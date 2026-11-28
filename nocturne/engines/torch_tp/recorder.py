"""Recorder and sampler of one rank process (session 10, A3). Off unless the run asks for a timeline.
Recorder: a cursor on time.monotonic_ns(). lap(kind) closes the span from the cursor to now with that kind, so every
instant of a step belongs to exactly one span. Kinds: compute (ends after a device sync, so it is real device time),
host_copy (tensor to the host, sum, tensor back to the device), send (bytes, destination), recv_wait (blocked on the
upstream link or on a head rank, with source), allreduce_wait (blocked on a group peer), other (Python between steps).
Events stay in memory and are written once, at shutdown, as timeline_<rank>.jsonl: a header line, then one JSON array
per event in the order of COLS.
Sampler: a thread that reads cumulative counters every 20 ms (process CPU seconds; for the host sampler also host busy
CPU seconds and bytes per network interface) and, on a CUDA rank, nvidia-smi every 100 ms. Written as
samples_rank_<rank>.jsonl (timeline.py merges them into samples_<host>.jsonl). Rates are formed later from differences (timeline.py)."""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

COLS = ("kind", "t0_ns", "t1_ns", "step", "req", "peer", "bytes", "tokens")
KINDS = ("compute", "host_copy", "send", "recv_wait", "allreduce_wait", "other")
INTERVAL_S = 0.02
GPU_INTERVAL_MS = 100


def file_name(prefix: str, rank: str) -> str:
    return f"{prefix}_{rank.replace('/', '_')}.jsonl"


class Recorder:
    def __init__(self, name: str, gid: int, host: str, device: str):
        self.meta = {"rank": name, "gid": gid, "host": host, "device": device, "pid": os.getpid(), "clock": "monotonic_ns",
                     "cols": list(COLS)}
        self.ev: list[tuple] = []
        self.t = time.monotonic_ns()
        self.step, self.req, self.tokens = -1, 0, 0

    def begin(self, req: int, tokens: int) -> None:
        """A new step starts; the cursor stays where the previous step left it."""
        self.step += 1
        self.req, self.tokens = req, tokens

    def skip(self) -> None:
        """Move the cursor to now without a span (the time between requests on the coordinator)."""
        self.t = time.monotonic_ns()

    def lap(self, kind: str, peer: str | None = None, nbytes: int = 0) -> None:
        t = time.monotonic_ns()
        self.ev.append((kind, self.t, t, self.step, self.req, peer, nbytes, self.tokens))
        self.t = t

    def write(self, run_dir: str | Path) -> Path:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / file_name("timeline", self.meta["rank"])
        with p.open("w") as f:
            f.write(json.dumps(dict(self.meta, events=len(self.ev))) + "\n")
            for e in self.ev:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
        return p


def ifaces_for(addrs: list[str]) -> list[str]:
    """Names of the interfaces that carry the given addresses (the ones the rank listens on)."""
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for name, lst in psutil.net_if_addrs().items():
        if any(a.address in addrs for a in lst) and name not in out:
            out.append(name)
    return out


def parse_smi(line: str) -> tuple[float, float] | None:
    """"37, 2048" (utilization percent, memory used MiB) from nvidia-smi csv,noheader,nounits."""
    parts = [x.strip() for x in line.split(",")]
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def smi_argv(device: str) -> list[str]:
    idx = device.split(":")[1] if ":" in device else "0"
    return ["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", str(GPU_INTERVAL_MS)]


class Sampler:
    def __init__(self, rank: str, host: str, device: str, ifaces: list[str], host_level: bool, interval_s: float = INTERVAL_S):
        self.meta = {"rank": rank, "host": host, "device": device, "pid": os.getpid(), "ifaces": ifaces, "host_level": host_level,
                     "interval_s": interval_s, "clock": "monotonic_ns", "notes": []}
        self.rows: list[dict] = []
        self.ifaces, self.host_level, self.interval = ifaces, host_level, interval_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi = None
        self.gpu = host_level and device.split(":")[0] == "cuda"

    def start(self) -> "Sampler":
        try:
            import psutil
            self._ps = psutil
            self._proc = psutil.Process()
            self.meta["ncpu"] = psutil.cpu_count()
            self._threads.append(threading.Thread(target=self._loop, daemon=True))
        except ImportError:
            self.meta["notes"].append("psutil missing: no CPU and network samples")
        if self.gpu:
            if shutil.which("nvidia-smi"):
                self._smi = subprocess.Popen(smi_argv(self.meta["device"]), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self._threads.append(threading.Thread(target=self._gpu_loop, daemon=True))
            else:
                self.meta["notes"].append("nvidia-smi not found: no GPU samples")
        for t in self._threads:
            t.start()
        return self

    def _sample(self) -> dict:
        ps = self._ps
        c = self._proc.cpu_times()
        row = {"t": time.monotonic_ns(), "pc": c.user + c.system}
        if self.host_level:
            h = ps.cpu_times()
            row["hb"] = sum(h) - h.idle - getattr(h, "iowait", 0.0)
            if self.ifaces:
                io = ps.net_io_counters(pernic=True)
                row["net"] = {n: [io[n].bytes_recv, io[n].bytes_sent] for n in self.ifaces if n in io}
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.rows.append(self._sample())
            except Exception as e:      # noqa: BLE001 - a sampler must never end a run
                self.meta["notes"].append(f"sample failed: {type(e).__name__}: {e}")
                return
            self._stop.wait(self.interval)

    def _gpu_loop(self) -> None:
        for line in self._smi.stdout:
            v = parse_smi(line)
            if v:
                self.rows.append({"t": time.monotonic_ns(), "gu": v[0], "gm": v[1]})
            if self._stop.is_set():
                return

    def stop(self) -> None:
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()
            try:
                self._smi.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._smi.kill()
        for t in self._threads:
            t.join(timeout=3)

    def write(self, run_dir: str | Path) -> Path:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / file_name("samples_rank", self.meta["rank"])
        rows = sorted(self.rows, key=lambda r: r["t"])
        with p.open("w") as f:
            f.write(json.dumps(dict(self.meta, rows=len(rows))) + "\n")
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        return p
