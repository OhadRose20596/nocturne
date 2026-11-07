"""Utilization sampler, 100 ms per sample, one per peer. Returns busy fraction (0..1) over a time window.
Metal: `sudo -n powermetrics --samplers gpu_power -i 100` (GPU HW active residency).
CUDA: `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 100`.
CPU fallback: psutil on the local host, `top`-free loop over /proc/stat on a remote Linux host.
Remote peers are sampled through the same ssh wrapper as the Runner. Degrades to null plus a note.
With sink=<path>, every sample is also appended as "wall_time<TAB>percent" for the live view (nocturne.bench.live)."""
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

RESIDENCY = re.compile(r"GPU (?:HW )?active residency:\s*([\d.]+)%")
PERCENT = re.compile(r"^\s*([\d.]+)\s*$")

CMDS = {
    "metal": "sudo -n powermetrics --samplers gpu_power -i 100",
    "cuda": "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 100",
    "cpu_linux": "while :; do head -1 /proc/stat; sleep 0.1; done",
    # macOS has no /proc: `top -l` in logging mode prints "CPU usage: 12.5% user, 6.2% sys, 81.2% idle" every 100 ms
    "cpu_macos": "top -l 0 -s 0 -n 0 -stats pid 2>/dev/null | grep --line-buffered 'CPU usage'",
}
CPU_USAGE_MAC = re.compile(r"CPU usage:.*?([\d.]+)% idle")


class Sampler:
    def __init__(self, peer: dict, sink=None):
        self.peer = peer
        self.sink = open(sink, "a", buffering=1) if sink else None
        self.samples: list[tuple[float, float]] = []   # (wall time, percent)
        self.notes: list[str] = []
        self.proc = None
        self.thread = None
        self._prev_stat = None
        self._stop = threading.Event()

    def _argv(self):
        be = self.peer["backend"]
        if be == "metal":
            cmd = CMDS["metal"]
        elif be == "cuda":
            cmd = CMDS["cuda"]
        elif self.peer.get("ssh"):
            cmd = CMDS["cpu_macos"] if self.peer.get("os") == "macos" else CMDS["cpu_linux"]
        else:
            return None   # local CPU: psutil thread
        if self.peer.get("ssh"):
            return ["ssh", "-o", "BatchMode=yes", self.peer["ssh"], cmd]
        return ["bash", "-c", cmd]

    def _parse(self, line: str) -> float | None:
        m = RESIDENCY.search(line)
        if m:
            return float(m.group(1))
        m = PERCENT.match(line)
        if m and self.peer["backend"] == "cuda":
            return float(m.group(1))
        m = CPU_USAGE_MAC.search(line)
        if m:
            return 100.0 - float(m.group(1))
        if line.startswith("cpu "):
            parts = [int(x) for x in line.split()[1:8]]
            idle, total = parts[3] + parts[4], sum(parts)
            if self._prev_stat:
                di, dt = idle - self._prev_stat[0], total - self._prev_stat[1]
                self._prev_stat = (idle, total)
                return 100.0 * (1 - di / dt) if dt else None
            self._prev_stat = (idle, total)
        return None

    def _reader(self):
        for line in self.proc.stdout:
            v = self._parse(line)
            if v is not None:
                self._add(v)

    def _add(self, v: float):
        t = time.time()
        self.samples.append((t, v))
        if self.sink:
            self.sink.write(f"{t:.3f}\t{v:.1f}\n")

    def _psutil_loop(self):
        try:
            import psutil
        except ImportError:
            self.notes.append("psutil missing; cpu utilization null")
            return
        while not self._stop.is_set():
            self._add(psutil.cpu_percent(interval=0.1))

    def start(self):
        argv = self._argv()
        if argv is None:
            self.thread = threading.Thread(target=self._psutil_loop, daemon=True)
            self.thread.start()
            return
        try:
            self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            self.notes.append(f"{self.peer.get('name')}: sampler failed to start ({e})")
            return
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            err = (self.proc.stderr.read() or "").strip() if self.proc.stderr else ""
            if not self.samples:
                hint = "needs passwordless sudo for powermetrics" if self.peer["backend"] == "metal" else ""
                self.notes.append(f"{self.peer.get('name')}: no utilization samples. {hint} {err[-200:]}".strip())
        if self.thread:
            self.thread.join(timeout=2)
        if self.sink:
            self.sink.close()
            self.sink = None

    def busy_fraction(self, windows: list[tuple[float, float]]) -> float | None:
        vals = [v for t, v in self.samples if any(a <= t <= b for a, b in windows)]
        if not vals:
            return None
        return min(1.0, max(0.0, sum(vals) / len(vals) / 100.0))


def start_all(peers: list[dict], dry_run: bool = False, sink_dir=None) -> list[Sampler]:
    out = []
    for p in peers:
        s = Sampler(p, sink=(Path(sink_dir) / f"util_{p.get('name')}.tsv") if sink_dir else None)
        if dry_run:
            argv = s._argv()
            print(f"[probe:{p.get('name')}] $ {' '.join(shlex.quote(a) for a in argv) if argv else 'psutil.cpu_percent loop'}", flush=True)
        else:
            s.start()
        out.append(s)
    return out
