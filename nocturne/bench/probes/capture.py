"""Traffic probe on the worker port during decode. Two sources, both recorded:
  bytes/token  from tcpdump on the requester (wire bytes to and from every worker, decode windows only, with the
               direction of every packet: out = requester to worker, in = worker to requester);
  msgs/token   from the worker log written with GGML_RPC_DEBUG=1, parsed by scripts/12_rpc_trace.py.
Optionally a second tcpdump on the worker over ssh (same filter plus icmp); the background ping's echo pairs seen in
both captures give the worker clock offset. tcpdump needs sudo; without it bytes fields are null plus a note. The
capture interface is the one that owns the route to the peer (`route get` on macOS, `ip route get` on Linux), never
`any`, which on macOS can report a packet once per interface it crosses. Calibrate with
`python -m nocturne.bench.calibrate_capture`."""
import importlib.util
import platform
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACE_PY = ROOT / "scripts" / "12_rpc_trace.py"
# `tcpdump -tt -q -nn -l`: "1730000000.123456 IP a.b.c.d.50052 > e.f.g.h.5555: tcp 14336"
LINE = re.compile(r"^(\d+\.\d+) IP6? (\S+) > (\S+): tcp (\d+)")
# "1730000000.123456 IP a.b.c.d > e.f.g.h: ICMP echo request, id 12345, seq 7, length 64"
ICMP = re.compile(r"^(\d+\.\d+) IP6? (\S+) > (\S+): ICMP echo (request|reply), id (\d+), seq (\d+)")
RPC_TENSOR_BYTES = 304      # sizeof(rpc_tensor) at 1e7bcf3, ggml/src/ggml-rpc/ggml-rpc.cpp (struct padded to 8)


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("rpc_trace", TRACE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROUTE_IFACE_MAC = re.compile(r"^\s*interface:\s*(\S+)", re.MULTILINE)
ROUTE_IFACE_LINUX = re.compile(r"\bdev\s+(\S+)")


def iface_for(ip: str, system: str | None = None, run=subprocess.run) -> tuple[str | None, list[str]]:
    """Interface that owns the route to ip. (None, [note]) when the route lookup fails."""
    system = system or platform.system()
    argv = ["route", "-n", "get", ip] if system == "Darwin" else ["ip", "route", "get", ip]
    try:
        r = run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, [f"route lookup for {ip} failed ({e}); capture interface unknown"]
    m = (ROUTE_IFACE_MAC if system == "Darwin" else ROUTE_IFACE_LINUX).search(r.stdout or "")
    if not m:
        return None, [f"route lookup for {ip} gave no interface: {(r.stdout or r.stderr).strip()[:120]}"]
    return m.group(1), []


def packet_dir(src: str, dst: str, port: int) -> str:
    """out = towards the worker port, in = from it. tcpdump prints host.port; the port is the last dotted field."""
    return "out" if dst.rsplit(".", 1)[-1] == str(port) else ("in" if src.rsplit(".", 1)[-1] == str(port) else "?")


class Capture:
    """tcpdump on the requester. packets: (time, size, dir). icmp: (time, kind, id, seq) of the background ping, kept
    for the worker clock offset. With ssh_peer the same command runs on the worker over ssh (worker side capture)."""

    def __init__(self, port: int, iface: str | None = None, ssh_peer: dict | None = None, icmp: bool = True):
        if iface is None:
            raise ValueError("Capture needs the interface that owns the route to the worker; use iface_for(ip)")
        self.port, self.iface, self.ssh_peer, self.icmp_too = port, iface, ssh_peer, icmp
        self.packets: list[tuple[float, int, str]] = []
        self.icmp: list[tuple[float, str, int, int]] = []
        self.notes: list[str] = []
        self.proc = None
        self.thread = None

    def filter(self) -> str:
        return f"tcp port {self.port}" + (" or icmp" if self.icmp_too else "")

    def argv(self):
        base = ["sudo", "-n", "tcpdump", "-i", self.iface, "-nn", "-q", "-tt", "-l", self.filter()]
        if self.ssh_peer and self.ssh_peer.get("ssh"):
            import shlex
            return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.ssh_peer["ssh"], " ".join(shlex.quote(a) for a in base)]
        return base

    def start(self):
        try:
            self.proc = subprocess.Popen(self.argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            self.notes.append(f"tcpdump not started ({e}); bytes_per_token null")
            return
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        time.sleep(0.5)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().strip()
            self.notes.append(f"tcpdump exited early (needs passwordless sudo?): {err[-200:]}; bytes_per_token null")
            self.proc = None

    def _reader(self):
        for line in self.proc.stdout:
            self.feed(line)

    def feed(self, line: str) -> None:
        m = LINE.match(line)
        if m:
            self.packets.append((float(m.group(1)), int(m.group(4)), packet_dir(m.group(2), m.group(3), self.port)))
            return
        m = ICMP.match(line)
        if m:
            self.icmp.append((float(m.group(1)), m.group(4), int(m.group(5)), int(m.group(6))))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)

    def per_token(self, windows: list[tuple[float, float, int]], compute_steps: int | None = None) -> dict:
        """windows: (start, end, n_tokens) per prompt. Concurrent prompts (A4) overlap in time, so the
        windows are merged into a union and total bytes are divided by total tokens; a byte is never counted twice."""
        if self.proc is None and not self.packets:
            return {"bytes_per_token": None, "bytes_per_step": None, "packets_per_token": None}
        return bytes_per_token(self.packets, windows, compute_steps)

    def per_decode_step(self, windows, decode_steps: int | None) -> dict:
        if self.proc is None and not self.packets:
            return {"bytes_per_step_decode": None, "bytes_per_step_decode_out": None, "bytes_per_step_decode_in": None}
        return bytes_per_decode_step(self.packets, windows, decode_steps)

    def save(self, path) -> None:
        Path(path).write_text("".join(f"{t:.6f}\t{s}\t{d}\n" for t, s, d in self.packets))

    def save_icmp(self, path) -> None:
        Path(path).write_text("".join(f"{t:.6f}\t{k}\t{i}\t{q}\n" for t, k, i, q in self.icmp))


def load_capture(path) -> list[tuple[float, int, str]]:
    """capture.tsv rows; two column files from before 2026-11-02 carry no direction and get '?'."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        f = line.split("\t")
        out.append((float(f[0]), int(f[1]), f[2] if len(f) > 2 else "?"))
    return out


def load_icmp(path) -> list[tuple[float, str, int, int]]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line:
            t, k, i, q = line.split("\t")
            out.append((float(t), k, int(i), int(q)))
    return out


def clock_offset(req_icmp, worker_icmp) -> tuple[float | None, float | None, int]:
    """Worker clock minus requester clock. For each echo (id, seq) seen as a request on both sides, the worker saw it
    one way delay after the requester sent it; the one way delay is taken as half the requester's own request to reply
    time for that seq. Returns (median offset, half interquartile range, pairs). None when fewer than 5 pairs."""
    import statistics
    req_sent = {(i, q): t for t, k, i, q in req_icmp if k == "request"}
    req_back = {(i, q): t for t, k, i, q in req_icmp if k == "reply"}
    w_seen = {(i, q): t for t, k, i, q in worker_icmp if k == "request"}
    offs = []
    for key, t_w in w_seen.items():
        if key in req_sent and key in req_back:
            rtt = req_back[key] - req_sent[key]
            if 0 <= rtt < 5.0:
                offs.append(t_w - req_sent[key] - rtt / 2)
    if len(offs) < 5:
        return None, None, len(offs)
    q = statistics.quantiles(offs, n=4)
    return statistics.median(offs), (q[2] - q[0]) / 2, len(offs)


def merge_windows(windows) -> list[tuple[float, float]]:
    """Union of [start, end] intervals."""
    iv = sorted((a, z) for a, z, *_ in windows if z >= a)
    out: list[list[float]] = []
    for a, z in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], z)
        else:
            out.append([a, z])
    return [(a, z) for a, z in out]


def _in_windows(packets, windows):
    union = merge_windows([w for w in windows if w[2]])
    return [p for p in packets if any(a <= p[0] <= z for a, z in union)]


def bytes_per_token(packets, windows: list[tuple[float, float, int]], compute_steps: int | None = None) -> dict:
    """bytes_per_token = bytes in the union of windows / total tokens; bytes_per_step = same bytes / compute_steps.
    packets are (time, size) or (time, size, dir)."""
    total_tokens = sum(n for _, _, n in windows if n)
    if not total_tokens:
        return {"bytes_per_token": None, "bytes_per_step": None, "packets_per_token": None}
    sel = _in_windows(packets, windows)
    total = sum(p[1] for p in sel)
    return {"bytes_per_token": total / total_tokens,
            "bytes_per_step": (total / compute_steps) if compute_steps else None,
            "packets_per_token": sum(1 for p in sel if p[1] > 0) / total_tokens}


def bytes_per_decode_step(packets, windows, decode_steps: int | None) -> dict:
    """Bytes inside the decode windows per decode step (sum of predicted_n - 1), split by direction when the capture
    has one. Windows of prompts with fewer than 2 tokens hold no step and are skipped."""
    if not decode_steps:
        return {"bytes_per_step_decode": None, "bytes_per_step_decode_out": None, "bytes_per_step_decode_in": None}
    sel = _in_windows(packets, [w for w in windows if w[2] and w[2] >= 2])
    total = sum(p[1] for p in sel)
    dirs = {p[2] for p in sel if len(p) > 2}
    has_dir = bool(dirs) and dirs <= {"in", "out"}
    out = sum(p[1] for p in sel if len(p) > 2 and p[2] == "out") if has_dir else None
    inn = sum(p[1] for p in sel if len(p) > 2 and p[2] == "in") if has_dir else None
    return {"bytes_per_step_decode": total / decode_steps,
            "bytes_per_step_decode_out": (out / decode_steps) if has_dir else None,
            "bytes_per_step_decode_in": (inn / decode_steps) if has_dir else None}


WORKER_CMD = re.compile(r"^(\d+\.\d+) \[(\w+)\](.*)")
N_TENSORS = re.compile(r"n_tensors:\s*(\d+)")


def decode_comm_from_worker_log(text: str, windows, decode_steps: int | None) -> tuple[dict, list[str]]:
    """llama.cpp per decode step inside the decode windows, from the timestamped worker log: blocking_syncs_per_step
    (get_tensor, the only command the client waits for), messages_per_step (every command, mean), and
    graph_meta_bytes_per_step (sum of n_tensors over graph_compute lines times sizeof(rpc_tensor)). The stamps are the
    requester's receive clock, the same clock as the decode windows."""
    empty = {"blocking_syncs_per_step": None, "messages_per_step": None, "graph_meta_bytes_per_step": None}
    if not decode_steps:
        return empty, ["no decode steps; per step comm null"]
    union = merge_windows([w for w in windows if w[2] and w[2] >= 2])
    if not union:
        return empty, ["no decode windows; per step comm null"]
    stamped = False
    cmds = 0
    gets = 0
    meta = 0
    for line in text.splitlines():
        m = WORKER_CMD.match(line)
        if not m:
            continue
        stamped = True
        t = float(m.group(1))
        if not any(a <= t <= z for a, z in union):
            continue
        cmds += 1
        if m.group(2) == "get_tensor":
            gets += 1
        elif m.group(2) == "graph_compute":
            k = N_TENSORS.search(m.group(3))
            if k:
                meta += int(k.group(1)) * RPC_TENSOR_BYTES
    if not stamped:
        return empty, ["worker log has no timestamps; per step comm null"]
    return {"blocking_syncs_per_step": gets / decode_steps, "messages_per_step": cmds / decode_steps,
            "graph_meta_bytes_per_step": meta / decode_steps}, []


def msgs_per_step_from_worker_log(text: str, tail: int = 32) -> tuple[float | None, list[str]]:
    s, n = step_stats_from_worker_log(text, tail)
    return s["msgs_per_step_median"] if s else None, n


def step_stats_from_worker_log(text: str, tail: int = 32) -> tuple[dict | None, list[str]]:
    """Per compute step of the last session: median RPC messages per step (set_tensor, get_tensor and the
    graph_compute or graph_recompute message itself, so 1,1 on 7B gives 7, as in the RESULTS.md trace table)
    and the number of compute steps. Messages per *token* = msgs_per_step * compute_steps / tokens generated,
    because a speculative or batched step (A2, A4) produces several tokens per step."""
    if not TRACE_PY.exists():
        return None, ["scripts/12_rpc_trace.py missing; msgs_per_token null"]
    mod = _load_trace_module()
    sessions, cur = [], []
    for raw in text.splitlines():
        if "[hello]" in raw and cur:
            sessions.append(cur); cur = []
        cur.append(raw)
    if cur:
        sessions.append(cur)
    sessions = [s for s in sessions if any("[graph_" in l for l in s)]
    if not sessions:
        return None, ["worker log has no graph_compute lines (GGML_RPC_DEBUG=1 not set?); msgs_per_token null"]
    steps = [s for s in mod.parse(sessions[-1]) if s["kind"] in ("graph_compute", "graph_recompute")]
    if not steps:
        return None, ["no compute steps parsed from worker log"]
    tail_steps = steps[-tail:]
    return {"msgs_per_step_median": statistics.median([sum(s["cmds"].values()) + 1 for s in tail_steps]),
            "compute_steps": len(steps),
            "payload_in_per_step_median": statistics.median([s["bytes_in"] for s in tail_steps]),
            "payload_out_per_step_median": statistics.median([s["bytes_out"] for s in tail_steps])}, []


def msgs_per_token(stats: dict | None, tokens_generated: int) -> float | None:
    if not stats or not tokens_generated:
        return None
    return stats["msgs_per_step_median"] * stats["compute_steps"] / tokens_generated


def comm_from_steps(stats: dict | None, tokens_generated: int) -> dict:
    """v2 comm fields derived from the worker log stats: per step is measured, per token is derived."""
    if not stats:
        return {}
    steps = stats["compute_steps"]
    tps = (tokens_generated / steps) if steps and tokens_generated else None
    return {"rpc_commands_per_step": stats["msgs_per_step_median"], "tokens_per_step": tps,
            "rpc_commands_per_token": (stats["msgs_per_step_median"] / tps) if tps else None,
            "compute_steps": steps, "tokens_generated": tokens_generated or None}


WORKER_TS = re.compile(r"^(\d+\.\d+)\s+\[(set_tensor|set_tensor_hash|graph_compute|graph_recompute)\]")


def transfer_window_from_worker_log(text: str) -> tuple[float | None, list[str]]:
    """Seconds between the first and last set_tensor / set_tensor_hash before the first graph message, from a worker
    log whose lines the runner prefixed with a wall clock. None plus a note when the log carries no timestamps."""
    first = last = None
    stamped = False
    for line in text.splitlines():
        m = WORKER_TS.match(line)
        if not m:
            continue
        stamped = True
        if m.group(2).startswith("graph_"):
            if first is not None:
                break
            continue
        t = float(m.group(1))
        first = t if first is None else first
        last = t
    if not stamped:
        return None, ["worker log has no timestamps (runner stamps them since session 2, 2026-10-26); setup.transfer_s null"]
    if first is None or last is None:
        return None, ["no set_tensor lines before the first graph message; setup.transfer_s null"]
    return last - first, []
