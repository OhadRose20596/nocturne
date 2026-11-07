"""Process memory sampler, 500 ms, one per peer. Peak over a time window in GB.
macOS: `ps -o rss= -p <pid>` of llama-server (requester, pid known) or ggml-rpc-server (worker, pgrep by port).
`footprint` is the better number on macOS but needs root for other users' processes, so rss is used and named in source.
CUDA: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500` (device memory, not process).
Remote peers run the loop over ssh. No samples -> None plus a `not_measured` note."""
import shlex
import subprocess
import threading
import time


def loop_cmd(peer: dict, pid: int | None, rpc_port: int, pattern: str | None = None) -> str | None:
    if peer["backend"] == "cuda":
        return "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500"
    proc = pattern or ("llama-server" if peer.get("role") == "requester" else f"ggml-rpc-server.*-p {rpc_port}")
    pidexpr = str(pid) if pid else f"$(pgrep -f '{proc}' | head -1)"
    return f"while :; do ps -o rss= -p {pidexpr} 2>/dev/null | head -1; sleep 0.5; done"


class MemSampler:
    def __init__(self, peer: dict, pid: int | None = None, rpc_port: int = 50052, pattern: str | None = None):
        self.peer, self.pid, self.rpc_port, self.pattern = peer, pid, rpc_port, pattern
        self.samples: list[tuple[float, float]] = []     # (wall time, GB)
        self.notes: list[str] = []
        self.proc = None
        self.thread = None
        self.unit = "mib" if peer["backend"] == "cuda" else "kib"
        self.source = ("nvidia-smi memory.used (device)" if peer["backend"] == "cuda"
                       else "ps rss of " + ("llama-server" if peer.get("role") == "requester" else "ggml-rpc-server"))

    def argv(self) -> list[str]:
        cmd = loop_cmd(self.peer, self.pid, self.rpc_port, self.pattern)
        if self.peer.get("ssh"):
            return ["ssh", "-o", "BatchMode=yes", self.peer["ssh"], cmd]
        return ["bash", "-c", cmd]

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                v = float(line)
            except ValueError:
                continue
            gb = v / (1024 * 1024) if self.unit == "kib" else v / 1024
            self.samples.append((time.time(), gb))

    def start(self):
        try:
            self.proc = subprocess.Popen(self.argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            self.notes.append(f"{self.peer.get('name')}: memory sampler failed to start ({e}); memory_peak_gb not_measured")
            return
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)
        if not self.samples:
            self.notes.append(f"{self.peer.get('name')}: memory_peak_gb not_measured (no samples from {self.source})")

    def peak_gb(self, windows: list[tuple[float, float]] | None = None) -> float | None:
        vals = [v for t, v in self.samples if not windows or any(a <= t <= b for a, b in windows)]
        if not vals and self.samples:
            vals = [v for _, v in self.samples]      # windows missed every sample: whole-run peak, still a measurement
        return round(max(vals), 3) if vals else None


def start_all(peers: list[dict], requester_pid: int | None, rpc_port: int, dry_run: bool = False,
              patterns: dict | None = None) -> list[MemSampler]:
    out = []
    for p in peers:
        s = MemSampler(p, requester_pid if p["role"] == "requester" else None, rpc_port, (patterns or {}).get(p["role"]))
        if dry_run:
            print(f"[probe:{p.get('name')}] $ {' '.join(shlex.quote(a) for a in s.argv())}", flush=True)
        else:
            s.start()
        out.append(s)
    return out
