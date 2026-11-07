"""Network probes. Pre-run: ping median over 50 packets and iperf3 for 10 s (the peer must run `iperf3 -s`), labelled
`probe` in the record. During the run: PingMonitor, a background ping to the first worker at 10 ms (100 ms when the
host refuses the short interval without root), saved as ping.tsv, percentiles inside the decode windows. awdl_state
reads the awdl0 interface. Missing tool or failure -> field null plus a note. Never raises."""
import json
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

PING_TIME = re.compile(r"time[=<]\s*([\d.]+)\s*ms")


def ping_median_ms(ip: str, count: int = 50, interval: float = 0.2) -> tuple[float | None, list[str]]:
    notes = []
    if not shutil.which("ping"):
        return None, ["ping not installed"]
    try:
        r = subprocess.run(["ping", "-c", str(count), "-i", str(interval), ip],
                           capture_output=True, text=True, timeout=count * interval + 30)
    except subprocess.TimeoutExpired:
        return None, [f"ping {ip} timed out"]
    times = [float(x) for x in PING_TIME.findall(r.stdout)]
    if not times:
        return None, [f"ping {ip}: no replies"]
    if len(times) < count:
        notes.append(f"ping {ip}: {count - len(times)} of {count} packets lost")
    return statistics.median(times), notes


def iperf3_mbps(ip: str, seconds: int = 10) -> tuple[float | None, list[str]]:
    if not shutil.which("iperf3"):
        return None, ["iperf3 not installed on the requester; bw_mbps null"]
    try:
        r = subprocess.run(["iperf3", "-c", ip, "-t", str(seconds), "-J"],
                           capture_output=True, text=True, timeout=seconds + 30)
        data = json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None, [f"iperf3 to {ip} failed or returned no JSON"]
    if "error" in data:
        return None, [f"iperf3 to {ip}: {data['error']} (is `iperf3 -s` running on the peer?)"]
    try:
        bps = data["end"]["sum_received"]["bits_per_second"]
    except KeyError:
        return None, [f"iperf3 to {ip}: no sum_received in output"]
    return bps / 1e6, []


def measure(ip: str | None, network_type: str, dry_run: bool = False) -> tuple[dict, list[str]]:
    """Returns (network dict for the record, notes)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    now = now[:-2] + ":" + now[-2:]
    if ip is None or network_type == "local":
        return {"type": network_type, "rtt_ms_median": None, "bw_mbps": None, "measured_at": now}, ["no peer: network probe skipped"]
    if dry_run:
        print(f"[probe] $ ping -c 50 -i 0.2 {ip}\n[probe] $ iperf3 -c {ip} -t 10 -J", flush=True)
        return {"type": network_type, "rtt_ms_median": None, "bw_mbps": None, "measured_at": now}, ["dry run"]
    rtt, n1 = ping_median_ms(ip)
    bw, n2 = iperf3_mbps(ip)
    return {"type": network_type, "rtt_ms_median": rtt, "bw_mbps": bw, "measured_at": now}, n1 + n2


PING_SEQ = re.compile(r"icmp_seq=(\d+).*?time[=<]\s*([\d.]+)\s*ms")


class PingMonitor:
    """`ping -i <interval> <ip>` for the whole run. samples: (wall time of the reply line, seq, rtt ms)."""

    def __init__(self, ip: str, interval_s: float = 0.01, sink=None):
        self.ip, self.interval_s = ip, interval_s
        self.sink = sink
        self.samples: list[tuple[float, int, float]] = []
        self.notes: list[str] = []
        self.proc = None
        self.thread = None
        self.t_start = None

    def argv(self) -> list[str]:
        return ["ping", "-i", f"{self.interval_s:g}", self.ip]

    def start(self):
        if not shutil.which("ping"):
            self.notes.append("ping not installed; ping_decode_ms null")
            return
        self.t_start = time.time()
        self.proc = subprocess.Popen(self.argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.5)
        if self.proc.poll() is not None and self.interval_s < 0.1:
            err = (self.proc.stderr.read() or "").strip()
            self.notes.append(f"ping -i {self.interval_s:g} refused ({err[-120:]}); background ping at 0.1 s instead")
            self.interval_s = 0.1
            self.proc = subprocess.Popen(self.argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        for line in self.proc.stdout:
            self.feed(line)

    def feed(self, line: str) -> None:
        m = PING_SEQ.search(line)
        if m:
            self.samples.append((time.time(), int(m.group(1)), float(m.group(2))))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)
        if self.proc and not self.samples:
            self.notes.append(f"background ping to {self.ip}: no replies; ping_decode_ms null")

    def save(self, path) -> None:
        Path(path).write_text("".join(f"{t:.6f}\t{q}\t{r:.3f}\n" for t, q, r in self.samples))

    def in_windows(self, windows) -> dict | None:
        return ping_percentiles(self.samples, windows, self.interval_s)


def ping_percentiles(samples, windows, interval_s: float | None = None) -> dict | None:
    """p50, p95, p99 of the replies whose wall time falls in a window, n replies, loss = 1 - n / expected (expected
    from the window length and interval). None when no reply is inside the windows."""
    sel = sorted(r for t, _, r in samples if any(a <= t <= b for a, b in windows))
    if not sel:
        return None
    def pct(p):
        return sel[min(len(sel) - 1, int(round(p * (len(sel) - 1))))]
    out = {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "n": len(sel)}
    if interval_s:
        expected = sum(b - a for a, b in windows) / interval_s
        out["interval_s"] = interval_s
        out["loss"] = max(0.0, 1 - len(sel) / expected) if expected > 0 else None
    return out


def awdl_state(run=subprocess.run) -> str:
    """up, down or unknown from `ifconfig awdl0` (macOS). Other systems: unknown."""
    try:
        r = run(["ifconfig", "awdl0"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if r.returncode != 0 or "awdl0" not in (r.stdout or ""):
        return "unknown"
    flags = (r.stdout.splitlines() or [""])[0]
    return "up" if "<UP," in flags or ",UP," in flags or "<UP>" in flags else "down"
