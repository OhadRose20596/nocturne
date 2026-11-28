"""Clock state of this host (session 10, A1). Standard library only, so any python3 runs it.
Usage: python3 nocturne/bench/probes/timesync.py [--server time.apple.com] [--samples 8] [--json]
Measures the offset to one NTP server with SNTP (RFC 4330): offset = ((t2 - t1) + (t3 - t4)) / 2, the server clock minus
this clock, taken from the sample with the smallest round trip delay. Also reports what the OS time service says
(macOS: sntp; Linux and WSL2: chronyc or timedatectl; WSL2 also asks Windows through w32tm.exe). Sets nothing.
The time of the last sync is what the OS tool tells: chronyc "Ref time", w32tm "Last Successful Sync Time". macOS has no
such line; there the report holds only the time of this measurement (`at`), and of a `--set` when the script made one."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import time

NTP_EPOCH = 2208988800          # seconds from 1900 to 1970
DEFAULT_SERVER = os.environ.get("NOCTURNE_NTP_SERVER", "time.apple.com")
W32TM = "/mnt/c/Windows/System32/w32tm.exe"


def _ntp_to_unix(sec: int, frac: int) -> float:
    return sec - NTP_EPOCH + frac / 2 ** 32


def offset_delay(t1: float, t2: float, t3: float, t4: float) -> tuple[float, float]:
    """(server minus local, round trip delay) from the four SNTP timestamps. t1, t4 local; t2, t3 server."""
    return ((t2 - t1) + (t3 - t4)) / 2, (t4 - t1) - (t3 - t2)


def parse_reply(data: bytes) -> tuple[float, float, int]:
    """(t2 receive, t3 transmit, stratum) of a 48 byte NTP reply."""
    if len(data) < 48:
        raise ValueError(f"NTP reply of {len(data)} bytes")
    stratum = data[1]
    r_sec, r_frac, x_sec, x_frac = struct.unpack("!IIII", data[32:48])
    return _ntp_to_unix(r_sec, r_frac), _ntp_to_unix(x_sec, x_frac), stratum


def sntp_once(server: str, timeout: float = 2.0) -> tuple[float, float, int, str]:
    """One query: (offset, delay, stratum, address)."""
    addr = socket.getaddrinfo(server, 123, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t1 = time.time()
        s.sendto(b"\x23" + 47 * b"\0", addr)          # LI 0, version 4, mode 3 (client)
        data, _ = s.recvfrom(256)
        t4 = time.time()
    t2, t3, stratum = parse_reply(data)
    if stratum == 0:
        raise ValueError("kiss of death reply (stratum 0)")
    off, delay = offset_delay(t1, t2, t3, t4)
    return off, delay, stratum, addr[0]


def best(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """The (offset, delay) with the smallest delay: the least queued, so the most symmetric path."""
    return min(samples, key=lambda s: s[1])


def measure(server: str = DEFAULT_SERVER, samples: int = 8, query=sntp_once, pause: float = 0.05) -> dict:
    got, err, stratum, ip = [], None, None, None
    for _ in range(samples):
        try:
            off, delay, stratum, ip = query(server)
            got.append((off, delay))
        except (OSError, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(pause)
    out = {"server": server, "server_ip": ip, "stratum": stratum, "samples": len(got), "asked": samples}
    if not got:
        return dict(out, offset_s=None, delay_s=None, error=err or "no reply")
    off, delay = best(got)
    return dict(out, offset_s=off, delay_s=delay, bound_s=delay / 2, spread_s=max(o for o, _ in got) - min(o for o, _ in got))


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return (r.stdout or b"").decode("utf-8", "replace").replace("\x00", "") + (r.stderr or b"").decode("utf-8", "replace").replace("\x00", "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_chronyc(text: str) -> dict | None:
    m = re.search(r"System time\s*:\s*([\d.]+) seconds (fast|slow) of NTP time", text)
    if not m:
        return None
    ref = re.search(r"Reference ID\s*:\s*\S+ \(([^)]*)\)", text)
    last = re.search(r"Ref time \(UTC\)\s*:\s*(.+?)\s*$", text, re.M)
    v = float(m.group(1))
    # "fast of NTP time": this clock is ahead, so server minus local is negative
    return {"tool": "chronyc", "source": ref.group(1) if ref else None, "offset_s": -v if m.group(2) == "fast" else v,
            "last_sync_utc": last.group(1) if last else None}


def parse_timedatectl(text: str) -> dict | None:
    m = re.search(r"Offset:\s*([+-]?[\d.]+)\s*(us|ms|s)\b", text)
    if not m:
        return None
    scale = {"us": 1e-6, "ms": 1e-3, "s": 1.0}[m.group(2)]
    srv = re.search(r"Server:\s*(\S+)(?: \(([^)]*)\))?", text)
    return {"tool": "timedatectl", "source": (srv.group(2) or srv.group(1)) if srv else None, "offset_s": float(m.group(1)) * scale}


def parse_w32tm(text: str) -> dict | None:
    src = re.search(r"^Source:\s*(.+?)\s*$", text, re.M)
    if not src:
        return None
    last = re.search(r"^Last Successful Sync Time:\s*(.+?)\s*$", text, re.M)
    return {"tool": "w32tm", "source": src.group(1), "last_sync": last.group(1) if last else None}


def parse_sntp(text: str) -> dict | None:
    m = re.search(r"^([+-][\d.]+) \+/- ([\d.]+) (\S+)", text, re.M)
    return {"tool": "sntp", "source": m.group(3), "offset_s": float(m.group(1)), "err_s": float(m.group(2))} if m else None


def is_wsl() -> bool:
    try:
        return "microsoft" in open("/proc/version").read().lower()
    except OSError:
        return False


def os_status(server: str, run=_run) -> dict:
    """What the OS time service reports. Read only."""
    if platform.system() == "Darwin":
        return parse_sntp(run(["sntp", "-t", "3", server])) or {"tool": "sntp", "error": "no reply"}
    st = parse_chronyc(run(["chronyc", "tracking"])) or parse_timedatectl(run(["timedatectl", "timesync-status"])) \
        or {"tool": None, "error": "no chrony and no systemd-timesyncd answer"}
    if is_wsl():
        st["wsl"] = True
        if os.path.exists(W32TM):
            st["windows"] = parse_w32tm(run([W32TM, "/query", "/status"])) or {"tool": "w32tm", "error": "no answer (service stopped or no permission)"}
    return st


SET_STATE = os.path.expanduser("~/.nocturne/timesync_set.json")


def last_set() -> dict | None:
    """What `17_timesync.sh --set` wrote the last time it stepped this clock: {"at": ..., "server": ...}."""
    try:
        return json.loads(open(SET_STATE).read())
    except (OSError, ValueError):
        return None


def report(server: str = DEFAULT_SERVER, samples: int = 8) -> dict:
    return {"host": socket.gethostname().split(".")[0], "os": platform.system(), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "at_unix": time.time(), "sntp": measure(server, samples), "os_service": os_status(server), "last_set": last_set()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = report(a.server, a.samples)
    if a.json:
        print(json.dumps(r))
    else:
        s, o = r["sntp"], r["os_service"]
        off = "none" if s["offset_s"] is None else f"{s['offset_s'] * 1000:+.3f} ms (bound {s['bound_s'] * 1000:.3f} ms)"
        print(f"timesync host={r['host']} os={r['os']} server={s['server']} offset={off} samples={s['samples']}/{s['asked']} "
              f"os_tool={o.get('tool')} os_source={o.get('source')} os_offset_s={o.get('offset_s')} at={r['at']}")
    return 0 if r["sntp"]["offset_s"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
