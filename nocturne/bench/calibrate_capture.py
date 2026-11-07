#!/usr/bin/env python3
"""One-off check of the traffic counter: send a known payload through the same tcpdump path and compare.
Usage: python -m nocturne.bench.calibrate_capture [--peers FILE] [--network wifi] [--mb 10] [--port 50099] [--local]
Listener `nc -l <ip> <port>` runs on the first worker over ssh (bound to its address for the network, never 0.0.0.0;
security deferred by ADR-009). With --local both ends are 127.0.0.1 on the loopback interface. Prints measured wire
bytes against the payload and the ratio; a ratio near 1.0 means no double counting, 2.0 means every packet is seen
twice. The result belongs in nocturne/bench/README.md under "Capture calibration"."""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .probes import capture
from .runner import Runner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peers", type=Path)
    ap.add_argument("--network", choices=config.NETWORKS, default="wifi")
    ap.add_argument("--mb", type=int, default=10)
    ap.add_argument("--port", type=int, default=50099)
    ap.add_argument("--local", action="store_true", help="loopback only: nc -l on 127.0.0.1, no ssh")
    a = ap.parse_args(argv)
    runner = Runner()
    if a.local:
        ip, peer = "127.0.0.1", {"name": "local", "ssh": None}
    else:
        cfg = config.load_peers(a.peers)
        w = cfg["peers"][cfg["roles"]["workers"][0]]
        w["name"] = cfg["roles"]["workers"][0]
        ip, peer = (w.get("networks") or {}).get(a.network) or w["bind"], w
    iface, notes = capture.iface_for(ip)
    for n in notes:
        print("note:", n)
    if iface is None:
        sys.exit("no interface for the route; cannot calibrate")
    cap = capture.Capture(a.port, iface)
    listener = runner.spawn(peer, ["nc", "-l", ip, str(a.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    cap.start()
    if cap.proc is None:
        sys.exit("; ".join(cap.notes))
    t0 = time.time()
    payload = os.urandom(a.mb * 1024 * 1024)
    # A stdlib socket, not `nc -N`: the macOS nc has no -N and would hang open (I025 6). The listener side stays nc -l.
    send_err = None
    try:
        with socket.create_connection((ip, a.port), timeout=30) as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
    except OSError as e:
        send_err = str(e)
    t1 = time.time()
    time.sleep(1.0)
    cap.stop()
    runner.stop_all()
    expected = len(payload)
    in_window = [(t, s) for t, s in cap.packets if t0 - 0.5 <= t <= t1 + 0.5]
    measured = sum(s for _, s in in_window)
    if send_err:
        sys.exit(f"send failed: {send_err}; is nc -l running on {ip}:{a.port}?")
    print(f"interface {iface}, port {a.port}, payload {expected} bytes sent")
    print(f"tcpdump payload bytes in window: {measured} over {len(in_window)} packets, ratio {measured / expected:.3f}")
    print("ratio ~1.0: counter exact (tcp payload only, no double counting); ~2.0: each packet counted twice; ")
    print("1.0 to 1.1: acks and retransmits, expected on Wi-Fi")
    if listener is not None and listener.poll() is None:
        listener.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
