"""Link record before a run: TCP connect time from this machine to each peer's ssh port (20 samples, 0.3 s apart) and the
interface type the join reports gave. Usage: python link_record.py <out.json> [peer ...]   (peers: name host port)"""
import json
import socket
import statistics
import sys
import time

PEERS = {"air": ("192.168.1.219", 22), "gtx": ("192.168.1.168", 2222)}


def sample(host, port, n=20):
    out = []
    for _ in range(n):
        t = time.time()
        try:
            socket.create_connection((host, port), 3).close()
            out.append((time.time() - t) * 1000)
        except OSError:
            out.append(None)
        time.sleep(0.3)
    ok = [x for x in out if x is not None]
    return {"samples": n, "ok": len(ok), "min_ms": min(ok) if ok else None, "median_ms": statistics.median(ok) if ok else None,
            "max_ms": max(ok) if ok else None}


if __name__ == "__main__":
    res = {name: sample(*PEERS[name]) for name in (sys.argv[2:] or PEERS)}
    res["date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res))
