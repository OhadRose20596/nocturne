"""Exchange microbenchmark on the engine's own transport, no model and no GPU: both sides send a frame and receive the
peer's frame, `frames` times per step, the way LinkComm.all_reduce does. Reports ms per exchange and per step.
  listen:  python -m nocturne.engines.torch_tp.microbench listen --bind <addr> --port 50053          (worker side)
  run:     python -m nocturne.engines.torch_tp.microbench run --peer <addr>:50053 --frames 56 --bytes 7168 --steps 300 [--mode thread|inline]
mode thread is the engine's path (a thread per send); inline is sendall then recv on the main thread. The listener
reads frames, bytes, steps and mode from the HELLO. The port accepts any peer; security deferred by ADR-009."""
import argparse
import json
import os
import statistics
import sys
import time

from . import transport


def exchange(link: transport.Link, payload: bytes, mode: str) -> None:
    if mode == "thread":
        t = link.send_async(transport.KIND_ALLREDUCE, payload, "bfloat16", (len(payload) // 2,))
        link.recv()
        t.join()
    else:
        link.send(transport.KIND_ALLREDUCE, payload, "bfloat16", (len(payload) // 2,))
        link.recv()


def run_steps(link: transport.Link, frames: int, nbytes: int, steps: int, mode: str, warmup: int = 5) -> dict:
    payload = os.urandom(nbytes)
    per_frame, per_step = [], []
    for s in range(warmup + steps):
        t0 = time.perf_counter_ns()
        for _ in range(frames):
            f0 = time.perf_counter_ns()
            exchange(link, payload, mode)
            if s >= warmup:
                per_frame.append((time.perf_counter_ns() - f0) / 1e6)
        if s >= warmup:
            per_step.append((time.perf_counter_ns() - t0) / 1e6)
    q = statistics.quantiles(per_frame, n=100)
    return {"frames": frames, "bytes": nbytes, "steps": steps, "mode": mode,
            "exchange_ms": {"mean": statistics.fmean(per_frame), "p50": statistics.median(per_frame), "p95": q[94], "p99": q[98],
                            "min": min(per_frame), "max": max(per_frame)},
            "step_ms": {"mean": statistics.fmean(per_step), "p50": statistics.median(per_step), "min": min(per_step), "max": max(per_step)},
            "gaps_over_30ms": sum(1 for x in per_frame if x > 30), "gap_time_share": sum(x for x in per_frame if x > 30) / sum(per_frame)}


def listen(bind: str, port: int) -> None:
    srv = transport.listen(bind, port)
    while True:
        sock, addr = srv.accept()
        link = transport.Link(sock)
        try:
            kind, payload, *_ = link.recv()
        except (ConnectionError, OSError):
            link.close()
            continue
        if kind != transport.KIND_HELLO:
            link.close()
            continue
        h = json.loads(payload.decode())
        print(f"[microbench] peer {addr}: {h}", flush=True)
        link.send(transport.KIND_HELLO, b"ok", "uint8")
        try:
            for _ in range(h["warmup"] + h["steps"]):
                pl = os.urandom(h["bytes"])
                for _ in range(h["frames"]):
                    exchange(link, pl, h["mode"])
            kind, *_ = link.recv()
            if kind == transport.KIND_SHUTDOWN:
                print("[microbench] done", flush=True)
        except (ConnectionError, OSError) as e:
            print(f"[microbench] peer went away: {e}", flush=True)
        link.close()
        if h.get("once"):
            srv.close()
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("listen"); l.add_argument("--bind", required=True); l.add_argument("--port", type=int, default=50053)
    r = sub.add_parser("run"); r.add_argument("--peer", required=True); r.add_argument("--frames", type=int, default=56)
    r.add_argument("--bytes", type=int, default=7168); r.add_argument("--steps", type=int, default=300)
    r.add_argument("--mode", choices=["thread", "inline"], default="thread"); r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--once", action="store_true", help="the listener exits after this run")
    r.add_argument("--json", help="write the summary here as well")
    a = ap.parse_args(argv)
    if a.cmd == "listen":
        listen(a.bind, a.port)
        return 0
    host, port = a.peer.rsplit(":", 1)
    link = transport.Link(transport.connect(host, int(port)))
    hello = {"frames": a.frames, "bytes": a.bytes, "steps": a.steps, "mode": a.mode, "warmup": a.warmup, "once": a.once}
    link.send(transport.KIND_HELLO, json.dumps(hello).encode(), "uint8")
    kind, *_ = link.recv()
    if kind != transport.KIND_HELLO:
        sys.exit("listener did not answer the hello")
    res = run_steps(link, a.frames, a.bytes, a.steps, a.mode, a.warmup)
    res["peer"] = a.peer
    link.send(transport.KIND_SHUTDOWN)
    link.close()
    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
