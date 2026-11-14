"""Serve the live view of a bench run on 127.0.0.1. Attaches to the newest results/logs/<id>/ unless --logs is given.
Usage: python -m nocturne.bench.live [--logs DIR] [--port 8091] [--host 127.0.0.1] [--follow] [--replay] [--speed 1.0]
Replay is automatic when the directory holds a finished run (a done event, or files older than 30 s)."""
import argparse
import sys
import time
from pathlib import Path

from . import server, tail

RESULTS = Path(__file__).resolve().parents[1] / "results"


def newest_logdir(results: Path = RESULTS) -> Path | None:
    dirs = [d for d in (results / "logs").glob("*") if d.is_dir() and any(d.iterdir())]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def is_finished(logdir: Path, idle_s: float = 30.0) -> bool:
    ev = logdir / "events.jsonl"
    if ev.exists() and '"event": "done"' in ev.read_text(errors="replace").replace('"event":"done"', '"event": "done"'):
        return True
    files = [p for p in logdir.glob("*") if p.is_file()]
    return bool(files) and time.time() - max(p.stat().st_mtime for p in files) > idle_s


def start(logdir: Path, port: int = 8091, replay: bool | None = None, speed: float = 1.0, host: str = "127.0.0.1"):
    """Used by run.py --live too. Returns (server, url, feeder)."""
    mode = "replay" if (replay if replay is not None else is_finished(logdir)) else "live"
    state = tail.RunState(logdir.name, mode, tail.peers_from_record(logdir))
    feeder = tail.Replayer(logdir, state, speed) if mode == "replay" else tail.Tailer(logdir, state)
    srv, url = server.serve(state, port, host)
    feeder.start()
    return srv, url, feeder


def attach(srv, logdir: Path):
    """Point an already running server (one socket per invocation, run.py --live) at a new run's log directory.
    Returns (server, url, feeder) like start()."""
    state = tail.RunState(logdir.name, "live", tail.peers_from_record(logdir))
    srv.RequestHandlerClass.state = state
    feeder = tail.Tailer(logdir, state)
    feeder.start()
    return srv, f"http://{srv.server_address[0]}:{srv.server_address[1]}/", feeder


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; a LAN address lets the other Mac open the page (no auth, ADR-009: home LAN only)")
    ap.add_argument("--follow", action="store_true", help="switch to each newer results/logs/<id>/ as a batch moves on")
    a = ap.parse_args(argv)
    logdir = a.logs or newest_logdir()
    if logdir is None or not logdir.is_dir():
        sys.exit("no log directory: pass --logs results/logs/<id>")
    if not any(logdir.iterdir()):
        sys.exit(f"{logdir} is empty: this run failed before any log was written")
    srv, url, feeder = start(logdir, a.port, True if a.replay else None, a.speed, a.host)
    print(f"live view: {url}  ({feeder.state.mode} of {logdir.name}); Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(3)
            if a.follow:
                new = newest_logdir()
                if new is not None and new != logdir:
                    feeder.stop()
                    logdir = new
                    srv, url, feeder = attach(srv, logdir)
                    print(f"live view: now following {logdir.name}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        feeder.stop()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
