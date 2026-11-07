"""State of one run, fed line by line from its log directory. Live mode tails the files; replay mode feeds the
same lines on a clock built from server.log timestamps (worker lines are interleaved by step count, since the
worker log has no clock). Read-only: nothing here writes under results/."""
import collections
import json
import re
import threading
import time
from pathlib import Path

from . import parsers

WORKER_GLOB = "worker_*.log"
PHASES = ["probe", "load", "prefill", "decode", "done"]


class RunState:
    def __init__(self, run_id: str, mode: str, peers: list[dict] | None = None):
        self.lock = threading.Lock()
        self.run_id, self.mode = run_id, mode
        self.t0 = time.time()
        self.anchor: float | None = None          # wall time of server.log ts 0 (set by the launch event)
        self.phase = "probe"
        self.phases: list[dict] = [{"name": "probe", "t": self.t0}]
        self.peers: dict[str, dict] = {}
        self.requester: str | None = None
        self.trackers: dict[str, parsers.StepTracker] = {}
        self.msgs: collections.deque = collections.deque(maxlen=4000)
        self.seq = 0
        self.slots: dict[int, dict] = {}
        self.tokens = {"generated": 0, "requests_done": 0, "tps_current": None, "prompt_in_flight": [],
                       "draft_accepted": 0, "draft_generated": 0}
        self.load = {"start_ts": None, "ready_ts": None, "cache_state": None, "offloaded": None, "load_s": None}
        self.pending_prompt_ids: collections.deque = collections.deque()
        self.open_requests = 0                    # engines without slot logs (torch_tp) drive state from events
        self.req_open: dict | None = None         # {prompt_id, t} of the request in flight (events only)
        self.last_step_ms: float | None = None    # from the last response event; estimates progress in flight
        self.last_n_gen: int | None = None
        self.done = False
        self.notes: list[str] = []
        for p in peers or []:
            self.add_peer(p["name"], p.get("role", "worker"), p.get("backend"))

    # ---- peers ---------------------------------------------------------
    def add_peer(self, name: str, role: str, backend: str | None = None):
        if name not in self.peers:
            self.peers[name] = {"name": name, "role": role, "backend": backend, "state": "idle", "util": None,
                                "util_measured": False, "buffers_mib": {}, "buffers_gb": 0.0, "mem_mib": None,
                                "device": None, "last_t": None}
            if role == "worker":
                self.trackers[name] = parsers.StepTracker()
        if role == "requester":
            self.requester = name
        if backend and not self.peers[name]["backend"]:
            self.peers[name]["backend"] = backend
        return self.peers[name]

    def _requester(self) -> dict:
        if self.requester is None:
            self.add_peer("requester", "requester")
        return self.peers[self.requester]

    def _worker_for_device(self, device: str) -> dict | None:
        if device.startswith("RPC"):
            ip = device[device.index("[") + 1:-1].rsplit(":", 1)[0]
            for n, p in self.peers.items():
                if p["role"] == "worker" and (p.get("ip") == ip or len(self.trackers) == 1):
                    p.setdefault("ip", ip)
                    return p
            return self.add_peer(ip, "worker")
        return self._requester()

    # ---- phases --------------------------------------------------------
    def set_phase(self, name: str, t: float):
        if name != self.phase:
            self.phase = name
            self.phases.append({"name": name, "t": t})

    def wall(self, ts: float | None, now: float) -> float:
        if self.mode == "replay" and ts is not None and self.anchor is not None:
            return self.anchor + ts
        return now

    # ---- feeders -------------------------------------------------------
    def feed_server(self, line: str, now: float):
        ev = parsers.parse_server_line(line)
        if not ev:
            return
        with self.lock:
            t = self.wall(ev["ts"], now)
            k = ev["kind"]
            if k == "device":
                if ev["device"].startswith("RPC"):
                    w = self._worker_for_device(ev["device"] + "[" + ev["desc"] + "]")
                    w["mem_mib"] = ev["mib"]
                else:
                    r = self._requester()
                    if r["device"] is None:
                        r["device"], r["mem_mib"] = ev["device"], ev["mib"]
                        r["backend"] = r["backend"] or ("metal" if ev["device"].startswith("MTL") else
                                                        "cuda" if ev["device"].startswith("CUDA") else r["backend"])
            elif k == "load_start":
                self.load["start_ts"] = ev["ts"]
                if self.anchor is None and ev["ts"] is not None:
                    self.anchor = now - ev["ts"]
                self.set_phase("load", t)
                self._requester()["state"] = "loading"
                for w in self.trackers.values():
                    w.state = "loading"
            elif k == "buffer":
                p = self._worker_for_device(ev["device"])
                p["buffers_mib"][f"{ev['device']} {ev['what']}"] = ev["mib"]
                p["buffers_gb"] = round(sum(p["buffers_mib"].values()) / 1024, 3)
            elif k == "offloaded":
                self.load["offloaded"] = ev["text"]
            elif k == "listening":
                self.load["ready_ts"] = ev["ts"]
                if self.load["start_ts"] is not None and ev["ts"] is not None:
                    self.load["load_s"] = round(ev["ts"] - self.load["start_ts"], 2)
                self._requester()["state"] = "idle"
            elif k == "task_start":
                pid = self.pending_prompt_ids.popleft() if self.pending_prompt_ids else None
                self.slots[ev["slot"]] = {"task": ev["task"], "prompt_id": pid, "n_prompt": ev.get("n_prompt"), "n_gen": 0,
                                          "state": "prefill", "t_start": t, "tps": None}
                self.tokens["prompt_in_flight"] = [s["prompt_id"] or f"task {s['task']}" for s in self.slots.values()
                                                   if s["state"] in ("prefill", "decode")]
                self.set_phase("prefill", t)
                self._requester()["state"] = "computing"
            elif k == "prompt_tokens" and ev["slot"] in self.slots:
                self.slots[ev["slot"]]["n_prompt"] = ev["n_prompt"]
            elif k == "prompt_done" and ev["slot"] in self.slots:
                s = self.slots[ev["slot"]]
                s["state"], s["prompt_ms"], s["n_prompt"] = "decode", ev["ms"], ev["n"]
                self.set_phase("decode", t)
            elif k == "progress" and ev["slot"] in self.slots:
                s = self.slots[ev["slot"]]
                s["n_gen"], s["tps"] = ev["n_gen"], ev["tg_3s"]
                s["state"] = "decode"
                self.tokens["tps_current"] = ev["tg_3s"]
                self.set_phase("decode", t)
            elif k == "eval_done" and ev["slot"] in self.slots:
                s = self.slots[ev["slot"]]
                s["n_gen"], s["eval_ms"] = ev["n"], ev["ms"]
                s["tps"] = round(ev["n"] / (ev["ms"] / 1000), 2) if ev["ms"] else None
                self.tokens["tps_current"] = s["tps"]
                self.tokens["generated"] += ev["n"]
                self.tokens["requests_done"] += 1
                self.set_phase("decode", t)
            elif k == "draft":
                self.tokens["draft_accepted"] += ev["accepted"]
                self.tokens["draft_generated"] += ev["generated"]
            elif k == "release" and ev["slot"] in self.slots:
                self.slots[ev["slot"]]["state"] = "done"
                self.tokens["prompt_in_flight"] = [s["prompt_id"] or f"task {s['task']}" for s in self.slots.values()
                                                   if s["state"] in ("prefill", "decode")]
            elif k == "idle":
                self._requester()["state"] = "idle"

    def feed_worker(self, name: str, line: str, now: float):
        ev = parsers.parse_worker_line(line)
        if not ev:
            return
        with self.lock:
            if name not in self.trackers:
                self.add_peer(name, "worker")
            tr = self.trackers[name]
            msg = tr.feed(ev["cmd"], ev["size"])
            p = self.peers[name]
            p["last_t"] = now
            p["state"] = tr.state
            if tr.state == "computing":
                self._requester()["state"] = "waiting"
            elif tr.state == "waiting" and self.phase in ("prefill", "decode"):
                self._requester()["state"] = "computing"
            if ev["cmd"] == "set_tensor_hash":
                self.load["cache_state"] = "warm"
            elif ev["cmd"] == "set_tensor" and (ev["size"] or 0) > 10 << 20 and self.load["cache_state"] is None:
                self.load["cache_state"] = "cold"
            if msg:
                self.seq += 1
                msg.update({"seq": self.seq, "peer": name, "t": now})
                self.msgs.append(msg)

    def feed_util(self, name: str, line: str, now: float):
        s = parsers.parse_util_line(line)
        if not s:
            return
        with self.lock:
            p = self.add_peer(name, "requester" if name == self.requester else self.peers.get(name, {}).get("role", "worker"))
            p["util"], p["util_measured"] = min(100.0, s[1]), True

    def feed_event(self, line: str, now: float):
        ev = parsers.parse_event_line(line)
        if not ev:
            return
        with self.lock:
            t = ev.get("t", now) if self.mode == "replay" else now
            e = ev["event"]
            if e == "peers":
                for p in ev.get("peers", []):
                    self.add_peer(p["name"], p["role"], p.get("backend"))
                    if p.get("ip"):
                        self.peers[p["name"]]["ip"] = p["ip"]
            elif e == "launch":
                # live: the launch event is the clock anchor for server.log timestamps. In replay the Replayer set
                # the anchor from the schedule start; resetting it here shifted every later line (I025 10).
                if self.mode != "replay" or self.anchor is None:
                    self.anchor = t
                self.set_phase("load", t)
            elif e == "healthy":
                self.load["load_s"] = self.load["load_s"] or ev.get("load_s")
            elif e == "request":
                self.pending_prompt_ids.append(ev.get("prompt_id"))
                self.open_requests += 1
                self.req_open = {"prompt_id": ev.get("prompt_id"), "t": ev.get("t", t)}   # event time: a late attach replays old lines with t = now
                if self.phase == "load":
                    self.set_phase("prefill", t)       # the slot lines (if the engine writes them) take over from here
                if not self.slots:
                    self.tokens["prompt_in_flight"] = [ev.get("prompt_id")]
            elif e == "response":
                self.open_requests = max(0, self.open_requests - 1)
                n = ev.get("n_gen")
                if self.req_open and n and n > 1:
                    dt = max(ev.get("t", t) - self.req_open["t"], 1e-3)
                    self.last_step_ms, self.last_n_gen = dt * 1000.0 / (n - 1), n
                    if not self.slots:
                        self.set_phase("decode", t)
                        self.tokens["tps_current"] = (n - 1) / dt
                        self.tokens["generated"] += n
                        self.tokens["requests_done"] += 1
                self.req_open = None
                if not self.slots:
                    self.tokens["prompt_in_flight"] = []
            elif e == "done":
                self.done = True
                self.set_phase("done", t)
                for p in self.peers.values():
                    p["state"] = "idle"
            elif e in PHASES:
                self.set_phase(e, t)

    # ---- output --------------------------------------------------------
    def snapshot(self, now: float) -> dict:
        with self.lock:
            peers = []
            active = any(s["state"] in ("prefill", "decode") for s in self.slots.values()) or self.open_requests > 0
            worker_states = []
            for n, p in self.peers.items():
                q = dict(p)
                if p["role"] == "worker" and p["last_t"] and now - p["last_t"] > 2.0 and q["state"] in ("computing", "waiting"):
                    q["state"] = "idle"
                if p["role"] == "worker" and n in self.trackers:
                    q["counters"] = self.trackers[n].snapshot()
                    worker_states.append(q["state"])
                peers.append(q)
            # The requester's state is derived, not logged: it computes whenever a request is open and no worker
            # is computing, waits while a worker computes, loads until the server listens.
            for q in peers:
                if q["role"] == "requester" and not self.done:
                    if self.phase == "load":
                        q["state"] = "loading"
                    elif "computing" in worker_states:
                        q["state"] = "waiting"
                    elif active:
                        q["state"] = "computing"
                    else:
                        q["state"] = "idle"
            cur = next((p for p in self.phases if p["name"] == self.phase), None)
            tokens = dict(self.tokens)
            if self.req_open and not self.slots:
                el = max(now - self.req_open["t"], 0.0)
                est = min(self.last_n_gen, int(el * 1000.0 / self.last_step_ms) + 1) if self.last_step_ms and self.last_n_gen else None
                tokens["in_flight"] = {"prompt_id": self.req_open["prompt_id"], "elapsed_s": round(el, 1),
                                       "est_tokens": est, "est_of": self.last_n_gen}
            return {"run_id": self.run_id, "mode": self.mode, "now": now, "t0": self.phases[0]["t"],
                    "phase": self.phase, "phase_elapsed": round(now - cur["t"], 1) if cur else None,
                    "phases": self.phases, "peers": peers, "slots": self.slots, "tokens": tokens,
                    "load": self.load, "done": self.done, "notes": self.notes, "seq": self.seq}

    def messages_since(self, seq: int, limit: int = 300) -> list[dict]:
        with self.lock:
            out = [m for m in self.msgs if m["seq"] > seq]
        return out[-limit:]


# ---- feeding from a directory ------------------------------------------------

def peers_from_record(logdir: Path) -> list[dict]:
    """results/<id>.json next to logs/ names the peers; without it the log file names do."""
    rec = logdir.parent.parent / f"{logdir.name}.json"
    peers = []
    if rec.exists():
        try:
            d = json.loads(rec.read_text())
            cmds = d.get("commands", [])
            names = []
            for c in cmds:
                if c["host"] not in names:
                    names.append(c["host"])
            for p, name in zip(d.get("peers", []), names or [p["host"] for p in d.get("peers", [])]):
                peers.append({"name": name, "role": p["role"], "backend": p.get("backend")})
        except (json.JSONDecodeError, KeyError):
            peers = []
    if not peers:
        peers = [{"name": "requester", "role": "requester", "backend": None}]
        for w in sorted(logdir.glob(WORKER_GLOB)):
            peers.append({"name": w.stem[len("worker_"):], "role": "worker", "backend": None})
    return peers


def _route(state: RunState, path: Path, line: str, now: float):
    n = path.name
    if n == "server.log":
        state.feed_server(line, now)
    elif n.startswith("worker_") and n.endswith(".log"):
        state.feed_worker(n[len("worker_"):-4], line, now)
    elif n.startswith("util_") and n.endswith(".tsv"):
        state.feed_util(n[len("util_"):-4], line, now)
    elif n == "events.jsonl":
        state.feed_event(line, now)
    elif n == "placement.log":
        state.feed_server(line, now)


class Tailer(threading.Thread):
    """Polls the directory every `interval` seconds and feeds new lines. Files may appear after start."""

    def __init__(self, logdir: Path, state: RunState, interval: float = 0.1):
        super().__init__(daemon=True)
        self.logdir, self.state, self.interval = Path(logdir), state, interval
        self.offsets: dict[Path, int] = {}
        self.partial: dict[Path, str] = {}
        self._stop = threading.Event()

    def poll_once(self, now: float | None = None):
        now = now or time.time()
        for p in sorted(self.logdir.glob("*")):
            if p.name == "capture.tsv" or p.suffix == ".json" or p.name == "placement.log" or not p.is_file():
                continue
            off = self.offsets.get(p, 0)
            try:
                size = p.stat().st_size
                if size <= off:
                    continue
                with p.open("rb") as f:
                    f.seek(off)
                    chunk = f.read(size - off)
            except OSError:
                continue
            self.offsets[p] = size
            text = self.partial.pop(p, "") + chunk.decode("utf-8", errors="replace")
            lines = text.split("\n")
            if not text.endswith("\n"):
                self.partial[p] = lines.pop()
            else:
                lines.pop()
            for line in lines:
                _route(self.state, p, line, now)

    def run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ---- replay --------------------------------------------------------------------

def build_schedule(logdir: Path) -> list[tuple[float, str, str]]:
    """(relative time, file name, line) sorted by time. Server lines keep their own timestamps. Worker lines
    before the first graph message are spread over the load phase; compute steps are divided between requests
    in proportion to generated tokens and spread over each request's prefill plus decode span (approximate,
    the worker log has no clock)."""
    sched: list[tuple[float, str, str]] = []
    server = (logdir / "server.log").read_text(errors="replace").splitlines() if (logdir / "server.log").exists() else []
    last = 0.0
    load_start = ready = None
    requests: list[list[float]] = []       # [t_start, t_end, n_gen]
    open_by_slot: dict[int, list[float]] = {}
    for line in server:
        ts = parsers.server_ts(line)
        if ts is not None:
            last = ts
        sched.append((last, "server.log", line))
        ev = parsers.parse_server_line(line)
        if not ev:
            continue
        if ev["kind"] == "load_start":
            load_start = last
        elif ev["kind"] == "listening":
            ready = last
        elif ev["kind"] == "task_start":
            open_by_slot[ev["slot"]] = [last, last, 0]
            requests.append(open_by_slot[ev["slot"]])
        elif ev["kind"] == "eval_done" and ev["slot"] in open_by_slot:
            r = open_by_slot.pop(ev["slot"])
            r[1], r[2] = last, ev["n"]
    end = last
    events = logdir / "events.jsonl"
    launch_wall = None
    if events.exists():
        for line in events.read_text(errors="replace").splitlines():
            ev = parsers.parse_event_line(line)
            if ev and ev["event"] == "launch":
                launch_wall = ev.get("t")
        if launch_wall is not None:
            for line in events.read_text(errors="replace").splitlines():
                ev = parsers.parse_event_line(line)
                if ev and "t" in ev:
                    sched.append((ev["t"] - launch_wall, "events.jsonl", line))
            for u in logdir.glob("util_*.tsv"):
                for line in u.read_text(errors="replace").splitlines():
                    s = parsers.parse_util_line(line)
                    if s:
                        sched.append((s[0] - launch_wall, u.name, line))
    for w in sorted(logdir.glob(WORKER_GLOB)):
        lines = w.read_text(errors="replace").splitlines()
        if launch_wall is not None:
            stamped = [(parsers.parse_worker_line(l) or {}).get("ts") for l in lines]
            if any(stamped):
                # the runner stamped these lines: use the real clock, no interleaving needed
                last_t = (load_start or 0.0)
                for l, ts in zip(lines, stamped):
                    if ts is not None:
                        last_t = ts - launch_wall
                    sched.append((last_t, w.name, l))
                continue
        lines = [l.split(" ", 1)[1] if parsers.WORKER.match(l) and (parsers.parse_worker_line(l) or {}).get("ts") else l for l in lines]
        first_graph = next((i for i, l in enumerate(lines) if l.startswith("[graph_")), None)
        if first_graph is None:
            # no compute in this log: spread everything over the load phase
            a, b = (load_start or 0.0), (ready or end or 1.0)
            for i, l in enumerate(lines):
                sched.append((a + (b - a) * i / max(1, len(lines)), w.name, l))
            continue
        a, b = (load_start or 0.0), (ready or load_start or 1.0)
        for i, l in enumerate(lines[:first_graph]):
            sched.append((a + (b - a) * i / max(1, first_graph), w.name, l))
        steps: list[list[str]] = []
        for l in lines[first_graph:]:
            if l.startswith("[graph_"):
                steps.append([l])
            elif steps:
                steps[-1].append(l)
        total_tokens = sum(r[2] for r in requests) or 1
        reqs = [r for r in requests if r[1] > r[0]] or [[ready or 0.0, end, total_tokens]]
        idx = 0
        for ri, (t_a, t_b, n) in enumerate(reqs):
            k = len(steps) - idx if ri == len(reqs) - 1 else round(len(steps) * n / total_tokens)
            chunk = steps[idx: idx + k]
            idx += k
            for si, st in enumerate(chunk):
                s0 = t_a + (t_b - t_a) * si / max(1, len(chunk))
                s1 = t_a + (t_b - t_a) * (si + 1) / max(1, len(chunk))
                for li, l in enumerate(st):
                    sched.append((s0 + (s1 - s0) * li / max(1, len(st)), w.name, l))
        for st in steps[idx:]:
            for l in st:
                sched.append((end, w.name, l))
    sched.sort(key=lambda x: x[0])
    return sched


class Replayer(threading.Thread):
    def __init__(self, logdir: Path, state: RunState, speed: float = 1.0):
        super().__init__(daemon=True)
        self.logdir, self.state, self.speed = Path(logdir), state, speed
        self.schedule = build_schedule(self.logdir)
        self._stop = threading.Event()

    def run(self):
        if not self.schedule:
            self.state.notes.append("nothing to replay in this directory")
            return
        t_first = self.schedule[0][0]
        start = time.time()
        self.state.anchor = start - t_first
        for rel, name, line in self.schedule:
            due = start + (rel - t_first) / self.speed
            while not self._stop.is_set() and time.time() < due:
                time.sleep(min(0.05, due - time.time()))
            if self._stop.is_set():
                return
            _route(self.state, self.logdir / name, line, time.time())
        self.state.feed_event(json.dumps({"event": "done", "t": time.time()}), time.time())

    def stop(self):
        self._stop.set()
