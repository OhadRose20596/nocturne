"""Pure parsers for the files under results/logs/<id>/: server.log (llama-server, -lv 4), worker_<name>.log
(ggml-rpc-server with GGML_RPC_DEBUG=1), util_<peer>.tsv, events.jsonl. No I/O here; tests feed fixture lines."""
import json
import re
import statistics

# "0.15.308.380 I slot print_timing: ..." -> minutes.seconds.ms.us since server start
SERVER_TS = re.compile(r"^(\d+(?:\.\d+)*)\.(\d{3})\.(\d{3})\s+[A-Z]\s+(.*)$")
BUFFER = re.compile(r"(RPC\d*\[[^\]]+\]|MTL\d+(?:_Mapped)?|Metal(?:_Mapped)?|CUDA\d+(?:_Host)?|Vulkan\d+|CPU(?:_Mapped)?)\s+"
                    r"(model|KV|compute|output) buffer size\s*=\s*([\d.]+) MiB")
DEVICE = re.compile(r"common_param:\s+-\s+(\S+)\s+:\s+(.+?)\s+\((\d+) MiB")
SLOT = re.compile(r"id\s+(\d+)\s+\|\s+task\s+(-?\d+)\s+\|\s*(.*)$")
PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
EVAL = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens")
PROGRESS = re.compile(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s, tg_3s =\s*([\d.]+) t/s")
DRAFT = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated")
N_TOKENS = re.compile(r"task\.n_tokens = (\d+)")
# torch_tp rank 0 (no timestamp prefix): "[torch_tp] decode req-3 n_gen=32 ms_per_step=590.1"
TORCH = re.compile(r"\[torch_tp\] (request|prefill|decode|done|release) req-(\d+)(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")
WORKER = re.compile(r"^(?:(\d+\.\d+)\s+)?\[([a-z_~]+)\]\s*(.*)$")
SIZE = re.compile(r"\bsize:\s*(\d+)")

STEP_CMDS = ("graph_compute", "graph_recompute")
STREAM_CMDS = ("set_tensor", "get_tensor", "set_tensor_hash", "graph_compute", "graph_recompute")


def server_ts(line: str) -> float | None:
    """Seconds since llama-server start, from the line prefix. None when the line has no prefix."""
    m = SERVER_TS.match(line)
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(".")]      # [min, sec] or [hour, min, sec]; last is seconds
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs + int(m.group(2)) / 1e3 + int(m.group(3)) / 1e6


def parse_server_line(line: str) -> dict | None:
    """One typed event or None. Every event carries 'kind' and 'ts' (relative seconds, may be None)."""
    ts = server_ts(line)
    m = SERVER_TS.match(line)
    body = m.group(4) if m else line
    ev = {"ts": ts}
    if "[torch_tp]" in body:
        return _parse_torch_line(body, ev)
    if "load_model: loading model '" in body:
        return {**ev, "kind": "load_start", "path": body.split("'")[1]}
    if "common_speculative_init_result: loading draft model" in body:
        return {**ev, "kind": "draft_load"}
    b = BUFFER.search(body)
    if b:
        return {**ev, "kind": "buffer", "device": b.group(1), "what": b.group(2), "mib": float(b.group(3))}
    d = DEVICE.search(body)
    if d:
        return {**ev, "kind": "device", "device": d.group(1), "desc": d.group(2), "mib": int(d.group(3))}
    if "load_tensors: offloaded" in body:
        return {**ev, "kind": "offloaded", "text": body.split("load_tensors: ")[-1]}
    if "llama_server: listening on" in body:
        return {**ev, "kind": "listening", "url": body.rsplit(" ", 1)[-1]}
    if "all slots are idle" in body:
        return {**ev, "kind": "idle"}
    s = SLOT.search(body)
    if s:
        slot, task, rest = int(s.group(1)), int(s.group(2)), s.group(3)
        base = {**ev, "slot": slot, "task": task}
        if "processing task" in rest:
            return {**base, "kind": "task_start"}
        n = N_TOKENS.search(rest)
        if n:
            return {**base, "kind": "prompt_tokens", "n_prompt": int(n.group(1))}
        p = PROMPT_EVAL.search(rest)
        if p:
            return {**base, "kind": "prompt_done", "ms": float(p.group(1)), "n": int(p.group(2))}
        e = EVAL.search(rest)
        if e and "prompt eval" not in rest:
            return {**base, "kind": "eval_done", "ms": float(e.group(1)), "n": int(e.group(2))}
        g = PROGRESS.search(rest)
        if g:
            return {**base, "kind": "progress", "n_gen": int(g.group(1)), "tg": float(g.group(2)), "tg_3s": float(g.group(3))}
        dr = DRAFT.search(rest)
        if dr:
            return {**base, "kind": "draft", "rate": float(dr.group(1)), "accepted": int(dr.group(2)), "generated": int(dr.group(3))}
        if "stop processing" in rest:
            return {**base, "kind": "release"}
    return None


def _parse_torch_line(body: str, ev: dict) -> dict | None:
    """torch_tp rank 0 lines mapped onto the llama-server event kinds (one slot, task = request number)."""
    if "[torch_tp] loading " in body:
        return {**ev, "kind": "load_start", "path": body.split("[torch_tp] loading ", 1)[1].strip()}
    if "[torch_tp] serving http" in body:
        return {**ev, "kind": "listening", "url": body.split("serving ", 1)[1].split(" ")[0]}
    m = TORCH.search(body)
    if not m:
        return None
    what, task = m.group(1), int(m.group(2))
    kv = {k: v for k, v in KV.findall(m.group(3))}
    base = {**ev, "slot": 0, "task": task}
    try:
        if what == "request":
            return {**base, "kind": "task_start", "n_prompt": int(kv["prompt_n"])}
        if what == "prefill":
            return {**base, "kind": "prompt_done", "ms": float(kv["ms"]), "n": int(kv["n"])}
        if what == "decode":
            tg = 1000.0 / float(kv["ms_per_step"]) if float(kv["ms_per_step"]) > 0 else 0.0
            return {**base, "kind": "progress", "n_gen": int(kv["n_gen"]), "tg": tg, "tg_3s": tg}
        if what == "done":
            return {**base, "kind": "eval_done", "ms": float(kv["predicted_ms"]), "n": int(kv["n_gen"])}
    except (KeyError, ValueError):
        return None
    return {**base, "kind": "release"}


def parse_worker_line(line: str) -> dict | None:
    """{'cmd': 'set_tensor', 'size': 14336, 'ts': wall time or None}. The runner prefixes worker lines with a wall
    clock since 2026-10-26; older logs have none."""
    m = WORKER.match(line)
    if not m:
        return None
    s = SIZE.search(m.group(3))
    return {"cmd": m.group(2), "size": int(s.group(1)) if s else None, "ts": float(m.group(1)) if m.group(1) else None}


def parse_util_line(line: str) -> tuple[float, float] | None:
    try:
        t, v = line.split("\t")
        return float(t), float(v)
    except ValueError:
        return None


def parse_event_line(line: str) -> dict | None:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and "event" in d else None


class StepTracker:
    """Decode-step bookkeeping for one worker log. A step starts at graph_compute or graph_recompute; the
    worker computes until it answers the next get_tensor, then waits for the requester. Lines before the first
    graph message are the load phase."""

    def __init__(self, recent: int = 32):
        self.step = 0
        self.cmds_this_step = 0
        self.cmds_total = 0
        self.bytes_in = 0          # set_tensor payload to the worker
        self.bytes_out = 0         # get_tensor payload back
        self.cache_hits = 0        # set_tensor_hash lines (weights served from the rpc-server cache)
        self.cache_bytes = 0
        self.state = "idle"        # loading | computing | waiting | idle
        self.loaded = False
        self.recent = recent
        self._per_step: list[int] = []

    def feed(self, cmd: str, size: int | None) -> dict | None:
        """Returns a stream message for the page, or None for commands that are not shown."""
        size = size or 0
        if cmd in STEP_CMDS:
            if self.step:
                self._per_step.append(self.cmds_this_step)
                self._per_step = self._per_step[-self.recent:]
            self.step += 1
            self.cmds_this_step = 1
            self.cmds_total += 1
            self.loaded = True
            self.state = "computing"
            return {"cmd": cmd, "size": None, "step": self.step, "step_start": True}
        if cmd == "set_tensor":
            self.bytes_in += size
        elif cmd == "get_tensor":
            self.bytes_out += size
            if self.loaded:
                self.state = "waiting"
        elif cmd == "set_tensor_hash":
            self.cache_hits += 1
            self.cache_bytes += size
        elif cmd in ("alloc_buffer", "buffer_clear") and not self.loaded:
            self.state = "loading"
        if cmd not in STREAM_CMDS:
            return None
        if self.loaded:
            self.cmds_this_step += 1
            self.cmds_total += 1
        elif cmd in ("set_tensor", "set_tensor_hash"):
            self.state = "loading"
        return {"cmd": cmd, "size": size, "step": self.step if self.loaded else 0, "step_start": False}

    def snapshot(self) -> dict:
        return {"step": self.step, "cmds_this_step": self.cmds_this_step, "cmds_total": self.cmds_total,
                "bytes_in": self.bytes_in, "bytes_out": self.bytes_out, "bytes_total": self.bytes_in + self.bytes_out,
                "cache_hits": self.cache_hits, "cache_bytes": self.cache_bytes, "state": self.state,
                "cmds_per_step_recent": statistics.median(self._per_step) if self._per_step else None}


def worker_backend(text_head: str) -> str | None:
    if "ggml_metal" in text_head:
        return "metal"
    if "ggml_cuda" in text_head or "CUDA" in text_head:
        return "cuda"
    if "Vulkan" in text_head:
        return "vulkan"
    return None


def format_sse(event: str, data) -> bytes:
    """One SSE frame. data is serialized as JSON on one line; multi-line payloads are split per the spec."""
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    lines = "".join(f"data: {l}\n" for l in payload.split("\n"))
    return f"event: {event}\n{lines}\n".encode()
