"""Rank 0 generation loop and rank 1 service loop over the transport.
Per decode step rank 0 sends one STEP frame (the embedded hidden state of the new token and its position) and both
ranks run the layers, exchanging one ALLREDUCE frame each way per all-reduce (2 per layer). The next token id is never
sent: rank 1 needs only the hidden state, which rank 0 embeds. Counters: all-reduce frames per decode step and bytes.
Greedy decoding, KV cache per rank, stop on EOS or max_tokens.
Rank 0 writes one line per request, per prefill, every `progress_every` decode tokens and at the end (the live view
reads them). When the link breaks mid request the request fails with LinkLost and the next request reconnects; rank 1
keeps its loaded slice across connections."""
import json
import time

import torch

from . import transport
from .config import Qwen2Config, load_config
from .loader import load_rank
from .model import Comm, Qwen2TP
from .shard import Shard

CPU_ADD_MAX = 1 << 16        # elements; the sum of larger (prefill) frames is formed on the device
REDUCE_DTYPES = ("bfloat16", "float32")


class LinkLost(RuntimeError):
    """The connection to rank 1 broke during a request. The request is lost; the next one reconnects."""


def chat_ids(tok, messages: list[dict]) -> list[int]:
    """Token ids of the chat template with the generation prompt. transformers returns a list of ints in older
    versions and a BatchEncoding (with a nested input_ids list) in newer ones; both become a flat list here."""
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def pick_device(name: str | None = None) -> str:
    if name:
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint16).cpu().numpy().tobytes() if t.dtype == torch.bfloat16 else t.contiguous().cpu().numpy().tobytes()


def _from_bytes(b, shape, dtype: str, device: str) -> torch.Tensor:
    import numpy as np
    if dtype == "bfloat16":
        return torch.from_numpy(np.frombuffer(b, dtype=np.uint16).copy()).view(torch.bfloat16).reshape(shape).to(device)
    return torch.from_numpy(np.frombuffer(b, dtype=np.dtype(dtype)).copy()).reshape(shape).to(device)


def _wire(xc: torch.Tensor) -> memoryview:
    """Zero copy byte view of a contiguous bf16 host tensor."""
    return memoryview(xc.view(torch.uint16).numpy()).cast("B")


STAGES = ("to_cpu", "send_queued", "recv_done", "to_device", "add")


class LinkComm(Comm):
    """Two ranks: send my partial, receive the peer's, add. The partial comes to the host once (.cpu(), which also
    waits for the device), goes on the wire as a view, the peer's frame lands in a reused receive buffer, the sum is
    formed on the host (in bf16, or in fp32 and rounded once when reduce_dtype is float32) and uploaded once. Frames
    above CPU_ADD_MAX elements (prefill) are summed on the device instead. Stage names in the trace keep their
    session 3 meaning where possible: to_cpu is the download with the device wait, send_queued the send, recv_done
    the wait for the peer's frame, to_device now holds the host add plus the upload, add only the device sync that
    makes the upload measurable."""

    def __init__(self, link: transport.Link, device: str, trace: bool = False, reduce_dtype: str = "bfloat16"):
        if reduce_dtype not in REDUCE_DTYPES:
            raise ValueError(f"reduce_dtype must be one of {REDUCE_DTYPES}")
        self.link, self.device, self.trace = link, device, trace
        self.reduce_fp32 = reduce_dtype == "float32"
        self.trace_steps: list[dict] = []
        self._cur: dict | None = None
        self._rbuf = bytearray()

    def begin_step(self) -> None:
        if self.trace:
            self._cur = {k: 0 for k in STAGES} | {"n": 0, "total": 0}

    def end_step(self) -> None:
        if self.trace and self._cur and self._cur["n"]:
            self.trace_steps.append(self._cur)
        self._cur = None

    def _recv_buf(self, nbytes: int) -> memoryview:
        if len(self._rbuf) < nbytes:
            self._rbuf = bytearray(nbytes)
        return memoryview(self._rbuf)[:nbytes]

    def _send(self, wire: memoryview, shape):
        if len(wire) <= transport.SMALL_FRAME:
            self.link.send(transport.KIND_ALLREDUCE, wire, "bfloat16", shape)
            return None
        return self.link.send_async(transport.KIND_ALLREDUCE, bytes(wire), "bfloat16", shape)

    def _recv(self, nbytes: int, pending) -> torch.Tensor:
        kind, payload, dtype, shape, _, _ = self.link.recv(into=self._recv_buf(nbytes))
        if pending is not None:
            pending.join()
        if kind != transport.KIND_ALLREDUCE:
            raise RuntimeError(f"expected an all-reduce frame, got {transport.KIND_NAMES.get(kind, kind)}")
        if dtype != "bfloat16":
            raise RuntimeError(f"all-reduce frame dtype {dtype}, expected bfloat16")
        buf = payload if isinstance(payload, memoryview) else bytearray(payload)
        return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape)

    def _sum(self, x: torch.Tensor, xc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        if xc.numel() > CPU_ADD_MAX:
            y = yc.to(self.device)
            return (x.float() + y.float()).to(x.dtype) if self.reduce_fp32 else x + y
        s = (xc.float() + yc.float()).to(torch.bfloat16) if self.reduce_fp32 else xc + yc
        return s.to(self.device)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self.trace:
            xc = x.contiguous().cpu()
            wire = _wire(xc)
            pending = self._send(wire, tuple(xc.shape))
            yc = self._recv(len(wire), pending)
            return self._sum(x, xc, yc)
        ns = time.perf_counter_ns
        t0 = ns()
        xc = x.contiguous().cpu(); wire = _wire(xc); t1 = ns()
        pending = self._send(wire, tuple(xc.shape)); t2 = ns()
        yc = self._recv(len(wire), pending); t3 = ns()
        out = self._sum(x, xc, yc); t4 = ns()
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        t5 = ns()
        if self._cur is None:
            self.begin_step()
        c = self._cur
        for k, d in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            c[k] += d
        c["n"] += 1
        c["total"] += t5 - t0
        return out

    def trace_summary(self, skip_first: int = 1) -> dict | None:
        """Mean ns per all-reduce per stage over the traced steps (prefill steps included unless skipped by the caller)
        and the step totals in ms."""
        steps = self.trace_steps[skip_first:] if len(self.trace_steps) > skip_first else self.trace_steps
        if not steps:
            return None
        n = sum(s["n"] for s in steps)
        per_ar = {k: sum(s[k] for s in steps) / n / 1e6 for k in STAGES}
        per_ar["total"] = sum(s["total"] for s in steps) / n / 1e6
        totals = sorted(s["total"] / 1e6 for s in steps)
        return {"steps": len(steps), "allreduces": n, "ms_per_allreduce": per_ar,
                "ms_per_step_in_allreduce": {"mean": sum(totals) / len(totals), "p50": totals[len(totals) // 2], "max": totals[-1]}}


def mem_metrics(device: str) -> dict:
    if device == "mps":
        return {"current_allocated_bytes": torch.mps.current_allocated_memory(), "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    if device == "cuda":
        return {"current_allocated_bytes": torch.cuda.memory_allocated(), "driver_allocated_bytes": torch.cuda.memory_reserved()}
    return {"current_allocated_bytes": None, "driver_allocated_bytes": None}


def _flush_print(*a, **kw):
    print(*a, **kw, flush=True)


class Rank0:
    def __init__(self, model_dir: str, world: int, max_seq_len: int, device: str | None = None, peer: str | None = None,
                 log=_flush_print, trace_allreduce: bool = False, reduce_dtype: str = "bfloat16", progress_every: int = 16):
        if reduce_dtype not in REDUCE_DTYPES:
            raise ValueError(f"reduce_dtype must be one of {REDUCE_DTYPES}")
        self.log = log
        self.trace_allreduce = trace_allreduce
        self.reduce_dtype = reduce_dtype
        self.progress_every = progress_every
        self.device = pick_device(device)
        self.log(f"[torch_tp] loading {model_dir}")
        self.cfg: Qwen2Config = load_config(model_dir)
        self.cfg.shard_ok(world)
        self.world, self.max_seq_len, self.peer = world, max_seq_len, peer
        self.shard = Shard(0, world, self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim, self.cfg.intermediate_size)
        t0 = time.time()
        weights, self.loaded_bytes = load_rank(model_dir, self.cfg, self.shard, self.device, log=log)
        self.model = Qwen2TP(self.cfg, weights, self.shard, self.device, max_seq_len)
        self.load_s = time.time() - t0
        log(f"[torch_tp] rank 0 loaded {self.loaded_bytes / 1e9:.2f} GB in {self.load_s:.1f} s on {self.device}")
        self.link: transport.Link | None = None
        self.comm: Comm = Comm()
        self.counters = transport.Counters()
        if world > 1:
            if not peer:
                raise ValueError("world > 1 needs --peer host:port of rank 1")
            self._connect(120.0)
        self.decode_steps = 0
        self.tokens_generated = 0
        self.prefill_steps = 0

    # ---- link ------------------------------------------------------------
    def _connect(self, timeout_s: float) -> None:
        host, port = self.peer.rsplit(":", 1)
        sock = transport.connect(host, int(port), timeout_s=timeout_s)
        link = transport.Link(sock, counters=self.counters)
        link.send(transport.KIND_HELLO, json.dumps({"world": self.world, "max_seq_len": self.max_seq_len,
                                                    "reduce_dtype": self.reduce_dtype}).encode(), "uint8")
        kind, payload, *_ = link.recv()
        if kind != transport.KIND_HELLO:
            link.close()
            raise RuntimeError("rank 1 did not answer the hello")
        self.link = link
        self.comm = LinkComm(link, self.device, trace=self.trace_allreduce, reduce_dtype=self.reduce_dtype)
        self.log(f"[torch_tp] rank 1 ready: {payload.decode()}")

    def _ensure_link(self) -> None:
        if self.world > 1 and self.link is None:
            self.log(f"[torch_tp] reconnecting to rank 1 at {self.peer}")
            self._connect(60.0)
            self.counters.reconnects += 1
            self.log("[torch_tp] reconnected")

    def _link_lost(self, e: Exception, rid: str) -> None:
        self.log(f"[torch_tp] link lost during {rid}: {e}; the request fails, the next request reconnects")
        if self.link is not None:
            self.link.close()
        self.link = None
        self.comm = Comm() if self.world == 1 else None

    # ---- generation --------------------------------------------------------
    def generate(self, ids: list[int], max_tokens: int, on_token=None, ignore_eos: bool = False, rid: str = "req-0") -> dict:
        """Greedy. Returns tokens, finish_reason and llama-server style timings: predicted_ms covers the predicted_n - 1
        steps after the first token, and predicted_per_second divides by those steps, as llama-server does."""
        self._ensure_link()
        self.log(f"[torch_tp] request {rid} prompt_n={len(ids)} max_tokens={max_tokens}")
        try:
            return self._generate(ids, max_tokens, on_token, ignore_eos, rid)
        except (ConnectionError, OSError) as e:
            self._link_lost(e, rid)
            raise LinkLost(str(e)) from e

    def _generate(self, ids, max_tokens, on_token, ignore_eos, rid) -> dict:
        self._reset()
        t0 = time.time()
        h = self.model.embed_tokens(ids)
        before = self.counters.snapshot() if self.link else None
        h = self._step(h, 0)
        nxt = int(torch.argmax(self.model.logits_last(h)).item())
        self.prefill_steps += 1
        t1 = time.time()
        self.log(f"[torch_tp] prefill {rid} n={len(ids)} ms={(t1 - t0) * 1000:.1f}")
        out = [nxt]
        if on_token:
            on_token(nxt)
        finish = "length"
        pos = len(ids)
        while True:
            if nxt in self.cfg.eos_token_ids and not ignore_eos:
                finish = "stop"
                break
            if len(out) >= max_tokens:
                break
            h = self.model.embed_tokens([nxt])
            h = self._step(h, pos)
            nxt = int(torch.argmax(self.model.logits_last(h)).item())
            pos += 1
            out.append(nxt)
            self.decode_steps += 1
            if on_token:
                on_token(nxt)
            if self.progress_every and len(out) % self.progress_every == 0:
                self.log(f"[torch_tp] decode {rid} n_gen={len(out)} ms_per_step={(time.time() - t1) * 1000 / (len(out) - 1):.1f}")
        t2 = time.time()
        n = len(out)
        self.tokens_generated += n
        after = self.counters.snapshot() if self.link else None
        predicted_ms = (t2 - t1) * 1000
        steps = n - 1
        tm = {"prompt_n": len(ids), "prompt_ms": (t1 - t0) * 1000, "predicted_n": n, "predicted_ms": predicted_ms,
              "predicted_per_token_ms": (predicted_ms / steps) if steps else None,
              "predicted_per_second": (steps / (predicted_ms / 1000)) if steps and predicted_ms > 0 else None,
              "prompt_per_second": len(ids) / (t1 - t0) if t1 > t0 else None}
        frames = None
        if before and after:
            frames = {k: after[k] - before[k] for k in after}
        self.log(f"[torch_tp] done {rid} n_gen={n} predicted_ms={predicted_ms:.1f} finish={finish}")
        self.log(f"[torch_tp] release {rid}")
        return {"tokens": out, "finish_reason": finish, "timings": tm, "frames": frames}

    def _reset(self):
        if self.link:
            self.link.send(transport.KIND_RESET)

    def _step(self, h: torch.Tensor, pos0: int) -> torch.Tensor:
        if self.link:
            self.link.send(transport.KIND_STEP, _to_bytes(h), "bfloat16", tuple(h.shape), aux0=pos0)
            if self.trace_allreduce:
                self.comm.begin_step()
        out = self.model.layers_forward(h, pos0, self.comm)
        if self.link and self.trace_allreduce:
            self.comm.end_step()
        return out

    def metrics(self) -> dict:
        ranks = {"0": dict(mem_metrics(self.device), loaded_tensor_bytes=self.loaded_bytes, device=self.device)}
        if self.link:
            try:
                self.link.send(transport.KIND_METRICS)
                kind, payload, *_ = self.link.recv()
                if kind == transport.KIND_METRICS_REPLY:
                    ranks["1"] = json.loads(payload.decode())
            except (ConnectionError, OSError) as e:
                self._link_lost(e, "metrics")
        m = {"world": self.world, "ranks": ranks, "counters": self.counters.snapshot() if self.world > 1 else None,
             "decode_steps": self.decode_steps, "prefill_steps": self.prefill_steps, "tokens_generated": self.tokens_generated,
             "layers": self.cfg.num_hidden_layers, "allreduces_per_step": 2 * self.cfg.num_hidden_layers,
             "torch": torch.__version__, "load_s": self.load_s, "reduce_dtype": self.reduce_dtype, "link_up": self.link is not None}
        if self.link and self.trace_allreduce:
            m["allreduce_trace"] = self.comm.trace_summary()
        return m

    def shutdown(self):
        if self.link:
            try:
                self.link.send(transport.KIND_SHUTDOWN)
            except OSError:
                pass
            self.link.close()
            self.link = None


def serve_rank(model_dir: str, rank: int, world: int, bind: str, port: int, device: str | None = None, log=_flush_print) -> None:
    """Rank >= 1: load the slice at the first hello, accept rank 0, run steps until SHUTDOWN. When rank 0 goes away the
    slice stays loaded and the socket accepts again (a reconnect takes the hello round trip, not a reload). A
    connection that closes before HELLO (a port probe) is ignored."""
    device = pick_device(device)
    cfg = load_config(model_dir)
    cfg.shard_ok(world)
    shard = Shard(rank, world, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    srv = transport.listen(bind, port)
    model = weights = None
    loaded_for = None
    nbytes = load_s = 0
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
        hello = json.loads(payload.decode())
        log(f"[torch_tp] rank {rank}: hello from {addr}, {hello}")
        max_seq = int(hello["max_seq_len"])
        if model is None or loaded_for != max_seq:
            model = weights = None
            if device == "mps":
                torch.mps.empty_cache()
            t0 = time.time()
            weights, nbytes = load_rank(model_dir, cfg, shard, device, log=log)
            model = Qwen2TP(cfg, weights, shard, device, max_seq)
            load_s = time.time() - t0
            loaded_for = max_seq
            log(f"[torch_tp] rank {rank} loaded {nbytes / 1e9:.2f} GB in {load_s:.1f} s on {device}")
            cached = False
        else:
            log(f"[torch_tp] rank {rank}: slice kept from the previous connection")
            cached = True
        link.send(transport.KIND_HELLO, json.dumps({"rank": rank, "loaded_bytes": nbytes, "device": device, "load_s": load_s,
                                                    "cached": cached}).encode(), "uint8")
        comm = LinkComm(link, device, reduce_dtype=hello.get("reduce_dtype", "bfloat16"))
        try:
            while True:
                kind, payload, dtype, shape, pos0, _ = link.recv()
                if kind == transport.KIND_STEP:
                    h = _from_bytes(payload, shape, dtype, device)
                    model.layers_forward(h, pos0, comm)
                elif kind == transport.KIND_RESET:
                    pass
                elif kind == transport.KIND_METRICS:
                    m = dict(mem_metrics(device), loaded_tensor_bytes=nbytes, device=device, load_s=load_s, counters=link.counters.snapshot())
                    link.send(transport.KIND_METRICS_REPLY, json.dumps(m).encode(), "uint8")
                elif kind == transport.KIND_SHUTDOWN:
                    log(f"[torch_tp] rank {rank}: shutdown")
                    link.close()
                    srv.close()
                    return
                else:
                    raise RuntimeError(f"unexpected frame {transport.KIND_NAMES.get(kind, kind)}")
        except (ConnectionError, OSError) as e:
            log(f"[torch_tp] rank {rank}: rank 0 went away ({e}); slice kept, waiting for a new rank 0")
            link.close()
