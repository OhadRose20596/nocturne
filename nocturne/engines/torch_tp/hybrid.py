"""Hybrid placement runtime (session 6): PP stages, each a TP group of any size with uneven shares.
Processes: one per rank. Rank 0 of the first stage is also the coordinator (tokenizer, HTTP, greedy choice).
Links: a `ctl` link from the coordinator to every other rank; inside a group one link per pair (all-to-all); from the
leader (group rank 0) of a stage to every rank of the next stage. The higher global rank connects to the lower one.
One decode step: the coordinator sends the token id to the other ranks of stage 0; stage 0 embeds (vocab split:
zeros outside a rank's rows, completed by an all-reduce); every stage runs its layers with two all-reduces per layer;
the leader sends the hidden state on; every rank of the head group sends (value, global index) of its local maximum
to the coordinator, which picks the largest value, ties to the lowest index. Control frames (metrics, shutdown)
travel the same path and every rank answers on its ctl link.
All-reduce: every rank sends its partial to every other rank and sums all partials in group rank order on the host,
in fp32 rounded to the activation dtype once (or in bf16 for the control run), so all ranks of a group hold bitwise
identical activations, whatever their device.
Session 10: after READY and before shutdown the coordinator measures every rank's clock offset on the ctl link
(clock.py). CLOCK frames travel only on ctl links: a rank of a later stage waits on its upstream link and on its ctl
link at once. With `timeline` in the plan every rank records spans (recorder.py) and writes them at shutdown.
Session 10 part 3 (overlap, docs/ENGINE_DESIGN.md section 11). Every option is off by default and off is the old path.
A rank never schedules: it handles frames in arrival order, with the slot each frame names (its own KV cache). A frame
flagged as the first of a pair makes a TP group read the partner and run the two in lockstep. Only the coordinator
schedules (generate_many): a slot is ready when its next input exists; the first stage runs for a ready slot while the
later stages still work on another; head results are read without blocking, per slot.
With `ship` in the plan a rank holds no model file: after its data links are up it receives its slices on the ctl
link into a cache and loads them from there (shipping.py). The coordinator sends to all ranks at once while it loads
its own slice from the local file. The model config always travels in the PLAN message.
Security: listeners accept any peer; deferred by ADR-009. Never bind 0.0.0.0 (CLAUDE.md rule 7)."""
import collections
import ctypes
import json
import os
import queue
import re
import select
import resource
import struct
import threading
import time

import numpy as np
import torch

from . import clock, shipping, transport
from .config import Qwen2Config, config_from_json, config_to_json, load_config
from .engine import LinkLost, REDUCE_DTYPES, _from_bytes, mem_metrics, pick_device
from .loader import BudgetExceeded, LoadSpec, load_from_source, load_slices
from .model import Comm, Qwen2TP, pick_global
from .plan import Dims, PlanError, RankSpec, check_overlap, chunk_ranges, needed_peers
from .recorder import CH_CTL, CH_GROUP, CH_PIPE, Recorder, Sampler, ifaces_for

# aux1 of a TOKEN or HIDDEN frame: flags in bits 0 to 7, the request number in bits 8 to 23 (recorder only), the slot in
# bits 24 to 31. aux0 of a HEAD frame: the token index in bits 0 to 23, the slot in bits 24 to 31.
FLAG_LOGITS = 1              # send this rank's logits with the head result
FLAG_NOHEAD = 2              # a prefill chunk that is not the last: no head result
FLAG_PAIR = 4                # the first frame of an interleaved pair: the next frame on the link is its partner
REQ_SHIFT, REQ_MASK = 8, 0xFFFF
SLOT_SHIFT = 24
IDX_MASK = (1 << SLOT_SHIFT) - 1
MAX_SLOTS = 256


def pack_aux(flags: int, req: int, slot: int) -> int:
    return flags | (req & REQ_MASK) << REQ_SHIFT | slot << SLOT_SHIFT
DATA_TIMEOUT_S = None
CTL_TIMEOUT_S = 1800.0
END_CLOCK_TIMEOUT_S = 20.0
TORCH_DT = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


class RankFailed(RuntimeError):
    pass


def _flush_print(*a, **kw):
    print(*a, **kw, flush=True)


# ---- memory of this process ------------------------------------------------------------------------------------------
def idle_footprint(device: str) -> dict:
    """Footprint of this process with the device initialized and nothing loaded (the runtime overhead)."""
    torch.zeros(1, device=device)
    sync(device)
    return footprint()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.split(":")[0] == "cuda":
        torch.cuda.synchronize(device)      # the rank's own device: without the argument torch waits for the current one (GPU 0)


def cuda_metrics(device: str) -> dict:
    """Name, VRAM total and free, and peak allocation of a CUDA rank (empty for other devices)."""
    if device.split(":")[0] != "cuda" or not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"cuda_name": torch.cuda.get_device_name(), "vram_total_bytes": int(total), "vram_free_bytes": int(free),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated())}


def footprint() -> dict:
    """phys_footprint and its lifetime maximum (macOS proc_pid_rusage, flavor 4), and peak RSS (getrusage)."""
    out = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        if os.uname().sysname == "Darwin":
            buf = (ctypes.c_uint64 * 48)()
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0:
                # rusage_info_v4: uuid (16 bytes = 2 words), then uint64 fields; phys_footprint is field 7,
                # lifetime_max_phys_footprint field 28
                out["phys_footprint_bytes"] = int(buf[2 + 7])
                out["lifetime_max_phys_footprint_bytes"] = int(buf[2 + 28])
        else:
            out["peak_rss_bytes"] *= 1024          # Linux reports KiB
    except OSError:
        pass
    return out


# ---- wire helpers --------------------------------------------------------------------------------------------------------
def _host_bytes(xc: torch.Tensor) -> tuple[memoryview, str]:
    if xc.dtype == torch.bfloat16:
        return memoryview(xc.view(torch.uint16).numpy()).cast("B"), "bfloat16"
    return memoryview(xc.numpy()).cast("B"), str(xc.dtype).replace("torch.", "")


def _from_wire(payload, shape, dtype: str) -> torch.Tensor:
    buf = payload if isinstance(payload, (bytearray, memoryview)) else bytearray(payload)
    if dtype == "bfloat16":
        return torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).reshape(shape)
    return torch.frombuffer(buf, dtype=TORCH_DT[dtype]).reshape(shape)


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bits_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


# ---- sending off the compute thread (--async-send) -------------------------------------------------------------------------
ASYNC_COPY = {"cuda": "pinned", "mps": "inline", "cpu": "inline"}
POOL_MAX_BYTES = 256 << 20


class Handoff:
    """One tensor on its way to the host. `host` is valid once `ready` is set."""
    __slots__ = ("host", "ready", "x", "event", "ctx", "pooled")

    def __init__(self, ctx):
        self.host = self.x = self.event = None
        self.ready, self.ctx, self.pooled = threading.Event(), ctx, False


class AsyncHead:
    """A head result still on the device: [max logit, local index] and the local logits."""
    __slots__ = ("pair", "logits")

    def __init__(self, pair, logits):
        self.pair, self.logits = pair, logits


class Sender:
    """One thread per rank. It takes work in the order it was handed over: wait until a tensor is on the host, then
    send its frames. So frames leave every link in hand-over order, and the compute thread never waits for the device
    or the socket. Where the copy to the host runs (`mode`): CUDA `pinned` (a non blocking copy into pinned memory,
    queued by the compute thread; this thread waits for its event), MPS and CPU `inline` (the compute thread copies, so
    only the send moves here). `thread` (this thread copies) is not used: on MPS two threads may not work on the device
    at once, a rank died with a Metal assertion (run OVs_15b_pp3_s2_c16_async_r1). NOCTURNE_ASYNC_COPY overrides the mode."""

    def __init__(self, device: str, rec: Recorder | None, on_fail=None):
        self.device, self.rec = device, rec
        self.on_fail = on_fail              # called once with the exception when this thread fails: the rank closes its links
        self.mode = os.environ.get("NOCTURNE_ASYNC_COPY") or ASYNC_COPY.get(device.split(":")[0], "inline")
        if self.mode not in ("pinned", "thread", "inline"):
            raise ValueError(f"NOCTURNE_ASYNC_COPY must be pinned, thread or inline, got {self.mode}")
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.error: BaseException | None = None
        self.pool: dict[tuple, list] = {}
        self.pool_bytes = 0
        self.pool_lock = threading.Lock()
        self.frames = 0
        self.thread = threading.Thread(target=self._loop, daemon=True, name="nocturne-sender")
        self.thread.start()

    # ---- compute thread ----
    def check(self) -> None:
        if self.error is not None:
            raise ConnectionError(f"the sender thread failed: {type(self.error).__name__}: {self.error}") from self.error

    def copy(self, x: torch.Tensor) -> Handoff:
        """Start bringing `x` to the host. Returns at once, except in `inline` mode."""
        self.check()
        rec = self.rec
        h = Handoff((rec.step, rec.req, rec.tokens, rec.slot) if rec else None)
        x = x.contiguous()
        if self.mode == "pinned":
            with self.pool_lock:
                free = self.pool.get((tuple(x.shape), x.dtype))
                buf = free.pop() if free else None
                if buf is not None:
                    self.pool_bytes -= buf.numel() * buf.element_size()
            if buf is None:
                buf = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
            buf.copy_(x, non_blocking=True)
            h.event = torch.cuda.Event()
            h.event.record(torch.cuda.current_stream(self.device))
            h.host, h.pooled = buf, True
        elif self.mode == "thread":
            h.x = x
        else:                               # the bytes are on the host already: nothing to wait for
            h.host = x.cpu()
            h.ready.set()
            return h
        self.q.put(("copy", h))
        return h

    def send(self, h: Handoff, link: transport.Link, kind: int, aux0: int, aux1: int, decode: bool, peer: str | None,
             ch: str | None = None) -> None:
        self.q.put(("send", h, link, kind, aux0, aux1, decode, peer, ch))

    def head(self, pair: Handoff, logits: Handoff | None, link: transport.Link, slot: int, rows: tuple[int, int], decode: bool,
             peer: str | None) -> None:
        self.q.put(("head", pair, logits, link, slot, rows, decode, peer))

    def free(self, h: Handoff) -> None:
        """The compute thread is done with `h.host`; the buffer returns to the pool after the frames queued so far."""
        self.q.put(("free", h))

    def host(self, h: Handoff) -> torch.Tensor:
        while not h.ready.wait(0.2):
            self.check()
        return h.host

    def flush(self) -> None:
        """Wait until everything handed over so far has left."""
        done = threading.Event()
        self.q.put(("flush", done))
        while not done.wait(0.2):
            self.check()

    def stop(self) -> None:
        self.q.put(("stop",))
        self.thread.join(timeout=3)

    # ---- sender thread ----
    def _loop(self) -> None:
        rec = self.rec
        try:
            while True:
                job = self.q.get()
                op = job[0]
                t0 = clock.now_ns()
                if op == "copy":
                    h = job[1]
                    if h.event is not None:
                        h.event.synchronize()
                        h.event = None
                    elif h.x is not None:
                        h.host, h.x = h.x.cpu(), None
                    h.ready.set()
                    if rec:
                        rec.span("tx_wait", t0, clock.now_ns(), h.ctx)
                elif op == "send":
                    _, h, link, kind, aux0, aux1, decode, peer, ch = job
                    wire, dt = _host_bytes(h.host)
                    link.send(kind, wire, dt, tuple(h.host.shape), aux0=aux0, aux1=aux1, decode=decode)
                    self.frames += 1
                    if rec:
                        rec.span("tx_send", t0, clock.now_ns(), h.ctx, peer, link.last_sent, ch)
                elif op == "head":
                    _, pair, logits, link, slot, rows, decode, peer = job
                    val, i = float(pair.host[0]), int(pair.host[1])
                    idx = rows[0] + i
                    if i >= rows[1] - rows[0] or idx > IDX_MASK:
                        raise ValueError(f"head index {i} is out of range (no maximum among the logits, or it does not fit the frame)")
                    aux0 = idx | slot << SLOT_SHIFT
                    if logits is not None:
                        wire, dt = _host_bytes(logits.host)
                        link.send(transport.KIND_HEAD, wire, dt, tuple(logits.host.shape), aux0=aux0, aux1=f32_bits(val), decode=decode)
                    else:
                        link.send(transport.KIND_HEAD, aux0=aux0, aux1=f32_bits(val), decode=decode)
                    self.frames += 1
                    if rec:
                        rec.span("tx_send", t0, clock.now_ns(), pair.ctx, peer, link.last_sent, CH_CTL)
                elif op == "free":
                    h = job[1]
                    with self.pool_lock:
                        if h.pooled and self.pool_bytes < POOL_MAX_BYTES:
                            self.pool.setdefault((tuple(h.host.shape), h.host.dtype), []).append(h.host)
                            self.pool_bytes += h.host.numel() * h.host.element_size()
                    h.host = None
                elif op == "flush":
                    job[1].set()
                else:
                    return
        except BaseException as e:      # noqa: BLE001 - the compute thread raises it at its next call (check)
            self.error = e
            if self.on_fail is not None:    # a compute thread blocked on a link makes no such call: closing the links wakes it
                self.on_fail(e)


# ---- all-reduce ----------------------------------------------------------------------------------------------------------
class GroupComm(Comm):
    """All-to-all all-reduce over a group of n ranks. `links` maps every other group rank to its Link."""

    def __init__(self, me: int, n: int, links: dict[int, transport.Link], device: str, reduce_dtype: str = "float32",
                 rec: Recorder | None = None, names: dict[int, str] | None = None, sender: "Sender | None" = None):
        if reduce_dtype not in REDUCE_DTYPES:
            raise ValueError(f"reduce_dtype must be one of {REDUCE_DTYPES}")
        self.me, self.n, self.links, self.device = me, n, links, device
        self.fp32 = reduce_dtype == "float32"
        self.calls = 0
        self.rec, self.names = rec, names or {}
        self.ctx: list | None = None        # recorder contexts of the two requests of a pair, by tag
        self.sender = sender
        self._tail: dict[int, threading.Thread] = {}      # per peer: the latest threaded send that may still be running
        self._out: dict[int, int] = {}      # per peer: bytes sent inline whose all-reduce has not been received yet

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.recv_sum(self.send_partial(x))

    def send_partial(self, x: torch.Tensor, tag: int | None = None):
        """First half of an all-reduce: bring my partial to the host and send it to every peer. Returns the ticket for
        recv_sum. `tag` (0 or 1) names the request of an interleaved pair, for the recorder only."""
        if self.n == 1:
            return x
        self.calls += 1
        rec = self.rec
        if rec and tag is not None and self.ctx:
            rec.use(self.ctx[tag])
        if self.sender is not None:         # --async-send: the copy and the sends are handed over, this thread goes on
            if rec:
                rec.lap("compute")          # no device sync here: what the compute thread spent since its last span
            h = self.sender.copy(x)
            for r in sorted(self.links):
                self.sender.send(h, self.links[r], transport.KIND_ALLREDUCE, 0, 0, x.shape[0] == 1, self.names.get(r), CH_GROUP)
            return (x.dtype, h, None, tag, 0)
        if rec:
            sync(self.device)
            rec.lap("compute")
        xc = x.contiguous().cpu()
        wire, dt = _host_bytes(xc)
        if rec:
            rec.lap("host_copy")
        nb = transport.frame_bytes(len(wire), xc.dim())
        dec = xc.shape[0] == 1
        pending = []
        inline = 0
        for r in sorted(self.links):
            prev = self._tail.get(r)
            if prev is not None and not prev.is_alive():
                prev = None
            # inline only when this frame and the inline frames of a pair partner fit the buffer the kernel granted:
            # every rank sends before it receives, so what is sent inline must fit without the peer reading
            if prev is None and self.links[r].fits_inline(len(wire), self._out.get(r, 0)):
                self.links[r].send(transport.KIND_ALLREDUCE, wire, dt, tuple(xc.shape), decode=dec)
                self._out[r] = self._out.get(r, 0) + len(wire)
                inline = len(wire)
            elif prev is None:
                self._tail[r] = self.links[r].send_async(transport.KIND_ALLREDUCE, bytes(wire), dt, tuple(xc.shape), decode=dec)
                pending.append(self._tail[r])
            else:
                # the first frame of a pair is still leaving on a thread: this one must not overtake it on the link
                self._tail[r] = threading.Thread(target=self._send_after, daemon=True,
                                                 args=(prev, self.links[r], bytes(wire), dt, tuple(xc.shape), dec))
                self._tail[r].start()
                pending.append(self._tail[r])
            if rec:
                rec.lap("send", self.names.get(r), nb, CH_GROUP)
        return (x.dtype, xc, pending, tag, inline)

    @staticmethod
    def _send_after(prev: threading.Thread, link: transport.Link, data: bytes, dt: str, shape: tuple, dec: bool) -> None:
        prev.join()
        link.send(transport.KIND_ALLREDUCE, data, dt, shape, decode=dec)

    def recv_sum(self, ticket) -> torch.Tensor:
        """Second half: receive every peer's partial and add all partials in group rank order."""
        if self.n == 1:
            return ticket
        dtype, xc, pending, tag, inline = ticket
        rec = self.rec
        if rec and tag is not None and self.ctx:
            rec.use(self.ctx[tag])
        parts = {}
        for r in sorted(self.links):
            kind, payload, pdt, shape, _, _ = self.links[r].recv()
            if kind != transport.KIND_ALLREDUCE:
                raise RuntimeError(f"expected an all-reduce frame from group rank {r}, got {transport.KIND_NAMES.get(kind, kind)}")
            if rec:
                rec.lap("allreduce_wait", self.names.get(r), self.links[r].last_recv, CH_GROUP)
            parts[r] = _from_wire(payload, shape, pdt)
            if inline:                      # the peer answered this all-reduce: it has read, or is reading, what was sent for it
                self._out[r] = self._out.get(r, 0) - inline
        if pending is None:                 # handed to the sender thread: my partial is on the host once its copy is done
            parts[self.me] = self.sender.host(xc)
        else:
            parts[self.me] = xc
            for t in pending:
                t.join()
        if self.fp32:
            acc = parts[0].float()
            for r in range(1, self.n):
                acc = acc + parts[r].float()
            out = acc.to(dtype)
        else:
            out = parts[0].clone()
            for r in range(1, self.n):
                out = out + parts[r]
        out = out.to(self.device)
        if pending is None:
            self.sender.free(xc)
        if rec:
            if self.sender is None:
                sync(self.device)
            rec.lap("host_copy")
        return out


# ---- one rank ------------------------------------------------------------------------------------------------------------
class RankRuntime:
    def __init__(self, spec: RankSpec, specs: list[RankSpec], cfg: Qwen2Config, model_dir: str, device: str, max_seq: int,
                 reduce_dtype: str, phase: transport.Phase, log=_flush_print, enforce_budget: bool = True,
                 dtype: str = "bfloat16", timeline: bool = False, slots: int = 1, async_send: bool = False):
        if not 1 <= slots <= MAX_SLOTS:
            raise ValueError(f"slots must be 1 to {MAX_SLOTS}, got {slots}")
        self.slots = slots
        self.log = log
        self.rec = Recorder(spec.name, spec.gid, spec.host, device) if timeline else None
        self.sender = Sender(device, self.rec, on_fail=self._sender_failed) if async_send else None
        self.sampler: Sampler | None = None
        self.source = None                  # a shipping.CacheSource when the weights were shipped
        self.shipping: dict | None = None
        self.spec, self.specs, self.cfg, self.model_dir = spec, specs, cfg, model_dir
        self.dtype = TORCH_DT[dtype]
        self.device, self.max_seq, self.reduce_dtype, self.phase, self.log = device, max_seq, reduce_dtype, phase, log
        self.enforce_budget = enforce_budget
        self.group = [s for s in specs if s.stage == spec.stage]
        nxt = [s for s in specs if s.stage == spec.stage + 1]
        self.down_specs = nxt if spec.group_rank == 0 else []
        self.up_spec = next((s for s in specs if s.stage == spec.stage - 1 and s.group_rank == 0), None)
        self.ctl: transport.Link | None = None
        self.up: transport.Link | None = None
        self.down: list[transport.Link] = []
        self.links: dict[str, transport.Link] = {}
        self.model = None
        self.loaded_bytes = self.kv_bytes = 0
        self.load_s = 0.0
        self.steps = 0
        self.idle = idle_footprint(device)

    def peers(self) -> dict[int, str]:
        return needed_peers(self.spec, self.specs)

    def _sender_failed(self, e: BaseException) -> None:
        """The sender thread died (--async-send). The compute thread may be blocked on a link and would never learn of
        it, so every link of this rank is closed: its own receive raises, and so do its peers'."""
        self.log(f"[hybrid] {self.spec.name}: the sender thread failed: {type(e).__name__}: {e}; closing this rank's links")
        for link in list(self.links.values()):
            link.close()

    def load(self) -> None:
        d = Dims.of(self.cfg)         # slice_spec needs no byte info
        sl = self.spec.slice_spec(d)
        ls = LoadSpec(self.spec.layers, sl, self.spec.embed_rows, self.spec.head_rows, self.spec.final_norm)
        budget = self.spec.budget if self.enforce_budget else None
        t0 = time.time()
        try:
            if self.source is not None:
                w, self.loaded_bytes = load_from_source(self.source, self.cfg, ls, self.device, self.dtype, budget, plain=self.source.plain)
            else:
                w, self.loaded_bytes = load_slices(self.model_dir, self.cfg, ls, self.device, dtype=self.dtype, budget=budget, log=self.log)
            self.model = Qwen2TP(self.cfg, w, sl, self.device, self.max_seq, dtype=self.dtype, layers=self.spec.layers,
                                 embed_rows=self.spec.embed_rows, head_rows=self.spec.head_rows, slots=self.slots)
        except torch.cuda.OutOfMemoryError as e:
            free, total = torch.cuda.mem_get_info()
            what = getattr(e, "tensor_name", None) or "weights or KV cache"
            raise RuntimeError(f"{self.spec.name}: CUDA out of memory loading {what}, free {free / 1e9:.2f} GB of "
                               f"{total / 1e9:.2f} GB") from e
        self.kv_bytes = sum(2 * l.k_cache.numel() * l.k_cache.element_size() for l in self.model.layers)
        if budget is not None and self.loaded_bytes + self.kv_bytes > budget:
            raise BudgetExceeded(f"weights {self.loaded_bytes / 1e9:.3f} GB plus KV {self.kv_bytes / 1e9:.3f} GB "
                                 f"over the budget of {budget / 1e9:.3f} GB")
        sync(self.device)
        self.load_s = time.time() - t0
        self.log(f"[hybrid] {self.spec.name} loaded {self.loaded_bytes / 1e9:.3f} GB (+ KV {self.kv_bytes / 1e9:.3f}) "
                 f"in {self.load_s:.1f} s on {self.device}")

    def attach(self, links: dict[int, transport.Link], ctl: transport.Link | None) -> None:
        by_gid = {s.gid: s for s in self.specs}
        self.ctl = ctl
        if ctl is not None:
            self.links["ctl"] = ctl
        grp = {}
        for gid, role in self.peers().items():
            link = links[gid]
            self.links[by_gid[gid].name] = link
            if role == "grp":
                grp[by_gid[gid].group_rank] = link
            elif role == "up":
                self.up = link
            else:
                self.down.append(link)
        self.down_order = [links[s.gid] for s in self.down_specs]
        self.down_names = [s.name for s in self.down_specs]
        self.up_name = self.specs[0].name if self.spec.stage == 0 else self.up_spec.name
        self.comm = GroupComm(self.spec.group_rank, self.spec.group_size, grp, self.device, self.reduce_dtype, rec=self.rec,
                              names={s.group_rank: s.name for s in self.group}, sender=self.sender)

    @torch.inference_mode()
    def step(self, ids: list[int] | None = None, h: torch.Tensor | None = None, pos0: int = 0, flags: int = 0):
        """One step of one request. `flags` is the frame's aux1: flag bits, request number, slot."""
        T = len(ids) if ids is not None else h.shape[0]
        self.phase.decode = T == 1
        slot = flags >> SLOT_SHIFT
        if ids is not None:
            h = self.model.embed_tokens(ids)
            if self.spec.group_size > 1:
                h = self.comm.all_reduce(h)
        h = self.model.layers_forward(h, pos0, self.comm, slot)
        return self._finish(h, pos0, flags)

    def _finish(self, h: torch.Tensor, pos0: int, flags: int):
        """The end of a step: the hidden state to the next stage, and the head result when this rank holds head rows
        and the frame asks for one."""
        rec = self.rec
        if self.sender is not None:         # --async-send: nothing here waits for the device, except a head result this process needs itself
            if self.down_order:
                hd = self.sender.copy(h)
                for link, name in zip(self.down_order, self.down_names):
                    self.sender.send(hd, link, transport.KIND_HIDDEN, pos0, flags, h.shape[0] == 1, name, CH_PIPE)
                self.sender.free(hd)
            self.steps += 1
            res = None
            if self.spec.head_rows is not None and not flags & FLAG_NOHEAD:
                res = AsyncHead(*self.model.head_device(h)) if self.spec.gid else self.model.head_argmax(h)
            if rec:
                rec.lap("compute")
            return res
        if self.down_order:
            if rec:
                sync(self.device)
                rec.lap("compute")
            hc = h.contiguous().cpu()
            wire, dt = _host_bytes(hc)
            data = bytes(wire)
            if rec:
                rec.lap("host_copy")
            for link, name in zip(self.down_order, self.down_names):
                link.send(transport.KIND_HIDDEN, data, dt, tuple(hc.shape), aux0=pos0, aux1=flags, decode=hc.shape[0] == 1)
                if rec:
                    rec.lap("send", name, link.last_sent, CH_PIPE)
        self.steps += 1
        if self.spec.head_rows is not None and not flags & FLAG_NOHEAD:
            res = self.model.head_argmax(h)
            if rec:
                sync(self.device)
                rec.lap("compute")
            return res
        if rec and not self.down_order:      # neither forwards nor answers with a head result: close the compute span
            sync(self.device)
            rec.lap("compute")
        return None

    @torch.inference_mode()
    def step_pair(self, a: tuple, b: tuple, ctxs: tuple | None = None):
        """Two requests in lockstep through this rank's layers (--tp-interleave 2). a and b are (ids, h, pos0, flags) of
        the two frames, in their order on the link. Returns the two head results. `ctxs`: their recorder contexts."""
        (ids_a, ha, pos_a, fl_a), (ids_b, hb, pos_b, fl_b) = a, b
        rec, comm = self.rec, self.comm
        comm.ctx = list(ctxs) if rec and ctxs else None
        try:
            if ids_a is not None:
                ha = self.model.embed_tokens(ids_a)
            if ids_b is not None:
                hb = self.model.embed_tokens(ids_b)
            if self.spec.group_size > 1 and (ids_a is not None or ids_b is not None):
                ta = comm.send_partial(ha, 0) if ids_a is not None else None
                tb = comm.send_partial(hb, 1) if ids_b is not None else None
                if ta is not None:
                    ha = comm.recv_sum(ta)
                if tb is not None:
                    hb = comm.recv_sum(tb)
            ha, hb = self.model.layers_forward_pair(ha, pos_a, fl_a >> SLOT_SHIFT, hb, pos_b, fl_b >> SLOT_SHIFT, comm)
        finally:
            comm.ctx = None
        if rec and ctxs:
            rec.use(ctxs[0])
        ra = self._finish(ha, pos_a, fl_a)
        if rec and ctxs:
            rec.use(ctxs[1])
        rb = self._finish(hb, pos_b, fl_b)
        return ra, rb

    def flush(self) -> None:
        """With --async-send: wait until every frame handed over has left (before a control frame or a counter read)."""
        if self.sender is not None:
            self.sender.flush()

    def forward_control(self, kind: int, aux1: int = 0) -> None:
        self.flush()
        for link in self.down_order:
            link.send(kind, aux1=aux1)

    def start_sampler(self, listen_addrs: list[str], host_level: bool) -> None:
        cuda = any(s.host == self.spec.host and s.device.split(":")[0] == "cuda" for s in self.specs)
        self.sampler = Sampler(self.spec.name, self.spec.host, self.device, ifaces_for(listen_addrs), host_level, gpu=cuda).start()

    def close(self, run_dir: str | None) -> None:
        """Stop the sampler and write the timeline and the samples (only when the run asked for a timeline)."""
        if self.sampler is not None:
            self.sampler.stop()
            if run_dir:
                self.sampler.write(run_dir)
            self.sampler = None
        if self.sender is not None:
            self.sender.stop()
        if self.rec is not None and run_dir:
            self.rec.write(run_dir)

    def metrics(self) -> dict:
        self.flush()
        m = {"name": self.spec.name, "gid": self.spec.gid, "host": self.spec.host, "device": self.device,
             "pid": os.getpid(), "torch": torch.__version__, "loaded_tensor_bytes": self.loaded_bytes,
             "kv_bytes": self.kv_bytes, "idle": self.idle, "planned": self.spec.planned, "budget": self.spec.budget, "load_s": self.load_s,
             "steps": self.steps, "allreduce_calls": self.comm.calls, "links": {k: l.counters.snapshot() for k, l in self.links.items()},
             "shipping": self.shipping, "slots": self.slots,
             "socket_buffers": {k: [l.sndbuf, l.rcvbuf] for k, l in self.links.items()},
             "async_send": {"copy": self.sender.mode, "frames": self.sender.frames} if self.sender else None}
        m.update(mem_metrics(self.device))
        m.update(cuda_metrics(self.device))
        m.update(footprint())
        return m

    def _unit(self, kind: int, payload, dtype, shape, aux0: int, aux1: int) -> tuple:
        """(ids, hidden state, pos0, flags) of a TOKEN or HIDDEN frame."""
        if kind == transport.KIND_TOKEN:
            return (np.frombuffer(payload, dtype=np.int64).tolist(), None, aux0, aux1)
        h = _from_bytes(payload, shape, dtype, self.device)
        if self.rec:
            self.rec.lap("host_copy")
        return (None, h, aux0, aux1)

    def _send_head(self, res, flags: int, unit: tuple) -> None:
        """A head rank's answer for one step, on its ctl link: the best value and its index, with the slot."""
        if res is None:
            return
        dec = (len(unit[0]) if unit[0] is not None else unit[1].shape[0]) == 1
        if isinstance(res, AsyncHead):
            pair = self.sender.copy(res.pair)
            lg = self.sender.copy(res.logits) if flags & FLAG_LOGITS else None
            self.sender.head(pair, lg, self.ctl, flags >> SLOT_SHIFT, self.spec.head_rows, dec, self.specs[0].name)
            self.sender.free(pair)
            if lg is not None:
                self.sender.free(lg)
            return
        val, idx, lg = res
        if idx > IDX_MASK:
            raise ValueError(f"token index {idx} does not fit the head frame ({IDX_MASK} at most)")
        aux0 = idx | (flags >> SLOT_SHIFT) << SLOT_SHIFT
        if flags & FLAG_LOGITS:
            lc = lg.contiguous().cpu()
            wire, dt = _host_bytes(lc)
            self.ctl.send(transport.KIND_HEAD, wire, dt, tuple(lc.shape), aux0=aux0, aux1=f32_bits(val), decode=dec)
        else:
            self.ctl.send(transport.KIND_HEAD, aux0=aux0, aux1=f32_bits(val), decode=dec)
        if self.rec:
            self.rec.lap("send", self.specs[0].name, self.ctl.last_sent, CH_CTL)

    def serve(self) -> None:
        """Remote rank loop: read the upstream link (ctl for stage 0, the previous leader otherwise). A rank of a later
        stage also watches its ctl link, where only a clock exchange can arrive."""
        up = self.ctl if self.spec.stage == 0 else self.up
        both = None if up is self.ctl else [up.sock, self.ctl.sock]
        up_ch = CH_CTL if up is self.ctl else CH_PIPE
        rec = self.rec
        while True:
            if both is not None and up.sock not in select.select(both, [], [])[0]:       # only the ctl link has something
                kind, _, _, _, _, aux1 = self.ctl.recv()
                if kind != transport.KIND_CLOCK or aux1 != clock.BEGIN:
                    raise RuntimeError(f"{self.spec.name}: unexpected frame {transport.KIND_NAMES.get(kind, kind)} on the ctl link")
                self.flush()
                clock.answer(self.ctl)
                continue
            kind, payload, dtype, shape, aux0, aux1 = up.recv()
            if kind in (transport.KIND_TOKEN, transport.KIND_HIDDEN):
                ctx = None
                if rec:
                    ctx = rec.begin((aux1 >> REQ_SHIFT) & REQ_MASK, shape[0], aux1 >> SLOT_SHIFT)
                    rec.lap("recv_wait", self.up_name, up.last_recv, up_ch)
                one = self._unit(kind, payload, dtype, shape, aux0, aux1)
                if aux1 & FLAG_PAIR and self.spec.group_size > 1:
                    # a pair: the partner is the next frame on this link; a stage of one rank just takes them in turn
                    kind, payload, dtype, shape, aux0b, aux1b = up.recv()
                    if kind not in (transport.KIND_TOKEN, transport.KIND_HIDDEN):
                        raise RuntimeError(f"{self.spec.name}: a pair was announced, got {transport.KIND_NAMES.get(kind, kind)}")
                    ctx_b = None
                    if rec:
                        ctx_b = rec.begin((aux1b >> REQ_SHIFT) & REQ_MASK, shape[0], aux1b >> SLOT_SHIFT)
                        rec.lap("recv_wait", self.up_name, up.last_recv, up_ch)
                    two = self._unit(kind, payload, dtype, shape, aux0b, aux1b)
                    if rec:
                        rec.use(ctx)
                    ra, rb = self.step_pair(one, two, (ctx, ctx_b) if rec else None)
                    if rec:
                        rec.use(ctx)
                    self._send_head(ra, aux1, one)
                    if rec:
                        rec.use(ctx_b)
                    self._send_head(rb, aux1b, two)
                else:
                    self._send_head(self.step(ids=one[0], h=one[1], pos0=one[2], flags=one[3]), aux1, one)
            elif kind == transport.KIND_CLOCK and up is self.ctl:        # a rank of the first stage: its upstream link is its ctl link
                self.flush()
                clock.answer(self.ctl)
            elif kind == transport.KIND_METRICS:
                self.forward_control(kind)
                self.ctl.send(transport.KIND_METRICS_REPLY, json.dumps(self.metrics()).encode(), "uint8")
            elif kind == transport.KIND_RESET:
                self.forward_control(kind)
            elif kind == transport.KIND_SHUTDOWN:
                self.forward_control(kind)
                self.log(f"[hybrid] {self.spec.name}: shutdown")
                return
            else:
                raise RuntimeError(f"{self.spec.name}: unexpected frame {transport.KIND_NAMES.get(kind, kind)}")


# ---- links -----------------------------------------------------------------------------------------------------------------
class Listener:
    """Accepts on every chosen address; each connection must open with HELLO {"role": "ctl"|"data", "gid": n}."""

    def __init__(self, addrs: list[str], port: int, phase: transport.Phase, log=_flush_print):
        self.socks = [transport.listen(a, port, backlog=64) for a in addrs]
        self.q: queue.Queue = queue.Queue()
        self.phase, self.log = phase, log
        for s in self.socks:
            threading.Thread(target=self._loop, args=(s,), daemon=True).start()

    def _loop(self, srv) -> None:
        while True:
            try:
                sock, addr = srv.accept()
            except OSError:
                return
            sock.settimeout(None)
            link = transport.Link(sock, phase=self.phase)
            try:
                kind, payload, *_ = link.recv()
            except (ConnectionError, OSError):
                link.close()
                continue
            if kind != transport.KIND_HELLO:
                link.close()
                continue
            self.q.put((json.loads(bytes(payload).decode()), link))

    def close(self) -> None:
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass


def connect_link(addr: str, port: int, hello: dict, phase: transport.Phase, timeout_s: float = 300.0) -> transport.Link:
    sock = transport.connect(addr, port, timeout_s=timeout_s)
    sock.settimeout(None)
    link = transport.Link(sock, phase=phase)
    link.send(transport.KIND_HELLO, json.dumps(hello).encode(), "uint8")
    return link


def address(target: RankSpec, from_host: str, hosts: dict[str, str]) -> str:
    return "127.0.0.1" if target.host == from_host else hosts[target.host]


def build_links(rt: RankRuntime, hosts: dict[str, str], listener: Listener, pending: list, timeout_s: float = 600.0
                ) -> dict[int, transport.Link]:
    """Connect to every needed peer with a lower gid; accept every needed peer with a higher gid. `pending` holds
    hellos the listener delivered before they were needed."""
    by_gid = {s.gid: s for s in rt.specs}
    me = rt.spec.gid
    need = rt.peers()
    links: dict[int, transport.Link] = {}
    for gid in sorted(g for g in need if g < me):
        t = by_gid[gid]
        links[gid] = connect_link(address(t, rt.spec.host, hosts), t.port, {"role": "data", "gid": me}, rt.phase)
    want = {g for g in need if g > me}
    for hello, link in list(pending):
        if hello.get("role") == "data" and hello.get("gid") in want:
            links[hello["gid"]] = link
            pending.remove((hello, link))
    t0 = time.time()
    while want - set(links):
        left = timeout_s - (time.time() - t0)
        if left <= 0:
            raise TimeoutError(f"{rt.spec.name}: peers {sorted(want - set(links))} did not connect in {timeout_s:.0f} s")
        hello, link = listener.q.get(timeout=left)
        if hello.get("role") == "data" and hello.get("gid") in want:
            links[hello["gid"]] = link
        else:
            pending.append((hello, link))
    return links


RUN_TAG = re.compile(r"[A-Za-z0-9._-]+")


def default_run_dir(tag: str | None) -> str:
    """~/.nocturne/runs/<tag>. The tag may come from the wire (the PLAN message), so anything that is not a plain name
    (letters, digits, dot, underscore, dash) becomes `untagged`."""
    ok = isinstance(tag, str) and RUN_TAG.fullmatch(tag) and tag not in (".", "..")
    return os.path.join(os.path.expanduser("~/.nocturne/runs"), tag if ok else "untagged")


def host_samplers(specs: list[RankSpec]) -> set[int]:
    """The gid that samples host level counters on each host: the lowest gid there."""
    first: dict[str, int] = {}
    for s in specs:
        first.setdefault(s.host, s.gid)
    return set(first.values())


def rank_main(addrs: list[str], port: int, model_dir: str | None, device: str | None, log=_flush_print,
              run_dir: str | None = None, cache_dir: str | None = None) -> int:
    """A rank process (gid > 0): wait for the coordinator's ctl link and plan, link up, load, serve."""
    device = pick_device(device)
    phase = transport.Phase()
    listener = Listener(addrs, port, phase, log)
    pending: list = []
    while True:
        hello, link = listener.q.get()
        if hello.get("role") == "ctl":
            ctl = link
            break
        pending.append((hello, link))
    kind, payload, *_ = ctl.recv()
    if kind != transport.KIND_PLAN:
        raise RuntimeError(f"expected the plan, got {transport.KIND_NAMES.get(kind, kind)}")
    pl = json.loads(bytes(payload).decode())
    specs = [RankSpec.from_json(s) for s in pl["specs"]]
    spec = specs[pl["gid"]]
    ship = bool(pl.get("ship"))
    cfg = config_from_json(pl["cfg"]) if pl.get("cfg") else load_config(model_dir)
    timeline = bool(pl.get("timeline"))
    out_dir = (run_dir or default_run_dir(pl.get("run_tag"))) if timeline else None
    rt = RankRuntime(spec, specs, cfg, model_dir, device, pl["max_seq_len"], pl["reduce_dtype"], phase, log,
                     enforce_budget=pl.get("enforce_budget", True), dtype=pl.get("dtype", "bfloat16"), timeline=timeline,
                     slots=int(pl.get("slots") or 1), async_send=bool(pl.get("async_send")))
    if timeline:
        rt.start_sampler(addrs, spec.gid in host_samplers(specs))
    try:
        links = build_links(rt, pl["hosts"], listener, pending)
        if not ship and not model_dir:       # reported after the links are up, so the coordinator reads it instead of waiting for a link
            raise RuntimeError("started without --model, and the plan ships no weights")
        if ship:
            root = shipping.cache_root(cache_dir)
            manifest, rt.shipping = shipping.receive_slices(ctl, root, log, verify=bool(pl.get("verify_cache")))
            rt.source = shipping.CacheSource(manifest, root)
        if device == "mps" and pl.get("mps_fraction_cap") and spec.budget:
            torch.mps.set_per_process_memory_fraction(spec.budget / torch.mps.recommended_max_memory())
        rt.load()
    except Exception as e:      # noqa: BLE001 - every failure is reported to the coordinator with its text
        log(f"[hybrid] {spec.name}: failed: {type(e).__name__}: {e}")
        ctl.send(transport.KIND_ERROR, json.dumps({"rank": spec.name, "error": f"{type(e).__name__}: {e}"}).encode(), "uint8")
        time.sleep(1.0)
        rt.close(None)
        return 1
    rt.attach(links, ctl)
    ctl.send(transport.KIND_READY, json.dumps(rt.metrics()).encode(), "uint8")
    try:
        clock.answer(ctl)
        rt.serve()
    except (ConnectionError, OSError) as e:
        log(f"[hybrid] {spec.name}: link lost ({e}); exiting")
        return 1
    finally:
        listener.close()
        rt.close(out_dir)
    return 0


# ---- coordinator -------------------------------------------------------------------------------------------------------
class Coordinator:
    """Rank 0 of the first stage plus the generation loop. Same interface as engine.Rank0 for server.py."""

    def __init__(self, specs: list[RankSpec], model_dir: str, hosts: dict[str, str], listen_addrs: list[str],
                 max_seq_len: int, reduce_dtype: str, prefill_max: int, device: str | None = None, log=_flush_print,
                 enforce_budget: bool = True, mps_fraction_cap: bool = False, ready_timeout_s: float = 1800.0,
                 dtype: str = "bfloat16", timeline: bool = False, run_dir: str | None = None, run_tag: str | None = None,
                 ship_weights: bool = False, cache_dir: str | None = None, verify_cache: bool = False, slots: int = 1,
                 async_send: bool = False):
        self.log, self.specs, self.max_seq_len, self.prefill_max = log, specs, max_seq_len, prefill_max
        self.slots, self.async_send = slots, async_send
        self.ship_weights, self.cache_dir, self.verify_cache = ship_weights, cache_dir, verify_cache
        self.shipping: dict | None = None
        self.run_dir = (run_dir or default_run_dir(run_tag)) if timeline else None
        self.clock: dict = {"start": {}, "end": {}}
        self.requests: list[dict] = []
        self.req_seq = 0
        self.reduce_dtype = reduce_dtype
        self.cfg: Qwen2Config = load_config(model_dir)
        self.world = len(specs)
        self.phase = transport.Phase()
        me = specs[0]
        self.device = pick_device(device or me.device)
        self.rank = RankRuntime(me, specs, self.cfg, model_dir, self.device, max_seq_len, reduce_dtype, self.phase, log,
                                enforce_budget=enforce_budget, dtype=dtype, timeline=timeline, slots=slots,
                                async_send=async_send)
        self.listener = Listener(listen_addrs, me.port, self.phase, log)
        self.ctl: dict[int, transport.Link] = {}
        if timeline:
            self.rank.start_sampler(listen_addrs, True)
        try:
            self._bring_up(specs, hosts, model_dir, max_seq_len, reduce_dtype, enforce_budget, mps_fraction_cap, dtype,
                           timeline, run_tag, ready_timeout_s)
        except BaseException:
            self.rank.close(None)
            for link in list(self.ctl.values()):      # ranks blocked on their ctl link see the close and exit
                link.close()
            self.listener.close()
            raise

    def _bring_up(self, specs, hosts, model_dir, max_seq_len, reduce_dtype, enforce_budget, mps_fraction_cap, dtype,
                  timeline, run_tag, ready_timeout_s) -> None:
        log, me = self.log, specs[0]
        ship = self.ship_weights and len(specs) > 1
        source = manifests = memo = None
        if ship:        # every slice of every rank is named and hashed before any rank is told to expect weights
            t0 = time.time()
            root = shipping.cache_root(self.cache_dir)
            model_sha, sums = shipping.model_identity(model_dir, root / "file_hashes.json")
            source = shipping.open_source(model_dir)
            memo, memo_path = shipping.load_memo(root, model_sha)
            known = len(memo)
            manifests = {s.gid: shipping.build_manifest(source, self.cfg, shipping.load_spec_of(s, self.cfg), model_sha, memo) for s in specs[1:]}
            if len(memo) != known:
                shipping.save_memo(memo, memo_path)
            self.shipping = {"model_sha256": model_sha, "model_files": sums, "kind": source.kind, "manifest_s": time.time() - t0,
                             "slices_hashed": len(memo) - known, "ranks": {}}
            log(f"[ship] manifests for {len(manifests)} ranks, {sum(m['total_bytes'] for m in manifests.values()) / 1e9:.3f} GB, "
                f"{len(memo) - known} slices hashed in {time.time() - t0:.1f} s")
        plan_msg = {"specs": [s.to_json() for s in specs], "hosts": hosts, "max_seq_len": max_seq_len,
                    "reduce_dtype": reduce_dtype, "enforce_budget": enforce_budget, "mps_fraction_cap": mps_fraction_cap,
                    "dtype": dtype, "timeline": timeline, "run_tag": run_tag, "ship": ship, "cfg": config_to_json(self.cfg),
                    "verify_cache": self.verify_cache, "slots": self.slots, "async_send": self.async_send}
        for s in specs[1:]:
            link = connect_link(address(s, me.host, hosts), s.port, {"role": "ctl"}, self.phase)
            link.sock.settimeout(CTL_TIMEOUT_S)
            link.send(transport.KIND_PLAN, json.dumps(dict(plan_msg, gid=s.gid)).encode(), "uint8")
            self.ctl[s.gid] = link
        links = build_links(self.rank, hosts, self.listener, [])
        threads = []
        if ship:
            stats, ship_errors, threads = shipping.ship_all(self.ctl, source, manifests, {s.gid: s.name for s in specs})
        self.rank.load()
        self.rank.attach(links, None)
        if ship:
            for t in threads:
                t.join()
            self.shipping["ranks"] = stats
            self.shipping["ship_s"] = max((v["ship_s"] for v in stats.values()), default=0.0)
            if ship_errors:
                raise RankFailed("; ".join(sorted(ship_errors)))
        self.ready = {}
        errors = []
        for gid, link in self.ctl.items():
            link.sock.settimeout(ready_timeout_s)
            kind, payload, *_ = link.recv()
            link.sock.settimeout(CTL_TIMEOUT_S)
            msg = json.loads(bytes(payload).decode())
            if kind == transport.KIND_ERROR:
                errors.append(f"{msg['rank']}: {msg['error']}")
            elif kind == transport.KIND_READY:
                self.ready[gid] = msg
                if self.shipping is not None and msg.get("shipping"):
                    self.shipping["ranks"].setdefault(specs[gid].name, {}).update(msg["shipping"])
            else:
                errors.append(f"{specs[gid].name}: unexpected {transport.KIND_NAMES.get(kind, kind)}")
        if errors:
            raise RankFailed("; ".join(errors))
        for gid, link in self.ctl.items():
            self.rank.links[f"ctl:{specs[gid].name}"] = link
        self.stage0_remote = [s for s in specs if s.stage == 0 and s.gid != 0]
        self.head_remote = [s for s in specs if s.head_rows is not None and s.gid != 0]
        self.decode_steps = self.prefill_steps = self.tokens_generated = 0
        for gid, link in self.ctl.items():        # every rank answers clock requests right after READY
            self.clock["start"][specs[gid].name] = clock.measure(link)
        log(f"[hybrid] all {len(specs)} ranks ready")

    # ---- one step -------------------------------------------------------------------------------------------------------
    def _step(self, ids: list[int], pos0: int, want_logits: bool = False) -> tuple[int, torch.Tensor | None]:
        rec = self.rank.rec
        # the request number rides in the upper bits only when the recorder is on; without it the frames are as before session 10
        flags = (FLAG_LOGITS if want_logits else 0) | (((self.req_seq & REQ_MASK) << REQ_SHIFT) if rec else 0)
        self.phase.decode = len(ids) == 1
        if rec:
            rec.begin(self.req_seq, len(ids))
            rec.lap("other")
        data = np.asarray(ids, dtype=np.int64).tobytes()
        for s in self.stage0_remote:
            self.ctl[s.gid].send(transport.KIND_TOKEN, data, "int64", (len(ids),), aux0=pos0, aux1=flags)
            if rec:
                rec.lap("send", s.name, self.ctl[s.gid].last_sent, CH_CTL)
        res = self.rank.step(ids=ids, pos0=pos0, flags=flags)
        cands, parts = [], []
        if res is not None:
            cands.append((res[0], res[1]))
            if want_logits:
                parts.append((self.specs[0].head_rows[0], res[2].float().cpu()))
        for s in self.head_remote:
            kind, payload, dtype, shape, aux0, aux1 = self.ctl[s.gid].recv()
            if kind == transport.KIND_ERROR:
                raise RankFailed(bytes(payload).decode())
            if kind != transport.KIND_HEAD:
                raise RuntimeError(f"expected a head frame from {s.name}, got {transport.KIND_NAMES.get(kind, kind)}")
            if rec:
                rec.lap("recv_wait", s.name, self.ctl[s.gid].last_recv, CH_CTL)
            cands.append((bits_f32(aux1), aux0))
            if want_logits:
                parts.append((s.head_rows[0], _from_wire(payload, shape, dtype).float()))
        tok = pick_global(cands)
        logits = torch.cat([p for _, p in sorted(parts, key=lambda x: x[0])]) if want_logits else None
        return tok, logits

    def generate(self, ids: list[int], max_tokens: int, on_token=None, ignore_eos: bool = False, rid: str = "req-0",
                 first_logits: bool = False) -> dict:
        if len(ids) > self.prefill_max:
            raise ValueError(f"prompt of {len(ids)} tokens is longer than prefill_max {self.prefill_max} (workspace plan)")
        if len(ids) + max_tokens > self.max_seq_len:
            raise ValueError(f"{len(ids)} + {max_tokens} tokens exceed max_seq_len {self.max_seq_len}")
        self.log(f"[hybrid] request {rid} prompt_n={len(ids)} max_tokens={max_tokens}")
        self.req_seq += 1
        rec = self.rank.rec
        if rec:
            rec.skip()
        m0 = clock.now_ns()
        try:
            t0 = time.time()
            nxt, lg = self._step(ids, 0, want_logits=first_logits)
            t1 = time.time()
            m1 = clock.now_ns()
            self.prefill_steps += 1
            out = [nxt]
            if on_token:
                on_token(nxt)
            finish, pos = "length", len(ids)
            while True:
                if nxt in self.cfg.eos_token_ids and not ignore_eos:
                    finish = "stop"
                    break
                if len(out) >= max_tokens:
                    break
                nxt, _ = self._step([nxt], pos)
                pos += 1
                out.append(nxt)
                self.decode_steps += 1
                if on_token:
                    on_token(nxt)
            t2 = time.time()
            if rec:
                rec.lap("other")
        except (ConnectionError, OSError) as e:
            raise LinkLost(str(e)) from e
        self.requests.append({"req": self.req_seq, "rid": rid, "prompt_n": len(ids), "n_gen": len(out), "t0_mono_ns": m0,
                              "first_token_mono_ns": m1, "t1_mono_ns": clock.now_ns()})
        self.tokens_generated += len(out)
        steps = len(out) - 1
        tm = {"prompt_n": len(ids), "prompt_ms": (t1 - t0) * 1000, "predicted_n": len(out), "predicted_ms": (t2 - t1) * 1000,
              "predicted_per_token_ms": (t2 - t1) * 1000 / steps if steps else None}
        self.log(f"[hybrid] done {rid} n_gen={len(out)} finish={finish}")
        return {"tokens": out, "finish_reason": finish, "timings": tm, "frames": None, "first_logits": lg}

    # ---- several requests in flight (overlap) ---------------------------------------------------------------------------
    def generate_many(self, reqs: list[dict], prefill_chunk: int = 0, tp_interleave: int = 1) -> list[dict]:
        """Run every request with up to `self.slots` in flight. A request is {"ids", "max_tokens", "ignore_eos", "rid",
        "first_logits"}. Returns one result per request, in the order given, each as generate() returns it plus its slot.
        A slot is ready when its next input exists: a prefill chunk, or the token just chosen. While a slot is ready the
        first stage runs for it; head results of other slots are read between steps without blocking. With
        tp_interleave 2 two ready slots go through as a pair. The tokens of a request do not depend on any of this:
        every step reads and writes only its own slot."""
        try:
            check_overlap(self.slots, prefill_chunk, tp_interleave)
        except PlanError as e:
            raise ValueError(str(e)) from e
        self.schedule = {"pair_steps": 0, "single_steps": 0, "max_in_flight": 0}      # steps that ran as a pair or alone; most units in flight
        for r in reqs:
            if len(r["ids"]) > self.prefill_max:
                raise ValueError(f"prompt of {len(r['ids'])} tokens is longer than prefill_max {self.prefill_max} (workspace plan)")
            if len(r["ids"]) + r["max_tokens"] > self.max_seq_len:
                raise ValueError(f"{len(r['ids'])} + {r['max_tokens']} tokens exceed max_seq_len {self.max_seq_len}")
        rec = self.rank.rec
        if rec:
            rec.skip()
        waiting = collections.deque(enumerate(reqs))
        table: list = [None] * self.slots
        ready: collections.deque = collections.deque()
        results: list = [None] * len(reqs)
        n_heads = len(self.head_remote) + (1 if self.specs[0].head_rows is not None else 0)

        def start(s: int) -> None:
            if not waiting:
                table[s] = None
                return
            i, r = waiting.popleft()
            self.req_seq += 1
            ids = list(r["ids"])
            chunks = chunk_ranges(len(ids), prefill_chunk)
            st = {"i": i, "r": r, "req": self.req_seq, "out": [], "pos": len(ids), "units": collections.deque(), "chunks": len(chunks),
                  "t0": time.time(), "m0": clock.now_ns(), "t1": None, "m1": None, "logits": None, "cands": None, "parts": None,
                  "want_logits": False, "decode": False, "ctx": None}
            for k, (a, b) in enumerate(chunks):
                last = k == len(chunks) - 1
                flags = (0 if last else FLAG_NOHEAD) | (FLAG_LOGITS if last and r.get("first_logits") else 0)
                st["units"].append((ids[a:b], a, flags))
            table[s] = st
            ready.append(s)
            self.log(f"[hybrid] request {r.get('rid', i)} prompt_n={len(ids)} max_tokens={r['max_tokens']} slot={s} chunks={len(chunks)}")

        def launch(s: int, pair_first: bool = False) -> tuple:
            """Take the next unit of a slot and send its token ids to the other ranks of the first stage."""
            st = table[s]
            ids, pos0, flags = st["units"].popleft()
            aux = pack_aux(flags | (FLAG_PAIR if pair_first else 0), st["req"] if rec else 0, s)
            st["decode"] = len(ids) == 1
            ctx = None
            if rec:
                ctx = st["ctx"] = rec.begin(st["req"], len(ids), s)
                rec.lap("other")
            data = np.asarray(ids, dtype=np.int64).tobytes()
            for sp in self.stage0_remote:
                self.ctl[sp.gid].send(transport.KIND_TOKEN, data, "int64", (len(ids),), aux0=pos0, aux1=aux, decode=len(ids) == 1)
                if rec:
                    rec.lap("send", sp.name, self.ctl[sp.gid].last_sent, CH_CTL)
            if st["decode"]:
                self.decode_steps += 1
            else:
                self.prefill_steps += 1
            if not flags & FLAG_NOHEAD:      # a unit that ends in a head result is in flight until that result is complete
                st["flying"] = True
                self.schedule["max_in_flight"] = max(self.schedule["max_in_flight"], sum(1 for x in table if x and x.get("flying")))
            return (ids, None, pos0, aux), ctx

        def after(s: int, unit: tuple, res) -> None:
            """A unit has left the first stage. With a head result due, wait for it; a prefill chunk that is not the
            last asks for none, so the slot is ready again at once."""
            st = table[s]
            flags = unit[3]
            if flags & FLAG_NOHEAD:
                ready.append(s)
                return
            st["cands"], st["parts"], st["want_logits"] = [], [], bool(flags & FLAG_LOGITS)
            if res is not None:
                st["cands"].append((res[0], res[1]))
                if st["want_logits"]:
                    st["parts"].append((self.specs[0].head_rows[0], res[2].float().cpu()))
            if len(st["cands"]) == n_heads:
                complete(s)

        def complete(s: int) -> None:
            st = table[s]
            r = st["r"]
            st["flying"] = False
            tok = pick_global(st["cands"])
            if st["want_logits"]:
                st["logits"] = torch.cat([p for _, p in sorted(st["parts"], key=lambda x: x[0])])
            st["cands"] = None
            if not st["out"]:
                st["t1"], st["m1"] = time.time(), clock.now_ns()
            st["out"].append(tok)
            if r.get("on_token"):
                r["on_token"](tok)
            stop = tok in self.cfg.eos_token_ids and not r.get("ignore_eos")
            if stop or len(st["out"]) >= r["max_tokens"]:
                t2 = time.time()
                steps = len(st["out"]) - 1
                self.tokens_generated += len(st["out"])
                self.requests.append({"req": st["req"], "rid": r.get("rid", st["i"]), "prompt_n": len(r["ids"]), "n_gen": len(st["out"]),
                                      "t0_mono_ns": st["m0"], "first_token_mono_ns": st["m1"], "t1_mono_ns": clock.now_ns(), "slot": s})
                tm = {"prompt_n": len(r["ids"]), "prompt_ms": (st["t1"] - st["t0"]) * 1000, "predicted_n": len(st["out"]),
                      "predicted_ms": (t2 - st["t1"]) * 1000, "predicted_per_token_ms": (t2 - st["t1"]) * 1000 / steps if steps else None}
                results[st["i"]] = {"tokens": st["out"], "finish_reason": "stop" if stop else "length", "timings": tm, "frames": None,
                                    "first_logits": st["logits"], "slot": s, "chunks": st["chunks"], "t0": st["t0"], "t1": t2}
                self.log(f"[hybrid] done {r.get('rid', st['i'])} n_gen={len(st['out'])} finish={'stop' if stop else 'length'} slot={s}")
                start(s)
            else:
                st["units"].append(([tok], st["pos"], 0))
                st["pos"] += 1
                ready.append(s)

        def drain(block: bool) -> None:
            """Read the head results that have arrived, from every head rank. block: wait until at least one came."""
            links = {self.ctl[sp.gid].sock: sp for sp in self.head_remote}
            if not links:
                if block:
                    raise RuntimeError("no slot is ready and no head rank is remote: the scheduler has nothing to wait for")
                return
            wait = CTL_TIMEOUT_S if block else 0
            while True:
                got = select.select(list(links), [], [], wait)[0]
                if not got:
                    if block and wait:
                        raise TimeoutError(f"no head result in {CTL_TIMEOUT_S:.0f} s")
                    return
                for sock in got:
                    sp = links[sock]
                    link = self.ctl[sp.gid]
                    kind, payload, dtype, shape, aux0, aux1 = link.recv(
                        decode=lambda k, a0, a1: bool(a0 >> SLOT_SHIFT < len(table) and table[a0 >> SLOT_SHIFT] and table[a0 >> SLOT_SHIFT]["decode"]))
                    if kind == transport.KIND_ERROR:
                        raise RankFailed(bytes(payload).decode())
                    if kind != transport.KIND_HEAD:
                        raise RuntimeError(f"expected a head frame from {sp.name}, got {transport.KIND_NAMES.get(kind, kind)}")
                    s = aux0 >> SLOT_SHIFT
                    st = table[s] if s < len(table) else None
                    if st is None or st["cands"] is None:
                        raise RuntimeError(f"head frame from {sp.name} for slot {s}, which waits for none")
                    if rec:
                        rec.use(st["ctx"])
                        rec.lap("recv_wait", sp.name, link.last_recv, CH_CTL)
                    st["cands"].append((bits_f32(aux1), aux0 & IDX_MASK))
                    if st["want_logits"]:
                        st["parts"].append((sp.head_rows[0], _from_wire(payload, shape, dtype).float()))
                    if len(st["cands"]) == n_heads:
                        complete(s)
                wait = 0                         # after the first frames, take only what is already there

        try:
            for s in range(self.slots):
                start(s)
            while any(st is not None for st in table):
                drain(block=False)
                if len(ready) >= 2 and tp_interleave == 2:
                    sa, sb = ready.popleft(), ready.popleft()
                    self.schedule["pair_steps"] += 2
                    (ua, ca), (ub, cb) = launch(sa, pair_first=True), launch(sb)
                    if self.rank.spec.group_size > 1:
                        if rec:
                            rec.use(ca)
                        ra, rb = self.rank.step_pair(ua, ub, (ca, cb) if rec else None)
                    else:                        # the first stage is one rank: the pair flag travels on to a later group
                        if rec:
                            rec.use(ca)
                        ra = self.rank.step(ids=ua[0], pos0=ua[2], flags=ua[3])
                        if rec:
                            rec.use(cb)
                        rb = self.rank.step(ids=ub[0], pos0=ub[2], flags=ub[3])
                    after(sa, ua, ra)
                    after(sb, ub, rb)
                elif ready:
                    s = ready.popleft()
                    self.schedule["single_steps"] += 1
                    u, _ = launch(s)
                    after(s, u, self.rank.step(ids=u[0], pos0=u[2], flags=u[3]))
                elif any(st is not None for st in table):      # the read above may have finished the last request
                    drain(block=True)
            if rec:
                rec.lap("other")
        except (ConnectionError, OSError) as e:
            raise LinkLost(str(e)) from e
        return results

    # ---- control ----------------------------------------------------------------------------------------------------------
    def metrics(self) -> dict:
        self.rank.flush()
        for s in self.stage0_remote:
            self.ctl[s.gid].send(transport.KIND_METRICS)
        self.rank.forward_control(transport.KIND_METRICS)
        ranks = {self.specs[0].name: self.rank.metrics()}
        for gid, link in self.ctl.items():
            kind, payload, *_ = link.recv()
            if kind != transport.KIND_METRICS_REPLY:
                raise RuntimeError(f"expected metrics from {self.specs[gid].name}, got {transport.KIND_NAMES.get(kind, kind)}")
            ranks[self.specs[gid].name] = json.loads(bytes(payload).decode())
        return {"world": self.world, "ranks": ranks, "decode_steps": self.decode_steps, "prefill_steps": self.prefill_steps,
                "tokens_generated": self.tokens_generated, "torch": torch.__version__, "reduce_dtype": self.reduce_dtype}

    def clock_sync(self) -> dict:
        """Measure every rank's clock offset again (the end measurement). Every frame of it, the begin frame included,
        travels on that rank's ctl link and on no data link."""
        for gid, link in self.ctl.items():
            link.send(transport.KIND_CLOCK, aux1=clock.BEGIN)
            self.clock["end"][self.specs[gid].name] = clock.measure(link)
        return self.clock

    def clock_record(self) -> dict:
        """Per rank: start and end measurements and the drift between them. The coordinator's own rank is zero."""
        out = {self.specs[0].name: {"start": dict(clock.ZERO), "end": dict(clock.ZERO), "drift": None, "host": self.specs[0].host}}
        for s in self.specs[1:]:
            st, en = self.clock["start"].get(s.name), self.clock["end"].get(s.name)
            out[s.name] = {"start": st, "end": en, "drift": clock.drift(st, en) if st else None, "host": s.host}
        return out

    def shutdown(self) -> None:
        try:
            if not self.clock["end"] and self.ctl and getattr(self, "stage0_remote", None) is not None:
                for link in self.ctl.values():       # a dead rank must not hold the shutdown for the ctl timeout
                    link.sock.settimeout(END_CLOCK_TIMEOUT_S)
                self.clock_sync()
        except (OSError, RuntimeError) as e:
            self.clock["end"] = {}
            self.log(f"[hybrid] end clock measurement failed: {e}")
        try:
            for s in self.stage0_remote:
                self.ctl[s.gid].send(transport.KIND_SHUTDOWN)
            self.rank.forward_control(transport.KIND_SHUTDOWN)
        except OSError:
            pass
        time.sleep(0.5)
        for link in list(self.ctl.values()):
            link.close()
        self.listener.close()
        self.rank.close(self.run_dir)


DATA_KIND_NAMES = ("step", "allreduce", "token", "hidden", "head")


def data_frames(metrics: dict) -> int:
    """All data frames sent by every process (prefill and decode), from a metrics snapshot."""
    return sum(c["by_kind_sent"].get(k, 0) for r in metrics["ranks"].values() for c in r["links"].values() for k in DATA_KIND_NAMES)


def decode_frames(metrics: dict) -> dict:
    """Sum of decode data frames and bytes sent by every process, from a metrics snapshot."""
    f = b = 0
    per_link = {}
    for rname, r in metrics["ranks"].items():
        for lname, c in r["links"].items():
            d = c.get("decode", {})
            f += d.get("sent_frames", 0)
            b += d.get("sent_bytes", 0)
            per_link[f"{rname}->{lname}"] = {"sent_frames": d.get("sent_frames", 0), "sent_bytes": d.get("sent_bytes", 0),
                                             "recv_frames": d.get("recv_frames", 0), "recv_bytes": d.get("recv_bytes", 0)}
    return {"sent_frames": f, "sent_bytes": b, "per_link": per_link}
