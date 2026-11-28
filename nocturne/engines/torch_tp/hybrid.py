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
(clock.py). With `timeline` in the plan every rank records spans (recorder.py) and writes them at shutdown.
Security: listeners accept any peer; deferred by ADR-009. Never bind 0.0.0.0 (CLAUDE.md rule 7)."""
import ctypes
import json
import os
import queue
import resource
import struct
import threading
import time

import numpy as np
import torch

from . import clock, transport
from .config import Qwen2Config, load_config
from .engine import LinkLost, REDUCE_DTYPES, _from_bytes, mem_metrics, pick_device
from .loader import BudgetExceeded, LoadSpec, load_slices
from .model import Comm, Qwen2TP, pick_global
from .plan import Dims, RankSpec, needed_peers
from .recorder import Recorder, Sampler, ifaces_for

FLAG_LOGITS = 1
REQ_SHIFT = 8                # aux1 of a TOKEN or HIDDEN frame: flags in the low byte, the request number above it
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
        torch.cuda.synchronize()


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


# ---- all-reduce ----------------------------------------------------------------------------------------------------------
class GroupComm(Comm):
    """All-to-all all-reduce over a group of n ranks. `links` maps every other group rank to its Link."""

    def __init__(self, me: int, n: int, links: dict[int, transport.Link], device: str, reduce_dtype: str = "float32",
                 rec: Recorder | None = None, names: dict[int, str] | None = None):
        if reduce_dtype not in REDUCE_DTYPES:
            raise ValueError(f"reduce_dtype must be one of {REDUCE_DTYPES}")
        self.me, self.n, self.links, self.device = me, n, links, device
        self.fp32 = reduce_dtype == "float32"
        self.calls = 0
        self.rec, self.names = rec, names or {}

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.n == 1:
            return x
        self.calls += 1
        rec = self.rec
        if rec:
            sync(self.device)
            rec.lap("compute")
        xc = x.contiguous().cpu()
        wire, dt = _host_bytes(xc)
        if rec:
            rec.lap("host_copy")
        nb = transport.frame_bytes(len(wire), xc.dim())
        pending = []
        for r in sorted(self.links):
            if len(wire) <= transport.SMALL_FRAME:
                self.links[r].send(transport.KIND_ALLREDUCE, wire, dt, tuple(xc.shape))
            else:
                pending.append(self.links[r].send_async(transport.KIND_ALLREDUCE, bytes(wire), dt, tuple(xc.shape)))
            if rec:
                rec.lap("send", self.names.get(r), nb)
        parts = {self.me: xc}
        for r in sorted(self.links):
            kind, payload, pdt, shape, _, _ = self.links[r].recv()
            if kind != transport.KIND_ALLREDUCE:
                raise RuntimeError(f"expected an all-reduce frame from group rank {r}, got {transport.KIND_NAMES.get(kind, kind)}")
            if rec:
                rec.lap("allreduce_wait", self.names.get(r), self.links[r].last_recv)
            parts[r] = _from_wire(payload, shape, pdt)
        for t in pending:
            t.join()
        if self.fp32:
            acc = parts[0].float()
            for r in range(1, self.n):
                acc = acc + parts[r].float()
            out = acc.to(x.dtype)
        else:
            out = parts[0].clone()
            for r in range(1, self.n):
                out = out + parts[r]
        out = out.to(self.device)
        if rec:
            sync(self.device)
            rec.lap("host_copy")
        return out


# ---- one rank ------------------------------------------------------------------------------------------------------------
class RankRuntime:
    def __init__(self, spec: RankSpec, specs: list[RankSpec], cfg: Qwen2Config, model_dir: str, device: str, max_seq: int,
                 reduce_dtype: str, phase: transport.Phase, log=_flush_print, enforce_budget: bool = True,
                 dtype: str = "bfloat16", timeline: bool = False):
        self.rec = Recorder(spec.name, spec.gid, spec.host, device) if timeline else None
        self.sampler: Sampler | None = None
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

    def load(self) -> None:
        d = Dims.of(self.cfg)         # slice_spec needs no byte info
        sl = self.spec.slice_spec(d)
        ls = LoadSpec(self.spec.layers, sl, self.spec.embed_rows, self.spec.head_rows, self.spec.final_norm)
        budget = self.spec.budget if self.enforce_budget else None
        t0 = time.time()
        try:
            w, self.loaded_bytes = load_slices(self.model_dir, self.cfg, ls, self.device, dtype=self.dtype, budget=budget, log=self.log)
            self.model = Qwen2TP(self.cfg, w, sl, self.device, self.max_seq, dtype=self.dtype, layers=self.spec.layers,
                                 embed_rows=self.spec.embed_rows, head_rows=self.spec.head_rows)
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
                              names={s.group_rank: s.name for s in self.group})

    @torch.inference_mode()
    def step(self, ids: list[int] | None = None, h: torch.Tensor | None = None, pos0: int = 0, flags: int = 0):
        T = len(ids) if ids is not None else h.shape[0]
        self.phase.decode = T == 1
        if ids is not None:
            h = self.model.embed_tokens(ids)
            if self.spec.group_size > 1:
                h = self.comm.all_reduce(h)
        h = self.model.layers_forward(h, pos0, self.comm)
        rec = self.rec
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
                link.send(transport.KIND_HIDDEN, data, dt, tuple(hc.shape), aux0=pos0, aux1=flags)
                if rec:
                    rec.lap("send", name, link.last_sent)
        self.steps += 1
        if self.spec.head_rows is not None:
            res = self.model.head_argmax(h)
            if rec:
                sync(self.device)
                rec.lap("compute")
            return res
        return None

    def forward_control(self, kind: int, aux1: int = 0) -> None:
        for link in self.down_order:
            link.send(kind, aux1=aux1)

    def start_sampler(self, listen_addrs: list[str], host_level: bool) -> None:
        self.sampler = Sampler(self.spec.name, self.spec.host, self.device, ifaces_for(listen_addrs), host_level).start()

    def close(self, run_dir: str | None) -> None:
        """Stop the sampler and write the timeline and the samples (only when the run asked for a timeline)."""
        if self.sampler is not None:
            self.sampler.stop()
            if run_dir:
                self.sampler.write(run_dir)
            self.sampler = None
        if self.rec is not None and run_dir:
            self.rec.write(run_dir)

    def metrics(self) -> dict:
        m = {"name": self.spec.name, "gid": self.spec.gid, "host": self.spec.host, "device": self.device,
             "pid": os.getpid(), "torch": torch.__version__, "loaded_tensor_bytes": self.loaded_bytes,
             "kv_bytes": self.kv_bytes, "idle": self.idle, "planned": self.spec.planned, "budget": self.spec.budget, "load_s": self.load_s,
             "steps": self.steps, "allreduce_calls": self.comm.calls, "links": {k: l.counters.snapshot() for k, l in self.links.items()}}
        m.update(mem_metrics(self.device))
        m.update(cuda_metrics(self.device))
        m.update(footprint())
        return m

    def serve(self) -> None:
        """Remote rank loop: read the upstream link (ctl for stage 0, the previous leader otherwise)."""
        up = self.ctl if self.spec.stage == 0 else self.up
        rec = self.rec
        while True:
            kind, payload, dtype, shape, aux0, aux1 = up.recv()
            if kind in (transport.KIND_TOKEN, transport.KIND_HIDDEN):
                if rec:
                    rec.begin(aux1 >> REQ_SHIFT, shape[0])
                    rec.lap("recv_wait", self.up_name, up.last_recv)
                if kind == transport.KIND_TOKEN:
                    res = self.step(ids=np.frombuffer(payload, dtype=np.int64).tolist(), pos0=aux0, flags=aux1)
                else:
                    h = _from_bytes(payload, shape, dtype, self.device)
                    if rec:
                        rec.lap("host_copy")
                    res = self.step(h=h, pos0=aux0, flags=aux1)
                if res is not None:
                    val, idx, lg = res
                    if aux1 & FLAG_LOGITS:
                        lc = lg.contiguous().cpu()
                        wire, dt = _host_bytes(lc)
                        self.ctl.send(transport.KIND_HEAD, wire, dt, tuple(lc.shape), aux0=idx, aux1=f32_bits(val))
                    else:
                        self.ctl.send(transport.KIND_HEAD, aux0=idx, aux1=f32_bits(val))
                    if rec:
                        rec.lap("send", self.specs[0].name, self.ctl.last_sent)
            elif kind == transport.KIND_CLOCK:
                self.forward_control(kind, aux1=clock.BEGIN)
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


def default_run_dir(tag: str | None) -> str:
    return os.path.join(os.path.expanduser("~/.nocturne/runs"), tag or "untagged")


def host_samplers(specs: list[RankSpec]) -> set[int]:
    """The gid that samples host level counters on each host: the lowest gid there."""
    first: dict[str, int] = {}
    for s in specs:
        first.setdefault(s.host, s.gid)
    return set(first.values())


def rank_main(addrs: list[str], port: int, model_dir: str, device: str | None, log=_flush_print,
              run_dir: str | None = None) -> int:
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
    cfg = load_config(model_dir)
    timeline = bool(pl.get("timeline"))
    out_dir = (run_dir or default_run_dir(pl.get("run_tag"))) if timeline else None
    rt = RankRuntime(spec, specs, cfg, model_dir, device, pl["max_seq_len"], pl["reduce_dtype"], phase, log,
                     enforce_budget=pl.get("enforce_budget", True), dtype=pl.get("dtype", "bfloat16"), timeline=timeline)
    if timeline:
        rt.start_sampler(addrs, spec.gid in host_samplers(specs))
    try:
        links = build_links(rt, pl["hosts"], listener, pending)
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
                 dtype: str = "bfloat16", timeline: bool = False, run_dir: str | None = None, run_tag: str | None = None):
        self.log, self.specs, self.max_seq_len, self.prefill_max = log, specs, max_seq_len, prefill_max
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
                                enforce_budget=enforce_budget, dtype=dtype, timeline=timeline)
        self.listener = Listener(listen_addrs, me.port, self.phase, log)
        self.ctl: dict[int, transport.Link] = {}
        if timeline:
            self.rank.start_sampler(listen_addrs, True)
        try:
            self._bring_up(specs, hosts, model_dir, max_seq_len, reduce_dtype, enforce_budget, mps_fraction_cap, dtype,
                           timeline, run_tag, ready_timeout_s)
        except BaseException:
            self.rank.close(None)
            raise

    def _bring_up(self, specs, hosts, model_dir, max_seq_len, reduce_dtype, enforce_budget, mps_fraction_cap, dtype,
                  timeline, run_tag, ready_timeout_s) -> None:
        log, me = self.log, specs[0]
        plan_msg = {"specs": [s.to_json() for s in specs], "hosts": hosts, "max_seq_len": max_seq_len,
                    "reduce_dtype": reduce_dtype, "enforce_budget": enforce_budget, "mps_fraction_cap": mps_fraction_cap,
                    "dtype": dtype, "timeline": timeline, "run_tag": run_tag}
        for s in specs[1:]:
            link = connect_link(address(s, me.host, hosts), s.port, {"role": "ctl"}, self.phase)
            link.sock.settimeout(CTL_TIMEOUT_S)
            link.send(transport.KIND_PLAN, json.dumps(dict(plan_msg, gid=s.gid)).encode(), "uint8")
            self.ctl[s.gid] = link
        links = build_links(self.rank, hosts, self.listener, [])
        self.rank.load()
        self.rank.attach(links, None)
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
        flags = (FLAG_LOGITS if want_logits else 0) | (self.req_seq << REQ_SHIFT)
        self.phase.decode = len(ids) == 1
        rec = self.rank.rec
        if rec:
            rec.begin(self.req_seq, len(ids))
            rec.lap("other")
        data = np.asarray(ids, dtype=np.int64).tobytes()
        for s in self.stage0_remote:
            self.ctl[s.gid].send(transport.KIND_TOKEN, data, "int64", (len(ids),), aux0=pos0, aux1=flags)
            if rec:
                rec.lap("send", s.name, self.ctl[s.gid].last_sent)
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
                rec.lap("recv_wait", s.name, self.ctl[s.gid].last_recv)
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
        m0 = time.monotonic_ns()
        try:
            t0 = time.time()
            nxt, lg = self._step(ids, 0, want_logits=first_logits)
            t1 = time.time()
            m1 = time.monotonic_ns()
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
                              "first_token_mono_ns": m1, "t1_mono_ns": time.monotonic_ns()})
        self.tokens_generated += len(out)
        steps = len(out) - 1
        tm = {"prompt_n": len(ids), "prompt_ms": (t1 - t0) * 1000, "predicted_n": len(out), "predicted_ms": (t2 - t1) * 1000,
              "predicted_per_token_ms": (t2 - t1) * 1000 / steps if steps else None}
        self.log(f"[hybrid] done {rid} n_gen={len(out)} finish={finish}")
        return {"tokens": out, "finish_reason": finish, "timings": tm, "frames": None, "first_logits": lg}

    # ---- control ----------------------------------------------------------------------------------------------------------
    def metrics(self) -> dict:
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
        """Measure every rank's clock offset again (the end measurement). The begin frame travels the data path, as
        metrics do; then every rank answers on its ctl link."""
        for s in self.stage0_remote:
            self.ctl[s.gid].send(transport.KIND_CLOCK, aux1=clock.BEGIN)
        self.rank.forward_control(transport.KIND_CLOCK, aux1=clock.BEGIN)
        for gid, link in self.ctl.items():
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
