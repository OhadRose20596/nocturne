"""torch_tp engine: sharding math, transport framing, and the sharded two-rank forward against the single-rank forward
on a tiny random Qwen2 written to disk (loader, model, engine and transport all exercised over loopback).
Runs only where torch is installed (the ~/dev/nocturne-torch venv); the bench venv skips it."""
import json
import socket
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from nocturne.engines.torch_tp import transport  # noqa: E402
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.engine import Rank0, serve_rank  # noqa: E402
from nocturne.engines.torch_tp.shard import Shard, shard_tensor  # noqa: E402

# Tolerance fixed before the real runs (session 3). The only difference between the sharded and the single-rank path
# is where rounding happens in o_proj and down_proj: one fp32 accumulation inside the matmul against two bf16 partials
# rounded and then added, so two extra bf16 roundings per all-reduce that propagate down the residual stream.
# First guess was 2e-2 absolute at hidden scale 1. Measured on the tiny 2-layer random model (2026-10-29): max abs
# error 0.039 at max |h| = 3.25, which is 5 ulps of the element where it occurs (|h| = 1.98, ulp 2^-7), mean error
# 0.005, argmax unchanged. The bound is therefore stated in bf16 ulps of the max hidden magnitude: 8 ulps for a
# 2-layer block (4 all-reduces), plus mean abs error below 1e-2 and the same argmax. fp32 must agree to 1e-4.
TOL_FP32 = 1e-4
TOL_BF16_ULPS = 8
TOL_BF16_MEAN = 1e-2


def bf16_ulp(x: float) -> float:
    import math
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


def check_bf16(out, ref, what: str):
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    mx, mean = diff.max().item(), diff.mean().item()
    assert mx <= TOL_BF16_ULPS * bf16_ulp(scale), f"{what}: max abs err {mx} > {TOL_BF16_ULPS} ulps at |h|max {scale} ({bf16_ulp(scale)})"
    assert mean <= TOL_BF16_MEAN, f"{what}: mean abs err {mean} > {TOL_BF16_MEAN}"
    return mx, mean

TINY = {"model_type": "qwen2", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False,
        "max_position_embeddings": 512, "eos_token_id": 2, "torch_dtype": "bfloat16"}


def write_tiny_model(d: Path, seed=0) -> Path:
    g = torch.Generator().manual_seed(seed)
    c = TINY
    hd = c["hidden_size"] // c["num_attention_heads"]
    w = {"model.embed_tokens.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5,
         "model.norm.weight": torch.ones(c["hidden_size"]), "lm_head.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        r = lambda *s: torch.randn(*s, generator=g) * 0.1
        w.update({p + "input_layernorm.weight": torch.ones(c["hidden_size"]), p + "post_attention_layernorm.weight": torch.ones(c["hidden_size"]),
                  p + "self_attn.q_proj.weight": r(c["num_attention_heads"] * hd, c["hidden_size"]), p + "self_attn.q_proj.bias": r(c["num_attention_heads"] * hd),
                  p + "self_attn.k_proj.weight": r(c["num_key_value_heads"] * hd, c["hidden_size"]), p + "self_attn.k_proj.bias": r(c["num_key_value_heads"] * hd),
                  p + "self_attn.v_proj.weight": r(c["num_key_value_heads"] * hd, c["hidden_size"]), p + "self_attn.v_proj.bias": r(c["num_key_value_heads"] * hd),
                  p + "self_attn.o_proj.weight": r(c["hidden_size"], c["num_attention_heads"] * hd),
                  p + "mlp.gate_proj.weight": r(c["intermediate_size"], c["hidden_size"]), p + "mlp.up_proj.weight": r(c["intermediate_size"], c["hidden_size"]),
                  p + "mlp.down_proj.weight": r(c["hidden_size"], c["intermediate_size"])})
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(c))
    safetensors_torch.save_file({k: v.to(torch.bfloat16) for k, v in w.items()}, str(d / "model.safetensors"))
    return d


def test_shard_slices_gqa_aware():
    s0 = Shard(0, 2, n_heads=28, n_kv_heads=4, head_dim=128, intermediate=18944)     # Qwen2.5-7B
    s1 = Shard(1, 2, 28, 4, 128, 18944)
    assert s0.q_rows() == slice(0, 14 * 128) and s1.q_rows() == slice(14 * 128, 28 * 128)
    assert s0.kv_rows() == slice(0, 2 * 128) and s1.kv_rows() == slice(2 * 128, 4 * 128)
    assert s1.mlp_rows() == slice(9472, 18944)
    assert s0.slice_for("self_attn.o_proj.weight") == (1, slice(0, 1792)) and s0.slice_for("mlp.down_proj.weight")[0] == 1
    assert s0.slice_for("input_layernorm.weight") is None
    Shard(0, 2, 12, 2, 128, 8960)                      # 1.5B divides
    with pytest.raises(ValueError):
        Shard(0, 3, 28, 4, 128, 18944)                 # 4 kv heads do not divide by 3
    w = torch.arange(24.0).view(6, 4)
    assert torch.equal(shard_tensor(w, (0, slice(3, 6))), w[3:6]) and torch.equal(shard_tensor(w, (1, slice(0, 2))), w[:, :2])


def test_transport_frames_roundtrip_and_counters():
    srv = transport.listen("127.0.0.1", 0)
    port = srv.getsockname()[1]
    got = {}

    def side_b():
        s, _ = srv.accept()
        link = transport.Link(s)
        kind, payload, dtype, shape, a0, a1 = link.recv()
        got.update(kind=kind, payload=payload, dtype=dtype, shape=shape, a0=a0, a1=a1)
        link.send(transport.KIND_ALLREDUCE, payload, dtype, shape)
        link.close()
    t = threading.Thread(target=side_b); t.start()
    a = transport.Link(transport.connect("127.0.0.1", port))
    x = torch.randn(3, 5).to(torch.bfloat16)
    from nocturne.engines.torch_tp.engine import _from_bytes, _to_bytes
    a.send(transport.KIND_STEP, _to_bytes(x), "bfloat16", (3, 5), aux0=7)
    kind, payload, dtype, shape, *_ = a.recv()
    t.join()
    assert got["kind"] == transport.KIND_STEP and got["shape"] == (3, 5) and got["a0"] == 7 and got["dtype"] == "bfloat16"
    assert kind == transport.KIND_ALLREDUCE and torch.equal(_from_bytes(payload, shape, dtype, "cpu"), x)
    c = a.counters.snapshot()
    assert c["sent_frames"] == 1 and c["recv_frames"] == 1 and c["step_frames"] == 1 and c["sent_bytes"] == 16 + 8 + 30
    with pytest.raises(ValueError):
        transport.listen("0.0.0.0", 0)
    srv.close()


def test_two_ranks_match_single_rank_over_loopback(tmp_path):
    dtype = torch.bfloat16
    from nocturne.engines.torch_tp import engine as E
    from nocturne.engines.torch_tp.model import Qwen2TP
    from nocturne.engines.torch_tp.loader import load_rank
    model_dir = write_tiny_model(tmp_path / "tiny")
    cfg = load_config(model_dir)
    # single rank reference in the requested dtype
    sh = Shard(0, 1, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    w1, _ = load_rank(model_dir, cfg, sh, "cpu", dtype=dtype, log=lambda *a: None)
    m1 = Qwen2TP(cfg, w1, sh, "cpu", 64, dtype=dtype)
    ids = [5, 17, 99, 3, 42, 8]
    h = m1.embed_tokens(ids)
    ref_prefill = m1.layers_forward(h, 0, E.Comm())
    ref_logits = m1.logits_last(ref_prefill)
    ref_dec = m1.layers_forward(m1.embed_tokens([int(ref_logits.argmax())]), len(ids), E.Comm())
    # two ranks in process: rank 1 served on loopback in a thread, rank 0 driven directly
    free = socket.socket(); free.bind(("127.0.0.1", 0)); port = free.getsockname()[1]; free.close()
    th = threading.Thread(target=serve_rank, args=(str(model_dir), 1, 2, "127.0.0.1", port, "cpu"), kwargs={"log": lambda *a: None}, daemon=True)
    th.start()
    # Rank0 loads in bf16 by default; load in the test dtype by hand through the same pieces
    sh0 = Shard(0, 2, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    w0, _ = load_rank(model_dir, cfg, sh0, "cpu", dtype=dtype, log=lambda *a: None)
    m0 = Qwen2TP(cfg, w0, sh0, "cpu", 64, dtype=dtype)
    link = transport.Link(transport.connect("127.0.0.1", port))
    link.send(transport.KIND_HELLO, json.dumps({"world": 2, "max_seq_len": 64}).encode(), "uint8")
    assert link.recv()[0] == transport.KIND_HELLO
    comm = E.LinkComm(link, "cpu")
    link.send(transport.KIND_STEP, E._to_bytes(h.to(dtype)), "bfloat16", tuple(h.shape), aux0=0)
    out_prefill = m0.layers_forward(h.to(dtype), 0, comm)
    check_bf16(out_prefill, ref_prefill, "prefill")
    nxt = int(m0.logits_last(out_prefill).argmax())
    assert nxt == int(ref_logits.argmax())
    h2 = m0.embed_tokens([nxt])
    link.send(transport.KIND_STEP, E._to_bytes(h2), "bfloat16", tuple(h2.shape), aux0=len(ids))
    out_dec = m0.layers_forward(h2, len(ids), comm)
    check_bf16(out_dec, ref_dec, "decode")
    c = link.counters.snapshot()
    assert c["allreduce_frames"] == 2 * 2 * cfg.num_hidden_layers and c["step_frames"] == 2     # 2 steps x 2 per layer
    link.send(transport.KIND_METRICS); kind, payload, *_ = link.recv()
    assert kind == transport.KIND_METRICS_REPLY and json.loads(payload)["loaded_tensor_bytes"] > 0
    link.send(transport.KIND_SHUTDOWN); link.close(); th.join(timeout=5)


def test_two_ranks_in_process_fp32_exact_to_1e4(tmp_path):
    """Both ranks in one process with a pair comm: fp32 sharded forward equals the unsharded one to 1e-4."""
    from nocturne.engines.torch_tp import engine as E
    from nocturne.engines.torch_tp.model import Comm, Qwen2TP
    from nocturne.engines.torch_tp.loader import load_rank
    model_dir = write_tiny_model(tmp_path / "tiny32")
    cfg = load_config(model_dir)
    mk = lambda r, w: Shard(r, w, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    ws = [load_rank(model_dir, cfg, mk(r, 2), "cpu", dtype=torch.float32, log=lambda *a: None)[0] for r in (0, 1)]
    ms = [Qwen2TP(cfg, ws[r], mk(r, 2), "cpu", 64, dtype=torch.float32) for r in (0, 1)]
    w1, _ = load_rank(model_dir, cfg, mk(0, 1), "cpu", dtype=torch.float32, log=lambda *a: None)
    m1 = Qwen2TP(cfg, w1, mk(0, 1), "cpu", 64, dtype=torch.float32)

    class Pair(Comm):           # lock step: rank 0 and rank 1 alternate calls; the partner's partial is stored
        def __init__(self): self.box = []
        def all_reduce(self, x):
            self.box.append(x)
            if len(self.box) == 2:
                s = self.box[0] + self.box[1]; self.box.clear(); self.last = s
                return s
            return None
    # run layer by layer in lock step: easiest is to run rank 0 and rank 1 in two threads with a barrier-like exchange
    import queue
    q01, q10 = queue.Queue(), queue.Queue()

    class Side(Comm):
        def __init__(self, out, inp): self.out, self.inp = out, inp
        def all_reduce(self, x):
            self.out.put(x); y = self.inp.get(); return x + y
    ids = [1, 2, 3, 4, 5, 6, 7]
    h = m1.embed_tokens(ids)
    res = {}
    def r1(): res[1] = ms[1].layers_forward(h.clone(), 0, Side(q10, q01))
    t = threading.Thread(target=r1); t.start()
    res[0] = ms[0].layers_forward(h.clone(), 0, Side(q01, q10)); t.join()
    ref = m1.layers_forward(h, 0, Comm())
    err = (res[0] - ref).abs().max().item()
    assert err <= TOL_FP32, err
    assert torch.equal(res[0], res[1])       # commutative add: bit identical on both ranks


def test_microbench_exchange_on_loopback_and_trace_summary():
    from nocturne.engines.torch_tp import microbench
    from nocturne.engines.torch_tp.engine import LinkComm, STAGES
    srv = transport.listen("127.0.0.1", 0)
    port = srv.getsockname()[1]

    def peer():
        sock, _ = srv.accept()
        link = transport.Link(sock)
        kind, payload, *_ = link.recv()
        h = json.loads(payload.decode())
        link.send(transport.KIND_HELLO, b"ok", "uint8")
        for _ in range(h["warmup"] + h["steps"]):
            for _ in range(h["frames"]):
                microbench.exchange(link, b"\0" * h["bytes"], h["mode"])
        link.close(); srv.close()
    t = threading.Thread(target=peer, daemon=True); t.start()
    link = transport.Link(transport.connect("127.0.0.1", port))
    link.send(transport.KIND_HELLO, json.dumps({"frames": 4, "bytes": 64, "steps": 3, "mode": "inline", "warmup": 1}).encode(), "uint8")
    assert link.recv()[0] == transport.KIND_HELLO
    res = microbench.run_steps(link, 4, 64, 3, "inline", warmup=1)
    link.close(); t.join(2)
    assert res["frames"] == 4 and res["steps"] == 3 and res["exchange_ms"]["p50"] < 50 and res["gaps_over_30ms"] == 0
    # the trace summary averages per all-reduce over the traced steps and skips the first (prefill) step
    class L:        # a Link stand-in: all_reduce is not called here, only the accounting
        counters = None
    c = LinkComm(L(), "cpu", trace=True)
    c.begin_step(); c._cur.update({k: 1_000_000 for k in STAGES} | {"n": 2, "total": 10_000_000}); c.end_step()
    c.begin_step(); c._cur.update({k: 2_000_000 for k in STAGES} | {"n": 2, "total": 20_000_000}); c.end_step()
    s = c.trace_summary()
    assert s["steps"] == 1 and s["allreduces"] == 2 and s["ms_per_allreduce"]["to_cpu"] == 1.0 and s["ms_per_allreduce"]["total"] == 10.0
    assert LinkComm(L(), "cpu", trace=True).trace_summary() is None


def _pair():
    """Two connected Links on loopback."""
    srv = transport.listen("127.0.0.1", 0)
    port = srv.getsockname()[1]
    box = {}
    t = threading.Thread(target=lambda: box.update(sock=srv.accept()[0]))
    t.start()
    a = transport.Link(transport.connect("127.0.0.1", port))
    t.join()
    srv.close()
    return a, transport.Link(box["sock"])


def test_recv_into_buffer_and_large_frame_thread():
    a, b = _pair()
    x = torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3)
    from nocturne.engines.torch_tp.engine import _wire
    a.send(transport.KIND_ALLREDUCE, _wire(x), "bfloat16", (2, 3))
    buf = bytearray(12)
    kind, payload, dtype, shape, *_ = b.recv(into=memoryview(buf))
    assert kind == transport.KIND_ALLREDUCE and shape == (2, 3) and isinstance(payload, memoryview)
    assert torch.equal(torch.frombuffer(buf, dtype=torch.bfloat16).reshape(2, 3), x)
    # a buffer of the wrong size is ignored and a fresh bytes object comes back
    a.send(transport.KIND_STEP, b"\1\2\3\4", "uint8", (4,), aux0=9)
    kind, payload, *_ = b.recv(into=memoryview(bytearray(2)))
    assert kind == transport.KIND_STEP and payload == b"\1\2\3\4"
    # a frame above SMALL_FRAME goes through the engine's thread path and round-trips intact
    from nocturne.engines.torch_tp.engine import LinkComm
    big = torch.ones(transport.SMALL_FRAME // 2 + 8, dtype=torch.bfloat16)
    ca, cb = LinkComm(a, "cpu"), LinkComm(b, "cpu")
    out = {}
    t = threading.Thread(target=lambda: out.update(b=cb.all_reduce(big * 2)))
    t.start()
    out["a"] = ca.all_reduce(big)
    t.join()
    assert torch.equal(out["a"], big * 3) and torch.equal(out["b"], big * 3)
    a.close(); b.close()


@pytest.mark.parametrize("reduce_dtype", ["bfloat16", "float32"])
def test_linkcomm_sum_matches_on_both_ranks(reduce_dtype):
    from nocturne.engines.torch_tp.engine import LinkComm
    a, b = _pair()
    ca, cb = LinkComm(a, "cpu", reduce_dtype=reduce_dtype), LinkComm(b, "cpu", reduce_dtype=reduce_dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3584, generator=g).to(torch.bfloat16)
    y = torch.randn(1, 3584, generator=g).to(torch.bfloat16)
    out = {}
    t = threading.Thread(target=lambda: out.update(b=cb.all_reduce(y)))
    t.start()
    out["a"] = ca.all_reduce(x)
    t.join()
    want = (x.float() + y.float()).to(torch.bfloat16) if reduce_dtype == "float32" else x + y
    assert torch.equal(out["a"], want) and torch.equal(out["b"], want)
    # the receive buffer is reused: a second round with new values must not see the old ones
    t = threading.Thread(target=lambda: out.update(b=cb.all_reduce(x)))
    t.start()
    out["a"] = ca.all_reduce(x)
    t.join()
    assert torch.equal(out["a"], x + x) and torch.equal(out["b"], x + x)
    assert a.counters.snapshot()["allreduce_frames"] == 2
    with pytest.raises(ValueError):
        LinkComm(a, "cpu", reduce_dtype="float16")
    a.close(); b.close()


def test_rank0_reconnects_after_link_loss_and_logs_request_lines(tmp_path):
    from nocturne.engines.torch_tp.engine import LinkLost
    from nocturne.bench.live import parsers
    model_dir = write_tiny_model(tmp_path / "tiny")
    free = socket.socket(); free.bind(("127.0.0.1", 0)); port = free.getsockname()[1]; free.close()
    worker_log: list[str] = []
    th = threading.Thread(target=serve_rank, args=(str(model_dir), 1, 2, "127.0.0.1", port, "cpu"),
                          kwargs={"log": worker_log.append}, daemon=True)
    th.start()
    lines: list[str] = []
    r0 = Rank0(str(model_dir), 2, 64, "cpu", f"127.0.0.1:{port}", log=lines.append, progress_every=2)
    ids = [5, 17, 99, 3]
    res = r0.generate(ids, 4, ignore_eos=True, rid="req-1")
    assert len(res["tokens"]) == 4 and res["finish_reason"] == "length"
    kinds = [(parsers.parse_server_line(l) or {}).get("kind") for l in lines]
    assert kinds[:2] == ["load_start", None]
    assert [k for k in kinds if k] == ["load_start", "task_start", "prompt_done", "progress", "progress", "eval_done", "release"]
    # the link breaks under rank 0: the request fails, rank 1 keeps its slice, the next request reconnects
    r0.link.sock.shutdown(socket.SHUT_RDWR)
    with pytest.raises(LinkLost):
        r0.generate(ids, 4, ignore_eos=True, rid="req-2")
    assert r0.link is None and any("link lost during req-2" in l for l in lines)
    res2 = r0.generate(ids, 4, ignore_eos=True, rid="req-3")
    assert res2["tokens"] == res["tokens"]
    assert r0.counters.snapshot()["reconnects"] == 1
    assert any("slice kept" in l for l in worker_log) and any('"cached": true' in l for l in lines)
    m = r0.metrics()
    assert m["link_up"] and m["counters"]["reconnects"] == 1 and m["reduce_dtype"] == "bfloat16"
    r0.shutdown(); th.join(timeout=5)
