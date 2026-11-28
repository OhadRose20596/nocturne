"""Session 10, part 3: compute while data is on the network. Slots (several requests in flight), chunked prefill,
two requests interleaved in a TP group, sends off the compute thread, interleaved pipeline stages.
The rule under test everywhere: the tokens of a request do not depend on how the work was scheduled.
Loopback in threads with tiny random Qwen2 models (fp32, CPU)."""
import json
import threading

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from nocturne.engines.torch_tp import hybrid, transport  # noqa: E402
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.plan import (Dims, Placement, PlanError, RankIn, StageIn, chunk_ranges, frames_per_step,  # noqa: E402
                                            frames_prefill, host_totals, placement_from_dict, plan, PROCESS_OVERHEAD)

from .test_hybrid import free_ports  # noqa: E402
from .test_torch_tp import TINY  # noqa: E402

PROMPTS = [[5, 17, 99, 3, 42], [8, 61], [7, 7, 23, 90, 11, 4, 100], [33, 2, 64], [120, 19, 6, 55, 71, 12]]


def write_tiny(d, layers=4, seed=3, **over):
    """The tiny Qwen2 of test_torch_tp with a chosen number of layers (interleaved stages need four). `over`: other
    config values (a large vocabulary makes a large head frame)."""
    g = torch.Generator().manual_seed(seed)
    c = dict(TINY, num_hidden_layers=layers, **over)
    hd = c["hidden_size"] // c["num_attention_heads"]
    r = lambda *s: torch.randn(*s, generator=g) * 0.1
    w = {"model.embed_tokens.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5,
         "model.norm.weight": torch.ones(c["hidden_size"]), "lm_head.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5}
    for i in range(layers):
        p = f"model.layers.{i}."
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


@pytest.fixture(scope="module")
def tiny4(tmp_path_factory):
    return write_tiny(tmp_path_factory.mktemp("s10o") / "t4")


def R(i, host="h"):
    return RankIn(host, "cpu", i)


def pp3(p):
    return Placement("m", (StageIn("s0", (0, 1), (R(p[0]),), embed=True), StageIn("s1", (1, 3), (R(p[1]),)),
                           StageIn("s2", (3, 4), (R(p[2]),), lm_head=True)), max_seq_len=64, granule=16)


def tp3(p):
    return Placement("m", (StageIn("s0", (0, 4), (R(p[0]), R(p[1]), R(p[2])), embed=True, lm_head=True),), max_seq_len=64, granule=16)


def hybrid_tp_first(p):
    """A TP pair, then one rank, then a head group of two."""
    return Placement("m", (StageIn("s0", (0, 2), (R(p[0]), R(p[1])), embed=True), StageIn("s1", (2, 4), (R(p[2]),)),
                           StageIn("head", None, (R(p[3]), R(p[4])), lm_head=True)), max_seq_len=64, granule=16)


def hybrid_pp_first(p):
    """One rank, then a TP group of three with the head: a pair made by the coordinator travels through the first stage."""
    return Placement("m", (StageIn("s0", (0, 1), (R(p[0]),), embed=True),
                           StageIn("s1", (1, 4), (R(p[1]), R(p[2]), R(p[3])), lm_head=True)), max_seq_len=64, granule=16)


def boot(model, placement, slots=1, device="cpu", **kw):
    cfg = load_config(model)
    specs = plan(Dims.of(cfg), placement, kv_slots=slots)
    threads = []
    for s in specs[1:]:
        t = threading.Thread(target=hybrid.rank_main, args=(["127.0.0.1"], s.port, str(model), device),
                             kwargs={"log": lambda *a: None, "run_dir": kw.get("run_dir")}, daemon=True)
        t.start()
        threads.append(t)
    co = hybrid.Coordinator(specs, str(model), {"h": "127.0.0.1", "g": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                            placement.prefill_max, device=device, log=lambda *a: None, dtype="float32", slots=slots, **kw)
    return specs, co, threads


def reqs(prompts=PROMPTS, n=6):
    return [{"ids": list(p), "max_tokens": n, "ignore_eos": True, "rid": f"p{i}", "first_logits": True} for i, p in enumerate(prompts)]


def alone(model, make, prompts=PROMPTS, n=6):
    """Every request by itself, on the old path (generate)."""
    specs, co, th = boot(model, make(free_ports(8)))
    try:
        out = [co.generate(list(p), n, ignore_eos=True, first_logits=True, rid=f"p{i}") for i, p in enumerate(prompts)]
        m = co.metrics()
    finally:
        co.shutdown()
    for t in th:
        t.join(timeout=10)
    return specs, out, m


def many(model, make, slots, chunk=0, interleave=1, prompts=PROMPTS, n=6, **kw):
    specs, co, th = boot(model, make(free_ports(8)), slots=slots, **kw)
    try:
        m0 = co.metrics()
        out = co.generate_many(reqs(prompts, n), prefill_chunk=chunk, tp_interleave=interleave)
        m = co.metrics()
    finally:
        co.shutdown()
    for t in th:
        t.join(timeout=10)
    return specs, out, m, m0, co


def same(a, b, logits="equal"):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x["tokens"] == y["tokens"], f"request {i}"
        if logits == "equal":
            assert torch.equal(x["first_logits"], y["first_logits"]), f"request {i}: first step logits differ"
        elif logits == "close":
            assert torch.allclose(x["first_logits"], y["first_logits"], atol=1e-4), f"request {i}: first step logits are not close"


def data_frames(m):
    """(decode data frames sent, all data frames sent) over every link of a metrics snapshot."""
    kinds = ("allreduce", "token", "hidden", "head")
    dec = tot = 0
    for r in m["ranks"].values():
        for c in r["links"].values():
            dec += c["decode"]["sent_frames"]
            tot += sum(c["by_kind_sent"].get(k, 0) for k in kinds)
    return dec, tot


# ---- O1: several requests in flight -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [pp3, hybrid_tp_first, hybrid_pp_first, tp3])
@pytest.mark.parametrize("slots", [2, 3])
def test_each_request_in_flight_gives_the_tokens_it_gives_alone(tiny4, make, slots):
    _, ref, _ = alone(tiny4, make)
    specs, out, m, m0, co = many(tiny4, make, slots)
    same(ref, out)
    assert sorted({o["slot"] for o in out}) == list(range(slots))        # every slot was used
    assert [r["rid"] for r in sorted(co.requests, key=lambda r: r["req"])] == [f"p{i}" for i in range(len(PROMPTS))]
    steps = m["decode_steps"] - m0["decode_steps"]
    dec = data_frames(m)[0] - data_frames(m0)[0]
    assert steps == len(PROMPTS) * 5 and dec == steps * frames_per_step(specs)["total"]      # the frame formula, per step of one request


def test_one_slot_through_the_scheduler_is_the_old_path_frame_for_frame(tiny4):
    """Options off must mean the old bytes. generate_many with one slot sends what generate sends."""
    sent = {"old": [], "new": []}
    real = transport.Link.send

    def spy(key):
        def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
            if kind in transport.DATA_KINDS:
                sent[key].append((kind, bytes(payload), dtype, tuple(shape), aux0, aux1))
            return real(self, kind, payload, dtype, shape, aux0, aux1, **kw)
        return send

    for key, fn in (("old", lambda: alone(tiny4, hybrid_tp_first)[1]), ("new", lambda: many(tiny4, hybrid_tp_first, 1)[1])):
        transport.Link.send = spy(key)
        try:
            res = fn()
        finally:
            transport.Link.send = real
        sent[key + "_res"] = res
    same(sent["old_res"], sent["new_res"])
    assert len(sent["old"]) == len(sent["new"]) > 100
    assert sorted(sent["old"]) == sorted(sent["new"])                      # the same frames, byte for byte (threads may order them differently)
    assert all(aux1 >> 8 == 0 for kind, *_, aux1 in sent["new"] if kind in (transport.KIND_TOKEN, transport.KIND_HIDDEN))
    assert all(aux0 >> 24 == 0 for kind, *_, aux0, _ in sent["new"] if kind == transport.KIND_HEAD)


def test_the_slot_travels_in_every_token_hidden_and_head_frame(tiny4):
    real = transport.Link.send
    seen = {"th": set(), "head": set(), "flags": set()}

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
        if kind in (transport.KIND_TOKEN, transport.KIND_HIDDEN):
            seen["th"].add(aux1 >> hybrid.SLOT_SHIFT)
            seen["flags"].add(aux1 & 0xFF)
            assert (aux1 >> hybrid.REQ_SHIFT) & hybrid.REQ_MASK == 0       # no recorder: no request number
        elif kind == transport.KIND_HEAD:
            seen["head"].add(aux0 >> hybrid.SLOT_SHIFT)
            assert aux0 & hybrid.IDX_MASK < 128
        return real(self, kind, payload, dtype, shape, aux0, aux1, **kw)
    transport.Link.send = send
    try:
        many(tiny4, pp3, 3)
    finally:
        transport.Link.send = real
    assert seen["th"] == {0, 1, 2} and seen["head"] == {0, 1, 2}
    assert hybrid.pack_aux(hybrid.FLAG_LOGITS, 7, 2) == 1 | 7 << 8 | 2 << 24 and hybrid.pack_aux(0, 0, 0) == 0


def test_slots_need_room_and_are_checked(tiny4):
    with pytest.raises(PlanError, match="slots must be 1 or more"):
        boot(tiny4, pp3(free_ports(3)), slots=0)
    cfg0 = load_config(tiny4)
    spec0 = plan(Dims.of(cfg0), pp3(free_ports(3)))
    with pytest.raises(ValueError, match="slots must be 1 to 256"):
        hybrid.RankRuntime(spec0[0], spec0, cfg0, str(tiny4), "cpu", 64, "float32", transport.Phase(), slots=257)
    specs, co, th = boot(tiny4, pp3(free_ports(3)), slots=2)
    try:
        with pytest.raises(ValueError, match="needs --slots 2"):
            hybrid.Coordinator.generate_many(type("C", (), {"slots": 1})(), [], tp_interleave=2)
        with pytest.raises(ValueError, match="longer than prefill_max"):
            co.generate_many([{"ids": list(range(1, 60)) * 5, "max_tokens": 2}])
        kv = [sum(2 * l.k_cache.numel() * l.k_cache.element_size() for l in co.rank.model.layers)]
        assert co.rank.model.layers[0].k_cache.shape[0] == 2 and kv[0] == co.rank.kv_bytes
    finally:
        co.shutdown()
    cfg = load_config(tiny4)
    one, two = plan(Dims.of(cfg), pp3(free_ports(3))), plan(Dims.of(cfg), pp3(free_ports(3)), kv_slots=2)
    for a, b in zip(one, two):
        assert b.planned["kv"] == 2 * a.planned["kv"] and b.planned["weights"] == a.planned["weights"]
    with pytest.raises(PlanError, match="slots must be 1 or more"):
        plan(Dims.of(cfg), pp3(free_ports(3)), kv_slots=0)


# ---- O2: chunked prefill ----------------------------------------------------------------------------------------------------
def test_chunk_ranges_cover_the_prompt_and_never_end_in_one_token():
    assert chunk_ranges(47, 0) == [(0, 47)] and chunk_ranges(10, 16) == [(0, 10)] and chunk_ranges(16, 16) == [(0, 16)]
    assert chunk_ranges(47, 16) == [(0, 16), (16, 32), (32, 47)]
    assert chunk_ranges(49, 16) == [(0, 16), (16, 32), (32, 49)]           # a remainder of one joins the chunk before it
    assert chunk_ranges(3, 2) == [(0, 3)] and chunk_ranges(5, 2) == [(0, 2), (2, 5)]
    for n in range(1, 70):
        for c in (1, 2, 3, 7, 16):
            r = chunk_ranges(n, c)
            assert r[0][0] == 0 and r[-1][1] == n and all(a[1] == b[0] for a, b in zip(r, r[1:]))
            assert all(b - a >= 2 for a, b in r) or n == 1 or c == 1 and all(b - a >= 1 for a, b in r)


@pytest.mark.parametrize("make", [pp3, hybrid_tp_first, hybrid_pp_first])
@pytest.mark.parametrize("slots,chunk", [(1, 2), (1, 3), (3, 2)])
def test_chunked_prefill_gives_the_same_tokens_and_the_predicted_frames(tiny4, make, slots, chunk):
    _, ref, _ = alone(tiny4, make)
    specs, out, m, m0, co = many(tiny4, make, slots, chunk=chunk)
    same(ref, out, logits="close")
    worst = max((a["first_logits"] - b["first_logits"]).abs().max().item() for a, b in zip(ref, out))
    assert worst <= 1e-4, worst                                             # the same math on other shapes: last bit at most
    n_chunks = [len(chunk_ranges(len(p), chunk)) for p in PROMPTS]
    assert [o["chunks"] for o in out] == n_chunks and max(n_chunks) >= 2
    dec, tot = (b - a for a, b in zip(data_frames(m0), data_frames(m)))
    assert dec == (m["decode_steps"] - m0["decode_steps"]) * frames_per_step(specs)["total"]
    assert tot - dec == sum(frames_prefill(specs, k) for k in n_chunks)     # prefill frames: per chunk, head only for the last
    assert m["prefill_steps"] - m0["prefill_steps"] == sum(n_chunks)


def test_a_chunk_that_is_not_the_last_asks_for_no_head_result(tiny4):
    real = transport.Link.send
    heads, nohead = [], []

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
        if kind == transport.KIND_HEAD:
            heads.append(aux0)
        elif kind == transport.KIND_HIDDEN and aux1 & hybrid.FLAG_NOHEAD:
            nohead.append((aux0, shape[0]))
        return real(self, kind, payload, dtype, shape, aux0, aux1, **kw)
    transport.Link.send = send
    try:
        many(tiny4, pp3, 1, chunk=2, prompts=[PROMPTS[2]], n=2)
    finally:
        transport.Link.send = real
    assert len(heads) == 2                                                   # the last chunk and one decode step
    assert sorted(set(nohead)) == [(0, 2), (2, 2)]                           # two chunks of two tokens at offsets 0 and 2, on both hops


# ---- O3: two requests interleaved in a TP group ----------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [tp3, hybrid_tp_first, hybrid_pp_first])
@pytest.mark.parametrize("slots", [2, 3])
def test_interleaved_requests_give_the_same_tokens_and_the_same_sums(tiny4, make, slots):
    _, ref, _ = alone(tiny4, make)
    specs, out, m, m0, co = many(tiny4, make, slots, interleave=2)
    same(ref, out)                                                           # first step logits bit equal: the sums are the same bytes
    steps = m["decode_steps"] - m0["decode_steps"]
    assert data_frames(m)[0] - data_frames(m0)[0] == steps * frames_per_step(specs)["total"]


def test_a_pair_is_sent_before_it_is_received_and_keeps_its_order_on_every_link(tiny4):
    """In a group of two: A's partial leaves, then B's, and only then A's sum is read. So A travels while B is computed."""
    real_send, real_recv = transport.Link.send, transport.Link.recv
    log = []

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
        if kind == transport.KIND_ALLREDUCE:
            log.append(("send", threading.get_ident(), id(self)))
        return real_send(self, kind, payload, dtype, shape, aux0, aux1, **kw)

    def recv(self, *a, **kw):
        out = real_recv(self, *a, **kw)
        if out[0] == transport.KIND_ALLREDUCE:
            log.append(("recv", threading.get_ident(), id(self)))
        return out
    transport.Link.send, transport.Link.recv = send, recv
    two = lambda p: Placement("m", (StageIn("s0", (0, 4), (R(p[0]), R(p[1])), embed=True, lm_head=True),), max_seq_len=64, granule=16)
    try:
        many(tiny4, two, 2, interleave=2, prompts=PROMPTS[:2], n=4)
    finally:
        transport.Link.send, transport.Link.recv = real_send, real_recv
    for tid in {t for _, t, _ in log}:
        mine = [k for k, t, _ in log if t == tid]
        pairs = "".join("s" if k == "send" else "r" for k in mine)
        assert "ssrr" in pairs, pairs[:60]                                    # two sends, then two receives: the lockstep
        assert pairs.replace("ssrr", "").replace("sr", "") == "", pairs[:80]  # nothing but pairs and single steps


def test_interleave_needs_two_slots_and_a_value_of_one_or_two(tiny4):
    from nocturne.engines.torch_tp.model import Comm
    specs, co, th = boot(tiny4, tp3(free_ports(3)), slots=2)
    try:
        with pytest.raises(ValueError, match="tp-interleave must be 1 or 2"):
            co.generate_many(reqs(PROMPTS[:1]), tp_interleave=3)
        h = torch.zeros(1, 64)
        with pytest.raises(ValueError, match="a pair needs two slots"):
            co.rank.model.layers_forward_pair(h, 0, 1, h, 0, 1, Comm())
        with pytest.raises(ValueError, match="slot 5 of 2"):
            co.rank.model.layers_forward(h, 0, Comm(), 5)
    finally:
        co.shutdown()


# ---- all three together -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [hybrid_tp_first, hybrid_pp_first, tp3, pp3])
def test_slots_chunks_and_interleave_together(tiny4, make):
    _, ref, _ = alone(tiny4, make)
    specs, out, m, m0, co = many(tiny4, make, 3, chunk=2, interleave=2)
    same(ref, out, logits="close")
    n_chunks = [len(chunk_ranges(len(p), 2)) for p in PROMPTS]
    dec, tot = (b - a for a, b in zip(data_frames(m0), data_frames(m)))
    assert dec == (m["decode_steps"] - m0["decode_steps"]) * frames_per_step(specs)["total"]
    assert tot - dec == sum(frames_prefill(specs, k) for k in n_chunks)


# ---- O5: interleaved pipeline stages and the planner's per host sum ---------------------------------------------------------------
def interleaved(p):
    """Two hosts, four stages: h holds layers 0 and 2, g holds layers 1 and 3 with the head."""
    return Placement("m", (StageIn("a", (0, 1), (R(p[0], "h"),), embed=True), StageIn("b", (1, 2), (R(p[1], "g"),)),
                           StageIn("c", (2, 3), (R(p[2], "h"),)), StageIn("d", (3, 4), (R(p[3], "g"),), lm_head=True)), max_seq_len=64, granule=16)


def contiguous(p):
    return Placement("m", (StageIn("a", (0, 2), (R(p[0], "h"),), embed=True), StageIn("d", (2, 4), (R(p[1], "g"),), lm_head=True)),
                     max_seq_len=64, granule=16)


def test_interleaved_stages_give_the_tokens_of_the_contiguous_placement(tiny4):
    _, ref, m_ref = alone(tiny4, contiguous)
    specs, out, m = alone(tiny4, interleaved)
    same(ref, out)
    assert [s.host for s in specs] == ["h", "g", "h", "g"] and frames_per_step(specs)["hops"] == 3
    specs, out, m, m0, co = many(tiny4, interleaved, 3, chunk=2)
    same(ref, out, logits="close")
    assert data_frames(m)[0] - data_frames(m0)[0] == (m["decode_steps"] - m0["decode_steps"]) * frames_per_step(specs)["total"]


def test_planner_sums_all_rank_processes_of_a_host_against_the_host_budget(tiny4):
    cfg = Dims.of(load_config(tiny4))
    specs = plan(cfg, interleaved([1, 2, 3, 4]))
    ht = host_totals(specs)
    assert ht["h"]["ranks"] == ["a/0", "c/0"] and ht["g"]["ranks"] == ["b/0", "d/0"] and ht["h"]["processes"] == 2
    assert ht["h"]["planned"] == specs[0].planned["footprint"] + specs[2].planned["footprint"]
    assert ht["h"]["total"] == ht["h"]["planned"] + 2 * PROCESS_OVERHEAD["cpu"] and ht["h"]["budget"] is None
    cuda = Placement("m", tuple(StageIn(s.name, s.layers, tuple(RankIn(r.host, "cuda", r.port) for r in s.ranks), s.embed, s.lm_head)
                                for s in interleaved([1, 2, 3, 4]).stages), max_seq_len=64, granule=16)
    sc = plan(cfg, cuda)
    hc = host_totals(sc)
    assert hc["h"]["process_overhead"] == 2 * PROCESS_OVERHEAD["cuda"] == 300 << 20         # a CUDA context per rank process
    need_h = hc["h"]["total"] / 1e9
    ok = Placement(**{**cuda.__dict__, "host_budget_gb": (("g", 10.0), ("h", need_h + 0.001))})
    assert len(plan(cfg, ok)) == 4
    tight = Placement(**{**cuda.__dict__, "host_budget_gb": (("g", 10.0), ("h", need_h - 0.01))})
    with pytest.raises(PlanError, match=r"host h: its 2 rank processes \(a/0, c/0\) plan .* > host budget"):
        plan(cfg, tight)
    assert len(plan(cfg, tight, check_budget=False)) == 4
    # each rank alone would fit that budget: only the sum per host catches it
    assert all(s.planned["total"] + PROCESS_OVERHEAD["cuda"] < (need_h - 0.01) * 1e9 for s in sc if s.host == "h")
    with pytest.raises(PlanError, match="names hosts that hold no rank"):
        plan(cfg, Placement(**{**cuda.__dict__, "host_budget_gb": (("nobody", 1.0),)}))
    y = {"model": "m", "host_budget_gb": {"h": 7.5, "g": 7}, "stages": [{"layers": [0, 4], "ranks": [{"host": "h", "port": 1}]}]}
    assert placement_from_dict(y).host_budget_gb == (("g", 7.0), ("h", 7.5))
    more = plan(cfg, cuda, kv_slots=3)
    assert host_totals(more)["h"]["planned"] - hc["h"]["planned"] == 2 * (sc[0].planned["kv"] + sc[2].planned["kv"])


# ---- O4: sends off the compute thread ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["inline", "thread"])
@pytest.mark.parametrize("make", [pp3, tp3, hybrid_tp_first, hybrid_pp_first])
def test_async_send_gives_the_same_tokens_logits_and_frames(tiny4, make, mode, monkeypatch):
    monkeypatch.setenv("NOCTURNE_ASYNC_COPY", mode)
    _, ref, m_ref = alone(tiny4, make)
    specs, co, th = boot(tiny4, make(free_ports(8)), async_send=True)
    try:
        out = [co.generate(list(p), 6, ignore_eos=True, first_logits=True, rid=f"p{i}") for i, p in enumerate(PROMPTS)]
        m = co.metrics()
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    same(ref, out)
    assert data_frames(m) == data_frames(m_ref)
    for name, r in m["ranks"].items():
        assert r["async_send"]["copy"] == mode
        # data links byte for byte; a ctl link also carries metrics replies (JSON of another length), so its decode data counts
        pick = lambda links: {k: v["decode"] if k.startswith("ctl") else (v["sent_bytes"], v["recv_bytes"]) for k, v in links.items()}
        assert pick(r["links"]) == pick(m_ref["ranks"][name]["links"]), name
    assert all(r["async_send"] is None for r in m_ref["ranks"].values())
    # every data frame a rank sends, except the coordinator's token frames, left through its sender thread
    tokens = sum(c["by_kind_sent"].get("token", 0) for c in m["ranks"][specs[0].name]["links"].values())
    local_head = 0          # the coordinator's own head result is not a frame
    assert sum(r["async_send"]["frames"] for r in m["ranks"].values()) == data_frames(m)[1] - tokens - local_head


@pytest.mark.parametrize("make,slots,chunk,interleave", [(pp3, 3, 3, 1), (tp3, 2, 0, 2), (hybrid_tp_first, 3, 3, 2), (hybrid_pp_first, 2, 3, 2)])
def test_async_send_with_slots_chunks_and_pairs(tiny4, make, slots, chunk, interleave, monkeypatch):
    monkeypatch.setenv("NOCTURNE_ASYNC_COPY", "thread")
    _, ref, _ = alone(tiny4, make)
    _, sync_out, m_sync, m0s, _ = many(tiny4, make, slots, chunk=chunk, interleave=interleave)
    _, out, m, m0, _ = many(tiny4, make, slots, chunk=chunk, interleave=interleave, async_send=True)
    same(ref, out, logits="equal" if not chunk else "close")
    same(sync_out, out)
    assert data_frames(m) == data_frames(m_sync)


def test_head_on_the_device_is_the_head_argmax_also_with_ties(tiny4):
    from nocturne.engines.torch_tp.model import Qwen2TP
    m = Qwen2TP.__new__(Qwen2TP)
    m.head_rows = (40, 52)
    for lg in ([0.5, 3.0, -1.0, 3.0, 2.0, 3.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [7.0] * 12, [-5.0] * 11 + [-4.0]):
        t = torch.tensor(lg, dtype=torch.bfloat16)
        m.logits_last = lambda h, t=t: t
        val, idx, _ = m.head_argmax(None)
        pair, lgd = m.head_device(None)
        assert (float(pair[0]), 40 + int(pair[1])) == (val, idx)
        assert hybrid.f32_bits(float(pair[0])) == hybrid.f32_bits(val)
        assert torch.equal(lgd, t)


class _FakeLink:
    """A link that writes down what it is asked to send. `slow`: seconds the first frame takes."""

    fits_inline = transport.Link.fits_inline

    def __init__(self, slow=0.0, sndbuf=8 << 20):
        import time as _t
        self.sent, self.slow, self.last_sent, self._sleep = [], slow, 0, _t.sleep
        self.lock = threading.Lock()
        self.sndbuf = self.rcvbuf = sndbuf

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, decode=None):
        with self.lock:
            first = not self.sent and not getattr(self, "_started", False)
            self._started = True
        if first and self.slow:
            self._sleep(self.slow)
        self.sent.append((kind, bytes(payload), aux0, aux1))

    def close(self):
        self.closed = True

    def recv(self, *a, **kw):
        """The peer's partial for the next all-reduce: zeros of the shape given in `answers` (a list the test fills)."""
        shape = self.answers.pop(0)
        self.last_recv = 0
        return transport.KIND_ALLREDUCE, bytearray(4 * shape[0] * shape[1]), "float32", shape, 0, 0


def test_a_large_pair_keeps_its_order_on_the_link_when_frames_leave_on_threads(monkeypatch):
    """Frames over SMALL_FRAME leave on a thread each. The second frame of a pair must wait for the first one on its link."""
    monkeypatch.setattr(transport, "SMALL_FRAME", 8)
    link = _FakeLink(slow=0.15)
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
    a, b = torch.full((4, 8), 1.0), torch.full((1, 8), 2.0)
    ta = comm.send_partial(a, 0)
    tb = comm.send_partial(b, 1)            # small enough to go inline if nothing were in front of it... it is not: 32 bytes > 8
    for t in ta[2] + tb[2]:
        t.join(timeout=5)
    assert [len(p) for _, p, _, _ in link.sent] == [a.numel() * 4, b.numel() * 4]
    # a small frame behind a large one waits too
    monkeypatch.setattr(transport, "SMALL_FRAME", 64)
    link = _FakeLink(slow=0.15)
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
    ta = comm.send_partial(a, 0)            # 128 bytes: a thread
    tb = comm.send_partial(b, 1)            # 32 bytes: inline, unless the first is still leaving
    for t in ta[2] + tb[2]:
        t.join(timeout=5)
    assert [len(p) for _, p, _, _ in link.sent] == [a.numel() * 4, b.numel() * 4]


def test_large_pairs_through_real_links_give_the_same_tokens(tiny4, monkeypatch):
    monkeypatch.setattr(transport, "SMALL_FRAME", 16)
    _, ref, _ = alone(tiny4, tp3)
    _, out, _, _, _ = many(tiny4, tp3, 2, interleave=2)
    same(ref, out)


def test_sender_keeps_hand_over_order_and_reports_a_failure_to_the_compute_thread():
    link = _FakeLink(slow=0.1)
    s = hybrid.Sender("cpu", None)
    hs = []
    for k in range(6):
        h = s.copy(torch.full((1, 4), float(k)))
        s.send(h, link, transport.KIND_HIDDEN, k, 0, True, "x")
        s.free(h)
        hs.append(h)
    s.flush()
    assert [a0 for _, _, a0, _ in link.sent] == list(range(6))
    assert [torch.frombuffer(bytearray(p), dtype=torch.float32)[0].item() for _, p, _, _ in link.sent] == [float(k) for k in range(6)]

    # the compute thread gets its partial only once the copy is done, however long the sender is held up before it
    slow = _FakeLink(slow=0.3)
    h0 = s.copy(torch.ones(1, 4))
    s.send(h0, slow, transport.KIND_HIDDEN, 0, 0, True, "x")
    s.mode = "thread"
    h1 = s.copy(torch.full((1, 4), 9.0))
    got = s.host(h1)
    assert got is not None and torch.equal(got, torch.full((1, 4), 9.0)) and len(slow.sent) == 1
    s.mode = "inline"

    class Broken(_FakeLink):
        def send(self, *a, **kw):
            raise ConnectionError("peer closed the connection")
    h = s.copy(torch.zeros(1, 4))
    s.send(h, Broken(), transport.KIND_HIDDEN, 0, 0, True, "x")
    import time
    for _ in range(100):                    # the sender thread stores its error within a moment
        if s.error is not None:
            break
        time.sleep(0.02)
    with pytest.raises(ConnectionError, match="sender thread failed"):
        s.copy(torch.zeros(1, 4))
    with pytest.raises(ConnectionError, match="sender thread failed"):
        s.flush()


def test_async_send_records_sender_spans_and_the_bytes_of_every_link(tiny4, tmp_path, monkeypatch):
    from nocturne.engines.torch_tp import timeline
    monkeypatch.setenv("NOCTURNE_ASYNC_COPY", "thread")
    specs, co, th = boot(tiny4, hybrid_tp_first(free_ports(8)), slots=2, async_send=True, timeline=True, run_dir=str(tmp_path))
    try:
        m0 = co.metrics()
        co.generate_many(reqs(PROMPTS[:3], 5), tp_interleave=2)
        m = co.metrics()
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    run = timeline.load(tmp_path)
    kinds = {e["kind"] for rk in run["ranks"] for e in rk["events"]}
    assert {"tx_wait", "tx_send"} <= kinds
    sent = timeline.all_bytes(run)
    checked = 0
    for name, r in m["ranks"].items():
        for peer, c in r["links"].items():
            if peer.startswith("ctl") or (name, peer) == ("s0/0", "s0/1"):
                continue        # ctl links carry control frames too; the coordinator's token frames to s0/1 travel on one
            c0 = m0["ranks"][name]["links"][peer]
            ctl_n = lambda x: sum(n for k, n in x["by_kind_sent"].items() if k not in ("allreduce", "token", "hidden", "head"))
            # between the two snapshots a data link carries data frames and forwarded metrics requests (bare headers of 16 bytes)
            assert sent.get(f"{name} -> {peer}", 0) == c["sent_bytes"] - c0["sent_bytes"] - 16 * (ctl_n(c) - ctl_n(c0)), (name, peer)
            checked += 1
    assert checked == 9


# ---- the overlap share ----------------------------------------------------------------------------------------------------------
def _ev(kind, t0, t1, peer=None, nbytes=0, step=0, req=1, slot=0):
    return {"kind": kind, "t0": float(t0), "t1": float(t1), "step": step, "req": req, "peer": peer, "bytes": nbytes, "tokens": 1, "slot": slot}


def _run(a_events, b_events):
    mk = lambda n, g, ev: {"meta": {"rank": n, "gid": g, "host": "h" + n, "device": "cpu"}, "events": ev, "aligned": True}
    return {"ranks": [mk("a", 0, a_events), mk("b", 1, b_events)], "samples": []}


def test_interval_helpers():
    from nocturne.engines.torch_tp.timeline import intersect, length, union
    assert union([(5, 7), (0, 2), (1, 3), (7, 8), (4, 4)]) == [[0, 3], [5, 8]]
    assert intersect([[0, 3], [5, 8]], [[2, 6], [7, 20]]) == [[2, 3], [5, 6], [7, 8]]
    assert length([[2, 3], [5, 6], [7, 8]]) == 3


def test_a_frame_is_in_flight_from_the_send_start_to_the_end_of_the_wait_matched_by_order():
    from nocturne.engines.torch_tp.timeline import in_flight
    a = [_ev("compute", 0, 10), _ev("send", 10, 12, "b", 100), _ev("compute", 12, 40, req=2, slot=1), _ev("send", 40, 41, "b", 100, req=2, slot=1)]
    b = [_ev("recv_wait", 0, 20, "a", 100), _ev("compute", 20, 60), _ev("recv_wait", 60, 61, "a", 100, req=2, slot=1)]
    fl = in_flight(_run(a, b))["a -> b"]
    assert (fl["sent"], fl["received"], fl["bytes_match"]) == (2, 2, True)
    assert fl["frames"] == [[10.0, 20.0, 100, 1, 0], [40.0, 61.0, 100, 2, 1]]
    # a receive that ends before its send started (clock error) gives an empty interval, never a negative one
    b2 = [_ev("recv_wait", 0, 5, "a", 100)]
    assert in_flight(_run(a[:2], b2))["a -> b"]["frames"] == [[10.0, 10.0, 100, 1, 0]]


def test_overlap_share_is_compute_time_with_a_frame_in_flight_over_the_request_window():
    from nocturne.engines.torch_tp.timeline import overlap
    # rank a: computes 0-10, sends 10-12 (in flight until b's wait ends at 20), computes 12-40 for the next request
    a = [_ev("compute", 0, 10), _ev("send", 10, 12, "b", 100), _ev("compute", 12, 40, req=2, slot=1)]
    b = [_ev("recv_wait", 0, 20, "a", 100), _ev("compute", 20, 60), _ev("other", 60, 100)]
    o = overlap(_run(a, b), [{"t0_mono_ns": 0, "t1_mono_ns": 100}])
    ra, rb = o["per_rank"]["a"], o["per_rank"]["b"]
    assert o["window_ms"] == pytest.approx(100 / 1e6)
    assert ra["busy_share"] == pytest.approx(0.38)                     # 10 + 28 of 100
    assert ra["overlap_share"] == pytest.approx(0.08)                  # computing 12-20 while its frame is in flight
    assert ra["overlap_out_share"] == pytest.approx(0.08) and ra["overlap_in_share"] == 0
    assert rb["busy_share"] == pytest.approx(0.40) and rb["overlap_share"] == 0
    assert ra["in_flight_share"] == pytest.approx(0.10)
    assert o["unmatched_links"] == [] and o["async_send"] is False
    # the window is the union of the request intervals: time outside it does not count
    o2 = overlap(_run(a, b), [{"t0_mono_ns": 0, "t1_mono_ns": 20}, {"t0_mono_ns": 15, "t1_mono_ns": 50}])
    assert o2["window_ms"] == pytest.approx(50 / 1e6)
    assert o2["per_rank"]["a"]["busy_share"] == pytest.approx(38 / 50)
    assert o2["per_rank"]["b"]["busy_share"] == pytest.approx(30 / 50)
    # two ranks on one device: the device is busy when either computes
    r = _run(a, b)
    for rk in r["ranks"]:
        rk["meta"]["host"], rk["meta"]["device"] = "h", "cuda:0"
    od = overlap(r, [{"t0_mono_ns": 0, "t1_mono_ns": 100}])["per_device"]["h cuda:0"]
    assert od["ranks"] == ["a", "b"] and od["busy_share"] == pytest.approx(0.58)      # 0-10, 12-60
    # a send without its receive is reported
    assert overlap(_run(a + [_ev("send", 41, 42, "b", 7)], b), None)["unmatched_links"] == ["a -> b"]


@pytest.mark.parametrize("async_send", [False, True])
def test_a_recorded_run_with_slots_matches_every_send_with_a_receive(tiny4, tmp_path, async_send, monkeypatch):
    from nocturne.engines.torch_tp import timeline
    monkeypatch.setenv("NOCTURNE_ASYNC_COPY", "thread")
    specs, co, th = boot(tiny4, hybrid_tp_first(free_ports(8)), slots=3, async_send=async_send, timeline=True, run_dir=str(tmp_path))
    try:
        out = co.generate_many(reqs(PROMPTS, 5), prefill_chunk=3, tp_interleave=2)
        requests = list(co.requests)
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    summ = timeline.build(tmp_path, co.clock_record(), requests)
    o = summ["overlap"]
    assert o["unmatched_links"] == [], o["links"]
    assert o["async_send"] is async_send
    assert set(o["per_rank"]) == {s.name for s in specs}
    assert all(0 <= v["overlap_share"] <= v["busy_share"] <= 1 for v in o["per_rank"].values())
    assert sum(l["frames"] for l in o["links"].values()) > 0
    slots = {e.get("slot") for rk in timeline.load(tmp_path)["ranks"] for e in rk["events"] if e["step"] >= 0}
    assert slots == {0, 1, 2}
    page = (tmp_path / "timeline.html").read_text()
    assert '"overlap":{' in page and '"flights":[[' in page


def test_default_copy_mode_per_device_and_the_override(monkeypatch):
    monkeypatch.delenv("NOCTURNE_ASYNC_COPY", raising=False)
    assert hybrid.ASYNC_COPY == {"cuda": "pinned", "mps": "inline", "cpu": "inline"}      # MPS: two threads may not use the device at once
    s = hybrid.Sender("cpu", None)
    assert s.mode == "inline"
    s.stop()
    monkeypatch.setenv("NOCTURNE_ASYNC_COPY", "sideways")
    with pytest.raises(ValueError, match="NOCTURNE_ASYNC_COPY"):
        hybrid.Sender("cpu", None)


def test_a_plain_run_folder_is_a_reference_and_missing_prompts_are_named(tmp_path):
    import numpy as np
    from nocturne.engines.torch_tp import hybrid_run
    d = tmp_path / "X_off_r1"
    d.mkdir()
    prompts = [{"id": "a", "kind": "eval", "tokens": [1, 2, 3], "score": 1.0}, {"id": "L", "kind": "long", "tokens": [5, 6, 7, 8], "score": None}]
    (d / "record.json").write_text(json.dumps({"run_id": "X_off_r1", "outcome": "completed", "prompts": prompts}))
    np.savez(d / "first_logits.npz", a=np.zeros(4, np.float32), L=np.ones(4, np.float32))
    ref, lg = hybrid_run.load_reference(str(d))
    out = [dict(prompts[0]), {"id": "L", "kind": "long", "tokens": [5, 6, 9, 8], "score": None}, {"id": "new", "kind": "long", "tokens": [1], "score": None}]
    c = hybrid_run.compare(out, {"a": np.zeros(4, np.float32), "L": np.full(4, 1.5, np.float32), "new": np.zeros(4, np.float32)}, ref, lg)
    assert (c["identical"], c["of"], c["long_first_div"], c["long_identical"], c["not_in_reference"]) == (1, 1, {"L": 2}, False, ["new"])
    assert c["logit_maxdiff"] == pytest.approx(0.5)
    # pass 2, P3: a comparison that did not happen is not a pass
    no_long = hybrid_run.compare([dict(prompts[0]), {"id": "new", "kind": "long", "tokens": [1], "score": None}],
                                 {"a": np.zeros(4, np.float32)}, ref, lg)
    assert (no_long["identical"], no_long["long_identical"], no_long["long_compared"], no_long["not_in_reference"]) == (1, None, 0, ["new"])
    nothing = hybrid_run.compare([{"id": "z", "kind": "eval", "tokens": [1], "score": 1.0}], {"z": np.zeros(4, np.float32)}, ref, lg)
    assert (nothing["of"], nothing["criterion_met"], nothing["logit_maxdiff"], nothing["long_identical"]) == (0, False, None, None)
    part = hybrid_run.compare([dict(prompts[0]), {"id": "z", "kind": "eval", "tokens": [1], "score": 1.0}],
                              {"a": np.zeros(4, np.float32)}, ref, lg)
    assert part["criterion_met"] is False and part["not_in_reference"] == ["z"]      # an eval prompt the reference lacks
    (d / "record.json").write_text(json.dumps({"run_id": "X_off_r1", "outcome": "error", "prompts": prompts}))
    with pytest.raises(ValueError, match="did not complete"):
        hybrid_run.load_reference(str(d))


# ---- after the first review (REVIEW_overlap.md) ------------------------------------------------------------------------------
GOLDEN = {       # produced by tests/bench/golden_frames.py on the engine of commit 965385f, before part 3
    "pp3": {"frames": 60, "sha256": "2217c89fa75ee8faa26b42909f2010c4949c8475a63a417919f5a9f196c8e8dc"},
    "pp3_timeline": {"frames": 60, "sha256": "2bb58b65ffc96b6d89dad5211dd92e0736a04ec4a96814f9f9c540f6f0a4b75f"},
    "tp3": {"frames": 1160, "sha256": "19e4fbb7529ee0540d5288d34ff67205c9dc29e434bb20824f4172ba425b17d1"},
    "tp3_timeline": {"frames": 1160, "sha256": "e460f1202fa98df54ad5c63d5396a0b64cd359d0db2e49b092464cc4e7cffe5d"},
    "hybrid": {"frames": 320, "sha256": "11124a461014963896d4f189bc5b973c7423babaaacc9416b8e781c099a32081"},
    "hybrid_timeline": {"frames": 320, "sha256": "93166f7711adff37352862de41bfc5239d12df89a621bcf16f8bd2ebf52a497a"},
}


def test_options_off_send_the_frames_of_the_engine_before_part_3(tmp_path):
    """F8. Every data frame of a fixed run, by kind, dtype, shape, length and header fields, and the frames by kind per
    link, against hashes made with the old code. A new bit in a header field of the default path changes the hash."""
    from . import golden_frames
    assert golden_frames.all_records(golden_frames.write_model(tmp_path / "t4")) == GOLDEN


def test_async_send_closes_a_compute_span_before_every_hand_over(tiny4, tmp_path):
    """F1. A TP rank's compute before an all-reduce must not be booked as waiting for the all-reduce."""
    from nocturne.engines.torch_tp import timeline
    specs, co, th = boot(tiny4, tp3(free_ports(8)), async_send=True, timeline=True, run_dir=str(tmp_path))
    try:
        co.generate(list(PROMPTS[0]), 5, ignore_eos=True)
        m = co.metrics()
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    run = timeline.load(tmp_path)
    assert len(run["ranks"]) == 3
    for rk in run["ranks"]:
        calls = m["ranks"][rk["meta"]["rank"]]["allreduce_calls"]
        compute = sum(1 for e in rk["events"] if e["kind"] == "compute")
        assert calls > 0 and compute >= calls, (rk["meta"]["rank"], compute, calls)


@pytest.mark.parametrize("interleave,async_send", [(1, False), (2, False), (2, True)])
def test_frames_in_flight_are_matched_per_socket_in_a_tp_group_with_slots(tiny4, tmp_path, interleave, async_send):
    """F2. A first stage rank with head rows has two sockets to the coordinator. Their frames must not be matched as one sequence."""
    from nocturne.engines.torch_tp import timeline
    specs, co, th = boot(tiny4, tp3(free_ports(8)), slots=3, async_send=async_send, timeline=True, run_dir=str(tmp_path))
    try:
        co.generate_many(reqs(PROMPTS, 6), tp_interleave=interleave)
        requests = list(co.requests)
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    o = timeline.build(tmp_path, co.clock_record(), requests)["overlap"]
    assert o["unmatched_links"] == []
    assert all(l["sent"] == l["received"] == l["frames"] for l in o["links"].values())


def test_in_flight_keeps_two_sockets_of_one_pair_of_ranks_apart():
    from nocturne.engines.torch_tp.timeline import in_flight
    ev = lambda kind, t0, t1, peer, n, ch: dict(_ev(kind, t0, t1, peer, n), ch=ch)
    # b sends a head frame on the control link, then a partial on the group link; a reads the partial first
    b = [ev("send", 10, 11, "a", 16, "c"), ev("send", 20, 21, "a", 500, "g")]
    a = [ev("allreduce_wait", 0, 25, "b", 500, "g"), ev("recv_wait", 25, 40, "b", 16, "c")]
    fl = in_flight(_run(a, b))["b -> a"]
    assert fl["bytes_match"] and (fl["sent"], fl["received"]) == (2, 2)
    assert fl["frames"] == [[10.0, 40.0, 16, 1, 0], [20.0, 25.0, 500, 1, 0]]
    # files from before the channel column: one sequence per pair of ranks, and the mismatch is reported
    for e in a + b:
        e.pop("ch")
    assert in_flight(_run(a, b))["b -> a"]["bytes_match"] is False


def test_a_failure_of_the_sender_thread_ends_the_run_at_once(tiny4, monkeypatch):
    """F3. The head rank's sender raises (no maximum among the logits). The rank is blocked on its upstream link and
    makes no further call to the sender, so the sender must close the rank's links."""
    from nocturne.engines.torch_tp.engine import LinkLost
    from nocturne.engines.torch_tp.model import Qwen2TP
    real = Qwen2TP.head_device

    def no_maximum(self, h):
        pair, lg = real(self, h)
        return torch.stack([pair[0], torch.tensor(float(lg.shape[0]))]), lg
    monkeypatch.setattr(Qwen2TP, "head_device", no_maximum)
    logs = []
    specs, co, th = boot(tiny4, pp3(free_ports(8)), async_send=True)
    co.rank.log = logs.append
    out = {}

    def go():
        try:
            co.generate(list(PROMPTS[0]), 4, ignore_eos=True)
        except Exception as e:      # noqa: BLE001
            out["e"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=20)
    alive = t.is_alive()
    co.shutdown()
    assert not alive, "the coordinator still waits for a head result that will never come"
    assert isinstance(out.get("e"), (LinkLost, hybrid.RankFailed, ConnectionError)), out
    assert "sender thread failed: ValueError" in str(out["e"]), out       # pass 2, P7: the record names the real error


def test_the_scheduler_counts_pairs_singles_and_units_in_flight(tiny4):
    """F4 and F7. Two requests are really in flight at once, and the record says how many steps ran as a pair."""
    _, _, _, _, co = many(tiny4, pp3, 2)
    assert co.schedule["max_in_flight"] == 2 and co.schedule["pair_steps"] == 0 and co.schedule["single_steps"] > 0
    _, _, _, _, co = many(tiny4, pp3, 3)
    assert co.schedule["max_in_flight"] == 3
    _, _, _, _, co = many(tiny4, tp3, 2, interleave=2)
    assert co.schedule["pair_steps"] > 0 and co.schedule["max_in_flight"] == 2
    _, _, _, _, co = many(tiny4, pp3, 1, chunk=3)
    assert co.schedule["max_in_flight"] == 1 and co.schedule["pair_steps"] == 0


def test_a_chunk_leaves_the_first_stage_before_the_head_result_of_the_prompt(tiny4):
    """F7, O2. With one slot the first stage sends every chunk of a prompt on without waiting: several hidden state
    frames leave the coordinator before any head frame comes back."""
    real_send, real_recv = transport.Link.send, transport.Link.recv
    ev, me = [], threading.get_ident()

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
        if kind == transport.KIND_HIDDEN and threading.get_ident() == me:
            ev.append("h")
        return real_send(self, kind, payload, dtype, shape, aux0, aux1, **kw)

    def recv(self, *a, **kw):
        out = real_recv(self, *a, **kw)
        if out[0] == transport.KIND_HEAD:        # only the coordinator receives head frames (on its reader thread)
            ev.append("r")
        return out
    transport.Link.send, transport.Link.recv = send, recv
    try:
        many(tiny4, pp3, 1, chunk=3, prompts=[PROMPTS[2]], n=3)       # 7 tokens: chunks of 3, 4
    finally:
        transport.Link.send, transport.Link.recv = real_send, real_recv
    assert "".join(ev).startswith("hhr"), "".join(ev)


def test_async_send_returns_before_the_frame_has_left():
    """F7, O4. With the sender thread the compute thread does not wait for a slow link."""
    import time
    link = _FakeLink(slow=0.4)
    s = hybrid.Sender("cpu", None)
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu", sender=s)
    t0 = time.perf_counter()
    ticket = comm.send_partial(torch.ones(1, 8))
    dt = time.perf_counter() - t0
    assert dt < 0.2 and link.sent == [], (dt, link.sent)
    s.flush()
    assert len(link.sent) == 1 and ticket[2] is None
    s.stop()


def test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted(monkeypatch):
    """F6. Linux grants far less than the 8 MiB asked for. Every rank sends before it receives, so what is sent on the
    caller's thread must fit the granted buffer; in a pair that is two frames."""
    link = _FakeLink(sndbuf=425984)                 # what gtx reports after the request for 8 MiB
    assert link.fits_inline(200_000) and not link.fits_inline(220_000)
    assert link.fits_inline(100_000, outstanding=100_000) and not link.fits_inline(120_000, outstanding=100_000)
    assert _FakeLink().fits_inline(1 << 20) and not _FakeLink().fits_inline((1 << 20) + 1)     # the Macs: SMALL_FRAME decides, as before
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
    x, y = torch.ones(1, 30_000), torch.ones(1, 30_000)           # 120 kB each as float32
    ta = comm.send_partial(x, 0)
    tb = comm.send_partial(y, 1)
    assert ta[2] == [] and len(tb[2]) == 1, "the second frame of the pair must leave on a thread: both do not fit 212992 bytes"
    for t in tb[2]:
        t.join(timeout=5)
    assert [len(p) for _, p, _, _ in link.sent] == [120_000, 120_000]
    assert comm._out == {1: 240_000}            # both frames wait for their answer, the one on the thread too


def test_a_pair_of_large_frames_cannot_block_both_ranks_with_small_socket_buffers(monkeypatch):
    """F6, the mechanism on real sockets: two ranks, buffers as small as Linux grants, a pair of 600 kB frames each way."""
    import socket as _s
    monkeypatch.setattr(transport, "SOCK_BUF", 212992)
    srv = _s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = _s.create_connection(srv.getsockname())
    d, _ = srv.accept()
    la, lb = transport.Link(c), transport.Link(d)
    assert la.sndbuf <= 2 * 212992 + 4096
    comms = [hybrid.GroupComm(0, 2, {1: la}, "cpu"), hybrid.GroupComm(1, 2, {0: lb}, "cpu")]
    xs = [torch.full((1, 150_000), 1.0), torch.full((1, 150_000), 2.0)]       # 600 kB each
    out = [None, None]

    def rank(i):
        ta = comms[i].send_partial(xs[0] * (i + 1), 0)
        tb = comms[i].send_partial(xs[1] * (i + 1), 1)
        out[i] = (comms[i].recv_sum(ta), comms[i].recv_sum(tb))
    ts = [threading.Thread(target=rank, args=(i,), daemon=True) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in ts), "both ranks are blocked in the send of the second frame"
    assert float(out[0][0][0, 0]) == 3.0 and float(out[1][1][0, 0]) == 6.0
    for l in (la, lb):
        l.close()
    srv.close()


def test_small_socket_buffers_do_not_change_the_tokens(tiny4, monkeypatch):
    monkeypatch.setattr(transport, "SOCK_BUF", 4096)
    _, ref, _ = alone(tiny4, tp3)
    _, out, m, _, _ = many(tiny4, tp3, 2, interleave=2)
    same(ref, out)
    assert all(len(v) == 2 and v[0] > 0 for r in m["ranks"].values() for v in r["socket_buffers"].values())


def test_overlap_options_are_checked_before_anything_starts(tmp_path):
    """F10 and F11."""
    from nocturne.engines.torch_tp import coordinate, hybrid_run
    from nocturne.engines.torch_tp.plan import check_overlap
    check_overlap(1, 0, 1)
    check_overlap(256, 2, 2)
    for bad, text in [((0, 0, 1), "slots"), ((257, 0, 1), "slots"), ((1, 1, 1), "prefill-chunk"), ((1, -4, 1), "prefill-chunk"),
                      ((2, 0, 3), "tp-interleave"), ((1, 0, 2), "needs --slots 2")]:
        with pytest.raises(PlanError, match=text):
            check_overlap(*bad)
    with pytest.raises(PlanError):
        coordinate.overlap_options({"prefill_chunk": 1})
    assert coordinate.overlap_options({}) == {"slots": 1, "prefill_chunk": 0, "tp_interleave": 1, "async_send": False, "on": False}
    for argv in (["--slots", "300"], ["--prefill-chunk", "1"], ["--tp-interleave", "2"]):
        with pytest.raises(SystemExit):
            hybrid_run.main(["--placement", "none.yaml", "--run-id", "x", "--session-dir", str(tmp_path)] + argv)
    with pytest.raises(SystemExit):         # pass 2, P3: a reference that cannot be read stops the run before it starts
        hybrid_run.main(["--placement", "none.yaml", "--run-id", "x", "--session-dir", str(tmp_path), "--reference", str(tmp_path / "no_such_run")])
    assert not list(tmp_path.iterdir())
    # pass 3, Q4: a reference with an empty logits file, and one without prompts, stop the run with a message too
    import numpy as np
    for name, rec, npz in (("empty_logits", {"run_id": "r", "outcome": "completed", "prompts": [{"id": "a", "kind": "eval", "tokens": [1]}]}, b""),
                           ("no_prompts", {"run_id": "r", "outcome": "completed", "prompts": []}, None)):
        d = tmp_path / "refs" / name
        d.mkdir(parents=True)
        (d / "record.json").write_text(json.dumps(rec))
        if npz is None:
            np.savez(d / "first_logits.npz", a=np.zeros(2, np.float32))
        else:
            (d / "first_logits.npz").write_bytes(npz)
        with pytest.raises(SystemExit):
            hybrid_run.main(["--placement", "none.yaml", "--run-id", "x", "--session-dir", str(tmp_path / "s"), "--reference", str(d)])
    assert not (tmp_path / "s").exists()


def test_a_host_with_several_rank_processes_and_no_host_budget_is_marked_unchecked(tiny4):
    """F5. Old placements keep working, and the plan says in words that nobody compared the sum with the machine."""
    from nocturne.engines.torch_tp.plan import format_plan
    d = Dims.of(load_config(tiny4))
    p = interleaved([1, 2, 3, 4])
    specs = plan(d, p)
    ht = host_totals(specs, p.host_budget_gb)
    assert all(h["unchecked"] for h in ht.values())
    assert "the sum is not checked" in format_plan(specs, d, p.host_budget_gb)
    one = host_totals(plan(d, tp3([1, 2, 3])))
    assert one["h"]["unchecked"] is True and one["h"]["processes"] == 3


def test_a_pair_plans_the_work_space_of_two_requests_for_group_ranks(tiny4):
    """F15."""
    d = Dims.of(load_config(tiny4))
    a, b = plan(d, hybrid_tp_first([1, 2, 3, 4, 5])), plan(d, hybrid_tp_first([1, 2, 3, 4, 5]), pair=True)
    for x, y in zip(a, b):
        assert y.planned["workspace"] == x.planned["workspace"] * (2 if x.group_size > 1 else 1), x.name


def test_the_old_path_masks_the_request_number_so_it_never_reaches_the_slot_bits(tiny4, tmp_path):
    """F9."""
    specs, co, th = boot(tiny4, pp3(free_ports(8)), timeline=True, run_dir=str(tmp_path))
    try:
        co.req_seq = 65535
        ref = co.generate(list(PROMPTS[0]), 3, ignore_eos=True)
        assert len(ref["tokens"]) == 3
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pinned copy needs a CUDA device")
@pytest.mark.parametrize("make,slots,interleave", [(pp3, 3, 1), (tp3, 2, 2), (hybrid_tp_first, 3, 2)])
def test_pinned_copy_on_cuda_gives_the_tokens_and_logits_of_the_plain_path(tiny4, make, slots, interleave):
    """F13. The CUDA default of --async-send: non blocking copies into pooled pinned buffers, events, the sender thread."""
    specs, co, th = boot(tiny4, make(free_ports(8)), device="cuda")
    try:
        ref = [co.generate(list(p), 8, ignore_eos=True, first_logits=True, rid=f"p{i}") for i, p in enumerate(PROMPTS)]
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    specs, co, th = boot(tiny4, make(free_ports(8)), slots=slots, device="cuda", async_send=True)
    try:
        out = co.generate_many(reqs(PROMPTS, 8), tp_interleave=interleave)
        m = co.metrics()
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    same(ref, out)
    assert all(r["async_send"]["copy"] == "pinned" for r in m["ranks"].values())


# ---- after the second review (REVIEW_overlap_pass2.md) -------------------------------------------------------------------------
def _socket_pair(monkeypatch, buf=212992):
    import socket as _s
    monkeypatch.setattr(transport, "SOCK_BUF", buf)
    srv = _s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = _s.create_connection(srv.getsockname())
    d, _ = srv.accept()
    srv.close()
    return transport.Link(c), transport.Link(d)


def _capacity(monkeypatch) -> int:
    """Bytes the two kernels take on one tuned link while nobody reads (it differs per kernel, so it is measured)."""
    a, b = _socket_pair(monkeypatch)
    import time
    a.sock.setblocking(False)
    n, chunk, quiet = 0, b"x" * 8192, None
    while quiet is None or time.monotonic() - quiet < 0.4:      # the receive side takes more a moment after the first refusal
        try:
            n += a.sock.send(chunk)
            quiet = None
        except BlockingIOError:
            quiet = quiet or time.monotonic()
            time.sleep(0.01)
    a.close()
    b.close()
    return n


def test_a_frame_on_a_helper_thread_counts_for_its_pair_partner(monkeypatch):
    """Pass 2, P1. A is too large for the inline rule and leaves on a thread; it fits the kernels, so the thread ends and A
    sits unread. B is small. Sent inline on top of A it overflows the buffers, and every rank of the group blocks in that
    send. B must count A and leave on a thread as well."""
    import time
    cap = _capacity(monkeypatch)
    la, lb = _socket_pair(monkeypatch)
    a_bytes, b_bytes = (cap - 60_000) // 4 * 4, 100_000
    assert a_bytes > la.sndbuf // 2 and b_bytes <= la.sndbuf // 2 and a_bytes + b_bytes > cap, (cap, la.sndbuf)
    comms = [hybrid.GroupComm(0, 2, {1: la}, "cpu"), hybrid.GroupComm(1, 2, {0: lb}, "cpu")]
    out = [None, None]

    def rank(i):
        ta = comms[i].send_partial(torch.full((1, a_bytes // 4), float(i + 1)), 0)
        time.sleep(0.15)                    # the compute of B: time for A's thread to end
        tb = comms[i].send_partial(torch.full((1, b_bytes // 4), float(i + 1)), 1)
        out[i] = (comms[i].recv_sum(ta), comms[i].recv_sum(tb))
    ts = [threading.Thread(target=rank, args=(i,), daemon=True) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in ts), "both ranks are blocked in the inline send of the small frame"
    assert float(out[0][0][0, 0]) == 3.0 and float(out[1][1][0, 0]) == 3.0
    assert comms[0]._out == {1: 0} and comms[1]._out == {0: 0}
    for l in (la, lb):
        l.close()


def test_the_count_of_unanswered_bytes_returns_to_zero_on_every_link_of_a_group_of_three(tiny4, monkeypatch):
    """Pass 2, P2. With small buffers some frames of a pair go inline and some on threads, per link. Every link's count
    must come back to zero; a count that drifts below zero switches the inline rule off."""
    monkeypatch.setattr(transport, "SOCK_BUF", 4096)
    _, ref, _ = alone(tiny4, tp3)
    _, out, _, _, co = many(tiny4, tp3, 3, interleave=2)
    same(ref, out)
    assert co.rank.comm._out == {1: 0, 2: 0}


@pytest.mark.parametrize("make", [tp3, hybrid_tp_first])
def test_the_old_path_records_channels_that_match_on_every_link(tiny4, tmp_path, make):
    """Pass 2, P6. Every options-off run with a timeline takes Coordinator._step: its token sends and head waits carry the
    control channel, so every send finds its receive."""
    from nocturne.engines.torch_tp import timeline
    specs, co, th = boot(tiny4, make(free_ports(8)), timeline=True, run_dir=str(tmp_path))
    try:
        for i, p in enumerate(PROMPTS[:3]):
            co.generate(list(p), 4, ignore_eos=True, rid=f"p{i}")
        requests = list(co.requests)
    finally:
        co.shutdown()
    for x in th:
        x.join(timeout=10)
    o = timeline.build(tmp_path, co.clock_record(), requests)["overlap"]
    assert o["unmatched_links"] == [] and sum(l["frames"] for l in o["links"].values()) > 0
    chans = {(e["kind"], e.get("ch")) for rk in timeline.load(tmp_path)["ranks"] for e in rk["events"] if e["bytes"]}
    assert all(ch in ("g", "p", "c") for _, ch in chans), chans


def test_a_head_frame_for_a_slot_outside_the_table_is_named(tiny4):
    """Pass 2, P6 (F12 of pass 1): the message names the rank and the slot, not an index error from inside the receive."""
    specs, co, th = boot(tiny4, pp3(free_ports(8)), slots=2)
    real = hybrid.RankRuntime._send_head

    def wrong_slot(self, res, flags, unit):
        return real(self, res, flags | 7 << hybrid.SLOT_SHIFT, unit)
    hybrid.RankRuntime._send_head = wrong_slot
    try:
        with pytest.raises(RuntimeError, match="head frame from s2/0 for slot 7"):
            co.generate_many(reqs(PROMPTS[:1], 3))
    finally:
        hybrid.RankRuntime._send_head = real
        co.shutdown()


def test_the_plan_command_shows_the_host_budget_it_checked(tmp_path, tiny4, capsys):
    """Pass 2, P4."""
    import yaml
    from nocturne.engines.torch_tp import __main__ as cli
    pl = {"model": str(tiny4), "max_seq_len": 64, "granule": 16, "host_budget_gb": {"h": 1.0}, "stages": [
        {"name": "a0", "layers": [0, 2], "embed": True, "ranks": [{"host": "h", "device": "cpu", "port": 1}]},
        {"name": "b0", "layers": [2, 4], "lm_head": True, "ranks": [{"host": "h", "device": "cpu", "port": 2}]}]}
    (tmp_path / "pl.yaml").write_text(yaml.safe_dump(pl))
    assert cli.main(["plan", "--placement", str(tmp_path / "pl.yaml")]) == 0
    text = capsys.readouterr().out
    assert "host budget MB" in text and "not checked" not in text and "| 1000 |" in text


# ---- after the third review (REVIEW_overlap_pass3.md) --------------------------------------------------------------------------
def test_the_small_partner_of_a_threaded_frame_that_has_ended_does_not_go_inline():
    """Pass 3, Q1: the decision of P1 without a kernel. A leaves on a thread and the thread ends, so A sits unread in the
    buffers. B is small. It must count A and leave on a thread too. (The socket test above depends on what a kernel takes.)"""
    link = _FakeLink(sndbuf=425984)
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
    ta = comm.send_partial(torch.ones(1, 75_000), 0)        # 300 kB: over half the granted buffer, a thread
    assert len(ta[2]) == 1
    ta[2][0].join(timeout=5)                                # the thread has ended
    tb = comm.send_partial(torch.ones(1, 25_000), 1)        # 100 kB: alone it would fit
    assert len(tb[2]) == 1, "the small partner of a threaded frame went inline on top of it"
    tb[2][0].join(timeout=5)
    assert [len(p) for _, p, _, _ in link.sent] == [300_000, 100_000] and comm._out == {1: 400_000}


def test_in_a_group_of_three_every_link_keeps_its_own_count():
    """Pass 3, Q1 (P2 of pass 2): link 1 is fast, link 2 is slow, so the second frame of a pair goes inline to peer 1 and
    behind the first frame to peer 2. After both all-reduces are received both counts are zero; the old code subtracted
    the inline amount from link 2 as well and left it negative."""
    fast, slow = _FakeLink(sndbuf=425984), _FakeLink(slow=0.4, sndbuf=425984)
    comm = hybrid.GroupComm(0, 3, {1: fast, 2: slow}, "cpu")
    a, b = torch.ones(1, 75_000), torch.ones(1, 10_000)     # 300 kB (threads), 40 kB
    ta = comm.send_partial(a, 0)
    for t in ta[2]:
        if not slow._started or t is not comm._tail.get(2):
            t.join(timeout=2)                               # the thread to the fast peer ends; the one to the slow peer is alive
    tb = comm.send_partial(b, 1)
    assert comm._out == {1: 340_000, 2: 340_000}
    for l in (fast, slow):
        l.answers = [(1, 75_000), (1, 10_000)]
    comm.recv_sum(ta)
    comm.recv_sum(tb)
    assert comm._out == {1: 0, 2: 0}
    assert [len(p) for _, p, _, _ in slow.sent] == [300_000, 40_000] and [len(p) for _, p, _, _ in fast.sent] == [300_000, 40_000]


# ---- after the Codex review (REVIEW_CODEX_overlap.md) ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def big_vocab(tmp_path_factory):
    """Two layers, hidden size 8, a vocabulary of 524288: with float32 logits a head rank of two sends 1 MiB per first step."""
    return write_tiny(tmp_path_factory.mktemp("s10o_big") / "big", layers=2, hidden_size=8, intermediate_size=32, num_attention_heads=2,
                      num_key_value_heads=2, vocab_size=1 << 19)


def tp2(p):
    return Placement("m", (StageIn("s0", (0, 2), (R(p[0]), R(p[1])), embed=True, lm_head=True),), max_seq_len=64, granule=16)


@pytest.mark.parametrize("async_send", [False, True])
def test_a_large_head_frame_does_not_block_the_work_of_the_next_slot(big_vocab, monkeypatch, async_send):
    """Codex 1 (high). The peer's head frame for slot A (first step logits, larger than the socket buffers) meets the
    coordinator's all-reduce for slot B. The peer cannot finish the head frame unless the coordinator reads it, and the
    coordinator waits for the peer's partial, which comes after the head frame. The head frame is held until the
    coordinator has started slot B, so the order does not depend on timing."""
    prompts = [[5, 17, 99], [8, 61, 7]]
    monkeypatch.setattr(transport, "SOCK_BUF", 4096)
    _, ref, _ = alone(big_vocab, tp2, prompts, 2)
    gate, seen = threading.Event(), {"token": 0, "head": 0}
    real = transport.Link.send

    def send(self, kind, *a, **kw):
        if kind == transport.KIND_TOKEN:
            seen["token"] += 1
            if seen["token"] == 2:
                gate.set()
        if kind == transport.KIND_HEAD:
            seen["head"] += 1
            if seen["head"] == 1:
                gate.wait(10)
        return real(self, kind, *a, **kw)
    monkeypatch.setattr(transport.Link, "send", send)
    specs, co, th = boot(big_vocab, tp2(free_ports(8)), slots=2, async_send=async_send)
    assert specs[1].head_rows[1] - specs[1].head_rows[0] == 1 << 18         # a head frame of 1 MiB from the peer
    out = []
    w = threading.Thread(target=lambda: out.append(co.generate_many(reqs(prompts, 2))), daemon=True)
    w.start()
    w.join(30)
    stuck = w.is_alive()
    if stuck:                               # free the threads of this test
        for link in list(co.rank.links.values()):
            link.close()
    else:
        co.shutdown()
    assert not stuck, "the coordinator and its peer wait for each other"
    same(ref, out[0])


def test_head_frames_are_read_while_the_scheduler_is_busy_and_the_reader_ends_with_the_call(tiny4):
    """Codex 1. The reader thread lives only inside generate_many, and the control links are free for metrics afterwards."""
    specs, co, th = boot(tiny4, pp3(free_ports(8)), slots=2)
    names = lambda: {t.name for t in threading.enumerate()}
    try:
        seen = []
        r = reqs(PROMPTS[:3], 4)
        r[0]["on_token"] = lambda tok: seen.append("nocturne-head-reader" in names())
        out = co.generate_many(r)
        assert seen and all(seen) and "nocturne-head-reader" not in names()
        m = co.metrics()                    # the control links answer: no reader is left on them
        assert m["ranks"]["s2/0"]["steps"] > 0
        out2 = co.generate_many(reqs(PROMPTS[:3], 4))       # a second call starts its own reader
        same(out, out2)
    finally:
        co.shutdown()


def test_a_lost_control_link_ends_a_run_with_slots(tiny4):
    """Codex 1. The reader thread hands a failure on its link to the scheduler, which raises it."""
    from nocturne.engines.torch_tp.engine import LinkLost
    specs, co, th = boot(tiny4, pp3(free_ports(8)), slots=2)
    r = reqs(PROMPTS[:2], 40)
    r[0]["on_token"] = lambda tok: co.ctl[2].sock.shutdown(2)       # the head rank's control link breaks at the first token
    try:
        with pytest.raises(LinkLost):
            co.generate_many(r)
    finally:
        for link in list(co.rank.links.values()):
            link.close()
        co.listener.close()


PLAN_KEYS_BEFORE = ["specs", "hosts", "max_seq_len", "reduce_dtype", "enforce_budget", "mps_fraction_cap", "dtype", "timeline", "run_tag",
                    "ship", "cfg", "verify_cache", "gid"]       # the plan frame of commit 965385f, in its order


def _plans(model, monkeypatch, **kw):
    real, got = transport.Link.send, []

    def send(self, kind, payload=b"", *a, **k):
        if kind == transport.KIND_PLAN:
            got.append(json.loads(bytes(payload).decode()))
        return real(self, kind, payload, *a, **k)
    monkeypatch.setattr(transport.Link, "send", send)
    specs, co, th = boot(model, pp3(free_ports(8)), **kw)
    co.shutdown()
    monkeypatch.setattr(transport.Link, "send", real)
    return got


def test_the_plan_frame_names_an_overlap_option_only_when_it_is_set(tiny4, monkeypatch):
    """Codex 2. With the options off the plan frame has the fields of before part 3, in their order: no added bytes."""
    off = _plans(tiny4, monkeypatch)
    assert len(off) == 2 and all(list(p) == PLAN_KEYS_BEFORE for p in off)
    on = _plans(tiny4, monkeypatch, slots=2, async_send=True)
    assert all(p["slots"] == 2 and p["async_send"] is True for p in on)
    only_slots = _plans(tiny4, monkeypatch, slots=3)
    assert all(p["slots"] == 3 and "async_send" not in p for p in only_slots)


def test_a_request_number_past_the_bits_of_the_frame_stays_apart_in_the_record(tiny4, tmp_path, monkeypatch):
    """Codex 3. The frame carries the request number modulo 65536. A rank rebuilds the full number from the last one it
    saw, so the requests past 65535 of one run stay apart on every rank. The engine part runs with 3 bits (modulo 8)."""
    from nocturne.engines.torch_tp import timeline
    from nocturne.engines.torch_tp.recorder import Recorder
    r = Recorder("x", 1, "h", "cpu")
    full = [1, 2, 1, 3] + list(range(4, 16)) + [16, 15, 17, 18, 16, 19]          # two requests in flight take turns across the wrap
    assert [r.full_req(v % 16, 16) for v in full] == full
    r = Recorder("x", 1, "h", "cpu")
    seq = list(range(1, 200000, 7))                                              # a long run, far past 65536
    assert [r.full_req(v % 65536, 1 << 16) for v in seq] == seq
    monkeypatch.setattr(hybrid, "REQ_MASK", 7)
    specs, co, th = boot(tiny4, pp3(free_ports(8)), timeline=True, run_dir=str(tmp_path))
    try:
        for k in range(12):
            co.generate(list(PROMPTS[k % 5]), 2, ignore_eos=True)
    finally:
        co.shutdown()
    for t in th:
        t.join(timeout=10)
    run = timeline.load(tmp_path)
    assert len(run["ranks"]) == 3
    for rk in run["ranks"]:
        assert {e["req"] for e in rk["events"] if e["step"] >= 0} == set(range(1, 13)), rk["meta"]["rank"]


class _FailingLink(_FakeLink):
    """The first frame cannot be sent; every later one can."""

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, decode=None):
        with self.lock:
            first = not getattr(self, "_started", False)
            self._started = True
        if first:
            self._sleep(self.slow)
            raise OSError("no route")
        self.sent.append((kind, bytes(payload), aux0, aux1))


def test_a_frame_behind_one_that_failed_is_not_sent_and_the_all_reduce_raises(monkeypatch):
    """Codex 4. Two frames of a pair leave on threads. The first fails. The second must not take its place on the link
    (the peer tells all-reduce frames apart by order), the link is closed, and the all-reduce raises the failure."""
    monkeypatch.setattr(transport, "SMALL_FRAME", 8)
    link = _FailingLink(slow=0.2)           # the second frame is queued while the first is still on its way to failing
    link.answers = [(4, 8), (1, 8)]
    comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
    ta = comm.send_partial(torch.full((4, 8), 1.0), 0)
    tb = comm.send_partial(torch.full((1, 8), 2.0), 1)
    for t in ta[2] + tb[2]:
        t.join(timeout=5)
    assert link.sent == [] and link.closed is True
    assert isinstance(ta[2][0].error, OSError) and isinstance(tb[2][0].error, ConnectionError)
    with pytest.raises(ConnectionError, match="an all-reduce frame could not be sent: OSError: no route"):
        comm.recv_sum(ta)
    with pytest.raises(ConnectionError, match="could not be sent"):
        comm.recv_sum(tb)
    # a later all-reduce on that link does not start: its frame would be read as the one that never left
    with pytest.raises(ConnectionError, match="could not be sent"):
        comm.send_partial(torch.full((1, 8), 3.0))
    assert link.sent == []


def test_a_run_that_differs_in_scheduling_only_must_give_every_token(tmp_path):
    """Codex 5. Against a run of the same placement, dtype and prefill chunk, one changed token is a failure; the rule
    of seven of eight stays for a reference with other numerics."""
    import numpy as np
    from nocturne.engines.torch_tp import hybrid_run
    ev = [{"id": f"e{k}", "kind": "eval", "tokens": [k, k + 1, k + 2], "score": 1.0} for k in range(8)]
    lng = {"id": "L", "kind": "long", "tokens": list(range(40)), "score": None}
    ref = {"run_id": "ref", "prompts": {p["id"]: p for p in ev + [lng]}}
    one_off = [dict(p) for p in ev] + [dict(lng)]
    one_off[3] = dict(one_off[3], tokens=[3, 4, 99])
    assert hybrid_run.compare(one_off, {}, ref, {})["criterion_met"] is True              # seven of eight, same scores
    c = hybrid_run.compare(one_off, {}, ref, {}, exact=True)
    assert (c["identical"], c["of"], c["criterion_met"], c["rule"]) == (7, 8, False, "every token")
    long_off = [dict(p) for p in ev] + [dict(lng, tokens=list(range(39)) + [7])]
    assert hybrid_run.compare(long_off, {}, ref, {})["criterion_met"] is True
    assert hybrid_run.compare(long_off, {}, ref, {}, exact=True)["criterion_met"] is False
    assert hybrid_run.compare([dict(p) for p in ev] + [dict(lng)], {}, ref, {}, exact=True)["criterion_met"] is True
    # what makes a comparison exact: the reference is a run folder with the same placement file and hash, dtype and prefill chunk
    d = tmp_path / "X_off_r1"
    d.mkdir()
    rec = {"run_id": "X_off_r1", "outcome": "completed", "prompts": ev, "placement": {"file": "p.yaml", "sha256": "abc"},
           "args": {"dtype": None, "prefill_chunk": 0, "slots": 1}}
    (d / "record.json").write_text(json.dumps(rec))
    np.savez(d / "first_logits.npz", e0=np.zeros(4, np.float32))
    got, _ = hybrid_run.load_reference(str(d))
    mine = lambda **a: hybrid_run.numerics_of({"placement": {"file": "p.yaml", "sha256": "abc"}, "args": dict({"dtype": None, "prefill_chunk": 0}, **a)})
    assert got["numerics"] == mine(slots=3, tp_interleave=2, async_send=True)             # scheduling options do not count
    assert got["numerics"] != mine(prefill_chunk=16) and got["numerics"] != mine(dtype="float32")
    assert got["numerics"] != hybrid_run.numerics_of({"placement": {"file": "p.yaml", "sha256": "other"}, "args": {}})
    assert hybrid_run.numerics_of({"run_id": "old"}) is None
    me = {"placement": {"file": "p.yaml", "sha256": "abc"}, "args": {"dtype": None, "prefill_chunk": 0, "slots": 4}}
    assert hybrid_run.scheduling_only(me, got) is True
    assert hybrid_run.scheduling_only(dict(me, args={"dtype": None, "prefill_chunk": 16}), got) is False
    assert hybrid_run.scheduling_only(me, {"run_id": "R0", "prompts": {}}) is False          # a reference.json of another placement
    assert hybrid_run.scheduling_only({"run_id": "old"}, {"numerics": None}) is False


def test_a_prompt_of_one_token_is_a_prefill_step(tiny4):
    """Codex 6. The step counts follow the scheduled unit, as on the old path, not the number of tokens in it."""
    specs, co, th = boot(tiny4, tp3(free_ports(8)), slots=2)
    try:
        out = co.generate_many([{"ids": [5], "max_tokens": 3, "ignore_eos": True, "rid": "one", "first_logits": True}])
        assert (co.prefill_steps, co.decode_steps) == (1, 2) and len(out[0]["tokens"]) == 3
    finally:
        co.shutdown()
    specs, co, th = boot(tiny4, tp3(free_ports(8)))
    try:
        ref = co.generate([5], 3, ignore_eos=True, first_logits=True)
        assert (co.prefill_steps, co.decode_steps) == (1, 2) and ref["tokens"] == out[0]["tokens"]
    finally:
        co.shutdown()


def test_a_kv_tensor_of_many_slots_opens_the_mps_heap(tiny4):
    """Codex 7. One K tensor holds every slot. At 256 slots it is 16 MiB here, over the 10 MiB at which MPS opens its
    1 GiB heap, although every weight matrix is small: the host total must not pass a budget below that heap."""
    from nocturne.engines.torch_tp.plan import MPS_HEAP
    cfg = Dims.of(load_config(tiny4))
    pl = Placement("m", (StageIn("s0", (0, 4), (RankIn("h", "mps", 1),), embed=True, lm_head=True),), max_seq_len=1024, prefill_max=16, granule=16)
    assert plan(cfg, pl)[0].planned["mps_floor"] == 0                   # one slot: a K tensor of 64 KiB
    assert plan(cfg, pl, kv_slots=128)[0].planned["mps_floor"] == 0     # 8 MiB
    sp = plan(cfg, pl, kv_slots=256)[0]
    assert sp.planned["kv"] // (2 * 4) == 16 << 20 and sp.planned["mps_floor"] == MPS_HEAP == sp.planned["footprint"]
    with pytest.raises(PlanError, match="host budget"):
        plan(cfg, Placement(**{**pl.__dict__, "host_budget_gb": (("h", 0.65),)}), kv_slots=256)
    cpu = Placement(**{**pl.__dict__, "stages": (StageIn("s0", (0, 4), (RankIn("h", "cpu", 1),), embed=True, lm_head=True),)})
    assert plan(cfg, cpu, kv_slots=256)[0].planned["mps_floor"] == 0


def test_a_run_with_an_unmatched_link_has_a_busy_share_and_no_overlap_share():
    """Codex 8. One receive is missing, so every later frame of that socket would be matched to the wrong receive."""
    from nocturne.engines.torch_tp.timeline import overlap
    a = [_ev("send", 2, 2.5, "b", 100), _ev("send", 3, 3.5, "b", 100), _ev("compute", 4, 7)]
    b = [_ev("recv_wait", 7.5, 8, "a", 100)]
    o = overlap(_run(a, b), [{"t0_mono_ns": 0, "t1_mono_ns": 10}])
    assert o["unmatched_links"] == ["a -> b"]
    for v in list(o["per_rank"].values()) + list(o["per_device"].values()):
        assert v["overlap_share"] is None and v["overlap_out_share"] is None and v["overlap_in_share"] is None and v["in_flight_share"] is None
    assert o["per_rank"]["a"]["busy_share"] == pytest.approx(0.3) and o["per_rank"]["b"]["busy_share"] == 0
    full = overlap(_run(a, b + [_ev("recv_wait", 8, 9, "a", 100)]), [{"t0_mono_ns": 0, "t1_mono_ns": 10}])
    assert full["unmatched_links"] == [] and full["per_rank"]["a"]["overlap_share"] == pytest.approx(0.3)


def test_the_pool_of_pinned_buffers_stays_within_its_limit(monkeypatch):
    """Codex 9. The limit counts the buffer that returns: one large buffer is not kept past it."""
    monkeypatch.setattr(hybrid, "POOL_MAX_BYTES", 96)
    s = hybrid.Sender("cpu", None)
    try:
        for n in (16, 16, 16):              # 64 bytes each
            h = hybrid.Handoff(None)
            h.host, h.pooled = torch.zeros(n, dtype=torch.float32), True
            s.free(h)
        s.flush()
        assert s.pool_bytes == 64 and sum(len(v) for v in s.pool.values()) == 1
        h = hybrid.Handoff(None)
        h.host, h.pooled = torch.zeros(8, dtype=torch.float32), True         # 32 bytes: fits beside the 64
        s.free(h)
        s.flush()
        assert s.pool_bytes == 96
    finally:
        s.stop()


def test_the_coordinator_empties_its_sender_before_a_clock_or_shutdown_frame(tiny4):
    """Codex 10. Design section 11.5: no control frame overtakes a data frame. The coordinator's own sender is flushed
    before CLOCK and before SHUTDOWN, and a sender that is stuck does not hold the shutdown."""
    specs, co, th = boot(tiny4, pp3(free_ports(8)), slots=2, async_send=True)
    ev = []
    real_flush = co.rank.flush
    co.rank.flush = lambda *a: (ev.append("flush"), real_flush(*a))[1]
    real = transport.Link.send

    def send(self, kind, *a, **kw):
        if threading.current_thread() is threading.main_thread() and kind in (transport.KIND_CLOCK, transport.KIND_SHUTDOWN):
            ev.append(transport.KIND_NAMES[kind])
        return real(self, kind, *a, **kw)
    transport.Link.send = send
    try:
        co.generate_many(reqs(PROMPTS[:2], 3))
        del ev[:]
        co.clock_sync()
        assert ev[0] == "flush" and "clock" in ev
        del ev[:]
        co.shutdown()                       # the end clock is measured already: the next control frame is SHUTDOWN
        assert ev[0] == "flush" and ev.index("flush") < ev.index("shutdown")
    finally:
        transport.Link.send = real
    # a sender that never finishes: flush gives up after its timeout
    hold = threading.Event()

    class Stuck(_FakeLink):
        def send(self, *a, **kw):
            hold.wait(10)
    s = hybrid.Sender("cpu", None)
    h = s.copy(torch.zeros(1, 4))
    s.send(h, Stuck(), transport.KIND_HIDDEN, 0, 0, True, "x")
    with pytest.raises(TimeoutError, match="still holds frames"):
        s.flush(0.5)
    hold.set()
    s.stop()
