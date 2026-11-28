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


def write_tiny(d, layers=4, seed=3):
    """The tiny Qwen2 of test_torch_tp with a chosen number of layers (interleaved stages need four)."""
    g = torch.Generator().manual_seed(seed)
    c = dict(TINY, num_hidden_layers=layers)
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


def boot(model, placement, slots=1, **kw):
    cfg = load_config(model)
    specs = plan(Dims.of(cfg), placement, kv_slots=slots)
    threads = []
    for s in specs[1:]:
        t = threading.Thread(target=hybrid.rank_main, args=(["127.0.0.1"], s.port, str(model), "cpu"), kwargs={"log": lambda *a: None}, daemon=True)
        t.start()
        threads.append(t)
    co = hybrid.Coordinator(specs, str(model), {"h": "127.0.0.1", "g": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                            placement.prefill_max, device="cpu", log=lambda *a: None, dtype="float32", slots=slots, **kw)
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
        with pytest.raises(ValueError, match="tp_interleave must be 1 or 2"):
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
