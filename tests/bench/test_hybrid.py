"""Session 6 hybrid placement: planner (pure), uneven slices, vocab parallel embedding and head, N rank all-reduce, and the
stage pipeline end to end over loopback with a tiny random Qwen2 (fp32, CPU)."""
import random
import socket
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from nocturne.engines.torch_tp import transport  # noqa: E402
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.loader import BudgetExceeded, LoadSpec, load_slices  # noqa: E402
from nocturne.engines.torch_tp.model import Comm, Qwen2TP, pick_global  # noqa: E402
from nocturne.engines.torch_tp.plan import (Dims, Placement, PlanError, RankIn, StageIn, frames_per_step,  # noqa: E402
                                            largest_remainder, plan, placement_from_dict)
from nocturne.engines.torch_tp.shard import SliceSpec, kv_range  # noqa: E402

from .test_torch_tp import TINY, write_tiny_model  # noqa: E402

Q7B = Dims(hidden=3584, intermediate=18944, n_layers=28, n_heads=28, n_kv_heads=4, vocab=152064, tied=False)
Q15B = Dims(hidden=1536, intermediate=8960, n_layers=28, n_heads=12, n_kv_heads=2, vocab=151936, tied=True)
Q05B = Dims(hidden=896, intermediate=4864, n_layers=24, n_heads=14, n_kv_heads=2, vocab=151936, tied=True)


_PORT = iter(range(40000, 60000))


def st(name, layers, ranks, **kw):
    return StageIn(name, layers, tuple(RankIn(**{"port": next(_PORT), **r}) for r in ranks), **kw)


def P(*stages, **kw):
    return Placement("m", tuple(stages), **kw)


# ---- planner -------------------------------------------------------------------------------------------------------------
def check_coverage(d: Dims, specs):
    by_stage = {}
    for s in specs:
        by_stage.setdefault(s.stage, []).append(s)
    for ranks in by_stage.values():
        if ranks[0].n_layers:
            heads = [h for s in ranks for h in range(s.q0, s.q1)]
            rows = [m for s in ranks for m in range(s.m0, s.m1)]
            assert sorted(heads) == list(range(d.n_heads)), "every head once"
            assert sorted(rows) == list(range(d.intermediate)), "every MLP row once"
            kv = {k for s in ranks for k in range(s.kv0, s.kv1)}
            assert kv == set(range(d.n_kv_heads)), "every KV head held"
            for s in ranks:
                assert (s.kv0, s.kv1) == kv_range(s.q0, s.q1, d.n_heads, d.n_kv_heads)
        for attr in ("embed_rows", "head_rows"):
            got = [r for s in ranks if getattr(s, attr) for r in range(*getattr(s, attr))]
            if got:
                assert sorted(got) == list(range(d.vocab)), f"every vocab row once in {attr}"


@pytest.mark.parametrize("d", [Q7B, Q15B, Q05B])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 16])
def test_plan_covers_every_head_row_and_vocab_row_once(d, n):
    rnd = random.Random(n)
    ranks = [dict(host="h", weight=rnd.choice([1, 2, 3])) for _ in range(n)]
    specs = plan(d, P(st("s0", (0, 3), ranks, embed=True), st("s1", (3, d.n_layers), [dict(host="h")]),
                      st("head", None, ranks, lm_head=True)))
    check_coverage(d, specs)
    for s in specs:
        if s.stage == 0:
            assert s.m0 % 128 == 0 and (s.m1 % 128 == 0 or s.m1 == d.intermediate)


def test_plan_matches_preregistered_numbers():
    p1 = plan(Q7B, P(st("s0", (0, 28), [dict(host="m5", weight=3), dict(host="air", weight=2)], embed=True, lm_head=True)))
    assert [(s.q1 - s.q0, s.m1 - s.m0) for s in p1] == [(17, 11392), (11, 7552)]
    assert [s.kv1 - s.kv0 for s in p1] == [3, 2] and p1[0].planned["kv_heads_replicated"] == 1
    assert [round(s.planned["layer_bytes"] / 1e6, 1) for s in p1] == [281.7, 186.3]
    p2 = plan(Q7B, P(st("s0", (0, 28), [dict(host="a", heads=14), dict(host="a", heads=7), dict(host="b", heads=7)],
                        embed=True, lm_head=True)))
    assert [s.m1 - s.m0 for s in p2] == [5888, 6528, 6528] and p2[0].planned["kv_heads_replicated"] == 0
    p4 = plan(Q7B, P(st("s0", (0, 28), [dict(host="a")] * 5, embed=True, lm_head=True)))
    assert [s.q1 - s.q0 for s in p4] == [6, 6, 6, 5, 5] and p4[0].planned["kv_heads_replicated"] == 4
    p3 = plan(Q15B, P(st("s0", (0, 28), [dict(host="a")] * 3, embed=True, lm_head=True)))
    assert [s.kv1 - s.kv0 for s in p3] == [1, 2, 1]


def a6_placement(budget=0.40):
    b = {"budget_gb": budget} if budget else {}
    return P(st("s0", (0, 12), [dict(host="m5")], embed=True),
             st("s1", (12, 14), [dict(host="m5", **b), dict(host="m5", **b), dict(host="air", **b)]),
             st("s2", (14, 28), [dict(host="air")]),
             st("head", None, [dict(host="m5", **b), dict(host="air", **b), dict(host="air", **b)], lm_head=True))


def test_weak_nodes_fit_hybrid_and_pure_pp_is_rejected_with_names():
    specs = plan(Q7B, a6_placement())
    s1 = [s for s in specs if s.stage_name == "s1"]
    assert [s.q1 - s.q0 for s in s1] == [10, 9, 9]
    assert all(s.planned["total"] <= s.budget for s in specs if s.budget)
    assert frames_per_step(specs)["total"] == 34
    pp = P(st("s0", (0, 12), [dict(host="m5")], embed=True), st("s1a", (12, 13), [dict(host="m5", budget_gb=0.40)]),
           st("s1b", (13, 14), [dict(host="m5", budget_gb=0.40)]), st("s2", (14, 28), [dict(host="air")]),
           st("head", None, [dict(host="m5", budget_gb=0.40)], lm_head=True))
    with pytest.raises(PlanError) as e:
        plan(Q7B, pp)
    assert "s1a/0" in str(e.value) and "s1b/0" in str(e.value) and "head/0" in str(e.value)
    assert len(plan(Q7B, pp, check_budget=False)) == 5


def test_ohads_shape_eight_stages_of_pairs_frames():
    stages = [st(f"s{i}", (3 * i, 3 * i + 3), [dict(host="a"), dict(host="b")], embed=i == 0, lm_head=i == 7) for i in range(8)]
    specs = plan(Q05B, P(*stages))
    assert len(specs) == 16 and frames_per_step(specs) == {"token": 1, "embed": 2, "layers": 96, "hops": 14, "head": 2, "total": 115}
    assert all(s.planned["kv_heads_replicated"] == 0 for s in specs)


def test_group_larger_than_heads_gives_mlp_only_ranks_and_rejects_empty_ranks():
    specs = plan(Q05B, P(st("s0", (0, 24), [dict(host="h")] * 16, embed=True, lm_head=True)))
    check_coverage(Q05B, specs)
    zero = [s for s in specs if s.q0 == s.q1]
    assert zero and all(s.m1 > s.m0 for s in zero)
    tiny = Dims(hidden=64, intermediate=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab=128, tied=False)
    with pytest.raises(PlanError, match="holds no heads"):
        plan(tiny, P(st("s0", (0, 2), [dict(host="h")] * 6), st("h", None, [dict(host="h")], lm_head=True)), )


def test_plan_validation_errors():
    with pytest.raises(PlanError, match="do not continue"):
        plan(Q05B, P(st("s0", (0, 10), [dict(host="a")]), st("s1", (11, 24), [dict(host="a")])))
    with pytest.raises(PlanError, match="mixed device"):
        plan(Q05B, P(st("s0", (0, 24), [dict(host="a"), dict(host="a", device="cpu")])))
    with pytest.raises(PlanError, match="heads override"):
        plan(Q05B, P(st("s0", (0, 24), [dict(host="a", heads=7), dict(host="a", heads=6)])))
    with pytest.raises(PlanError, match="coordinator"):
        plan(Q05B, P(st("s0", (0, 24), [dict(host="a")]), coordinator_host="b"))


def test_largest_remainder_and_yaml_parse():
    assert largest_remainder([3, 2], 28) == [17, 11] and sum(largest_remainder([1, 1, 1], 148)) == 148
    y = {"model": "m", "coordinator": {"host": "m5"}, "stages": [{"name": "s0", "layers": [0, 24], "embed": True,
         "ranks": [{"host": "m5", "port": 1, "budget_gb": 0.4, "heads": 14}]}]}
    p = placement_from_dict(y)
    assert p.stages[0].layers == (0, 24) and p.stages[0].ranks[0].heads == 14 and p.coordinator_host == "m5"


# ---- slices, embedding and head --------------------------------------------------------------------------------------------
def test_slice_spec_kv_map_and_zero_heads():
    s = SliceSpec(28, 4, 128, 18944, 17, 28, 0, 0)
    assert (s.kv0, s.kv1) == (2, 4) and s.kv_index() == [0] * 4 + [1] * 7
    z = SliceSpec(14, 2, 64, 4864, 7, 7, 0, 128)
    assert z.heads_local == 0 and z.kv_heads_local == 0 and z.q_rows() == slice(448, 448)


def test_vocab_parallel_embedding_and_argmax_are_exact(tmp_path):
    d = write_tiny_model(tmp_path / "t")
    cfg = load_config(d)
    full = SliceSpec(4, 2, 16, 128, 0, 4, 0, 128)
    w, _ = load_slices(d, cfg, LoadSpec((0, 0), full, (0, 128), (0, 128), True), "cpu", dtype=torch.float32)
    ref = Qwen2TP(cfg, w, full, "cpu", 16, dtype=torch.float32, layers=(0, 0))
    ids = [0, 5, 47, 48, 100, 127]
    parts, cands = [], []
    h = torch.randn(3, 64)
    for v in [(0, 48), (48, 96), (96, 128)]:
        wv, _ = load_slices(d, cfg, LoadSpec((0, 0), full, v, v, True), "cpu", dtype=torch.float32)
        m = Qwen2TP(cfg, wv, full, "cpu", 16, dtype=torch.float32, layers=(0, 0), embed_rows=v, head_rows=v)
        parts.append(m.embed_tokens(ids))
        val, idx, _ = m.head_argmax(h)
        cands.append((val, idx))
    assert torch.equal(sum(parts), ref.embed_tokens(ids))
    assert pick_global(cands) == int(ref.logits_last(h).argmax())
    assert pick_global([(1.0, 9), (2.0, 70), (2.0, 40)]) == 40          # ties to the lowest index


def test_budget_is_enforced_at_load(tmp_path):
    d = write_tiny_model(tmp_path / "t")
    cfg = load_config(d)
    sl = SliceSpec(4, 2, 16, 128, 0, 4, 0, 128)
    w, n = load_slices(d, cfg, LoadSpec((0, 2), sl, None, None, False), "cpu")
    with pytest.raises(BudgetExceeded):
        load_slices(d, cfg, LoadSpec((0, 2), sl, None, None, False), "cpu", budget=n - 1)
    dims = Dims.of(cfg)
    specs = plan(dims, P(st("s0", (0, 2), [dict(host="h")] * 3, embed=True, lm_head=True), granule=16))
    for s in specs:      # planned weight bytes equal loaded tensor bytes
        _, got = load_slices(d, cfg, LoadSpec(s.layers, s.slice_spec(dims), s.embed_rows, s.head_rows, s.final_norm), "cpu")
        assert got == s.planned["weights"], s.name


# ---- N rank all-reduce -------------------------------------------------------------------------------------------------------
def mesh(n):
    """Group links over socketpairs: links[i][j] is rank i's link to rank j."""
    links = [dict() for _ in range(n)]
    srv = transport.listen("127.0.0.1", 0, backlog=64)
    port = srv.getsockname()[1]
    for i in range(n):
        for j in range(i + 1, n):
            a = transport.connect("127.0.0.1", port)
            b, _ = srv.accept()
            links[i][j], links[j][i] = transport.Link(a), transport.Link(b)
    srv.close()
    return links


@pytest.mark.parametrize("n,heads", [(3, None), (5, [1, 1, 1, 1, 0])])
def test_n_ranks_fp32_match_one_rank_and_are_identical(tmp_path, n, heads):
    from nocturne.engines.torch_tp.hybrid import GroupComm
    d = write_tiny_model(tmp_path / "t")
    cfg = load_config(d)
    dims = Dims.of(cfg)
    ranks = [dict(host="h", **({"heads": heads[i]} if heads else {})) for i in range(n)]
    specs = plan(dims, P(st("s0", (0, 2), ranks, embed=True, lm_head=True), granule=16))
    ms = []
    for s in specs:
        sl = s.slice_spec(dims)
        w, _ = load_slices(d, cfg, LoadSpec(s.layers, sl, s.embed_rows, s.head_rows, s.final_norm), "cpu", dtype=torch.float32)
        ms.append(Qwen2TP(cfg, w, sl, "cpu", 32, dtype=torch.float32, embed_rows=s.embed_rows, head_rows=s.head_rows))
    one = SliceSpec(4, 2, 16, 128, 0, 4, 0, 128)
    w1, _ = load_slices(d, cfg, LoadSpec((0, 2), one, (0, 128), (0, 128), True), "cpu", dtype=torch.float32)
    ref = Qwen2TP(cfg, w1, one, "cpu", 32, dtype=torch.float32)
    ids = [3, 9, 27, 81, 120, 1, 2]
    href = ref.layers_forward(ref.embed_tokens(ids), 0, Comm())
    lk = mesh(n)
    out = [None] * n

    def run(i):
        c = GroupComm(i, n, lk[i], "cpu", "float32")
        h = c.all_reduce(ms[i].embed_tokens(ids))
        out[i] = ms[i].layers_forward(h, 0, c)
    th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert (out[0] - href).abs().max().item() <= 1e-5
    assert all(torch.equal(out[0], o) for o in out[1:])
    assert pick_global([ms[i].head_argmax(out[i])[:2] for i in range(n)]) == int(ref.logits_last(href).argmax())


# ---- the pipeline end to end ---------------------------------------------------------------------------------------------------
def free_ports(k):
    socks = [socket.socket() for _ in range(k)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def run_pipeline(model_dir, placement, n_tokens=6, ids=(5, 17, 99, 3, 42, 8)):
    from nocturne.engines.torch_tp.hybrid import Coordinator, decode_frames, rank_main
    cfg = load_config(model_dir)
    specs = plan(Dims.of(cfg), placement)
    for s in specs[1:]:
        threading.Thread(target=rank_main, args=(["127.0.0.1"], s.port, str(model_dir), "cpu"),
                         kwargs={"log": lambda *a: None}, daemon=True).start()
    co = Coordinator(specs, str(model_dir), {"h": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                     placement.prefill_max, device="cpu", log=lambda *a: None, dtype="float32")
    try:
        res = co.generate(list(ids), n_tokens, ignore_eos=True, first_logits=True)
        m = co.metrics()
    finally:
        co.shutdown()
    return specs, res, m, decode_frames(m)


def test_pipeline_of_stages_matches_one_rank_and_the_frame_formula(tmp_path):
    d = write_tiny_model(tmp_path / "t")
    pts = free_ports(8)
    one = P(st("s0", (0, 2), [dict(host="h", device="cpu", port=pts[0])], embed=True, lm_head=True), max_seq_len=64)
    _, ref, _, _ = run_pipeline(d, one)
    hyb = P(StageIn("s0", (0, 1), (RankIn("h", "cpu", pts[1]), RankIn("h", "cpu", pts[2])), embed=True),
            StageIn("s1", (1, 2), (RankIn("h", "cpu", pts[3], heads=2), RankIn("h", "cpu", pts[4], heads=1),
                                   RankIn("h", "cpu", pts[5], heads=1))),
            StageIn("head", None, (RankIn("h", "cpu", pts[6]), RankIn("h", "cpu", pts[7])), lm_head=True),
            max_seq_len=64, granule=16)
    specs, res, m, fr = run_pipeline(d, hyb)
    assert res["tokens"] == ref["tokens"]
    assert (res["first_logits"] - ref["first_logits"]).abs().max().item() <= 1e-4
    steps = m["decode_steps"]
    f = frames_per_step(specs)
    assert f == {"token": 1, "embed": 2, "layers": 2 * 2 + 1 * 2 * 6, "hops": 3 + 2, "head": 2, "total": 26}
    assert fr["sent_frames"] == f["total"] * steps
    for r in m["ranks"].values():       # planned bf16 weight bytes equal loaded tensor bytes (fp32 here: twice)
        assert r["loaded_tensor_bytes"] == 2 * r["planned"]["weights"], r["name"]


def test_budget_failure_is_reported_with_the_rank_name(tmp_path):
    from nocturne.engines.torch_tp.hybrid import RankFailed
    d = write_tiny_model(tmp_path / "t")
    pts = free_ports(2)
    pl = P(st("s0", (0, 1), [dict(host="h", device="cpu", port=pts[0])], embed=True),
           StageIn("s1", (1, 2), (RankIn("h", "cpu", pts[1], budget_gb=1e-5),), lm_head=True), max_seq_len=64)
    from nocturne.engines.torch_tp.hybrid import Coordinator, rank_main
    cfg = load_config(d)
    specs = plan(Dims.of(cfg), pl, check_budget=False)
    threading.Thread(target=rank_main, args=(["127.0.0.1"], pts[1], str(d), "cpu"), kwargs={"log": lambda *a: None},
                     daemon=True).start()
    with pytest.raises(RankFailed, match="s1/0: BudgetExceeded"):
        Coordinator(specs, str(d), {"h": "127.0.0.1"}, ["127.0.0.1"], 64, "float32", 256, device="cpu", log=lambda *a: None)


def test_tiny_config_is_what_the_tests_assume():
    assert TINY["num_attention_heads"] == 4 and TINY["num_key_value_heads"] == 2 and TINY["intermediate_size"] == 128
