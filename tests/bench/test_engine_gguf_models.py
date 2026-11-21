"""Engine GGUF, model level (V1): tiny random qwen2, llama (Llama 3 rope scaling) and qwen3 models made as safetensors and as
GGUF by llama.cpp's own converter and quantizer. The engine loads both and must agree with them and with transformers."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gguf")
from . import gguf_tiny  # noqa: E402

if not gguf_tiny.available():
    pytest.skip("llama.cpp converter, llama-quantize or the local Qwen tokenizer files are missing", allow_module_level=True)

from nocturne.engines.torch_tp import quant  # noqa: E402
from nocturne.engines.torch_tp.config import load_config, llama3_factors  # noqa: E402
from nocturne.engines.torch_tp.gguf_source import GgufSource  # noqa: E402
from nocturne.engines.torch_tp.loader import LoadSpec, load_slices  # noqa: E402
from nocturne.engines.torch_tp.model import Comm, Qwen2TP  # noqa: E402
from nocturne.engines.torch_tp.plan import Dims, Placement, PlanError, RankIn, StageIn, plan  # noqa: E402
from nocturne.engines.torch_tp.shard import Shard  # noqa: E402

ARCHS = gguf_tiny.ARCHS


@pytest.fixture(scope="module")
def models():
    return {a: gguf_tiny.build(a) for a in ARCHS}


def load_all(path, cfg, dtype=torch.bfloat16):
    sh = Shard(0, 1, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    w, nbytes = load_slices(path, cfg, LoadSpec.from_shard(cfg, sh), "cpu", dtype=dtype, log=lambda *a: None)
    return w, sh, nbytes


def dense(t, dtype=torch.bfloat16):
    return t.weight(dtype) if isinstance(t, quant.QuantLinear) else t


@pytest.mark.parametrize("arch", ARCHS)
def test_config_from_gguf_matches_hf_config(models, arch):
    a, b = load_config(models[arch]["hf"]), load_config(models[arch]["bf16"])
    for f in ("arch_name", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "vocab_size", "tie_word_embeddings", "rope_theta"):
        assert getattr(a, f) == getattr(b, f), f
    assert abs(a.rms_norm_eps - b.rms_norm_eps) < 1e-9
    assert (a.rope_factors is None) == (b.rope_factors is None)
    if a.rope_factors:
        assert np.allclose(a.rope_factors, b.rope_factors, rtol=1e-6)
    assert 151645 in b.eos_token_ids


def test_llama3_factors_equal_the_converters_rope_freqs(models):
    """The converter writes rope_freqs.weight from the HF config; our config formula must give the same numbers."""
    f = load_config(models["llama"]["hf"]).rope_factors
    g = load_config(models["llama"]["bf16"]).rope_factors
    assert len(f) == 32 and max(abs(x - y) / y for x, y in zip(f, g)) < 1e-6
    assert min(f) == 1.0 and max(f) == 8.0                       # both regimes of the scaling are present
    assert llama3_factors(64, 500000.0, {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                         "original_max_position_embeddings": 2048}) == f


@pytest.mark.parametrize("source", ("hf", "bf16"))
def test_llama3_rotary_matches_transformers_at_long_positions(models, source):
    """Independent check of rope scaling: transformers' own LlamaRotaryEmbedding, positions where the scaling matters."""
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from nocturne.engines.torch_tp.model import rotary
    hf = LlamaRotaryEmbedding(AutoConfig.from_pretrained(str(models["llama"]["hf"])))
    pos = torch.tensor([0, 1, 100, 2047, 5000, 8000])
    cos_ref, sin_ref = hf(torch.zeros(1, 1, 64, dtype=torch.bfloat16), pos[None])
    cfg = load_config(models["llama"][source])
    cos, sin = rotary(pos, cfg.head_dim, cfg.rope_theta, torch.bfloat16, "cpu", cfg.rope_factors)
    assert (cos.float() - cos_ref[0].float()).abs().max() <= 2 ** -7 * 2          # at most 2 bf16 ulps at |x| near 1
    assert (sin.float() - sin_ref[0].float()).abs().max() <= 2 ** -7 * 2
    plain, _ = rotary(pos, cfg.head_dim, cfg.rope_theta, torch.bfloat16, "cpu", None)
    assert (plain.float() - cos_ref[0].float()).abs().max() > 0.05                    # the scaling is not a no-op on this config


def test_qwen3_head_size_comes_from_key_length_not_hidden_over_heads(models):
    for kind in ("hf", "bf16"):
        cfg = load_config(models["qwen3"][kind])
        assert cfg.hidden_size // cfg.num_attention_heads == 32 and cfg.head_dim == 64


@pytest.mark.parametrize("arch", ARCHS)
def test_bf16_gguf_weights_equal_safetensors_exactly(models, arch):
    """Includes the llama Q and K un-permutation: the converter permuted them, the loader must put them back."""
    cfg = load_config(models[arch]["hf"])
    ws, _, _ = load_all(models[arch]["hf"], cfg)
    wg, _, _ = load_all(models[arch]["bf16"], load_config(models[arch]["bf16"]))
    assert set(ws) == set(wg)
    for name in ws:
        a, b = dense(ws[name]), dense(wg[name])
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a, b), f"{arch} {name}: max abs diff {(a.float() - b.float()).abs().max()}"


def raw_bf16(src, name):
    """The tensor exactly as stored in the file (gguf-py hands BF16 over as bytes)."""
    b = np.ascontiguousarray(np.asarray(src._tensors[name].data)).reshape(-1).view(np.int16).reshape(256, 256)
    return torch.from_numpy(b.copy()).view(torch.bfloat16)


def test_llama_permute_really_happens_in_the_file(models):
    """Guards the test above against passing for the wrong reason: the stored rows differ from the HF order."""
    src = GgufSource(models["llama"]["bf16"])
    stored, _ = src.read("model.layers.0.self_attn.q_proj.weight")                  # un-permuted by the reader
    raw = raw_bf16(src, "model.layers.0.self_attn.q_proj.weight")
    assert not torch.equal(stored, raw) and torch.equal(stored.sort(dim=0).values, raw.sort(dim=0).values)
    qwen = GgufSource(models["qwen2"]["bf16"])
    s2, _ = qwen.read("model.layers.0.self_attn.q_proj.weight")
    r2 = raw_bf16(qwen, "model.layers.0.self_attn.q_proj.weight")
    assert torch.equal(s2, r2), "qwen2 is not permuted by the converter"


@pytest.mark.parametrize("arch", ("llama", "qwen2"))
def test_unpermute_agrees_with_transformers_own_reverse_permute(models, arch):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
    p = models[arch]["q8_0"]
    conf = AutoConfig.from_pretrained(str(p.parent), gguf_file=p.name)
    with torch.device("meta"):
        shell = AutoModelForCausalLM.from_config(conf)
    ck = load_gguf_checkpoint(str(p), return_tensors=True, model_to_load=shell)["tensors"]
    src = GgufSource(p)
    for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight", "model.layers.1.self_attn.v_proj.weight"):
        raw, qt = src.read(name)
        ours = quant.dequant(raw, qt, torch.float32)
        assert torch.equal(ours, ck[name].float()), f"{arch} {name}"


@pytest.mark.parametrize("arch", ARCHS)
def test_sharded_reads_equal_slices_of_the_full_read(models, arch):
    src = GgufSource(models[arch]["q8_0"])
    name = "model.layers.1.self_attn.q_proj.weight"
    full, _ = src.read(name)
    part, _ = src.read(name, rows=slice(64, 192))
    assert torch.equal(part, full[64:192])
    o_full, _ = src.read("model.layers.1.self_attn.o_proj.weight")
    o_part, _ = src.read("model.layers.1.self_attn.o_proj.weight", cols=slice(64, 192))
    assert torch.equal(o_part, o_full[:, 2:6])                   # block of 32 columns
    with pytest.raises(ValueError, match="not aligned to the Q8_0 block of 32"):
        src.read("model.layers.1.self_attn.o_proj.weight", cols=slice(10, 64))


def hf_logits(hf_dir, ids):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.bfloat16).eval()
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].float()


def engine_logits(path, ids):
    cfg = load_config(path)
    w, sh, _ = load_all(path, cfg)
    m = Qwen2TP(cfg, w, sh, "cpu", 64, dtype=torch.bfloat16)
    h = m.layers_forward(m.embed_tokens(ids), 0, Comm())
    return m.logits_last(h).float()


IDS = [5, 17, 99, 3, 42, 8, 1000, 77]


@pytest.mark.parametrize("arch", ARCHS)
def test_engine_on_safetensors_matches_transformers(models, arch):
    ref = hf_logits(models[arch]["hf"], IDS)
    ours = engine_logits(models[arch]["hf"], IDS)
    assert int(ours.argmax()) == int(ref.argmax())
    assert (ours - ref).abs().max() <= 0.05 * ref.abs().max() + 0.05, (ours - ref).abs().max()


@pytest.mark.parametrize("arch", ARCHS)
def test_engine_on_bf16_gguf_equals_engine_on_safetensors(models, arch):
    a = engine_logits(models[arch]["hf"], IDS)
    b = engine_logits(models[arch]["bf16"], IDS)
    assert torch.equal(a, b) or (a - b).abs().max() <= 1e-2 * a.abs().max(), (a - b).abs().max()
    assert int(a.argmax()) == int(b.argmax())


def reference_dense_weights(path, cfg):
    """The same GGUF with every quantized matrix expanded by gguf-py's numpy dequantizer, cast to bf16."""
    import gguf
    w, sh, _ = load_all(path, cfg)
    src = GgufSource(path)
    for name, t in list(w.items()):
        if isinstance(t, quant.QuantLinear) and t.qtype not in quant.DENSE:
            rows = t.data.shape[0]
            flat = t.data.reshape(rows, -1).numpy()
            w[name] = gguf.quants.dequantize(flat, getattr(gguf.GGMLQuantizationType, t.qtype)).reshape(rows, -1)
            w[name] = torch.from_numpy(np.ascontiguousarray(w[name])).to(torch.bfloat16)
    return w, sh


def logits_of(cfg, w, sh):
    m = Qwen2TP(cfg, w, sh, "cpu", 64, dtype=torch.bfloat16)
    return m.logits_last(m.layers_forward(m.embed_tokens(IDS), 0, Comm())).float()


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("kind", ("q8_0", "q4_k_m"))
def test_quantized_path_equals_dense_path_on_gguf_py_dequantized_weights(models, arch, kind):
    """Isolates the engine's quantized path from quantization noise: same file, expanded by someone else's code."""
    path = models[arch][kind]
    cfg = load_config(path)
    w, sh, _ = load_all(path, cfg)
    wd, shd = reference_dense_weights(path, cfg)
    ours, ref = logits_of(cfg, w, sh), logits_of(cfg, wd, shd)
    assert torch.isfinite(ours).all()
    assert torch.equal(ours, ref), f"{arch} {kind}: max abs diff {(ours - ref).abs().max()}"


@pytest.mark.parametrize("arch", ARCHS)
def test_q8_0_logits_track_the_bf16_ones(models, arch):
    ref = engine_logits(models[arch]["bf16"], IDS)
    q = engine_logits(models[arch]["q8_0"], IDS)
    assert (q - ref).abs().mean() / ref.abs().mean() < 0.05


def test_k_quant_file_mixes_types_and_loads(models):
    src = GgufSource(models["llama"]["q4_k_m"])
    types = {m.qtype for m in src.meta().values()}
    assert {"Q4_K", "F32"} <= types and len(types) >= 3, types
    cfg = load_config(models["llama"]["q4_k_m"])
    w, _, nbytes = load_all(models["llama"]["q4_k_m"], cfg)
    expect = sum(m.nbytes if len(m.shape) == 2 else m.shape[0] * 2 for m in src.meta().values())      # 1-D tensors are held as bf16
    assert nbytes == expect, (nbytes, expect)


# ---- planner: bytes and block alignment ---------------------------------------------------------------------------------
def P(*stages, **kw):
    return Placement(model="m", stages=tuple(stages), **kw)


def st(name, layers, ranks, **kw):
    return StageIn(name=name, layers=layers, ranks=tuple(RankIn(**r) for r in ranks), **kw)


def test_planned_weight_bytes_equal_loaded_bytes_for_every_rank(models):
    for arch in ARCHS:
        for kind in ("q8_0", "q4_k_m"):
            path = models[arch][kind]
            cfg = load_config(path)
            d = Dims.of(cfg, GgufSource(path).meta())
            specs = plan(d, P(st("s0", (0, 2), [dict(host="h", port=1), dict(host="h", port=2)], embed=True, lm_head=True)),
                         check_budget=False)
            for s in specs:
                sl = s.slice_spec(d)
                _, nbytes = load_slices(path, cfg, LoadSpec(s.layers, sl, s.embed_rows, s.head_rows, s.final_norm), "cpu",
                                        log=lambda *a: None)
                assert abs(nbytes - s.planned["weights"]) <= 0.001 * nbytes + 64, f"{arch} {kind} {s.name}: loaded {nbytes} planned {s.planned['weights']}"


def test_planner_aligns_head_splits_to_k_quant_blocks(models):
    """o_proj of the Q4_K_M file is Q4_K (256 columns per block): head_dim 64 means heads move in units of 4."""
    path = models["qwen2"]["q4_k_m"]
    d = Dims.of(load_config(path), GgufSource(path).meta())
    specs = plan(d, P(st("s", (0, 2), [dict(host="h", port=1), dict(host="h", port=2)], embed=True, lm_head=True)), check_budget=False)
    assert d.quant.o[0] == "Q4_K"
    for s in specs:
        assert (s.q0 * d.head_dim) % 256 == 0 and (s.q1 * d.head_dim) % 256 == 0 or s.q1 == d.n_heads or s.q0 == 0
        assert s.m0 % 256 == 0 and (s.m1 % 256 == 0 or s.m1 == d.intermediate)


def test_planner_rejects_unaligned_head_split_and_names_the_rule(models):
    """head_dim 64 and a 256 column block: only whole groups of 4 heads may move, and this model has exactly 4."""
    path = models["qwen2"]["q4_k_m"]
    d = Dims.of(load_config(path), GgufSource(path).meta())
    for heads in ((1, 3), (2, 2), (3, 1)):
        with pytest.raises(PlanError, match=r"multiple of its block of 256 columns.*block boundaries"):
            plan(d, P(st("s", (0, 2), [dict(host="h", port=1, heads=heads[0]), dict(host="h", port=2, heads=heads[1])],
                         embed=True, lm_head=True)), check_budget=False)
    specs = plan(d, P(st("s", (0, 2), [dict(host="h", port=1), dict(host="h", port=2)], embed=True, lm_head=True)), check_budget=False)
    assert [(s.q0, s.q1) for s in specs] == [(0, 4), (4, 4)]          # the planner's own split moves 4 heads at a time


def test_q8_0_allows_any_head_split_that_is_a_multiple_of_the_block(models):
    path = models["qwen2"]["q8_0"]
    d = Dims.of(load_config(path), GgufSource(path).meta())          # head_dim 64, block 32: every head boundary is fine
    specs = plan(d, P(st("s", (0, 2), [dict(host="h", port=1, heads=1), dict(host="h", port=2, heads=3)], embed=True, lm_head=True)),
                 check_budget=False)
    assert [(s.q0, s.q1) for s in specs] == [(0, 1), (1, 4)]


def test_transient_and_mps_floor_in_the_plan(models):
    path = models["llama"]["q8_0"]
    cfg = load_config(path)
    d = Dims.of(cfg, GgufSource(path).meta())
    pl = lambda dims: plan(dims, P(st("s", (0, 2), [dict(host="h", port=1, device="mps")], embed=True, lm_head=True)), check_budget=False)[0]
    s, dense = pl(d), pl(Dims.of(cfg))
    biggest = d.matrix_elems(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size, cfg.vocab_size)
    assert s.planned["workspace"] - dense.planned["workspace"] == min(biggest, quant.CHUNK_ELEMS) * quant.CHUNK_WORKING_BYTES_PER_ELEM
    assert s.planned["mps_floor"] == 1 << 30 and s.planned["footprint"] >= 1 << 30


def test_eos_ids_include_end_of_generation_tokens_found_by_text(models):
    """Review F1: the GGUF eos key alone is one id; llama.cpp also stops on the other end tokens of the vocabulary."""
    from pathlib import Path
    q = Path.home() / "models" / "gguf" / "Qwen2.5-1.5B-Instruct-Q8_0.gguf"
    if q.exists():
        assert {151643, 151645} <= set(load_config(q).eos_token_ids)
    cfg = load_config(models["qwen3"]["bf16"])
    assert 151645 in cfg.eos_token_ids


def test_planned_bytes_exact_for_mixed_layer_types_and_independent_gate_up():
    """Review F5 and F6: per layer sums, up and gate may differ, no off by one from float means."""
    import random
    from nocturne.engines.torch_tp.plan import QuantInfo
    rnd = random.Random(7)
    bpe = lambda t: quant.BLOCK[t][1] / quant.BLOCK[t][0]
    for L in (7, 13, 28, 36):
        qi = QuantInfo(*[tuple(rnd.choice(("Q4_K", "Q6_K", "Q5_K")) for _ in range(L)) for _ in range(7)], "Q4_K", "Q6_K")
        d = Dims(hidden=512, intermediate=1536, n_layers=L, n_heads=8, n_kv_heads=2, vocab=1024, tied=False, hd=64, qkv_bias=False, quant=qi)
        sp = plan(d, P(st("s", (0, L), [dict(host="h", port=1)])), check_budget=False)[0]
        per_layer = sum(8 * 64 * 512 * (bpe(qi.q[i]) + bpe(qi.o[i])) + 2 * 64 * 512 * (bpe(qi.k[i]) + bpe(qi.v[i]))
                        + 1536 * 512 * (bpe(qi.gate[i]) + bpe(qi.up[i]) + bpe(qi.down[i])) + 2 * 512 * 2 for i in range(L))
        want = per_layer + sp.planned["vocab_bytes"] + 512 * 2           # plus the final norm
        assert abs(sp.planned["weights"] - want) <= 0.5, (L, sp.planned["weights"], want)


def test_llama_with_attention_bias_is_rejected_not_silently_wrong(tmp_path):
    """Codex finding 3: the engine applies only the qwen2 q, k, v bias, so other bias settings must fail loudly."""
    import json
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "attention_bias": True, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 64}))
    with pytest.raises(ValueError, match="attention_bias"):
        load_config(tmp_path)


def test_mps_floor_only_when_a_tensor_reaches_10_mib():
    """Codex finding 4: tiny ranks get no 1 GiB floor, big ones do."""
    from nocturne.engines.torch_tp.plan import QuantInfo
    QI = lambda L: QuantInfo(*[("Q8_0",) * L] * 7, "Q8_0", "Q8_0")
    tiny = Dims(hidden=32, intermediate=64, n_layers=1, n_heads=2, n_kv_heads=1, vocab=64, tied=False, hd=16, qkv_bias=False,
                quant=QI(1))
    pl = lambda d: plan(d, P(st("s", (0, d.n_layers), [dict(host="h", port=1, device="mps")], embed=True, lm_head=True), prefill_max=4, max_seq_len=16),
                        check_budget=False)[0].planned
    assert pl(tiny)["mps_floor"] == 0 and pl(tiny)["footprint"] == pl(tiny)["total"]
    big = Dims(hidden=4096, intermediate=11008, n_layers=2, n_heads=32, n_kv_heads=8, vocab=32000, tied=False, hd=128, qkv_bias=False,
               quant=QI(2))
    assert pl(big)["mps_floor"] == 1 << 30
