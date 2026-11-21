"""Engine GGUF tests (V1). Every check compares against an implementation we did not write: gguf-py's numpy dequantizer,
transformers' GGUF loader, the converter's own permutation. Tests that need local model files skip when they are absent."""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
from gguf.quants import dequantize as ref_dequantize  # noqa: E402

from nocturne.engines.torch_tp import quant  # noqa: E402
from nocturne.engines.torch_tp.arch import engine_name  # noqa: E402

MODELS = Path(os.path.expanduser("~/models"))
QTYPES = ["Q8_0", "Q4_0", "Q5_0", "Q4_K", "Q5_K", "Q6_K"]
SCALE_FIELDS = {"Q8_0": [(0, 2)], "Q4_0": [(0, 2)], "Q5_0": [(0, 2)], "Q4_K": [(0, 2), (2, 4)], "Q5_K": [(0, 2), (2, 4)], "Q6_K": [(208, 210)]}


def random_blocks(qtype: str, n: int, seed: int) -> np.ndarray:
    """Random payload bytes with valid float16 scale fields (random fp16 bytes would be nan or inf)."""
    rng = np.random.default_rng(seed)
    bs, ts = quant.BLOCK[qtype]
    b = rng.integers(0, 256, size=(n, ts), dtype=np.uint8)
    for a, z in SCALE_FIELDS[qtype]:
        b[:, a:z] = (rng.standard_normal((n, 1)).astype(np.float16) * 0.05).view(np.uint8).reshape(n, 2)
    return b


@pytest.mark.parametrize("qtype", QTYPES)
def test_dequant_random_blocks_exact(qtype):
    b = random_blocks(qtype, 300, seed=1)
    ref = ref_dequantize(b, getattr(gguf.GGMLQuantizationType, qtype)).reshape(300, -1)
    ours = quant.dequant_blocks(torch.from_numpy(b), qtype).numpy()
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref), f"max abs diff {np.abs(ours - ref).max()}"


@pytest.mark.parametrize("qtype", QTYPES)
def test_dequant_chunked_matches_unchunked(qtype, monkeypatch):
    bs, ts = quant.BLOCK[qtype]
    raw = torch.from_numpy(random_blocks(qtype, 24, 2)).reshape(6, 4, ts)
    full = quant.dequant(raw, qtype, torch.float32)
    monkeypatch.setattr(quant, "CHUNK_ELEMS", 2 * 4 * bs)       # two rows per chunk
    assert torch.equal(quant.dequant(raw, qtype, torch.float32), full)


def test_unsupported_type_is_refused_with_message():
    with pytest.raises(quant.UnsupportedQuant, match="Q2_K"):
        quant.check_supported("Q2_K")


def _gguf_files():
    return sorted(MODELS.glob("*.gguf")) if MODELS.exists() else []


SMALL = [p for p in _gguf_files() if p.stat().st_size < 2_000_000_000]


@pytest.mark.parametrize("path", SMALL, ids=lambda p: p.name)
def test_real_tensors_match_gguf_py(path):
    from nocturne.engines.torch_tp.gguf_source import GgufSource
    src = GgufSource(path)
    seen = set()
    for t in src.reader.tensors:
        key = (t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name, t.tensor_type.name)
        if key in seen or t.tensor_type.name not in quant.BLOCK:
            continue
        seen.add(key)
        if t.tensor_type.name in quant.DENSE:
            continue
        en = engine_name(t.name)
        raw, qt = src.read(en, rows=slice(0, 4))
        ref = ref_dequantize(np.asarray(t.data)[:4], t.tensor_type)
        ours = quant.dequant(raw, qt, torch.float32).numpy()
        assert np.array_equal(ours, ref), f"{path.name} {t.name} {qt}: max abs diff {np.abs(ours - ref).max()}"


@pytest.mark.parametrize("path", [p for p in SMALL if "0.5b" in p.name.lower()], ids=lambda p: p.name)
def test_real_tensors_match_transformers_loader(path):
    """A second reading of names and values: transformers' own GGUF name map and un-permute. It dequantizes with gguf-py's
    code too, so for the values it is not independent of the first test; for names and permutation it is."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
    from nocturne.engines.torch_tp.gguf_source import GgufSource
    conf = AutoConfig.from_pretrained(str(path.parent), gguf_file=path.name)
    with torch.device("meta"):
        shell = AutoModelForCausalLM.from_config(conf)
    ck = load_gguf_checkpoint(str(path), return_tensors=True, model_to_load=shell)["tensors"]
    src = GgufSource(path)
    checked = 0
    for en, m in src.meta().items():
        if m.qtype in quant.DENSE or len(m.shape) != 2 or en not in ck:
            continue
        raw, qt = src.read(en, rows=slice(0, 8))
        ours = quant.dequant(raw, qt, torch.float32)
        theirs = ck[en][:8].float()
        assert torch.equal(ours, theirs), f"{en} {qt}: max abs diff {(ours - theirs).abs().max()}"
        checked += 1
        if checked >= 40:
            break
    assert checked >= 10


@pytest.mark.parametrize("path", _gguf_files(), ids=lambda p: p.name)
def test_name_map_covers_every_tensor_exactly_once(path):
    from nocturne.engines.torch_tp.gguf_source import GgufSource
    src = GgufSource(path)
    mapped = [engine_name(t.name) for t in src.reader.tensors]
    weights = [n for n in mapped if n]
    assert len(weights) == len(set(weights)), "two GGUF tensors map to one engine name"
    assert len(weights) == len(src.meta())
    cfg = src.config()
    expect_layer = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                    "input_layernorm.weight", "post_attention_layernorm.weight"}
    if cfg.arch.qkv_bias:
        expect_layer |= {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}
    if cfg.arch.qk_norm:
        expect_layer |= {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}
    for i in range(cfg.num_hidden_layers):
        have = {n.split(".", 3)[3] for n in weights if n.startswith(f"model.layers.{i}.")}
        assert have == expect_layer, f"layer {i}: missing {expect_layer - have}, extra {have - expect_layer}"
    assert {"model.embed_tokens.weight", "model.norm.weight"} <= set(weights)


def test_name_map_agrees_with_gguf_py_tensor_name_map():
    """The converter's own map (gguf.get_tensor_name_map) names HF tensors the same way as our table."""
    m = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, 4)
    for hf, ours in (("model.layers.2.self_attn.q_proj", "blk.2.attn_q"), ("model.layers.2.mlp.down_proj", "blk.2.ffn_down"),
                     ("model.layers.2.input_layernorm", "blk.2.attn_norm"), ("model.layers.2.post_attention_layernorm", "blk.2.ffn_norm"),
                     ("model.embed_tokens", "token_embd"), ("lm_head", "output"), ("model.norm", "output_norm")):
        assert m.get_name(hf) == ours
        assert engine_name(ours + ".weight") == hf + ".weight"


def test_unknown_gguf_tensor_raises():
    with pytest.raises(KeyError, match="no engine name"):
        engine_name("blk.0.attn_qkv.weight")
