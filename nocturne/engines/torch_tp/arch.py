"""Architecture table: one row per supported model family, not one class per model.
Engine tensor names are the Hugging Face names (model.layers.N.self_attn.q_proj.weight, ...). GGUF names map onto them.
The converter (llama.cpp convert_hf_to_gguf.py, conversion/llama.py) permutes the rows of q_proj and k_proj for llama
(`undo_permute = True`) so that ggml's interleaved rope works; Qwen2 and Qwen3 are not permuted. We rotate with the
Hugging Face half split layout, so a llama GGUF needs the permutation undone on load."""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    name: str                  # GGUF general.architecture and HF model_type
    qkv_bias: bool
    qk_norm: bool              # per head RMSNorm of q and k before rope (Qwen3)
    gguf_rope_permuted: bool   # the GGUF holds q and k rows in ggml's interleaved order


ARCHS = {
    "qwen2": Arch("qwen2", qkv_bias=True, qk_norm=False, gguf_rope_permuted=False),
    "llama": Arch("llama", qkv_bias=False, qk_norm=False, gguf_rope_permuted=True),
    "qwen3": Arch("qwen3", qkv_bias=False, qk_norm=True, gguf_rope_permuted=False),
}


def get_arch(name: str) -> Arch:
    if name not in ARCHS:
        raise ValueError(f"architecture {name!r} is not supported; supported: {', '.join(ARCHS)}")
    return ARCHS[name]


# GGUF block tensor suffix -> engine suffix (after model.layers.N.)
BLOCK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight", "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
}
TOP_MAP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
           "output.weight": "lm_head.weight"}
IGNORED = ("rope_freqs.weight",)           # read into the config, not a weight
_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def engine_name(gguf_name: str) -> str | None:
    """Engine name of a GGUF tensor, None for the ones that are not weights (rope_freqs). Raises on an unknown tensor."""
    if gguf_name in IGNORED:
        return None
    if gguf_name in TOP_MAP:
        return TOP_MAP[gguf_name]
    m = _BLK.match(gguf_name)
    if m and m.group(2) in BLOCK_MAP:
        return f"model.layers.{m.group(1)}.{BLOCK_MAP[m.group(2)]}"
    raise KeyError(f"GGUF tensor {gguf_name!r} has no engine name")
