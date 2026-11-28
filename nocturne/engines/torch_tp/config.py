"""Model config for the torch engine, from a Hugging Face config.json or from GGUF metadata (gguf_source.py).
One dataclass for every architecture in arch.py. Only the fields the engine needs."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .arch import Arch, get_arch


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]
    torch_dtype: str = "bfloat16"
    arch_name: str = "qwen2"
    head_dim: int = 0                          # 0 means hidden_size // num_attention_heads
    rope_factors: tuple[float, ...] | None = None    # per frequency divisors (Llama 3 scaling), None for plain rope

    def __post_init__(self):
        if not self.head_dim:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)
        get_arch(self.arch_name)

    @property
    def arch(self) -> Arch:
        return get_arch(self.arch_name)

    def shard_ok(self, world: int) -> None:
        for name, n in (("num_attention_heads", self.num_attention_heads), ("num_key_value_heads", self.num_key_value_heads),
                        ("intermediate_size", self.intermediate_size)):
            if n % world:
                raise ValueError(f"{name}={n} not divisible by world size {world}")


Qwen2Config = ModelConfig       # name kept for old imports


def config_to_json(cfg: ModelConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def config_from_json(d: dict) -> ModelConfig:
    """A ModelConfig that travelled in the PLAN message (JSON turns its tuples into lists)."""
    d = dict(d)
    d["eos_token_ids"] = tuple(d.get("eos_token_ids") or ())
    if d.get("rope_factors") is not None:
        d["rope_factors"] = tuple(d["rope_factors"])
    return ModelConfig(**d)


def llama3_factors(head_dim: int, theta: float, rs: dict) -> tuple[float, ...]:
    """Per frequency divisors, the formula of convert_hf_to_gguf (conversion/llama.py, generate_extra_tensors)."""
    freqs = [1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)]
    factor = rs.get("factor", 8.0)
    low, high = rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0)
    old = rs.get("original_max_position_embeddings", 8192)
    out = []
    for f in freqs:
        wl = 2 * math.pi / f
        if wl < old / high:
            out.append(1.0)
        elif wl > old / low:
            out.append(float(factor))
        else:
            smooth = (old / wl - low) / (high - low)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return tuple(out)


def load_config(model_dir: str | Path) -> ModelConfig:
    """A safetensors directory (config.json) or a .gguf file."""
    p = Path(model_dir)
    if p.suffix == ".gguf":
        from .gguf_source import open_gguf
        return open_gguf(p).config()
    d = json.loads((p / "config.json").read_text())
    arch = d.get("model_type")
    get_arch(arch)
    eos = d.get("eos_token_id", [])
    eos = tuple(eos) if isinstance(eos, list) else (eos,)
    gen = p / "generation_config.json"
    if gen.exists():
        g = json.loads(gen.read_text()).get("eos_token_id", [])
        eos = tuple(sorted(set(eos) | set(g if isinstance(g, list) else [g])))
    if d.get("attention_bias") or d.get("mlp_bias"):
        raise ValueError(f"{arch} with attention_bias or mlp_bias is not supported: the engine applies only the q, k, v bias of qwen2")
    rp = d.get("rope_parameters") or {}
    theta = d.get("rope_theta", rp.get("rope_theta", 1e6 if arch != "llama" else 10000.0))
    hd = d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"]
    rs = d.get("rope_scaling") or (rp if rp.get("rope_type") else None)
    factors = None
    if rs and rs.get("rope_type", rs.get("type")) == "llama3":
        factors = llama3_factors(hd, theta, rs)
    elif rs and rs.get("rope_type", rs.get("type")) not in (None, "default"):
        raise ValueError(f"rope scaling {rs.get('rope_type', rs.get('type'))!r} is not supported")
    return ModelConfig(
        hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"], num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"], num_key_value_heads=d["num_key_value_heads"], vocab_size=d["vocab_size"],
        rms_norm_eps=d.get("rms_norm_eps", 1e-6), rope_theta=theta,
        tie_word_embeddings=d.get("tie_word_embeddings", False), max_position_embeddings=d.get("max_position_embeddings", 32768),
        eos_token_ids=eos, torch_dtype=d.get("torch_dtype", d.get("dtype", "bfloat16")), arch_name=arch, head_dim=hd,
        rope_factors=factors)
