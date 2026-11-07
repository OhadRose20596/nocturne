"""Qwen2 config from the Hugging Face config.json. Only the fields the engine needs."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen2Config:
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

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def shard_ok(self, world: int) -> None:
        for name, n in (("num_attention_heads", self.num_attention_heads), ("num_key_value_heads", self.num_key_value_heads),
                        ("intermediate_size", self.intermediate_size)):
            if n % world:
                raise ValueError(f"{name}={n} not divisible by world size {world}")


def load_config(model_dir: str | Path) -> Qwen2Config:
    d = json.loads((Path(model_dir) / "config.json").read_text())
    if d.get("model_type") != "qwen2":
        raise ValueError(f"model_type {d.get('model_type')!r} is not qwen2; this engine implements Qwen2 only")
    eos = d.get("eos_token_id", [])
    eos = tuple(eos) if isinstance(eos, list) else (eos,)
    gen = Path(model_dir) / "generation_config.json"
    if gen.exists():
        g = json.loads(gen.read_text()).get("eos_token_id", [])
        eos = tuple(sorted(set(eos) | set(g if isinstance(g, list) else [g])))
    return Qwen2Config(
        hidden_size=d["hidden_size"], intermediate_size=d["intermediate_size"], num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"], num_key_value_heads=d["num_key_value_heads"], vocab_size=d["vocab_size"],
        rms_norm_eps=d.get("rms_norm_eps", 1e-6), rope_theta=d.get("rope_theta", 1e6),
        tie_word_embeddings=d.get("tie_word_embeddings", False), max_position_embeddings=d.get("max_position_embeddings", 32768),
        eos_token_ids=eos, torch_dtype=d.get("torch_dtype", "bfloat16"))
