"""Load one rank's slice of a Qwen2 safetensors checkpoint with safe_open and tensor slicing. The full model is never
materialized: every tensor is read as the slice the rank needs, straight to the compute device."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import Qwen2Config
from .shard import Shard, shard_tensor

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def weight_files(model_dir: Path) -> dict[str, Path]:
    """tensor name -> file."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: model_dir / v for k, v in wm.items()}
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {model_dir}")
    with safe_open(str(single), framework="pt", device="cpu") as f:
        return {k: single for k in f.keys()}


def load_rank(model_dir: str | Path, cfg: Qwen2Config, shard: Shard, device: str, dtype: torch.dtype = torch.bfloat16,
              log=print) -> tuple[dict[str, torch.Tensor], int]:
    """{name: tensor on device} for this rank, and the bytes loaded. Rank 0 also holds embed_tokens, final norm and
    lm_head (replicated nowhere else); other ranks hold only their layer slices."""
    model_dir = Path(model_dir)
    files = weight_files(model_dir)
    out: dict[str, torch.Tensor] = {}
    nbytes = 0
    by_file: dict[Path, list[str]] = {}
    for name, f in files.items():
        by_file.setdefault(f, []).append(name)
    for f, names in sorted(by_file.items()):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for name in sorted(names):
                if name.startswith("model.layers."):
                    local = name.split(".", 3)[3]
                    spec = shard.slice_for(local)
                    t = shard_tensor(sf.get_slice(name), spec)
                elif name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                    if shard.rank != 0:
                        continue
                    t = sf.get_tensor(name)
                else:
                    log(f"[torch_tp] skipping unknown tensor {name}")
                    continue
                t = t.to(dtype) if t.is_floating_point() else t
                out[name] = t.to(device)
                nbytes += t.numel() * t.element_size()
    if shard.rank == 0 and "lm_head.weight" not in out:
        if not cfg.tie_word_embeddings:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
        out["lm_head.weight"] = out["model.embed_tokens.weight"]
    return out, nbytes
