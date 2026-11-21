"""Load one rank's slice of a Qwen2 safetensors checkpoint with safe_open and tensor slicing. The full model is never
materialized: every tensor is read as the slice the rank needs, straight to the compute device.
LoadSpec says what a rank holds: a layer range, the head and MLP slice inside those layers, and vocab rows of the
embedding and of the output head. With tied embeddings the head rows come from the embedding tensor; a rank holding
both with the same rows keeps one tensor. A budget (bytes) is enforced while loading: the load stops with
BudgetExceeded at the first tensor that crosses it."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .config import ModelConfig as Qwen2Config
from .quant import QuantLinear
from .shard import Shard

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
ROW_SPLIT_BIASES = ("self_attn.o_proj.bias", "mlp.down_proj.bias")


class BudgetExceeded(MemoryError):
    pass


@dataclass(frozen=True)
class LoadSpec:
    layers: tuple[int, int]
    slice: object                       # SliceSpec or Shard
    embed_rows: tuple[int, int] | None
    head_rows: tuple[int, int] | None
    final_norm: bool

    @classmethod
    def from_shard(cls, cfg: Qwen2Config, shard: Shard) -> "LoadSpec":
        full = (0, cfg.vocab_size) if shard.rank == 0 else None
        return cls((0, cfg.num_hidden_layers), shard, full, full, shard.rank == 0)


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


def _take(sf, name: str, spec: tuple[int, slice] | None) -> torch.Tensor:
    """The slice of tensor `name`; an empty slice gives an empty tensor of the right shape."""
    s = sf.get_slice(name)
    if spec is None:
        return s[:]
    dim, sl = spec
    shape = list(s.get_shape())
    a, b, _ = sl.indices(shape[dim])
    if b <= a:
        shape[dim] = 0
        return torch.empty(shape, dtype=torch.bfloat16)
    return s[a:b] if dim == 0 else s[:, a:b]


def _wanted(names, cfg: Qwen2Config, spec: LoadSpec) -> dict[str, tuple]:
    """{engine name: (source name, (dim, slice) | None)} for what a rank holds, from the names the source has."""
    l0, l1 = spec.layers
    want: dict[str, tuple] = {}
    for name in names:
        if name.startswith("model.layers."):
            i, local = int(name.split(".")[2]), name.split(".", 3)[3]
            if l0 <= i < l1:
                if local in ROW_SPLIT_BIASES:
                    if getattr(spec.slice, "row_bias", True):
                        want[name] = (name, None)
                else:
                    want[name] = (name, spec.slice.slice_for(local))
    if spec.embed_rows:
        want["model.embed_tokens.weight"] = ("model.embed_tokens.weight", (0, slice(*spec.embed_rows)))
    if spec.head_rows:
        src = "lm_head.weight" if "lm_head.weight" in names else "model.embed_tokens.weight"
        if src == "model.embed_tokens.weight" and not cfg.tie_word_embeddings:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is false")
        shared = src == "model.embed_tokens.weight" and spec.embed_rows == spec.head_rows
        if not shared:
            want["lm_head.weight"] = (src, (0, slice(*spec.head_rows)))
    if spec.final_norm:
        want["model.norm.weight"] = ("model.norm.weight", None)
    return want


def _load_gguf(source, cfg: Qwen2Config, spec: LoadSpec, device: str, dtype: torch.dtype, budget: int | None):
    meta = source.meta()
    want = _wanted(meta.keys(), cfg, spec)
    out: dict = {}
    nbytes = 0
    for out_name in sorted(want):
        src, sl = want[out_name]
        rows = cols = None
        if sl is not None:
            dim, s = sl
            rows, cols = (s, None) if dim == 0 else (None, s)
        t, qtype = source.read(src, rows, cols)
        if t.dim() == 1:
            t = t.to(dtype) if t.is_floating_point() else t
            n = t.numel() * t.element_size()
            val = t
        else:
            n = t.numel() * t.element_size()
            val = QuantLinear(t, qtype)
        if budget is not None and nbytes + n > budget:
            raise BudgetExceeded(f"load: {out_name} ({n / 1e6:.1f} MB) would bring tensor bytes to "
                                 f"{(nbytes + n) / 1e9:.3f} GB, over the budget of {budget / 1e9:.3f} GB")
        try:
            if isinstance(val, QuantLinear):
                val.data = val.data.to(device)
            else:
                val = val.to(device)
        except torch.cuda.OutOfMemoryError as e:
            e.tensor_name = out_name
            raise
        out[out_name] = val
        nbytes += n
    if spec.head_rows and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["model.embed_tokens.weight"]
    return out, nbytes


def load_slices(model_dir: str | Path, cfg: Qwen2Config, spec: LoadSpec, device: str, dtype: torch.dtype = torch.bfloat16,
                budget: int | None = None, log=print) -> tuple[dict[str, torch.Tensor], int]:
    """{name: tensor on device}, bytes loaded. Names keep the checkpoint's names; embedding and head hold only their
    rows (the model knows the offsets from the spec). A .gguf path loads quantized blocks as QuantLinear (2-D weights)."""
    model_dir = Path(model_dir)
    if model_dir.suffix == ".gguf":
        from .gguf_source import GgufSource
        return _load_gguf(GgufSource(model_dir), cfg, spec, device, dtype, budget)
    files = weight_files(model_dir)
    want = _wanted(files.keys(), cfg, spec)
    by_file: dict[Path, list[str]] = {}
    for out_name, (src, _) in want.items():
        by_file.setdefault(files[src], []).append(out_name)
    out: dict[str, torch.Tensor] = {}
    nbytes = 0
    for f, names in sorted(by_file.items()):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for out_name in sorted(names):
                src, sl = want[out_name]
                t = _take(sf, src, sl)
                t = t.to(dtype) if t.is_floating_point() else t
                n = t.numel() * t.element_size()
                if budget is not None and nbytes + n > budget:
                    raise BudgetExceeded(f"load: {out_name} ({n / 1e6:.1f} MB) would bring tensor bytes to "
                                         f"{(nbytes + n) / 1e9:.3f} GB, over the budget of {budget / 1e9:.3f} GB")
                try:
                    out[out_name] = t.to(device)
                except torch.cuda.OutOfMemoryError as e:
                    e.tensor_name = out_name
                    raise
                nbytes += n
    if spec.head_rows and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["model.embed_tokens.weight"]
    return out, nbytes


def load_rank(model_dir: str | Path, cfg: Qwen2Config, shard: Shard, device: str, dtype: torch.dtype = torch.bfloat16,
              log=print) -> tuple[dict[str, torch.Tensor], int]:
    """Session 3 entry point: an even Shard; rank 0 also holds embed_tokens, final norm and lm_head (whole)."""
    return load_slices(model_dir, cfg, LoadSpec.from_shard(cfg, shard), device, dtype, log=log)
