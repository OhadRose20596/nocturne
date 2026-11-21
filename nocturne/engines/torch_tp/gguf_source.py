"""GGUF as a weight source (engine GGUF). Reads the header with `gguf.GGUFReader` (gguf-py of llama.cpp), maps tensor names
to engine names, and returns the raw quantized blocks of a slice. The weights are never dequantized here.
TensorSource is the interface the loader uses: today a local file, later a stream of blocks from the coordinator.
Engine shape is [out, in] (GGUF stores ne = [in, out]); a quantized tensor is returned as uint8 [rows, blocks, type_size]."""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .arch import engine_name, get_arch
from .config import ModelConfig
from .quant import BLOCK, DENSE, UnsupportedQuant, check_supported

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")     # memmap views are copied by .to(device)


EOG_TEXTS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>")


@dataclass(frozen=True)
class TensorMeta:
    gguf_name: str
    qtype: str
    shape: tuple[int, ...]       # engine order: [out, in] or [n]
    nbytes: int


class TensorSource(Protocol):
    def meta(self) -> dict[str, TensorMeta]: ...
    def read(self, name: str, rows: slice | None = None, cols: slice | None = None) -> tuple[torch.Tensor, str]: ...


class GgufSource:
    def __init__(self, path: str | Path):
        import gguf
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")
        self.reader = gguf.GGUFReader(str(self.path))
        self.arch_name = self._str("general.architecture")
        self.arch = get_arch(self.arch_name)
        self._tensors = {}
        self._meta = {}
        for t in self.reader.tensors:
            en = engine_name(t.name)
            if en is None:
                self._tensors[t.name] = t
                continue
            if en in self._meta:
                raise ValueError(f"two GGUF tensors map to {en}: {self._meta[en].gguf_name} and {t.name}")
            qtype = t.tensor_type.name
            check_supported(qtype)
            self._tensors[en] = t
            self._meta[en] = TensorMeta(t.name, qtype, tuple(int(x) for x in reversed(t.shape)), int(t.n_bytes))

    def _field(self, key: str):
        f = self.reader.fields.get(key)
        return None if f is None else f.contents()

    def _str(self, key: str) -> str:
        v = self._field(key)
        if v is None:
            raise KeyError(f"GGUF metadata key {key} missing in {self.path}")
        return str(v)

    def _num(self, key: str, default=None):
        v = self._field(f"{self.arch_name}.{key}")
        if v is None:
            if default is None:
                raise KeyError(f"GGUF metadata key {self.arch_name}.{key} missing in {self.path}")
            return default
        return v

    def meta(self) -> dict[str, TensorMeta]:
        return self._meta

    def config(self) -> ModelConfig:
        a = self.arch_name
        n_heads = int(self._num("attention.head_count"))
        hidden = int(self._num("embedding_length"))
        hd = int(self._num("attention.key_length", hidden // n_heads))
        rope_dim = int(self._num("rope.dimension_count", hd))
        if rope_dim != hd:
            raise ValueError(f"rope dimension count {rope_dim} differs from head size {hd}: partial rotary is not supported")
        st = self._field(f"{a}.rope.scaling.type")
        if st not in (None, "none", "llama3"):
            raise ValueError(f"rope scaling type {st!r} is not supported")
        factors = None
        if "rope_freqs.weight" in self._tensors:
            rf = self._tensors["rope_freqs.weight"]
            factors = tuple(float(x) for x in np.asarray(rf.data).reshape(-1))
        emb = self._meta["model.embed_tokens.weight"]
        eos = [int(self._field("tokenizer.ggml.eos_token_id"))] if self._field("tokenizer.ggml.eos_token_id") is not None else []
        for k in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"):
            v = self._field(k)
            if v is not None and int(v) not in eos:
                eos.append(int(v))
        toks = self._field("tokenizer.ggml.tokens")
        if toks is not None:                    # end of generation tokens by their text, as llama.cpp does
            for i, t in enumerate(toks):
                if str(t) in EOG_TEXTS and i not in eos:
                    eos.append(i)
        return ModelConfig(
            hidden_size=hidden, intermediate_size=int(self._num("feed_forward_length")), num_hidden_layers=int(self._num("block_count")),
            num_attention_heads=n_heads, num_key_value_heads=int(self._num("attention.head_count_kv", n_heads)),
            vocab_size=emb.shape[0], rms_norm_eps=float(self._num("attention.layer_norm_rms_epsilon", 1e-6)),
            rope_theta=float(self._num("rope.freq_base", 10000.0)), tie_word_embeddings="lm_head.weight" not in self._meta,
            max_position_embeddings=int(self._num("context_length", 32768)), eos_token_ids=tuple(sorted(eos)),
            arch_name=a, head_dim=hd, rope_factors=factors)

    def _unpermute_index(self, name: str, n_heads: int, rows: int) -> torch.Tensor:
        """take[k] = stored row of engine row k: undoes LlamaModel.permute (conversion/llama.py)."""
        half = rows // n_heads // 2
        s = torch.arange(rows).reshape(n_heads, half, 2)
        return s.transpose(1, 2).reshape(-1)

    def read(self, name: str, rows: slice | None = None, cols: slice | None = None) -> tuple[torch.Tensor, str]:
        """Rows are output rows of the engine tensor (HF order), cols are input columns in elements and must sit on block
        boundaries. Returns (tensor on cpu, qtype)."""
        m = self._meta[name]
        t = self._tensors[name]
        qtype = m.qtype
        cfg = self.config() if self.arch.gguf_rope_permuted else None
        data = np.asarray(t.data)
        if qtype == "BF16":                      # gguf-py hands BF16 over as raw bytes
            data = np.ascontiguousarray(data).reshape(-1).view(np.int16).reshape(m.shape)
        elif qtype in DENSE:
            data = data.reshape(m.shape)
        else:
            bs, ts = BLOCK[qtype]
            out_f, in_f = m.shape
            if in_f % bs:
                raise ValueError(f"{name}: {in_f} input columns is not a multiple of the {qtype} block size {bs}")
            data = data.reshape(out_f, in_f // bs, ts)
        n = m.shape[0]
        if rows is not None:
            r0, r1, _ = rows.indices(n)
            if self.arch.gguf_rope_permuted and cfg is not None and name.endswith(("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias")):
                nh = cfg.num_attention_heads if "q_proj" in name else cfg.num_key_value_heads
                idx = self._unpermute_index(name, nh, n)[r0:r1].numpy()
                data = data[idx]
            else:
                data = data[r0:r1]
        elif self.arch.gguf_rope_permuted and cfg is not None and name.endswith(("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias")):
            nh = cfg.num_attention_heads if "q_proj" in name else cfg.num_key_value_heads
            data = data[self._unpermute_index(name, nh, n).numpy()]
        if cols is not None:
            c0, c1, _ = cols.indices(m.shape[1])
            if qtype in DENSE:
                data = data[:, c0:c1]
            else:
                bs = BLOCK[qtype][0]
                if c0 % bs or c1 % bs:
                    raise ValueError(f"{name}: column range [{c0}, {c1}) is not aligned to the {qtype} block of {bs} columns")
                data = data[:, c0 // bs:c1 // bs]
        data = np.ascontiguousarray(data)
        if qtype == "BF16":
            return torch.from_numpy(data).view(torch.bfloat16), qtype
        return torch.from_numpy(data), qtype


def weight_meta(path: str) -> dict | None:
    """Tensor metadata of a .gguf file (for the planner's byte accounting); None for a safetensors directory."""
    return GgufSource(path).meta() if str(path).endswith(".gguf") else None


def load_tokenizer(path: str):
    """Tokenizer for a safetensors directory or a .gguf file (built by transformers from the GGUF metadata)."""
    from transformers import AutoTokenizer
    p = Path(os.path.expanduser(path))
    if p.suffix == ".gguf":
        return AutoTokenizer.from_pretrained(str(p.parent), gguf_file=p.name)
    return AutoTokenizer.from_pretrained(str(p))
