"""Perplexity of the torch engine on a text slice, the way llama-perplexity scores it, to compare on the same GGUF file.
Usage: python ppl_engine.py <model.gguf|dir> <text.txt> [--chunk 128] [--chunks 4] [--device mps]
Token ids come from llama-tokenize (--no-bos; the Qwen models add no BOS), so both programs score the same ids.
Chunk k is tokens [k*chunk, (k+1)*chunk). Like llama-perplexity, tokens in the second half of a chunk are scored:
logits at positions j = chunk/2 .. chunk-2 predict token j+1 (63 predictions for a chunk of 128). PPL = exp(mean nll) over all
scored tokens of all chunks. Each chunk starts from an empty KV cache."""
import argparse
import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.loader import LoadSpec, load_slices  # noqa: E402
from nocturne.engines.torch_tp.model import Comm, Qwen2TP, rms_norm  # noqa: E402
from nocturne.engines.torch_tp.quant import linear  # noqa: E402
from nocturne.engines.torch_tp.shard import Shard  # noqa: E402

TOKENIZE = os.path.expanduser("~/dev/llama.cpp/build/bin/llama-tokenize")


def token_ids(model: str, text: str) -> list[int]:
    out = subprocess.run([TOKENIZE, "-m", model, "-f", text, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True, check=True)
    return ast.literal_eval(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ids-from", default=None, help="a GGUF file for llama-tokenize when the model is a safetensors directory")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = os.path.expanduser(a.model)
    ids = token_ids(os.path.expanduser(a.ids_from or model), a.text)
    need = a.chunk * a.chunks
    if len(ids) < need:
        sys.exit(f"text has {len(ids)} tokens, need {need}")
    cfg = load_config(model)
    sh = Shard(0, 1, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
    w, nbytes = load_slices(model, cfg, LoadSpec.from_shard(cfg, sh), a.device, dtype=torch.bfloat16)
    m = Qwen2TP(cfg, w, sh, a.device, a.chunk + 8, dtype=torch.bfloat16)
    nll, count, per_chunk = 0.0, 0, []
    first = a.chunk // 2
    for k in range(a.chunks):
        t = ids[k * a.chunk:(k + 1) * a.chunk]
        h = m.layers_forward(m.embed_tokens(t[:-1]), 0, Comm())
        x = rms_norm(h[first:], m.norm, cfg.rms_norm_eps)
        lg = linear(x, m.lm_head).float()
        lp = F.log_softmax(lg, dim=-1)
        tgt = torch.tensor(t[first + 1:], device=lp.device)
        c_nll = -lp.gather(1, tgt[:, None]).sum().item()
        nll += c_nll
        count += len(t[first + 1:])
        per_chunk.append(math.exp(c_nll / len(t[first + 1:])))
    ppl = math.exp(nll / count)
    res = {"model": model, "device": a.device, "chunk": a.chunk, "chunks": a.chunks, "scored_tokens": count, "ppl": ppl,
           "ppl_per_chunk": per_chunk, "ppl_first_chunk": per_chunk[0], "tensor_bytes": nbytes}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
