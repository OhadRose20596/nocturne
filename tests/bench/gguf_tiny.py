"""Builds tiny random models (qwen2, llama with Llama 3 rope scaling, qwen3) as safetensors and as GGUF made by llama.cpp's own
converter (bf16, q8_0) and quantizer (Q4_K_M). Cached under ~/.cache/nocturne-tests, keyed by the converter version.
The converter wants sentencepiece on import for a path these models never take, so a stub module is put on its PYTHONPATH.
Tokenizer files come from the local Qwen2.5 0.5B directory (only their presence matters to the converter)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA = Path(os.path.expanduser("~/dev/llama.cpp"))
CONVERT = LLAMA / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA / "build" / "bin" / "llama-quantize"
TOK_SRC = Path(os.path.expanduser("~/models/hf/Qwen2.5-0.5B-Instruct"))
CACHE = Path(os.path.expanduser("~/.cache/nocturne-tests/tiny_gguf_v1"))
ARCHS = ("qwen2", "llama", "qwen3")


def available() -> bool:
    return CONVERT.exists() and QUANTIZE.exists() and TOK_SRC.exists()


def _config(arch: str):
    from transformers import LlamaConfig, Qwen2Config, Qwen3Config
    common = dict(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=4096, bos_token_id=151643, eos_token_id=151645, head_dim=64)
    if arch == "llama":
        return LlamaConfig(**common, rope_theta=500000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                           rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                         "original_max_position_embeddings": 2048})
    if arch == "qwen2":
        c = {k: v for k, v in common.items() if k != "head_dim"}
        return Qwen2Config(**c, rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=False)
    return Qwen3Config(**common, rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)


def _run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def build(arch: str) -> dict:
    """{'hf': dir, 'bf16': gguf, 'q8_0': gguf, 'q4_k_m': gguf} for one architecture."""
    import torch
    from transformers import AutoModelForCausalLM
    d = CACHE / arch
    out = {"hf": d / "hf", "bf16": d / "bf16.gguf", "q8_0": d / "q8_0.gguf", "q4_k_m": d / "q4_k_m.gguf"}
    if all(Path(p).exists() for p in out.values()):
        return out
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    torch.manual_seed({"qwen2": 1, "llama": 2, "qwen3": 3}[arch])
    m = AutoModelForCausalLM.from_config(_config(arch)).to(torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.uniform_(0.6, 1.4)
            elif n.endswith(".bias"):
                p.normal_(0, 0.2)
            elif "embed" in n or "lm_head" in n:
                p.normal_(0, 0.5)
            else:
                p.normal_(0, 0.08)
    m = m.to(torch.bfloat16)
    m.save_pretrained(out["hf"])
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(TOK_SRC / f, out["hf"] / f)
    stub = d / "stub"
    stub.mkdir()
    (stub / "sentencepiece.py").write_text("class SentencePieceProcessor:\n    pass\n")
    env = {**os.environ, "PYTHONPATH": str(stub)}
    for t in ("bf16", "q8_0"):
        _run([sys.executable, str(CONVERT), str(out["hf"]), "--outfile", str(out[t]), "--outtype", t], env)
    _run([str(QUANTIZE), str(out["bf16"]), str(out["q4_k_m"]), "Q4_K_M"])
    return out
