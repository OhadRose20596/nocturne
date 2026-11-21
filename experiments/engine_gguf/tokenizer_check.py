"""Does the tokenizer built from a GGUF file give the ids of the Hugging Face tokenizer, and are the EOS ids the same?
Usage: python tokenizer_check.py <model.gguf> <hf dir> [--json]
Compares chat template ids for every eval prompt (eval_set and eval_long), plain text ids for a few strings, and the EOS sets."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.engine import chat_ids  # noqa: E402
from nocturne.engines.torch_tp.gguf_source import load_tokenizer  # noqa: E402

PROMPTS = REPO / "nocturne" / "bench" / "prompts"
TEXTS = ["Hello world", "  leading spaces and\ttabs", "naïve café 日本語 🙂", "1234567890 + 1 = ?", "def f(x):\n    return x ** 2\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("hf")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    g, h = load_tokenizer(os.path.expanduser(a.gguf)), load_tokenizer(os.path.expanduser(a.hf))
    rows = []
    for name in ("eval_set.jsonl", "eval_long.jsonl"):
        rows += [r for r in (json.loads(l) for l in (PROMPTS / name).read_text().splitlines() if l.strip()) if "id" in r]
    chat_diff = [r["id"] for r in rows if chat_ids(g, [{"role": "user", "content": r["prompt"]}]) != chat_ids(h, [{"role": "user", "content": r["prompt"]}])]
    text_diff = [t for t in TEXTS if g.encode(t, add_special_tokens=False) != h.encode(t, add_special_tokens=False)]
    eg, eh = set(load_config(os.path.expanduser(a.gguf)).eos_token_ids), set(load_config(os.path.expanduser(a.hf)).eos_token_ids)
    res = {"gguf": a.gguf, "hf": a.hf, "prompts": len(rows), "chat_template_ids_differ": chat_diff, "plain_text_ids_differ": text_diff,
           "eos_gguf": sorted(eg), "eos_hf": sorted(eh), "eos_equal": eg == eh}
    print(json.dumps(res) if a.json else json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
