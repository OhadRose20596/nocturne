"""Greedy token ids from llama.cpp (llama-server on localhost) for the eval prompts, on the same GGUF the engine reads.
Usage: python llama_ref.py <model.gguf> <out.json> [--subset 8] [--tokens 32] [--long 1] [--fixed 256] [--ctx 4096]
Prompts are tokenized by transformers from the GGUF itself (the same ids the engine gets), sent as token ids, temperature 0.
The server binds 127.0.0.1 only. Output: {prompt id: token ids}."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from nocturne.engines.torch_tp.engine import chat_ids  # noqa: E402
from nocturne.engines.torch_tp.gguf_source import load_tokenizer  # noqa: E402

BIN = Path(os.path.expanduser("~/dev/llama.cpp/build/bin/llama-server"))
PROMPTS = REPO / "nocturne" / "bench" / "prompts"


def rows(name, n):
    out = [json.loads(l) for l in (PROMPTS / name).read_text().splitlines() if l.strip()]
    return [r for r in out if "id" in r][:n]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--long", type=int, default=1)
    ap.add_argument("--fixed", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=18099)
    a = ap.parse_args()
    tok = load_tokenizer(a.model)
    srv = subprocess.Popen([str(BIN), "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port), "-c", str(a.ctx), "-ngl", "99",
                            "--no-warmup", "--parallel", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    raise RuntimeError("llama-server exited: " + srv.stderr.read().decode()[-800:])
                time.sleep(1)
        res = {}
        for r, fixed in [(r, None) for r in rows("eval_set.jsonl", a.subset)] + [(r, a.fixed) for r in rows("eval_long.jsonl", a.long)]:
            ids = chat_ids(tok, [{"role": "user", "content": r["prompt"]}])
            n = fixed or a.tokens
            body = {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "cache_prompt": False, "return_tokens": True,
                    "ignore_eos": bool(fixed), "samplers": ["top_k"]}
            out = post(a.port, body)
            res[r["id"]] = out["tokens"]
            print(r["id"], len(out["tokens"]), flush=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "prompts": res}))
    finally:
        srv.terminate()
        srv.wait(timeout=30)


if __name__ == "__main__":
    main()
