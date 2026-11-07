"""Stage 1 and 2 check: our engine's greedy tokens against transformers `generate` with the same chat template.
Usage: python -m nocturne.engines.torch_tp compare --model DIR [--world 1|2 --peer host:port] [--n 20] [--max-tokens 32]
Prints one line per prompt (match or first divergence) and a summary; exit code 1 when any prompt differs."""
import json
import sys
import time
from pathlib import Path

import torch

from .engine import Rank0, chat_ids

EVAL_SET = Path(__file__).resolve().parents[2] / "bench" / "prompts" / "eval_set.jsonl"


def load_prompts(path: Path, n: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:n]


def reference_tokens(model_dir: str, tok, ids: list[int], max_tokens: int, device: str, model=None) -> list[int]:
    from transformers import AutoModelForCausalLM
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device).eval()
    inp = torch.tensor([ids], device=device)
    with torch.inference_mode():
        out = model.generate(inp, max_new_tokens=max_tokens, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out[0, len(ids):].tolist(), model


def run(model_dir: str, world: int, peer: str | None, n: int, max_tokens: int, device: str | None, max_seq_len: int) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    eng = Rank0(model_dir, world, max_seq_len, device, peer)
    ref_model = None
    rows = load_prompts(EVAL_SET, n)
    bad = 0
    tps = []
    for r in rows:
        ids = chat_ids(tok, [{"role": "user", "content": r["prompt"]}])
        mt = min(max_tokens, r.get("max_tokens", max_tokens))
        res = eng.generate(ids, mt)
        ours = res["tokens"]
        if ours and ours[-1] in eng.cfg.eos_token_ids:
            ours_cmp = ours
        else:
            ours_cmp = ours
        ref, ref_model = reference_tokens(model_dir, tok, ids, mt, eng.device, ref_model)
        # transformers stops after the EOS token and includes it; ours includes it too
        same = ours_cmp == ref
        tps.append(res["timings"]["predicted_per_second"] or 0.0)
        if not same:
            bad += 1
            i = next((k for k, (a, b) in enumerate(zip(ours_cmp, ref)) if a != b), min(len(ours_cmp), len(ref)))
            print(f"{r['id']}: DIFF at token {i}: ours {tok.decode(ours_cmp)!r} ref {tok.decode(ref)!r}", flush=True)
        else:
            print(f"{r['id']}: match ({len(ours)} tokens, {res['timings']['predicted_per_second'] or 0:.1f} tok/s"
                  + (f", frames {res['frames']['allreduce_frames']}" if res.get("frames") else "") + ")", flush=True)
    print(f"\n{len(rows) - bad}/{len(rows)} identical to transformers generate; median decode {sorted(tps)[len(tps) // 2]:.1f} tok/s; "
          f"world {world}, device {eng.device}, torch {torch.__version__}")
    if world > 1:
        print("metrics:", json.dumps(eng.metrics()))
    eng.shutdown()
    return 1 if bad else 0
