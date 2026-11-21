"""Compare greedy token ids of two sources prompt by prompt: identical count and first divergence.
Usage: python compare_tokens.py A B [--json]    A, B: a run folder (record.json), a record.json, a reference.json, or a llama_ref.py output.
Eval prompts are compared over their own length; the long prompt is listed apart (first divergence only)."""
import argparse
import json
from pathlib import Path


def load(p: str) -> dict[str, list[int]]:
    p = Path(p)
    if p.is_dir():
        p = p / "record.json"
    d = json.loads(p.read_text())
    if "prompts" in d and isinstance(d["prompts"], list):         # a hybrid_run record
        return {x["id"]: x["tokens"] for x in d["prompts"]}
    if "prompts" in d and isinstance(d["prompts"], dict):
        return {k: (v["tokens"] if isinstance(v, dict) else v) for k, v in d["prompts"].items()}
    raise ValueError(f"{p}: no prompts")


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(a, b):
    rows, ident, n = {}, 0, 0
    for k in a:
        if k not in b:
            continue
        d = first_div(a[k], b[k])
        long = k.startswith("longopen") or max(len(a[k]), len(b[k])) >= 200
        rows[k] = {"identical": d is None, "first_div": d, "len_a": len(a[k]), "len_b": len(b[k]), "long": long}
        if not long:
            n += 1
            ident += d is None
    return {"identical": ident, "of": n, "prompts": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", action="store_true")
    x = ap.parse_args()
    r = compare(load(x.a), load(x.b))
    if x.json:
        print(json.dumps(r))
    else:
        print(f"eval prompts identical: {r['identical']} of {r['of']}")
        for k, v in r["prompts"].items():
            print(f"  {k:14s} {'same' if v['identical'] else 'first divergence at ' + str(v['first_div']):28s} lengths {v['len_a']} {v['len_b']}{'  (long)' if v['long'] else ''}")
