"""V3 table of session 10: one row per run of experiments/session10_cuda_pair/runs, from record.json.
Usage: python experiments/session10_cuda_pair/v3_table.py [run id prefix]   (prints Markdown; compares tokens with the recorded part 1 run of the same placement)"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent / "engine_gguf" / "runs"
SAME_PLACEMENT = {"Q8_7b_pp3.yaml": "C1_q8_7b_pp3_r1", "Q8_7b_tp3.yaml": "C2_q8_7b_tp3_r1", "Q8_7b_hybrid.yaml": "C3_q8_7b_hybrid_r1"}


def tokens(rec):
    return {p["id"]: p["tokens"] for p in rec.get("prompts") or []}


def against(rec, other_id):
    o = json.loads((G / other_id / "record.json").read_text())
    a, b = tokens(rec), tokens(o)
    ev = [p["id"] for p in rec["prompts"] if p["kind"] == "eval"]
    lg = [p["id"] for p in rec["prompts"] if p["kind"] == "long"]
    return f"{sum(a[k] == b[k] for k in ev)}/{len(ev)}, long " + ("identical" if all(a[k] == b[k] for k in lg) else "differs")


def mean_ms(rec):
    v = [p["ms_per_step"] for p in rec["prompts"] if p["ms_per_step"]]
    return sum(v) / len(v)


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else ""
    print("| run | placement | weights | timeline | identical to the one rank run (8; long first divergence) | same tokens as the recorded part 1 run | "
          "frames per step measured / predicted | bytes per step, recorder / counters | shipped GB (from cache GB) | mean ms per step |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted(HERE.glob(f"runs/{pre}*/record.json")):
        r = json.loads(d.read_text())
        if r.get("outcome") != "completed":
            print(f"| {r['run_id']} | {Path(r['placement']['file']).name} | | | outcome: {r.get('outcome')} | | | | | |")
            continue
        pl = Path(r["placement"]["file"]).name
        c = next(iter(r["correctness"].values()))
        ship = r.get("shipping")
        if ship:
            s = sum(x["shipped_bytes"] for x in ship["ranks"].values()) / 1e9
            k = sum(x.get("cache_bytes", 0) for x in ship["ranks"].values()) / 1e9
            sh = f"{s:.3f} ({k:.3f})"
        else:
            sh = "-"
        tl = r.get("timeline") or {}
        by = f"{tl['decode_bytes_per_step']:.0f} / {r['frames']['bytes_per_step']:.0f}" if tl.get("decode_bytes_per_step") is not None else f"- / {r['frames']['bytes_per_step']:.0f}"
        same = against(r, SAME_PLACEMENT[pl]) if pl in SAME_PLACEMENT else "-"
        print(f"| {r['run_id']} | {pl} | {'shipped' if ship else 'local'} | {'on' if r['args'].get('timeline') else 'off'} | "
              f"{c['identical']}/{c['of']}; {(list(c['long_first_div'].values()) or ['no long prompt'])[0]} | {same} | "
              f"{r['frames']['measured_per_step']:.1f} / {r['frames']['predicted']['total']} | {by} | {sh} | {mean_ms(r):.1f} |")


if __name__ == "__main__":
    main()
