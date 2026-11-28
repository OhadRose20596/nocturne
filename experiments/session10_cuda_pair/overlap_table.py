"""Tables of session 10 part 3 (overlap), from record.json of runs under experiments/session10_cuda_pair/runs.
Usage: python experiments/session10_cuda_pair/overlap_table.py <run id prefix> [more prefixes]   (prints Markdown: the runs, then the options-off gate)
Run it with the engine's Python from the repository root. Comparisons are computed here from the raw files of the runs (tokens in
record.json, first step logits in first_logits.npz), against every reference the run names."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nocturne.engines.torch_tp.hybrid_run import compare, load_reference  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
# options off: the recorded part 1 run of the same placement (VERIFY.md), same recorder setting
PART1 = {("S10_q8_15b_pp3.yaml", True): "V3codex_q8_15b_pp3_tl_r1", ("S10_q8_15b_tp3.yaml", True): "V3post_q8_15b_tp3_tl_r4",
         ("S10_q8_15b_tp3.yaml", False): "V3codex_q8_15b_tp3_notl_r1", ("S10_q8_15b_hybrid.yaml", True): "V3final_q8_15b_hybrid_tl_r1",
         ("Q8_7b_pp3.yaml", True): "V3final_q8_7b_pp3_tl_r1", ("Q8_7b_tp3.yaml", True): "V3final_q8_7b_tp3_tl_r1",
         ("Q8_7b_hybrid.yaml", True): "V3codex_q8_7b_hybrid_tl_r1"}


def load(rid):
    return json.loads((RUNS / rid / "record.json").read_text())


def options(r):
    o = r.get("overlap") or {}
    parts = []
    if o.get("slots", 1) > 1:
        parts.append(f"slots {o['slots']}")
    if o.get("prefill_chunk"):
        parts.append(f"chunk {o['prefill_chunk']}")
    if o.get("tp_interleave", 1) > 1:
        parts.append("pair")
    if o.get("async_send"):
        modes = sorted({(x.get("async_send") or {}).get("copy") or "?" for x in ranks(r)})
        parts.append("async (" + ", ".join(modes) + ")")
    return ", ".join(parts) or "off"


def ranks(r):
    return list(r["ranks"].values()) if isinstance(r.get("ranks"), dict) else (r.get("ranks") or [])


def against(c):
    lg = list(c["long_first_div"].values())
    return f"{c['identical']}/{c['of']}; {', '.join(str(x) for x in lg) if lg else 'no long prompt'}"


def comparisons(r):
    lg = dict(np.load(RUNS / r["run_id"] / "first_logits.npz"))
    return [compare(r["prompts"], lg, *load_reference(ref)) for ref in r["args"].get("reference") or []]


def shares(r):
    o = (r.get("timeline") or {}).get("overlap")
    if not o:
        return "-", "-", "-"
    pr = o["per_rank"]
    busy = ", ".join(f"{k} {100 * v['busy_share']:.0f}" for k, v in pr.items())
    over = ", ".join(f"{k} {100 * v['overlap_share']:.1f} ({100 * v['overlap_out_share']:.1f})" for k, v in pr.items())
    dev = ", ".join(f"{k}: {100 * v['busy_share']:.0f}" for k, v in o["per_device"].items())
    return busy, over, dev + ("" if not o["unmatched_links"] else f"; UNMATCHED {o['unmatched_links']}")


def main():
    recs = []
    for pre in sys.argv[1:] or [""]:
        for d in sorted(RUNS.glob(f"{pre}*/record.json")):
            recs.append(json.loads(d.read_text()))
    print("| run | placement | options | identical to the one rank run (8; long first divergence) | against the options-off run of the placement "
          "(8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | "
          "bytes per decode step | tokens per s | busy % per rank | overlap % per rank (sent part) | busy % per device |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        pl = Path(r["placement"]["file"]).name
        if r.get("outcome") != "completed":
            print(f"| {r['run_id']} | {pl} | {options(r)} | outcome: {r.get('outcome')} | | | | | | | | |")
            continue
        cs = comparisons(r)
        off = (f"{cs[1]['identical']}/{cs[1]['of']}; {'identical' if cs[1]['long_identical'] else 'first divergence ' + str(list(cs[1]['long_first_div'].values()))}; "
               f"{cs[1]['logit_maxdiff']:.4g}") if len(cs) > 1 else "-"
        f = r["frames"]
        busy, over, dev = shares(r)
        t = r.get("throughput") or {}
        print(f"| {r['run_id']} | {pl} | {options(r)} | {against(cs[0])} | {off} | {f['measured_per_step']:.1f} / {f['predicted']['total']} | "
              f"{f.get('prefill_measured')} / {f.get('prefill_predicted')} | {f['bytes_per_step']:.0f} | {t.get('tokens_per_s', 0):.2f} | {busy} | {over} | {dev} |")
    gate = [r for r in recs if r.get("outcome") == "completed" and not (r.get("overlap") or {}).get("on") and not (r.get("overlap") or {}).get("async_send")
            and (Path(r["placement"]["file"]).name, bool(r["args"].get("timeline"))) in PART1]
    if gate:
        print("\n| options-off run | recorded part 1 run | same tokens, 9 prompts | frames per step | bytes per step | frames and bytes per link per step |")
        print("|---|---|---|---|---|---|")
        for r in gate:
            old = load(PART1[(Path(r["placement"]["file"]).name, bool(r["args"].get("timeline")))])
            tok = [p["tokens"] for p in r["prompts"]] == [p["tokens"] for p in old["prompts"]]
            a, b = r["frames"], old["frames"]
            print(f"| {r['run_id']} | {old['run_id']} | {'yes' if tok else 'NO'} | {a['measured_per_step']:.1f} = {b['measured_per_step']:.1f} | "
                  f"{a['bytes_per_step']:.0f} {'=' if a['bytes_per_step'] == b['bytes_per_step'] else '!='} {b['bytes_per_step']:.0f} | "
                  f"{'identical' if a['per_link_per_step'] == b['per_link_per_step'] else 'DIFFERENT'} |")


if __name__ == "__main__":
    main()
