#!/usr/bin/env python3
"""Reads results/*.json and prints the matrix as Markdown: one 4x4 table per (model, network, requester host, engine),
plus a BASELINE line. Cell text: operational success, quality_pass against the baseline (separately), accuracy, ms per
decode step pooled over prompts (schema v3, the number to cite) with blocking syncs per step, then S, B, memory, and
the old per prompt median decode TPS labelled as such. results/derived_v3.json (python -m nocturne.bench.derive_v3)
supplies the per step comm values of v1 and v2 records. Records listed in results/EXCLUDE.yaml (id: reason) are left
out and listed at the end.
Usage: python -m nocturne.bench.report [--results DIR] [--model NAME] [--network wifi|wired|tailscale] [--include-excluded]"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from . import config, derive_v3, record

LEGEND = ("ms/step: sum(predicted_ms) / sum(predicted_n - 1) over prompts, pooled, with its token weighted tok/s; syncs: blocking "
          "waits per decode step (llama.cpp get_tensor, torch_tp all-reduces plus STEP); S: RPC commands or frames per token; "
          "B: wire bytes per token; dec median: the per prompt median of predicted_n / predicted_ms used until 2026-11-02, "
          "which counts the first token as a step and overstates speed on short prompts.\n")

ROWS = ["A", "B", "C", "D"]
COLS = [1, 2, 3, 4]
EXCLUDE_NAME = "EXCLUDE.yaml"


def fmt(x, nd=2, unit=""):
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{nd}f}{unit}"
    return f"{x}{unit}"


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def load_exclusions(results_dir: Path) -> dict[str, str]:
    """{record id: reason}. The file is hand written; the records themselves are never deleted or edited."""
    p = Path(results_dir) / EXCLUDE_NAME
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return {str(k): str(v) for k, v in (data.get("exclude") or {}).items()}


def requester_host(r: dict) -> str:
    return next((p["host"] for p in r["peers"] if p["role"] == "requester"), "?")


def _v3(r: dict, derived: dict, block: str, key: str):
    """A v3 value from the record, else from derived_v3.json for an older record."""
    v = r.get(block, {}).get(key)
    if v is None:
        v = ((derived.get(r["id"]) or {}).get(block) or {}).get(key)
    return v


def summarize(recs: list[dict], derived: dict | None = None) -> dict:
    """Aggregate several runs of one cell into one dict of medians. `ok` is operational success; `quality` counts
    quality_pass true, false and null separately (I025 9): a run can be ok and still fail quality."""
    derived = derived or {}
    peak = {}
    for r in recs:
        for h, v in r["memory_buffers_gb"].items():
            if v is not None:
                peak[h] = max(peak.get(h, 0.0), v)
    q = [r.get("quality_pass") for r in recs]
    return {
        "runs": len(recs),
        "ok": sum(1 for r in recs if r["success"]),
        "quality_pass": sum(1 for v in q if v is True),
        "quality_fail": sum(1 for v in q if v is False),
        "quality_unknown": sum(1 for v in q if v is None),
        "failure_classes": sorted({r["failure_class"] for r in recs if r["failure_class"]}),
        "task_accuracy": med([r["correctness"]["task_accuracy"] for r in recs]),
        "S": med([r["comm"]["rpc_commands_per_token"] for r in recs]),
        "B": med([r["comm"]["bytes_per_token"] for r in recs]),
        "peak_gb": max(peak.values()) if peak else None,
        "decode_tps": med([r["timing"]["decode_tps_median"] for r in recs]),
        "ms_step": med([_v3(r, derived, "timing", "decode_ms_per_step_pooled") for r in recs]),
        "tps_pooled": med([_v3(r, derived, "timing", "decode_tps_pooled") for r in recs]),
        "syncs": med([_v3(r, derived, "comm", "blocking_syncs_per_step") for r in recs]),
    }


def cell_text(s: dict | None) -> str:
    if s is None:
        return "not run"
    fc = f" ({','.join(s['failure_classes'])})" if s["failure_classes"] else ""
    graded = s["quality_pass"] + s["quality_fail"]
    quality = f"quality {s['quality_pass']}/{graded}" if graded else "quality n/a"
    if s["quality_unknown"] and graded:
        quality += f" (+{s['quality_unknown']} no baseline)"
    return (f"ok {s['ok']}/{s['runs']}{fc}<br>{quality}, acc {fmt(s['task_accuracy'])}<br>"
            f"ms/step {fmt(s['ms_step'], 1)} ({fmt(s['tps_pooled'], 1, ' t/s')} pooled), syncs {fmt(s['syncs'], 1)}<br>"
            f"S {fmt(s['S'], 1)} B {fmt(s['B'], 0)}<br>mem {fmt(s['peak_gb'], 1, ' GB')}; dec median {fmt(s['decode_tps'], 1, ' t/s')} (old)")


def group(recs: list[dict]):
    """Cells keyed by (row, col) under (model, network, requester host, engine); a variant cell such as A2b or B1t
    (cell.name longer than row+col) is keyed by its name and rendered as its own row under the matrix."""
    g = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = (r["model"]["name"], r["network"]["type"], requester_host(r), r["engine"].get("name") or "?")
        name = r["cell"].get("name") or f"{r['cell']['row']}{r['cell']['col']}"
        base = f"{r['cell']['row']}{r['cell']['col']}"
        g[key][(r["cell"]["row"], r["cell"]["col"]) if name in (base, "BASELINE") else name].append(r)
    return g


def render(recs: list[dict], model: str | None = None, network: str | None = None, excluded: dict | None = None,
           derived: dict | None = None) -> str:
    excluded = excluded or {}
    derived = derived or {}
    kept = [r for r in recs if r["id"] not in excluded]
    out = []
    cells_cfg = config.load_cells()
    grouped = group(kept)
    # BASELINE records carry network "local" whatever network the cells ran on: index them by (model, host, engine)
    # and show the matching one above every section of that triple.
    baselines = {(m, host, engine): cells[("BASELINE", 0)] for (m, _, host, engine), cells in grouped.items()
                 if ("BASELINE", 0) in cells}
    for (m, net, host, engine), cells in sorted(grouped.items()):
        if model and m != model or network and net != network:
            continue
        base = baselines.get((m, host, engine))
        only_baseline = set(cells) <= {("BASELINE", 0)}
        if only_baseline and any(k[0] == m and k[2] == host and k[3] == engine and k[1] != net for k in grouped):
            continue            # its cells are in another section, where the BASELINE line is shown
        out.append(f"### {m} over {net}, requester {host}, engine {engine}\n")
        if base:
            s = summarize(base, derived)
            out.append(f"BASELINE ({base[0]['network']['type']}): ok {s['ok']}/{s['runs']}, acc {fmt(s['task_accuracy'])}, "
                       f"ms/step {fmt(s['ms_step'], 1)} ({fmt(s['tps_pooled'], 1)} t/s pooled), dec median {fmt(s['decode_tps'], 1)} t/s (old), "
                       f"mem {fmt(s['peak_gb'], 1)} GB\n")
        else:
            out.append("BASELINE: none for this engine on this host (quality_pass is null for every cell below)\n")
        if only_baseline:
            out.append("")
            continue
        out.append("| partition \\ decode | 1 single | 2 speculative | 3 outline | 4 independent |")
        out.append("|---|---|---|---|---|")
        for row in ROWS:
            tds = [cell_text(summarize(cells[(row, c)], derived) if (row, c) in cells else None) for c in COLS]
            out.append(f"| {row} | " + " | ".join(tds) + " |")
        for name in sorted(k for k in cells if isinstance(k, str)):
            col = int(name[1])
            tds = [cell_text(summarize(cells[name], derived)) if c == col else "" for c in COLS]
            try:
                note = cells_cfg["cells"][name].get("notes", "").split(".")[0]
            except KeyError:
                note = "cell no longer in cells.yaml"
            out.append(f"| {name} ({note}) | " + " | ".join(tds) + " |")
        out.append("")
    if out:
        out.insert(0, LEGEND)
    present = {r["id"] for r in recs}
    listed = [(i, why) for i, why in sorted(excluded.items()) if i in present]
    if listed:
        out.append(f"Excluded ({len(listed)}, results/{EXCLUDE_NAME}; the files are kept):\n")
        out += [f"- `{i}`: {why}" for i, why in listed]
        out.append("")
    return "\n".join(out) if out else "no records\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=record.RESULTS_DIR)
    ap.add_argument("--model")
    ap.add_argument("--network")
    ap.add_argument("--include-excluded", action="store_true", help=f"ignore {EXCLUDE_NAME}")
    a = ap.parse_args(argv)
    excluded = {} if a.include_excluded else load_exclusions(a.results)
    print(render(record.read_all(a.results), a.model, a.network, excluded, derive_v3.load(a.results)))


if __name__ == "__main__":
    main()
