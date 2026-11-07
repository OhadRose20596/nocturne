"""Session 5 memory edge table from results/*.json (records tagged c<ctx>[bal]) and their logs/<id>/server.log.
Usage: python -m nocturne.bench.derive_edge [--results DIR] [--out experiments/session5_memory_edge]"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from . import record
from .probes import memlog

ROOT = Path(__file__).resolve().parents[2]
TAG = re.compile(r"_c(\d+)(bal|prop)?_r\d+$")
DEVLINE = re.compile(r"-\s+(MTL\d+|RPC\d+)\s*:\s*(.+?)\s*\((\d+) MiB, \d+ MiB free\)")
WDEV = re.compile(r"MTL\d+: .*?\((\d+) MiB, \d+ MiB free\)")
NODES = ("m5", "air")
PARTITION_ORDER = ["local M5", "local Air", "PP default", "PP balanced", "TP", "TP 3:2", "TP torch"]


def partition(r: dict, var) -> str:
    c = r["cell"]["name"]
    if c == "BASELINE":
        req = next(p for p in r["peers"] if p["role"] == "requester")
        return "local M5" if req["host"] == "MMF4FWRG4R" else "local Air"
    if c == "A1":
        return "PP balanced" if var == "bal" else "PP default"
    if c == "B1":
        return "TP 3:2" if var == "prop" else "TP"
    if c == "B1t":
        return "TP torch"
    return c


def capacity_mib(r: dict, logs: Path) -> dict:
    """Metal working set per node, MiB, as the engine reports it (requester device line, worker device line)."""
    out = {}
    d = logs / r["id"]
    s = (d / "server.log").read_text(errors="replace") if (d / "server.log").exists() else ""
    for m in DEVLINE.finditer(s):
        out["m5" if m.group(1).startswith("MTL") else "air"] = int(m.group(3))
    w = d / "worker_air.log"
    if w.exists():
        m = WDEV.search(w.read_text(errors="replace"))
        if m:
            out["air"] = int(m.group(1))
    for p in r["peers"]:
        name = "m5" if p["role"] == "requester" and partition(r, None) != "local Air" else "air"
        if name not in out and p.get("metal_working_set_gb"):
            out[name] = int(p["metal_working_set_gb"] * 1000)
    return out


def corrected_breakdown(r: dict, logs: Path) -> dict:
    """Tensor mode records written before the Meta() share fix carry the requester's share on every node; recompute
    from the run's own server.log with the -ts shares of the record's flags."""
    bd = r.get("memory_breakdown_gb") or {}
    log = logs / r["id"] / "server.log"
    if any("Meta()" in l for l in r.get("memory_log_lines") or []) and log.exists():
        workers = [p["host"] for p in r["peers"] if p["role"] == "worker"]
        bd, _ = memlog.breakdown(log.read_text(errors="replace"), "m5", {}, ["m5"] + ["air" for _ in workers],
                                 memlog.ts_shares(r["engine"].get("flags")))
    return bd


def oom_evidence(r: dict, logs: Path) -> bool:
    d = logs / r["id"]
    return any(memlog.OOM.search(p.read_text(errors="replace")) for p in d.glob("*.log")) if d.exists() else False


def node_view(r: dict, bd: dict | None = None) -> dict:
    """GiB per node: device total (model + kv + compute + output, CPU buffers excluded) from the record breakdown.
    In TP mode the engine reports one Meta() device; the breakdown gives each node its per device share."""
    bd = bd if bd is not None else (r.get("memory_breakdown_gb") or {})
    out = {}
    for h, d in bd.items():
        key = "m5" if h.lower() in ("m5", "mmf4fwrg4r") else ("air" if h.lower() in ("air", "mac", "ohads-macbook-air-2") else h)
        if partition(r, None) == "local Air":
            key = "air"
        out[key] = {k: d.get(k, 0.0) for k in ("model", "kv", "compute", "output", "cpu")}
        out[key]["gpu"] = round(sum(d.get(k, 0.0) for k in ("model", "kv", "compute", "output")), 3)
    return out


def rows(results: Path) -> list[dict]:
    logs = results / "logs"
    groups: dict = {}
    for r in record.read_all(results):
        m = TAG.search(r["id"])
        if not m:
            continue
        ctx = r["setup"].get("ctx") or int(m.group(1))
        key = (r["model"]["name"], r["model"].get("quant", ""), ctx, partition(r, m.group(2)))
        groups.setdefault(key, []).append(r)
    out = []
    for (model, quant, ctx, part), rs in groups.items():
        rs.sort(key=lambda x: x["date"])
        notes = []
        for x in rs:
            budget = any("exceed budget" in n for n in x["notes"])
            if x["failure_class"] == "oom" and budget and not oom_evidence(x, logs) and all(
                    not (v or {}).get("error") for v in x["per_prompt"].values()) and x["per_prompt"]:
                x["success"], x["failure_class"] = True, None
                notes.append(f"{x['id']}: budget check false positive (Meta() share given to every node), no OOM in any log, all requests finished")
        ok = [x for x in rs if x["success"]]
        ref = (ok or rs)[-1]
        nv = node_view(ref, corrected_breakdown(ref, logs))
        cap = capacity_mib(ref, logs)
        row = {"model": model, "quant": quant, "ctx": ctx, "partition": part,
               "fits": "yes" if ok else "no", "runs_ok": f"{len(ok)}/{len(rs)}",
               "failure_class": ";".join(sorted({x["failure_class"] for x in rs if x["failure_class"]})) or "",
               "run_ids": " ".join(x["id"] for x in rs), "note": " ".join(notes)}
        for n in NODES:
            v = nv.get(n)
            for k in ("model", "kv", "compute"):
                row[f"{n}_{k}_gib"] = round(v[k], 3) if v else None
            row[f"{n}_total_gib"] = v["gpu"] if v else None
            row[f"{n}_line_gib"] = round(cap[n] / 1024, 2) if n in cap and v else None
            row[f"{n}_headroom_gib"] = round(cap[n] / 1024 - v["gpu"], 2) if v and n in cap else None
        row["bytes_per_token"] = ref["comm"].get("bytes_per_token")
        row["task_accuracy"] = ref["correctness"].get("task_accuracy")
        row["exact_match_vs_baseline"] = ref["correctness"].get("exact_match_ratio")
        ms = [x["timing"].get("decode_ms_per_step_pooled") for x in ok if x["timing"].get("decode_ms_per_step_pooled")]
        row["ms_per_step_background"] = round(statistics.median(ms), 1) if ms else None
        row["_outputs"] = ref.get("outputs") if ok else None
        out.append(row)
    # identity against PP default at the same (model, ctx): the same engine, the reference when no BASELINE fits
    for row in out:
        pp = next((x for x in out if x["model"] == row["model"] and x["ctx"] == row["ctx"] and x["partition"] == "PP default"), None)
        a, b = row["_outputs"], (pp or {}).get("_outputs")
        row["identical_to_pp_default"] = (f"{sum(a[i] == b.get(i) for i in a)}/{len(a)}" if a and b else None)
    for row in out:
        row.pop("_outputs")
    out.sort(key=lambda x: (x["model"], x["ctx"], PARTITION_ORDER.index(x["partition"]) if x["partition"] in PARTITION_ORDER else 99))
    return out


def write(out_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(out_rows[0]) if out_rows else []
    with (out_dir / "EDGE_TABLE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    show = ["model", "ctx", "partition", "fits", "runs_ok", "failure_class",
            "m5_model_gib", "m5_kv_gib", "m5_compute_gib", "m5_total_gib", "m5_headroom_gib",
            "air_model_gib", "air_kv_gib", "air_compute_gib", "air_total_gib", "air_headroom_gib",
            "task_accuracy", "identical_to_pp_default", "bytes_per_token", "ms_per_step_background"]
    fmt = lambda v: "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["# Session 5 memory edge table", "",
             "Generated by `python -m nocturne.bench.derive_edge`. GiB from the engine's buffer lines (model, KV, compute; "
             "CPU buffers excluded). Line: the Metal working set the engine reports per device (M5 18186 MiB, Air 12124 MiB). "
             "TP: llama.cpp reports one Meta() device; each node is given the per device share. ms per step is background (T1).",
             "", "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for r in out_rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in show) + " |")
    lines += ["", "## Run ids", ""]
    for r in out_rows:
        lines.append(f"- {r['model']} ctx {r['ctx']} {r['partition']}: {r['run_ids']}")
    (out_dir / "EDGE_TABLE.md").write_text("\n".join(lines) + "\n")


EVIDENCE = re.compile(r"buffer size|llama_kv_cache: size|common_memory_breakdown_print|MiB free\)|offloaded|OutOfMemory|"
                      r"failed to allocate|out of memory|-ts|tensor_split|SPLIT_MODE", re.IGNORECASE)


def write_evidence(results: Path, out_dir: Path, ids: list[str]) -> None:
    """The memory and failure lines of each run's logs (results/logs/ is gitignored) next to the table."""
    ev = out_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    for i in ids:
        d = results / "logs" / i
        if not d.exists():
            continue
        parts = []
        for f in sorted(d.glob("*.log")):
            hits = [l.rstrip() for l in f.read_text(errors="replace").splitlines() if EVIDENCE.search(l)]
            parts.append(f"== {f.name} ({len(hits)} lines)\n" + "\n".join(hits[:200]))
        (ev / f"{i}.txt").write_text("\n".join(parts) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=record.RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "session5_memory_edge")
    a = ap.parse_args(argv)
    rs = rows(a.results)
    write(rs, a.out)
    write_evidence(a.results, a.out, [i for r in rs for i in r["run_ids"].split()])
    print(f"{len(rs)} rows -> {a.out / 'EDGE_TABLE.md'}")


if __name__ == "__main__":
    main()
