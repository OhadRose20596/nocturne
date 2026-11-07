#!/usr/bin/env python3
"""Recompute the schema v2 values of one record from its logs and print them. Writes nothing.
Usage: python -m nocturne.bench.recompute --id <record id> [--results DIR] [--json]
From results/logs/<id>/: worker_*.log -> rpc_commands_per_step, tokens_per_step, rpc_commands_per_token,
setup.transfer_s (only when the log is timestamped); capture.tsv -> bytes_per_step, bytes_per_token, packets_per_token;
per_prompt -> ttft_server_s_median; outputs -> quality_pass against the matching baseline on the same prompt ids.
Values the logs cannot give (memory_peak_gb, ttft_s client side, capture_iface) are printed as null with the reason."""
import argparse
import json
import sys
from pathlib import Path

from . import prompts, record
from .check import correctness
from .probes import capture


def recompute(rec: dict, logdir: Path, all_recs: list[dict] | None = None) -> dict:
    v2 = record.upgrade(rec)
    out = {"id": rec["id"], "schema_version_on_disk": rec.get("schema_version", 1), "comm": {}, "setup": {}, "timing": {},
           "memory_peak_gb": {h: None for h in v2["memory_buffers_gb"]}, "quality_pass": None, "notes": []}
    per = rec.get("per_prompt", {})
    n_gen = [(v.get("timings") or {}).get("predicted_n") or 0 for v in per.values() if not v.get("error")]
    total = sum(n_gen)
    logs = sorted(logdir.glob("worker_*.log"))
    steps = None
    if logs and total:
        text = logs[0].read_text(errors="replace")
        stats, notes = capture.step_stats_from_worker_log(text)
        out["notes"] += notes
        if stats:
            steps = stats["compute_steps"]
            out["comm"].update(capture.comm_from_steps(stats, total))
        t, notes = capture.transfer_window_from_worker_log(text)
        out["setup"]["transfer_s"] = t
        out["notes"] += notes
    else:
        out["notes"].append("no worker log or no generated tokens; rpc_commands fields null")
    cap = logdir / "capture.tsv"
    if cap.exists() and total:
        pk = [(float(t), int(b)) for t, b in (l.split("\t") for l in cap.read_text().splitlines() if l)]
        win = []
        for v in per.values():
            tm = v.get("timings") or {}
            if v.get("t_done") and tm.get("predicted_ms"):
                win.append((v["t_done"] - tm["predicted_ms"] / 1000, v["t_done"], tm.get("predicted_n") or 0))
        out["comm"].update(capture.bytes_per_token(pk, win, steps))
    else:
        out["notes"].append("no capture.tsv; bytes fields null")
    out["comm"]["capture_iface"] = None
    out["comm"]["serial_waits_per_token"] = None
    out["setup"]["load_total_s"] = v2["setup"]["load_total_s"]
    out["timing"]["ttft_server_s_median"] = v2["timing"]["ttft_server_s_median"]
    out["timing"]["ttft_s_median"] = None
    out["notes"].append("memory_peak_gb and client ttft_s need a live probe; not recoverable from logs")
    ids = sorted(rec.get("outputs", {}))
    rows = [r for r in prompts.load() if r["id"] in ids]
    if rec["cell"]["row"] == "BASELINE":
        acc = correctness.task_accuracy(rec["outputs"], rows)
        out["correctness"] = {"task_accuracy": acc, "baseline_task_accuracy": acc}
        out["quality_pass"] = True if acc is not None else None
    else:
        req = next(p["host"] for p in rec["peers"] if p["role"] == "requester")
        base = record.find_baseline(all_recs or [], v2["model"], req, rec["engine"]["commit"], v2["prompts"])
        if base is None:
            out["notes"].append("no matching baseline; quality_pass null")
            out["correctness"] = {"task_accuracy": correctness.task_accuracy(rec["outputs"], rows), "baseline_task_accuracy": None}
        else:
            out["correctness"] = correctness.compare(rec["outputs"], base["outputs"], rows)
            out["correctness"]["baseline_id"] = base["id"]
            out["quality_pass"] = correctness.passes(out["correctness"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", required=True)
    ap.add_argument("--results", type=Path, default=record.RESULTS_DIR)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    path = a.results / f"{a.id}.json"
    if not path.exists():
        sys.exit(f"no record {path}")
    rec = json.loads(path.read_text())
    out = recompute(rec, a.results / "logs" / a.id, record.read_all(a.results))
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        for k in ("comm", "setup", "timing", "memory_peak_gb", "correctness"):
            print(f"{k}:")
            for kk, v in out.get(k, {}).items():
                print(f"  {kk}: {v}")
        print(f"quality_pass: {out['quality_pass']}")
        for n in out["notes"]:
            print(f"note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
