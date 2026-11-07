#!/usr/bin/env python3
"""Schema v3 values for every existing record, from the records and their logs, written to results/derived_v3.json.
The records themselves are never modified (RESULTS.md and results/*.json are evidence).
Usage: python -m nocturne.bench.derive_v3 [--results DIR] [--out FILE] [--id ID]
Per record: timing.decode_ms_per_step_pooled, decode_tps_pooled, decode_steps_total (from per_prompt, every version);
comm.blocking_syncs_per_step, messages_per_step, graph_meta_bytes_per_step (llama.cpp worker log inside the decode
windows, when the log is timestamped); comm.bytes_per_step_decode and its directions (capture.tsv; two column
captures from before 2026-11-02 give the total only); torch_tp blocking syncs from the engine's per step count."""
import argparse
import datetime
import json
import sys
from pathlib import Path

from . import record
from .probes import capture


def windows_of(rec: dict) -> tuple[list[tuple[float, float, int]], int | None]:
    win, steps = [], 0
    for v in (rec.get("per_prompt") or {}).values():
        tm = v.get("timings") or {}
        n, ms = tm.get("predicted_n") or 0, tm.get("predicted_ms")
        if v.get("error") or not v.get("t_done") or ms is None or not n:
            continue
        win.append((v["t_done"] - ms / 1000.0, v["t_done"], n))
        steps += max(0, n - 1)
    return win, (steps or None)


def derive_one(raw: dict, logdir: Path) -> dict:
    v3 = record.upgrade(raw) if raw.get("schema_version", 1) < record.SCHEMA_VERSION else json.loads(json.dumps(raw))
    out = {"schema_version_on_disk": raw.get("schema_version", 1), "engine": v3["engine"].get("name"),
           "timing": {k: v3["timing"].get(k) for k in ("decode_tps_median", "decode_ms_per_step_pooled", "decode_tps_pooled", "decode_steps_total")},
           "comm": {k: None for k in record.V3_COMM if k not in ("worker_capture", "clock_offset_s", "clock_offset_unc_s")},
           "notes": []}
    if raw.get("schema_version", 1) >= record.SCHEMA_VERSION:
        out["comm"].update({k: v3["comm"].get(k) for k in out["comm"]})
        out["notes"].append("v3 record: values copied, nothing derived")
        return out
    win, steps = windows_of(v3)
    if not steps:
        out["notes"].append("no decode steps (no per_prompt timings); comm per step null")
        return out
    if v3["engine"].get("name") == "torch_tp":
        per = v3["comm"].get("rpc_commands_per_step")
        if per:
            out["comm"]["blocking_syncs_per_step"] = out["comm"]["messages_per_step"] = round(per) + 1
            out["notes"].append("torch_tp: all-reduce frames per step from the engine counters plus the STEP frame")
    else:
        logs = sorted(logdir.glob("worker_*.log"))
        if logs:
            dc, nn = capture.decode_comm_from_worker_log(logs[0].read_text(errors="replace"), win, steps)
            out["comm"].update(dc)
            out["notes"] += nn
        else:
            out["notes"].append("no worker log; blocking_syncs, messages and graph_meta per step null")
    cap = logdir / "capture.tsv"
    if cap.exists():
        pk = capture.load_capture(cap)
        out["comm"].update(capture.bytes_per_decode_step(pk, win, steps))
        if pk and pk[0][2] == "?":
            out["notes"].append("capture.tsv has no direction (before 2026-11-02); in and out null")
    else:
        out["notes"].append("no capture.tsv; bytes_per_step_decode null")
    return out


def derive_all(results_dir: Path, only: str | None = None) -> dict:
    out = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "schema_version": record.SCHEMA_VERSION,
           "source": "python -m nocturne.bench.derive_v3; records and logs read only", "records": {}}
    for raw in record.read_all(results_dir, raw=True):
        if only and raw["id"] != only:
            continue
        out["records"][raw["id"]] = derive_one(raw, results_dir / "logs" / raw["id"])
    return out


def load(results_dir: Path = record.RESULTS_DIR) -> dict:
    p = Path(results_dir) / "derived_v3.json"
    return json.loads(p.read_text())["records"] if p.exists() else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=record.RESULTS_DIR)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--id")
    a = ap.parse_args(argv)
    d = derive_all(a.results, a.id)
    out = a.out or (a.results / "derived_v3.json")
    if a.id:
        print(json.dumps(d["records"], indent=2))
        return 0
    out.write_text(json.dumps(d, indent=2) + "\n")
    print(f"{len(d['records'])} records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
