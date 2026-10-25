#!/usr/bin/env python3
"""RPC command trace per decode step from a worker log produced with GGML_RPC_DEBUG=1.
A step is the span between two graph_compute or graph_recompute events. Reports commands, bytes in and out per step for the tail.
Usage: python3 scripts/12_rpc_trace.py <worker_log> [--tail 32] [--label NAME]
Writes experiments/r3/trace/<run_id>/{summary.json,summary.md}.
"""
import argparse, json, re, statistics, sys, datetime, socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "r3" / "trace"
LINE = re.compile(r"\[(\w+)\]\s*(.*)")
SIZE = re.compile(r"size:\s*(\d+)")
NODES = re.compile(r"n_nodes:\s*(\d+)")

def split_sessions(path):
    """One worker log holds many client sessions ([hello] ... [~impl] closing socket). Return the line blocks
    of sessions that actually computed something (preflight/--list-devices sessions have no graph events)."""
    sessions, cur = [], []
    for raw in Path(path).read_text(errors="replace").splitlines():
        if "[hello]" in raw and cur:
            sessions.append(cur); cur = []
        cur.append(raw)
    if cur:
        sessions.append(cur)
    return [b for b in sessions if any("[graph_" in l for l in b)]

def parse(lines):
    steps, cur = [], None
    for raw in lines:
        m = LINE.search(raw)
        if not m:
            continue
        cmd, rest = m.group(1), m.group(2)
        if cmd in ("graph_compute", "graph_recompute"):
            if cur is not None:
                steps.append(cur)
            n = NODES.search(rest)
            cur = {"kind": cmd, "n_nodes": int(n.group(1)) if n else None, "cmds": {}, "bytes_in": 0, "bytes_out": 0}
            continue
        if cur is None:
            cur = {"kind": "pre", "n_nodes": None, "cmds": {}, "bytes_in": 0, "bytes_out": 0}
        cur["cmds"][cmd] = cur["cmds"].get(cmd, 0) + 1
        s = SIZE.search(rest)
        if s:
            if cmd == "set_tensor":
                cur["bytes_in"] += int(s.group(1))
            elif cmd == "get_tensor":
                cur["bytes_out"] += int(s.group(1))
    if cur is not None:
        steps.append(cur)
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tail", type=int, default=32, help="steady-state steps to summarize (from the end)")
    ap.add_argument("--label", default="trace")
    a = ap.parse_args()
    sessions = split_sessions(a.log)
    if not sessions:
        print("no [graph_compute] lines found — was the worker started with GGML_RPC_DEBUG=1, and did a requester run?"); return 1
    if len(sessions) > 1:
        print(f"note: {len(sessions)} compute sessions in this log; reporting each (use one worker per split to avoid guessing)")
    rc = 0
    for si, lines in enumerate(sessions, 1):
        rc |= report(parse(lines), a, f"{a.label}_s{si}" if len(sessions) > 1 else a.label)
    return rc

def report(steps, a, label):
    compute_steps = [s for s in steps if s["kind"] in ("graph_compute", "graph_recompute")]
    tail = compute_steps[-a.tail:] if len(compute_steps) > a.tail else compute_steps
    head = compute_steps[:max(0, len(compute_steps) - len(tail))]

    def agg(group):
        if not group:
            return {}
        types = sorted({c for s in group for c in s["cmds"]})
        per_type = {t: statistics.median([s["cmds"].get(t, 0) for s in group]) for t in types}
        return {"steps": len(group),
                "recompute_share": sum(1 for s in group if s["kind"] == "graph_recompute") / len(group),
                "median_cmds_per_step_by_type": per_type,
                "median_total_cmds_per_step": statistics.median([sum(s["cmds"].values()) for s in group]),
                "median_bytes_in_per_step": statistics.median([s["bytes_in"] for s in group]),
                "median_bytes_out_per_step": statistics.median([s["bytes_out"] for s in group]),
                "max_bytes_out_per_step": max(s["bytes_out"] for s in group),
                "n_nodes_full_graph": next((s["n_nodes"] for s in reversed(group) if s["n_nodes"]), None) or next((s["n_nodes"] for s in reversed(compute_steps) if s["n_nodes"]), None)}

    pre = steps[0] if steps and steps[0]["kind"] == "pre" else None
    summary = {
        "run_id": f"{datetime.datetime.utcnow():%Y%m%d-%H%M%S}_{socket.gethostname().split('.')[0]}_trace_{label}",
        "log": str(a.log), "total_compute_steps": len(compute_steps),
        "load_phase": {"cmds": pre["cmds"], "bytes_in": pre["bytes_in"]} if pre else None,
        "early_steps(load+prefill)": agg(head), "steady_state_tail": agg(tail),
        "note": "counts are server-side commands per compute step; set_tensor/graph_compute are async on the client, get_tensor waits. Not RTTs.",
    }
    d = OUT / summary["run_id"]; d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    t = summary["steady_state_tail"] or {}
    md = (f"| {label} | {len(compute_steps)} | {t.get('steps')} | {t.get('recompute_share', 0):.2f} | "
          f"{t.get('median_total_cmds_per_step')} | {json.dumps(t.get('median_cmds_per_step_by_type'))} | "
          f"{t.get('median_bytes_in_per_step')} | {t.get('median_bytes_out_per_step')} | {t.get('n_nodes_full_graph')} |")
    summary["session_label"] = label
    (d / "summary.md").write_text("| label | compute steps | tail steps | recompute share | cmds/step | by type | bytes in/step | bytes out/step | n_nodes |\n|---|---|---|---|---|---|---|---|---|\n" + md + "\n")
    print(json.dumps(summary, indent=2, default=str)); print(md); print(f"saved: {d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
