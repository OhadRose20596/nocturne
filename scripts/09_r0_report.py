#!/usr/bin/env python3
"""Build R0 Tables A/B/C and the acceptance checks from experiments/r0/runs/*/.
Reads raw artefacts only. Writes experiments/r0/REPORT.md and bench_r0.csv.
Usage: python3 scripts/09_r0_report.py
"""
import csv, json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
R0 = ROOT / "experiments" / "r0"
RUNS = R0 / "runs"

def load_runs():
    runs = []
    for d in sorted(RUNS.glob("*/")):
        m = d / "manifest.json"
        if not m.exists():
            continue
        try:
            man = json.loads(m.read_text())
        except Exception as e:
            print(f"!! {d.name}: bad manifest ({e})", file=sys.stderr)
            continue
        man["_dir"] = d
        runs.append(man)
    return runs

def bench_results(run):
    """Return {test: (avg_ts, std_ts)} from bench.json, or {} if missing/invalid."""
    f = run["_dir"] / "bench.json"
    if not f.exists() or f.stat().st_size == 0:
        return {}
    try:
        rows = json.loads(f.read_text())
    except Exception:
        return {}
    out = {}
    configs = {str(r.get("tensor_split", "")) for r in rows}
    multi = len(configs) > 1 or (len(rows) > 2)
    for r in rows:
        try:
            name = r.get("test")
            if not name:
                n_p, n_g = int(r.get("n_prompt") or 0), int(r.get("n_gen") or 0)
                name = f"pp{n_p}" if n_p > 0 else f"tg{n_g}"
            if multi:
                name = f"{name}@ts={r.get('tensor_split')}"
            out[name] = (float(r["avg_ts"]), float(r.get("stddev_ts", 0)))
        except (KeyError, ValueError, TypeError):
            pass
    return out

def fmt(v):
    if not v:
        return "—"
    a, s = v
    pct = (100 * s / a) if a else 0
    flag = "" if pct < 5 else " ⚠"
    return f"{a:.2f} ± {s:.2f}{flag}"

def expected_sides(split):
    """split 'a,b' = REMOTE (RPC) weight a, local weight b.
    llama.cpp puts --rpc devices at the front of the device list (src/llama.cpp: "add RPC servers at the
    front of the list"); verified from `layer N assigned to device RPC0/MTL0` logs on 2026-10-10.
    Returns (want_local, want_remote)."""
    try:
        parts = [float(x) for x in str(split).split(",")]
        return any(p > 0 for p in parts[1:]), parts[0] > 0
    except Exception:
        return True, True

def placement_summary(run):
    f = run["_dir"] / "placement.log"
    if not f.exists():
        return "no log", False
    txt = f.read_text(errors="replace")
    rpc = [(n, sz) for n, sz in re.findall(r"load_tensors:\s+(RPC\d*\[[^\]]+\])\s+model buffer size\s*=\s*([\d.]+ \w+)", txt) if float(sz.split()[0]) > 0]
    loc = [(n, sz) for n, sz in re.findall(r"load_tensors:\s+((?:MTL|Metal|CUDA|Vulkan)[\w_]*)\s+model buffer size\s*=\s*([\d.]+ \w+)", txt) if float(sz.split()[0]) > 0]
    cpu = re.findall(r"load_tensors:\s+(CPU[\w_]*)\s+model buffer size\s*=\s*([\d.]+ \w+)", txt)
    off = re.search(r"offloaded (\d+)/(\d+) layers", txt)
    parts = []
    assigned = re.findall(r"layer\s+(\d+) assigned to device (\w+)", txt)
    if assigned:
        n = max(int(i) for i, _ in assigned) + 1
        last = assigned[-n:]
        counts = {}
        for _, dev in last:
            counts[dev] = counts.get(dev, 0) + 1
        parts.append("layers: " + ", ".join(f"{d}={c}" for d, c in counts.items()))
    parts += [f"{n} {sz}" for n, sz in rpc + loc + cpu]
    if off:
        parts.append(f"offloaded {off.group(1)}/{off.group(2)}")
    want_local, want_remote = expected_sides(run.get("split"))
    ok = (bool(loc) or not want_local) and (bool(rpc) or not want_remote) and bool(off)
    if want_local and not want_remote and rpc:
        ok = False   # 1,0 but something landed on the worker
    return ("; ".join(parts) if parts else "no load_tensors lines"), ok

def net_summary(run):
    f = run["_dir"] / "net.txt"
    if not f.exists():
        return {}
    txt = f.read_text(errors="replace")
    d = {}
    m = re.search(r"(?:rtt|round-trip)[^=]*=\s*([\d.]+)/([\d.]+)/([\d.]+)", txt)
    if m:
        d["rtt_min"], d["rtt_avg"], d["rtt_max"] = m.groups()
    m = re.search(r"packets transmitted, (\d+) (?:packets )?received", txt)
    if m:
        d["recv"] = m.group(1)
    bws = re.findall(r"([\d.]+ [KMG]bits/sec)\s+(?:\d+\s+)?receiver", txt)
    if bws:
        d["iperf_fwd"] = bws[0]
        if len(bws) > 1:
            d["iperf_rev"] = bws[1]
    m = re.search(r"==\s*route\s*\n(.*)", txt)
    if m:
        d["route"] = m.group(1).strip()[:60]
    ts = re.search(r"pong from \S+ \(([^)]*)\) via (\S+)", txt)
    if ts:
        d["ts_path"] = f"{ts.group(2)}"
    return d

def main():
    runs = load_runs()
    if not runs:
        print(f"no runs under {RUNS}. Run scripts/08_r0.sh first.")
        return 1
    bench = [r for r in runs if r["kind"] == "bench"]
    place = [r for r in runs if r["kind"] == "placement"]
    net = [r for r in runs if r["kind"] == "net"]

    out = []
    w = out.append
    w("# R0 report — generated by scripts/09_r0_report.py\n")
    w(f"Runs: {len(bench)} bench, {len(place)} placement, {len(net)} net. Source: `experiments/r0/runs/`.\n")

    shas = {r["llama_sha"] for r in runs if r.get("llama_sha")}
    hashes = {r["model_sha256"] for r in runs if r.get("model_sha256") not in (None, "", "missing", "unavailable")}
    pptg = {(r.get("pp"), r.get("tg"), r.get("reps")) for r in bench}
    w("## Consistency\n")
    w(f"- llama.cpp SHA(s): {', '.join(s[:10] for s in shas) or '—'}  {'✅' if len(shas) == 1 else '❌ more than one SHA — rows are not comparable'}")
    w(f"- model sha256(s): {', '.join(h[:12] for h in hashes) or '—'}  {'✅' if len(hashes) == 1 else '❌ more than one model hash'}")
    w(f"- PP/TG/REPS set(s): {sorted(pptg)}  {'✅' if len(pptg) == 1 else '❌ mixed benchmark sizes'}\n")

    w("## Table A — single machine (role=local)\n")
    w("| host | backend | label | pp t/s | tg t/s | run_id |")
    w("|---|---|---|---|---|---|")
    a_rows = [r for r in bench if r.get("role") == "local"]
    for r in a_rows:
        res = bench_results(r)
        pp = next((v for k, v in res.items() if k.startswith("pp")), None)
        tg = next((v for k, v in res.items() if k.startswith("tg")), None)
        w(f"| {r['host']} | {r['backend']} | {r['label']} | {fmt(pp)} | {fmt(tg)} | `{r['run_id']}` |")
    if not a_rows:
        w("| — | | | | | |")
    w("")

    w("## Table C — network path (one row per net run)\n")
    w("| date (UTC) | from | to | location | link | overlay | path | RTT min/avg/max ms | recv | iperf3 fwd | iperf3 rev | route |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in net:
        d = net_summary(r)
        rtt = "/".join(d.get(k, "?") for k in ("rtt_min", "rtt_avg", "rtt_max")) if "rtt_avg" in d else "—"
        path = r.get("path") or d.get("ts_path", "")
        w(f"| {r['started_utc']} | {r['host']} | {r['peer']} | {r.get('location','')} | {r.get('link','')} | {r.get('overlay','')} | {path} | {rtt} | {d.get('recv','?')}/20 | {d.get('iperf_fwd','—')} | {d.get('iperf_rev','—')} | {d.get('route','')} |")
    if not net:
        w("| — | | | | | | | | | | | |")
    w("")

    w("## Table B — requester + remote worker (role=requester)\n")
    w("| requester | worker | location | link | overlay | split | pp t/s | tg t/s | placement log | verified | run_id |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    b_rows_all = [r for r in bench if r.get("role") == "requester"]
    invalid = [r for r in b_rows_all if any("@ts=" in k for k in bench_results(r))]   # >1 llama-bench config = old ',' syntax
    b_rows = [r for r in b_rows_all if r not in invalid]
    for r in invalid:
        w(f"| {r['host']} | {r['peer']} | {r.get('location','')} | {r.get('link','')} | {r.get('overlay','')} | {r.get('split','')} | — | — | **INVALID: bench.json holds several tensor-split configs (old `-ts a,b` syntax); move to runs_superseded_ts_syntax/** | ❌ | `{r['run_id']}` |")
    for r in b_rows:
        res = bench_results(r)
        pp = next((v for k, v in res.items() if k.startswith("pp")), None)
        tg = next((v for k, v in res.items() if k.startswith("tg")), None)
        own = r["_dir"] / "placement.log"
        if own.exists() and own.stat().st_size > 0:
            psum, pok = placement_summary(r)          # evidence from the measured run itself (preferred)
            psum = "bench run: " + psum
        else:
            cands = [p for p in place if p["host"] == r["host"] and p["peer"] == r["peer"] and p.get("split") == r.get("split") and p["started_utc"] <= r["started_utc"]]
            if cands:
                psum, pok = placement_summary(cands[-1]); psum = "separate cli run: " + psum
            else:
                psum, pok = "no placement evidence", False
        w(f"| {r['host']} | {r['peer']} | {r.get('location','')} | {r.get('link','')} | {r.get('overlay','')} | {r.get('split','')} | {fmt(pp)} | {fmt(tg)} | {psum} | {'✅' if pok else '❌'} | `{r['run_id']}` |")
    if not b_rows:
        w("| — | | | | | | | | | | |")
    w("")

    w("## Acceptance (R0_PROTOCOL §6)\n")
    hosts_local = defaultdict(set)
    for r in a_rows:
        hosts_local[r["host"]].add(r["label"])
    ok_a = sum(1 for h, labs in hosts_local.items() if {"gpu_all_layers", "cpu_only"} <= labs)
    w(f"- A: {ok_a} host(s) with both gpu_all_layers and cpu_only rows  {'✅' if ok_a >= 2 else '⏳ need 2'}")
    combos = defaultdict(set)
    for r in b_rows:
        combos[(r["host"], r["peer"], r.get("link"))].add(r.get("split"))
    full = [k for k, s in combos.items() if {"0,1", "1,1", "1,0"} <= s]
    w(f"- B: {len(full)} (requester, worker, link) combination(s) with all three splits  {'✅' if len(full) >= 4 else '⏳ need 4 = 2 directions × 2 links'}")
    noisy = []
    for r in bench:
        for k, (a, s) in bench_results(r).items():
            if a and 100 * s / a >= 5:
                noisy.append(f"{r['run_id']}:{k}")
    w(f"- variance < 5 %: {'✅' if not noisy else '⚠ ' + ', '.join(noisy[:6])}")
    unverified = [r["run_id"] for r in b_rows if not next((placement_summary(p)[1] for p in reversed(place) if p["host"] == r["host"] and p["peer"] == r["peer"] and p.get("split") == r.get("split") and p["started_utc"] <= r["started_utc"]), False)]
    if not b_rows:
        w("- placement verified for every B row: n/a (no B rows yet)")
    else:
        w(f"- placement verified for every B row: {'✅' if not unverified else '❌ ' + ', '.join(unverified[:6])}")
    failed = [r["run_id"] for r in runs if r.get("exit_code", 0) not in (0, "0")]
    if failed:
        w(f"- non-zero exit codes: {', '.join(failed)} — read their stderr.log before using the rows")
    w("")

    report = "\n".join(out)
    (R0 / "REPORT.md").write_text(report)
    print(report)

    with (R0 / "bench_r0.csv").open("w", newline="") as f:
        cw = csv.writer(f, quoting=csv.QUOTE_ALL)
        cw.writerow(["run_id", "started_utc", "host", "role", "peer", "location", "link", "overlay", "split", "label", "test", "avg_ts", "stddev_ts", "llama_sha", "model_sha256", "pp", "tg", "reps"])
        for r in bench:
            for test, (a, s) in bench_results(r).items():
                cw.writerow([r["run_id"], r["started_utc"], r["host"], r.get("role"), r.get("peer"), r.get("location"), r.get("link"), r.get("overlay"), r.get("split"), r["label"], test, a, s, r["llama_sha"], r["model_sha256"], r["pp"], r["tg"], r["reps"]])
    print(f"\nwritten: {R0/'REPORT.md'}  {R0/'bench_r0.csv'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
