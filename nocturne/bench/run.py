#!/usr/bin/env python3
"""Run one cell of the matrix on the configured peers, measure, check correctness, write one record per run.
Usage: python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --runs 3 [--subset 10] [--dry-run] [--live]
       python -m nocturne.bench.run --cell BASELINE --host m5 --model qwen2.5-7b-q4km
Every launched command is printed and stored in the record under "commands"."""
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

from . import adapters, config, prompts, record
from .adapters.base import EngineUnsupported
from .adapters.llamacpp_rpc import LoadFailed
from .check import correctness
from .check.graders import GRADERS
from .probes import capture, mem, net, util
from .runner import Runner

ROOT = Path(__file__).resolve().parents[2]
RESULTS_MD = ROOT / "experiments" / "RESULTS.md"
RESULTS_MD_HEADER = (
    "\n## Bench matrix (nocturne/bench, from 2026-10-26)\n\n"
    "One row per successful run. Full record: nocturne/bench/results/<id>.json.\n\n"
    "| date | id | requester | workers | llama.cpp | model | network | rtt ms | bw Mbps | decode t/s | S msgs/tok | B bytes/tok | peak GB | acc | base acc |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def peers_for_record(cfg: dict, requester: str, workers: list[str]) -> list[dict]:
    out = []
    for name, role in [(requester, "requester")] + [(w, "worker") for w in workers]:
        p = cfg["peers"][name]
        out.append({"host": p["host"], "role": role, "backend": p["backend"], "os": p["os"], "mem_gb": p["mem_gb"]})
    return out


def append_results_md(rec: dict, path: Path = RESULTS_MD) -> None:
    if not path.exists():
        return
    text = path.read_text()
    with path.open("a") as f:
        if "## Bench matrix (nocturne/bench" not in text:
            f.write(RESULTS_MD_HEADER)
        req = next(p["host"] for p in rec["peers"] if p["role"] == "requester")
        workers = ",".join(p["host"] for p in rec["peers"] if p["role"] == "worker") or "-"
        # Same columns as the rows written since 2026-10-26 (the table is append-only): S = rpc_commands_per_token,
        # B = bytes_per_token, "peak GB" = device buffers (memory_buffers_gb). From 2026-11-02 (schema v3) "decode t/s"
        # is the pooled value (sum(predicted_n - 1) / sum(predicted_ms)) and the network column carries the run
        # conditions: awdl state, added one-way delay, fixed tokens, prompt set.
        n, t, c, m, cr = rec["network"], rec["timing"], rec["comm"], rec["memory_buffers_gb"], rec["correctness"]
        fmt = lambda v, nd=1: "null" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        eng = rec["engine"]
        engine_col = eng["commit"] if eng.get("name") in ("llama.cpp", "", None) else f"{eng.get('name')} {eng['commit']}"
        pooled = t.get("decode_tps_pooled")
        tps_col = fmt(pooled) + " pooled" if pooled is not None else fmt(t["decode_tps_median"])
        cond = [n["type"]]
        if n.get("awdl"):
            cond.append(f"awdl {n['awdl']}")
        if n.get("added_delay_ms") is not None:
            cond.append(f"delay {fmt(n['added_delay_ms'])} ms")
        pr = rec.get("prompts", {})
        if pr.get("fixed_tokens"):
            cond.append(f"fixed {pr['fixed_tokens']}")
        if pr.get("set") and pr.get("set") != "eval_set":
            cond.append(pr["set"])
        f.write(f"| {rec['date']} | {rec['id']} | {req} | {workers} | {engine_col} | {rec['model']['name']} | "
                f"{'; '.join(cond)} | {fmt(n['rtt_ms_median'])} | {fmt(n['bw_mbps'], 0)} | {tps_col} | "
                f"{fmt(c['rpc_commands_per_token'])} | {fmt(c['bytes_per_token'], 0)} | "
                f"{fmt(max(m.values()) if m else None)} | {fmt(cr['task_accuracy'], 2)} | {fmt(cr['baseline_task_accuracy'], 2)} |\n")


def remote_iface_for(runner, peer: dict, ip: str) -> tuple[str | None, list[str]]:
    """Interface on `peer` that owns the route back to `ip` (route get over ssh), for the worker side capture."""
    r = runner.run(peer, ["route", "-n", "get", ip] if peer.get("os") == "macos" else ["ip", "route", "get", ip], timeout=30)
    pat = capture.ROUTE_IFACE_MAC if peer.get("os") == "macos" else capture.ROUTE_IFACE_LINUX
    m = pat.search(r.stdout or "")
    if not m:
        return None, [f"worker capture not started: route lookup for {ip} on {peer.get('name')} gave no interface"]
    return m.group(1), []


def safe_stop(eng, timeout_s: float = 90) -> str | None:
    """Run eng.stop() with a bound. Returns None when it finished, else a one-line reason."""
    import threading
    err: list[str] = []

    def target():
        try:
            eng.stop()
        except Exception as e:  # noqa: BLE001
            err.append(f"{type(e).__name__}: {e}")

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return f"stop() still running after {timeout_s} s"
    return err[0] if err else None


class EventLog:
    """Phase and request markers for the live view: results/logs/<id>/events.jsonl. Not part of the record."""

    def __init__(self, path: Path | None):
        self.fh = path.open("a", buffering=1) if path else None

    def __call__(self, ev: dict) -> None:
        if self.fh:
            ev.setdefault("t", time.time())
            self.fh.write(json.dumps(ev) + "\n")

    def close(self):
        if self.fh:
            self.fh.close()


def run_once(a, cells, cfg, cell_cfg, run_idx: int, results_dir: Path, live_srv=None) -> dict:
    model_cfg = cfg["models"][a.model]
    requester = a.host or cfg["roles"]["requester"]
    workers = [] if a.cell == "BASELINE" else list(cfg["roles"]["workers"])
    network = "local" if a.cell == "BASELINE" else a.network
    cfg = dict(cfg, roles={"requester": requester, "workers": workers})
    for name, p in cfg["peers"].items():
        p["name"] = name

    tag = getattr(a, "tag", None)
    rec = record.empty_record(a.cell, cell_cfg, network, a.model, run_idx, host=f"{requester}_{tag}" if tag else requester)
    rec["peers"] = peers_for_record(cfg, requester, workers)
    rec["model"].update({"quant": model_cfg.get("quant", ""), "sha256": None})
    prompt_path = prompts.path_for(getattr(a, "prompt_set", None))
    pmeta = prompts.meta(prompt_path)
    fixed_tokens = getattr(a, "fixed_tokens", None)
    rec["prompts"] = {"set": pmeta["set"], "version": pmeta["version"], "subset": a.subset, "ids": [], "generation": {},
                      "fixed_tokens": fixed_tokens}
    notes = rec["notes"]
    workdir = results_dir / "logs" / rec["id"]
    if not a.dry_run:
        workdir.mkdir(parents=True, exist_ok=True)
    events = EventLog(None if a.dry_run else workdir / "events.jsonl")
    events({"event": "peers", "peers": [{"name": n, "role": "requester" if n == requester else "worker",
                                         "backend": cfg["peers"][n]["backend"],
                                         "ip": (cfg["peers"][n].get("networks") or {}).get(network) or cfg["peers"][n].get("bind")}
                                        for n in [requester] + workers]})
    live = None
    if live_srv is not None and not a.dry_run:
        # one listening socket for the whole invocation (I025 4): the server is started once in main(), each repeat
        # attaches a new tailer on its own log directory
        from .live.__main__ import attach as live_attach
        live = live_attach(live_srv, workdir)
        print(f"[live] {live[1]}", flush=True)

    if cell_cfg["status"] != "ready":
        rec["failure_class"] = "tool_unsupported"
        notes.append(f"cell {a.cell} status is {cell_cfg['status']}: {cell_cfg.get('notes', '')}".strip())
        return rec

    runner = Runner(dry_run=a.dry_run)
    # BASELINE is one cell for every engine (ADR-007 amendment 2): the adapter comes from models.<name>.engine in
    # peers.yaml (default llamacpp_rpc) and the flag template from cells.yaml flags_by_engine.<adapter> when present.
    adapter_name = cell_cfg["adapter"]
    if a.cell == "BASELINE":
        adapter_name = model_cfg.get("engine") or cell_cfg["adapter"]
        by_engine = cell_cfg.get("flags_by_engine") or {}
        if adapter_name in by_engine:
            cell_cfg = dict(cell_cfg, flags=list(by_engine[adapter_name]))
    try:
        Engine = adapters.get(adapter_name)
    except EngineUnsupported as e:
        rec["failure_class"] = "tool_unsupported"; notes.append(str(e)); return rec
    eng = Engine(a.cell, cell_cfg, cfg, model_cfg, network, runner, workdir, ctx=a.ctx,
                 load_timeout_s=a.load_timeout, request_timeout_s=a.request_timeout)
    eng.on_event = events
    for ov in getattr(a, "override_flag", None) or []:
        flag, _, value = ov.partition("=")
        eng.overrides[flag] = value
        notes.append(f"flag override: {flag} {value} (second data point, not the cell's template; recorded as a note, "
                     f"listed in results/EXCLUDE.yaml, not a RESULTS.md table row)")

    rows = prompts.load(subset=a.subset, path=prompt_path)
    if cell_cfg["decode"] == "independent":
        rows = prompts.load(categories=["long_structured"], path=prompt_path) or rows
        k = a.k or len(rows)
    else:
        k = 1
    rec["prompts"]["ids"] = [r["id"] for r in rows]
    if hasattr(eng, "fixed_tokens"):
        eng.fixed_tokens = fixed_tokens
    elif fixed_tokens:
        notes.append(f"--fixed-tokens {fixed_tokens} ignored: the {adapter_name} adapter has no fixed token mode")
    if getattr(a, "trace_allreduce", False):
        if hasattr(eng, "trace_allreduce"):
            eng.trace_allreduce = True
            notes.append("all-reduce tracing on (--trace-allreduce): each all-reduce is stamped per stage and the add is synchronized; small overhead")
        else:
            notes.append("--trace-allreduce ignored: not a torch_tp run")
    # grading needs a grader per category and a run that stops on its own; fixed-token runs keep only the identity check
    gradable = [] if fixed_tokens else [r for r in rows if r["category"] in GRADERS]
    if fixed_tokens:
        notes.append(f"fixed tokens {fixed_tokens}: every prompt decodes exactly {fixed_tokens} tokens (ignore_eos); task accuracy not graded")
    elif len(gradable) < len(rows):
        notes.append(f"{len(rows) - len(gradable)} prompts have no grader (category without one); task accuracy covers the rest")

    peer_ip = None
    if workers:
        w0 = cfg["peers"][workers[0]]
        peer_ip = (w0.get("networks") or {}).get(network) or w0["bind"]
    events({"event": "probe"})
    probe, n_notes = net.measure(peer_ip, network, dry_run=a.dry_run)
    rec["network"].update(probe)
    rec["network"]["rtt_source"] = "probe" if probe.get("rtt_ms_median") is not None else None
    rec["network"]["awdl"] = "unknown" if a.dry_run else net.awdl_state()
    rec["network"]["added_delay_ms"] = getattr(a, "added_delay_ms", None)
    notes += n_notes
    pinger = None
    if peer_ip and not a.dry_run:
        pinger = net.PingMonitor(peer_ip)
        pinger.start()
    elif peer_ip:
        print(f"[probe] $ ping -i 0.01 {peer_ip}   (background, whole run)", flush=True)

    comps = []
    samplers = []
    mem_samplers = []
    cap = None
    wcap = None
    try:
        eng.prepare()
        if cell_cfg["decode"] == "speculative":
            rec["draft_model"] = {"name": Path(model_cfg["draft"]).name, "quant": "", "sha256": None}
        events({"event": "workers"})
        eng.start_workers()
        events({"event": "launch"})
        eng.launch(k=k)
        events({"event": "healthy", "load_s": getattr(eng, "load_s", None)})
        util_peers = [cfg["peers"][n] for n in [requester] + workers]
        if eng.util_backend():
            util_peers = [dict(p, backend=eng.util_backend()) for p in util_peers]
            notes.append(f"utilization sampled as {eng.util_backend()} on every peer: the engine does not use the tagged accelerator")
        samplers = util.start_all(util_peers, dry_run=a.dry_run, sink_dir=None if a.dry_run else workdir)
        mem_samplers = mem.start_all([dict(cfg["peers"][n], role="requester" if n == requester else "worker")
                                      for n in [requester] + workers],
                                     requester_pid=eng.server_pid() if hasattr(eng, "server_pid") else None,
                                     rpc_port=eng.worker_port(), dry_run=a.dry_run, patterns=eng.mem_patterns())
        if workers:
            iface, i_notes = ("<iface of route to %s>" % peer_ip, []) if a.dry_run else capture.iface_for(peer_ip)
            notes += i_notes
            if iface is not None:
                cap = capture.Capture(eng.worker_port(), iface)
                rec["comm"]["capture_iface"] = iface
                if a.dry_run:
                    print("[probe] $ " + " ".join(cap.argv()), flush=True)
                else:
                    cap.start()
            else:
                notes.append("tcpdump not started: no capture interface; bytes_per_token null")
            if getattr(a, "capture_worker", False):
                w0 = cfg["peers"][workers[0]]
                req_ip = (cfg["peers"][requester].get("networks") or {}).get(network) or cfg["peers"][requester].get("bind")
                wiface, w_notes = ("<iface of route to %s on %s>" % (req_ip, workers[0]), []) if a.dry_run else remote_iface_for(runner, w0, req_ip)
                notes += w_notes
                if wiface is not None:
                    wcap = capture.Capture(eng.worker_port(), wiface, ssh_peer=w0)
                    if a.dry_run:
                        print(f"[probe:{workers[0]}] $ " + " ".join(wcap.argv()), flush=True)
                    else:
                        wcap.start()
                        rec["comm"]["worker_capture"] = True
        if rows and not getattr(a, "no_warmup", False) and hasattr(eng, "warmup"):
            events({"event": "warmup"})
            rec["timing"]["warmup_s"] = eng.warmup(rows[0])
            if rec["timing"]["warmup_s"] is not None:
                notes.append(f"warm-up request ({rows[0]['id']}, 8 tokens) took {rec['timing']['warmup_s']:.1f} s; not in the timings")
        events({"event": "prompts", "n": len(rows), "k": k})
        comps = eng.complete(rows, concurrency=k)
    except EngineUnsupported as e:
        rec["failure_class"] = "tool_unsupported"; notes.append(str(e))
    except LoadFailed as e:
        rec["failure_class"] = e.failure_class; notes.append(str(e))
    except Exception as e:  # keep the record; never lose a run to a probe bug
        rec["failure_class"] = "engine_abort"; notes.append(f"{type(e).__name__}: {e}")
    finally:
        for s in samplers + mem_samplers:
            s.stop()
        if cap:
            cap.stop()
        if wcap:
            wcap.stop()
        if pinger:
            pinger.stop()
        server_alive = eng.server_alive()
        # A hung stop (ssh pkill with no answer, a process ignoring SIGTERM) must not lose the record (I024): the stop
        # runs in a thread with a bound, the record is written with failure_class engine_abort and the partial data.
        stop_err = safe_stop(eng, timeout_s=90)
        if stop_err:
            rec["failure_class"] = rec["failure_class"] or "engine_abort"
            notes.append(f"engine stop did not finish cleanly: {stop_err}; record holds the partial data")
        events({"event": "done"})
        events.close()
        if live:
            time.sleep(1.0)          # let the page receive the last state
            live[2].stop()

    rec["commands"] = runner.as_records()
    if not a.dry_run:
        try:
            real = eng.real_hostnames()
        except Exception as e:  # a dead ssh must not lose the record
            real, _ = {}, notes.append(f"hostname lookup failed: {e}")
        for name, p in zip([requester] + workers, rec["peers"]):   # same order as peers_for_record
            if name in real and real[name] != p["host"]:
                notes.append(f"peer {name}: peers.yaml host '{p['host']}' but hostname -s is '{real[name]}'; record uses the real one")
                p["host"] = real[name]
    if a.dry_run:
        notes.append("dry run: nothing executed")
        rec["engine"] = {"name": eng.name, "commit": "dry-run", "flags": eng.build_flags(k)}
        return rec

    ctr = eng.counters()
    rec["engine"] = ctr["engine"]
    rec["setup"] = ctr["setup"]
    notes += ctr.get("setup_notes", [])
    rec["memory_buffers_gb"] = ctr["memory_buffers_gb"]
    rec["model"]["sha256"] = ctr["model_sha256"]
    rec["prompts"]["generation"] = ctr.get("generation", {})
    if rec["draft_model"]:
        rec["draft_model"]["sha256"] = ctr["draft_sha256"]
    if ctr["placement_lines"]:
        (workdir / "placement.log").write_text("\n".join(ctr["placement_lines"]) + "\n")
    if ctr.get("tp_outcome"):
        tp = ctr["tp_outcome"]
        notes.append(f"tensor mode outcome ({tp['outcome']}): {tp['detail']}; Meta device line: {tp.get('meta_device', False)}; "
                     f"flash_attn forced: {tp.get('flash_attn_forced')}")
        if tp["outcome"] in ("b", "c") and rec["failure_class"] in (None, "load_fail", "engine_abort"):
            rec["failure_class"] = "tool_unsupported"
    ok = [c for c in comps if not c.error]
    windows = [(*c.decode_window, c.n_gen) for c in ok if c.decode_window and c.n_gen]
    tokens_generated = sum(c.n_gen or 0 for c in ok)
    decode_steps = sum(c.decode_steps for c in ok) or None
    steps = None
    if ctr.get("comm"):
        rec["comm"].update(ctr["comm"])          # engines without an RPC log fill comm themselves (row B)
    for n, t in ctr["worker_logs"].items():
        stats, nn = capture.step_stats_from_worker_log(t)
        notes += nn
        if stats is not None:
            steps = stats["compute_steps"]
            rec["comm"].update(capture.comm_from_steps(stats, tokens_generated))
            rec["comm"]["source"] = ("rpc_commands: worker GGML_RPC_DEBUG log (12_rpc_trace parse, graph message included), "
                                     "median per step over the last 32 steps; per token = per step / tokens_per_step; "
                                     "blocking_syncs, messages and graph_meta per decode step: the same log inside the decode "
                                     "windows / sum(predicted_n - 1); bytes: tcpdump on the requester's routed interface, "
                                     "union of decode windows, per decode step and per direction")
            dc, nn = capture.decode_comm_from_worker_log(t, windows, decode_steps)
            notes += nn
            rec["comm"].update(dc)
            break
    if cap:
        rec["comm"].update(cap.per_token(windows, steps))
        rec["comm"].update(cap.per_decode_step(windows, decode_steps))
        cap.save(workdir / "capture.tsv")
        cap.save_icmp(workdir / "icmp.tsv")
        notes += cap.notes
    if wcap:
        wcap.save(workdir / "capture_worker.tsv")
        wcap.save_icmp(workdir / "icmp_worker.tsv")
        notes += [f"worker capture: {n}" for n in wcap.notes]
        off, unc, pairs = capture.clock_offset(cap.icmp if cap else [], wcap.icmp)
        rec["comm"]["clock_offset_s"], rec["comm"]["clock_offset_unc_s"] = off, unc
        notes.append(f"worker clock offset from {pairs} echo pairs: {off} s (half IQR {unc})" if off is not None
                     else f"worker clock offset not estimated ({pairs} echo pairs seen on both sides)")
    if pinger:
        pinger.save(workdir / "ping.tsv")
        rec["network"]["ping_decode_ms"] = pinger.in_windows([(w[0], w[1]) for w in windows])
        notes += pinger.notes
    for s in samplers:
        rec["utilization"][s.peer["name"]] = s.busy_fraction([(w[0], w[1]) for w in windows])
        notes += s.notes
    for s in mem_samplers:
        rec["memory_peak_gb"][s.peer["name"]] = s.peak_gb([(w[0], w[1]) for w in windows])
        notes += s.notes
    rec["checks"]["memory_measured"] = bool(rec["memory_peak_gb"]) and all(v is not None for v in rec["memory_peak_gb"].values())
    rec["timing"].update({"decode_tps_median": med([c.decode_tps for c in ok]),
                          "prefill_tps_median": med([c.prefill_tps for c in ok]),
                          "ttft_s_median": med([c.raw.get("ttft_client_s") for c in ok]),
                          "ttft_server_s_median": med([c.ttft_s for c in ok])})
    if cell_cfg["decode"] == "independent" and ok:
        t_start = min(c.t_send for c in ok); t_end = max(c.t_done for c in ok)
        agg = sum(c.n_gen or 0 for c in ok) / (t_end - t_start) if t_end > t_start else None
        notes.append(f"A4 aggregate: {sum(c.n_gen or 0 for c in ok)} tokens over {len(ok)} concurrent requests, "
                     f"{agg:.2f} tok/s wall" if agg else "A4 aggregate unavailable")
    rec["outputs"] = {c.prompt_id: c.text for c in comps}
    scores, skipped = correctness.grade_all(rec["outputs"], gradable)
    if skipped:
        notes.append(f"skipped_unsandboxed: {len(skipped)} code prompts not graded (no sandbox-exec or unshare on this host); "
                     f"task_accuracy excludes them")
    rec["per_prompt"] = {c.prompt_id: {"timings": c.raw.get("timings"), "finish_reason": c.raw.get("finish_reason"),
                                       "error": c.error, "decode_tps": c.decode_tps, "decode_ms_per_step": c.decode_ms_per_step,
                                       "ttft_client_s": c.raw.get("ttft_client_s"),
                                       "score": scores.get(c.prompt_id), "t_send": c.t_send, "t_done": c.t_done} for c in comps}
    rec["timing"].update(record.pooled_timing(rec["per_prompt"]))

    # Correctness against the requester's BASELINE: same model (name, sha256, quant), commit, prompt set and version,
    # generation flags, run on the requester host. Baseline accuracy is recomputed on the prompt ids that ran here.
    all_recs = record.read_all(results_dir)
    requester_host = next(p["host"] for p in rec["peers"] if p["role"] == "requester")
    acc = correctness.task_accuracy(rec["outputs"], gradable) if gradable else None
    if a.cell == "BASELINE":
        rec["correctness"] = {"exact_match_ratio": None, "first_divergence_median": None,
                              "task_accuracy": acc, "baseline_task_accuracy": acc}
    else:
        base = record.find_baseline(all_recs, rec["model"], requester_host, rec["engine"]["commit"], rec["prompts"],
                                    engine=rec["engine"]["name"])
        if base is None:
            rec["correctness"]["task_accuracy"] = acc
            notes.append(f"no matching BASELINE record for {a.model} (sha {str(rec['model']['sha256'])[:8]}, {rec['model']['quant']}) "
                         f"with engine {rec['engine']['name']} on {requester_host} at {rec['engine']['commit']} "
                         f"with prompts {rec['prompts']['set']} v{rec['prompts']['version']}; "
                         f"run: python -m nocturne.bench.run --cell BASELINE --host {requester} --model {a.model}")
        else:
            rec["checks"]["baseline_id"] = base["id"]
            rec["correctness"] = correctness.compare(rec["outputs"], base["outputs"], rows, graded=gradable)
    if cell_cfg["decode"] == "speculative":
        a1 = [r for r in all_recs if r["cell"]["row"] == "A" and r["cell"]["col"] == 1 and r["success"]
              and r["model"]["name"] == a.model and r["network"]["type"] == network
              and record.same_commit(r["engine"]["commit"], rec["engine"]["commit"])
              and any(record.same_host(p["host"], requester_host) and p["role"] == "requester" for p in r["peers"])]
        if not a1:
            rec["checks"]["a2_identity"] = None
            notes.append("A1 identity check not run")
        else:
            a1.sort(key=lambda r: r["date"])
            rec["checks"]["a1_id"] = a1[-1]["id"]
            diff = [i for i in rec["outputs"] if rec["outputs"][i] != a1[-1]["outputs"].get(i)]
            rec["checks"]["a2_identity"] = not diff
            if diff:
                rec["failure_class"] = rec["failure_class"] or "wrong_output"
                notes.append(f"A2 greedy output differs from A1 ({a1[-1]['id']}) on {len(diff)} prompts: {diff[:5]}")
            else:
                notes.append(f"A2 greedy output identical to A1 ({a1[-1]['id']})")
        tm = [c.raw.get("timings", {}) for c in ok]
        drafted = sum(t.get("draft_n", 0) for t in tm); accepted = sum(t.get("draft_n_accepted", 0) for t in tm)
        if drafted:
            notes.append(f"draft acceptance: {accepted}/{drafted} = {accepted / drafted:.2f}")

    # success = the operational conditions: workers loaded, every request finished with stop or length, no device
    # buffers above a host's budget, memory measured (or --allow-unmeasured-memory), record valid and written.
    # quality_pass = accuracy within 0.05 of the baseline on the same prompts; reported separately, never folded in.
    if rec["failure_class"] is None:
        if not comps or any(c.error for c in comps):
            errs = [c.error for c in comps if c.error][:3]
            if not server_alive and ctr["oom_in_log"]:
                rec["failure_class"] = "oom"
            elif not server_alive or ctr["net_err_in_log"]:
                rec["failure_class"] = "network_abort" if ctr["net_err_in_log"] else "engine_abort"
            else:
                rec["failure_class"] = "engine_abort"
            notes.append(f"request errors: {errs}" if errs else "no completions")
        else:
            for name, p in zip([requester] + workers, rec["peers"]):
                for field in ("memory_buffers_gb", "memory_peak_gb"):
                    v = rec[field].get(name)
                    if v is not None and v > p["mem_gb"]:
                        rec["failure_class"] = "oom"
                        notes.append(f"{name}: {field} {v:.1f} GB exceed budget {p['mem_gb']} GB")
            if rec["failure_class"] is None and not rec["checks"]["memory_measured"]:
                if a.allow_unmeasured_memory:
                    notes.append("memory_peak_gb missing on some peer; counted as success because --allow-unmeasured-memory was passed")
                else:
                    rec["failure_class"] = "unmeasured_memory"
                    notes.append("memory_peak_gb missing on some peer; not a success (pass --allow-unmeasured-memory to accept)")
    rec["quality_pass"] = correctness.passes(rec["correctness"])
    if rec["quality_pass"] is None:
        notes.append("quality_pass null: no baseline or no graded prompt")
    rec["success"] = rec["failure_class"] is None
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--network", choices=config.NETWORKS, default="wifi")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--subset", type=int)
    ap.add_argument("--host", help="requester peer name (BASELINE, or to override roles.requester)")
    ap.add_argument("--k", type=int, help="parallel slots for decode=independent (default: number of long prompts)")
    ap.add_argument("--prompt-set", help="prompts/<name>.jsonl instead of eval_set (e.g. eval_long)")
    ap.add_argument("--fixed-tokens", type=int, help="every prompt decodes exactly N tokens (max_tokens N, ignore_eos); accuracy not graded")
    ap.add_argument("--capture-worker", action="store_true", help="second tcpdump on the worker over ssh (needs passwordless sudo there); clock offset from the ping echoes")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up request before the prompts")
    ap.add_argument("--trace-allreduce", action="store_true", help="torch_tp: per stage timing of every all-reduce, summary in the record notes and allreduce_trace.json")
    ap.add_argument("--added-delay-ms", type=float, help="one-way delay added on this host with scripts/16_delay.sh, recorded in network.added_delay_ms")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--peers", type=Path)
    ap.add_argument("--override-flag", action="append", metavar="FLAG=VALUE",
                    help="replace a flag value in the cell template; write it as --override-flag=--split-mode=row (argparse needs the =); use with --tag and --no-results-md")
    ap.add_argument("--tag", help="suffix for the record id (e.g. row), keeps a data point apart from the cell's own records")
    ap.add_argument("--results", type=Path, default=record.RESULTS_DIR)
    ap.add_argument("--no-results-md", action="store_true", help="do not append to experiments/RESULTS.md")
    ap.add_argument("--allow-unmeasured-memory", action="store_true",
                    help="count a run as success although memory_peak_gb is null on some peer (noted in the record)")
    ap.add_argument("--dry-run", action="store_true", help="print every command, execute nothing, write no record")
    ap.add_argument("--live", action="store_true", help="serve the live view on 127.0.0.1 during the run (python -m nocturne.bench.live)")
    ap.add_argument("--live-port", type=int, default=8091)
    a = ap.parse_args(argv)

    cells = config.load_cells()
    cell_cfg = config.cell(cells, a.cell)
    cfg = config.load_peers(a.peers if a.peers else (config.PEERS_EXAMPLE if a.dry_run and not config.PEERS_PATH.exists() else None))
    if a.model not in cfg["models"]:
        sys.exit(f"model {a.model!r} not in peers.yaml; known: {', '.join(cfg['models'])}")
    if (a.host or cfg["roles"]["requester"]) not in cfg["peers"]:
        sys.exit(f"requester {a.host!r} not in peers.yaml")
    if a.prompt_set:
        prompts.path_for(a.prompt_set)          # unknown set: KeyError before anything runs
    if a.cell != "BASELINE" and not a.dry_run:
        for tool in ("ping",):
            if not shutil.which(tool):
                sys.exit(f"{tool} not found")

    live_srv = None
    if a.live and not a.dry_run:
        from .live import server as live_server
        live_srv, url = live_server.serve(None, a.live_port)
        print(f"[live] {url} (one socket for all {a.runs} runs)", flush=True)

    rc = 0
    for i in range(a.runs):
        requester_name = a.host or cfg["roles"]["requester"]
        idx = record.next_run_index(a.results, a.cell, "local" if a.cell == "BASELINE" else a.network, a.model,
                                    host=f"{requester_name}_{a.tag}" if a.tag else requester_name) if not a.dry_run else i + 1
        rec = run_once(a, cells, cfg, cell_cfg, idx, a.results, live_srv)
        if a.dry_run:
            print(f"\n[dry-run] record id would be {rec['id']}; flags: {' '.join(rec['engine']['flags'])}")
            print(f"[dry-run] {len(rec['commands'])} commands logged, nothing executed")
            record.validate(rec)
            continue
        path = record.write(rec, a.results)
        status = "OK" if rec["success"] else f"FAIL ({rec['failure_class']})"
        q = {True: "quality_pass", False: "QUALITY FAIL", None: "quality unknown"}[rec["quality_pass"]]
        t, c = rec["timing"], rec["comm"]
        print(f"\n{rec['id']}: {status}, {q}  decode {t['decode_tps_median']}  cmds/step {c['rpc_commands_per_step']}  "
              f"cmds/tok {c['rpc_commands_per_token']}  B/tok {c['bytes_per_token']}  acc {rec['correctness']['task_accuracy']}  -> {path}")
        for n in rec["notes"]:
            print(f"  note: {n}")
        if rec["success"] and not a.no_results_md:
            append_results_md(rec)
        rc |= 0 if rec["success"] else 1
    if live_srv is not None:
        live_srv.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
