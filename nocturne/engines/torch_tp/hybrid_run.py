"""Run driver: one recorded run of a placement (session 6; since session 10 this process is only the operator).
Usage: python -m nocturne.engines.torch_tp.hybrid_run --placement experiments/session6_hybrid/placements/P1_loopback.yaml
         --run-id P1_loopback_qwen2.5-7b-bf16_r1 [--reduce-dtype bfloat16] [--reference R0_qwen2.5-7b-bf16_mps_r1 ...]
         [--save-reference] [--transformers] [--force] [--mps-fraction-cap] [--session-dir DIR] [--timeline]
The operator starts the rank processes and the coordinator process on their hosts (operate.py); the coordinator is the
host named in the placement and may be another machine, on macOS or Linux. It plans there, with its model file.
Writes <session-dir>/runs/<run_id>/record.json (default session dir experiments/session6_hybrid) and the logs next to it.
--timeline adds timeline_<rank>.jsonl.gz, samples_<host>.jsonl, trace.json.gz, breakdown.csv, timeline.html (docs/MEASUREMENT.md).
R0 runs (--save-reference) also write reference.json; first_logits.npz (first step logits, fp32) is always written."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SESSION6 = ROOT / "experiments" / "session6_hybrid"
OUT = SESSION6
RUNS = OUT / "runs"
HASHES = OUT / "model_hashes.json"
ENGINE = Path(__file__).resolve().parent
PROMPTS = ROOT / "nocturne" / "bench" / "prompts"
SHIP_DEFAULT = True


def sh(peer: dict, cmd: str, timeout: float = 120) -> subprocess.CompletedProcess:
    if peer.get("ssh"):
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer["ssh"], cmd],
                              capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)


def engine_hash_local() -> str:
    h = hashlib.sha256()
    for f in sorted(ENGINE.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


ENGINE_HASH_REMOTE = ("import hashlib, pathlib, nocturne.engines.torch_tp as m; d = pathlib.Path(m.__file__).parent; "
                      "h = hashlib.sha256(); [h.update(f.name.encode() + f.read_bytes()) for f in sorted(d.glob('*.py'))]; "
                      "print(h.hexdigest())")


def peer_os(peer: dict, run=None) -> str:
    """"Darwin" or "Linux": the `os` of the peers.yaml entry (macos, darwin, linux, wsl2, ubuntu), else `uname -s`."""
    run = run or sh
    v = str(peer.get("os") or "").lower()
    if v:
        return "Darwin" if v in ("macos", "darwin", "mac") else "Linux"
    r = run(peer, "uname -s", 20)
    return "Darwin" if r.stdout.strip() == "Darwin" else "Linux"


from .operate import shell_path  # noqa: E402


def _model_files(mdir: str) -> tuple[str, str]:
    """(quoted directory, file list): the weight files of a safetensors directory (a glob, left unquoted), or the one
    .gguf file as a quoted literal name."""
    import shlex
    if mdir.endswith(".gguf"):
        d, _, f = mdir.rpartition("/")
        return shell_path(d or "."), "-- " + shlex.quote(f)
    return shell_path(mdir), "*.json *.safetensors"


def stat_cmd(osname: str, mdir: str) -> str:
    fmt = "stat -f '%N %z %m'" if osname == "Darwin" else "stat -c '%n %s %Y'"
    d, files = _model_files(mdir)
    return f"cd {d} && {fmt} {files}"


def hash_cmd(osname: str, mdir: str) -> str:
    tool = "shasum -a 256" if osname == "Darwin" else "sha256sum"
    d, files = _model_files(mdir)
    return f"cd {d} && {tool} {files}"


def model_hashes(peer: dict, host: str, mdir: str, run=None, osname: str | None = None) -> dict:
    """sha256 of every file of the model directory on one host, cached by (size, mtime)."""
    run = run or sh
    osname = osname or peer_os(peer, run)
    cache = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    r = run(peer, stat_cmd(osname, mdir), 120)
    stat = r.stdout.strip()
    key = f"{host}:{mdir}"
    if stat and cache.get(key, {}).get("stat") == stat:
        return cache[key]["sha256"]
    r = run(peer, hash_cmd(osname, mdir), 1800)
    if r.returncode:
        raise RuntimeError(f"hashing {mdir} on {host} failed: {r.stderr[-300:]}")
    sums = {l.split(None, 1)[1].lstrip("*"): l.split(None, 1)[0] for l in r.stdout.splitlines() if l.strip()}
    cache[key] = {"stat": stat, "sha256": sums}
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    HASHES.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return sums


NVIDIA_SMI = "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader"


def host_state(peer: dict, local: bool, osname: str, has_cuda: bool, run=None, log=print) -> dict:
    """Sleep, AWDL, VPN and GPU state of one host. Mac only items are "n/a" on Linux."""
    from ...bench.probes import net
    run = run or sh
    info = {"os": osname}
    if osname == "Darwin":
        r = run(peer, "pgrep -x caffeinate", 120)
        if r.returncode:
            log(f"[run] caffeinate not running on {peer.get('name')}; starting it")
            if local:
                subprocess.Popen(["caffeinate", "-dimsu"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                run(peer, "nohup caffeinate -dimsu >/dev/null 2>&1 &", 120)
            time.sleep(1)
            r = run(peer, "pgrep -x caffeinate", 120)
        info["caffeinate"] = "on" if r.returncode == 0 else "off"
        if local:
            info["awdl"], info["vpn_state"] = net.awdl_state(), net.vpn_state()
        else:
            r = run(peer, "ifconfig awdl0 | head -1", 120)
            info["awdl"] = "up" if "<UP" in r.stdout or ",UP," in r.stdout else ("down" if r.stdout else "unknown")
            r = run(peer, "scutil --nc list", 120)
            info["vpn_state"] = "connected" if "(Connected)" in r.stdout else ("disconnected" if r.returncode == 0 else "unknown")
    else:
        info["caffeinate"] = info["awdl"] = info["vpn_state"] = "n/a"
    if has_cuda:
        r = run(peer, NVIDIA_SMI, 60)
        info["nvidia_smi"] = r.stdout.strip() if r.returncode == 0 else f"unavailable: {r.stderr.strip()[-200:]}"
    return info


TIMESYNC = "-m nocturne.bench.probes.timesync --json --samples 4"


def _json_line(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def model_hosts(op, ship: bool) -> list[str]:
    """Hosts that must hold the model file: only the coordinator host when the weights are shipped, else every host."""
    return [op.coord] if ship else sorted({s.host for s in op.slots})


def preflight(op, log, run=None, ship: bool = False, timeline: bool = False) -> dict:
    """ssh, sleep prevention (Mac), the same engine files and the clock state on every host of the placement; the model
    files on every host that needs them (all the same); what the coordinator host needs to tokenize."""
    run = run or sh
    out = {"hosts": {}, "coordinator": op.coord, "ship_weights": ship}
    local_hash = engine_hash_local()
    sums = {}
    for host in sorted({s.host for s in op.slots}):
        peer = op.peer(host)
        if peer.get("ssh"):
            r = run(peer, "true", 20)
            if r.returncode:
                raise RuntimeError(f"ssh to {host} failed: {r.stderr.strip()[-200:]}")
        osname = peer_os(peer, run)
        has_cuda = any(s.host == host and s.device.split(":")[0] == "cuda" for s in op.slots)
        info = host_state(peer, not peer.get("ssh"), osname, has_cuda, run, log)
        py = shell_path(op.python(host))
        r = run(peer, f"{py} -c \"{ENGINE_HASH_REMOTE}\"")
        remote = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if remote != local_hash:
            raise RuntimeError(f"engine files on {host} differ from the operator's ({remote[:12] or r.stderr.strip()[-120:]} vs "
                               f"{local_hash[:12]}); git pull on {host} (same SHA) or copy them")
        info["engine_sha256"] = local_hash
        r = run(peer, f"{py} {TIMESYNC}", 60)
        info["timesync"] = _json_line(r.stdout) or {"error": (r.stderr or r.stdout).strip()[-200:] or "no output"}
        if timeline:
            r = run(peer, f"{py} -c 'import psutil; print(psutil.__version__)'", 30)
            info["psutil"] = r.stdout.strip() if r.returncode == 0 else None
        if host == op.coord:
            r = run(peer, f"{py} -c 'import transformers; print(transformers.__version__)'", 60)
            if r.returncode:
                raise RuntimeError(f"the coordinator host {host} cannot import transformers (it tokenizes the prompts): {r.stderr.strip()[-200:]}")
            info["transformers"] = r.stdout.strip()
        if host in model_hosts(op, ship):
            from .launch import model_dir_of
            sums[host] = model_hashes(peer, host, model_dir_of(op.placement.model, op.peers, host), run, osname)
            info["model_sha256"] = sums[host]
        out["hosts"][host] = info
    if len({json.dumps(v, sort_keys=True) for v in sums.values()}) > 1:
        raise RuntimeError(f"model files differ between hosts: {sums}")
    return out


def clock_report(clocks: dict, pre: dict, coord: str) -> dict:
    """Adds to every rank's clock entry what the OS time sync predicts for it and the gap to the engine's measurement."""
    from . import clock
    sntp = {h: (v.get("timesync") or {}).get("sntp", {}).get("offset_s") for h, v in pre.get("hosts", {}).items()}
    bound = {h: (v.get("timesync") or {}).get("sntp", {}).get("bound_s") for h, v in pre.get("hosts", {}).items()}
    for name, c in clocks.items():
        m = c.get("end") or c.get("start")
        if not m:
            continue
        h = c.get("host")
        if h == coord:
            c["os"] = {"os_predicted_ns": 0, "gap_ns": m["offset_wall_ns"], "disagree": abs(m["offset_wall_ns"]) > clock.DISAGREE_NS,
                       "same_host": True}
        else:
            c["os"] = clock.compare_with_os(m["offset_wall_ns"], sntp.get(coord), sntp.get(h))
            c["os"]["os_bound_ns"] = round(((bound.get(coord) or 0) + (bound.get(h) or 0)) * 1e9) if c["os"]["gap_ns"] is not None else None
    return clocks


def load_prompts(subset: int, tokens: int, n_long: int, fixed: int) -> list[dict]:
    """The job's prompt list: `subset` eval prompts with `tokens` greedy tokens, `n_long` long prompts with `fixed` tokens."""
    rows = [json.loads(l) for l in (PROMPTS / "eval_set.jsonl").read_text().splitlines() if l.strip()][:subset]
    longs = [json.loads(l) for l in (PROMPTS / "eval_long.jsonl").read_text().splitlines() if l.strip()]
    longs = [r for r in longs if "id" in r][:n_long]
    out = [{"id": r["id"], "kind": "eval", "prompt": r["prompt"], "n": tokens, "ignore_eos": False, "cap": int(r.get("max_tokens") or tokens),
            "category": r["category"], "expected": r["expected"]} for r in rows]
    return out + [{"id": r["id"], "kind": "long", "prompt": r["prompt"], "n": fixed, "ignore_eos": True} for r in longs]


def find_run(run_id: str) -> Path:
    """The run folder of an earlier run: this session's runs first, then session 6. A run id with a slash is a path to
    a run folder, absolute or from the repository root (a reference recorded by another session)."""
    if "/" in run_id:
        for cand in (Path(os.path.expanduser(run_id)), ROOT / run_id):
            if cand.is_dir():
                return cand
        raise FileNotFoundError(f"no run folder at {run_id}")
    for base in (RUNS, SESSION6 / "runs"):
        if (base / run_id).exists():
            return base / run_id
    raise FileNotFoundError(f"no run folder {run_id} in {RUNS} or {SESSION6 / 'runs'}")


def load_reference(run_id: str) -> tuple[dict, dict]:
    """Prompts and first step logits of a reference run. A run saved with --save-reference has reference.json; any other
    completed run folder serves as well, from its record.json (the options-off run of a placement, for a run with options)."""
    d = find_run(run_id)
    if (d / "reference.json").exists():
        ref = json.loads((d / "reference.json").read_text())
    else:
        rec = json.loads((d / "record.json").read_text())
        if rec.get("outcome") != "completed":
            raise ValueError(f"reference run {run_id} did not complete (outcome {rec.get('outcome')})")
        ref = {"run_id": rec["run_id"], "prompts": {p["id"]: p for p in rec["prompts"]}}
    lg = dict(np.load(d / "first_logits.npz"))
    return ref, lg


def first_div(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--placement", required=True)
    ap.add_argument("--session-dir", default=str(SESSION6), help="where runs/ and model_hashes.json live; references are also "
                    "searched in experiments/session6_hybrid")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--peers", default="~/.nocturne/peers.yaml")
    ap.add_argument("--subset", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=32, help="greedy tokens per prompt for the identity check (EOS stops)")
    ap.add_argument("--long", type=int, default=1, help="eval_long prompts with --fixed tokens each")
    ap.add_argument("--fixed", type=int, default=256)
    ap.add_argument("--reduce-dtype", choices=["float32", "bfloat16"])
    ap.add_argument("--reference", nargs="*", default=None, help="R0 run ids to compare with (default: by model and device)")
    ap.add_argument("--save-reference", action="store_true")
    ap.add_argument("--transformers", action="store_true", help="R0: compare with transformers generate after the run (on this machine)")
    ap.add_argument("--transformers-dir", default=None, help="safetensors directory for --transformers (default: the placement's model; "
                    "needed when the placement reads a GGUF file)")
    ap.add_argument("--tokenizer-dir", default=None, help="tokenizer directory for the prompts, on the coordinator host (default: the "
                    "placement's model; the tokenizer of a GGUF file is built by transformers from its metadata)")
    ap.add_argument("--force", action="store_true", help="skip the planner's budget check (ranks still enforce budgets)")
    ap.add_argument("--mps-fraction-cap", action="store_true", help="also set torch's MPS memory fraction to the budget")
    ap.add_argument("--timeline", action="store_true", help="record compute, waits and bytes per rank and write the timeline files")
    ap.add_argument("--ship-weights", action=argparse.BooleanOptionalAction, default=SHIP_DEFAULT,
                    help="the coordinator sends every rank its slices; only the coordinator host holds the model")
    ap.add_argument("--cache-dir", default=None, help="slice cache of the ranks (default ~/.nocturne/cache on each host)")
    ap.add_argument("--verify-cache", action="store_true", help="every rank hashes its cached slices again before it trusts them "
                    "(one sequential read of the cached bytes) and asks again for any that differ")
    ap.add_argument("--stall-timeout", type=float, default=300.0, help="end the run when the coordinator process gives no sign of life "
                    "(output or its 20 s heartbeat) for this many seconds; 0 waits for ever. A liveness check, not a time limit")
    ap.add_argument("--slots", type=int, default=1, help="requests in flight (1: one at a time, as before). Each slot has its own KV cache")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="tokens per prefill chunk (0: the whole prompt at once, as before)")
    ap.add_argument("--tp-interleave", type=int, default=1, choices=[1, 2], help="2: two requests share a TP group in lockstep (needs --slots 2 or more)")
    ap.add_argument("--async-send", action="store_true", help="copy to the host and send off the compute thread")
    ap.add_argument("--dtype", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    from .plan import PlanError, check_overlap
    try:
        check_overlap(a.slots, a.prefill_chunk, a.tp_interleave)
    except PlanError as e:
        ap.error(str(e))
    global OUT, RUNS, HASHES
    OUT = Path(a.session_dir).resolve()
    RUNS, HASHES = OUT / "runs", OUT / "model_hashes.json"
    for r in a.reference or []:             # a reference that cannot be read costs a moment here, not a run
        try:
            load_reference(r)
        except (OSError, ValueError, KeyError) as e:
            ap.error(f"--reference {r}: {type(e).__name__}: {e}")

    from ...bench.check.graders import grade
    from .operate import Operator, valid_tag
    from .plan import PlanError

    if not valid_tag(a.run_id):      # before any folder or log is created: the id names folders here and on every host
        print(f"run id {a.run_id!r}: letters, digits, dot, underscore and dash only")
        return 2
    run_dir = RUNS / a.run_id
    if (run_dir / "record.json").exists():
        print(f"{run_dir} exists; run ids are never reused (raw run folders are evidence)")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = (run_dir / "coordinator.log").open("a")

    def log(*x):
        line = " ".join(str(v) for v in x)
        print(line, flush=True)
        logf.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        logf.flush()

    pl_path = Path(a.placement)
    rec = {"schema": "session6_hybrid/1", "run_id": a.run_id, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "placement": {"file": str(pl_path.resolve().relative_to(ROOT)) if pl_path.resolve().is_relative_to(ROOT) else str(pl_path),
                         "sha256": hashlib.sha256(pl_path.read_bytes()).hexdigest(), "yaml": pl_path.read_text()},
           "git": {"sha": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                   "dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "nocturne/engines"],
                                                capture_output=True, text=True).stdout.strip())},
           "args": vars(a), "outcome": None}

    def save():
        (run_dir / "record.json").write_text(json.dumps(rec, indent=1, default=str))

    try:
        op = Operator(str(pl_path), a.peers, run_dir, tag=a.run_id, log=log, ship_weights=a.ship_weights, cache_dir=a.cache_dir)
    except PlanError as e:
        rec["outcome"], rec["error"] = "plan_rejected", str(e)
        log(f"[run] the placement is rejected: {e}")
        save()
        return 3
    except Exception as e:      # noqa: BLE001 - a bad tag, a peers entry without bind or backend, a missing file
        rec["outcome"], rec["error"] = "setup_failed", f"{type(e).__name__}: {e}"
        log(f"[run] setup failed: {rec['error']}")
        save()
        return 4
    import socket
    rec["model"] = {"key": op.placement.model, "dir": op.model_dir}
    rec["coordinator"] = {"host": op.coord, "hostname": op.peer(op.coord).get("host"), "operator_hostname": socket.gethostname().split(".")[0],
                          "remote": bool(op.peer(op.coord).get("ssh")), "run_dir": op.host_dir(op.coord)}
    job = op.job(load_prompts(a.subset, a.tokens, a.long, a.fixed), force=a.force, mps_fraction_cap=a.mps_fraction_cap,
                 reduce_dtype=a.reduce_dtype, timeline=a.timeline, tokenizer_dir=a.tokenizer_dir, dtype=a.dtype,
                 verify_cache=a.verify_cache, slots=a.slots, prefill_chunk=a.prefill_chunk, tp_interleave=a.tp_interleave,
                 async_send=a.async_send)
    meta = {q["id"]: q for q in job["prompts"]}
    try:
        op.push_job(job)
        pl = op.plan()
    except Exception as e:      # noqa: BLE001
        pl = {"ok": False, "fatal": True, "error": f"{type(e).__name__}: {e}"}
    if not pl["ok"]:
        rec["outcome"], rec["error"] = ("preflight_failed" if pl.get("fatal") else "plan_rejected"), pl["error"]
        log(f"[run] {'the coordinator host could not plan' if pl.get('fatal') else 'planner rejected the placement'}: {pl['error']}")
        rec["commands"] = op.commands()
        save()
        return 4 if pl.get("fatal") else 3
    rec["reduce_dtype"] = pl["reduce_dtype"]
    rec["planned"] = pl["specs"]
    rec["planner_host"] = pl.get("planner_host")
    rec["planned_hosts"] = pl.get("hosts")
    for h, t in sorted((pl.get("hosts") or {}).items()):
        if t.get("unchecked"):
            log(f"[run] note: host {h} holds {t['processes']} rank processes ({t['total'] / 1e9:.3f} GB planned together) and the placement "
                f"gives no host_budget_gb for it: the sum is not checked")
    pred = pl["frames_predicted"]
    rec["frames"] = {"predicted": pred}
    try:
        rec["preflight"] = preflight(op, log, ship=a.ship_weights, timeline=a.timeline)
        rec["model"]["sha256"] = rec["preflight"]["hosts"][op.coord]["model_sha256"]
    except Exception as e:      # noqa: BLE001
        rec["outcome"], rec["error"] = "preflight_failed", str(e)
        log(f"[run] preflight failed: {e}")
        rec["commands"] = op.commands()
        save()
        return 4
    save()
    crec = {}
    try:
        op.start_ranks()
        rc = op.run_coordinator(a.stall_timeout)
        if op.stalled:               # saved now, before collection and cleanup can fail
            rec["outcome"], rec["error"] = "coordinator_stalled", op.stalled
            save()
        late = op.wait_ranks(5.0 if op.stalled else 60.0)
        if late:
            log(f"[run] ranks still running a minute after the coordinator ended: {late}")
        got = op.collect()
        rec["collected"] = got
        cpath = run_dir / "coordinator_record.json"
        if op.stalled:
            crec = {"outcome": "coordinator_stalled", "error": op.stalled}
        elif cpath.exists():
            crec = json.loads(cpath.read_text())
        else:
            crec = {"outcome": "error", "error": f"the coordinator process on {op.coord} ended with rc {rc} and left no record"}
    except Exception as e:      # noqa: BLE001
        crec = {"outcome": "error", "error": f"{type(e).__name__}: {e}"}
        log(traceback.format_exc())
    finally:
        try:
            op.stop()
        except Exception as e:      # noqa: BLE001
            log(f"[run] stop: {e}")
        rec["commands"] = op.commands()
    for k in ("load_s", "link_counters", "ranks", "requests", "shipping", "torch_coordinator", "error", "overlap", "throughput"):
        if k in crec:
            rec[k] = crec[k]
    rec["frames"].update({k: v for k, v in (crec.get("frames") or {}).items() if k != "predicted"})
    rec["outcome"] = crec.get("outcome") or "error"
    if rec.get("shipping"):       # two hashes of the model: the coordinator's own, and the operator's shell command on that host
        rec["shipping"]["agrees_with_preflight"] = rec["shipping"].get("model_files") == rec["preflight"]["hosts"][op.coord].get("model_sha256")
        by_name = {r["name"]: r for r in rec.get("ranks") or []}
        for name, sh in rec["shipping"].get("ranks", {}).items():      # shipped bytes are stored bytes; the loader narrows 1-D float tensors
            r = by_name.get(name)
            if r and "manifest_bytes" in sh:
                sh.update({"loaded_tensor_bytes": r["loaded_tensor_bytes"], "planned_weight_bytes": r["planned"]["weights"],
                           "stored_minus_loaded_bytes": sh["manifest_bytes"] - r["loaded_tensor_bytes"]})
    if crec.get("clock"):
        rec["clock"] = clock_report(crec["clock"], rec["preflight"], op.coord)
    out = crec.get("prompts") or []
    for item in out:
        q = meta[item["id"]]
        if item["kind"] == "eval":
            item.update({"category": q["category"], "cap": q["cap"], "score": grade(q["category"], item["text"], q["expected"])})
    rec["prompts"] = out
    save()
    if rec["outcome"] != "completed":
        log(f"[run] outcome {rec['outcome']}: {rec.get('error')}")
        return 1

    if a.timeline:
        from . import timeline
        try:
            rec["timeline"] = timeline.build(run_dir, rec.get("clock"), rec.get("requests"),
                                             {"run_id": a.run_id, "model": op.placement.model, "placement": rec["placement"]["file"],
                                              "date": rec["date"]})
            if rec["timeline"]:
                log(f"[run] timeline: {rec['timeline']['decode_bytes_per_step']:.0f} bytes per decode step by the recorder, "
                    f"{rec['frames'].get('bytes_per_step')} by the link counters")
        except Exception as e:      # noqa: BLE001 - a broken chart must not lose the run
            rec["timeline"] = {"error": f"{type(e).__name__}: {e}"}
            log(traceback.format_exc())
    logits = dict(np.load(run_dir / "first_logits.npz"))
    if a.save_reference:
        (run_dir / "reference.json").write_text(json.dumps({"run_id": a.run_id, "model": op.placement.model,
                                                            "device": op.slots[0].device, "prompts": {p["id"]: p for p in out}}, indent=1))
    if a.transformers:
        rec["transformers"] = transformers_check(op, out, a, log)
    refs = a.reference
    if refs is None:
        devs = sorted({s.device.split(":")[0] for s in op.slots})
        refs = [latest_reference(op.placement.model, d) for d in devs]
        refs = [r for r in refs if r and r != a.run_id]
    rec["correctness"] = {r: compare(out, logits, *load_reference(r)) for r in refs}
    for r, c in rec["correctness"].items():
        ld = "not compared" if c["logit_maxdiff"] is None else f"{c['logit_maxdiff']:.4g}"
        log(f"[run] vs {r}: identical {c['identical']}/{c['of']}, accuracy {c['accuracy']} (reference {c['ref_accuracy']}), "
            f"max first step logit diff {ld}, long prompt first divergence {c['long_first_div']}, "
            f"criterion {'met' if c['criterion_met'] else 'NOT met'}"
            + (f"; not in the reference: {c['not_in_reference']}" if c["not_in_reference"] else ""))
    log(f"[run] frames per decode step: measured {rec['frames']['measured_per_step']}, predicted {pred['total']}; "
        f"prefill frames: measured {rec['frames'].get('prefill_measured')}, predicted {rec['frames'].get('prefill_predicted')}")
    if (rec.get("overlap") or {}).get("on"):
        o, t = rec["overlap"], rec.get("throughput") or {}
        log(f"[run] overlap: slots {o['slots']}, prefill chunk {o['prefill_chunk']}, tp interleave {o['tp_interleave']}, async send {o['async_send']}; "
            f"{t.get('tokens')} tokens in {t.get('wall_s', 0):.1f} s")
    save()
    return 0


def latest_reference(model: str, device: str) -> str | None:
    c = []
    for base in dict.fromkeys([RUNS, SESSION6 / "runs"]):
        c = sorted(base.glob(f"R0_{model}_{device}_r*/reference.json"), key=lambda p: p.stat().st_mtime)
        if c:
            break
    return c[-1].parent.name if c else None


def compare(out: list[dict], logits: dict, ref: dict, ref_logits: dict) -> dict:
    rp = ref["prompts"]
    missing = [p["id"] for p in out if p["id"] not in rp]       # a reference recorded with fewer prompts: compare what it has
    ev_ids = {p["id"] for p in out if p["kind"] == "eval"}
    out = [p for p in out if p["id"] in rp]
    ev = [p for p in out if p["kind"] == "eval"]
    same = [p["tokens"] == rp[p["id"]]["tokens"] for p in ev]
    divs = {p["id"]: first_div(p["tokens"], rp[p["id"]]["tokens"]) for p in out}
    scores = [p["score"] for p in ev if p["score"] is not None]
    rscores = [rp[p["id"]]["score"] for p in ev if rp[p["id"]].get("score") is not None]
    ld = {k: float(np.abs(v - ref_logits[k]).max()) for k, v in logits.items() if k in ref_logits}
    acc = sum(scores) / len(scores) if scores else None
    racc = sum(rscores) / len(rscores) if rscores else None
    longs = [p for p in out if p["kind"] == "long"]
    return {"reference": ref["run_id"], "identical": sum(same), "of": len(ev), "first_div": divs, "accuracy": acc,
            "ref_accuracy": racc, "logit_maxdiff": max(ld.values()) if ld else None, "logit_maxdiff_per_prompt": ld,
            "long_first_div": {p["id"]: divs[p["id"]] for p in longs},
            # None when the reference holds no long prompt of this run: nothing was compared, which is not "identical"
            "long_identical": all(p["tokens"] == rp[p["id"]]["tokens"] for p in longs) if longs else None,
            "long_compared": len(longs), "not_in_reference": missing,
            "criterion_met": bool(ev) and not any(i in ev_ids for i in missing) and sum(same) >= min(7, len(ev)) and acc == racc}


def transformers_check(L, out: list[dict], a, log) -> dict:
    """One rank reference against transformers generate on the same device, same prompts, same token counts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import chat_ids
    dev = L.slots[0].device
    tdir = os.path.expanduser(a.transformers_dir if a.transformers_dir else L.model_dir)
    from .gguf_source import load_tokenizer
    tok = load_tokenizer(os.path.expanduser(a.tokenizer_dir) if a.tokenizer_dir else tdir)
    model = AutoModelForCausalLM.from_pretrained(tdir, torch_dtype=torch.bfloat16).to(dev).eval()
    rows = {json.loads(l)["id"]: json.loads(l) for l in (PROMPTS / "eval_set.jsonl").read_text().splitlines() if l.strip()}
    longs = {r["id"]: r for r in (json.loads(l) for l in (PROMPTS / "eval_long.jsonl").read_text().splitlines() if l.strip()) if "id" in r}
    res = {}
    for p in out:
        r = rows.get(p["id"]) or longs[p["id"]]
        ids = chat_ids(tok, [{"role": "user", "content": r["prompt"]}])
        n = len(p["tokens"])
        with torch.inference_mode():
            kw = {"min_new_tokens": n} if p["kind"] == "long" else {}
            g = model.generate(torch.tensor([ids], device=dev), max_new_tokens=n, do_sample=False, num_beams=1,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
        ref = g[0, len(ids):].tolist()
        res[p["id"]] = {"identical": ref == p["tokens"], "first_div": first_div(ref, p["tokens"])}
        log(f"[run] transformers {p['id']}: {'identical' if ref == p['tokens'] else 'differs at ' + str(res[p['id']]['first_div'])}")
    del model
    return {"identical": sum(v["identical"] for k, v in res.items() if k in rows), "of": sum(1 for k in res if k in rows), "per_prompt": res}


if __name__ == "__main__":
    sys.exit(main())
