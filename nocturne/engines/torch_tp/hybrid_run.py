"""Session 6 run driver: one recorded run of a placement.
Usage: python -m nocturne.engines.torch_tp.hybrid_run --placement experiments/session6_hybrid/placements/P1_loopback.yaml
         --run-id P1_loopback_qwen2.5-7b-bf16_r1 [--reduce-dtype bfloat16] [--reference R0_qwen2.5-7b-bf16_mps_r1 ...]
         [--save-reference] [--transformers] [--force] [--mps-fraction-cap] [--session-dir DIR]
Writes <session-dir>/runs/<run_id>/record.json (default session dir experiments/session6_hybrid) and the launcher and rank logs next to it. R0 runs
(--save-reference) also write reference.json and first_logits.npz (first step logits, fp32)."""
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


def _model_files(mdir: str) -> tuple[str, str]:
    """(directory, file glob): the weight files of a safetensors directory, or the one .gguf file."""
    if mdir.endswith(".gguf"):
        d, _, f = mdir.rpartition("/")
        return d or ".", f
    return mdir, "*.json *.safetensors"


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
    sums = {l.split()[1].lstrip("*"): l.split()[0] for l in r.stdout.splitlines() if l.strip()}
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


def preflight(L, log, run=None) -> dict:
    """Sleep prevention (Mac) and ssh on every host of the placement, the same engine files, the same model files."""
    run = run or sh
    out = {"hosts": {}}
    local_hash = engine_hash_local()
    sums = {}
    for host in sorted({s.host for s in L.specs}):
        peer = L._peer(host)
        if peer.get("ssh"):
            r = run(peer, "true", 20)
            if r.returncode:
                raise RuntimeError(f"ssh to {host} failed: {r.stderr.strip()[-200:]}")
        osname = peer_os(peer, run)
        has_cuda = any(s.host == host and s.device.split(":")[0] == "cuda" for s in L.specs)
        info = host_state(peer, not peer.get("ssh"), osname, has_cuda, run, log)
        if peer.get("ssh"):
            r = run(peer, f"{peer['python']} -c \"{ENGINE_HASH_REMOTE}\"")
            remote = r.stdout.strip()
            if remote != local_hash:
                raise RuntimeError(f"engine files on {host} differ from this machine ({remote[:12]} vs {local_hash[:12]}); "
                                   f"git pull on {host} (same SHA) or copy them")
        info["engine_sha256"] = local_hash
        sums[host] = model_hashes(peer, host, L.model_dir, run, osname)
        info["model_sha256"] = sums[host]
        out["hosts"][host] = info
    if len({json.dumps(v, sort_keys=True) for v in sums.values()}) > 1:
        raise RuntimeError(f"model files differ between hosts: {sums}")
    return out


def find_run(run_id: str) -> Path:
    """The run folder of an earlier run: this session's runs first, then session 6."""
    for base in (RUNS, SESSION6 / "runs"):
        if (base / run_id).exists():
            return base / run_id
    raise FileNotFoundError(f"no run folder {run_id} in {RUNS} or {SESSION6 / 'runs'}")


def load_reference(run_id: str) -> tuple[dict, dict]:
    d = find_run(run_id)
    ref = json.loads((d / "reference.json").read_text())
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
    ap.add_argument("--transformers", action="store_true", help="R0: compare with transformers generate after the run")
    ap.add_argument("--transformers-dir", default=None, help="safetensors directory for --transformers (default: the placement's model; "
                    "needed when the placement reads a GGUF file)")
    ap.add_argument("--tokenizer-dir", default=None, help="tokenizer directory for the prompts (default: the placement's model; the "
                    "tokenizer of the GGUF file is built by transformers from its metadata)")
    ap.add_argument("--force", action="store_true", help="skip the planner's budget check (ranks still enforce budgets)")
    ap.add_argument("--mps-fraction-cap", action="store_true", help="also set torch's MPS memory fraction to the budget")
    a = ap.parse_args(argv)
    global OUT, RUNS, HASHES
    OUT = Path(a.session_dir).resolve()
    RUNS, HASHES = OUT / "runs", OUT / "model_hashes.json"

    import torch
    from .gguf_source import load_tokenizer
    from ...bench.check.graders import grade
    from .engine import chat_ids
    from .hybrid import RankFailed, decode_frames
    from .launch import Launch
    from .plan import PlanError, frames_per_step

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
           "torch_coordinator": torch.__version__, "args": vars(a), "outcome": None}

    def save():
        (run_dir / "record.json").write_text(json.dumps(rec, indent=1, default=str))

    try:
        L = Launch(str(pl_path), a.peers, str(run_dir), force=a.force, mps_fraction_cap=a.mps_fraction_cap,
                   reduce_dtype=a.reduce_dtype, tag=a.run_id, log=log)
    except PlanError as e:
        rec["outcome"], rec["error"] = "plan_rejected", str(e)
        log(f"[run] planner rejected the placement: {e}")
        save()
        return 3
    rec["model"] = {"key": L.placement.model, "dir": L.model_dir}
    rec["reduce_dtype"] = L.placement.reduce_dtype
    rec["planned"] = [s.to_json() for s in L.specs]
    pred = frames_per_step(L.specs)
    rec["frames"] = {"predicted": pred}
    try:
        rec["preflight"] = preflight(L, log)
        rec["model"]["sha256"] = rec["preflight"]["hosts"][L.specs[0].host]["model_sha256"]
    except Exception as e:      # noqa: BLE001
        rec["outcome"], rec["error"] = "preflight_failed", str(e)
        log(f"[run] preflight failed: {e}")
        save()
        return 4
    save()
    co = None
    try:
        t0 = time.time()
        co = L.start()
        rec["load_s"] = time.time() - t0
        tok = load_tokenizer(os.path.expanduser(a.tokenizer_dir) if a.tokenizer_dir else L.local_model_dir)
        rows = [json.loads(l) for l in (PROMPTS / "eval_set.jsonl").read_text().splitlines() if l.strip()][:a.subset]
        longs = [json.loads(l) for l in (PROMPTS / "eval_long.jsonl").read_text().splitlines() if l.strip()]
        longs = [r for r in longs if "id" in r][:a.long]
        m0 = co.metrics()
        out, logits = [], {}
        for r, fixed in [(r, None) for r in rows] + [(r, a.fixed) for r in longs]:
            ids = chat_ids(tok, [{"role": "user", "content": r["prompt"]}])
            n = fixed or a.tokens
            res = co.generate(ids, n, ignore_eos=bool(fixed), rid=r["id"], first_logits=True)
            logits[r["id"]] = res["first_logits"].numpy().astype(np.float32)
            item = {"id": r["id"], "kind": "long" if fixed else "eval", "prompt_n": len(ids), "tokens": res["tokens"],
                    "finish": res["finish_reason"], "ms_per_step": res["timings"]["predicted_per_token_ms"]}
            if not fixed:
                cap = int(r.get("max_tokens") or a.tokens)
                text = tok.decode([t for t in res["tokens"][:cap] if t not in co.cfg.eos_token_ids], skip_special_tokens=True)
                item.update({"category": r["category"], "cap": cap, "text": text, "score": grade(r["category"], text, r["expected"])})
            else:
                item["text"] = tok.decode(res["tokens"], skip_special_tokens=True)
            out.append(item)
            log(f"[run] {r['id']}: {len(res['tokens'])} tokens")
        m1 = co.metrics()
        steps = m1["decode_steps"] - m0["decode_steps"]
        f0, f1 = decode_frames(m0), decode_frames(m1)
        rec["frames"].update({"decode_steps": steps, "measured_per_step": (f1["sent_frames"] - f0["sent_frames"]) / steps if steps else None,
                              "bytes_per_step": (f1["sent_bytes"] - f0["sent_bytes"]) / steps if steps else None,
                              "per_link_per_step": {k: {kk: (v[kk] - f0["per_link"].get(k, {}).get(kk, 0)) / steps for kk in v}
                                                    for k, v in f1["per_link"].items()} if steps else None})
        rec["frames"]["formula_holds"] = rec["frames"]["measured_per_step"] == pred["total"]
        rec["link_counters"] = {k: v["links"] for k, v in m1["ranks"].items()}
        rec["ranks"] = [{k: v for k, v in r.items() if k != "links"} for r in m1["ranks"].values()]
        rec["prompts"] = out
        rec["outcome"] = "completed"
    except RankFailed as e:
        rec["outcome"], rec["error"] = "rank_failed", str(e)
        log(f"[run] a rank failed: {e}")
    except (ConnectionError, OSError) as e:
        rec["outcome"], rec["error"] = "rpc_disconnect", f"{type(e).__name__}: {e}"
        log(f"[run] link lost: {e}")
    except Exception as e:      # noqa: BLE001
        rec["outcome"], rec["error"] = "error", f"{type(e).__name__}: {e}"
        log(traceback.format_exc())
    finally:
        try:
            L.stop()
        except Exception as e:      # noqa: BLE001
            log(f"[run] stop: {e}")
        rec["commands"] = L.commands()
        save()
    if rec["outcome"] != "completed":
        return 1

    if a.save_reference:
        (run_dir / "reference.json").write_text(json.dumps({"run_id": a.run_id, "model": L.placement.model,
                                                            "device": L.specs[0].device, "prompts": {p["id"]: p for p in out}}, indent=1))
        np.savez_compressed(run_dir / "first_logits.npz", **logits)
    if a.transformers:
        import gc
        L.coordinator.rank.model = None       # free the engine's weights before transformers loads its own copy
        L.coordinator = co = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        rec["transformers"] = transformers_check(L, out, a, log)
    refs = a.reference
    if refs is None:
        devs = sorted({s.device.split(":")[0] for s in L.specs})
        refs = [latest_reference(L.placement.model, d) for d in devs]
        refs = [r for r in refs if r and r != a.run_id]
    rec["correctness"] = {r: compare(out, logits, *load_reference(r)) for r in refs}
    for r, c in rec["correctness"].items():
        log(f"[run] vs {r}: identical {c['identical']}/{c['of']}, accuracy {c['accuracy']} (reference {c['ref_accuracy']}), "
            f"max first step logit diff {c['logit_maxdiff']:.4g}, long prompt first divergence {c['long_first_div']}, "
            f"criterion {'met' if c['criterion_met'] else 'NOT met'}")
    log(f"[run] frames per decode step: measured {rec['frames']['measured_per_step']}, predicted {pred['total']}")
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
    ev = [p for p in out if p["kind"] == "eval"]
    rp = ref["prompts"]
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
            "criterion_met": sum(same) >= min(7, len(ev)) and acc == racc}


def transformers_check(L, out: list[dict], a, log) -> dict:
    """One rank reference against transformers generate on the same device, same prompts, same token counts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import chat_ids
    dev = L.specs[0].device
    tdir = os.path.expanduser(a.transformers_dir) if a.transformers_dir else L.local_model_dir
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.tokenizer_dir) if a.tokenizer_dir else tdir)
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
