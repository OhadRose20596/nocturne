"""The coordinator side of one recorded run (session 10, part B). Runs on the host the placement names, on macOS or Linux.
Usage: python -m nocturne.engines.torch_tp.coordinate --job <dir>/job.json [--plan-only] [--tag T]
It reads a job file written by the operator (hybrid_run on any machine), plans with the model file of this host, links to
the rank processes the operator started, runs the prompts, and writes coordinator_record.json and first_logits.npz next
to the job file. Only this process needs the model file and the tokenizer. docs/ENGINE_DESIGN.md has the whole flow."""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

PLAN_MARK = "PLAN_JSON "


def short_hostname() -> str:
    return socket.gethostname().split(".")[0]


def check_host(job: dict, hostname: str | None = None) -> None:
    """The coordinator must run on the host the placement names (peers.<host>.host is its hostname)."""
    want = job.get("coordinator_hostname")
    me = hostname or short_hostname()
    if want and want.split(".")[0].lower() != me.lower():
        raise RuntimeError(f"the coordinator host is {job.get('coordinator')} ({want}); this machine is {me}")


def model_info(job: dict):
    """(placement, model path on this host, config, dims). Needs the model file."""
    from .config import load_config
    from .gguf_source import weight_meta
    from .plan import Dims, Placement, placement_from_dict
    p = placement_from_dict(job["placement"])
    if job.get("reduce_dtype"):
        p = Placement(**{**p.__dict__, "reduce_dtype": job["reduce_dtype"]})
    path = os.path.expanduser(job["model_dir"])
    cfg = load_config(path)
    return p, path, cfg, Dims.of(cfg, weight_meta(path))


def plan_job(job: dict) -> dict:
    """The planner's answer for a job: the rank specs and the predicted frames, or the rejection text."""
    from .plan import PlanError, format_plan, frames_per_step, plan
    p, path, cfg, dims = model_info(job)
    try:
        specs = plan(dims, p, check_budget=not job.get("force"))
    except PlanError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "specs": [s.to_json() for s in specs], "frames_predicted": frames_per_step(specs), "model_dir": path,
            "reduce_dtype": p.reduce_dtype, "table": format_plan(specs, dims), "planner_host": short_hostname()}


def run_job(job: dict, run_dir: str | Path, log=print) -> dict:
    """Plan, link, run the prompts, measure. Returns the coordinator record (also written to coordinator_record.json)."""
    import numpy as np
    import torch
    from .engine import chat_ids
    from .gguf_source import load_tokenizer
    from .hybrid import Coordinator, RankFailed, decode_frames
    from .plan import PlanError, RankSpec, frames_per_step, listen_addrs, plan
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"outcome": None, "host": short_hostname(), "torch_coordinator": torch.__version__, "pid": os.getpid(),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def save():
        (run_dir / "coordinator_record.json").write_text(json.dumps(rec, indent=1, default=str))

    co = None
    try:
        check_host(job)
        p, path, cfg, dims = model_info(job)
        specs = plan(dims, p, check_budget=not job.get("force"))
        rec["planned"] = [s.to_json() for s in specs]
        rec["reduce_dtype"] = p.reduce_dtype
        pred = frames_per_step(specs)
        rec["frames"] = {"predicted": pred}
        t0 = time.time()
        co = Coordinator(specs, path, job["hosts"], listen_addrs(specs, job["hosts"])[0], p.max_seq_len, p.reduce_dtype, p.prefill_max,
                         device=specs[0].device, log=log, enforce_budget=True, mps_fraction_cap=bool(job.get("mps_fraction_cap")),
                         dtype=job.get("dtype") or "bfloat16", timeline=bool(job.get("timeline")), run_dir=str(run_dir),
                         run_tag=job.get("tag"), **shipping_args(job))
        rec["load_s"] = time.time() - t0
        tok = None
        if any("prompt" in q for q in job["prompts"]):
            tok = load_tokenizer(os.path.expanduser(job["tokenizer_dir"]) if job.get("tokenizer_dir") else path)
        m0 = co.metrics()
        out, logits = [], {}
        for q in job["prompts"]:
            ids = [int(x) for x in q["ids"]] if "ids" in q else chat_ids(tok, [{"role": "user", "content": q["prompt"]}])
            res = co.generate(ids, int(q["n"]), ignore_eos=bool(q.get("ignore_eos")), rid=q["id"], first_logits=True)
            logits[q["id"]] = res["first_logits"].numpy().astype(np.float32)
            item = {"id": q["id"], "kind": q.get("kind", "eval"), "prompt_n": len(ids), "tokens": res["tokens"],
                    "finish": res["finish_reason"], "ms_per_step": res["timings"]["predicted_per_token_ms"], "text": None}
            if tok is not None:
                keep = res["tokens"] if q.get("ignore_eos") else [t for t in res["tokens"][:int(q.get("cap") or q["n"])] if t not in cfg.eos_token_ids]
                item["text"] = tok.decode(keep, skip_special_tokens=True)
            out.append(item)
            log(f"[run] {q['id']}: {len(res['tokens'])} tokens")
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
        rec["requests"] = co.requests
        rec["eos_token_ids"] = list(cfg.eos_token_ids)
        np.savez_compressed(run_dir / "first_logits.npz", **logits)
        rec["outcome"] = "completed"
        try:        # the results are stored; a link that drops now costs only the end clock measurement
            co.clock_sync()
        except (OSError, RuntimeError) as e:
            co.clock["end"] = {}
            rec["clock_end_error"] = f"{type(e).__name__}: {e}"
            log(f"[run] end clock measurement failed: {e}")
    except PlanError as e:
        rec["outcome"], rec["error"] = "plan_rejected", str(e)
        log(f"[run] planner rejected the placement: {e}")
    except RankFailed as e:
        rec["outcome"], rec["error"] = "rank_failed", str(e)
        log(f"[run] a rank failed: {e}")
    except (ConnectionError, OSError) as e:
        rec["outcome"], rec["error"] = "rpc_disconnect", f"{type(e).__name__}: {e}"
        log(f"[run] link lost: {e}")
    except Exception as e:      # noqa: BLE001 - the record names every failure
        rec["outcome"], rec["error"] = "error", f"{type(e).__name__}: {e}"
        log(traceback.format_exc())
    finally:
        if co is not None:
            try:
                co.shutdown()
            except Exception as e:      # noqa: BLE001
                log(f"[run] shutdown: {e}")
            rec["clock"] = co.clock_record()
            rec["shipping"] = getattr(co, "shipping", None)
        save()
    return rec


def exit_on_sigterm() -> None:
    """SIGTERM becomes a normal exit, so `finally` blocks and atexit handlers run (the sampler stops its nvidia-smi)."""
    import signal
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))


def shipping_args(job: dict) -> dict:
    return {"ship_weights": bool(job.get("ship_weights")), "cache_dir": job.get("cache_dir")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--tag", default="", help="run tag, lets the operator find the process")
    a = ap.parse_args(argv)
    exit_on_sigterm()
    job_path = Path(os.path.expanduser(a.job))
    job = json.loads(job_path.read_text())

    def log(*x):
        print(" ".join(str(v) for v in x), flush=True)

    if a.plan_only:
        try:
            check_host(job)
            res = plan_job(job)
        except Exception as e:      # noqa: BLE001
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "fatal": True}
        (job_path.parent / "plan.json").write_text(json.dumps(res, indent=1))
        print(PLAN_MARK + json.dumps(res), flush=True)
        return 0 if res["ok"] else 3
    rec = run_job(job, job_path.parent, log)
    return {"completed": 0, "plan_rejected": 3}.get(rec["outcome"], 1)


if __name__ == "__main__":
    sys.exit(main())
