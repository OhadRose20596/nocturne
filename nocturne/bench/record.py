"""Record writer. One JSON record per run, validated against schema/record.schema.json (v3) before writing.
File name: <cell>_<network>_<model>_r<n>.json under results/. Readers call upgrade() so v1 and v2 records
(schema/record.schema.v1.json, record.schema.v2.json) present v3 field names; files on disk are never rewritten."""
import datetime
import json
import re
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema" / "record.schema.json"
SCHEMA_V1_PATH = HERE / "schema" / "record.schema.v1.json"
SCHEMA_V2_PATH = HERE / "schema" / "record.schema.v2.json"
SCHEMA_VERSION = 3
RESULTS_DIR = HERE / "results"
FAILURE_CLASSES = (None, "load_fail", "oom", "network_abort", "engine_abort", "wrong_output", "tool_unsupported",
                   "unmeasured_memory")


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate(rec: dict) -> None:
    v = rec.get("schema_version", SCHEMA_VERSION)
    path = {1: SCHEMA_V1_PATH, 2: SCHEMA_V2_PATH}.get(v, SCHEMA_PATH)
    jsonschema.validate(rec, json.loads(path.read_text()), cls=jsonschema.Draft202012Validator)


def pooled_timing(per_prompt: dict) -> dict:
    """decode_ms_per_step_pooled = sum(predicted_ms) / sum(predicted_n - 1) over prompts with predicted_n >= 2.
    predicted_ms covers the steps after the first token (llama-server and torch_tp alike), so each prompt has n - 1
    steps; the per prompt view adds decode_ms_per_step to each entry (null when predicted_n < 2)."""
    ms = steps = 0
    for v in (per_prompt or {}).values():
        tm = (v or {}).get("timings") or {}
        n, t = tm.get("predicted_n"), tm.get("predicted_ms")
        if v.get("error") or not n or t is None:
            v["decode_ms_per_step"] = None
            continue
        v["decode_ms_per_step"] = (t / (n - 1)) if n >= 2 else None
        if n >= 2:
            ms += t
            steps += n - 1
    pooled = (ms / steps) if steps else None
    return {"decode_ms_per_step_pooled": pooled, "decode_tps_pooled": (1000.0 / pooled) if pooled else None,
            "decode_steps_total": steps or None}


V3_TIMING = {"decode_ms_per_step_pooled": None, "decode_tps_pooled": None, "decode_steps_total": None, "warmup_s": None}
V3_COMM = {"blocking_syncs_per_step": None, "messages_per_step": None, "bytes_per_step_decode": None,
           "bytes_per_step_decode_out": None, "bytes_per_step_decode_in": None, "graph_meta_bytes_per_step": None,
           "worker_capture": False, "clock_offset_s": None, "clock_offset_unc_s": None}
V3_NETWORK = {"rtt_source": "probe", "ping_decode_ms": None, "awdl": None, "added_delay_ms": None, "added_delay_measured_ms": None}


def upgrade(rec: dict) -> dict:
    """v1 -> v2 -> v3 view of a record, in memory only. Values an older record did not measure stay null; the pooled
    timing is derived from per_prompt, which every version carries."""
    if rec.get("schema_version", 1) >= SCHEMA_VERSION:
        return rec
    r = json.loads(json.dumps(rec))
    if r.get("schema_version", 1) == 1:
        r = _upgrade_v1_to_v2(r)
    r["schema_version"] = SCHEMA_VERSION
    for k, v in V3_TIMING.items():
        r["timing"].setdefault(k, v)
    r["timing"].update({k: v for k, v in pooled_timing(r.get("per_prompt", {})).items()})
    for k, v in V3_COMM.items():
        r["comm"].setdefault(k, v)
    for k, v in V3_NETWORK.items():
        r["network"].setdefault(k, v)
    r["network"]["rtt_source"] = "probe" if r["network"].get("rtt_ms_median") is not None else None
    r["prompts"].setdefault("fixed_tokens", None)
    r["comm"]["source"] = (r["comm"].get("source") or "") + " [v%d record, v3 view in memory]" % rec.get("schema_version", 1)
    return r


def _upgrade_v1_to_v2(r: dict) -> dict:
    r["schema_version"] = 2
    r["cell"].setdefault("name", "BASELINE" if r["cell"]["row"] == "BASELINE" else f"{r['cell']['row']}{r['cell']['col']}")
    c = r.get("comm", {})
    steps, toks = c.get("compute_steps"), c.get("tokens_generated")
    per_step = c.get("msgs_per_step_median")
    tps = (toks / steps) if steps and toks else None
    bpt = c.get("bytes_per_token_median")
    r["comm"] = {"rpc_commands_per_step": per_step, "tokens_per_step": tps,
                 "rpc_commands_per_token": c.get("msgs_per_token_median"),
                 "bytes_per_step": (bpt * tps) if bpt is not None and tps else None, "bytes_per_token": bpt,
                 "packets_per_token": c.get("packets_per_token_median"), "serial_waits_per_token": None,
                 "compute_steps": steps, "tokens_generated": toks, "capture_iface": None,
                 "source": (c.get("source") or "") + " [v1 record, names upgraded in memory]"}
    st = r.get("setup", {})
    r["setup"] = {"load_total_s": st.get("weight_transfer_s"), "transfer_s": None,
                  "cache_state": st.get("cache_state"), "bytes_sent_gb": st.get("bytes_sent_gb")}
    t = r.get("timing", {})
    r["timing"] = {"decode_tps_median": t.get("decode_tps_median"), "prefill_tps_median": t.get("prefill_tps_median"),
                   "ttft_s_median": None, "ttft_server_s_median": t.get("ttft_s_median")}
    r["memory_buffers_gb"] = r.pop("memory_peak_gb", {})
    r["memory_peak_gb"] = {h: None for h in r["memory_buffers_gb"]}
    cr = r.get("correctness", {})
    acc, base = cr.get("task_accuracy"), cr.get("baseline_task_accuracy")
    r["quality_pass"] = (acc >= base - 0.05) if acc is not None and base is not None else None
    # v1 success already required the accuracy rule, so success is unchanged; it is now also true for records that
    # failed only on wrong_output through the threshold (none exist as of 2026-10-26).
    r["prompts"].setdefault("ids", sorted(r.get("outputs", {})))
    r["prompts"].setdefault("generation", {"temperature": 0, "seed": 42, "samplers": ["temperature"], "repeat_penalty": 1.0})
    r["checks"] = {"a2_identity": None, "memory_measured": False, "baseline_id": None, "a1_id": None}
    for n in r.get("notes", []):
        if n.startswith("A2 greedy output identical to A1"):
            r["checks"]["a2_identity"] = True
        elif n.startswith("A2 greedy output differs from A1"):
            r["checks"]["a2_identity"] = False
    return r


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def record_id(cell: str, network: str, model: str, run: int, host: str | None = None) -> str:
    # host (the requester peer name) keeps records from different machines apart: results/ lives in iCloud,
    # so two Macs writing BASELINE_local_<model>_r1.json would overwrite each other.
    h = f"_{safe(host)}" if host else ""
    return f"{safe(cell)}_{safe(network)}_{safe(model)}{h}_r{run}"


def empty_record(cell_name: str, cell_cfg: dict, network: str, model_name: str, run: int, host: str | None = None) -> dict:
    """A record with every required field present and null where not yet measured."""
    row = "BASELINE" if cell_name == "BASELINE" else cell_name[0]
    col = 0 if cell_name == "BASELINE" else int(re.match(r"[A-D](\d)", cell_name).group(1))    # A2b -> col 2
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id(cell_name, network, model_name, run, host),
        "cell": {"partition": cell_cfg["partition"], "decode": cell_cfg["decode"], "row": row, "col": col, "name": cell_name},
        "date": now_iso(),
        "model": {"name": model_name, "quant": "", "sha256": None},
        "draft_model": None,
        "engine": {"name": "", "commit": "", "flags": []},
        "peers": [],
        "network": dict({"type": network, "rtt_ms_median": None, "bw_mbps": None, "measured_at": None}, **V3_NETWORK),
        "prompts": {"set": "", "version": "", "subset": None, "ids": [], "generation": {}, "fixed_tokens": None},
        "success": False,
        "failure_class": None,
        "quality_pass": None,
        "setup": {"load_total_s": None, "transfer_s": None, "cache_state": None, "bytes_sent_gb": None},
        "timing": dict({"decode_tps_median": None, "prefill_tps_median": None, "ttft_s_median": None, "ttft_server_s_median": None},
                       **V3_TIMING),
        "comm": dict({"rpc_commands_per_step": None, "tokens_per_step": None, "rpc_commands_per_token": None,
                      "bytes_per_step": None, "bytes_per_token": None, "packets_per_token": None,
                      "serial_waits_per_token": None, "compute_steps": None, "tokens_generated": None,
                      "capture_iface": None, "source": None}, **V3_COMM),
        "memory_buffers_gb": {},
        "memory_peak_gb": {},
        "utilization": {},
        "correctness": {"exact_match_ratio": None, "first_divergence_median": None,
                        "task_accuracy": None, "baseline_task_accuracy": None},
        "fault_test": None,
        "checks": {"a2_identity": None, "memory_measured": False, "baseline_id": None, "a1_id": None},
        "notes": [],
        "outputs": {},
        "per_prompt": {},
        "commands": [],
    }


def next_run_index(results_dir: Path, cell: str, network: str, model: str, host: str | None = None) -> int:
    h = f"_{safe(host)}" if host else ""
    prefix = f"{safe(cell)}_{safe(network)}_{safe(model)}{h}_r"
    n = 0
    for p in results_dir.glob(prefix + "*.json"):
        m = re.search(r"_r(\d+)\.json$", p.name)
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def write(rec: dict, results_dir: Path = RESULTS_DIR) -> Path:
    validate(rec)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{rec['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def read_all(results_dir: Path = RESULTS_DIR, raw: bool = False) -> list[dict]:
    """All records, upgraded to the v3 view unless raw=True. Files are never modified."""
    recs = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "id" not in d or "cell" not in d:
            continue          # derived_v3.json and other non record files live in the same directory
        recs.append(d if raw else upgrade(d))
    return recs


def same_commit(a: str, b: str) -> bool:
    """git rev-parse --short gives 7 chars on one Mac and 9 on the other; compare the common 7-char prefix."""
    a, b = (a or "").lower(), (b or "").lower()
    return bool(a and b) and a[:7] == b[:7]


def same_host(a: str, b: str) -> bool:
    return (a or "").lower() == (b or "").lower()


def same_engine(a: str | None, b: str | None) -> bool:
    """Engine name equality (ADR-007 amendment 2: each engine has its own BASELINE). A missing name on either side
    matches, so a caller that does not know the engine yet (recompute of a v1 record) still finds the llama.cpp one."""
    if not a or not b:
        return True
    return a.strip().lower() == b.strip().lower()


def _gen_equal(a: dict | None, b: dict | None) -> bool:
    """Deterministic generation flags must match when both records carry them; v1 records carry the fixed set."""
    if not a or not b:
        return True
    keys = set(a) | set(b)
    return all(a.get(k) == b.get(k) for k in keys)


def matches_baseline(r: dict, model: dict, requester_host: str, commit: str, prompts: dict | None = None,
                     engine: str | None = None) -> bool:
    """model: {name, quant, sha256}; prompts: {set, version, generation}; engine: engine.name of the record being graded.
    sha256 and quant are compared when both sides have them (a v1 BASELINE written before sha caching has null and
    still matches on name)."""
    if r["cell"]["row"] != "BASELINE" or not r["success"] or r["model"]["name"] != model["name"]:
        return False
    if not same_commit(r["engine"]["commit"], commit):
        return False
    if not same_engine(r["engine"].get("name"), engine):
        return False
    if not any(same_host(p["host"], requester_host) and p["role"] == "requester" for p in r["peers"]):
        return False
    for k in ("sha256", "quant"):
        if r["model"].get(k) and model.get(k) and r["model"][k] != model[k]:
            return False
    if prompts:
        if r["prompts"].get("set") != prompts.get("set") or r["prompts"].get("version") != prompts.get("version"):
            return False
        if not _gen_equal(r["prompts"].get("generation"), prompts.get("generation")):
            return False
    return True


def find_baseline(recs: list[dict], model, requester_host: str, commit: str, prompts: dict | None = None,
                  engine: str | None = None) -> dict | None:
    """Latest successful BASELINE for the same model (name, and sha256 and quant when known), same engine, same commit,
    same prompt set and version, same generation flags, run on the requester host. `model` may be a name (v1 call) or
    a dict."""
    if isinstance(model, str):
        model = {"name": model}
    cands = [r for r in recs if matches_baseline(r, model, requester_host, commit, prompts, engine)]
    cands.sort(key=lambda r: r["date"])
    return cands[-1] if cands else None
