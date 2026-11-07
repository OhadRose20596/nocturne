import json
from pathlib import Path

import jsonschema
import pytest

from nocturne.bench import config, record, report


def test_cells_yaml_complete():
    cells = config.load_cells()
    names = set(cells["cells"])
    assert names == {"BASELINE", "A2b", "B1t", "B1d", "B2d", "B3d", "B4d"} | {f"{r}{c}" for r in "ACD" for c in "1234"} | {"B1"}
    for n, c in cells["cells"].items():
        assert c["status"] in ("ready", "later", "tool_unsupported"), n
        assert c["adapter"], n
        assert isinstance(c["flags"], list), n
    assert {n for n, c in cells["cells"].items() if c["status"] == "ready"} == {"BASELINE", "A1", "A2", "A2b", "A4", "B1", "B1t", "B1d"}
    assert cells["cells"]["B4d"]["status"] == "tool_unsupported" and cells["cells"]["B2d"]["status"] == "tool_unsupported"
    assert cells["cells"]["B3d"]["status"] == "later"


def test_peers_example_loads(tmp_path):
    cfg = config.load_peers(config.PEERS_EXAMPLE)
    assert cfg["roles"]["requester"] in cfg["peers"]
    for w in cfg["roles"]["workers"]:
        assert w in cfg["peers"]
    for p in cfg["peers"].values():
        assert p["bind"] != "0.0.0.0"


def test_peers_reject_wildcard_bind(tmp_path):
    bad = tmp_path / "peers.yaml"
    bad.write_text(config.PEERS_EXAMPLE.read_text().replace("bind: 192.168.1.20", "bind: 0.0.0.0"))
    with pytest.raises(ValueError):
        config.load_peers(bad)


def test_peers_missing_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load_peers(tmp_path / "nope.yaml")


def _rec(cell="A1", net="wifi", model="qwen2.5-7b-q4km", run=1, **over):
    cells = config.load_cells()
    r = record.empty_record(cell, config.cell(cells, cell), net, model, run)
    r["peers"] = [{"host": "m5", "role": "requester", "backend": "metal", "os": "macos", "mem_gb": 24}]
    r["engine"] = {"name": "llama.cpp", "commit": "1e7bcf3", "flags": ["-m", "x.gguf"]}
    for k, v in over.items():
        r[k] = v
    return r


def test_empty_record_validates():
    record.validate(_rec())
    record.validate(_rec("BASELINE"))
    b = _rec("BASELINE")
    assert b["cell"] == {"partition": "none", "decode": "single", "row": "BASELINE", "col": 0, "name": "BASELINE"}
    a4 = _rec("A4")
    assert a4["cell"]["row"] == "A" and a4["cell"]["col"] == 4 and a4["cell"]["decode"] == "independent"
    a2b = _rec("A2b")
    assert a2b["cell"] == {"partition": "PP", "decode": "speculative", "row": "A", "col": 2, "name": "A2b"}
    record.validate(a2b)


def test_schema_rejects_bad_values():
    r = _rec(); r["failure_class"] = "cosmic_rays"
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
    r = _rec(); r["utilization"] = {"m5": 1.5}
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
    r = _rec(); r["model"]["sha256"] = "abc"
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
    r = _rec(); r["peers"] = []
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
    r = _rec(); r["extra"] = 1
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
    r = _rec(); del r["comm"]
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)


def test_write_read_and_run_index(tmp_path):
    assert record.next_run_index(tmp_path, "A1", "wifi", "qwen2.5-7b-q4km") == 1
    p = record.write(_rec(run=1), tmp_path)
    assert p.name == "A1_wifi_qwen2.5-7b-q4km_r1.json"
    assert record.next_run_index(tmp_path, "A1", "wifi", "qwen2.5-7b-q4km") == 2
    record.write(_rec(run=2), tmp_path)
    recs = record.read_all(tmp_path)
    assert [r["id"] for r in recs] == ["A1_wifi_qwen2.5-7b-q4km_r1", "A1_wifi_qwen2.5-7b-q4km_r2"]
    assert json.loads(p.read_text())["schema_version"] == 3


def test_write_refuses_invalid(tmp_path):
    r = _rec(); r["success"] = "yes"
    with pytest.raises(jsonschema.ValidationError):
        record.write(r, tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_find_baseline(tmp_path):
    b1 = _rec("BASELINE", net="local", run=1, success=True); b1["date"] = "2026-10-20T10:00:00+02:00"
    b2 = _rec("BASELINE", net="local", run=2, success=True); b2["date"] = "2026-10-21T10:00:00+02:00"
    b3 = _rec("BASELINE", net="local", run=3, success=False); b3["date"] = "2026-10-22T10:00:00+02:00"
    other = _rec("BASELINE", net="local", run=4, success=True); other["engine"]["commit"] = "deadbee"
    recs = [b1, b2, b3, other, _rec("A1")]
    hit = record.find_baseline(recs, "qwen2.5-7b-q4km", "m5", "1e7bcf3")
    assert hit is b2
    assert record.find_baseline(recs, "qwen2.5-7b-q4km", "air", "1e7bcf3") is None


def test_report_table(tmp_path):
    base = _rec("BASELINE", net="wifi", success=True)
    base["correctness"]["task_accuracy"] = 0.9; base["timing"]["decode_tps_median"] = 25.0
    a1 = _rec("A1", success=True)
    a1["correctness"]["task_accuracy"] = 0.88; a1["comm"]["rpc_commands_per_token"] = 7.0
    a1["comm"]["bytes_per_token"] = 14336.0; a1["memory_buffers_gb"] = {"m5": 5.1, "air": 4.0}
    a1["timing"]["decode_tps_median"] = 7.7
    a1b = _rec("A1", run=2, success=False, failure_class="oom")
    a2 = _rec("A2", success=False, failure_class="wrong_output")
    a2b = _rec("A2b", success=True); a2b["timing"]["decode_tps_median"] = 12.5
    a1["quality_pass"] = True; a1b["quality_pass"] = False
    for r in (base, a1, a1b, a2, a2b):
        record.write(r, tmp_path)
    md = report.render(record.read_all(tmp_path))
    assert ("| A2b (A2 with a shorter, more selective draft) |  | ok 1/1<br>quality n/a, acc null<br>ms/step null (null pooled), syncs null"
            "<br>S null B null<br>mem null; dec median 12.5 t/s (old) |  |  |") in md
    assert "### qwen2.5-7b-q4km over wifi, requester m5, engine llama.cpp" in md
    assert "BASELINE (wifi): ok 1/1, acc 0.90, ms/step null (null t/s pooled), dec median 25.0 t/s (old)" in md and md.startswith("ms/step:")
    # operational success and quality_pass are shown apart (I025 9): 1 of 2 ok, 1 of 2 passed quality
    assert "| A | ok 1/2 (oom)<br>quality 1/2, acc 0.88<br>ms/step null (null pooled), syncs null<br>S 7.0 B 14336<br>mem 5.1 GB; dec median 7.7 t/s (old) | ok 0/1 (wrong_output)" in md
    # v3 values come from the record when it has them, else from derived_v3.json for older records
    derived = {a1["id"]: {"timing": {"decode_ms_per_step_pooled": 110.0, "decode_tps_pooled": 9.09}, "comm": {"blocking_syncs_per_step": 1.0}}}
    mdd = report.render(record.read_all(tmp_path), derived=derived)
    assert "ms/step 110.0 (9.1 t/s pooled), syncs 1.0<br>S 7.0" in mdd
    assert "| B | not run | not run | not run | not run |" in md
    assert report.render([]) == "no records\n"
    assert report.render(record.read_all(tmp_path), network="wired") == "no records\n"
    # an excluded record disappears from the table and is listed at the end; the file is untouched
    (tmp_path / "EXCLUDE.yaml").write_text("exclude:\n  " + a1b["id"] + ": smoke run\n")
    ex = report.load_exclusions(tmp_path)
    md2 = report.render(record.read_all(tmp_path), excluded=ex)
    assert "| A | ok 1/1<br>quality 1/1, acc 0.88" in md2 and f"- `{a1b['id']}`: smoke run" in md2
    assert (tmp_path / f"{a1b['id']}.json").exists()
    # engine name separates groups: a torch BASELINE on the same host is its own section
    tb = _rec("BASELINE", net="local", model="qwen2.5-7b-bf16", success=True)
    tb["engine"] = {"name": "torch_tp", "commit": "abc1234", "flags": []}
    b1t = _rec("B1t", model="qwen2.5-7b-bf16", success=True)
    b1t["engine"] = {"name": "torch_tp", "commit": "abc1234", "flags": []}
    for r in (tb, b1t):
        record.write(r, tmp_path)
    md3 = report.render(record.read_all(tmp_path))
    # the local BASELINE is shown above the wifi section of the same (model, host, engine), not as its own section
    assert "### qwen2.5-7b-bf16 over wifi, requester m5, engine torch_tp\n\nBASELINE (local): ok 1/1" in md3
    assert "### qwen2.5-7b-bf16 over local" not in md3 and "| B1t (" in md3


def test_find_baseline_requires_same_engine():
    base = _rec("BASELINE", net="local", success=True)
    base["engine"]["name"] = "llama.cpp"
    assert record.find_baseline([base], "qwen2.5-7b-q4km", "m5", "1e7bcf3", engine="llama.cpp") is base
    assert record.find_baseline([base], "qwen2.5-7b-q4km", "m5", "1e7bcf3", engine="torch_tp") is None
    assert record.find_baseline([base], "qwen2.5-7b-q4km", "m5", "1e7bcf3") is base       # engine unknown: v1 call
    tb = _rec("BASELINE", net="local", success=True)
    tb["engine"] = {"name": "torch_tp", "commit": "1e7bcf3", "flags": []}
    assert record.find_baseline([base, tb], "qwen2.5-7b-q4km", "m5", "1e7bcf3", engine="torch_tp") is tb


def test_upgrade_v1_record_in_memory():
    v1 = json.loads(Path(record.RESULTS_DIR / "A2_wifi_qwen2.5-7b-q4km_m5_r2.json").read_text())
    assert v1["schema_version"] == 1
    record.validate(v1)                       # v1 files still validate against the kept v1 schema
    u = record.upgrade(v1)
    record.validate(u)
    assert v1["schema_version"] == 1 and "msgs_per_token_median" in v1["comm"]     # source untouched
    c = u["comm"]
    assert c["rpc_commands_per_step"] == 7.0 and abs(c["rpc_commands_per_token"] - 2.245) < 0.01
    assert abs(c["tokens_per_step"] - 714 / 229) < 1e-9 and c["serial_waits_per_token"] is None
    assert u["setup"]["load_total_s"] == v1["setup"]["weight_transfer_s"] and u["setup"]["transfer_s"] is None
    assert u["timing"]["ttft_server_s_median"] == v1["timing"]["ttft_s_median"] and u["timing"]["ttft_s_median"] is None
    assert u["memory_buffers_gb"] == v1["memory_peak_gb"] and u["memory_peak_gb"] == {"m5": None, "air": None}
    assert u["quality_pass"] is True and u["checks"]["a2_identity"] is True
    assert record.upgrade(u) is u
    assert u["schema_version"] == 3 and u["network"]["rtt_source"] == "probe" and u["prompts"]["fixed_tokens"] is None
    assert u["comm"]["blocking_syncs_per_step"] is None and u["comm"]["worker_capture"] is False
    # pooled timing is derived from per_prompt for every version: 714 tokens over 33 prompts is 681 steps
    assert u["timing"]["decode_steps_total"] == 681 and 80 < u["timing"]["decode_ms_per_step_pooled"] < 120
    assert abs(u["timing"]["decode_tps_pooled"] - 1000 / u["timing"]["decode_ms_per_step_pooled"]) < 1e-9


def test_upgrade_v2_record_in_memory():
    v2 = json.loads(Path(record.RESULTS_DIR / "B1_wifi_qwen2.5-7b-q4km_m5_r3.json").read_text())
    assert v2["schema_version"] == 2
    record.validate(v2)
    u = record.upgrade(v2)
    record.validate(u)
    assert v2["schema_version"] == 2 and "decode_ms_per_step_pooled" not in v2["timing"]
    assert u["schema_version"] == 3 and u["comm"]["source"].endswith("[v2 record, v3 view in memory]")
    assert abs(u["timing"]["decode_ms_per_step_pooled"] - 1605.98) < 0.1 and abs(u["timing"]["decode_tps_median"] - 0.742) < 0.001
    assert u["per_prompt"]["long_03"]["decode_ms_per_step"] == v2["per_prompt"]["long_03"]["timings"]["predicted_ms"] / 113


def test_pooled_timing():
    per = {"a": {"timings": {"predicted_n": 5, "predicted_ms": 400.0}}, "b": {"timings": {"predicted_n": 1, "predicted_ms": 0.0}},
           "c": {"timings": {"predicted_n": 3, "predicted_ms": 100.0}}, "d": {"timings": None, "error": "boom"}}
    t = record.pooled_timing(per)
    assert t == {"decode_ms_per_step_pooled": 500 / 6, "decode_tps_pooled": 1000 / (500 / 6), "decode_steps_total": 6}
    assert per["a"]["decode_ms_per_step"] == 100.0 and per["b"]["decode_ms_per_step"] is None and per["d"]["decode_ms_per_step"] is None
    assert record.pooled_timing({}) == {"decode_ms_per_step_pooled": None, "decode_tps_pooled": None, "decode_steps_total": None}


def test_schema_v3_new_fields_validate():
    r = _rec()
    r["timing"].update({"decode_ms_per_step_pooled": 110.0, "decode_tps_pooled": 9.09, "decode_steps_total": 681, "warmup_s": 1.5})
    r["comm"].update({"blocking_syncs_per_step": 57.0, "messages_per_step": 232.0, "bytes_per_step_decode": 1.9e6,
                      "bytes_per_step_decode_out": 8e5, "bytes_per_step_decode_in": 1.1e6, "graph_meta_bytes_per_step": 1.66e7,
                      "worker_capture": True, "clock_offset_s": 2.5, "clock_offset_unc_s": 0.001})
    r["network"].update({"ping_decode_ms": {"p50": 8.1, "p95": 20.0, "p99": 60.0, "n": 5000, "interval_s": 0.01, "loss": 0.01},
                         "awdl": "down", "added_delay_ms": 10.0, "added_delay_measured_ms": 9.8})
    r["prompts"]["fixed_tokens"] = 256
    record.validate(r)
    r["network"]["awdl"] = "sideways"
    with pytest.raises(jsonschema.ValidationError):
        record.validate(r)
