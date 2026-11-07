"""torch_tp adapter: flags, commands on each machine, bind rule, counters mapping. Nothing runs (dry run)."""
from pathlib import Path

import pytest

from nocturne.bench import config
from nocturne.bench.adapters import torch_tp as T
from nocturne.bench.adapters.base import EngineUnsupported
from nocturne.bench.runner import Runner


def _cfg(workers=("air",)):
    cfg = config.load_peers(config.PEERS_EXAMPLE)
    for n, p in cfg["peers"].items():
        p["name"] = n
    cfg["roles"] = {"requester": "m5", "workers": list(workers)}
    return cfg


def _eng(cell="B1t", workers=("air",), model="qwen2.5-7b-bf16", flags=None):
    cells = config.load_cells(); cfg = _cfg(workers)
    cc = dict(config.cell(cells, cell))
    if flags is not None:
        cc["flags"] = flags
    return T.TorchTp(cell, cc, cfg, cfg["models"][model], "wifi", Runner(dry_run=True, echo=False), Path("/nonexistent"))


def test_b1t_commands_per_machine():
    e = _eng()
    e.prepare(); e.start_workers(); e.launch(); e.complete([{"id": "x", "prompt": "hi", "max_tokens": 8}]); e.stop()
    cmds = e.runner.as_records()
    worker = next(c for c in cmds if "torch_tp worker" in c["cmd"])
    assert worker["host"] == "air" and "--bind 192.168.1.10 --port 50053" in worker["cmd"] and "--rank 1 --world 2" in worker["cmd"]
    assert "0.0.0.0" not in worker["cmd"] and "/Users/ohad/dev/nocturne-torch/bin/python -m nocturne.engines.torch_tp" in worker["cmd"]
    serve = next(c for c in cmds if "torch_tp serve" in c["cmd"])
    assert serve["host"] == "m5" and "--world 2" in serve["cmd"] and "--peer 192.168.1.10:50053" in serve["cmd"]
    assert "--host 127.0.0.1 --port 8092" in serve["cmd"] and "--max-seq-len 4096" in serve["cmd"]
    assert e.worker_port() == 50053 and e.mem_patterns()["worker"] == "torch_tp worker.*--port 50053"


def test_baseline_world_one_has_no_worker():
    cells = config.load_cells()
    flags = cells["cells"]["BASELINE"]["flags_by_engine"]["torch_tp"]
    e = _eng("BASELINE", workers=(), model="qwen2.5-1.5b-bf16", flags=flags)
    e.prepare(); e.start_workers(); e.launch(); e.stop()
    cmds = [c["cmd"] for c in e.runner.as_records()]
    assert not any("torch_tp worker" in c for c in cmds)
    serve = next(c for c in cmds if "torch_tp serve" in c)
    assert "--world 1" in serve and "--peer" not in serve and "Qwen2.5-1.5B-Instruct" in serve


def test_rejects_bad_configs():
    with pytest.raises(EngineUnsupported):
        _eng(workers=("air", "air")).prepare()
    cfg_e = _eng(model="qwen2.5-7b-q4km")          # a GGUF model has no torch_model
    with pytest.raises(EngineUnsupported):
        cfg_e.prepare()
    e = _eng(); e.peers["peers"]["air"].pop("python")
    with pytest.raises(EngineUnsupported):
        e.prepare()


def test_counters_from_metrics():
    e = _eng(); e.flags = e.build_flags(); e.load_s = 30.0
    e.metrics = {"world": 2, "torch": "2.14.1", "decode_steps": 90, "prefill_steps": 10, "tokens_generated": 100,
                 "ranks": {"0": {"driver_allocated_bytes": 8.0e9, "device": "mps"}, "1": {"driver_allocated_bytes": 7.6e9, "device": "mps"}},
                 "counters": {"sent_frames": 5610, "recv_frames": 5600, "sent_bytes": 1_000_000, "recv_bytes": 900_000, "allreduce_frames": 5600, "step_frames": 100}}
    c = e.counters()
    assert c["memory_buffers_gb"] == {"m5": 8.0, "air": 7.6}
    assert c["comm"]["rpc_commands_per_step"] == 56.0 and c["comm"]["rpc_commands_per_token"] == 56.0
    assert c["comm"]["engine_frames_per_token"] == 56.1 and c["comm"]["engine_bytes_per_token"] == 19000.0
    assert c["engine"]["runtime"] == "torch 2.14.1 mps" and c["engine"]["world"] == 2 and c["engine"]["format"] == "safetensors-bf16"
    from nocturne.bench import record
    r = record.empty_record("B1t", config.load_cells()["cells"]["B1t"], "wifi", "qwen2.5-7b-bf16", 1)
    r["peers"] = [{"host": "m5", "role": "requester", "backend": "metal", "os": "macos", "mem_gb": 24}]
    r["engine"] = c["engine"]; r["comm"].update(c["comm"]); r["memory_buffers_gb"] = c["memory_buffers_gb"]
    record.validate(r)
