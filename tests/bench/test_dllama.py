"""Row B adapter: flags, node-count gate, worker bind, stats parsing. Nothing runs; the dry run prints the commands."""
from pathlib import Path

import pytest

from nocturne.bench import config
from nocturne.bench.adapters import distributed_llama as dl
from nocturne.bench.adapters.base import EngineUnsupported
from nocturne.bench.adapters.llamacpp_rpc import LlamaCppRpc
from nocturne.bench.runner import Runner


def _cfg(workers=("air",)):
    cfg = config.load_peers(config.PEERS_EXAMPLE)
    for n, p in cfg["peers"].items():
        p["name"] = n
    cfg["roles"] = {"requester": "m5", "workers": list(workers)}
    return cfg


def _eng(cell="B1d", workers=("air",), model="qwen3-8b-dllama"):
    cells = config.load_cells(); cfg = _cfg(workers)
    return dl.DistributedLlama(cell, config.cell(cells, cell), cfg, cfg["models"][model], "wifi",
                               Runner(dry_run=True, echo=False), Path("/nonexistent"))


def test_b1_flags_and_ports():
    e = _eng()
    f = e.build_flags()
    assert f[f.index("--workers") + 1] == "192.168.1.10:9999" and f[f.index("--buffer-float-type") + 1] == "q80"
    assert f[f.index("--tokenizer") + 1].endswith("dllama_tokenizer_qwen3_8b_q40.t")
    assert e.worker_port() == 9999 and e.mem_patterns()["worker"] == "dllama worker.*--port 9999"


def test_prepare_dry_run_logs_commands_and_worker_binds_to_network_address():
    e = _eng()
    e.prepare(); e.start_workers(); e.launch(); e.complete([{"id": "x", "prompt": "hi", "max_tokens": 8}]); e.stop()
    cmds = [c["cmd"] for c in e.runner.as_records()]
    worker = next(c for c in cmds if "dllama worker" in c)
    assert "--host 192.168.1.10 --port 9999 --nthreads 4" in worker and "0.0.0.0" not in worker
    api = next(c for c in cmds if "dllama-api --model" in c)
    assert "--host 127.0.0.1 --port 9990" in api and "--temperature 0 --seed 42" in api
    assert any("dllama inference" in c and "--steps 8" in c for c in cmds)
    assert e.counters()["engine"]["format"] == "dllama-q40" and e.counters()["comm"]["rpc_commands_per_step"] is None
    assert e.util_backend() == "cpu"


def test_tool_stats_are_kib():
    e = _eng()
    e.tool_stats = {"steps": 2, "sent_kb_per_token": 512, "recv_kb_per_token": 256, "pred_ms_median": 1, "sync_ms_median": 1}
    c = e.counters()["comm"]
    assert c["tool_bytes_sent_per_token"] == 512 * 1024 and c["tool_bytes_recv_per_token"] == 256 * 1024


def test_request_is_not_streamed(monkeypatch):
    """usage is present only in the non streaming body (I025 2)."""
    e = _eng(); e.runner.dry_run = False
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    def post(url, json, timeout):
        seen.update(json); return R()
    monkeypatch.setattr(dl.httpx, "post", post)
    c = e._one({"id": "p", "prompt": "2+2", "max_tokens": 8})
    assert seen["stream"] is False and c.n_gen == 3 and c.n_prompt == 7 and c.text == "4" and c.error is None
    assert c.prompt_ms is None and c.predicted_ms is not None


def test_prepare_rejects_bad_node_count_and_other_decodes():
    with pytest.raises(EngineUnsupported):
        _eng(workers=("air", "air")).prepare()         # root + 2 = 3 nodes
    with pytest.raises(EngineUnsupported):
        _eng("B2d").prepare()
    with pytest.raises(EngineUnsupported):
        _eng("B4d").prepare()


def test_tool_stats_regex():
    line = "🔶 Pred  123 ms Sync   45 ms | Sent   512 kB Recv   256 kB | token"
    assert dl.PRED.findall(line) == [("123", "45", "512", "256")]
    ev = "🔷️ Eval  900 ms Sync   10 ms | Sent  2048 kB Recv  1024 kB | (46 tokens)"
    assert dl.EVAL.findall(ev) == [("900", "10", "2048", "1024", "46")]


def test_a2b_flags():
    cells = config.load_cells(); cfg = _cfg()
    e = LlamaCppRpc("A2b", config.cell(cells, "A2b"), cfg, cfg["models"]["qwen2.5-7b-q4km"], "wifi",
                    Runner(dry_run=True, echo=False), Path("/nonexistent"))
    f = e.build_flags()
    assert f[f.index("--spec-draft-n-max") + 1] == "4" and f[f.index("--spec-draft-p-min") + 1] == "0.75"
    assert f[f.index("--spec-type") + 1] == "draft-simple"
