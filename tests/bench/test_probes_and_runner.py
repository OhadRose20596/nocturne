import time
from pathlib import Path

from nocturne.bench import config, run as runmod
from nocturne.bench.adapters.llamacpp_rpc import LlamaCppRpc
from nocturne.bench.probes import capture, net, util
from nocturne.bench.runner import Runner

WORKER_LOG = """[hello] version 3.0.0
[set_tensor] id: 1 size: 20000000
[set_tensor_hash] id: 2 hash: 1234 result: 1
[graph_compute] n_nodes: 957
[set_tensor] id: 3 size: 14336
[get_tensor] id: 4 size: 608256
[graph_compute] n_nodes: 957
[set_tensor] id: 3 size: 14336
[set_tensor] id: 5 size: 100
[graph_recompute] n_nodes: 957
[set_tensor] id: 3 size: 14336
[set_tensor] id: 5 size: 100
[set_tensor] id: 6 size: 100
[get_tensor] id: 4 size: 608256
[graph_recompute] n_nodes: 957
[set_tensor] id: 3 size: 14336
[set_tensor] id: 5 size: 100
[set_tensor] id: 6 size: 100
[get_tensor] id: 4 size: 608256
"""


def _cfg():
    cfg = config.load_peers(config.PEERS_EXAMPLE)
    for n, p in cfg["peers"].items():
        p["name"] = n
    return cfg


def _engine(cell, ts=None, dry=True):
    cells = config.load_cells()
    cfg = _cfg(); cfg["tensor_split"] = ts
    return LlamaCppRpc(cell, config.cell(cells, cell), cfg, cfg["models"]["qwen2.5-7b-q4km"], "wifi",
                       Runner(dry_run=dry, echo=False), Path("/nonexistent"))


def test_build_flags_a1_no_ts_drops_pair():
    f = _engine("A1").build_flags()
    assert "-ts" not in f and "--rpc" in f and f[f.index("--rpc") + 1] == "192.168.1.10:50052"
    assert f[f.index("--host") + 1] == "127.0.0.1"
    assert f[f.index("--temp") + 1] == "0" and f[f.index("--seed") + 1] == "42"


def test_build_flags_a1_with_ts_remote_first():
    f = _engine("A1", ts=[3, 2]).build_flags()
    assert f[f.index("-ts") + 1] == "3,2"


def test_build_flags_a2_and_a4():
    f = _engine("A2").build_flags()
    assert f.index("--rpc") < f.index("--model-draft")
    assert f[f.index("--model-draft") + 1].endswith("Qwen2.5-0.5B-Instruct-Q8_0.gguf")
    f = _engine("A4").build_flags(k=3)
    assert f[f.index("-np") + 1] == "3" and "-cb" in f and f[f.index("-c") + 1] == str(4096 * 3)


def test_runner_logs_and_tilde_expands_remote():
    r = Runner(dry_run=True, echo=False)
    r.run({"name": "air", "ssh": "u@h"}, ["~/dev/bin/x", "--flag", "a b"])
    r.run({"name": "m5", "ssh": None}, ["/usr/bin/true"], env={"GGML_RPC_DEBUG": "1"})
    recs = r.as_records()
    assert recs[0]["host"] == "air" and recs[0]["cmd"].startswith("ssh ") and '"$HOME"/dev/bin/x' in recs[0]["cmd"]
    assert "'a b'" in recs[0]["cmd"]
    assert recs[1] == {"host": "m5", "cmd": "GGML_RPC_DEBUG=1 /usr/bin/true"}
    assert r.spawn({"name": "m5"}, ["sleep", "1"]) is None


def test_runner_executes_locally():
    r = Runner(echo=False)
    out = r.run({"name": "local", "ssh": None}, ["echo", "hi there"])
    assert out.returncode == 0 and out.stdout.strip() == "hi there"


def test_ping_monitor_parse_percentiles_and_awdl():
    m = net.PingMonitor("192.0.2.1")
    assert m.argv() == ["ping", "-i", "0.01", "192.0.2.1"]
    for i, rtt in enumerate([8.0, 9.0, 7.5, 60.0, 8.2]):
        m.feed(f"64 bytes from 192.0.2.1: icmp_seq={i} ttl=64 time={rtt} ms")
    m.feed("Request timeout for icmp_seq 5")
    assert len(m.samples) == 5 and m.samples[3][1] == 3 and m.samples[3][2] == 60.0
    now = m.samples[0][0]
    p = m.in_windows([(now - 1, now + 1)])
    assert p["p50"] == 8.2 and p["p99"] == 60.0 and p["n"] == 5 and p["interval_s"] == 0.01 and 0 <= p["loss"] <= 1
    assert m.in_windows([(now + 100, now + 101)]) is None
    assert net.ping_percentiles([(1.0, 0, 5.0)], [(0, 2)]) == {"p50": 5.0, "p95": 5.0, "p99": 5.0, "n": 1}
    m.stop()           # never started; must not raise

    class R:
        def __init__(self, out, rc=0): self.stdout, self.stderr, self.returncode = out, "", rc
    assert net.awdl_state(run=lambda *a, **k: R("awdl0: flags=8943<UP,BROADCAST,SMART,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1500\n")) == "up"
    assert net.awdl_state(run=lambda *a, **k: R("awdl0: flags=8902<BROADCAST,PROMISC,SIMPLEX,MULTICAST> mtu 1500\n")) == "down"
    assert net.awdl_state(run=lambda *a, **k: R("ifconfig: interface awdl0 does not exist", 1)) == "unknown"


def test_ping_regex_and_missing_peer():
    assert net.PING_TIME.findall("64 bytes from 1.2.3.4: icmp_seq=0 ttl=64 time=1.234 ms\ntime<1 ms") == ["1.234", "1"]
    d, notes = net.measure(None, "local")
    assert d["rtt_ms_median"] is None and d["bw_mbps"] is None and notes
    d, notes = net.measure("192.0.2.1", "wifi", dry_run=True)
    assert d["type"] == "wifi" and notes == ["dry run"]


def test_capture_parse_and_per_token():
    c = capture.Capture(50052, "en0")
    assert c.argv()[:6] == ["sudo", "-n", "tcpdump", "-i", "en0", "-nn"]
    for line in ["1700000000.100000 IP 192.168.1.20.5555 > 192.168.1.10.50052: tcp 14336",
                 "1700000000.200000 IP 192.168.1.10.50052 > 192.168.1.20.5555: tcp 0",
                 "1700000000.300000 IP 192.168.1.10.50052 > 192.168.1.20.5555: tcp 608256",
                 "1700000001.500000 IP 192.168.1.10.50052 > 192.168.1.20.5555: tcp 100",
                 "garbage line"]:
        c.feed(line)
    assert len(c.packets) == 4 and [p[2] for p in c.packets] == ["out", "in", "in", "in"]
    r = c.per_token([(1700000000.0, 1700000001.0, 2)], compute_steps=2)
    assert r["bytes_per_token"] == (14336 + 608256) / 2 and r["bytes_per_step"] == r["bytes_per_token"]
    assert r["packets_per_token"] == 1.0
    empty = capture.Capture(50052, "en0")
    assert empty.per_token([(0, 1, 2)]) == {"bytes_per_token": None, "bytes_per_step": None, "packets_per_token": None}


def test_msgs_per_step_from_worker_log():
    s, notes = capture.msgs_per_step_from_worker_log(WORKER_LOG, tail=2)
    assert s == 5.0 and notes == []   # 3 set_tensor + 1 get_tensor + the graph message
    s, notes = capture.msgs_per_step_from_worker_log("[hello]\n[set_tensor] size: 5\n")
    assert s is None and notes


def test_util_parse_and_busy_fraction():
    s = util.Sampler({"name": "m5", "backend": "metal", "ssh": None})
    assert s._parse("GPU HW active residency:  42.50% (389 MHz: ...)") == 42.5
    assert s._parse("GPU idle residency: 57.50%") is None
    cu = util.Sampler({"name": "box", "backend": "cuda", "ssh": "u@h"})
    assert cu._parse(" 87 ") == 87.0
    cpu = util.Sampler({"name": "lin", "backend": "cpu", "ssh": "u@h"})
    assert cpu._parse("cpu  100 0 100 800 0 0 0") is None
    assert cpu._parse("cpu  150 0 150 900 0 0 0") == 50.0
    now = time.time()
    s.samples = [(now - 10, 90.0), (now, 40.0), (now + 0.05, 60.0)]
    assert s.busy_fraction([(now - 0.1, now + 0.1)]) == 0.5
    assert s.busy_fraction([(now + 100, now + 101)]) is None
    s.stop()  # never started; must not raise


def test_append_results_md(tmp_path):
    md = tmp_path / "RESULTS.md"; md.write_text("# Experiments Log\n")
    rec = {"date": "2026-10-26T10:00:00+02:00", "id": "A1_wifi_m_r1", "engine": {"commit": "1e7bcf3"},
           "model": {"name": "m"}, "network": {"type": "wifi", "rtt_ms_median": 3.25, "bw_mbps": 240.0},
           "timing": {"decode_tps_median": 7.7}, "comm": {"rpc_commands_per_token": 7.0, "bytes_per_token": 14336.0},
           "memory_buffers_gb": {"m5": 2.2, "air": 2.0}, "correctness": {"task_accuracy": 0.9, "baseline_task_accuracy": 0.91},
           "peers": [{"host": "M5", "role": "requester"}, {"host": "Air", "role": "worker"}]}
    runmod.append_results_md(rec, md); runmod.append_results_md(rec, md)
    text = md.read_text()
    assert text.count("## Bench matrix") == 1
    assert text.count("| 2026-10-26T10:00:00+02:00 | A1_wifi_m_r1 | M5 | Air | 1e7bcf3 | m | wifi | 3.2 | 240 | 7.7 | 7.0 | 14336 | 2.2 | 0.90 | 0.91 |") == 2
    assert text.startswith("# Experiments Log\n")
    # a v3 record: pooled tok/s in the decode column, run conditions in the network column
    rec["timing"]["decode_tps_pooled"] = 6.2
    rec["network"].update({"awdl": "down", "added_delay_ms": 10.0}); rec["prompts"] = {"fixed_tokens": 256, "set": "eval_long"}
    runmod.append_results_md(rec, md)
    assert "| wifi; awdl down; delay 10.0 ms; fixed 256; eval_long | 3.2 | 240 | 6.2 pooled |" in md.read_text()


def test_stop_all_kills_process_group():
    """A `bash -c` leader with a sleeping child: terminating the leader alone leaves the child (I024)."""
    import os, signal, time
    from nocturne.bench.runner import Runner
    r = Runner(echo=False)
    p = r.spawn({"name": "local", "ssh": None}, ["bash", "-c", "sleep 300 & echo $!; wait"], stdout=__import__("subprocess").PIPE)
    child = int(p.stdout.readline().strip())
    assert os.getpgid(child) == p.pid        # own session: the child is in the leader's group
    r.stop_all(grace=2.0)
    time.sleep(0.3)
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:
        os.kill(child, signal.SIGKILL)
    assert not alive and p.poll() is not None


def test_safe_stop_bounds_a_hung_stop():
    from nocturne.bench.run import safe_stop
    import time

    class Hang:
        def stop(self): time.sleep(5)

    class Boom:
        def stop(self): raise RuntimeError("ssh died")

    class Fine:
        def stop(self): pass
    assert safe_stop(Hang(), timeout_s=0.3).startswith("stop() still running")
    assert safe_stop(Boom()) == "RuntimeError: ssh died"
    assert safe_stop(Fine()) is None


def test_baseline_adapter_follows_model_engine(monkeypatch, tmp_path):
    """BASELINE is one cell; models.<name>.engine picks the adapter and flags_by_engine the template (amendment 2)."""
    import argparse
    from nocturne.bench import adapters, config, run
    cells = config.load_cells()
    cfg = config.load_peers(config.PEERS_EXAMPLE)
    cfg["models"]["fake-torch"] = {"engine": "torch_tp", "torch_model": "/m", "quant": "bf16"}
    captured = {}

    def fake_get(name):
        captured["adapter"] = name
        raise adapters.EngineUnsupported("not registered in this test")
    monkeypatch.setattr(adapters, "get", fake_get)
    a = argparse.Namespace(cell="BASELINE", host="m5", model="fake-torch", network="wifi", subset=None, dry_run=True,
                           live=False, live_port=0, ctx=4096, load_timeout=1, request_timeout=1, k=None,
                           allow_unmeasured_memory=False)
    rec = run.run_once(a, cells, cfg, config.cell(cells, "BASELINE"), 1, tmp_path)
    assert captured["adapter"] == "torch_tp" and rec["failure_class"] == "tool_unsupported"
    a.model = "qwen2.5-7b-q4km"
    run.run_once(a, cells, cfg, config.cell(cells, "BASELINE"), 1, tmp_path)
    assert captured["adapter"] == "llamacpp_rpc"


def _llama(cell, cfg=None):
    from pathlib import Path
    from nocturne.bench import config
    from nocturne.bench.adapters.llamacpp_rpc import LlamaCppRpc
    from nocturne.bench.runner import Runner
    cells = config.load_cells(); cfg = cfg or config.load_peers(config.PEERS_EXAMPLE)
    for n, p in cfg["peers"].items():
        p["name"] = n
    cfg["roles"] = {"requester": "m5", "workers": ["air"]}
    return LlamaCppRpc(cell, config.cell(cells, cell), cfg, cfg["models"]["qwen2.5-7b-q4km"], "wifi",
                       Runner(dry_run=True, echo=False), Path("/nonexistent"))


def test_b1_flags_tensor_mode_and_override():
    e = _llama("B1")
    f = e.build_flags()
    assert f[f.index("--split-mode") + 1] == "tensor" and f[f.index("-fit") + 1] == "off" and "-ts" not in f
    assert "--rpc" in f and f[f.index("--rpc") + 1] == "192.168.1.10:50052"
    assert e.tensor_mode()
    e.overrides["--split-mode"] = "row"
    g = e.build_flags()
    assert g[g.index("--split-mode") + 1] == "row" and not e.tensor_mode()


def test_b1_classify_and_placement(tmp_path):
    from nocturne.bench.adapters import llamacpp_rpc as L
    e = _llama("B1"); e.server_log = tmp_path / "server.log"
    e.server_log.write_text("0.1 I llama_model_load: creating a Meta device with 2 devices\n"
                            "0.2 E llama_model_load: error loading model: LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'qwen2'\n")
    assert e._classify_log() == "tool_unsupported"
    assert "not implemented for architecture" in e.first_error_line()
    assert L.PLACEMENT.search("load_tensors:  RPC0[192.168.1.10:50052] model buffer size =  2000.00 MiB")
    assert L.PLACEMENT.search("llama_init_from_model: enabling flash_attn since it is required for SPLIT_MODE_TENSOR")
    assert L.PLACEMENT.search("GGML_ASSERT(ggml_backend_dev_is_meta(dev)) failed")
    # outcome (b): loaded, every buffer local
    e.load_s = 12.0; e.server = object()         # a launched server; preflight failures report no outcome
    e.server_log.write_text("0.1 I load_tensors:  MTL0_Mapped model buffer size =  4168.09 MiB\n")
    c = e.counters()
    assert c["tp_outcome"]["outcome"] == "b"
    # outcome (a): both sides
    e.server_log.write_text("0.1 I load_tensors:  MTL0_Mapped model buffer size =  2084.00 MiB\n"
                            "0.1 I load_tensors:  RPC0[192.168.1.10:50052] model buffer size =  2084.00 MiB\n")
    c = e.counters()
    assert c["tp_outcome"]["outcome"] == "a" and set(c["memory_buffers_gb"]) == {"m5", "air"}
    # outcome (a) as the pinned build prints it: one Meta() buffer, per device share, attributed to every device
    e.server_log.write_text("0.1 I llama_prepare_model_devices: creating a Meta device for tensor parallelism from 2 devices:\n"
                            "0.1 I load_tensors:   CPU_Mapped model buffer size =   292.36 MiB\n"
                            "0.1 I load_tensors:       Meta() model buffer size =  2084.43 MiB\n"
                            "0.2 I llama_kv_cache:     Meta() KV buffer size =   112.00 MiB\n"
                            "0.2 I llama_init_from_model: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n")
    c = e.counters()
    assert c["tp_outcome"]["outcome"] == "a" and c["tp_outcome"]["flash_attn_forced"] and c["tp_outcome"]["meta_device"]
    assert abs(c["memory_buffers_gb"]["air"] - (2084.43 + 112) / 1024) < 1e-3
    assert abs(c["memory_buffers_gb"]["m5"] - (2084.43 + 112 + 292.36) / 1024) < 1e-3
    # A1 has no tp outcome
    a1 = _llama("A1"); a1.server_log = tmp_path / "none.log"
    assert a1.counters()["tp_outcome"] is None
