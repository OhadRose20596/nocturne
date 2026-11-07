from nocturne.bench import record
from nocturne.bench.adapters import llamacpp_rpc
from nocturne.bench.probes import memlog, net

LOG = """0.05.678.113 I load_tensors:   CPU_Mapped model buffer size =   417.66 MiB
0.05.678.115 I load_tensors:  MTL0_Mapped model buffer size = 10939.06 MiB
0.05.678.116 I load_tensors: RPC0[192.168.1.219:50052] model buffer size =  7569.29 MiB
4.12.225.030 I llama_context:        CPU  output buffer size =     0.58 MiB
4.12.291.760 I llama_kv_cache:       MTL0 KV buffer size =   296.00 MiB
4.12.452.249 I llama_kv_cache: RPC0[192.168.1.219:50052] KV buffer size =   216.00 MiB
4.12.658.189 I sched_reserve: RPC0[192.168.1.219:50052] compute buffer size =   184.01 MiB
4.12.658.191 I sched_reserve:       MTL0 compute buffer size =   184.01 MiB
4.12.658.192 I sched_reserve:        CPU compute buffer size =    22.01 MiB
"""


def test_breakdown_by_node():
    bd, lines = memlog.breakdown(LOG, "m5", {"192.168.1.219": "air"}, ["m5", "air"])
    assert len(lines) == 9
    assert bd["air"] == {"model": round(7569.29 / 1024, 3), "kv": round(216 / 1024, 3), "compute": round(184.01 / 1024, 3),
                         "output": 0.0, "cpu": 0.0}
    assert bd["m5"]["kv"] == round(296 / 1024, 3) and bd["m5"]["model"] == round(10939.06 / 1024, 3)
    assert bd["m5"]["cpu"] == round((417.66 + 0.58 + 22.01) / 1024, 3)


def test_buf_regex_counts_compute_lines():
    assert len(llamacpp_rpc.BUF.findall(LOG)) == 9


def test_oom_classes():
    metal = "ggml_metal_synchronize: kIOGPUCommandBufferCallbackErrorOutOfMemory"
    assert memlog.classify_oom(LOG + metal, {}, loaded=True) == "oom_requester_compute"
    assert memlog.classify_oom(metal, {}, loaded=False) == "oom_requester_load"
    assert memlog.classify_oom("ggml_backend_rpc_buffer_type_alloc_buffer: failed to allocate RPC0 buffer", {}, False) == "oom_worker_load"
    assert memlog.classify_oom("fine", {"air": "ggml_metal: failed to allocate buffer"}, True) == "oom_worker_compute"
    assert memlog.classify_oom(LOG, {}, True) is None


def test_ts_override_applies_when_template_has_no_split():
    from tests.bench.test_probes_and_runner import _engine
    e = _engine("A1")
    e.overrides["-ts"] = "1,1"
    f = e.build_flags(1)
    assert f[f.index("-ts") + 1] == "1,1"


def test_vpn_state():
    class R:
        returncode = 0
        stdout = "Available network connection services in the current set (*=enabled):\n* (Disconnected) X VPN\n"
    assert net.vpn_state(lambda *a, **k: R()) == "disconnected"
    R.stdout = "* (Connected)    X VPN"
    assert net.vpn_state(lambda *a, **k: R()) == "connected"


def test_v3_record_upgrades_to_v4():
    r = record.empty_record("BASELINE", {"partition": "none", "decode": "plain"}, "local", "m", 1)
    r["schema_version"] = 3
    for k in ("memory_breakdown_gb", "memory_log_lines"):
        r.pop(k)
    r["setup"].pop("ctx")
    r["network"].pop("vpn_state")
    u = record.upgrade(r)
    assert u["schema_version"] == 4 and u["setup"]["ctx"] is None and u["memory_breakdown_gb"] == {}


META = """0.05.755.050 I load_tensors:   CPU_Mapped model buffer size =   417.66 MiB
0.05.755.052 I load_tensors:       Meta() model buffer size = 11117.50 MiB
4.26.267.123 I llama_kv_cache:     Meta() KV buffer size =  5120.00 MiB
4.26.479.044 I llama_kv_cache: size = 8192.00 MiB ( 32768 cells,  64 layers,  1/1 seqs), K (f16): 4096.00 MiB, V (f16): 4096.00 MiB
4.26.751.802 I sched_reserve:     Meta() compute buffer size =   232.42 MiB
"""


def test_meta_split_by_ts_shares():
    assert memlog.ts_shares(["-m", "x", "-ts", "3,2"]) == [3.0, 2.0] and memlog.ts_shares(["-m", "x"]) is None
    bd, lines = memlog.breakdown(META, "m5", {}, ["m5", "air"], [3.0, 2.0])
    assert bd["m5"]["model"] == round(11117.5 / 1024, 3) and bd["air"]["model"] == round(11117.5 * 2 / 3 / 1024, 3)
    assert bd["m5"]["kv"] == 5.0 and bd["air"]["kv"] == 3.0
    assert bd["air"]["compute"] == bd["m5"]["compute"] and lines[-1].startswith("llama_kv_cache: size")
