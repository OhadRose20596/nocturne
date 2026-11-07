"""Live view: SSE framing, log parsers on lines copied from results/logs/ (A1_wifi r2, A2_wifi r2, A4_wifi r2),
step tracking, replay scheduling and the HTTP server."""
import http.client
import json
import time
from pathlib import Path

from nocturne.bench.live import parsers, server, tail
from nocturne.bench.live.__main__ import is_finished, start

SERVER_LINES = [
    "0.00.116.020 I cmn  common_param:   - MTL0    : Apple M5 (18186 MiB, 18185 MiB free)",
    "0.00.220.501 I cmn  common_param:   - RPC0    : 192.168.1.219:50052 (12124 MiB, 12123 MiB free)",
    "0.00.226.012 I srv    load_model: loading model '/Users/orosentswaig/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'",
    "0.02.506.570 I load_tensors: offloaded 29/29 layers to GPU",
    "0.02.506.578 I load_tensors:  MTL0_Mapped model buffer size =  3082.78 MiB",
    "0.02.506.580 I load_tensors: RPC0[192.168.1.219:50052] model buffer size =  1586.90 MiB",
    "0.12.682.782 I llama_kv_cache: RPC0[192.168.1.219:50052] KV buffer size =    96.00 MiB",
    "0.13.246.491 I srv  llama_server: listening on http://127.0.0.1:8089",
    "0.14.706.271 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0",
    "0.14.706.275 I slot   operator(): id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 46",
    "0.15.308.380 I slot print_timing: id  0 | task 0 | prompt eval time =     375.90 ms /    46 tokens (    8.17 ms per token,   122.37 tokens per second)",
    "0.15.308.393 I slot print_timing: id  0 | task 0 |        eval time =     226.19 ms /     4 tokens (   75.40 ms per token,    13.26 tokens per second)",
    "0.15.308.591 I slot      release: id  0 | task 0 | stop processing: n_tokens = 49, truncated = 0",
    "0.15.308.604 I srv  update_slots: all slots are idle",
    "1.37.835.463 I slot print_timing: id  0 | task 405 | n_gen =    100, tg =   8.31 t/s, tg_3s =   8.40 t/s",
    "0.15.690.535 I slot print_timing: id  0 | task 0 | draft acceptance = 0.87500 (    7 accepted /     8 generated), mean len =  8.00",
    "\ttop_k = 40",
]

WORKER_LINES = [
    "[hello] version: 6.0.0",
    "[alloc_buffer] device: 0, size: 1663985664 -> remote_ptr: 76f6c50200, remote_size: 1663985664",
    "[set_tensor] buffer: 0x76f6c50200, data: 0x7000003800, offset: 0, size: 7225344",
    "[set_tensor_hash] buffer: 0x76f6c50200, data: 0x700103f000, offset: 0, size: 38191104, hash: 3599848693a0d283",
    "[graph_compute] device: 0, n_nodes: 408, n_tensors: 581",
    "[get_tensor] buffer: 0x75ccc4c380, data: 0x11634c000, offset: 0, size: 14336",
    "[set_tensor] buffer: 0x75ccc4c380, data: 0x11634c000, offset: 0, size: 14336",
    "[set_tensor] buffer: 0x75ccc4c380, data: 0x116a4c000, offset: 0, size: 4",
    "[set_tensor] buffer: 0x75ccc4c380, data: 0x116a4c800, offset: 0, size: 8",
    "[set_tensor] buffer: 0x75ccc4c380, data: 0x116a4d800, offset: 0, size: 8",
    "[set_tensor] buffer: 0x75ccc4c380, data: 0x116a4e800, offset: 0, size: 512",
    "[graph_recompute] device: 0",
    "[get_tensor] buffer: 0x75ccc4c380, data: 0x11634c000, offset: 0, size: 14336",
    "ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_rms_norm_mul_f32_4'",
]


def test_server_ts():
    assert parsers.server_ts("0.15.308.380 I x") == 15.30838
    assert abs(parsers.server_ts("1.37.835.463 I x") - 97.835463) < 1e-6
    assert parsers.server_ts("\ttop_k = 40") is None


def test_parse_server_lines():
    kinds = [(parsers.parse_server_line(l) or {}).get("kind") for l in SERVER_LINES]
    assert kinds == ["device", "device", "load_start", "offloaded", "buffer", "buffer", "buffer", "listening", "task_start",
                     "prompt_tokens", "prompt_done", "eval_done", "release", "idle", "progress", "draft", None]
    b = parsers.parse_server_line(SERVER_LINES[5])
    assert b["device"] == "RPC0[192.168.1.219:50052]" and b["what"] == "model" and b["mib"] == 1586.9
    e = parsers.parse_server_line(SERVER_LINES[11])
    assert e["slot"] == 0 and e["task"] == 0 and e["n"] == 4 and e["ms"] == 226.19
    g = parsers.parse_server_line(SERVER_LINES[14])
    assert g["n_gen"] == 100 and g["tg_3s"] == 8.4
    d = parsers.parse_server_line(SERVER_LINES[15])
    assert d["accepted"] == 7 and d["generated"] == 8


def test_parse_worker_lines_and_step_tracker():
    assert parsers.parse_worker_line(WORKER_LINES[2]) == {"cmd": "set_tensor", "size": 7225344, "ts": None}
    assert parsers.parse_worker_line(WORKER_LINES[11]) == {"cmd": "graph_recompute", "size": None, "ts": None}
    stamped = parsers.parse_worker_line("1793019800.833541 " + WORKER_LINES[2])
    assert stamped == {"cmd": "set_tensor", "size": 7225344, "ts": 1793019800.833541}
    assert parsers.parse_worker_line(WORKER_LINES[-1]) is None
    tr = parsers.StepTracker()
    shown = []
    for l in WORKER_LINES:
        ev = parsers.parse_worker_line(l)
        if ev:
            m = tr.feed(ev["cmd"], ev["size"])
            if m:
                shown.append(m)
    s = tr.snapshot()
    assert s["step"] == 2 and s["cache_hits"] == 1 and s["cache_bytes"] == 38191104
    # second step: graph_recompute + get_tensor so far; the first full step had 7 commands
    assert s["cmds_this_step"] == 2 and s["cmds_per_step_recent"] == 7
    assert s["bytes_out"] == 2 * 14336 and s["bytes_in"] == 7225344 + 14336 + 4 + 8 + 8 + 512
    assert s["state"] == "waiting"
    assert [m for m in shown if m["step_start"]][0]["step"] == 1
    assert shown[0]["cmd"] == "set_tensor" and shown[0]["step"] == 0   # load-phase line, no step yet


def test_format_sse():
    assert parsers.format_sse("state", {"a": 1}) == b'event: state\ndata: {"a":1}\n\n'
    assert parsers.format_sse("x", "l1\nl2") == b"event: x\ndata: l1\ndata: l2\n\n"


def test_run_state_from_fixture_lines():
    st = tail.RunState("fx", "live", [{"name": "m5", "role": "requester", "backend": "metal"},
                                       {"name": "air", "role": "worker", "backend": "metal"}])
    now = time.time()
    st.feed_event(json.dumps({"event": "request", "prompt_id": "arith_01"}), now)
    for l in SERVER_LINES[:9]:
        st.feed_server(l, now)
    snap = st.snapshot(now)
    air = next(p for p in snap["peers"] if p["name"] == "air"); m5 = next(p for p in snap["peers"] if p["name"] == "m5")
    assert m5["device"] == "MTL0" and m5["mem_mib"] == 18186 and air["mem_mib"] == 12124
    assert abs(m5["buffers_gb"] - 3082.78 / 1024) < 1e-3 and abs(air["buffers_gb"] - (1586.9 + 96) / 1024) < 1e-3
    assert snap["phase"] == "prefill" and snap["load"]["load_s"] == 13.02
    assert snap["tokens"]["prompt_in_flight"] == ["arith_01"]
    for l in SERVER_LINES[9:]:
        st.feed_server(l, now)
    for l in WORKER_LINES:
        st.feed_worker("air", l, now)
    snap = st.snapshot(now)
    assert snap["tokens"]["generated"] == 4 and snap["tokens"]["requests_done"] == 1 and snap["tokens"]["draft_accepted"] == 7
    assert snap["load"]["cache_state"] == "warm"
    assert snap["seq"] == 11 and len(st.messages_since(0)) == 11     # 2 load lines + 7 + 2 shown commands
    assert st.messages_since(9)[0]["seq"] == 10


def _fake_run(tmp_path: Path) -> Path:
    d = tmp_path / "logs" / "A1_test_r1"; d.mkdir(parents=True)
    (d / "server.log").write_text("\n".join(SERVER_LINES[:14]) + "\n")
    (d / "worker_air.log").write_text("\n".join(WORKER_LINES) + "\n")
    return d


def test_build_schedule_interleaves_worker_by_step(tmp_path):
    d = _fake_run(tmp_path)
    sched = tail.build_schedule(d)
    assert [t for t, _, _ in sched] == sorted(t for t, _, _ in sched)
    worker = [(t, l) for t, n, l in sched if n == "worker_air.log"]
    # load-phase lines sit between load_start (0.226) and listening (13.246); compute steps inside the request span
    assert 0.226 <= worker[0][0] < 13.25 and worker[0][1].startswith("[hello]")
    graph = [t for t, l in worker if l.startswith("[graph_")]
    assert all(14.706 <= t <= 15.309 for t in graph)
    assert not is_finished(d)                      # files just written, no done event: looks live
    (d / "events.jsonl").write_text(json.dumps({"event": "done", "t": 1.0}) + "\n")
    assert is_finished(d)


def test_tailer_reads_appended_lines(tmp_path):
    d = _fake_run(tmp_path)
    st = tail.RunState("x", "live")
    tl = tail.Tailer(d, st)
    tl.poll_once(1.0)
    assert st.snapshot(1.0)["tokens"]["generated"] == 4
    with (d / "worker_air.log").open("a") as f:
        f.write("[graph_recompute] device: 0\n[get_tensor] buffer: 0x1, data: 0x1, offset: 0, size: 14336\n")
    tl.poll_once(2.0)
    assert st.trackers["air"].step == 3


def test_http_server_and_sse(tmp_path):
    d = _fake_run(tmp_path)
    srv, url, feeder = start(d, port=0, replay=True, speed=1000)
    try:
        host, port = srv.server_address
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/"); r = c.getresponse(); body = r.read()
        assert r.status == 200 and b"<title>Nocturne bench live</title>" in body and b"cdn" not in body.lower()
        c.request("GET", "/events"); r = c.getresponse()
        assert r.getheader("Content-Type") == "text/event-stream"
        line = r.fp.readline()
        assert line == b"event: state\n"
        data = json.loads(r.fp.readline()[len("data: "):])
        assert data["run_id"] == "A1_test_r1" and data["mode"] == "replay"
        c.close()
        time.sleep(0.5)
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/state"); snap = json.loads(c.getresponse().read())
        assert snap["done"] and snap["tokens"]["generated"] == 4 and snap["phase"] == "done"
    finally:
        feeder.stop(); srv.shutdown()


def test_serve_falls_back_when_port_is_busy(tmp_path):
    st = tail.RunState("x", "live")
    a, url_a = server.serve(st, 0)
    try:
        b, url_b = server.serve(st, a.server_address[1])      # same port: must not raise
        assert b.server_address[1] != a.server_address[1] and url_b.startswith("http://127.0.0.1:")
        b.shutdown()
    finally:
        a.shutdown()


def test_server_swallows_connection_resets(capsys):
    srv = server.Server.__new__(server.Server)
    try:
        raise ConnectionResetError(54, "Connection reset by peer")
    except ConnectionResetError:
        srv.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""


def test_events_drive_phase_without_slot_logs():
    """torch_tp writes no llama.cpp slot lines: request/response events alone must leave the load phase."""
    st = tail.RunState("tt", "live", [{"name": "m5", "role": "requester", "backend": None}])
    now = 100.0
    st.feed_event(json.dumps({"event": "launch", "t": now}), now)
    assert st.phase == "load"
    st.feed_event(json.dumps({"event": "healthy", "load_s": 19.2}), now + 20)
    st.feed_event(json.dumps({"event": "request", "prompt_id": "longopen_01"}), now + 21)
    snap = st.snapshot(now + 22)
    assert snap["phase"] == "prefill"
    assert next(p for p in snap["peers"] if p["role"] == "requester")["state"] == "computing"
    st.feed_event(json.dumps({"event": "response", "prompt_id": "longopen_01", "n_gen": 256}), now + 200)
    assert st.phase == "decode"
    assert next(p for p in st.snapshot(now + 201)["peers"] if p["role"] == "requester")["state"] == "idle"


def test_in_flight_estimate_from_previous_response():
    st = tail.RunState("tt", "live", [{"name": "m5", "role": "requester", "backend": None}])
    now = 100.0
    st.feed_event(json.dumps({"event": "request", "prompt_id": "p1"}), now)
    assert st.snapshot(now + 5)["tokens"]["in_flight"]["est_tokens"] is None
    st.feed_event(json.dumps({"event": "response", "prompt_id": "p1", "n_gen": 256}), now + 255)   # 1 s per step
    tk = st.snapshot(now + 256)["tokens"]
    assert tk["requests_done"] == 1 and tk["generated"] == 256 and abs(tk["tps_current"] - 1.0) < 1e-6
    assert "in_flight" not in tk and tk["prompt_in_flight"] == []
    st.feed_event(json.dumps({"event": "request", "prompt_id": "p2"}), now + 256)
    f = st.snapshot(now + 256 + 100)["tokens"]["in_flight"]
    assert f["prompt_id"] == "p2" and f["est_tokens"] == 101 and f["est_of"] == 256
    assert st.snapshot(now + 256 + 1000)["tokens"]["in_flight"]["est_tokens"] == 256


def test_late_attach_uses_event_timestamps():
    st = tail.RunState("tt", "live", [{"name": "m5", "role": "requester", "backend": None}])
    now = 5000.0
    st.feed_event(json.dumps({"event": "request", "prompt_id": "p1", "t": 1000.0}), now)
    st.feed_event(json.dumps({"event": "response", "prompt_id": "p1", "n_gen": 256, "t": 1255.0}), now + 0.001)
    assert abs(st.snapshot(now)["tokens"]["tps_current"] - 1.0) < 1e-6


TORCH_LINES = [
    "[torch_tp] loading /Users/x/models/hf/Qwen2.5-7B-Instruct",
    "[torch_tp] rank 0 loaded 7.62 GB in 21.3 s on mps",
    "[torch_tp] serving http://127.0.0.1:8092 (world 2)",
    "[torch_tp] request req-1 prompt_n=78 max_tokens=256",
    "[torch_tp] prefill req-1 n=78 ms=2063.0",
    "[torch_tp] decode req-1 n_gen=16 ms_per_step=590.0",
    "[torch_tp] done req-1 n_gen=256 predicted_ms=150000.0 finish=length",
    "[torch_tp] release req-1",
]


def test_parse_torch_tp_lines():
    kinds = [(parsers.parse_server_line(l) or {}).get("kind") for l in TORCH_LINES]
    assert kinds == ["load_start", None, "listening", "task_start", "prompt_done", "progress", "eval_done", "release"]
    t = parsers.parse_server_line(TORCH_LINES[3])
    assert t["slot"] == 0 and t["task"] == 1 and t["n_prompt"] == 78 and t["ts"] is None
    g = parsers.parse_server_line(TORCH_LINES[5])
    assert g["n_gen"] == 16 and abs(g["tg_3s"] - 1000 / 590) < 1e-6
    assert parsers.parse_server_line("[torch_tp] decode req-1 n_gen=x") is None


def test_torch_tp_lines_drive_slots_and_phases():
    st = tail.RunState("B1t_test", "live")
    now = 1000.0
    st.feed_event(json.dumps({"event": "launch", "t": now}), now)
    for i, l in enumerate(TORCH_LINES[:3]):
        st.feed_server(l, now + 1 + i)
    st.feed_event(json.dumps({"event": "request", "prompt_id": "longopen_01", "t": now + 30}), now + 30)
    for i, l in enumerate(TORCH_LINES[3:]):
        st.feed_server(l, now + 30 + i)
    assert st.snapshot(now + 40)["phase"] == "decode"
    slot = st.slots[0]
    assert slot["prompt_id"] == "longopen_01" and slot["n_prompt"] == 78 and slot["n_gen"] == 256 and slot["state"] == "done"
    assert st.tokens["generated"] == 256 and st.tokens["requests_done"] == 1 and st.tokens["prompt_in_flight"] == []
    assert [p["name"] for p in st.phases] == ["probe", "load", "prefill", "decode"]
