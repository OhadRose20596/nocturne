"""Session 10, part A: OS time sync probe, engine clock exchange, recorder, sampler and the timeline outputs.
End to end over loopback with the tiny random Qwen2 of test_torch_tp (fp32, CPU)."""
import csv
import json
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from nocturne.bench.probes import timesync  # noqa: E402
from nocturne.engines.torch_tp import clock, recorder, timeline, transport  # noqa: E402
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.plan import Dims, Placement, RankIn, StageIn, plan  # noqa: E402

from .test_hybrid import free_ports  # noqa: E402
from .test_torch_tp import write_tiny_model  # noqa: E402


# ---- A1: the SNTP formula and the OS tool parsers ------------------------------------------------------------------------
def test_sntp_offset_is_server_minus_local_and_delay_excludes_the_server_hold():
    # local clock 0.250 s behind the server, 10 ms each way, the server holds the packet 1 ms
    t1 = 100.0
    t2 = t1 + 0.250 + 0.010
    t3 = t2 + 0.001
    t4 = t3 - 0.250 + 0.010
    off, delay = timesync.offset_delay(t1, t2, t3, t4)
    assert off == pytest.approx(0.250) and delay == pytest.approx(0.020)


def test_sntp_measure_keeps_the_sample_with_the_smallest_delay_and_reports_failures():
    seq = iter([(0.30, 0.080, 2, "1.2.3.4"), (0.26, 0.012, 2, "1.2.3.4"), (0.20, 0.200, 2, "1.2.3.4")])
    r = timesync.measure("x", 3, query=lambda s: next(seq), pause=0)
    assert r["offset_s"] == 0.26 and r["delay_s"] == 0.012 and r["bound_s"] == 0.006 and r["samples"] == 3

    def fail(server):
        raise OSError("no route")
    r = timesync.measure("x", 2, query=fail, pause=0)
    assert r["offset_s"] is None and "no route" in r["error"]


def test_ntp_reply_parsing():
    import struct
    data = bytes([0x24, 2]) + bytes(30) + struct.pack("!IIII", timesync.NTP_EPOCH + 10, 1 << 31, timesync.NTP_EPOCH + 11, 0)
    assert timesync.parse_reply(data) == (10.5, 11.0, 2)
    with pytest.raises(ValueError):
        timesync.parse_reply(b"short")


def test_os_tool_parsers_and_their_sign():
    c = timesync.parse_chronyc("Reference ID    : A9FEA97B (time.apple.com)\nSystem time     : 0.000120000 seconds fast of NTP time\n")
    assert c == {"tool": "chronyc", "source": "time.apple.com", "offset_s": -0.00012, "last_sync_utc": None}       # ahead: server minus local < 0
    real = "Reference ID    : 65D8456C (ntp-nts-3.ps5.canonical.com)\nStratum         : 3\nRef time (UTC)  : Sat Nov 28 06:18:21 2026\nSystem time     : 0.010830267 seconds slow of NTP time\n"
    assert timesync.parse_chronyc(real) == {"tool": "chronyc", "source": "ntp-nts-3.ps5.canonical.com", "offset_s": 0.010830267,
                                            "last_sync_utc": "Sat Nov 28 06:18:21 2026"}                # the output of gtx on 2026-11-28
    assert timesync.parse_chronyc("System time     : 0.5 seconds slow of NTP time")["offset_s"] == 0.5
    t = timesync.parse_timedatectl("       Server: 17.253.14.125 (time.apple.com)\n       Offset: -1.250ms\n")
    assert t["tool"] == "timedatectl" and t["source"] == "time.apple.com" and t["offset_s"] == pytest.approx(-0.00125)
    assert timesync.parse_sntp("+0.072619 +/- 0.060138 time.apple.com 17.253.14.45") == \
        {"tool": "sntp", "source": "time.apple.com", "offset_s": 0.072619, "err_s": 0.060138}
    w = timesync.parse_w32tm("Leap Indicator: 0\nSource: time.windows.com,0x9\nLast Successful Sync Time: 28/11/2026 06:00:01\n")
    assert w["source"] == "time.windows.com,0x9" and w["last_sync"].startswith("28/11")
    assert timesync.parse_chronyc("") is None and timesync.parse_timedatectl("") is None and timesync.parse_w32tm("") is None


def test_os_status_on_wsl_reads_the_linux_tool(monkeypatch):
    monkeypatch.setattr(timesync.platform, "system", lambda: "Linux")
    monkeypatch.setattr(timesync, "is_wsl", lambda: False)
    out = {"chronyc": "", "timedatectl": "Server: 1.2.3.4 (pool)\nOffset: +2ms"}
    st = timesync.os_status("x", run=lambda cmd, timeout=8.0: out[cmd[0]])
    assert st["tool"] == "timedatectl" and st["offset_s"] == pytest.approx(0.002)


def tcp_pair():
    """Two transport Links joined over loopback TCP."""
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname())
    a, _ = srv.accept()
    srv.close()
    return transport.Link(c), transport.Link(a)


# ---- A2: the clock estimate -------------------------------------------------------------------------------------------------
def _round(t0, rtt, off_mono, off_wall, wall0=10 ** 18, asym=0):
    """One exchange seen by the coordinator: the rank stamps at the middle of the round trip plus `asym`."""
    mid = t0 + rtt // 2 + asym
    return (t0, t0 + rtt, wall0 + t0, wall0 + mid + off_wall, mid + off_mono)


def test_estimate_is_remote_minus_local_and_uses_the_smallest_round_trip():
    samples = [_round(1_000, 900_000, 5_000_000, -3_000_000, asym=200_000), _round(2_000_000, 100_000, 5_000_000, -3_000_000),
               _round(4_000_000, 400_000, 5_000_000, -3_000_000, asym=-50_000)]
    e = clock.estimate(samples)
    assert e["rtt_ns"] == 100_000 and e["rounds"] == 3
    assert e["offset_mono_ns"] == 5_000_000          # remote ahead by 5 ms: positive
    assert e["offset_wall_ns"] == -3_000_000
    assert e["at_mono_ns"] == 2_000_000 + 50_000


def test_event_times_go_to_the_coordinator_clock_with_the_interpolated_offset():
    start = dict(clock.ZERO, offset_mono_ns=5_000_000, at_mono_ns=0)
    end = dict(clock.ZERO, offset_mono_ns=5_002_000, at_mono_ns=100_000_000_000)        # 20 ppb per ns: 2 us over 100 s
    assert clock.to_coordinator(5_000_000, start) == 0                                      # remote 5 ms is coordinator 0
    assert clock.to_coordinator(5_000_000, start, end) == pytest.approx(0, abs=1)
    assert clock.to_coordinator(100_000_000_000 + 5_002_000, start, end) == pytest.approx(100_000_000_000, abs=1)
    mid = 50_000_000_000 + 5_001_000
    assert clock.to_coordinator(mid, start, end) == pytest.approx(50_000_000_000, abs=1)
    d = clock.drift(start, end)
    assert d["drift_ns"] == 2_000 and d["ppm"] == pytest.approx(0.02)
    assert clock.drift(start, None) is None


def test_compare_with_os_sync_flags_a_gap_over_two_ms():
    # coordinator 10 ms behind the server, rank 4 ms behind: the rank's wall clock is 6 ms ahead of the coordinator's
    ok = clock.compare_with_os(6_500_000, 0.010, 0.004)
    assert ok["os_predicted_ns"] == 6_000_000 and ok["gap_ns"] == 500_000 and ok["disagree"] is False
    bad = clock.compare_with_os(-6_000_000, 0.010, 0.004)
    assert bad["gap_ns"] == -12_000_000 and bad["disagree"] is True
    assert clock.compare_with_os(1, None, 0.004)["disagree"] is None


def test_clock_exchange_over_a_real_link_same_host_gives_an_offset_inside_the_round_trip():
    la, lb = tcp_pair()
    t = threading.Thread(target=clock.answer, args=(lb,), daemon=True)
    t.start()
    e = clock.measure(la)
    t.join(timeout=5)
    assert not t.is_alive() and e["rounds"] == clock.ROUNDS
    assert abs(e["offset_mono_ns"]) <= e["rtt_ns"] and abs(e["offset_wall_ns"]) <= e["rtt_ns"] + 2_000_000
    la.close(); lb.close()


# ---- A3: recorder ---------------------------------------------------------------------------------------------------------------
def test_recorder_cursor_leaves_no_gap_inside_a_step(tmp_path):
    r = recorder.Recorder("s0/1", 1, "h", "cpu")
    r.begin(3, 1)
    for k in ("recv_wait", "compute", "host_copy", "send"):
        r.lap(k, "s0/0" if k != "compute" else None, 100 if k == "send" else 0)
    for a, b in zip(r.ev, r.ev[1:]):
        assert a[2] == b[1]
    r.skip()
    r.begin(4, 5)
    r.lap("compute")
    assert r.ev[-1][3] == 1 and r.ev[-1][4] == 4 and r.ev[-1][7] == 5 and r.ev[-1][1] >= r.ev[-2][2]
    head, ev = timeline.read_jsonl(r.write(tmp_path))
    assert head["rank"] == "s0/1" and head["events"] == 5 and head["cols"] == list(recorder.COLS) and len(ev) == 5
    assert (tmp_path / "timeline_s0_1.jsonl.gz").exists()


def test_nvidia_smi_line_parsing_and_command():
    assert recorder.parse_smi("1, 37, 2048\n") == (1, 37.0, 2048.0)
    assert recorder.parse_smi("0, [N/A], 12") is None and recorder.parse_smi("") is None and recorder.parse_smi("37, 2048") is None
    argv = recorder.smi_argv()
    assert argv[0] == "nvidia-smi" and "-i" not in argv and "--query-gpu=index,utilization.gpu,memory.used" in argv      # every GPU of the host


def test_run_tag_from_the_plan_cannot_leave_the_runs_folder():
    """Review F7: the tag comes from the wire when a rank was started without --run-dir."""
    import os
    from nocturne.engines.torch_tp.hybrid import default_run_dir
    root = os.path.expanduser("~/.nocturne/runs")
    assert default_run_dir("R1_ok-2.x") == os.path.join(root, "R1_ok-2.x")
    for bad in ("../../etc", "/abs/path", "a/b", "", None, "x\n"):
        assert default_run_dir(bad) == os.path.join(root, "untagged"), bad


def test_sampler_writes_cumulative_counters(tmp_path):
    pytest.importorskip("psutil")
    s = recorder.Sampler("s0/0", "h", "cpu", recorder.ifaces_for(["127.0.0.1"]), True, interval_s=0.005).start()
    x = 0
    for i in range(300_000):
        x += i * i
    __import__("time").sleep(0.06)
    s.stop()
    head, rows = timeline.read_jsonl(s.write(tmp_path))
    assert head["host_level"] and head["ifaces"] and len(rows) >= 5
    assert rows[-1]["pc"] > rows[0]["pc"] and rows[-1]["hb"] >= rows[0]["hb"] and rows[-1]["t"] > rows[0]["t"]
    assert all(set(r["net"]) == set(head["ifaces"]) for r in rows)
    assert not any(t.is_alive() for t in s._threads)


# ---- A3: end to end over loopback -------------------------------------------------------------------------------------------
def tp_pp_placement(pts):
    return Placement("m", (StageIn("s0", (0, 1), (RankIn("h", "cpu", pts[0]), RankIn("h", "cpu", pts[1]), RankIn("h", "cpu", pts[2])), embed=True),
                           StageIn("s1", (1, 2), (RankIn("h", "cpu", pts[3]),), lm_head=True)), max_seq_len=64, granule=16)


def run(model_dir, placement, tmp=None, timeline_on=False, n_tokens=6, prompts=((5, 17, 99, 3), (42, 8))):
    from nocturne.engines.torch_tp.hybrid import Coordinator, decode_frames, rank_main
    specs = plan(Dims.of(load_config(model_dir)), placement)
    threads = []
    for s in specs[1:]:
        t = threading.Thread(target=rank_main, args=(["127.0.0.1"], s.port, str(model_dir), "cpu"),
                             kwargs={"log": lambda *a: None, "run_dir": str(tmp) if tmp else None}, daemon=True)
        t.start()
        threads.append(t)
    co = Coordinator(specs, str(model_dir), {"h": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                     placement.prefill_max, device="cpu", log=lambda *a: None, dtype="float32", timeline=timeline_on,
                     run_dir=str(tmp) if tmp else None)
    try:
        m0 = co.metrics()
        out = [co.generate(list(p), n_tokens, ignore_eos=True, first_logits=True, rid=f"p{i}") for i, p in enumerate(prompts)]
        m1 = co.metrics()
    finally:
        co.shutdown()
    for t in threads:
        t.join(timeout=10)
    return specs, out, co, decode_frames(m0), decode_frames(m1), m1


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_model(tmp_path_factory.mktemp("s10") / "t")


def test_timeline_on_gives_identical_tokens_and_logits(tiny, tmp_path):
    pl_off, pl_on = tp_pp_placement(free_ports(4)), tp_pp_placement(free_ports(4))
    _, off, co_off, *_ = run(tiny, pl_off)
    _, on, co_on, *_ = run(tiny, pl_on, tmp_path, timeline_on=True)
    for a, b in zip(off, on):
        assert a["tokens"] == b["tokens"]
        assert torch.equal(a["first_logits"], b["first_logits"])
    assert co_off.rank.rec is None and co_off.run_dir is None
    assert sorted(p.name for p in tmp_path.glob("timeline_*.jsonl.gz")) == ["timeline_s0_0.jsonl.gz", "timeline_s0_1.jsonl.gz", "timeline_s0_2.jsonl.gz",
                                                                          "timeline_s1_0.jsonl.gz"]


def test_clock_is_measured_at_start_and_end_without_a_timeline(tiny):
    specs, _, co, *_ = run(tiny, tp_pp_placement(free_ports(4)))
    rec = co.clock_record()
    assert set(rec) == {s.name for s in specs}
    assert rec["s0/0"]["start"]["offset_mono_ns"] == 0
    for name in ("s0/1", "s0/2", "s1/0"):
        st, en = rec[name]["start"], rec[name]["end"]
        assert st["rounds"] == en["rounds"] == clock.ROUNDS
        assert abs(st["offset_mono_ns"]) <= st["rtt_ns"] and abs(en["offset_mono_ns"]) <= en["rtt_ns"]      # one host, one clock
        assert rec[name]["drift"]["span_s"] > 0
    assert [r["rid"] for r in co.requests] == ["p0", "p1"] and [r["req"] for r in co.requests] == [1, 2]


def test_recorded_spans_tile_every_step_and_bytes_equal_the_transport_counters(tiny, tmp_path):
    specs, out, co, f0, f1, m1 = run(tiny, tp_pp_placement(free_ports(4)), tmp_path, timeline_on=True)
    r = timeline.load(tmp_path, co.clock_record())
    assert [x["meta"]["rank"] for x in r["ranks"]] == [s.name for s in specs] and all(x["aligned"] for x in r["ranks"])
    steps = 2 * 6                                     # per prompt: one prefill step and five decode steps
    for rk in r["ranks"]:
        ev = [e for e in rk["events"] if e["step"] >= 0]
        assert {e["step"] for e in ev} == set(range(steps)), rk["meta"]["rank"]
        assert {e["req"] for e in ev} == {1, 2}
        assert {e["kind"] for e in ev} <= set(recorder.KINDS)
        if rk["meta"]["gid"]:
            for a, b in zip(ev, ev[1:]):               # a remote rank's spans leave no gap, across steps too
                assert a["t1"] == pytest.approx(b["t0"], abs=2)
    kinds = {rk["meta"]["rank"]: {e["kind"] for e in rk["events"]} for rk in r["ranks"]}
    assert {"compute", "host_copy", "send", "allreduce_wait"} <= kinds["s0/1"] and "recv_wait" in kinds["s0/1"]
    assert "allreduce_wait" not in kinds["s1/0"] and {"recv_wait", "compute", "send"} <= kinds["s1/0"]
    for name in ("s0/1", "s0/2"):       # review F17: a rank that neither forwards nor holds the head ends its step with compute
        ev = next(x["events"] for x in r["ranks"] if x["meta"]["rank"] == name)
        last = {}
        for e in ev:
            if e["step"] >= 0:
                last[e["step"]] = e["kind"]
        assert set(last.values()) == {"compute"}, name
    assert "other" in kinds["s0/0"] and "recv_wait" in kinds["s0/0"]          # the coordinator waits for the head rank
    rows = timeline.breakdown(r, co.requests)
    s = timeline.summary(rows)
    dsteps = m1["decode_steps"]
    assert s["decode_steps"] == dsteps == 10
    assert s["decode_bytes_total"] == f1["sent_bytes"] - f0["sent_bytes"]      # recorder against the Link counters
    assert all(l["bytes"] > 0 for l in s["links"].values())
    assert {"s0/0 -> s0/1", "s0/0 -> s1/0", "s1/0 -> s0/0", "s0/2 -> s0/0"} <= set(s["links"])
    # the group peers of a TP stage exchange the same bytes both ways
    assert s["links"]["s0/1 -> s0/2"]["bytes"] == s["links"]["s0/2 -> s0/1"]["bytes"]
    # direction: the last stage receives a hidden state (fp32 here, 64 values) and sends back a head frame with no payload
    last = next(r for r in rows if r["rank"] == "s1/0" and r["phase"] == "decode")
    assert last["sent"] == {"s0/0": transport.frame_bytes(0)} and last["recv"] == {"s0/0": transport.frame_bytes(64 * 4, 2)}
    assert s["links"]["s0/0 -> s1/0"]["bytes"] == 10 * transport.frame_bytes(64 * 4, 2) and s["links"]["s1/0 -> s0/0"]["bytes"] == 10 * 16
    for row in rows:                                   # the kinds add up to the step
        assert sum(row[k] for k in recorder.KINDS) == pytest.approx(row["t1"] - row["t0"], abs=50)


def test_build_writes_trace_breakdown_samples_and_html(tiny, tmp_path):
    pytest.importorskip("psutil")
    specs, out, co, *_ = run(tiny, tp_pp_placement(free_ports(4)), tmp_path, timeline_on=True, n_tokens=14)
    s = timeline.build(tmp_path, co.clock_record(), co.requests, {"run_id": "t"}, steps=3)
    assert s["decode_steps"] == 26 and not s["not_aligned"]
    tr = json.loads(__import__("gzip").open(tmp_path / "trace.json.gz", "rt").read())
    x = [e for e in tr["traceEvents"] if e["ph"] == "X"]
    assert len(x) == sum(s["events"].values()) and all(e["dur"] >= 0 for e in x)
    assert {e["args"]["name"] for e in tr["traceEvents"] if e["ph"] == "M" and e["name"] == "thread_name"} >= {"s0/0 compute", "s1/0 communication"}
    rows = list(csv.DictReader((tmp_path / "breakdown.csv").open()))
    assert len(rows) == 4 * 28 and {r["phase"] for r in rows} == {"prefill", "decode"}
    r0 = next(r for r in rows if r["rank"] == "s0/1" and r["phase"] == "decode")
    assert float(r0["compute_ms"]) > 0 and int(r0["sent_bytes"]) > 0 and "s0/2=" in r0["sent_by_peer"] and r0["rid"] in ("p0", "p1")
    html = (tmp_path / "timeline.html").read_text()
    assert "/*DATA*/null" not in html and '"first_step"' in html and "mean time per decode step (ms)" in html
    data = json.loads(html.split("const D = ", 1)[1].split(";\nconst $", 1)[0])
    assert data["window"]["last_step"] - data["window"]["first_step"] == 2 and len(data["ranks"]) == 4 and data["spans"]
    assert (tmp_path / "samples_h.jsonl").exists() and "h" in s["hosts"] and s["hosts"]["h"]["ranks"]
    head, _ = timeline.read_jsonl(tmp_path / "samples_rank_s0_0.jsonl.gz")
    assert head["host_level"] is True
    assert timeline.read_jsonl(tmp_path / "samples_rank_s0_1.jsonl.gz")[0]["host_level"] is False     # one host sampler per host
    assert timeline.main([str(tmp_path)]) == 0


def test_window_choice():
    rows = [{"rank": "a", "phase": "decode" if s % 20 else "prefill", "req": 1 + s // 20, "step": s} for s in range(40)]
    assert timeline.pick_window(rows, None, None, 3) == (9, 11)
    assert timeline.pick_window(rows, 2, 30, 2) == (30, 31)
    assert timeline.pick_window([], None, None, 3) is None
