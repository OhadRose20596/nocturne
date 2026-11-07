"""Regression tests for the 2026-10-26 metric fixes: S per token, union of decode windows, commit and host matching."""
from nocturne.bench import record
from nocturne.bench.probes import capture
from .test_probes_and_runner import WORKER_LOG


def test_step_stats_include_graph_message():
    stats, notes = capture.step_stats_from_worker_log(WORKER_LOG, tail=2)
    assert notes == []
    # last two steps: 3 set_tensor + 1 get_tensor + the graph_recompute message = 5
    assert stats["msgs_per_step_median"] == 5.0
    assert stats["compute_steps"] == 4
    assert stats["payload_out_per_step_median"] == 608256
    assert capture.msgs_per_step_from_worker_log(WORKER_LOG, tail=2)[0] == 5.0


def test_msgs_per_token_divides_by_tokens_per_step():
    stats = {"msgs_per_step_median": 7.0, "compute_steps": 716}
    assert abs(capture.msgs_per_token(stats, 714) - 7.0 * 716 / 714) < 1e-9       # A1: one token per step
    assert abs(capture.msgs_per_token({"msgs_per_step_median": 7.0, "compute_steps": 229}, 714) - 2.245) < 0.01  # A2
    assert capture.msgs_per_token(stats, 0) is None
    assert capture.msgs_per_token(None, 10) is None


def test_merge_windows():
    assert capture.merge_windows([(0, 10, 3), (5, 12, 4), (20, 25, 1)]) == [(0, 12), (20, 25)]
    assert capture.merge_windows([(5, 6), (0, 1)]) == [(0, 1), (5, 6)]
    assert capture.merge_windows([]) == []


def test_overlapping_windows_count_bytes_once():
    packets = [(1.0, 1000), (2.0, 1000), (3.0, 1000), (9.0, 500)]
    # three concurrent requests over the same 0..4 s window, 1 token each: 3000 bytes / 3 tokens
    r = capture.bytes_per_token(packets, [(0, 4, 1), (0, 4, 1), (0.5, 4, 1)], compute_steps=1)
    assert r["bytes_per_token"] == 1000.0 and r["packets_per_token"] == 1.0 and r["bytes_per_step"] == 3000.0
    # sequential prompts: both windows, total bytes / total tokens
    r = capture.bytes_per_token(packets, [(0, 4, 3), (8, 10, 1)])
    assert r["bytes_per_token"] == 3500 / 4 and r["bytes_per_step"] is None
    assert capture.bytes_per_token(packets, [(0, 4, 0)]) == {"bytes_per_token": None, "bytes_per_step": None, "packets_per_token": None}


def test_capture_save_and_load_with_direction(tmp_path):
    c = capture.Capture(50052, "en0"); c.packets = [(1.5, 10, "out"), (2.25, 0, "in")]
    c.save(tmp_path / "capture.tsv")
    assert (tmp_path / "capture.tsv").read_text() == "1.500000\t10\tout\n2.250000\t0\tin\n"
    assert capture.load_capture(tmp_path / "capture.tsv") == [(1.5, 10, "out"), (2.25, 0, "in")]
    (tmp_path / "old.tsv").write_text("1.500000\t10\n")          # session 3 format, no direction
    assert capture.load_capture(tmp_path / "old.tsv") == [(1.5, 10, "?")]


def test_capture_direction_icmp_and_decode_step_bytes():
    c = capture.Capture(50052, "en0")
    for line in ["1700000000.100000 IP 192.168.1.20.5555 > 192.168.1.10.50052: tcp 14336",
                 "1700000000.300000 IP 192.168.1.10.50052 > 192.168.1.20.5555: tcp 608256",
                 "1700000000.200000 IP 192.168.1.20 > 192.168.1.10: ICMP echo request, id 77, seq 3, length 64",
                 "1700000000.208000 IP 192.168.1.10 > 192.168.1.20: ICMP echo reply, id 77, seq 3, length 64",
                 "1700000005.000000 IP 192.168.1.20.5555 > 192.168.1.10.50052: tcp 999"]:
        c.feed(line)
    assert c.packets[0][2] == "out" and c.packets[1][2] == "in"
    assert c.icmp == [(1700000000.2, "request", 77, 3), (1700000000.208, "reply", 77, 3)]
    # two prompts: 3 tokens (2 steps) and 1 token (no step, its window is skipped); the late packet is outside
    r = c.per_decode_step([(1700000000.0, 1700000001.0, 3), (1700000004.9, 1700000005.1, 1)], decode_steps=2)
    assert r == {"bytes_per_step_decode": (14336 + 608256) / 2, "bytes_per_step_decode_out": 7168.0, "bytes_per_step_decode_in": 304128.0}
    # old two column captures carry no direction: totals only
    r = capture.bytes_per_decode_step([(1700000000.1, 100, "?")], [(1700000000.0, 1700000001.0, 2)], 1)
    assert r["bytes_per_step_decode"] == 100.0 and r["bytes_per_step_decode_in"] is None
    assert capture.bytes_per_decode_step([], [(0, 1, 2)], None)["bytes_per_step_decode"] is None


def test_clock_offset_from_echo_pairs():
    req = []
    worker = []
    for q in range(10):
        t = 1000.0 + q * 0.01
        req += [(t, "request", 5, q), (t + 0.008, "reply", 5, q)]
        worker.append((t + 0.004 + 2.5, "request", 5, q))            # worker clock 2.5 s ahead, one way 4 ms
    off, unc, n = capture.clock_offset(req, worker)
    assert n == 10 and abs(off - 2.5) < 1e-6 and unc < 1e-6
    assert capture.clock_offset(req[:4], worker[:2]) == (None, None, 2)


def test_decode_comm_from_worker_log():
    log = ("1000.0 [hello]\n"
           "1000.5 [set_tensor] size: 14336\n1000.6 [graph_compute] device: 0, n_nodes: 1, n_tensors: 26\n"
           "1000.7 [get_tensor] size: 28672\n1000.8 [graph_compute] device: 0, n_nodes: 7, n_tensors: 35\n"
           "1001.5 [set_tensor] size: 14336\n1001.6 [graph_compute] device: 0, n_nodes: 1, n_tensors: 26\n"
           "1001.7 [get_tensor] size: 28672\n1001.8 [graph_recompute] device: 0\n"
           "1005.0 [get_tensor] size: 1\n")
    c, notes = capture.decode_comm_from_worker_log(log, [(1000.4, 1002.0, 3)], decode_steps=2)
    assert notes == [] and c["blocking_syncs_per_step"] == 1.0 and c["messages_per_step"] == 4.0
    assert c["graph_meta_bytes_per_step"] == (26 + 35 + 26) * 304 / 2
    assert capture.decode_comm_from_worker_log("[set_tensor] size: 1\n", [(0, 1, 2)], 1)[0]["messages_per_step"] is None
    assert capture.decode_comm_from_worker_log(log, [(1000.4, 1002.0, 1)], 0)[0]["messages_per_step"] is None


def test_same_commit_and_host():
    assert record.same_commit("1e7bcf3da", "1e7bcf3")
    assert record.same_commit("1E7BCF3", "1e7bcf3da")
    assert not record.same_commit("1e7bcf3", "deadbee")
    assert not record.same_commit("", "1e7bcf3")
    assert record.same_host("Mac", "mac") and not record.same_host("Mac", "MMF4FWRG4R")


def test_find_baseline_across_hash_lengths_and_case():
    base = {"cell": {"row": "BASELINE"}, "model": {"name": "m"}, "engine": {"commit": "1e7bcf3"}, "success": True,
            "date": "2026-10-26T09:00:00+02:00", "peers": [{"host": "ohads-MacBook-Air-2", "role": "requester"}]}
    assert record.find_baseline([base], "m", "ohads-macbook-air-2", "1e7bcf3da") is base
    assert record.find_baseline([base], "m", "MMF4FWRG4R", "1e7bcf3da") is None


def test_iface_for_parses_route_output():
    class R:
        def __init__(self, out): self.stdout, self.stderr = out, ""
    mac = lambda *a, **k: R("   route to: 192.168.1.219\ndestination: 192.168.1.219\n  interface: en0\n      flags: <UP,HOST,DONE,LLINFO,WASCLONED,IFSCOPE,IFREF>\n")
    assert capture.iface_for("192.168.1.219", "Darwin", run=mac) == ("en0", [])
    lin = lambda *a, **k: R("192.168.1.219 dev wlan0 src 192.168.1.20 uid 1000\n    cache\n")
    assert capture.iface_for("192.168.1.219", "Linux", run=lin) == ("wlan0", [])
    iface, notes = capture.iface_for("10.9.9.9", "Darwin", run=lambda *a, **k: R(""))
    assert iface is None and notes


def test_comm_from_steps_and_transfer_window():
    c = capture.comm_from_steps({"msgs_per_step_median": 7.0, "compute_steps": 229}, 714)
    assert c["rpc_commands_per_step"] == 7.0 and abs(c["tokens_per_step"] - 714 / 229) < 1e-9
    assert abs(c["rpc_commands_per_token"] - 2.245) < 0.01 and c["compute_steps"] == 229
    assert capture.comm_from_steps(None, 10) == {}
    log = ("1000.000000 [hello] version: 6.0.0\n1000.500000 [set_tensor] buffer: 0x1, size: 7225344\n"
           "1003.250000 [set_tensor_hash] buffer: 0x1, size: 38191104, hash: 1\n1004.000000 [graph_compute] device: 0\n"
           "1004.100000 [set_tensor] buffer: 0x1, size: 14336\n")
    assert capture.transfer_window_from_worker_log(log) == (2.75, [])
    t, notes = capture.transfer_window_from_worker_log("[set_tensor] size: 5\n[graph_compute] device: 0\n")
    assert t is None and "no timestamps" in notes[0]


def test_find_baseline_matches_sha_quant_prompts_and_generation():
    gen = {"temperature": 0, "seed": 42, "samplers": ["temperature"], "repeat_penalty": 1.0}
    def base(**o):
        b = {"cell": {"row": "BASELINE"}, "model": {"name": "m", "quant": "Q4_K_M", "sha256": "a" * 64},
             "engine": {"commit": "1e7bcf3"}, "success": True, "date": "2026-10-26T09:00:00+02:00",
             "prompts": {"set": "eval_set", "version": "1", "generation": gen},
             "peers": [{"host": "Mac", "role": "requester"}]}
        for k, v in o.items():
            d = b
            *path, last = k.split(".")
            for part in path:
                d = d[part]
            d[last] = v
        return b
    model = {"name": "m", "quant": "Q4_K_M", "sha256": "a" * 64}
    pr = {"set": "eval_set", "version": "1", "generation": gen}
    assert record.find_baseline([base()], model, "mac", "1e7bcf3da", pr) is not None
    assert record.find_baseline([base(**{"model.sha256": "b" * 64})], model, "mac", "1e7bcf3", pr) is None
    assert record.find_baseline([base(**{"model.sha256": None})], model, "mac", "1e7bcf3", pr) is not None   # unknown sha matches
    assert record.find_baseline([base(**{"model.quant": "Q8_0"})], model, "mac", "1e7bcf3", pr) is None
    assert record.find_baseline([base(**{"prompts.version": "2"})], model, "mac", "1e7bcf3", pr) is None
    assert record.find_baseline([base(**{"prompts.generation": dict(gen, seed=7)})], model, "mac", "1e7bcf3", pr) is None
    assert record.find_baseline([base(**{"prompts.generation": {}})], model, "mac", "1e7bcf3", pr) is not None  # v1 record
    assert record.find_baseline([base()], "m", "mac", "1e7bcf3") is not None                                  # v1 call


def test_baseline_accuracy_on_same_prompt_ids():
    from nocturne.bench.check import correctness
    rows = [{"id": "arith_01", "category": "arithmetic", "expected": 936},
            {"id": "arith_02", "category": "arithmetic", "expected": 543}]
    base_out = {"arith_01": "936", "arith_02": "wrong", "fact_01": "Tokyo"}
    c = correctness.compare({"arith_01": "936", "arith_02": "543"}, base_out, rows)
    assert c["baseline_task_accuracy"] == 0.5 and c["task_accuracy"] == 1.0 and c["exact_match_ratio"] == 0.5
    assert correctness.passes(c) is True
    assert correctness.passes({"task_accuracy": 0.9, "baseline_task_accuracy": None}) is None
    assert correctness.passes({"task_accuracy": 0.9, "baseline_task_accuracy": 0.96}) is False


def test_code_grader_without_sandbox_is_skipped(monkeypatch):
    from nocturne.bench.check import correctness
    from nocturne.bench.check.graders import code
    monkeypatch.setattr(code, "_prefix", None)
    assert code.grade("def f(): return 1", {"tests": ["assert f() == 1"]}) is None
    rows = [{"id": "code_01", "category": "code", "expected": {"tests": ["assert f() == 1"]}},
            {"id": "arith_01", "category": "arithmetic", "expected": 936}]
    scores, skipped = correctness.grade_all({"code_01": "def f(): return 1", "arith_01": "936"}, rows)
    assert skipped == ["code_01"] and correctness.task_accuracy({"code_01": "x", "arith_01": "936"}, rows) == 1.0
    assert code.sandbox_prefix("Plan9") is None


def test_code_grader_sandboxed_denies_network():
    from nocturne.bench.check.graders import code
    if code.sandbox_prefix() is None:
        return   # no sandbox on this host: the grader skips, covered above
    assert code.grade("def is_even(n):\n    return n % 2 == 0", {"tests": ["assert is_even(2)", "assert not is_even(3)"]}) == 1.0
    net = "import socket\nsocket.create_connection(('192.0.2.1', 80), timeout=1)\ndef f(): return 1"
    assert code.grade(net, {"tests": ["assert f() == 1"]}, timeout_s=5) == 0.0
    assert code.grade("while True: pass", {"tests": ["assert 1"]}, timeout_s=1) == 0.0


def test_mem_sampler_commands_and_peak():
    from nocturne.bench.probes import mem
    assert mem.loop_cmd({"backend": "cuda"}, None, 50052).startswith("nvidia-smi --query-gpu=memory.used")
    assert "ps -o rss= -p 123" in mem.loop_cmd({"backend": "metal", "role": "requester"}, 123, 50052)
    assert "pgrep -f 'ggml-rpc-server.*-p 50052'" in mem.loop_cmd({"backend": "metal", "role": "worker"}, None, 50052)
    s = mem.MemSampler({"name": "air", "backend": "metal", "ssh": "u@h", "role": "worker"})
    assert s.argv()[:3] == ["ssh", "-o", "BatchMode=yes"]
    s.samples = [(1.0, 1.5), (2.0, 3.25), (9.0, 9.0)]
    assert s.peak_gb([(0.5, 2.5)]) == 3.25 and s.peak_gb() == 9.0
    assert s.peak_gb([(100, 101)]) == 9.0          # windows miss: whole-run peak
    assert mem.MemSampler({"name": "x", "backend": "metal", "ssh": None, "role": "worker"}).peak_gb() is None


def test_prompts_version_from_file(tmp_path):
    from nocturne.bench import prompts
    assert (prompts.SET_NAME, prompts.SET_VERSION) == ("eval_set", "1")
    f = tmp_path / "eval_set_v2.jsonl"
    f.write_text('{"id": "a", "category": "arithmetic", "prompt": "1+1", "expected": 2, "max_tokens": 4}\n')
    assert prompts.meta(f) == {"set": "eval_set", "version": "2"} and len(prompts.load(path=f)) == 1
    g = tmp_path / "other.jsonl"
    g.write_text('{"set": "other", "version": "3"}\n{"id": "a", "category": "arithmetic", "prompt": "1+1", "expected": 2, "max_tokens": 4}\n')
    assert prompts.meta(g)["version"] == "3" and prompts.load(path=g)[0]["id"] == "a"
