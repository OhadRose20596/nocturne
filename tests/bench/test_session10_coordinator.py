"""Session 10, part B: any host can be the coordinator and the planner. The operator (the machine with peers.yaml and ssh)
starts every process; the coordinator process plans and runs on the host the placement names."""
import io
import json
import os
import socket
import subprocess
import sys
import tarfile

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from nocturne.bench.runner import Runner  # noqa: E402
from nocturne.engines.torch_tp import coordinate, hybrid_run as hr, operate  # noqa: E402
from nocturne.engines.torch_tp.config import load_config  # noqa: E402
from nocturne.engines.torch_tp.launch import model_dir_of  # noqa: E402
from nocturne.engines.torch_tp.plan import (Dims, PlanError, coordinator_host, listen_addrs, needed_peers, placement_from_dict,  # noqa: E402
                                            plan, slots)

from .test_hybrid import Q7B, free_ports  # noqa: E402
from .test_hybrid_run_os import Fake  # noqa: E402
from .test_torch_tp import write_tiny_model  # noqa: E402

HOSTNAME = socket.gethostname().split(".")[0]


def cuda_pair_yaml(coord="gtx", model="qwen2.5-7b-q8"):
    return {"model": model, "coordinator": {"host": coord}, "max_seq_len": 512, "stages": [
        {"name": "s0", "layers": [0, 12], "embed": True, "ranks": [{"host": "gtx", "device": "cuda", "port": 50070}]},
        {"name": "s1", "layers": [12, 28], "lm_head": True, "ranks": [{"host": "pc", "device": "cuda", "port": 50070},
                                                                     {"host": "gtx", "device": "cuda", "port": 50071}]}]}


PEERS = {"models": {"qwen2.5-7b-q8": {"path": "~/models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf", "quant": "Q8_0"}},
         "peers": {"m5": {"host": "MMF4FWRG4R", "ssh": None, "os": "macos", "python": "/m5/python", "bind": "192.168.1.242"},
                   "gtx": {"host": "Ohad", "ssh": "gtx", "os": "linux", "python": "/home/ohadr/venv/bin/python", "bind": "192.168.1.168",
                           "models": {"qwen2.5-7b-q8": "/mnt/d/models/Qwen2.5-7B-Instruct-Q8_0.gguf"}},
                   "pc": {"host": "lioryahel", "ssh": "p16v", "os": "linux", "python": "/home/lior/venv/bin/python", "bind": "192.168.1.77"}}}


def write(tmp_path, placement, peers=PEERS):
    (tmp_path / "pl.yaml").write_text(yaml.safe_dump(placement))
    (tmp_path / "peers.yaml").write_text(yaml.safe_dump(peers))
    return str(tmp_path / "pl.yaml"), str(tmp_path / "peers.yaml")


# ---- the planner side -------------------------------------------------------------------------------------------------------
def test_slots_need_no_model_and_agree_with_the_full_plan():
    y = cuda_pair_yaml()
    y["allow_mixed_devices"] = True
    p = placement_from_dict(y)
    sl, specs = slots(p), plan(Q7B, p, check_budget=False)
    for a, b in zip(sl, specs, strict=True):
        for f in ("gid", "name", "stage", "stage_name", "group_rank", "group_size", "host", "device", "port"):
            assert getattr(a, f) == getattr(b, f), f
        assert needed_peers(a, sl) == needed_peers(b, specs)
    hosts = {"gtx": "192.168.1.168", "pc": "192.168.1.77"}
    assert listen_addrs(sl, hosts) == listen_addrs(specs, hosts)
    assert listen_addrs(sl, hosts)[0] == ["127.0.0.1", "192.168.1.168"]     # both ranks of the next stage connect to gid 0


def test_coordinator_host_is_rank_0_of_the_first_stage_and_must_match_the_key():
    assert coordinator_host(placement_from_dict(cuda_pair_yaml("gtx"))) == "gtx"
    y = cuda_pair_yaml("gtx")
    y.pop("coordinator")
    assert coordinator_host(placement_from_dict(y)) == "gtx"
    with pytest.raises(PlanError, match="on gtx, not on pc"):
        coordinator_host(placement_from_dict(cuda_pair_yaml("pc")))
    with pytest.raises(PlanError, match="on gtx, not on m5"):
        slots(placement_from_dict(cuda_pair_yaml("m5")))
    y = cuda_pair_yaml()
    y["stages"][1]["ranks"][1]["port"] = 50070
    with pytest.raises(PlanError, match="port 50070 used twice on gtx"):
        slots(placement_from_dict(y))


def test_model_path_may_differ_per_host():
    assert model_dir_of("qwen2.5-7b-q8", PEERS, "gtx") == "/mnt/d/models/Qwen2.5-7B-Instruct-Q8_0.gguf"
    assert model_dir_of("qwen2.5-7b-q8", PEERS, "pc") == "~/models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf"
    assert model_dir_of("qwen2.5-7b-q8", PEERS) == "~/models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf"
    assert model_dir_of("/abs/file.gguf", PEERS, "gtx") == "/abs/file.gguf"


def test_coordinator_process_refuses_another_host():
    job = {"coordinator": "gtx", "coordinator_hostname": "Ohad"}
    coordinate.check_host(job, "ohad")
    coordinate.check_host(dict(job, coordinator_hostname="Ohad.local"), "Ohad")
    coordinate.check_host({"coordinator": "x", "coordinator_hostname": None}, "anything")
    with pytest.raises(RuntimeError, match="coordinator host is gtx .Ohad.; this machine is MMF4FWRG4R"):
        coordinate.check_host(job, "MMF4FWRG4R")


# ---- the operator: who is started where -----------------------------------------------------------------------------------------
def test_operator_starts_the_coordinator_on_the_placement_host_and_ranks_on_theirs(tmp_path):
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"))
    op = operate.Operator(pl, peers, tmp_path / "run", tag="R1_test", ship_weights=False, runner=Runner(dry_run=True, echo=False))
    assert op.coord == "gtx" and op.model_dir == "/mnt/d/models/Qwen2.5-7B-Instruct-Q8_0.gguf"
    assert op.host_dir("gtx") == op.host_dir("pc") == "~/.nocturne/runs/R1_test" and op.collected_hosts() == ["gtx", "pc"]
    job = op.job([{"id": "p", "ids": [1], "n": 2}], timeline=True)
    assert job["coordinator_hostname"] == "Ohad" and job["hosts"] == {"gtx": "192.168.1.168", "pc": "192.168.1.77"}
    op.push_job(job)
    assert op.plan()["ok"]
    op.start_ranks()
    op.run_coordinator()
    op.stop()
    cmds = op.commands()
    by = {h: [c["cmd"] for c in cmds if c["host"] == h] for h in ("m5", "gtx", "pc")}
    assert by["m5"] == []                                              # the operator's own machine runs nothing
    assert "ssh" in by["gtx"][0] and "gtx" in by["gtx"][0] and "cat > ~/.nocturne/runs/R1_test/job.json" in by["gtx"][0]
    coord = [c for c in by["gtx"] if "torch_tp.coordinate" in c]
    assert len(coord) == 2 and "--plan-only" in coord[0] and "--plan-only" not in coord[1]
    assert all("/home/ohadr/venv/bin/python" in c and "R1_test/job.json" in c and "--tag R1_test" in c for c in coord)
    rank_gtx = [c for c in by["gtx"] if " rank " in c]
    rank_pc = [c for c in by["pc"] if " rank " in c]
    assert len(rank_gtx) == 1 and "--port 50071" in rank_gtx[0] and "--name s1/1" in rank_gtx[0]
    assert "--listen 127.0.0.1 " in rank_gtx[0]                         # only the coordinator, on the same host, connects to it
    assert len(rank_pc) == 1 and "/home/lior/venv/bin/python" in rank_pc[0] and "p16v" in rank_pc[0] and "--listen 192.168.1.77" in rank_pc[0]
    assert "--model /mnt/d/models/Qwen2.5-7B-Instruct-Q8_0.gguf" in rank_gtx[0]
    assert "--model" in rank_pc[0] and "models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf" in rank_pc[0]
    assert "0.0.0.0" not in json.dumps(cmds)
    assert any("pkill" in c for c in by["pc"]) and any("pkill" in c for c in by["gtx"])


def test_operator_gives_no_model_to_ranks_when_weights_are_shipped(tmp_path):
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"))
    op = operate.Operator(pl, peers, tmp_path / "run", tag="R2", ship_weights=True, cache_dir="/c", runner=Runner(dry_run=True, echo=False))
    for s in op.slots[1:]:
        argv = op.rank_argv(s)
        assert "--model" not in argv and argv[argv.index("--cache-dir") + 1] == "/c"
        assert argv[argv.index("--run-dir") + 1] == "~/.nocturne/runs/R2"
    assert hr.model_hosts(op, ship=True) == ["gtx"] and hr.model_hosts(op, ship=False) == ["gtx", "pc"]


def test_a_rank_without_device_gets_the_backend_of_its_host(tmp_path):
    y = cuda_pair_yaml("gtx")
    for st in y["stages"]:
        for r in st["ranks"]:
            r.pop("device")
    peers = json.loads(json.dumps(PEERS))
    peers["peers"]["gtx"]["backend"], peers["peers"]["pc"]["backend"] = "cuda", "metal"
    pl, pf = write(tmp_path, y, peers)
    op = operate.Operator(pl, pf, tmp_path, tag="t", runner=Runner(dry_run=True, echo=False))
    assert [s.device for s in op.slots] == ["cuda", "mps", "cuda"]
    assert op.job([])["placement"]["stages"][1]["ranks"][0]["device"] == "mps"      # the coordinator plans with the filled devices
    peers["peers"]["pc"].pop("backend")
    pl, pf = write(tmp_path, y, peers)
    with pytest.raises(ValueError, match="rank on pc has no device"):
        operate.Operator(pl, pf, tmp_path, tag="t")


def test_operator_refuses_a_bad_tag_and_zero_bind(tmp_path):
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"))
    with pytest.raises(ValueError, match="run tag"):
        operate.Operator(pl, peers, tmp_path, tag="a b; rm")
    bad = json.loads(json.dumps(PEERS))
    bad["peers"]["pc"]["bind"] = "0.0.0.0"
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"), bad)
    with pytest.raises(ValueError, match="0.0.0.0"):
        operate.Operator(pl, peers, tmp_path, tag="t")


def test_collected_tar_keeps_only_plain_files_of_the_run_folder(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (("./timeline_s0_0.jsonl.gz", b"a"), ("../evil", b"b"), ("sub/x", b"c"), (".hidden", b"d"), ("plan.json", b"{}")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("link")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ln)
    assert operate.safe_extract(buf.getvalue(), tmp_path) == ["plan.json", "timeline_s0_0.jsonl.gz"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plan.json", "timeline_s0_0.jsonl.gz"]


# ---- preflight on a Linux coordinator -----------------------------------------------------------------------------------------
def linux_table(transformers_rc=0):
    ts = json.dumps({"host": "x", "sntp": {"offset_s": 0.004, "bound_s": 0.001}, "os_service": {"tool": "chronyc"}})
    return {"true": (0, ""), "nvidia-smi": (0, "GTX 1070, 8192 MiB, 100 MiB, 560\n"), "hashlib": (0, hr.engine_hash_local() + "\n"),
            "timesync": (0, ts + "\n"), "import psutil": (0, "7.2.2\n"), "import transformers": (transformers_rc, "5.21.0\n"),
            "stat -c": (0, "Qwen2.5-7B-Instruct-Q8_0.gguf 8098525760 1700\n"), "sha256sum": (0, "ab12  Qwen2.5-7B-Instruct-Q8_0.gguf\n")}


def test_preflight_with_a_linux_coordinator_uses_linux_tools_and_only_its_model(tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "HASHES", tmp_path / "h.json")
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"))
    op = operate.Operator(pl, peers, tmp_path / "run", tag="t", ship_weights=True, runner=Runner(dry_run=True, echo=False))
    run = Fake(linux_table())
    pre = hr.preflight(op, lambda *a: None, run, ship=True, timeline=True)
    assert pre["coordinator"] == "gtx" and set(pre["hosts"]) == {"gtx", "pc"}
    g, pc = pre["hosts"]["gtx"], pre["hosts"]["pc"]
    assert g["caffeinate"] == "n/a" and g["os"] == "Linux" and "GTX 1070" in g["nvidia_smi"]
    assert g["model_sha256"] == {"Qwen2.5-7B-Instruct-Q8_0.gguf": "ab12"} and "model_sha256" not in pc
    assert g["transformers"] == "5.21.0" and "transformers" not in pc and g["psutil"] == pc["psutil"] == "7.2.2"
    assert g["timesync"]["sntp"]["offset_s"] == 0.004
    text = "\n".join(run.cmds)
    assert "caffeinate" not in text and "shasum" not in text and "scutil" not in text and "stat -f" not in text
    assert "cd /mnt/d/models && sha256sum" in text
    n_hash = sum("sha256sum" in c for c in run.cmds)
    pre2 = hr.preflight(op, lambda *a: None, Fake(linux_table()), ship=False)
    assert "model_sha256" in pre2["hosts"]["pc"] and n_hash == 1
    with pytest.raises(RuntimeError, match="coordinator host gtx cannot import transformers"):
        hr.preflight(op, lambda *a: None, Fake(linux_table(transformers_rc=1)), ship=True)


def test_preflight_stops_when_engine_files_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "HASHES", tmp_path / "h.json")
    pl, peers = write(tmp_path, cuda_pair_yaml("gtx"))
    op = operate.Operator(pl, peers, tmp_path / "run", tag="t", runner=Runner(dry_run=True, echo=False))
    with pytest.raises(RuntimeError, match="engine files on gtx differ from the operator's"):
        hr.preflight(op, lambda *a: None, Fake(dict(linux_table(), hashlib=(0, "deadbeef\n"))))


def test_clock_report_compares_the_engine_offset_with_the_os_sync():
    pre = {"hosts": {"gtx": {"timesync": {"sntp": {"offset_s": 0.010, "bound_s": 0.001}}},
                     "pc": {"timesync": {"sntp": {"offset_s": 0.004, "bound_s": 0.002}}}}}
    m = lambda wall: {"offset_mono_ns": 123, "offset_wall_ns": wall, "rtt_ns": 1000, "at_mono_ns": 0}
    clocks = {"s0/0": {"host": "gtx", "start": m(0), "end": m(0)}, "s1/0": {"host": "pc", "start": m(6_100_000), "end": m(6_200_000)},
              "s1/1": {"host": "gtx", "start": m(40_000), "end": None}, "s1/2": {"host": "pc", "start": m(-9_000_000), "end": None}}
    out = hr.clock_report(clocks, pre, "gtx")
    assert out["s1/0"]["os"] == {"os_predicted_ns": 6_000_000, "gap_ns": 200_000, "disagree": False, "os_bound_ns": 3_000_000}
    assert out["s1/1"]["os"]["same_host"] and out["s1/1"]["os"]["disagree"] is False
    assert out["s1/2"]["os"]["disagree"] is True and out["s1/2"]["os"]["gap_ns"] == -15_000_000
    assert hr.clock_report({"a": {"host": "pc", "start": m(1), "end": None}}, {"hosts": {}}, "gtx")["a"]["os"]["disagree"] is None


def test_prompt_list_of_the_job():
    ps = hr.load_prompts(3, 32, 1, 256)
    assert [p["kind"] for p in ps] == ["eval"] * 3 + ["long"] and ps[0]["n"] == 32 and ps[0]["cap"] == 16 and not ps[0]["ignore_eos"]
    assert ps[3]["n"] == 256 and ps[3]["ignore_eos"] and all("prompt" in p for p in ps)


# ---- end to end: the coordinator under a second peer name, in its own process ----------------------------------------------------
def two_name_setup(tmp_path, tiny, pts):
    peers = {"models": {}, "peers": {
        "opx": {"host": HOSTNAME, "ssh": None, "python": sys.executable, "bind": "127.0.0.1"},
        "co2": {"host": HOSTNAME, "ssh": None, "python": sys.executable, "bind": "127.0.0.1", "run_root": str(tmp_path / "co2_runs")}}}
    y = {"model": str(tiny), "coordinator": {"host": "co2"}, "max_seq_len": 64, "granule": 16, "stages": [
        {"name": "s0", "layers": [0, 1], "embed": True, "ranks": [{"host": "co2", "device": "cpu", "port": pts[0]},
                                                                  {"host": "opx", "device": "cpu", "port": pts[1]}]},
        {"name": "s1", "layers": [1, 2], "lm_head": True, "ranks": [{"host": "opx", "device": "cpu", "port": pts[2]}]}]}
    return write(tmp_path, y, peers)


PROMPTS = [{"id": "p0", "ids": [5, 17, 99, 3], "n": 6, "ignore_eos": True}, {"id": "p1", "ids": [42, 8], "n": 6, "ignore_eos": True}]


def reference_tokens(tiny):
    from .test_session10_timeline import run
    from nocturne.engines.torch_tp.plan import Placement, RankIn, StageIn
    one = Placement("m", (StageIn("s0", (0, 2), (RankIn("h", "cpu", free_ports(1)[0]),), embed=True, lm_head=True),), max_seq_len=64)
    _, out, *_ = run(tiny, one)
    return [o["tokens"] for o in out]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_model(tmp_path_factory.mktemp("s10b") / "t")


def test_loopback_run_with_the_coordinator_under_a_second_name(tmp_path, tiny):
    pytest.importorskip("psutil")
    pl, peers = two_name_setup(tmp_path, tiny, free_ports(3))
    run_dir = tmp_path / "run"
    lines = []
    op = operate.Operator(pl, peers, run_dir, tag="B3_loop", log=lines.append)
    assert op.coord == "co2" and op.collected_hosts() == ["co2"]
    job = op.job(PROMPTS, timeline=True, dtype="float32")
    try:
        op.push_job(job)
        p = op.plan()
        assert p["ok"] and [s["host"] for s in p["specs"]] == ["co2", "opx", "opx"] and p["planner_host"] == HOSTNAME
        op.start_ranks()
        rc = op.run_coordinator()
        got = op.collect()
    finally:
        op.stop()
    assert rc == 0, "\n".join(lines)
    assert not (run_dir / "coordinator_record.json").exists() or "coordinator_record.json" in got["co2"]
    assert {"coordinator_record.json", "first_logits.npz", "plan.json", "job.json", "timeline_s0_0.jsonl.gz", "samples_rank_s0_0.jsonl.gz"} <= set(got["co2"])
    assert (run_dir / "timeline_s0_1.jsonl.gz").exists() and (run_dir / "timeline_s1_0.jsonl.gz").exists()     # ranks of the operator's host
    crec = json.loads((run_dir / "coordinator_record.json").read_text())
    assert crec["outcome"] == "completed" and crec["pid"] != os.getpid() and crec["host"] == HOSTNAME
    assert [q["tokens"] for q in crec["prompts"]] == reference_tokens(tiny)
    assert crec["frames"]["formula_holds"] and set(crec["clock"]) == {"s0/0", "s0/1", "s1/0"}
    assert [r["rid"] for r in crec["requests"]] == ["p0", "p1"]
    assert all(c["host"] in ("co2", "opx") for c in op.commands())
    left = subprocess.run(["pgrep", "-f", "--tag B3_loop"], capture_output=True, text=True).stdout.split()
    assert left == [], f"processes left running: {left}"


def test_coordinator_process_reports_a_rejected_plan_and_a_wrong_host(tmp_path, tiny):
    pl, peers = two_name_setup(tmp_path, tiny, free_ports(3))
    y = yaml.safe_load(open(pl))
    y["stages"][1]["ranks"][0]["budget_gb"] = 1e-6
    open(pl, "w").write(yaml.safe_dump(y))
    op = operate.Operator(pl, peers, tmp_path / "run", tag="B3_rej", log=lambda *a: None)
    op.push_job(op.job(PROMPTS, dtype="float32"))
    p = op.plan()
    assert not p["ok"] and not p.get("fatal") and "over budget: s1/0 on opx" in p["error"]
    job = op.job(PROMPTS, dtype="float32")
    job["coordinator_hostname"] = "some-other-machine"
    op.push_job(job)
    p = op.plan()
    assert not p["ok"] and p["fatal"] and "this machine is" in p["error"]
