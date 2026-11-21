"""Session 7: mixed device groups in the planner, and the OS aware preflight helpers with fake command output."""
import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from nocturne.engines.torch_tp import hybrid_run as hr  # noqa: E402
from nocturne.engines.torch_tp.plan import Dims, Placement, PlanError, RankIn, StageIn, placement_from_dict, plan  # noqa: E402

Q05B = Dims(hidden=896, intermediate=4864, n_layers=24, n_heads=14, n_kv_heads=2, vocab=151936, tied=True)


def mixed(**kw):
    s = StageIn("s0", (0, 24), (RankIn("m5", "mps", 1), RankIn("pc", "cuda", 1)), embed=True, lm_head=True)
    return Placement("m", (s,), **kw)


def test_mixed_rejected_by_default():
    with pytest.raises(PlanError, match="mixed device"):
        plan(Q05B, mixed())


def test_mixed_accepted_with_key():
    specs = plan(Q05B, mixed(allow_mixed_devices=True))
    assert [s.device for s in specs] == ["mps", "cuda"]
    assert sum(s.q1 - s.q0 for s in specs) == 14


def test_key_parsed_from_yaml():
    y = {"model": "m", "allow_mixed_devices": True, "stages": [{"layers": [0, 24], "embed": True, "lm_head": True,
         "ranks": [{"host": "a", "device": "mps", "port": 1}, {"host": "a", "device": "cpu", "port": 2}]}]}
    assert placement_from_dict(y).allow_mixed_devices is True
    y.pop("allow_mixed_devices")
    assert placement_from_dict(y).allow_mixed_devices is False


class Fake:
    """A runner with canned output per command prefix; records every command."""

    def __init__(self, table):
        self.table, self.cmds = table, []

    def __call__(self, peer, cmd, timeout=120):
        self.cmds.append(cmd)
        for k, (rc, out) in self.table.items():
            if cmd.startswith(k) or k in cmd:
                return subprocess.CompletedProcess(cmd, rc, out, "")
        return subprocess.CompletedProcess(cmd, 1, "", "unknown")


def test_peer_os_from_entry_and_uname():
    assert hr.peer_os({"os": "macos"}) == "Darwin"
    assert hr.peer_os({"os": "linux"}) == "Linux"
    assert hr.peer_os({}, Fake({"uname -s": (0, "Darwin\n")})) == "Darwin"
    assert hr.peer_os({}, Fake({"uname -s": (0, "Linux\n")})) == "Linux"


def test_stat_and_hash_commands_per_os():
    assert "stat -f '%N %z %m'" in hr.stat_cmd("Darwin", "~/m") and "shasum -a 256" in hr.hash_cmd("Darwin", "~/m")
    assert "stat -c '%n %s %Y'" in hr.stat_cmd("Linux", "~/m") and "sha256sum" in hr.hash_cmd("Linux", "~/m")
    assert "shasum" not in hr.hash_cmd("Linux", "~/m")


def test_model_hashes_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(hr, "HASHES", tmp_path / "h.json")
    run = Fake({"cd ~/m && stat -c": (0, "config.json 10 1700\nmodel.safetensors 20 1701\n"),
                "cd ~/m && sha256sum": (0, "aa  config.json\nbb  model.safetensors\n")})
    sums = hr.model_hashes({}, "pc", "~/m", run, "Linux")
    assert sums == {"config.json": "aa", "model.safetensors": "bb"}
    n = len(run.cmds)
    assert hr.model_hashes({}, "pc", "~/m", run, "Linux") == sums      # cached by stat
    assert len(run.cmds) == n + 1


def test_host_state_linux_skips_mac_tools():
    run = Fake({"nvidia-smi": (0, "NVIDIA RTX 2000 Ada, 8188 MiB, 300 MiB, 560.94\n")})
    info = hr.host_state({"name": "pc"}, False, "Linux", True, run, lambda *a: None)
    assert info["caffeinate"] == info["awdl"] == info["vpn_state"] == "n/a"
    assert "RTX 2000 Ada" in info["nvidia_smi"]
    assert not any("caffeinate" in c or "awdl0" in c or "scutil" in c for c in run.cmds)


def test_host_state_linux_without_cuda_does_not_call_nvidia_smi():
    run = Fake({})
    info = hr.host_state({"name": "pc"}, True, "Linux", False, run, lambda *a: None)
    assert "nvidia_smi" not in info and run.cmds == []


def test_host_state_darwin_remote():
    run = Fake({"pgrep": (0, "123\n"), "ifconfig awdl0": (0, "awdl0: flags=8943<UP,BROADCAST>\n"), "scutil": (0, "No service\n")})
    info = hr.host_state({"name": "air"}, False, "Darwin", False, run, lambda *a: None)
    assert info["caffeinate"] == "on" and info["awdl"] == "up" and info["vpn_state"] == "disconnected"


def test_find_run_falls_back_to_session6(tmp_path, monkeypatch):
    s6 = tmp_path / "s6"
    (s6 / "runs" / "R0_x").mkdir(parents=True)
    monkeypatch.setattr(hr, "SESSION6", s6)
    monkeypatch.setattr(hr, "RUNS", tmp_path / "s7" / "runs")
    assert hr.find_run("R0_x") == s6 / "runs" / "R0_x"
    (tmp_path / "s7" / "runs" / "R0_x").mkdir(parents=True)
    assert hr.find_run("R0_x") == tmp_path / "s7" / "runs" / "R0_x"
    with pytest.raises(FileNotFoundError):
        hr.find_run("nope")


def test_gguf_hash_and_stat_commands_quote_file_names(tmp_path):
    """Codex finding 1: the weight file name is a literal for the shell, a directory glob stays a glob."""
    import subprocess
    from nocturne.engines.torch_tp import hybrid_run as H
    f = tmp_path / "model; printf CODEX_REPRO; #.gguf"
    f.write_bytes(b"x")
    g = tmp_path / "sp ace.gguf"
    g.write_bytes(b"y")
    for name in (f, g):
        r = subprocess.run(["bash", "-c", H.hash_cmd("Linux", str(name)).replace("sha256sum", "shasum -a 256")], capture_output=True, text=True)
        assert r.returncode == 0 and "CODEX_REPRO" not in r.stdout.replace(str(name.name), ""), r.stderr
        assert name.name in r.stdout
    assert H.hash_cmd("Linux", "~/models/hf/Qwen").endswith("*.json *.safetensors")
    assert H.hash_cmd("Linux", "~/models/hf/Qwen").startswith("cd ~/models/hf/Qwen")
