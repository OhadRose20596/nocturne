"""Join tool additions (engine GGUF, B6): network types by link, report facts, GGUF model listing and copy commands,
Linux lan_ip parsing. scripts/16_join_apply.py has a digit first, so it is loaded by path."""
import argparse
import importlib.util
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("join_apply", REPO / "scripts" / "16_join_apply.py")
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)

PEERS = """models:
  m: {path: x}
peers:
  m5:
    host: m5
    ssh: null
    bind: 192.168.1.242
    networks: {wifi: 192.168.1.242, wired: 10.0.0.2, tailscale: 100.64.0.2}
  air:
    host: air
    ssh: air
    bind: 192.168.1.219
    networks: {wifi: 192.168.1.219, hotspot: 172.20.10.2}
  bare:
    host: bare
    ssh: bare
    networks: {}
"""


def facts(ip, link, host="air"):
    return {"ip": ip, "host": host, "user": "u", "os": "macos", "device": "mps", "mem_gb": "16", "python": "/p", "link_type": link}


def networks(text, peer):
    lines = text.splitlines()
    s, e = ja.peer_block(lines, peer)
    return ja.get_key(lines, s, e, "networks")


def test_wired_report_adds_wired_and_keeps_wifi_and_hotspot():
    out = ja.update_peers(PEERS, "air", facts("10.0.0.1", "wired"), "air")
    n = networks(out, "air")
    assert "wifi: 192.168.1.219" in n and "wired: 10.0.0.1" in n and "hotspot: 172.20.10.2" in n


def test_wifi_report_replaces_wifi_in_place_and_keeps_wired():
    once = ja.update_peers(PEERS, "m5", facts("192.168.1.50", "wifi", "m5"), "m5")
    n = networks(once, "m5")
    assert n.startswith("{wifi: 192.168.1.50, wired: 10.0.0.2, tailscale: 100.64.0.2}")
    twice = ja.update_peers(once, "m5", facts("10.0.0.9", "wired", "m5"), "m5")
    n = networks(twice, "m5")
    assert "wifi: 192.168.1.50" in n and "wired: 10.0.0.9" in n and "tailscale: 100.64.0.2" in n


def test_one_type_never_overwrites_the_other():
    out = ja.update_peers(PEERS, "air", facts("10.0.0.1", "wired"), "air")
    out = ja.update_peers(out, "air", facts("192.168.1.220", "wifi"), "air")
    n = networks(out, "air")
    assert "wifi: 192.168.1.220" in n and "wired: 10.0.0.1" in n


def test_unknown_link_leaves_networks_alone(capsys):
    out = ja.update_peers(PEERS, "air", facts("192.168.9.9", "unknown"), "air")
    assert networks(out, "air") == networks(PEERS, "air")
    assert "does not say" in capsys.readouterr().out


def test_unknown_link_on_empty_networks_falls_back_to_wifi():
    out = ja.update_peers(PEERS, "bare", facts("192.168.9.9", "unknown", "bare"), "bare")
    assert networks(out, "bare") == "{wifi: 192.168.9.9}"


def test_new_peer_gets_its_link_type():
    out = ja.update_peers(PEERS, "newbox", facts("10.0.0.7", "wired", "newbox"), "newbox")
    assert "networks: {wired: 10.0.0.7}" in out


def test_bind_still_set_as_before():
    out = ja.update_peers(PEERS, "air", facts("10.0.0.1", "wired"), "air")
    lines = out.splitlines(); s, e = ja.peer_block(lines, "air")
    assert ja.get_key(lines, s, e, "bind") == "10.0.0.1"


REPORT = """# nocturne join report
fact report_version=1
fact host=box
fact user=u
fact os=linux
fact ip=192.168.1.5
fact link_type=wired
fact llama_commit=1e7bcf3da
fact llama_backend=cuda
fact ggml_rpc_server=/x/ggml-rpc-server
fact gguf_models=a.gguf:10,gguf/b.gguf:20

check llama_cpp PASS commit 1e7bcf3da (same as the pinned 1e7bcf3), backend cuda
check link PASS wired on eth0
check models PASS hf: none; gguf: 2 file(s)

--- uname
$ uname -a
"""


def test_report_parsing_with_new_fact_lines(tmp_path, capsys):
    p = tmp_path / "r.txt"; p.write_text(REPORT)
    f, c = ja.parse_report(p)
    assert f["llama_backend"] == "cuda" and f["link_type"] == "wired" and f["gguf_models"] == "a.gguf:10,gguf/b.gguf:20"
    assert ("llama_cpp", "PASS", "commit 1e7bcf3da (same as the pinned 1e7bcf3), backend cuda") in c
    ja.summarize(p)
    out = capsys.readouterr().out
    assert "wired" in out and "llama.cpp 1e7bcf3da, backend cuda" in out


def test_real_m5_report_has_the_new_lines():
    reports = sorted(pathlib.Path.home().glob("nocturne_join_*.txt"))
    if not reports:
        pytest.skip("no report; run: bash scripts/15_join.sh --check --self")
    f, c = ja.parse_report(reports[-1])
    names = {n for n, _, _ in c}
    assert {"llama_cpp", "link", "models"} <= names
    assert f["link_type"] in ("wifi", "wired", "unknown") and "llama_commit" in f and "gguf_models" in f


def test_linux_lan_ip_parsing():
    routes = "default via 10.8.0.1 dev tun0 proto static\ndefault via 192.168.1.1 dev eth0 proto dhcp metric 100\n"
    assert ja.linux_default_dev(routes) == "eth0"
    assert ja.linux_default_dev("") == ""
    addr = "2: eth0    inet 192.168.1.5/24 brd 192.168.1.255 scope global dynamic eth0\\       valid_lft 85000sec\n"
    assert ja.linux_addr(addr) == "192.168.1.5"
    assert ja.linux_addr("") == ""


def test_remote_listing_and_missing_models():
    out = "airhost\nQwen2.5-0.5B-Instruct\n@@gguf\nq.gguf 100\ngguf/sub dir.gguf 5\n"
    host, hf, gg = ja.parse_remote_listing(out)
    assert host == "airhost" and hf == ["Qwen2.5-0.5B-Instruct"] and gg == {"q.gguf": 100, "gguf/sub dir.gguf": 5}
    mine_gg = {"q.gguf": 100, "new.gguf": 7, "gguf/sub dir.gguf": 6}
    miss_hf, miss_gg = ja.models_missing(["Qwen2.5-0.5B-Instruct", "Other"], mine_gg, hf, gg)
    assert miss_hf == ["Other"] and miss_gg == ["new.gguf", "gguf/sub dir.gguf"]     # new, and one with another size


def test_gguf_commands_carry_sha256_and_dirs():
    lines = ja.gguf_commands("air", ["a.gguf", "gguf/b.gguf"], {"a.gguf": 2_000_000_000, "gguf/b.gguf": 1}, {"a.gguf": "abc"})
    text = "\n".join(lines)
    assert "mkdir -p ~/models ~/models/gguf" in text
    assert "scp ~/models/a.gguf air:~/models/ " in text and "scp ~/models/gguf/b.gguf air:~/models/gguf/ " in text
    assert "expect abc" in text and "(not computed)" in text


def test_file_sha256_cached(tmp_path):
    f = tmp_path / "m.gguf"; f.write_bytes(b"hello" * 1000)
    cache = tmp_path / "c.json"
    h = ja.file_sha256(f, cache)
    import hashlib
    assert h == hashlib.sha256(b"hello" * 1000).hexdigest() and cache.exists()
    cache.write_text(cache.read_text().replace(h, "cached"))
    assert ja.file_sha256(f, cache) == "cached"                 # served from the cache while size and mtime match


def test_local_models_lists_both_layouts(tmp_path):
    (tmp_path / "hf" / "M1").mkdir(parents=True); (tmp_path / "gguf").mkdir()
    (tmp_path / "a.gguf").write_bytes(b"1"); (tmp_path / "gguf" / "b.gguf").write_bytes(b"22")
    hf, gg = ja.local_models(tmp_path)
    assert hf == ["M1"] and gg == {"a.gguf": 1, "gguf/b.gguf": 2}


def test_apply_dry_run_on_copies_of_the_real_files(tmp_path, monkeypatch, capsys):
    home = pathlib.Path.home()
    real = [home / ".nocturne" / "peers.yaml", REPO / "nocturne" / "bench" / "peers.yaml"]
    reports = sorted(home.glob("nocturne_join_*.txt"))
    if not all(p.exists() for p in real) or not reports:
        pytest.skip("real peers.yaml files or a join report are missing")
    copies = []
    for i, p in enumerate(real):
        d = tmp_path / str(i); d.mkdir(); shutil.copy(p, d / p.name); copies.append(d / p.name)
    ssh = tmp_path / "ssh_config"; ssh.write_text((home / ".ssh" / "config").read_text() if (home / ".ssh" / "config").exists() else "")
    rep = tmp_path / "report.txt"; shutil.copy(reports[-1], rep)
    monkeypatch.setattr(ja, "PEERS_FILES", copies)
    monkeypatch.setattr(ja, "SSH_CONFIG", ssh)
    before = [c.read_text() for c in copies]
    args = argparse.Namespace(report=str(rep), peer="m5", dry_run=True, force=True, no_hash=True)
    ja.cmd_apply(args)
    assert [c.read_text() for c in copies] == before, "dry run must not write"
    capsys.readouterr()
    # the same, for real, on the copies: the report's own type is set, every other type survives
    monkeypatch.setattr(ja, "INBOX", tmp_path / "inbox")
    f, _ = ja.parse_report(rep)
    f["link_type"], f["ip"] = "wired", "10.0.0.2"
    text = rep.read_text().replace("fact link_type=" + reports_link(rep), "fact link_type=wired")
    rep.write_text(text.replace("fact ip=" + ja.parse_report(rep)[0]["ip"], "fact ip=10.0.0.2"))
    ja.cmd_apply(argparse.Namespace(report=str(rep), peer="m5", dry_run=False, force=True, no_hash=True))
    for c in copies:
        n = networks(c.read_text(), "m5")
        assert "wired: 10.0.0.2" in n and "tailscale: 100.64.0.2" in n and "hotspot: 172.20.10.3" in n and "wifi: 192.168.1.242" in n, n


def reports_link(path):
    return ja.parse_report(path)[0]["link_type"]


def test_shell_scripts_parse():
    for name in ("15_join.sh",):
        r = subprocess.run(["bash", "-n", str(REPO / "scripts" / name)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
