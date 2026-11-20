#!/usr/bin/env python3
"""Coordinator side of scripts/15_join: print the invite commands, receive reports, apply one to ssh config and peers.yaml.
Usage: python3 scripts/16_join_apply.py invite | listen [--once] | apply <report.txt> --peer <name> [--dry-run] [--force]   (docs/JOIN.md)
"""
import argparse
import datetime
import difflib
import http.server
import ipaddress
import pathlib
import re
import shutil
import subprocess
import sys

HOME = pathlib.Path.home()
REPO = pathlib.Path(__file__).resolve().parents[1]
SSH_CONFIG = HOME / ".ssh" / "config"
PEERS_FILES = [HOME / ".nocturne" / "peers.yaml", REPO / "nocturne" / "bench" / "peers.yaml"]
INBOX = HOME / ".nocturne" / "join"
RAW = "https://raw.githubusercontent.com/OhadRose20596/nocturne/main/scripts"
PORT = 50079


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def lan_ip():
    rc, out = sh("netstat -rn -f inet | awk '$1==\"default\" && $NF !~ /^(utun|ppp|ipsec)/ {print $NF; exit}'")
    rc, ip = sh("ipconfig getifaddr %s" % out) if out else (1, "")
    if rc != 0 or not ip:
        sys.exit("no LAN address on this machine")
    return ip


def pubkey():
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        p = HOME / ".ssh" / name
        if p.exists():
            return p.read_text().strip()
    sys.exit("no public key in ~/.ssh. Create one with: ssh-keygen -t ed25519")


def parse_report(path):
    facts, checks = {}, []
    for line in pathlib.Path(path).read_text(errors="replace").replace("\ufeff", "").splitlines():
        line = line.rstrip("\r")
        if line.startswith("fact ") and "=" in line:
            k, v = line[5:].split("=", 1)
            facts[k] = v
        elif line.startswith("check "):
            parts = line.split(" ", 3)
            if len(parts) == 4:
                checks.append(tuple(parts[1:]))
        elif line.startswith("--- "):
            break
    return facts, checks


def cmd_invite(_):
    ip, key = lan_ip(), pubkey()
    print("Start the receiver here first (it waits for the reports):\n")
    print("  python3 scripts/16_join_apply.py listen\n")
    print("macOS or Linux, in a terminal on the new machine:\n")
    print("  curl -fsSL %s/15_join.sh -o /tmp/15_join.sh && bash /tmp/15_join.sh --install --coordinator %s --key '%s'\n" % (RAW, ip, key))
    print("Windows with WSL2, in PowerShell on the new machine:\n")
    print("  iwr -UseBasicParsing %s/15_join.ps1 -OutFile $env:TEMP\\15_join.ps1; powershell -ExecutionPolicy Bypass -File $env:TEMP\\15_join.ps1 -Install -Coordinator %s -Key '%s'\n" % (RAW, ip, key))
    print("Drop --install / -Install on a machine that is already set up. The receiver listens on %s:%d (LAN only, no auth, ADR-009)." % (ip, PORT))


def cmd_listen(args):
    ip = lan_ip()
    INBOX.mkdir(parents=True, exist_ok=True)
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        timeout = 15

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if self.path != "/join" or n <= 0 or n > 2_000_000:
                self.send_response(400); self.end_headers(); return
            body = self.rfile.read(n).decode("utf-8", "replace")
            m = re.search(r"^fact host=(.*)$", body, re.M)
            host = re.sub(r"[^A-Za-z0-9_.-]", "_", m.group(1))[:40] if m else "unknown"
            out = INBOX / ("%s_%s.txt" % (host, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            out.write_text(body)
            self.send_response(200); self.end_headers(); self.wfile.write(b"saved\n")
            print("\nreport from %s (%s) saved: %s" % (host, self.client_address[0], out))
            summarize(out)
            print("next: python3 scripts/16_join_apply.py apply '%s' --peer <name>" % out)
            got.append(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer((ip, PORT), H)
    print("waiting for join reports on http://%s:%d/join  (Ctrl-C to stop)" % (ip, PORT))
    try:
        while not (args.once and got):
            srv.handle_request()
    except KeyboardInterrupt:
        pass
    srv.server_close()


def summarize(path):
    facts, checks = parse_report(path)
    print("  host %s  os %s%s  ip %s  device %s  torch %s  sha %s" % (
        facts.get("host"), facts.get("os"), " (WSL2)" if facts.get("wsl") == "1" else "",
        facts.get("ip"), facts.get("device"), facts.get("torch"), facts.get("sha")))
    for name, status, msg in checks:
        if status in ("FAIL", "FIXED"):
            print("  [%s] %s: %s" % (status, name, msg))
    return facts, checks


def is_comment(line):
    return line.lstrip().startswith("#")


def trim(lines, start, end):
    while end > start + 1 and (not lines[end - 1].strip() or is_comment(lines[end - 1])):
        end -= 1
    return end


def peer_block(lines, name):
    start = next((i for i, l in enumerate(lines) if re.match(r"^  %s:\s*(#.*)?$" % re.escape(name), l)), None)
    if start is None:
        return None, None
    end = start + 1
    while end < len(lines) and (lines[end].startswith("    ") or not lines[end].strip() or is_comment(lines[end])):
        end += 1
    return start, trim(lines, start, end)


def peers_end(lines):
    start = next((i for i, l in enumerate(lines) if re.match(r"^peers:\s*(#.*)?$", l)), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip() or is_comment(lines[end])):
        end += 1
    return trim(lines, start, end)


def set_key(lines, start, end, key, value):
    for i in range(start + 1, end):
        if re.match(r"^    %s:(\s|$)" % re.escape(key), lines[i]):
            lines[i] = "    %s: %s" % (key, value)
            return
    lines.insert(start + 1, "    %s: %s" % (key, value))


def get_key(lines, start, end, key):
    for i in range(start + 1, end):
        m = re.match(r"^    %s:\s*([^#]*)" % re.escape(key), lines[i])
        if m:
            return m.group(1).strip().strip('"')
    return None


def update_peers(text, peer, f, ssh_target, main=True):
    """main=True: the coordinator's own file, peers may be added. main=False: only addresses of an existing peer."""
    lines = text.splitlines()
    start, end = peer_block(lines, peer)
    ip = f["ip"]
    if start is None and not main:
        return text
    if start is None:
        at = peers_end(lines)
        if at is None:
            lines += ["peers:"]
            at = len(lines)
        block = [
            "  %s:" % peer,
            "    host: %s" % f["host"],
            "    ssh: %s" % ssh_target,
            "    os: %s" % f["os"],
            "    backend: %s" % {"mps": "metal"}.get(f.get("device", ""), f.get("device", "cpu")),
            "    mem_gb: %s" % f.get("mem_gb", "0"),
            '    python: "%s"' % f.get("python", ""),
            "    bind: %s" % ip,
            "    networks: {wifi: %s}" % ip,
        ]
        lines[at:at] = block
        return "\n".join(lines) + "\n"
    is_self = get_key(lines, start, end, "ssh") in ("null", None, "")
    set_key(lines, start, end, "bind", ip)
    start, end = peer_block(lines, peer)
    for i in range(start + 1, end):
        m = re.match(r"^(    networks:\s*)\{(.*)\}(.*)$", lines[i])
        if m:
            items = [x.strip() for x in m.group(2).split(",") if x.strip() and not x.strip().startswith("wifi:")]
            lines[i] = "%s{%s}%s" % (m.group(1), ", ".join(["wifi: %s" % ip] + items), m.group(3))
            break
        if re.match(r"^    networks:\s*(#.*)?$", lines[i]):
            sys.exit("peer '%s' has a block-style 'networks:' mapping. Set wifi: %s there by hand, or rewrite it as {wifi: ...}" % (peer, ip))
    else:
        set_key(lines, start, end, "networks", "{wifi: %s}" % ip)
        start, end = peer_block(lines, peer)
    if not is_self and (main or "@" in (get_key(lines, start, end, "ssh") or "")):
        set_key(lines, start, end, "ssh", ssh_target)
        start, end = peer_block(lines, peer)
    if main and f.get("python") and get_key(lines, start, end, "python") not in (None, f["python"]):
        set_key(lines, start, end, "python", '"%s"' % f["python"])
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def update_ssh_config(text, alias, f):
    lines = text.splitlines()
    start = None
    for i, l in enumerate(lines):
        m = re.match(r"^\s*Host\s+(.+)$", l, re.I)
        if m and alias in m.group(1).split():
            start = i
            break
    want = [("HostName", f["ip"]), ("Port", f.get("ssh_port", "22")), ("User", f["user"])]
    if start is None:
        block = ["# nocturne peer, written by scripts/16_join_apply.py", "Host %s" % alias]
        block += ["  %s %s" % kv for kv in want] + ["  ServerAliveInterval 30", ""]
        at = next((i for i, l in enumerate(lines) if re.match(r"^\s*(Host\s+\*\s*$|Match\s)", l, re.I)), None)
        if at is None:
            lines += [""] + block[:-1]
        else:
            lines[at:at] = block
        return "\n".join(lines) + "\n"
    end = start + 1
    while end < len(lines) and not re.match(r"^\s*(Host|Match)\s", lines[end], re.I):
        end += 1
    for key, val in want:
        for i in range(start + 1, end):
            if re.match(r"^\s*%s\s" % key, lines[i], re.I):
                lines[i] = "  %s %s" % (key, val)
                break
        else:
            lines.insert(start + 1, "  %s %s" % (key, val)); end += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


BLOCKING = ("network", "wsl_mirror", "ssh_server", "win_network")


def validate(facts, checks, force):
    for need in ("ip", "host", "user", "os"):
        if not facts.get(need):
            sys.exit("report has no '%s' fact, cannot apply" % need)
    try:
        ip = ipaddress.ip_address(facts["ip"])
    except ValueError:
        sys.exit("report ip is not an address: %r" % facts["ip"])
    for k in ("host", "user"):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", facts[k]):
            sys.exit("report %s has unexpected characters: %r" % (k, facts[k]))
    if not re.fullmatch(r"\d{1,5}", facts.get("ssh_port", "22")):
        sys.exit("report ssh_port is not a number: %r" % facts.get("ssh_port"))
    if facts.get("python") and not re.fullmatch(r"[A-Za-z0-9._/:~ -]+", facts["python"]):
        sys.exit("report python path has unexpected characters: %r" % facts["python"])
    problems = ["%s: %s" % (n, m) for n, st, m in checks if st == "FAIL" and n in BLOCKING]
    if facts.get("wsl") == "1" and facts.get("win_ip") and facts["win_ip"] != facts["ip"]:
        problems.append("WSL address %s differs from the Windows address %s (mirrored networking is off)" % (facts["ip"], facts["win_ip"]))
    if ip.is_loopback or ip.is_link_local:
        problems.append("address %s is not a LAN address" % ip)
    if re.match(r"^172\.20\.10\.", facts["ip"]):
        print("warning: %s is a phone hotspot address" % facts["ip"])
    if problems and not force:
        sys.exit("not applied, the report has blocking failures:\n  " + "\n  ".join(problems) + "\nfix them on the peer and run the join script again, or pass --force")


def write_with_diff(path, new, stamp, dry):
    old = path.read_text() if path.exists() else ""
    if old == new:
        print("unchanged: %s" % path); return
    print("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(path), str(path) + " (new)", n=1)))
    if dry:
        return
    if path.exists():
        bdir = INBOX / "backups"
        bdir.mkdir(parents=True, exist_ok=True)
        bak = bdir / ("%s_%s.bak_%s" % (path.parent.name, path.name, stamp))
        shutil.copy2(path, bak)
        print("backup: %s" % bak)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new)


def cmd_apply(args):
    facts, checks = summarize(args.report)
    validate(facts, checks, args.force)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.peer):
        sys.exit("--peer must be a simple name")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_peers = PEERS_FILES[0]
    if not main_peers.exists():
        sys.exit("%s does not exist. Create it first (template: nocturne/bench/peers.example.yaml or docs), then run apply again" % main_peers)
    lines = main_peers.read_text().splitlines()
    start, end = peer_block(lines, args.peer)
    current = get_key(lines, start, end, "ssh") if start is not None else None
    is_self = start is not None and current in ("null", None, "")
    use_alias = (not is_self) and (current is None or "@" not in current)
    alias = current if (current and use_alias) else args.peer
    ssh_target = alias if use_alias else "%s@%s" % (facts["user"], facts["ip"])
    if use_alias:
        write_with_diff(SSH_CONFIG, update_ssh_config(SSH_CONFIG.read_text() if SSH_CONFIG.exists() else "", alias, facts), stamp, args.dry_run)
    for pf in PEERS_FILES:
        if pf.exists():
            write_with_diff(pf, update_peers(pf.read_text(), args.peer, facts, ssh_target, main=(pf == main_peers)), stamp, args.dry_run)
    if args.dry_run or is_self:
        return
    print("\nverify from this machine:")
    rc, out = sh(["ping", "-c", "10", "-i", "0.3", facts["ip"]], 30)
    print("  ping: " + " | ".join(out.splitlines()[-2:]))
    rc, out = sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
                  ssh_target, "hostname -s; ls ~/models/hf 2>/dev/null; true"], 30)
    if rc != 0:
        print("  ssh %s: FAILED\n  %s" % (ssh_target, out.replace("\n", "\n  ")))
        sys.exit(1)
    got = out.splitlines()
    print("  ssh %s: ok (%s)" % (ssh_target, got[0] if got else "?"))
    mine = sorted(p.name for p in (HOME / "models" / "hf").glob("*") if p.is_dir())
    missing = [m for m in mine if m not in got[1:]]
    if missing:
        print("\nmodels on this machine that the peer does not have. Copy the ones you need:")
        print("  ssh %s 'mkdir -p ~/models/hf'" % ssh_target)
        for m in missing:
            print("  scp -r ~/models/hf/%s %s:~/models/hf/" % (m, ssh_target))
    fails = [c for c in checks if c[1] == "FAIL"]
    print("\npeer '%s' applied. %d failed check(s) in its report." % (args.peer, len(fails)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("invite").set_defaults(fn=cmd_invite)
    p = sub.add_parser("listen"); p.add_argument("--once", action="store_true"); p.set_defaults(fn=cmd_listen)
    p = sub.add_parser("apply"); p.add_argument("report"); p.add_argument("--peer", required=True)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
