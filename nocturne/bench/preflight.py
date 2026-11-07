"""Session 5 preflight: fails loudly (exit 1) when a precondition for a memory edge run is not met.
Usage: python -m nocturne.bench.preflight --model qwen2.5-32b-q4km [--min-free-gb 30]"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys

from . import config
from .probes import net

SHA = "1e7bcf3"


def sh(peer: dict, cmd: str, timeout: int = 900) -> tuple[int, str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", peer["ssh"], cmd] if peer.get("ssh") else ["bash", "-c", cmd]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-free-gb", type=float, default=30)
    ap.add_argument("--network", default="wifi")
    ap.add_argument("--peers", type=Path, default=Path("~/.nocturne/peers.yaml").expanduser())
    a = ap.parse_args(argv)
    cfg = config.load_peers(a.peers)
    m = cfg["models"][a.model]
    names = [cfg["roles"]["requester"]] + list(cfg["roles"]["workers"])
    bad, shas = [], {}
    for n in names:
        p = cfg["peers"][n]
        rc, out = sh(p, f"{p['bin']}/llama-server --version 2>&1 | grep commit")
        if SHA not in out:
            bad.append(f"{n}: llama.cpp SHA is not {SHA} ({out[-120:]})")
        rc, out = sh(p, f"test -x {p['bin']}/llama-server && test -x {p['bin']}/ggml-rpc-server && echo ok")
        if "ok" not in out:
            bad.append(f"{n}: binaries missing in {p['bin']}")
        rc, out = sh(p, f"shasum -a 256 ~/models/{os.path.basename(m['path'])} | cut -d' ' -f1")
        shas[n] = out.split()[-1] if rc == 0 and out else None
        rc, out = sh(p, "df -g ~ | tail -1 | awk '{print $4}'")
        free = float(out) if rc == 0 and out.replace('.', '').isdigit() else None
        print(f"[{n}] model sha {str(shas[n])[:12]}, free {free} GB")
        if free is None or free < a.min_free_gb:
            bad.append(f"{n}: free disk {free} GB under {a.min_free_gb}")
    if len(set(shas.values())) != 1 or None in shas.values():
        bad.append(f"model sha256 differs or missing: {shas}")
    vpn, awdl = net.vpn_state(), net.awdl_state()
    print(f"vpn_state {vpn}, awdl {awdl}")
    if vpn != "disconnected":
        bad.append(f"M5 vpn_state is {vpn}, must be disconnected")
    w = cfg["peers"][cfg["roles"]["workers"][0]]
    ip = (w.get("networks") or {}).get(a.network) or w["bind"]
    s = socket.socket()
    s.settimeout(3)
    up = s.connect_ex((ip, cfg["rpc_port"])) == 0
    s.close()
    print(f"worker {ip}:{cfg['rpc_port']} {'open' if up else 'closed (the bench starts it over ssh)'}")
    for b in bad:
        print("PREFLIGHT FAIL:", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
