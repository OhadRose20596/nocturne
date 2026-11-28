"""Launcher for a hybrid placement: plan, start every rank process (locally, or over ssh for another host), run the
coordinator in this process, stop every process at the end.
Addresses (CLAUDE.md rule 7): a rank listens on 127.0.0.1 when a peer on its own host connects to it, and on its
host's `bind` address from peers.yaml when a peer on another host does. Never 0.0.0.0. The ports accept any peer
without authentication: security is deferred by ADR-009."""
import os
import re
import socket
import tempfile
import threading
import time
from pathlib import Path

import yaml

from ...bench.runner import Runner
from .config import load_config
from .hybrid import Coordinator
from .gguf_source import weight_meta
from .plan import Dims, Placement, coordinator_host, listen_addrs, placement_from_dict, plan

PEERS = "~/.nocturne/peers.yaml"


def load_peers(path: str = PEERS) -> dict:
    return yaml.safe_load(Path(os.path.expanduser(path)).read_text())


def model_dir_of(model: str, peers: dict, host: str | None = None) -> str:
    """The weights of a model key of peers.yaml, or a path: a safetensors directory or a .gguf file. `torch_model` wins;
    else `path` when it is a .gguf (the llama.cpp file works for the torch engine too). Unexpanded (~ stays for ssh).
    A host may hold the model elsewhere: peers.<host>.models.<key> (a path) wins over the shared entry."""
    own = ((peers.get("peers") or {}).get(host) or {}).get("models") or {}
    if own.get(model):
        return own[model]
    m = (peers.get("models") or {}).get(model)
    if m:
        if m.get("torch_model"):
            return m["torch_model"]
        if str(m.get("path", "")).endswith(".gguf"):
            return m["path"]
        raise ValueError(f"model {model} has no torch_model and no .gguf path in peers.yaml")
    return model


def host_addrs(peers: dict, names) -> dict[str, str]:
    out = {}
    for n in names:
        p = (peers.get("peers") or {}).get(n)
        if not p or not p.get("bind"):
            raise ValueError(f"host {n} has no peers.yaml entry with a bind address")
        if p["bind"].startswith("0.0.0.0"):
            raise ValueError(f"host {n} binds 0.0.0.0; refuse (CLAUDE.md rule 7)")
        out[n] = p["bind"]
    return out


class Launch:
    def __init__(self, placement_path: str, peers_path: str = PEERS, log_dir: str | None = None, force: bool = False,
                 mps_fraction_cap: bool = False, reduce_dtype: str | None = None, tag: str | None = None, log=print):
        self.log = log
        self.placement_path = placement_path
        self.yaml = yaml.safe_load(Path(placement_path).read_text())
        self.placement: Placement = placement_from_dict(self.yaml)
        if reduce_dtype:
            self.placement = Placement(**{**self.placement.__dict__, "reduce_dtype": reduce_dtype})
        self.peers = load_peers(peers_path)
        self.model_dir = model_dir_of(self.placement.model, self.peers, coordinator_host(self.placement))
        self.local_model_dir = os.path.expanduser(self.model_dir)
        self.cfg = load_config(self.local_model_dir)
        self.dims = Dims.of(self.cfg, weight_meta(self.local_model_dir))
        self.force = force
        self.specs = plan(self.dims, self.placement, check_budget=not force)
        self.hosts = host_addrs(self.peers, sorted({s.host for s in self.specs}))
        self.listen = listen_addrs(self.specs, self.hosts)
        self.mps_fraction_cap = mps_fraction_cap
        self.tag = tag or f"hyb{int(time.time())}"
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="hybrid_"))
        self.runner = Runner(echo=False)
        self.procs = []
        self.coordinator: Coordinator | None = None
        coord = self.specs[0].host
        me = socket.gethostname().split(".")[0]
        if self.peers["peers"][coord].get("host", coord).split(".")[0] != me:
            raise RuntimeError(f"the coordinator host is {coord} ({self.peers['peers'][coord].get('host')}); this machine is {me}")

    def _peer(self, host: str) -> dict:
        return dict(self.peers["peers"][host], name=host)

    @staticmethod
    def _stamp(pipe, path: Path) -> None:
        with path.open("a") as f:
            for line in pipe:
                f.write(f"{time.strftime('%H:%M:%S')} {line}")
                f.flush()

    def start(self, ready_timeout_s: float = 1800.0) -> Coordinator:
        import subprocess
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logs = {}
        for s in self.specs[1:]:
            peer = self._peer(s.host)
            argv = [peer["python"], "-u", "-m", "nocturne.engines.torch_tp", "rank", "--listen", ",".join(self.listen[s.gid]),
                    "--port", str(s.port), "--model", model_dir_of(self.placement.model, self.peers, s.host), "--device", s.device,
                    "--name", s.name, "--tag", self.tag]
            p = self.runner.spawn(peer, argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.procs.append((s, p))
            path = self.log_dir / f"rank_{s.name.replace('/', '_')}.log"
            logs[s.name] = path
            threading.Thread(target=self._stamp, args=(p.stdout, path), daemon=True).start()
        t0 = time.time()
        for s, p in self.procs:       # every rank prints "listening on" once its sockets are open
            while time.time() - t0 < 180:
                if p.poll() is not None:
                    raise RuntimeError(f"rank {s.name} exited rc={p.returncode} at start; see {logs[s.name]}")
                if logs[s.name].exists() and "listening on" in logs[s.name].read_text(errors="replace"):
                    break
                time.sleep(0.3)
        me = self.specs[0]
        self.coordinator = Coordinator(self.specs, self.local_model_dir, self.hosts, self.listen[0], self.placement.max_seq_len,
                                       self.placement.reduce_dtype, self.placement.prefill_max, device=me.device, log=self.log,
                                       enforce_budget=True, mps_fraction_cap=self.mps_fraction_cap, ready_timeout_s=ready_timeout_s)
        return self.coordinator

    def commands(self) -> list[dict]:
        return self.runner.as_records()

    def stop(self) -> None:
        if self.coordinator:
            self.coordinator.shutdown()
        t0 = time.time()
        for s, p in self.procs:
            while p.poll() is None and time.time() - t0 < 15:
                time.sleep(0.2)
        self.runner.stop_all()
        for h in sorted({s.host for s, _ in self.procs}):
            self.runner.run(self._peer(h), ["pkill", "-f", "--", f"--tag {re.escape(self.tag)}( |$)"], timeout=30)
