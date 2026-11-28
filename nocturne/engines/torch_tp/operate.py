"""The operator of a run (session 10, part B): the machine that holds peers.yaml and has ssh to every host. It is not a
role in the engine. It starts one rank process per rank on its host, starts the coordinator process on the host the
placement names (coordinate.py), streams its log, and collects every host's run files. It never loads the model.
A host is reached over ssh when its peers.yaml entry has `ssh`, else with a local shell. Each host writes its run files
under ~/.nocturne/runs/<tag> (or peers.<host>.run_root/<tag>); a host without ssh writes straight into the run folder.
Ports accept any peer: security is deferred by ADR-009. Listen addresses come from peers.yaml, never 0.0.0.0."""
import io
import json
import re
import shlex
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import yaml

from ...bench.runner import Runner
from .coordinate import PLAN_MARK
from .launch import PEERS, host_addrs, load_peers, model_dir_of
from .plan import coordinator_host, listen_addrs, placement_from_dict, slots

TAG_OK = re.compile(r"^[A-Za-z0-9._-]+$")
BACKEND_DEVICE = {"metal": "mps", "mps": "mps", "cuda": "cuda", "cpu": "cpu"}


def fill_devices(y: dict, peers: dict) -> dict:
    """A rank without `device` gets the device of its host's `backend` in peers.yaml (the planner's own default is mps,
    which is wrong on a CUDA host). Returns the same dict."""
    for st in y.get("stages") or []:
        for r in st.get("ranks") or []:
            if not r.get("device"):
                be = str(((peers.get("peers") or {}).get(r.get("host")) or {}).get("backend") or "").lower()
                if be not in BACKEND_DEVICE:
                    raise ValueError(f"rank on {r.get('host')} has no device, and peers.yaml gives the host no backend (metal, cuda, cpu)")
                r["device"] = BACKEND_DEVICE[be]
    return y


def shell_path(p: str) -> str:
    """Quote a path for a remote shell, keeping a leading ~/ expandable."""
    return "~/" + shlex.quote(p[2:]) if p.startswith("~/") else shlex.quote(p)


def shell_argv(peer: dict, cmd: str) -> list[str]:
    if peer.get("ssh"):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer["ssh"], cmd]
    return ["bash", "-lc", cmd]


def kill_argv(tag: str) -> list[str]:
    """pkill for every process of one run. `--` ends pkill's own options (the pattern starts with two dashes), and the
    pattern ends at a space or at the end, so tag X_r1 does not match the processes of X_r13."""
    return ["pkill", "-f", "--", f"--tag {re.escape(tag)}( |$)"]


def safe_extract(data: bytes, dest: Path) -> list[str]:
    """Unpack a tar stream of plain files into dest. Names with a path, links and devices are skipped."""
    out = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for m in tf.getmembers():
            name = m.name[2:] if m.name.startswith("./") else m.name
            if not m.isfile() or not name or "/" in name or name.startswith("."):
                continue
            (dest / name).write_bytes(tf.extractfile(m).read())
            out.append(name)
    return sorted(out)


class Operator:
    def __init__(self, placement_path: str, peers_path: str = PEERS, run_dir: str | Path = ".", tag: str | None = None, log=print,
                 ship_weights: bool = False, cache_dir: str | None = None, runner: Runner | None = None):
        self.log = log
        self.placement_path = placement_path
        self.peers = load_peers(peers_path)
        self.yaml = fill_devices(yaml.safe_load(Path(placement_path).read_text()), self.peers)
        self.placement = placement_from_dict(self.yaml)
        self.slots = slots(self.placement)
        self.coord = coordinator_host(self.placement)
        self.hosts = host_addrs(self.peers, sorted({s.host for s in self.slots}))
        self.listen = listen_addrs(self.slots, self.hosts)
        self.model_dir = model_dir_of(self.placement.model, self.peers, self.coord)
        self.tag = tag or f"hyb{int(time.time())}"
        if not TAG_OK.match(self.tag):
            raise ValueError(f"run tag {self.tag!r}: letters, digits, dot, underscore and dash only (it names folders on every host)")
        self.run_dir = Path(run_dir)
        self.ship_weights, self.cache_dir = ship_weights, cache_dir
        self.runner = runner or Runner(echo=False)
        self.procs: list = []
        self.coord_proc = None

    # ---- hosts ----------------------------------------------------------------------------------------------------------
    def peer(self, host: str) -> dict:
        return dict(self.peers["peers"][host], name=host)

    _peer = peer

    def python(self, host: str) -> str:
        return self.peer(host).get("python") or "python3"

    def host_dir(self, host: str) -> str:
        """Where a host writes its run files."""
        p = self.peer(host)
        if p.get("run_root"):
            return f"{str(p['run_root']).rstrip('/')}/{self.tag}"
        return f"~/.nocturne/runs/{self.tag}" if p.get("ssh") else str(self.run_dir)

    def collected_hosts(self) -> list[str]:
        return sorted({s.host for s in self.slots if self.host_dir(s.host) != str(self.run_dir)})

    def sh(self, host: str, cmd: str, timeout: float = 120, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        """One shell command on a host; stdout and stderr as bytes. Logged like every other command."""
        peer = self.peer(host)
        full = shell_argv(peer, cmd)
        self.runner._record(peer, full)
        if self.runner.dry_run:
            return subprocess.CompletedProcess(full, 0, b"", b"")
        return subprocess.run(full, input=stdin, capture_output=True, timeout=timeout)

    def hostname(self, host: str) -> str | None:
        """The name the host gives itself now (a Mac's changes with the network); the peers.yaml entry when it does not answer."""
        try:
            r = self.sh(host, "hostname", 20)
        except (OSError, subprocess.SubprocessError):
            return self.peer(host).get("host")
        name = r.stdout.decode(errors="replace").strip().splitlines()[-1:] if r.returncode == 0 and r.stdout else []
        return name[0] if name else self.peer(host).get("host")

    # ---- job and plan ---------------------------------------------------------------------------------------------------
    def job(self, prompts: list[dict], **opts) -> dict:
        return {"run_id": self.tag, "tag": self.tag, "coordinator": self.coord, "coordinator_hostname": self.hostname(self.coord),
                "placement": self.yaml, "model_dir": self.model_dir, "hosts": self.hosts, "prompts": prompts,
                "ship_weights": self.ship_weights, "cache_dir": self.cache_dir, **opts}

    def job_path(self) -> str:
        return f"{self.host_dir(self.coord)}/job.json"

    def push_job(self, job: dict) -> None:
        d = shell_path(self.host_dir(self.coord))
        r = self.sh(self.coord, f"mkdir -p {d} && cat > {d}/job.json", 60, json.dumps(job, indent=1).encode())
        if r.returncode:
            raise RuntimeError(f"writing the job file on {self.coord} failed: {r.stderr.decode(errors='replace')[-300:]}")

    def coordinate_argv(self, plan_only: bool = False) -> list[str]:
        return [self.python(self.coord), "-u", "-m", "nocturne.engines.torch_tp.coordinate", "--job", self.job_path(), "--tag", self.tag] + \
               (["--plan-only"] if plan_only else [])

    def plan(self) -> dict:
        """Ask the coordinator host for the plan (it has the model). {"ok": bool, "specs" | "error", ...}."""
        r = self.runner.run(self.peer(self.coord), self.coordinate_argv(plan_only=True), timeout=600)
        if self.runner.dry_run:
            return {"ok": True, "specs": [], "frames_predicted": {}}
        for line in reversed((r.stdout or "").splitlines()):
            if line.startswith(PLAN_MARK):
                return json.loads(line[len(PLAN_MARK):])
        return {"ok": False, "fatal": True, "error": f"no plan from {self.coord} (rc {r.returncode}): {(r.stderr or r.stdout or '')[-400:]}"}

    # ---- processes ------------------------------------------------------------------------------------------------------
    def rank_argv(self, s) -> list[str]:
        argv = [self.python(s.host), "-u", "-m", "nocturne.engines.torch_tp", "rank", "--listen", ",".join(self.listen[s.gid]),
                "--port", str(s.port), "--device", s.device, "--name", s.name, "--tag", self.tag, "--run-dir", self.host_dir(s.host)]
        if not self.ship_weights:
            argv += ["--model", model_dir_of(self.placement.model, self.peers, s.host)]
        if self.cache_dir:
            argv += ["--cache-dir", self.cache_dir]
        return argv

    @staticmethod
    def _stamp(pipe, path: Path, echo=None) -> None:
        with path.open("a") as f:
            for line in pipe:
                f.write(f"{time.strftime('%H:%M:%S')} {line}")
                f.flush()
                if echo:
                    echo(line.rstrip("\n"))

    def rank_log(self, s) -> Path:
        return self.run_dir / f"rank_{s.name.replace('/', '_')}.log"

    def start_ranks(self, wait_s: float = 180.0) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for s in self.slots[1:]:
            p = self.runner.spawn(self.peer(s.host), self.rank_argv(s), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if p is None:
                continue
            self.procs.append((s, p))
            threading.Thread(target=self._stamp, args=(p.stdout, self.rank_log(s)), daemon=True).start()
        t0 = time.time()
        for s, p in self.procs:       # every rank prints "listening on" once its sockets are open
            path = self.rank_log(s)
            while True:
                if p.poll() is not None:
                    raise RuntimeError(f"rank {s.name} exited rc={p.returncode} at start; see {path}")
                if path.exists() and "listening on" in path.read_text(errors="replace"):
                    break
                if time.time() - t0 > wait_s:
                    raise RuntimeError(f"rank {s.name} on {s.host} did not open its port in {wait_s:.0f} s; see {path}")
                time.sleep(0.3)

    def run_coordinator(self) -> int:
        """Start the coordinator process on its host and wait for it. Its output goes to the operator's log."""
        p = self.runner.spawn(self.peer(self.coord), self.coordinate_argv(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p is None:
            return 0
        self.coord_proc = p
        for line in p.stdout:
            self.log(line.rstrip("\n"))
        return p.wait()

    def wait_ranks(self, timeout_s: float = 60.0) -> list[str]:
        """Wait until every rank process has ended (each writes its timeline files as it exits). Returns the names of
        the ranks still running at the timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout_s and any(p.poll() is None for _, p in self.procs):
            time.sleep(0.2)
        return [s.name for s, p in self.procs if p.poll() is None]

    def collect(self) -> dict[str, list[str]]:
        """Copy every other host's run files into the run folder (a tar stream over the same shell). Call it after
        wait_ranks: tar reports a file that changes while it is read."""
        got = {}
        for host in self.collected_hosts():
            d = shell_path(self.host_dir(host))
            try:
                r = self.sh(host, f"cd {d} 2>/dev/null && tar -cf - .", 600)
                if not r.stdout:
                    self.log(f"[run] no run files from {host} ({self.host_dir(host)}): {r.stderr.decode(errors='replace')[-200:]}")
                    got[host] = []
                    continue
                if r.returncode:      # GNU tar ends with 1 when a file changed while it was read; keep what arrived and say so
                    self.log(f"[run] tar on {host} ended with rc {r.returncode}: {r.stderr.decode(errors='replace').strip()[-200:]}")
                got[host] = safe_extract(r.stdout, self.run_dir)
            except (OSError, subprocess.SubprocessError, tarfile.TarError) as e:      # a host that fails here must not lose the others
                self.log(f"[run] collecting the run files of {host} failed: {type(e).__name__}: {e}")
                got[host] = []
        return got

    def commands(self) -> list[dict]:
        return self.runner.as_records()

    def stop(self) -> None:
        t0 = time.time()
        for s, p in self.procs:
            while p.poll() is None and time.time() - t0 < 15:
                time.sleep(0.2)
        self.runner.stop_all()
        for h in sorted({s.host for s in self.slots}):
            try:
                r = self.runner.run(self.peer(h), kill_argv(self.tag), timeout=30)
            except (OSError, subprocess.SubprocessError) as e:      # one host that does not answer must not skip the others
                self.log(f"[run] pkill on {h} did not run: {type(e).__name__}: {e}")
                continue
            if r.returncode not in (0, 1):
                self.log(f"[run] pkill on {h} failed (rc {r.returncode}): {(r.stderr or '').strip()[-200:]}")
