"""Runs commands locally or over ssh. Every command is logged verbatim (host + shell string) so it can be
rerun by hand. dry_run prints instead of executing. Background processes are tracked for stop()."""
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class Cmd:
    host: str
    cmd: str


@dataclass
class Runner:
    dry_run: bool = False
    log: list[Cmd] = field(default_factory=list)
    procs: list = field(default_factory=list)
    echo: bool = True

    @staticmethod
    def _q(a: str) -> str:
        """Quote one argument; keep a leading ~/ expandable on the remote shell."""
        if a.startswith("~/"):
            return '"$HOME"/' + shlex.quote(a[2:])
        return shlex.quote(a)

    def _wrap(self, peer: dict, argv: list[str], env: dict | None = None) -> list[str]:
        shell = " ".join(self._q(a) for a in argv)
        if env:
            shell = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shell
        if peer.get("ssh"):
            return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer["ssh"], shell]
        return ["bash", "-lc", shell]

    def _record(self, peer: dict, full: list[str]) -> str:
        text = " ".join(shlex.quote(a) if i < len(full) - 1 or not peer.get("ssh") else a for i, a in enumerate(full))
        if not peer.get("ssh"):
            text = full[-1]
        self.log.append(Cmd(peer.get("name", peer.get("host", "?")), text))
        if self.echo:
            print(f"[{peer.get('name', '?')}] $ {text}", flush=True)
        return text

    def run(self, peer: dict, argv: list[str], env=None, timeout: float | None = 120, check: bool = False):
        full = self._wrap(peer, argv, env)
        self._record(peer, full)
        if self.dry_run:
            return subprocess.CompletedProcess(full, 0, "", "")
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"command failed on {peer.get('name')}: rc={r.returncode}\n{r.stderr[-2000:]}")
        return r

    def spawn(self, peer: dict, argv: list[str], env=None, stdout=None, stderr=None):
        full = self._wrap(peer, argv, env)
        self._record(peer, full)
        if self.dry_run:
            return None
        # Own process group: `bash -lc` and `ssh` start children (llama-server, the local tcpdump reader loops), and
        # terminating the leader alone leaves them running (I024). stop_all signals the whole group.
        p = subprocess.Popen(full, stdout=stdout, stderr=stderr, text=True, start_new_session=True)
        self.procs.append(p)
        return p

    @staticmethod
    def _signal_group(p, sig) -> None:
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass

    def stop_all(self, grace: float = 5.0):
        for p in self.procs:
            if p.poll() is None:
                self._signal_group(p, signal.SIGTERM)
        t0 = time.time()
        for p in self.procs:
            while p.poll() is None and time.time() - t0 < grace:
                time.sleep(0.1)
            if p.poll() is None:
                self._signal_group(p, signal.SIGKILL)
        self.procs.clear()

    def as_records(self) -> list[dict]:
        return [{"host": c.host, "cmd": c.cmd} for c in self.log]
