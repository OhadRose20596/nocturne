"""Engine interface. One adapter per engine (llama.cpp RPC today; distributed-llama and the MoE row later).
run.py only talks to this interface. All I/O is through the Runner so dry runs print every command."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Completion:
    prompt_id: str
    text: str
    n_prompt: int | None
    n_gen: int | None
    prompt_ms: float | None
    predicted_ms: float | None
    t_send: float          # wall clock, time.time()
    t_done: float
    raw: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def decode_window(self) -> tuple[float, float] | None:
        """Wall clock [first generated token, last token]. Prefill excluded."""
        if self.predicted_ms is None:
            return None
        return (self.t_done - self.predicted_ms / 1000.0, self.t_done)

    @property
    def decode_tps(self):
        """predicted_n / predicted_ms, the v1 and v2 definition. predicted_ms covers n - 1 steps, so this overstates
        speed on short prompts; kept so new records compare with old ones. Cite decode_ms_per_step instead."""
        return (self.n_gen / (self.predicted_ms / 1000.0)) if self.n_gen and self.predicted_ms else None

    @property
    def decode_steps(self) -> int:
        return max(0, (self.n_gen or 0) - 1)

    @property
    def decode_ms_per_step(self):
        return (self.predicted_ms / self.decode_steps) if self.decode_steps and self.predicted_ms else None

    @property
    def prefill_tps(self):
        return (self.n_prompt / (self.prompt_ms / 1000.0)) if self.n_prompt and self.prompt_ms else None

    @property
    def ttft_s(self):
        return self.prompt_ms / 1000.0 if self.prompt_ms is not None else None


class Engine(ABC):
    name: str = "abstract"

    def __init__(self, cell_name: str, cell_cfg: dict, peers_cfg: dict, model_cfg: dict, network: str, runner, workdir):
        self.cell_name, self.cell, self.peers, self.model, self.network = cell_name, cell_cfg, peers_cfg, model_cfg, network
        self.runner, self.workdir = runner, workdir
        self.notes: list[str] = []
        self.on_event = None       # callable(dict) or None; run.py sets it for the live view (request / response)

    def emit(self, event: str, **fields) -> None:
        if self.on_event:
            self.on_event({"event": event, **fields})

    @abstractmethod
    def prepare(self) -> None:
        """Check binaries, flags and model files. Raise EngineUnsupported if the cell cannot run here."""

    @abstractmethod
    def start_workers(self) -> None: ...

    @abstractmethod
    def launch(self) -> None:
        """Start the requester side and block until it is ready to serve."""

    @abstractmethod
    def complete(self, prompts: list[dict], concurrency: int = 1) -> list[Completion]: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def counters(self) -> dict:
        """{"engine": {...}, "setup": {...}, "memory_peak_gb": {...}, "comm_msgs_per_step": float|None, "worker_logs": {...}}"""

    def requester_peer(self) -> dict:
        return self.peers["peers"][self.peers["roles"]["requester"]]

    def worker_peers(self) -> list[dict]:
        return [self.peers["peers"][w] for w in self.peers["roles"]["workers"]]

    def worker_port(self) -> int:
        """Port the workers listen on; tcpdump filters on it."""
        return int(self.peers["rpc_port"])

    def mem_patterns(self) -> dict:
        """pgrep patterns for the memory probe when the pid is unknown (remote workers)."""
        return {}

    def util_backend(self) -> str | None:
        """Backend the utilization sampler should read, when the engine does not use the peer's tagged accelerator
        (distributed-llama computes on CPU on a peer tagged metal). None: use the peer's backend."""
        return None

    def real_hostnames(self) -> dict[str, str]:
        """peer name -> `hostname -s` on that machine. peers.yaml `host` is a label; records carry the real one
        so that a BASELINE written on a machine matches a cell whose requester is that machine."""
        import socket
        out = {}
        for peer in [self.requester_peer()] + self.worker_peers():
            if self.runner.dry_run:
                out[peer["name"]] = peer["host"]
                continue
            if peer.get("ssh") is None:
                out[peer["name"]] = socket.gethostname().split(".")[0]
            else:
                r = self.runner.run(peer, ["hostname", "-s"], timeout=30)
                out[peer["name"]] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else peer["host"]
        return out


class EngineUnsupported(Exception):
    """The cell cannot run with this engine on these peers (missing tool, missing flag, missing file)."""
