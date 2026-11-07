"""torch_tp adapter (row B1t, ADR-007 amendment 2): rank 1 runs `python -m nocturne.engines.torch_tp worker` on the
worker over ssh, bound to the network address (never 0.0.0.0; the port accepts any rank 0, security deferred by
ADR-009); rank 0 runs `... serve` on the requester, HTTP on 127.0.0.1 in the llama-server dialect, so requests reuse
`stream_chat`. BASELINE is `--world 1` on the requester alone. comm comes from the engine's all-reduce frame counters
(source "engine counters: all-reduce frames"); tcpdump on torch_port stays the independent wire measurement; memory per
rank comes from /metrics (torch.mps allocations), because `ps -o rss` undercounts Metal."""
import hashlib
import os
import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .base import Completion, Engine, EngineUnsupported
from .llamacpp_rpc import LoadFailed, stream_chat

GENERATION = {"temperature": 0, "seed": 42, "samplers": ["temperature"], "repeat_penalty": 1.0}   # greedy; the engine is greedy only


class TorchTp(Engine):
    name = "torch_tp"

    def __init__(self, *a, ctx: int = 4096, load_timeout_s: float = 900, request_timeout_s: float = 900, **kw):
        super().__init__(*a, **kw)
        self.ctx, self.load_timeout_s, self.request_timeout_s = ctx, load_timeout_s, request_timeout_s
        self.server = None
        self.server_log = Path(self.workdir) / "server.log"
        self.worker_logs: dict[str, Path] = {}
        self.flags: list[str] = []
        self.commit = "unknown"
        self.load_s = None
        self.k = 1
        self.model_sha = None
        self.metrics: dict | None = None
        self.stamp_threads: list[threading.Thread] = []
        self.fixed_tokens: int | None = None
        self.trace_allreduce = False

    # ---- helpers -------------------------------------------------------
    def _python(self, peer: dict) -> str:
        if not peer.get("python"):
            raise EngineUnsupported(f"peer {peer['name']} has no `python` (torch venv interpreter) in peers.yaml")
        return peer["python"]

    def _worker_ip(self, w: dict) -> str:
        return (w.get("networks") or {}).get(self.network) or w["bind"]

    def worker_port(self) -> int:
        return int(self.peers.get("torch_port", 50053))

    def _api_port(self) -> int:
        return int(self.peers.get("torch_api_port", 8092))

    def _server_url(self) -> str:
        return f"http://127.0.0.1:{self._api_port()}"

    def _world(self) -> int:
        return 1 + len(self.worker_peers())

    def mem_patterns(self) -> dict:
        return {"requester": "torch_tp serve", "worker": f"torch_tp worker.*--port {self.worker_port()}"}

    @staticmethod
    def _stamp(pipe, log: Path) -> None:
        with log.open("w", buffering=1) as fh:
            for line in pipe:
                fh.write(f"{time.time():.6f} {line}" if line.strip() else line)

    # ---- Engine interface --------------------------------------------
    def prepare(self) -> None:
        if self.cell["decode"] != "single":
            raise EngineUnsupported(f"torch_tp: decode={self.cell['decode']} not implemented (greedy single stream only)")
        req, workers, dry = self.requester_peer(), self.worker_peers(), self.runner.dry_run
        mdir = self.model.get("torch_model")
        if not mdir:
            raise EngineUnsupported("model has no torch_model (safetensors directory) in peers.yaml")
        if len(workers) > 1:
            raise EngineUnsupported("torch_tp implements two ranks: one worker")
        for peer in [req] + workers:
            py = self._python(peer)
            r = self.runner.run(peer, ["test", "-x", py], timeout=30)
            if not dry and r.returncode != 0:
                raise EngineUnsupported(f"python {py} not executable on {peer['name']}; create the torch venv (README, row B)")
            r = self.runner.run(peer, ["test", "-f", str(Path(mdir) / "config.json")], timeout=30)
            if not dry and r.returncode != 0:
                raise EngineUnsupported(f"{mdir}/config.json missing on {peer['name']}; hf download the model there")
            r = self.runner.run(peer, [py, "-c", "import torch, transformers, safetensors; print(torch.__version__)"], timeout=120)
            if not dry and r.returncode != 0:
                raise EngineUnsupported(f"torch stack missing in {py} on {peer['name']}: {(r.stderr or '')[-200:]}")
        for w in workers:
            if self._worker_ip(w).startswith("0.0.0.0"):
                raise EngineUnsupported(f"worker {w['name']} bind is 0.0.0.0; refuse (CLAUDE.md rule 7)")
        root = Path(__file__).resolve().parents[3]
        r = self.runner.run(req, ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=30)
        self.commit = "dry-run" if dry else (r.stdout.strip()[:7] if r.returncode == 0 and r.stdout.strip() else "unknown")
        if not dry:
            self.model_sha = self._model_sha(Path(os.path.expanduser(mdir)))

    @staticmethod
    def _model_sha(mdir: Path) -> str | None:
        """sha256 of the safetensors index (or the single file's header) plus config.json: identifies the checkout
        without hashing 15 GB on every run; the record note names the method."""
        idx = mdir / "model.safetensors.index.json"
        h = hashlib.sha256((mdir / "config.json").read_bytes())
        if idx.exists():
            h.update(idx.read_bytes())
        elif (mdir / "model.safetensors").exists():
            with (mdir / "model.safetensors").open("rb") as f:
                h.update(f.read(1 << 20))
        else:
            return None
        return h.hexdigest()

    def start_workers(self) -> None:
        mdir = self.model["torch_model"]
        for w in self.worker_peers():
            log = Path(self.workdir) / f"worker_{w['name']}.log"
            self.worker_logs[w["name"]] = log
            argv = [self._python(w), "-m", "nocturne.engines.torch_tp", "worker", "--bind", self._worker_ip(w), "--port", str(self.worker_port()),
                    "--model", mdir, "--rank", "1", "--world", str(self._world())]
            if self.runner.dry_run:
                self.runner.spawn(w, argv)
                continue
            proc = self.runner.spawn(w, argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            t = threading.Thread(target=self._stamp, args=(proc.stdout, log), daemon=True)
            t.start()
            self.stamp_threads.append(t)
        if self.runner.dry_run:
            return
        # The worker loads its slice only after rank 0 says hello, so readiness here is "listening" in its log
        # (printed with flush) or the process still alive after 30 s; rank 0 retries the connect for 120 s.
        for w, proc in zip(self.worker_peers(), self.runner.procs[-len(self.worker_peers()):]):
            log = self.worker_logs[w["name"]]
            t0 = time.time()
            while time.time() - t0 < 30:
                if proc.poll() is not None:
                    raise LoadFailed("load_fail", f"torch_tp worker {w['name']} exited rc={proc.returncode} at start; see {log.name}")
                if log.exists() and "listening on" in log.read_text(errors="replace"):
                    return
                time.sleep(0.5)
            self.notes.append(f"worker {w['name']}: no 'listening on' line within 30 s; process alive, rank 0 will retry the connect")

    def build_flags(self, k: int = 1) -> list[str]:
        subst = {"{torch_model}": self.model["torch_model"], "{ctx}": str(self.ctx)}
        flags = [subst.get(t, t) for t in self.cell["flags"]]
        if "--world" in flags:
            flags[flags.index("--world") + 1] = str(self._world())
        return flags

    def launch(self, k: int = 1) -> None:
        req = self.requester_peer()
        self.k = k
        self.flags = self.build_flags(k)
        argv = [self._python(req), "-m", "nocturne.engines.torch_tp", "serve"] + self.flags + ["--host", "127.0.0.1", "--port", str(self._api_port())]
        for w in self.worker_peers():
            argv += ["--peer", f"{self._worker_ip(w)}:{self.worker_port()}"]
        if self.trace_allreduce and self.worker_peers():
            argv.append("--trace-allreduce")
        fh = None if self.runner.dry_run else self.server_log.open("w")
        t0 = time.time()
        self.server = self.runner.spawn(req, argv, stdout=fh, stderr=subprocess.STDOUT)
        if self.runner.dry_run:
            return
        url = self._server_url() + "/health"
        while time.time() - t0 < self.load_timeout_s:
            if self.server.poll() is not None:
                raise LoadFailed(self._classify_log(), f"torch_tp serve exited rc={self.server.returncode} during load; see server.log")
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    self.load_s = time.time() - t0
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise LoadFailed("load_fail", f"torch_tp serve not healthy after {self.load_timeout_s} s")

    def _classify_log(self) -> str:
        text = self.server_log.read_text(errors="replace") if self.server_log.exists() else ""
        if "out of memory" in text.lower() or "MPS backend out of memory" in text or "Insufficient Memory" in text:
            return "oom"
        if "Connection refused" in text or "timed out" in text or "peer closed" in text:
            return "network_abort"
        return "load_fail"

    def _one(self, p: dict) -> Completion:
        return stream_chat(self._server_url(), p, self.request_timeout_s, self.emit, GENERATION, self.fixed_tokens)

    def warmup(self, p: dict) -> float | None:
        if self.runner.dry_run:
            print(f"[requester] warm-up: POST {self._server_url()}/v1/chat/completions ({p['id']}, max_tokens 8)", flush=True)
            return None
        c = stream_chat(self._server_url(), dict(p, max_tokens=8), self.request_timeout_s, lambda *a, **k: None, GENERATION)
        return c.t_done - c.t_send

    def complete(self, prompts: list[dict], concurrency: int = 1) -> list[Completion]:
        if self.runner.dry_run:
            print(f"[requester] POST {self._server_url()}/v1/chat/completions x{len(prompts)} (one at a time), greedy, stream true"
                  + (f", fixed_tokens {self.fixed_tokens} with ignore_eos" if self.fixed_tokens else ""), flush=True)
            print(f"[requester] GET {self._server_url()}/metrics", flush=True)
            return []
        out = [self._one(p) for p in prompts]
        try:
            self.metrics = httpx.get(self._server_url() + "/metrics", timeout=60).json()
        except (httpx.HTTPError, ValueError) as e:
            self.notes.append(f"/metrics failed ({e}); engine counters and per rank memory null")
        return out

    def stop(self) -> None:
        self.runner.stop_all()
        for w in self.worker_peers():
            self.runner.run(w, ["pkill", "-f", f"torch_tp worker.*--port {self.worker_port()}"], timeout=20)

    def server_alive(self) -> bool:
        return self.server is not None and self.server.poll() is None

    def server_pid(self) -> int | None:
        return self.server.pid if self.server is not None else None

    def counters(self) -> dict:
        for t in self.stamp_threads:
            t.join(timeout=5)
        text = self.server_log.read_text(errors="replace") if self.server_log.exists() else ""
        worker_texts = {n: (p.read_text(errors="replace") if p.exists() else "") for n, p in self.worker_logs.items()}
        m = self.metrics or {}
        req = self.requester_peer()["name"]
        names = [req] + [w["name"] for w in self.worker_peers()]
        mem = {}
        for r, peer in enumerate(names):
            rk = (m.get("ranks") or {}).get(str(r)) or {}
            drv = rk.get("driver_allocated_bytes")
            mem[peer] = round(drv / 1e9, 3) if drv else None
        c = m.get("counters") or {}
        steps, toks = m.get("decode_steps") or 0, m.get("tokens_generated") or 0
        ar = c.get("allreduce_frames")
        per_step = m.get("allreduces_per_step")
        comm = {"rpc_commands_per_step": (ar / (steps + (m.get("prefill_steps") or 0))) if ar and (steps or m.get("prefill_steps")) else None,
                # every all-reduce blocks rank 0 on the peer's frame; the STEP frame blocks rank 1, so one more per step
                "blocking_syncs_per_step": (per_step + 1) if per_step else None,
                "messages_per_step": (per_step + 1) if per_step else None,
                "tokens_per_step": 1.0 if steps else None,
                "rpc_commands_per_token": (ar / toks) if ar and toks else None,
                "engine_frames_per_token": (c.get("sent_frames") / toks) if c.get("sent_frames") and toks else None,
                "engine_bytes_per_token": ((c.get("sent_bytes", 0) + c.get("recv_bytes", 0)) / toks) if toks and c else None,
                "compute_steps": (steps + (m.get("prefill_steps") or 0)) or None, "tokens_generated": toks or None,
                "source": "rpc_commands: engine counters: all-reduce frames sent by rank 0 (one per all-reduce, 2 per layer), per step "
                          "over prefill plus decode steps, per token over generated tokens; blocking_syncs and messages per step: "
                          "2 * layers all-reduces plus the STEP frame; engine_bytes: frame bytes sent plus "
                          "received by rank 0 over the whole run (prefill frames included) per generated token; bytes: tcpdump on "
                          "torch_port inside the decode windows, as in row A"}
        load_lines = [l for l in text.splitlines() if "[torch_tp]" in l]
        if m.get("allreduce_trace"):
            tr = m["allreduce_trace"]
            Path(self.workdir).mkdir(parents=True, exist_ok=True)
            (Path(self.workdir) / "allreduce_trace.json").write_text(json.dumps(tr, indent=2) + "\n")
            per = tr["ms_per_allreduce"]
            self.notes.append("all-reduce trace (ms per all-reduce, mean over %d steps): " % tr["steps"]
                              + ", ".join(f"{k} {v:.2f}" for k, v in per.items()) + "; saved as allreduce_trace.json")
        world = m.get("world") or self._world()
        return {
            "engine": {"name": self.name, "commit": self.commit, "flags": list(self.flags), "format": "safetensors-bf16",
                       "runtime": f"torch {m.get('torch', 'unknown')} " + ((m.get("ranks") or {}).get("0") or {}).get("device", "?"),
                       "world": world},
            "setup": {"load_total_s": self.load_s, "transfer_s": None, "cache_state": None, "bytes_sent_gb": None},
            "setup_notes": ["torch_tp: each rank reads its own slice from local safetensors; no weight transfer, so transfer_s, "
                            "cache_state and bytes_sent_gb are null. model.sha256 is sha256(config.json + safetensors index or "
                            "first MiB), not of the weights"] + list(self.notes),
            "memory_buffers_gb": mem,
            "generation": dict(GENERATION),
            "worker_logs": {},              # not llama.cpp RPC logs
            "comm": comm,
            "placement_lines": load_lines,
            "model_sha256": self.model_sha,
            "draft_sha256": None,
            "oom_in_log": "out of memory" in text.lower(),
            "net_err_in_log": any("Connection refused" in t or "went away" in t for t in worker_texts.values()),
            "raw_worker_logs": worker_texts,
            "tp_outcome": None,
        }
