"""distributed-llama adapter (row B, tensor parallel over TCP). Gate: ADR-007 amendment 1 (proposed 2026-10-26).
Workers run `dllama worker --host <bind> --port 9999 --nthreads N`; the root (requester) runs `dllama-api` on 127.0.0.1
and the prompts go to its /v1/chat/completions. The API prints no transfer counters, so after the prompts one
`dllama inference` pass on the first prompt reads the engine's own per-token Sent/Recv kB lines; tcpdump on the
worker port measures bytes the same way as row A. Facts read from the upstream sources on 2026-10-26 (src/app.cpp,
src/dllama.cpp, src/dllama-api.cpp, launch.py, docs/HOW_TO_RUN_LINUX_MACOS_WIN.md); nothing here has run yet.
Fixes from review I025 (2026-10-27): readiness never opens a bare TCP connection to the worker (it would take it as a
root and die), the request is not streamed (usage comes only in the non streaming body), the tool stats pass cannot
lose completions already received, Sent/Recv are KiB (times 1024), the utilization sampler is told the engine runs
on CPU although the peers are tagged metal."""
import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .base import Completion, Engine, EngineUnsupported
from .llamacpp_rpc import GENERATION, LoadFailed

# "🔶 Pred  123 ms Sync   45 ms | Sent   512 kB Recv   256 kB | token"  (src/dllama.cpp, generation step)
PRED = re.compile(r"Pred\s*(\d+) ms Sync\s*(\d+) ms \| Sent\s*(\d+) kB Recv\s*(\d+) kB")
EVAL = re.compile(r"Eval\s*(\d+) ms Sync\s*(\d+) ms \| Sent\s*(\d+) kB Recv\s*(\d+) kB \| \((\d+) tokens\)")


class DistributedLlama(Engine):
    name = "distributed-llama"

    def __init__(self, *a, ctx: int = 4096, load_timeout_s: float = 900, request_timeout_s: float = 900, **kw):
        super().__init__(*a, **kw)
        self.ctx, self.load_timeout_s, self.request_timeout_s = ctx, load_timeout_s, request_timeout_s
        self.api = None
        self.api_log = Path(self.workdir) / "server.log"
        self.worker_logs: dict[str, Path] = {}
        self.flags: list[str] = []
        self.commit = "unknown"
        self.load_s = None
        self.tool_stats: dict = {}
        self.stamp_threads: list[threading.Thread] = []

    # ---- helpers -------------------------------------------------------
    def _bin(self, peer: dict, name: str) -> str:
        if not peer.get("dllama_bin"):
            raise EngineUnsupported(f"peer {peer['name']} has no dllama_bin in peers.yaml")
        return str(Path(peer["dllama_bin"]) / name)

    def _worker_ip(self, w: dict) -> str:
        return (w.get("networks") or {}).get(self.network) or w["bind"]

    def worker_port(self) -> int:
        return int(self.peers.get("dllama_port", 9999))

    def _api_url(self) -> str:
        return f"http://127.0.0.1:{self.peers.get('dllama_api_port', 9990)}"

    def _tokenizer(self) -> str:
        if self.model.get("dllama_tokenizer"):
            return self.model["dllama_tokenizer"]
        m = Path(self.model["dllama_path"])
        return str(m.parent / f"dllama_tokenizer_{m.parent.name}.t")

    def _nthreads(self, peer: dict) -> str:
        return str(peer.get("dllama_nthreads") or 4)

    def mem_patterns(self) -> dict:
        return {"requester": "dllama-api", "worker": f"dllama worker.*--port {self.worker_port()}"}

    def util_backend(self) -> str | None:
        """This engine computes on CPU: sampling the GPU of a peer tagged metal would read the wrong resource (I025 7)."""
        return "cpu"

    @staticmethod
    def _stamp(pipe, log: Path) -> None:
        with log.open("w", buffering=1) as fh:
            for line in pipe:
                fh.write(f"{time.time():.6f} {line}" if line.strip() else line)

    # ---- Engine interface --------------------------------------------
    def prepare(self) -> None:
        if self.cell["decode"] != "single":
            raise EngineUnsupported(f"distributed-llama: decode={self.cell['decode']} is {self.cell['status']}: {self.cell.get('notes', '')}")
        req, workers, dry = self.requester_peer(), self.worker_peers(), self.runner.dry_run
        n = 1 + len(workers)
        if n & (n - 1):
            raise EngineUnsupported(f"distributed-llama needs 2^n nodes; root + {len(workers)} workers = {n}")
        if not self.model.get("dllama_path"):
            raise EngineUnsupported("model has no dllama_path in peers.yaml (converted .m file on the root)")
        for peer in [req] + workers:
            for b in ("dllama",) + (("dllama-api",) if peer is req else ()):
                r = self.runner.run(peer, ["test", "-x", self._bin(peer, b)], timeout=30)
                if not dry and r.returncode != 0:
                    raise EngineUnsupported(f"{b} not found in {peer['dllama_bin']} on {peer['name']} (make {b})")
        for f in (self.model["dllama_path"], self._tokenizer()):
            r = self.runner.run(req, ["test", "-f", f], timeout=30)
            if not dry and r.returncode != 0:
                raise EngineUnsupported(f"missing on root: {f} (python launch.py qwen3_8b_q40 in the distributed-llama checkout)")
        for w in workers:
            if self._worker_ip(w).startswith("0.0.0.0"):
                raise EngineUnsupported(f"worker {w['name']} bind is 0.0.0.0; refuse (CLAUDE.md rule 7)")
        r = self.runner.run(req, ["git", "-C", str(Path(req["dllama_bin"])), "rev-parse", "--short", "HEAD"], timeout=30)
        self.commit = "dry-run" if dry else (r.stdout.strip()[:7] if r.returncode == 0 and r.stdout.strip() else "unknown")

    def start_workers(self) -> None:
        for w in self.worker_peers():
            log = Path(self.workdir) / f"worker_{w['name']}.log"
            self.worker_logs[w["name"]] = log
            # --host: the worker binds to the chosen network address, never the 0.0.0.0 default (security: ADR-009)
            argv = [self._bin(w, "dllama"), "worker", "--host", self._worker_ip(w), "--port", str(self.worker_port()),
                    "--nthreads", self._nthreads(w)]
            if self.runner.dry_run:
                self.runner.spawn(w, argv)
                continue
            proc = self.runner.spawn(w, argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            t = threading.Thread(target=self._stamp, args=(proc.stdout, log), daemon=True)
            t.start()
            self.stamp_threads.append(t)
        if self.runner.dry_run:
            return
        # Readiness without touching the port (I025 1): a bare TCP connect is taken by the worker as its root and it
        # exits when the client goes away. The worker prints "Listening on host:port..." but its stdout is block
        # buffered when piped, so the line may arrive late; the process being alive after a short wait is enough, the
        # root (dllama-api) retries the connection itself ("wait for ... worker" loop in nn-network.cpp).
        for w, proc in zip(self.worker_peers(), self.runner.procs[-len(self.worker_peers()):]):
            log = self.worker_logs[w["name"]]
            t0 = time.time()
            seen = False
            while time.time() - t0 < 20:
                if proc.poll() is not None:
                    raise LoadFailed("load_fail", f"worker {w['name']} exited rc={proc.returncode} at start; see {log.name}")
                if log.exists() and "Listening on" in log.read_text(errors="replace"):
                    seen = True
                    break
                time.sleep(0.5)
            if not seen:
                self.notes.append(f"worker {w['name']}: 'Listening on' line not seen within 20 s (stdout buffered); "
                                  f"process alive, proceeding; dllama-api retries the connection")

    def build_flags(self, k: int = 1) -> list[str]:
        req = self.requester_peer()
        subst = {"{dllama_model}": self.model["dllama_path"], "{dllama_tokenizer}": self._tokenizer(),
                 "{nthreads}": self._nthreads(req), "{ctx}": str(self.ctx),
                 "{workers}": " ".join(f"{self._worker_ip(w)}:{self.worker_port()}" for w in self.worker_peers())}
        flags = []
        for tok in self.cell["flags"]:
            v = subst.get(tok, tok)
            flags += v.split(" ") if tok == "{workers}" else [v]     # --workers takes space separated host:port values
        return flags

    def launch(self, k: int = 1) -> None:
        req = self.requester_peer()
        self.flags = self.build_flags(k)
        argv = [self._bin(req, "dllama-api")] + self.flags + ["--host", "127.0.0.1", "--port", str(self.peers.get("dllama_api_port", 9990)),
                                                           "--temperature", "0", "--seed", "42"]
        fh = None if self.runner.dry_run else self.api_log.open("w")
        t0 = time.time()
        self.api = self.runner.spawn(req, argv, stdout=fh, stderr=subprocess.STDOUT)
        if self.runner.dry_run:
            return
        url = self._api_url() + "/v1/models"
        while time.time() - t0 < self.load_timeout_s:
            if self.api.poll() is not None:
                raise LoadFailed("load_fail", f"dllama-api exited rc={self.api.returncode} during load; see server.log")
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    self.load_s = time.time() - t0
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise LoadFailed("load_fail", f"dllama-api not answering after {self.load_timeout_s} s")

    def _one(self, p: dict) -> Completion:
        # Not streamed (I025 2): dllama-api returns `usage` only in the non streaming body (src/dllama-api.cpp, the
        # ChatCompletion with ChatUsage is built in the non stream branch), and n_gen comes from usage.completion_tokens.
        # Without a first token event the client side split between prefill and decode is unknown: prompt_ms is null
        # and predicted_ms is the whole request. decode_tps therefore includes prefill and is marked in `timings.source`.
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": p["max_tokens"],
                "temperature": 0, "seed": 42, "stream": False}
        t0 = time.time()
        self.emit("request", prompt_id=p["id"], t=t0)
        text, fin, t_first, usage = "", None, None, {}
        try:
            r = httpx.post(self._api_url() + "/v1/chat/completions", json=body, timeout=self.request_timeout_s)
            r.raise_for_status()
            data = r.json()
            t1 = time.time()
            for ch in data.get("choices") or []:
                text += (ch.get("message") or {}).get("content") or ""
                fin = ch.get("finish_reason") or fin
            usage = data.get("usage") or {}
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            self.emit("response", prompt_id=p["id"], t=time.time(), error=str(e))
            return Completion(p["id"], "", None, None, None, None, t0, time.time(), {}, error=f"{type(e).__name__}: {e}")
        n_gen = usage.get("completion_tokens")
        n_prompt = usage.get("prompt_tokens")
        prompt_ms = None
        predicted_ms = (t1 - t0) * 1000
        self.emit("response", prompt_id=p["id"], t=t1, n_gen=n_gen)
        return Completion(p["id"], text, n_prompt, n_gen, prompt_ms, predicted_ms, t0, t1,
                          {"timings": {"prompt_n": n_prompt, "predicted_n": n_gen, "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                                       "source": "client wall clock of the whole non streamed request; dllama-api returns "
                                                 "usage only, so predicted_ms includes prefill and prompt_ms is null"},
                           "finish_reason": fin or ("length" if n_gen == p["max_tokens"] else "stop"),
                           "ttft_client_s": None},
                          error=None if fin in (None, "stop", "length") else f"finish_reason={fin!r}")

    def complete(self, prompts: list[dict], concurrency: int = 1) -> list[Completion]:
        if self.runner.dry_run:
            print(f"[requester] POST {self._api_url()}/v1/chat/completions x{len(prompts)} (one at a time; dllama-api is serial), "
                  f"body: temperature 0, seed 42, stream true", flush=True)
            self._tool_stats_pass(prompts[0] if prompts else {"prompt": "Hello", "max_tokens": 32})
            return []
        out = [self._one(p) for p in prompts]
        if prompts:
            try:
                self._tool_stats_pass(prompts[0])
            except (subprocess.TimeoutExpired, OSError) as e:    # I025 8: the completions above are the measurement
                self.notes.append(f"dllama inference stats pass failed ({type(e).__name__}: {e}); tool byte counters null")
        return out

    def _tool_stats_pass(self, p: dict) -> None:
        """One `dllama inference` run after the API server is stopped: the only mode that prints Sent/Recv per token.
        Assumes the workers accept a second root session (runWorkerApp loops; to verify on the first real run)."""
        req = self.requester_peer()
        if self.api is not None and not self.runner.dry_run:
            self.api.terminate()
            try:
                self.api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.api.kill()
            self.api = None
            time.sleep(1.0)
        argv = [self._bin(req, "dllama"), "inference"] + self.build_flags() + ["--prompt", p["prompt"], "--steps", str(min(32, p.get("max_tokens", 32))),
                                                                             "--temperature", "0", "--seed", "42"]
        r = self.runner.run(req, argv, timeout=self.request_timeout_s)
        if self.runner.dry_run:
            return
        text = (r.stdout or "") + (r.stderr or "")
        (Path(self.workdir) / "tool_stats.log").write_text(text)
        preds = PRED.findall(text)
        if preds:
            self.tool_stats = {"steps": len(preds),
                               "sent_kb_per_token": sum(int(x[2]) for x in preds) / len(preds),
                               "recv_kb_per_token": sum(int(x[3]) for x in preds) / len(preds),
                               "pred_ms_median": sorted(int(x[0]) for x in preds)[len(preds) // 2],
                               "sync_ms_median": sorted(int(x[1]) for x in preds)[len(preds) // 2]}
        else:
            self.notes.append("dllama inference printed no Pred/Sent/Recv lines; tool byte counters null")

    def stop(self) -> None:
        self.runner.stop_all()
        for w in self.worker_peers():
            self.runner.run(w, ["pkill", "-f", f"dllama worker.*--port {self.worker_port()}"], timeout=20)

    def server_alive(self) -> bool:
        return self.api is not None and self.api.poll() is None

    def server_pid(self) -> int | None:
        return self.api.pid if self.api is not None else None

    def counters(self) -> dict:
        for t in self.stamp_threads:
            t.join(timeout=5)
        worker_texts = {n: (p.read_text(errors="replace") if p.exists() else "") for n, p in self.worker_logs.items()}
        ts = self.tool_stats
        return {
            "engine": {"name": self.name, "commit": self.commit, "flags": list(self.flags),
                       "format": self.model.get("format", "dllama-q40"), "tokenizer": self._tokenizer()},
            "setup": {"load_total_s": self.load_s, "transfer_s": None, "cache_state": None, "bytes_sent_gb": None},
            "setup_notes": ["distributed-llama: weights stay on the root and are sliced per node at start; no weight transfer "
                            "log, so transfer_s, cache_state and bytes_sent_gb are null"] + list(self.notes),
            "memory_buffers_gb": {},
            "generation": {"temperature": 0, "seed": 42},
            "worker_logs": {},            # not llama.cpp RPC logs: run.py must not parse them for rpc commands
            "comm": {"rpc_commands_per_step": None, "tokens_per_step": None, "rpc_commands_per_token": None,
                     # upstream prints kB but divides by 1024 (I025 13): KiB
                     "tool_bytes_sent_per_token": (ts["sent_kb_per_token"] * 1024) if ts else None,
                     "tool_bytes_recv_per_token": (ts["recv_kb_per_token"] * 1024) if ts else None,
                     "source": "rpc_commands null: this engine has no RPC log. tool_bytes: dllama inference Sent/Recv kB per "
                               "generated token on the first prompt (32 steps). bytes: tcpdump on the worker port as in row A."},
            "placement_lines": [],
            "model_sha256": None,
            "draft_sha256": None,
            "oom_in_log": False,
            "net_err_in_log": any("Connection refused" in t or "error" in t.lower() for t in worker_texts.values()),
            "raw_worker_logs": worker_texts,
        }
