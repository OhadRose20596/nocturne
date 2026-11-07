"""llama.cpp adapter: workers run ggml-rpc-server, the requester runs llama-server (single launch path for
BASELINE, A1, A2, A4). Sampling is pinned in the flags and in every request: temp 0, seed 42, greedy.
Flags were verified against llama.cpp 1e7bcf3 (see README, "Verified against 1e7bcf3")."""
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .base import Completion, Engine, EngineUnsupported

# -lv 4: at the default verbosity (3) this build does not print load_tensors / buffer size / offloaded lines,
# and memory_peak_gb and placement_lines come from them (checked on 1e7bcf3, 2026-10-26).
FIXED_SERVER_FLAGS = ["--host", "127.0.0.1", "--temp", "0", "--seed", "42", "--samplers", "temperature",
                      "--repeat-penalty", "1.0", "--metrics", "-lv", "4"]
# Pinned per request as well; recorded in prompts.generation and compared by find_baseline.
GENERATION = {"temperature": 0, "seed": 42, "samplers": ["temperature"], "repeat_penalty": 1.0}
REQUIRED_FLAGS = {"--rpc", "--tensor-split", "--parallel", "--temp", "--seed", "--samplers", "--repeat-penalty"}
SPEC_FLAGS = {"--model-draft", "--spec-draft-n-max", "--spec-draft-p-min", "--spec-type", "--device-draft"}
# Log lines look like "0.01.390.906 I load_tensors:  MTL0_Mapped model buffer size =  4168.09 MiB" (timestamp and level
# prefix, device names MTL0 / MTL0_Mapped / CPU_Mapped / RPC[ip:port]). In tensor mode (B1) the device is "Meta()" and
# the size is the per device share: the 7B Q4_K_M model buffer is 4168 MiB in layer mode and "Meta() model buffer size
# = 2084.43 MiB" with two devices (loopback run, 2026-10-27), and the worker received 2.3 GB of set_tensor payload.
BUF = re.compile(r"^(?:[\d.]+\s+[A-Z]\s+)?(?:load_tensors|llama_kv_cache|llama_context|graph_reserve|llama_kv_cache_unified|llama_memory_recurrent)?:?\s*"
                 r"(RPC\d*\[[^\]]+\]|Meta\(\)|Metal(?:_Mapped)?|MTL\d+(?:_Mapped)?|CUDA\d+(?:_Host)?|Vulkan\d+|CPU(?:_Mapped)?|CPU_AARCH64|CPU_REPACK)\s+"
                 r"(?:model|KV|compute|output) buffer size\s*=\s*([\d.]+) MiB", re.MULTILINE)
OOM = re.compile(r"out of memory|OutOfMemory|failed to allocate|kIOGPUCommandBufferCallbackErrorOutOfMemory|"
                 r"cudaErrorMemoryAllocation|ggml_metal_.*failed", re.IGNORECASE)
NET_ERR = re.compile(r"Connection refused|failed to connect|connection reset|Broken pipe|RPC.*(error|failed)|"
                     r"recv_msg failed|send_msg failed|recv failed \(bytes_recv=0|Remote RPC server crashed", re.IGNORECASE)
# Cell B1 (--split-mode tensor over RPC, supervisor question Q7). Lines that say this build cannot do it: the arch
# check in src/llama-model.cpp, fit.cpp with fit on, a GGML_ASSERT in ggml-backend-meta.cpp, "not implemented".
TP_UNSUPPORTED = re.compile(r"SPLIT_MODE_TENSOR.*(not implemented|requires|needs)|not implemented for SPLIT_MODE_TENSOR|"
                            r"GGML_ASSERT|not implemented|unsupported", re.IGNORECASE)
# placement.log keeps these lines (scripts/10_capacity.sh line 70 list plus the Meta device and flash_attn lines)
PLACEMENT = re.compile(r"load_tensors|offloaded \d+/\d+ layers|RPC\d*\[|model buffer size|assigned to device|failed to allocate|"
                       r"out of memory|OOM|ggml_metal.*alloc|llama_model_load: error|Meta device|tensor parallelism|"
                       r"SPLIT_MODE_TENSOR|not implemented|unsupported|GGML_ASSERT|flash_attn|split_mode|creating a Meta",
                       re.IGNORECASE)


def preflight_message(w: dict, binary: str, rc: int, stderr: str) -> str:
    """ssh exits 255 when the host is unreachable; `test -x` exits 1 when the file is missing. The 2026-10-26 A1 wifi r1
    failure was the first case (peers.yaml ssh still pointed at the hotspot address) reported as the second."""
    target = w.get("ssh") or "local"
    if w.get("ssh") and rc == 255:
        return (f"worker {w['name']}: ssh to {target} failed (rc 255: {stderr.strip()[-160:] or 'unreachable'}); "
                f"check peers.yaml ssh: for the current network, the binary {binary} was not checked")
    return f"worker {w['name']}: ggml-rpc-server not executable at {binary} (via ssh {target}, rc {rc}); check peers.yaml bin:"


class LoadFailed(Exception):
    def __init__(self, failure_class: str, msg: str):
        super().__init__(msg)
        self.failure_class = failure_class


def sha256_file(path: str, cache_dir: Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    c = cache_dir / f"{p.name}.sha256"
    if c.exists() and c.stat().st_mtime >= p.stat().st_mtime:
        return c.read_text().strip()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    c.write_text(h.hexdigest())
    return h.hexdigest()


def stream_chat(server_url: str, p: dict, timeout_s: float, emit, generation: dict, fixed_tokens: int | None = None) -> Completion:
    """One streamed /v1/chat/completions request in the llama-server dialect, shared with the torch_tp adapter whose
    server speaks the same format. ttft_s is the wall time to the first chunk that carries content; the final chunk
    carries `timings` (server-task.cpp, to_json_oaicompat_chat_stream, verified on 1e7bcf3). fixed_tokens N sends
    max_tokens N and ignore_eos true (a sampling field llama-server and the torch_tp server both read)."""
    body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": p["max_tokens"], **generation, "stream": True}
    if fixed_tokens:
        body.update({"max_tokens": fixed_tokens, "ignore_eos": True})
    t0 = time.time()
    emit("request", prompt_id=p["id"], t=t0)
    text, tm, fin, t_first, extra = "", {}, None, None, {}
    try:
        with httpx.stream("POST", server_url + "/v1/chat/completions", json=body, timeout=timeout_s) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content") or ""
                    if delta and t_first is None:
                        t_first = time.time()
                    text += delta
                    fin = ch.get("finish_reason") or fin
                if chunk.get("timings"):
                    tm = chunk["timings"]
                if chunk.get("frames"):
                    extra["frames"] = chunk["frames"]
        t1 = time.time()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        emit("response", prompt_id=p["id"], t=time.time(), error=str(e))
        return Completion(p["id"], "", None, None, None, None, t0, time.time(), {}, error=f"{type(e).__name__}: {e}")
    emit("response", prompt_id=p["id"], t=t1, n_gen=tm.get("predicted_n"))
    return Completion(p["id"], text, tm.get("prompt_n"), tm.get("predicted_n"), tm.get("prompt_ms"), tm.get("predicted_ms"), t0, t1,
                      {"timings": tm, "finish_reason": fin, "ttft_client_s": (t_first - t0) if t_first else None, **extra},
                      error=None if fin in ("stop", "length") else f"finish_reason={fin!r}")


class LlamaCppRpc(Engine):
    name = "llama.cpp"

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
        self.draft_sha = None
        self.stamp_threads: list[threading.Thread] = []
        self.overrides: dict[str, str] = {}      # run.py --override-flag: flag value replaced in the template (B1 row data point)
        self.fixed_tokens: int | None = None

    # ---- helpers -------------------------------------------------------
    def _bin(self, peer: dict, name: str) -> str:
        return str(Path(peer["bin"]) / name)

    def _worker_ip(self, worker: dict) -> str:
        # The worker listens on, and the requester connects to, the address of the chosen network.
        # `bind` is the fallback when peers.yaml has no entry for that network.
        nets = worker.get("networks") or {}
        return nets.get(self.network) or worker["bind"]

    def _endpoint(self, worker: dict) -> str:
        return f"{self._worker_ip(worker)}:{self.peers['rpc_port']}"

    def _server_url(self) -> str:
        return f"http://127.0.0.1:{self.peers['server_port']}"

    def _help(self, peer: dict, binary: str) -> str:
        r = self.runner.run(peer, [self._bin(peer, binary), "--help"], timeout=60)
        return (r.stdout or "") + (r.stderr or "")

    def _git_commit(self, peer: dict) -> str:
        src = Path(peer["bin"]).parent.parent
        r = self.runner.run(peer, ["git", "-C", str(src), "rev-parse", "--short", "HEAD"], timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:7]
        r = self.runner.run(peer, [self._bin(peer, "llama-server"), "--version"], timeout=60)
        m = re.search(r"\b([0-9a-f]{7,10})\b", (r.stdout or "") + (r.stderr or ""))
        return m.group(1)[:7] if m else "unknown"

    def tensor_mode(self) -> bool:
        f = self.flags or self.build_flags(self.k)
        return "--split-mode" in f and f[f.index("--split-mode") + 1] == "tensor"

    # ---- Engine interface --------------------------------------------
    def prepare(self) -> None:
        req = self.requester_peer()
        dry = self.runner.dry_run
        if not dry and not Path(self._bin(req, "llama-server")).is_file():
            raise EngineUnsupported(f"llama-server not found in {req['bin']} on {req['name']}")
        if not dry and not Path(self.model["path"]).is_file():
            raise EngineUnsupported(f"model file missing on requester: {self.model['path']}")
        help_text = "" if dry else self._help(req, "llama-server")
        if not dry:
            missing = {f for f in REQUIRED_FLAGS if f not in help_text}
            if self.cell["decode"] == "speculative":
                missing |= {f for f in SPEC_FLAGS if f not in help_text}
            if "--split-mode" in self.cell["flags"]:
                m = re.search(r"--split-mode \{([^}]*)\}", help_text)
                modes = set(m.group(1).split(",")) if m else set()
                want = self.overrides.get("--split-mode", self.cell["flags"][self.cell["flags"].index("--split-mode") + 1])
                if want not in modes:
                    raise EngineUnsupported(f"llama-server --split-mode has {sorted(modes)}, not {want!r}; this build cannot run the cell")
                if "-fit" in self.cell["flags"] and "--fit" not in help_text:
                    missing.add("--fit")
            if missing:
                raise EngineUnsupported(f"llama-server lacks flags {sorted(missing)}; see README 'Verified against 1e7bcf3'")
        if self.cell["decode"] == "speculative":
            draft = self.model.get("draft")
            if not draft:
                raise EngineUnsupported("cell needs a draft model: set models.<name>.draft in peers.yaml")
            if not dry and not Path(draft).is_file():
                raise EngineUnsupported(f"draft model missing on requester: {draft}")
        self.commit = "dry-run" if dry else self._git_commit(req)
        cache = Path(self.workdir).parent / ".cache"
        if not dry:
            self.model_sha = sha256_file(self.model["path"], cache)
            if self.cell["decode"] == "speculative":
                self.draft_sha = sha256_file(self.model["draft"], cache)
        for w in self.worker_peers():
            if self._worker_ip(w).startswith("0.0.0.0"):
                raise EngineUnsupported(f"worker {w['name']} bind is 0.0.0.0; refuse (CLAUDE.md rule 7)")
            r = self.runner.run(w, ["test", "-x", self._bin(w, "ggml-rpc-server")], timeout=30)
            if not dry and r.returncode != 0:
                raise EngineUnsupported(preflight_message(w, self._bin(w, "ggml-rpc-server"), r.returncode, r.stderr or ""))

    def start_workers(self) -> None:
        for w in self.worker_peers():
            log = Path(self.workdir) / f"worker_{w['name']}.log"
            self.worker_logs[w["name"]] = log
            argv = [self._bin(w, "ggml-rpc-server"), "-H", self._worker_ip(w), "-p", str(self.peers["rpc_port"]), "-c"]
            if w.get("device"):
                argv += ["-d", w["device"]]
            if self.runner.dry_run:
                self.runner.spawn(w, argv, env={"GGML_RPC_DEBUG": "1"})
                continue
            # The worker log has no clock of its own. The runner stamps every line with time.time() as it arrives
            # over the ssh pipe ("1793019800.833541 [set_tensor] ..."); setup.transfer_s and the live replay use it.
            proc = self.runner.spawn(w, argv, env={"GGML_RPC_DEBUG": "1"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            t = threading.Thread(target=self._stamp, args=(proc.stdout, log), daemon=True)
            t.start()
            self.stamp_threads.append(t)
        if self.runner.dry_run:
            return
        for w in self.worker_peers():
            host, port = self._endpoint(w).rsplit(":", 1)
            t0 = time.time()
            while time.time() - t0 < 60:
                try:
                    with socket.create_connection((host, int(port)), timeout=2):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                raise LoadFailed("network_abort", f"worker {w['name']} did not open {host}:{port} within 60 s")

    @staticmethod
    def _stamp(pipe, log: Path) -> None:
        with log.open("w", buffering=1) as fh:
            for line in pipe:
                fh.write(f"{time.time():.6f} {line}" if line.strip() else line)

    def build_flags(self, k: int = 1) -> list[str]:
        workers = self.worker_peers()
        subst = {
            "{model}": self.model["path"],
            "{draft}": self.model.get("draft") or "",
            "{rpc}": ",".join(self._endpoint(w) for w in workers),
            "{k}": str(k),
            "{ts}": "",
            "{ot}": "",
            # the requester's own device (e.g. MTL0): pins the draft model locally with --device-draft,
            # otherwise llama.cpp splits the draft over the RPC devices too (seen in A2 on 2026-10-26)
            "{req_device}": self.requester_peer().get("device") or "none",
        }
        ts = self.peers.get("tensor_split")
        if ts:
            subst["{ts}"] = ",".join(str(x) for x in ts)
        tmpl = list(self.cell["flags"])
        flags: list[str] = []
        i = 0
        while i < len(tmpl):
            tok = tmpl[i]
            if tok in ("-ts", "--tensor-split") and not subst["{ts}"]:
                i += 2
                continue
            if tok in self.overrides and i + 1 < len(tmpl):
                flags += [tok, self.overrides[tok]]
                i += 2
                continue
            flags.append(subst.get(tok, tok))
            i += 1
        flags += ["--port", str(self.peers["server_port"]), "-c", str(self.ctx * max(1, k))] + FIXED_SERVER_FLAGS
        return flags

    def launch(self, k: int = 1) -> None:
        req = self.requester_peer()
        self.k = k
        self.flags = self.build_flags(k)
        argv = [self._bin(req, "llama-server")] + self.flags
        fh = None if self.runner.dry_run else self.server_log.open("w")
        t0 = time.time()
        self.server = self.runner.spawn(req, argv, stdout=fh, stderr=subprocess.STDOUT)
        if self.runner.dry_run:
            return
        url = self._server_url() + "/health"
        while time.time() - t0 < self.load_timeout_s:
            if self.server.poll() is not None:
                raise LoadFailed(self._classify_log(), f"llama-server exited rc={self.server.returncode} during load")
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    self.load_s = time.time() - t0
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise LoadFailed("load_fail", f"llama-server not healthy after {self.load_timeout_s} s")

    def _classify_log(self) -> str:
        text = self.server_log.read_text(errors="replace") if self.server_log.exists() else ""
        if OOM.search(text):
            return "oom"
        if self.tensor_mode() and TP_UNSUPPORTED.search(text):
            return "tool_unsupported"
        if NET_ERR.search(text):
            return "network_abort"
        return "load_fail"

    def first_error_line(self) -> str | None:
        """First log line that explains a tensor-mode failure (B1 notes carry it verbatim)."""
        text = self.server_log.read_text(errors="replace") if self.server_log.exists() else ""
        for line in text.splitlines():
            if TP_UNSUPPORTED.search(line) or OOM.search(line) or " E " in line[:24] or "error" in line.lower():
                return line.strip()[:300]
        return None

    def _one(self, p: dict) -> Completion:
        return stream_chat(self._server_url(), p, self.request_timeout_s, self.emit, GENERATION, self.fixed_tokens)

    def warmup(self, p: dict) -> float | None:
        """One untimed request before the prompts (first request pays kernel and cache warm-up). Returns its wall time."""
        if self.runner.dry_run:
            print(f"[requester] warm-up: POST {self._server_url()}/v1/chat/completions ({p['id']}, max_tokens 8)", flush=True)
            return None
        c = stream_chat(self._server_url(), dict(p, max_tokens=8), self.request_timeout_s, lambda *a, **k: None, GENERATION)
        return c.t_done - c.t_send

    def complete(self, prompts: list[dict], concurrency: int = 1) -> list[Completion]:
        if self.runner.dry_run:
            print(f"[requester] POST {self._server_url()}/v1/chat/completions x{len(prompts)} (concurrency {concurrency}), "
                  f"body: {GENERATION}, stream true, max_tokens per prompt"
                  + (f", fixed_tokens {self.fixed_tokens} with ignore_eos" if self.fixed_tokens else ""), flush=True)
            return []
        if concurrency <= 1:
            return [self._one(p) for p in prompts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(self._one, prompts))

    def stop(self) -> None:
        self.runner.stop_all()
        port = str(self.peers["rpc_port"])
        for w in self.worker_peers():
            self.runner.run(w, ["pkill", "-f", f"ggml-rpc-server.*-p {port}"], timeout=20)

    def server_alive(self) -> bool:
        return self.server is not None and self.server.poll() is None

    def server_pid(self) -> int | None:
        return self.server.pid if self.server is not None else None

    def counters(self) -> dict:
        text = self.server_log.read_text(errors="replace") if self.server_log.exists() else ""
        by_host: dict[str, float] = {}
        req = self.requester_peer()["name"]
        ip_to_host = {self._endpoint(w).split(":")[0]: w["name"] for w in self.worker_peers()}
        meta_hosts = [req] + [w["name"] for w in self.worker_peers()]     # the Meta device spans every device in --rpc order
        for dev, mib in BUF.findall(text):
            if dev == "Meta()":        # per device share, held by each device of the Meta set
                for h in meta_hosts:
                    by_host[h] = by_host.get(h, 0.0) + float(mib) / 1024.0
                continue
            host = req
            if dev.startswith("RPC"):  # "RPC0[ip:port]" on 1e7bcf3 (device index), "RPC[ip:port]" on older builds
                endpoint = dev[dev.index("[") + 1:-1]
                host = ip_to_host.get(endpoint.rsplit(":", 1)[0], dev)
            by_host[host] = by_host.get(host, 0.0) + float(mib) / 1024.0
        for t in self.stamp_threads:
            t.join(timeout=5)
        worker_texts = {n: (p.read_text(errors="replace") if p.exists() else "") for n, p in self.worker_logs.items()}
        bytes_sent = 0
        hash_hits = 0
        transfer_s, transfer_notes = None, []
        for t in worker_texts.values():
            bytes_sent += sum(int(m) for m in re.findall(r"\[set_tensor\].*?size:\s*(\d+)", t))
            hash_hits += len(re.findall(r"\[set_tensor_hash\]", t))
            from ..probes.capture import transfer_window_from_worker_log
            ts, nn = transfer_window_from_worker_log(t)
            transfer_notes += nn
            if ts is not None:
                transfer_s = max(transfer_s or 0.0, ts)
        cache_state = None
        if worker_texts:
            cache_state = "warm" if hash_hits > 0 else "cold"
        placement = [l for l in text.splitlines() if PLACEMENT.search(l)]
        tp = None
        if self.tensor_mode() and self.server is not None:      # only when llama-server was launched (not a preflight failure)
            # Outcome of the tensor-mode attempt (cell B1): (a) both sides hold weights; (b) loaded but every buffer is
            # local, the RPC device was ignored; (c) did not load. run.py turns (b) and (c) into tool_unsupported.
            worker_gb = sum(v for h, v in by_host.items() if h != req)
            if self.load_s is None:
                tp = {"outcome": "c", "detail": "did not load: " + (self.first_error_line() or "no error line in server.log")}
            elif worker_gb < 0.1:
                tp = {"outcome": "b", "detail": f"loaded with all buffers local ({by_host}); RPC device ignored in tensor mode"}
            else:
                meta = re.findall(r"Meta\(\) model buffer size\s*=\s*([\d.]+) MiB", text)
                tp = {"outcome": "a", "detail": f"both sides hold weights: {by_host} GB per host"
                      + (f"; Meta() model buffer {meta[0]} MiB per device, {len(meta_hosts)} devices" if meta else "")}
            if "Meta device" in text or "creating a Meta" in text:
                tp["meta_device"] = True
            fa = re.search(r"enabling flash_attn since it is required for SPLIT_MODE_TENSOR", text)
            tp["flash_attn_forced"] = bool(fa)
        return {
            "tp_outcome": tp,
            "engine": {"name": self.name, "commit": self.commit, "flags": list(self.flags)},
            "setup": {"load_total_s": self.load_s, "transfer_s": transfer_s, "cache_state": cache_state,
                      "bytes_sent_gb": (bytes_sent / 1e9) if worker_texts else None},
            "setup_notes": transfer_notes,
            "memory_buffers_gb": {h: round(v, 3) for h, v in by_host.items()},
            "generation": dict(GENERATION),
            "worker_logs": worker_texts,
            "placement_lines": placement,
            "model_sha256": self.model_sha,
            "draft_sha256": self.draft_sha,
            "oom_in_log": bool(OOM.search(text)),
            "net_err_in_log": bool(NET_ERR.search(text)),
        }
