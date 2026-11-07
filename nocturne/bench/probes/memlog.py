"""Per node memory breakdown and OOM classification from the engine's own logs (pure functions, no I/O)."""
import re

DEV = (r"(RPC\d*\[[^\]]+\]|Meta\(\)|Metal(?:_Mapped)?|MTL\d+(?:_Mapped)?|CUDA\d+(?:_Host)?|Vulkan\d+|CPU(?:_Mapped)?|"
       r"CPU_AARCH64|CPU_REPACK)")
BUF_KIND = re.compile(r"^(?:[\d.]+\s+[A-Z]\s+)?\w+:\s*" + DEV + r"\s+(model|KV|compute|output) buffer size\s*=\s*([\d.]+) MiB",
                      re.MULTILINE)
KINDS = {"model": "model", "KV": "kv", "compute": "compute", "output": "output"}
KV_TOTAL = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB")
OOM = re.compile(r"out of memory|OutOfMemory|failed to allocate|kIOGPUCommandBufferCallbackErrorOutOfMemory|"
                 r"cudaErrorMemoryAllocation|ggml_metal_.*failed", re.IGNORECASE)


def ts_shares(flags: list) -> list | None:
    """-ts / --tensor-split values from an engine flag list, or None (the engine default)."""
    for i, f in enumerate(flags or []):
        if f in ("-ts", "--tensor-split") and i + 1 < len(flags):
            return [float(x) for x in flags[i + 1].split(",")]
    return None


def breakdown(text: str, requester: str, ip_to_host: dict, meta_hosts: list, shares: list | None = None) -> tuple[dict, list]:
    """{host: {model, kv, compute, output, cpu}} in GB and the matched log lines. CPU devices go to `cpu` whatever the
    kind; `output` is the logits buffer of llama_context, not the output layer (its weights sit in `model`).
    Tensor mode prints one Meta() line per kind holding the requester's share (meta_hosts[0]; checked on 1e7bcf3 against
    the worker's set_tensor bytes, 2026-11-07). The other devices get: weights by the -ts shares (equal when none),
    KV as the log's total KV minus the requester's share, compute equal to the requester's (not printed; an assumption)."""
    out: dict = {}
    lines = []
    sh = shares or [1.0] * len(meta_hosts)
    kv_total = KV_TOTAL.search(text)

    def add(host, key, mib):
        d = out.setdefault(host, {"model": 0.0, "kv": 0.0, "compute": 0.0, "output": 0.0, "cpu": 0.0})
        d[key] += mib / 1024.0

    for m in BUF_KIND.finditer(text):
        dev, kind, mib = m.group(1), m.group(2), float(m.group(3))
        lines.append(m.group(0).strip())
        if dev == "Meta()":
            add(meta_hosts[0], KINDS[kind], mib)
            rest = meta_hosts[1:]
            for i, h in enumerate(rest, start=1):
                if kind == "model":
                    add(h, "model", mib * sh[i] / sh[0])
                elif kind == "KV" and kv_total:
                    add(h, "kv", (float(kv_total.group(1)) - mib) * sh[i] / sum(sh[1:]))
                else:
                    add(h, KINDS[kind], mib)
            continue
        hosts = [requester]
        if dev.startswith("RPC"):
            hosts = [ip_to_host.get(dev[dev.index("[") + 1:-1].rsplit(":", 1)[0], dev)]
        for h in hosts:
            add(h, "cpu" if dev.startswith("CPU") else KINDS[kind], mib)
    if kv_total:
        lines.append(kv_total.group(0))
    return {h: {k: round(v, 3) for k, v in d.items()} for h, d in out.items()}, lines


def classify_oom(server_text: str, worker_texts: dict, loaded: bool) -> str | None:
    """oom_{requester,worker}_{load,compute}; None when no OOM evidence. A server line that names an RPC buffer is the
    worker failing to allocate, seen from the requester."""
    phase = "compute" if loaded else "load"
    if any(OOM.search(t) for t in worker_texts.values()):
        return f"oom_worker_{phase}"
    for line in server_text.splitlines():
        if OOM.search(line):
            return f"oom_worker_{phase}" if "RPC" in line else f"oom_requester_{phase}"
    return None
