"""The structure of every data frame of a small fixed run with every overlap option off, as one hash per placement.
Usage: python tests/bench/golden_frames.py   (run from a checkout of the commit whose frames are to be pinned; prints JSON)
Self-contained on purpose: it uses only the engine calls that exist since session 10 part 1, so the same file runs
against the engine of an older commit (cwd and PYTHONPATH decide which engine is imported). Payloads of float frames
and the head's token index depend on the CPU's arithmetic and are left out; kind, dtype, shape, payload length and the
header fields are kept, which is everything part 3 touched."""
import hashlib
import json
import socket
import sys
import tempfile
import threading
from pathlib import Path

import torch
from safetensors.torch import save_file

TINY = {"model_type": "qwen2", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "eos_token_id": 127,
        "max_position_embeddings": 256, "torch_dtype": "bfloat16"}
PROMPTS = [[5, 17, 99, 3, 42], [8, 61], [7, 7, 23, 90, 11, 4, 100], [33, 2, 64], [120, 19, 6, 55, 71, 12]]
DATA = ("allreduce", "token", "hidden", "head")


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def write_model(d: Path, seed: int = 3) -> Path:
    g = torch.Generator().manual_seed(seed)
    c = TINY
    hd = c["hidden_size"] // c["num_attention_heads"]
    r = lambda *s: torch.randn(*s, generator=g) * 0.1
    w = {"model.embed_tokens.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5,
         "model.norm.weight": torch.ones(c["hidden_size"]), "lm_head.weight": torch.randn(c["vocab_size"], c["hidden_size"], generator=g) * 0.5}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w.update({p + "input_layernorm.weight": torch.ones(c["hidden_size"]), p + "post_attention_layernorm.weight": torch.ones(c["hidden_size"]),
                  p + "self_attn.q_proj.weight": r(c["num_attention_heads"] * hd, c["hidden_size"]), p + "self_attn.q_proj.bias": r(c["num_attention_heads"] * hd),
                  p + "self_attn.k_proj.weight": r(c["num_key_value_heads"] * hd, c["hidden_size"]), p + "self_attn.k_proj.bias": r(c["num_key_value_heads"] * hd),
                  p + "self_attn.v_proj.weight": r(c["num_key_value_heads"] * hd, c["hidden_size"]), p + "self_attn.v_proj.bias": r(c["num_key_value_heads"] * hd),
                  p + "self_attn.o_proj.weight": r(c["hidden_size"], c["num_attention_heads"] * hd),
                  p + "mlp.gate_proj.weight": r(c["intermediate_size"], c["hidden_size"]), p + "mlp.up_proj.weight": r(c["intermediate_size"], c["hidden_size"]),
                  p + "mlp.down_proj.weight": r(c["hidden_size"], c["intermediate_size"])})
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(c))
    save_file({k: v.to(torch.bfloat16) for k, v in w.items()}, str(d / "model.safetensors"))
    return d


def placements(p):
    from nocturne.engines.torch_tp.plan import Placement, RankIn, StageIn
    R = lambda i: RankIn("h", "cpu", p[i])
    return {
        "pp3": Placement("m", (StageIn("s0", (0, 1), (R(0),), embed=True), StageIn("s1", (1, 3), (R(1),)),
                               StageIn("s2", (3, 4), (R(2),), lm_head=True)), max_seq_len=64, granule=16),
        "tp3": Placement("m", (StageIn("s0", (0, 4), (R(0), R(1), R(2)), embed=True, lm_head=True),), max_seq_len=64, granule=16),
        "hybrid": Placement("m", (StageIn("s0", (0, 2), (R(0), R(1)), embed=True), StageIn("s1", (2, 4), (R(2),)),
                                  StageIn("head", None, (R(3), R(4)), lm_head=True)), max_seq_len=64, granule=16),
    }


def record(model: Path, name: str, timeline: bool, n: int = 4) -> dict:
    """Run the fixed prompts one by one on the options-off path and describe every data frame that was sent."""
    from nocturne.engines.torch_tp import hybrid, transport
    from nocturne.engines.torch_tp.config import load_config
    from nocturne.engines.torch_tp.plan import Dims, plan
    placement = placements(free_ports(8))[name]
    cfg = load_config(model)
    specs = plan(Dims.of(cfg), placement)
    frames, lock = [], threading.Lock()
    real = transport.Link.send
    names = {v: k for k, v in vars(transport).items() if k.startswith("KIND_") and isinstance(v, int)}

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0, **kw):
        k = names.get(kind, "").replace("KIND_", "").lower()
        if k in DATA:
            with lock:      # the head frame's index and value are arithmetic; its slot bits (24 to 31) are structure
                frames.append((k, dtype, tuple(shape), len(payload), aux0 >> 24 if k == "head" else aux0, 0 if k == "head" else aux1))
        return real(self, kind, payload, dtype, shape, aux0, aux1, **kw)
    transport.Link.send = send
    run_dir = tempfile.mkdtemp() if timeline else None
    try:
        threads = []
        for s in specs[1:]:
            t = threading.Thread(target=hybrid.rank_main, args=(["127.0.0.1"], s.port, str(model), "cpu"),
                                 kwargs={"log": lambda *a: None, "run_dir": run_dir}, daemon=True)
            t.start()
            threads.append(t)
        co = hybrid.Coordinator(specs, str(model), {"h": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                                placement.prefill_max, device="cpu", log=lambda *a: None, dtype="float32", timeline=timeline, run_dir=run_dir)
        try:
            for i, p in enumerate(PROMPTS):
                co.generate(list(p), n, ignore_eos=True, first_logits=True, rid=f"p{i}")
            m = co.metrics()
        finally:
            co.shutdown()
        for t in threads:
            t.join(timeout=10)
    finally:
        transport.Link.send = real
    per_link = {r: {l: {k: v for k, v in c["by_kind_sent"].items() if k in DATA} for l, c in x["links"].items()} for r, x in m["ranks"].items()}
    body = json.dumps([sorted(frames, key=repr), per_link], sort_keys=True, default=list)
    return {"frames": len(frames), "sha256": hashlib.sha256(body.encode()).hexdigest()}


def all_records(model: Path) -> dict:
    return {f"{name}{'_timeline' if tl else ''}": record(model, name, tl) for name in ("pp3", "tp3", "hybrid") for tl in (False, True)}


if __name__ == "__main__":
    sys.path.insert(0, ".")
    print(json.dumps(all_records(write_model(Path(tempfile.mkdtemp()) / "t4")), indent=1))
