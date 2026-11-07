"""Adapter registry. cells.yaml names the adapter; run.py looks it up here."""
from .base import Engine, EngineUnsupported
from .distributed_llama import DistributedLlama
from .llamacpp_rpc import LlamaCppRpc
from .torch_tp import TorchTp

ADAPTERS: dict[str, type[Engine]] = {
    "llamacpp_rpc": LlamaCppRpc,
    "distributed_llama": DistributedLlama,     # row B1d (ADR-007 amendment 1, accepted 2026-10-27)
    "torch_tp": TorchTp,                       # row B1t, own PyTorch TP engine (ADR-007 amendment 2)
    # "reference": later session (row D)
}


def get(name: str) -> type[Engine]:
    if name not in ADAPTERS:
        raise EngineUnsupported(f"adapter {name!r} is not implemented in this session")
    return ADAPTERS[name]
