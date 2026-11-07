"""Load cells.yaml and peers.yaml. Pure parsing plus path expansion; no network."""
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CELLS_PATH = HERE / "cells.yaml"
PEERS_PATH = HERE / "peers.yaml"
PEERS_EXAMPLE = HERE / "peers.example.yaml"
NETWORKS = ("wifi", "wired", "tailscale", "hotspot")


def load_cells(path: Path = CELLS_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def load_peers(path: Path | None = None) -> dict:
    p = path or PEERS_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Copy {PEERS_EXAMPLE.name} to peers.yaml and edit it.")
    cfg = yaml.safe_load(p.read_text())
    for m in cfg.get("models", {}).values():
        for k in ("path", "draft", "dllama_path", "dllama_tokenizer"):
            if m.get(k):
                m[k] = os.path.expanduser(m[k])
        # torch_model stays as written (~/...): every rank reads it on its own machine, and the runner turns a
        # leading ~/ into "$HOME"/ on each peer, whose home may differ from the requester's.
    for peer in cfg["peers"].values():
        for k in ("bin", "dllama_bin", "python"):
            if peer.get(k) and peer.get("ssh") is None:
                peer[k] = os.path.expanduser(peer[k])
        if str(peer.get("bind", "")).startswith("0.0.0.0"):
            raise ValueError(f"peer {peer['host']}: bind must be a specific address, never 0.0.0.0")
    cfg.setdefault("rpc_port", 50052)
    cfg.setdefault("server_port", 8089)
    cfg.setdefault("tensor_split", None)
    cfg.setdefault("torch_port", 50053)
    cfg.setdefault("torch_api_port", 8092)
    return cfg


def cell(cells: dict, name: str) -> dict:
    try:
        return cells["cells"][name]
    except KeyError:
        raise KeyError(f"unknown cell {name!r}; known: {', '.join(cells['cells'])}") from None
