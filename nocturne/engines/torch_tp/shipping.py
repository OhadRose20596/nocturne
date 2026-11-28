"""Weight shipping (session 10, part C): the coordinator sends every rank the slices it computes with, so only the
coordinator host holds the model file. docs/ENGINE_DESIGN.md section 3.8; the frames are documented in transport.py.
A slice is the part of one tensor a rank holds (a row range or a column range from the plan). Its bytes are the bytes
the loader reads from the file: stored quantization blocks, or raw values of a dense tensor. A row split layer of a
quantized file is cut only at block boundaries (the planner guarantees it, the source refuses anything else).
Coordinator: build_manifest (names, ranges, type, bytes, sha256 per slice), send_slices (chunks with a CRC32 each).
Rank: receive_slices checks its cache (<cache>/<model sha256>/<slice sha256>), asks for what is missing, writes each
slice to a temporary file while hashing it, and renames it into the cache only when the sha256 matches. CacheSource
then serves the loader from memory mapped cache files, so a slice is never held twice in memory.
Security: the manifest comes from whoever connected (ADR-009, deferred). Hashes are checked to be 64 hex digits before
they name a file."""
import hashlib
import json
import os
import re
import shutil
import threading
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch

from . import transport
from .loader import LoadSpec, _take, _wanted, weight_files

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")     # cache files are mapped read only

CHUNK = 4 << 20
OP_MANIFEST, OP_NEED, OP_CHUNK, OP_END, OP_ACK, OP_DONE = range(1, 7)
OP_NAMES = {OP_MANIFEST: "manifest", OP_NEED: "need", OP_CHUNK: "chunk", OP_END: "end", OP_ACK: "ack", OP_DONE: "done"}
HEX64 = re.compile(r"[0-9a-f]{64}")
CODE_FILES = ("shipping.py", "gguf_source.py", "loader.py", "shard.py", "quant.py", "arch.py")
DEFAULT_CACHE = "~/.nocturne/cache"
DISK_MARGIN = 64 << 20
NP_DTYPE = {"torch.uint8": np.uint8, "torch.bfloat16": np.int16, "torch.float16": np.float16, "torch.float32": np.float32,
            "torch.int8": np.int8, "torch.int16": np.int16, "torch.int32": np.int32, "torch.int64": np.int64}


class ShipRejected(RuntimeError):
    pass


def cache_root(path: str | None = None) -> Path:
    return Path(os.path.expanduser(path or os.environ.get("NOCTURNE_CACHE") or DEFAULT_CACHE))


# ---- model identity ------------------------------------------------------------------------------------------------------
def file_sha256(path: Path, memo_file: Path | None = None) -> str:
    """sha256 of a file, remembered by (size, mtime) in memo_file."""
    st = path.stat()
    key, stamp = str(path.resolve()), [st.st_size, st.st_mtime_ns]
    memo = {}
    if memo_file and memo_file.exists():
        try:
            memo = json.loads(memo_file.read_text())
        except ValueError:
            memo = {}
    if memo.get(key, {}).get("stamp") == stamp:
        return memo[key]["sha256"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    if memo_file:
        memo[key] = {"stamp": stamp, "sha256": h.hexdigest()}
        memo_file.parent.mkdir(parents=True, exist_ok=True)
        memo_file.write_text(json.dumps(memo, indent=1, sort_keys=True))
    return h.hexdigest()


def model_id(sums: dict[str, str]) -> str:
    """One id for a model from {file name: sha256}: the file's own sha256 for one file, else a sha256 over the sorted list."""
    if len(sums) == 1:
        return next(iter(sums.values()))
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(sums.items())).encode()).hexdigest()


def model_files(model_path: str | Path) -> list[Path]:
    p = Path(model_path)
    if p.suffix == ".gguf":
        return [p]
    return sorted(f for f in p.iterdir() if f.suffix in (".json", ".safetensors"))


def model_identity(model_path: str | Path, memo_file: Path | None = None) -> tuple[str, dict[str, str]]:
    """(model sha256, {file name: sha256}) of a .gguf file or of the .json and .safetensors files of a directory."""
    sums = {f.name: file_sha256(f, memo_file) for f in model_files(model_path)}
    return model_id(sums), sums


# ---- sources on the coordinator ----------------------------------------------------------------------------------------------
class SafetensorsSource:
    """A safetensors directory as a TensorSource: read() gives the slice in the file's dtype, as the local loader takes it."""
    kind = "safetensors"

    def __init__(self, model_dir: str | Path):
        self.files = weight_files(Path(model_dir))

    def meta(self) -> dict:
        return self.files

    def read(self, name: str, rows: slice | None = None, cols: slice | None = None) -> tuple[torch.Tensor, str]:
        from safetensors import safe_open
        spec = (0, rows) if rows is not None else ((1, cols) if cols is not None else None)
        with safe_open(str(self.files[name]), framework="pt", device="cpu") as sf:
            t = _take(sf, name, spec)
        return t, str(t.dtype).replace("torch.", "")


def open_source(model_path: str | Path):
    p = Path(model_path)
    if p.suffix == ".gguf":
        from .gguf_source import open_gguf
        return open_gguf(p)
    return SafetensorsSource(p)


def load_spec_of(spec, cfg) -> LoadSpec:
    """What a rank loads, from its RankSpec (the same LoadSpec the rank builds for itself)."""
    from .plan import Dims
    return LoadSpec(tuple(spec.layers), spec.slice_spec(Dims.of(cfg)), spec.embed_rows, spec.head_rows, spec.final_norm)


# ---- manifest ----------------------------------------------------------------------------------------------------------------
def _range(s: slice | None) -> list[int] | None:
    return None if s is None else [int(s.start or 0), int(s.stop)]


def slice_key(src: str, rows, cols) -> str:
    return f"{src}|{rows}|{cols}"


def split_spec(sl) -> tuple[slice | None, slice | None]:
    """(rows, cols) of a loader slice spec (dim, slice) | None."""
    if sl is None:
        return None, None
    dim, s = sl
    return (s, None) if dim == 0 else (None, s)


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """The bytes of a host tensor, without a copy when it is contiguous."""
    t = t.contiguous()
    if t.numel() == 0:
        return memoryview(b"")
    a = t.view(torch.int16).numpy() if t.dtype == torch.bfloat16 else t.numpy()
    return memoryview(a).cast("B")


def _sha(mv: memoryview) -> str:
    h = hashlib.sha256()
    for off in range(0, len(mv), 8 << 20):
        h.update(mv[off:off + (8 << 20)])
    return h.hexdigest()


def build_manifest(source, cfg, load_spec: LoadSpec, model_sha: str, memo: dict | None = None) -> dict:
    """Every slice one rank loads: engine name, source tensor, row and column range, type, shape, bytes, sha256.
    `memo` ({slice key: entry}) remembers hashed slices of this model across ranks and runs."""
    want = _wanted(source.meta().keys(), cfg, load_spec)
    slices, total = [], 0
    for out_name in sorted(want):
        src, sl = want[out_name]
        rows, cols = split_spec(sl)
        r, c = _range(rows), _range(cols)
        key = slice_key(src, r, c)
        e = memo.get(key) if memo is not None else None
        if e is None:
            t, qtype = source.read(src, rows, cols)
            mv = tensor_bytes(t)
            e = {"qtype": qtype, "dtype": str(t.dtype), "shape": list(t.shape), "nbytes": len(mv), "sha256": _sha(mv)}
            if memo is not None:
                memo[key] = e
        slices.append({"name": out_name, "src": src, "rows": r, "cols": c, **e})
        total += e["nbytes"]
    return {"model_sha256": model_sha, "kind": source.kind, "slices": slices, "total_bytes": total}


def code_stamp() -> str:
    """sha256 over the files that decide which bytes a slice key means. A remembered hash is valid only for this code."""
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode() + Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def load_memo(root: Path, model_sha: str) -> tuple[dict, Path]:
    """{slice key: entry} remembered for this model, empty when it was written by other slicing code."""
    p = root / model_sha / "slice_hashes.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        d = {}
    return (d.get("slices") or {} if d.get("code") == code_stamp() else {}), p


def save_memo(memo: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"code": code_stamp(), "slices": memo}))
    os.replace(tmp, path)


# ---- coordinator: send -----------------------------------------------------------------------------------------------------------
def send_slices(link: transport.Link, source, manifest: dict, rank: str = "") -> dict:
    """Send the manifest, read the need list, stream every needed slice in chunks, wait for the ack of each one."""
    t0 = time.time()
    slices = manifest["slices"]
    link.send(transport.KIND_WEIGHTS, json.dumps(manifest).encode(), "uint8", aux0=OP_MANIFEST)
    kind, payload, _, _, aux0, _ = link.recv()
    if kind == transport.KIND_ERROR:
        raise ShipRejected(f"{rank}: {bytes(payload).decode(errors='replace')}")
    if kind != transport.KIND_WEIGHTS or aux0 & 0xFF != OP_NEED:
        raise ShipRejected(f"{rank}: expected the need list, got {transport.KIND_NAMES.get(kind, kind)}")
    need = json.loads(bytes(payload).decode())
    if any(not isinstance(i, int) or not 0 <= i < len(slices) for i in need):
        raise ShipRejected(f"{rank}: need list names a slice outside the manifest")
    shipped = chunks = 0
    for i in need:
        sl = slices[i]
        t, _ = source.read(sl["src"], None if sl["rows"] is None else slice(*sl["rows"]), None if sl["cols"] is None else slice(*sl["cols"]))
        mv = tensor_bytes(t)
        if len(mv) != sl["nbytes"]:
            raise ShipRejected(f"{rank}: slice {sl['name']} is {len(mv)} bytes now, the manifest says {sl['nbytes']} (the model file changed)")
        for off in range(0, len(mv), CHUNK):
            chunk = mv[off:off + CHUNK]
            link.send(transport.KIND_WEIGHTS, chunk, "uint8", aux0=OP_CHUNK | i << 8, aux1=zlib.crc32(chunk))
            chunks += 1
        link.send(transport.KIND_WEIGHTS, aux0=OP_END | i << 8)
        kind, payload, _, _, aux0, aux1 = link.recv()
        if kind == transport.KIND_ERROR:
            raise ShipRejected(f"{rank}: {bytes(payload).decode(errors='replace')}")
        if kind != transport.KIND_WEIGHTS or aux0 & 0xFF != OP_ACK or aux0 >> 8 != i:
            raise ShipRejected(f"{rank}: expected the ack of slice {sl['name']}, got {transport.KIND_NAMES.get(kind, kind)}")
        if aux1 != 0:
            raise ShipRejected(f"{rank}: slice {sl['name']} rejected: {bytes(payload).decode(errors='replace')}")
        shipped += sl["nbytes"]
    link.send(transport.KIND_WEIGHTS, aux0=OP_DONE)
    return {"manifest_bytes": manifest["total_bytes"], "manifest_slices": len(slices), "shipped_bytes": shipped, "shipped_slices": len(need),
            "chunks": chunks, "ship_s": time.time() - t0}


# ---- rank: receive ---------------------------------------------------------------------------------------------------------------
def check_manifest(m: dict) -> None:
    if not HEX64.fullmatch(str(m.get("model_sha256"))):
        raise ShipRejected("manifest: the model sha256 is not 64 hex digits")
    for sl in m["slices"]:
        if not HEX64.fullmatch(str(sl.get("sha256"))) or not isinstance(sl.get("nbytes"), int) or sl["nbytes"] < 0:
            raise ShipRejected(f"manifest: bad sha256 or size for slice {sl.get('name')}")


def unique_bytes(m: dict) -> int:
    """Bytes of the manifest with slices of identical content counted once (they share one cache file)."""
    return sum({s["sha256"]: s["nbytes"] for s in m["slices"]}.values())


def cached(d: Path, sl: dict) -> bool:
    p = d / sl["sha256"]
    return p.is_file() and p.stat().st_size == sl["nbytes"]


def receive_slices(link: transport.Link, root: Path, log=print) -> tuple[dict, dict]:
    """Rank side. Returns (manifest, stats). Raises ShipRejected after answering a bad slice with a rejecting ack."""
    t0 = time.time()
    kind, payload, _, _, aux0, _ = link.recv()
    if kind != transport.KIND_WEIGHTS or aux0 & 0xFF != OP_MANIFEST:
        raise ShipRejected(f"expected the weights manifest, got {transport.KIND_NAMES.get(kind, kind)}")
    m = json.loads(bytes(payload).decode())
    check_manifest(m)
    slices = m["slices"]
    d = root / m["model_sha256"]
    d.mkdir(parents=True, exist_ok=True)
    need, seen = [], set()
    for i, sl in enumerate(slices):
        if sl["sha256"] not in seen and not cached(d, sl):
            need.append(i)
        seen.add(sl["sha256"])
    missing = sum(slices[i]["nbytes"] for i in need)
    free = shutil.disk_usage(d).free
    if missing + DISK_MARGIN > free:
        raise ShipRejected(f"cache {d} has {free / 1e9:.2f} GB free, the missing slices need {missing / 1e9:.2f} GB")
    link.send(transport.KIND_WEIGHTS, json.dumps(need).encode(), "uint8", aux0=OP_NEED)
    received = chunks = checks = 0
    accepted: set[int] = set()
    cur = None          # the slice being received: index, temporary file, sha256, bytes, first error
    try:
        while True:
            kind, payload, _, _, aux0, aux1 = link.recv()
            if kind != transport.KIND_WEIGHTS:
                raise ShipRejected(f"expected a weights frame, got {transport.KIND_NAMES.get(kind, kind)}")
            op, i = aux0 & 0xFF, aux0 >> 8
            if op == OP_DONE:
                missing_now = [slices[k]["name"] for k in need if k not in accepted]
                if cur is not None or missing_now:
                    raise ShipRejected(f"the transfer ended before {len(missing_now)} needed slices arrived (first: {missing_now[0]})")
                break
            if op not in (OP_CHUNK, OP_END) or i not in need or i in accepted:
                raise ShipRejected(f"unexpected weights frame {OP_NAMES.get(op, op)} for slice {i}")
            sl = slices[i]
            if cur is None:
                tmp = d / f"{sl['sha256']}.part{os.getpid()}_{threading.get_ident()}"      # two ranks of one host may fetch the same slice
                cur = {"i": i, "tmp": tmp, "f": tmp.open("wb"), "sha": hashlib.sha256(), "n": 0, "err": None}
            elif cur["i"] != i:
                cur["err"] = cur["err"] or f"a frame of slice {i} arrived inside slice {cur['i']}"
            if op == OP_CHUNK:
                chunks += 1
                if zlib.crc32(payload) != aux1:
                    cur["err"] = cur["err"] or f"chunk {chunks} failed its CRC32"
                elif not cur["err"]:
                    cur["f"].write(payload)
                    cur["sha"].update(payload)
                    cur["n"] += len(payload)
                continue
            checks += 1
            err = cur["err"]
            if not err and cur["n"] != sl["nbytes"]:
                err = f"{cur['n']} bytes arrived, the manifest says {sl['nbytes']}"
            if not err and cur["sha"].hexdigest() != sl["sha256"]:
                err = "sha256 differs from the manifest"
            if err:
                link.send(transport.KIND_WEIGHTS, err.encode(), "uint8", aux0=OP_ACK | i << 8, aux1=1)
                raise ShipRejected(f"slice {sl['name']} rejected: {err}")
            cur["f"].flush()
            os.fsync(cur["f"].fileno())          # the file gets its cache name only when its bytes are on the disk
            cur["f"].close()
            os.replace(cur["tmp"], d / sl["sha256"])
            received += cur["n"]
            accepted.add(i)
            cur = None
            link.send(transport.KIND_WEIGHTS, aux0=OP_ACK | i << 8, aux1=0)
    finally:
        if cur is not None:      # any abnormal end: a rejected slice, a lost link, a full disk
            try:
                cur["f"].close()
            except OSError:
                pass
            cur["tmp"].unlink(missing_ok=True)
    total, unique = m["total_bytes"], unique_bytes(m)
    stats = {"manifest_bytes": total, "manifest_slices": len(slices), "received_bytes": received, "received_slices": len(accepted),
             "cache_bytes": unique - received, "duplicate_bytes": total - unique, "hash_checks": checks, "chunks": chunks,
             "recv_s": time.time() - t0, "cache_dir": str(d)}
    log(f"[ship] {received / 1e9:.3f} GB received in {len(need)} slices, {(unique - received) / 1e9:.3f} GB from the cache, {stats['recv_s']:.1f} s")
    return m, stats


# ---- rank: load from the cache -------------------------------------------------------------------------------------------------
class CacheSource:
    """The slices of a manifest as a TensorSource. read() takes exactly the ranges of the manifest: the slice is already cut."""

    def __init__(self, manifest: dict, root: Path):
        self.dir = root / manifest["model_sha256"]
        self.kind = manifest["kind"]
        self.plain = self.kind == "safetensors"
        self.by_key = {slice_key(s["src"], s["rows"], s["cols"]): s for s in manifest["slices"]}
        self.names = {s["src"] for s in manifest["slices"]}

    def meta(self) -> dict:
        return dict.fromkeys(self.names)

    def read(self, name: str, rows: slice | None = None, cols: slice | None = None) -> tuple[torch.Tensor, str]:
        key = slice_key(name, _range(rows), _range(cols))
        sl = self.by_key.get(key)
        if sl is None:
            raise KeyError(f"slice {key} is not in the shipped manifest")
        dt = sl["dtype"]
        if sl["nbytes"] == 0:
            t = torch.empty(sl["shape"], dtype=getattr(torch, dt.replace("torch.", "")))
            return t, sl["qtype"]
        a = np.memmap(self.dir / sl["sha256"], dtype=NP_DTYPE[dt], mode="r").reshape(sl["shape"])
        t = torch.from_numpy(a)
        return (t.view(torch.bfloat16) if dt == "torch.bfloat16" else t), sl["qtype"]


# ---- coordinator: all ranks at once ------------------------------------------------------------------------------------------------
def ship_all(links: dict, source, manifests: dict, names: dict) -> tuple[dict, list[str], list[threading.Thread]]:
    """One thread per ctl link, started here. Returns ({rank name: stats}, [error text], threads); the first two fill
    in while the threads run, so read them after joining."""
    stats, errors, lock = {}, [], threading.Lock()

    def one(gid):
        try:
            s = send_slices(links[gid], source, manifests[gid], names[gid])
            with lock:
                stats[names[gid]] = s
        except Exception as e:      # noqa: BLE001 - reported with the rank's name by the caller
            text = str(e) if isinstance(e, ShipRejected) else f"{names[gid]}: {type(e).__name__}: {e}"
            if not isinstance(e, ShipRejected):      # a rank that failed on its own (disk full) sent its reason before it closed
                try:
                    links[gid].sock.settimeout(3.0)
                    while True:
                        kind, payload, *_ = links[gid].recv()
                        if kind == transport.KIND_ERROR:
                            text += f" (the rank said: {bytes(payload).decode(errors='replace')})"
                            break
                except Exception:      # noqa: BLE001 - the first error text is what matters
                    pass
            with lock:
                errors.append(text)

    threads = [threading.Thread(target=one, args=(gid,), daemon=True) for gid in manifests]
    for t in threads:
        t.start()
    return stats, errors, threads
