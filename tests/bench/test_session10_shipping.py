"""Session 10, part C: the coordinator ships every rank its slices. Manifest, chunks with a CRC32, the slice cache, loading
through TensorSource, and the end to end loopback with ranks that have no model file (safetensors and GGUF)."""
import hashlib
import json
import threading
import zlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from nocturne.engines.torch_tp import shipping, transport  # noqa: E402
from nocturne.engines.torch_tp.config import config_from_json, config_to_json, load_config  # noqa: E402
from nocturne.engines.torch_tp.loader import load_from_source, load_slices  # noqa: E402
from nocturne.engines.torch_tp.plan import Dims, Placement, RankIn, StageIn, plan  # noqa: E402
from nocturne.engines.torch_tp.quant import BLOCK, DENSE, QuantLinear  # noqa: E402

from . import gguf_tiny  # noqa: E402
from .test_hybrid import free_ports  # noqa: E402
from .test_session10_timeline import tcp_pair  # noqa: E402
from .test_torch_tp import write_tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_model(tmp_path_factory.mktemp("s10c") / "t")


def tp_pp(pts, granule=16, heads=None):
    r = lambda i, h=None: RankIn("h", "cpu", pts[i], heads=h)
    hs = heads or (None, None)
    return Placement("m", (StageIn("s0", (0, 1), (r(0, hs[0]), r(1, hs[1])), embed=True),
                           StageIn("s1", (1, 2), (r(2),), lm_head=True)), max_seq_len=64, granule=granule)


def run(model, placement, cache=None, ship=False, dtype="float32", prompts=((5, 17, 99, 3), (42, 8)), n_tokens=5, hook=None):
    """Loopback run in threads. With ship the ranks get no model path, only a cache directory."""
    from nocturne.engines.torch_tp.gguf_source import weight_meta
    from nocturne.engines.torch_tp.hybrid import Coordinator, rank_main
    cfg = load_config(model)
    specs = plan(Dims.of(cfg, weight_meta(str(model))), placement)
    threads = []
    for s in specs[1:]:
        t = threading.Thread(target=rank_main, args=(["127.0.0.1"], s.port, None if ship else str(model), "cpu"),
                             kwargs={"log": lambda *a: None, "cache_dir": str(cache) if cache else None}, daemon=True)
        t.start()
        threads.append(t)
    if hook:
        hook()
    co = Coordinator(specs, str(model), {"h": "127.0.0.1"}, ["127.0.0.1"], placement.max_seq_len, placement.reduce_dtype,
                     placement.prefill_max, device="cpu", log=lambda *a: None, dtype=dtype, ship_weights=ship,
                     cache_dir=str(cache) if cache else None)
    try:
        out = [co.generate(list(p), n_tokens, ignore_eos=True, first_logits=True, rid=f"p{i}") for i, p in enumerate(prompts)]
        m = co.metrics()
    finally:
        co.shutdown()
    for t in threads:
        t.join(timeout=10)
    return specs, out, co, m


def same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x["tokens"] == y["tokens"]
        assert torch.equal(x["first_logits"], y["first_logits"])


# ---- identity and manifest ------------------------------------------------------------------------------------------------
def test_model_identity_of_a_file_and_of_a_directory(tmp_path, tiny):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"abc" * 1000)
    sha, sums = shipping.model_identity(f, tmp_path / "memo.json")
    assert sha == hashlib.sha256(b"abc" * 1000).hexdigest() and sums == {"m.gguf": sha}
    assert json.loads((tmp_path / "memo.json").read_text())[str(f.resolve())]["sha256"] == sha
    f.write_bytes(b"abd" * 1001)                           # a changed file (another size or mtime) is hashed again
    assert shipping.model_identity(f, tmp_path / "memo.json")[0] == hashlib.sha256(b"abd" * 1001).hexdigest() != sha
    dsha, dsums = shipping.model_identity(tiny)
    assert set(dsums) == {"config.json", "model.safetensors"} and dsha == shipping.model_id(dsums) != dsums["model.safetensors"]
    assert shipping.model_id({"b": "2", "a": "1"}) == hashlib.sha256(b"a 1\nb 2").hexdigest()


def test_config_survives_the_plan_message(tiny):
    for cfg in (load_config(tiny),):
        assert config_from_json(json.loads(json.dumps(config_to_json(cfg)))) == cfg
    if gguf_tiny.available():
        cfg = load_config(gguf_tiny.build("llama")["q8_0"])
        back = config_from_json(json.loads(json.dumps(config_to_json(cfg))))
        assert back == cfg and isinstance(back.rope_factors, tuple) and isinstance(back.eos_token_ids, tuple)


def manifests_for(model, placement):
    from nocturne.engines.torch_tp.gguf_source import weight_meta
    cfg = load_config(model)
    specs = plan(Dims.of(cfg, weight_meta(str(model))), placement)
    src = shipping.open_source(model)
    memo = {}
    return cfg, specs, src, [shipping.build_manifest(src, cfg, shipping.load_spec_of(s, cfg), "0" * 64, memo) for s in specs], memo


def test_manifest_lists_exactly_what_the_rank_loads_with_hashes_of_the_slices(tiny):
    cfg, specs, src, ms, memo = manifests_for(tiny, tp_pp(free_ports(3)))
    for spec, m in zip(specs, ms):
        w, loaded = load_slices(tiny, cfg, shipping.load_spec_of(spec, cfg), "cpu", dtype=torch.bfloat16, log=lambda *a: None)
        names = {s["name"] for s in m["slices"]}
        assert names == {k for k in w if not (k == "lm_head.weight" and "lm_head.weight" not in names)}
        assert m["total_bytes"] == sum(s["nbytes"] for s in m["slices"]) == loaded      # bf16 file, bf16 load: the same bytes
        assert m["total_bytes"] == spec.planned["weights"]                                 # shipped bytes are the planned bytes
        for s in m["slices"]:
            t, _ = src.read(s["src"], None if s["rows"] is None else slice(*s["rows"]), None if s["cols"] is None else slice(*s["cols"]))
            assert hashlib.sha256(bytes(shipping.tensor_bytes(t))).hexdigest() == s["sha256"] and list(t.shape) == s["shape"]
    q = next(s for s in ms[1]["slices"] if s["name"].endswith("layers.0.self_attn.q_proj.weight"))
    o = next(s for s in ms[1]["slices"] if s["name"].endswith("layers.0.self_attn.o_proj.weight"))
    assert q["rows"] == [specs[1].q0 * 16, specs[1].q1 * 16] and q["cols"] is None and o["cols"] == q["rows"] and o["rows"] is None
    assert len(memo) < sum(len(m["slices"]) for m in ms) or len(ms) == 1          # replicated tensors (norms) are hashed once


def test_manifest_slice_bounds_follow_the_plan_for_every_rank(tiny):
    """The slices of a group tile each tensor: the row or column ranges of the ranks are adjacent and cover it."""
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3), heads=(3, 1)))
    grp = [m for s, m in zip(specs, ms) if s.stage == 0]
    for name, full in (("model.layers.0.self_attn.q_proj.weight", 64), ("model.layers.0.mlp.gate_proj.weight", 128),
                       ("model.embed_tokens.weight", 128)):
        rngs = sorted(next(s["rows"] for s in m["slices"] if s["name"] == name) for m in grp)
        assert rngs[0][0] == 0 and rngs[-1][1] == full and all(a[1] == b[0] for a, b in zip(rngs, rngs[1:])), name
    assert sorted(next(s["rows"] for s in m["slices"] if s["name"].endswith("0.self_attn.q_proj.weight")) for m in grp) == [[0, 48], [48, 64]]
    cols = sorted(next(s["cols"] for s in m["slices"] if s["name"].endswith("0.mlp.down_proj.weight")) for m in grp)
    assert cols[0][0] == 0 and cols[-1][1] == 128 and cols[0][1] == cols[1][0]


# ---- the wire: chunks, CRC32, sha256, cache ----------------------------------------------------------------------------------
def ship_pair(src, manifest, root, expect_fail=False, verify=False):
    a, b = tcp_pair()
    res = {}

    def recv():
        try:
            res["m"], res["stats"] = shipping.receive_slices(b, root, lambda *x: None, verify=verify)
        except Exception as e:      # noqa: BLE001
            res["err"] = e
    t = threading.Thread(target=recv, daemon=True)
    t.start()
    try:
        res["sent"] = shipping.send_slices(a, src, manifest, "s0/1")
    except shipping.ShipRejected as e:
        res["send_err"] = e
    t.join(timeout=10)
    a.close(); b.close()
    return res


def test_slices_arrive_in_the_cache_and_a_second_transfer_ships_nothing(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(shipping, "CHUNK", 1000)                 # many chunks per slice
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    m = ms[1]
    r = ship_pair(src, m, tmp_path)
    uniq = shipping.unique_bytes(m)                           # the two norm vectors of a layer are both ones: one cache file
    assert r["sent"]["shipped_bytes"] == uniq == r["stats"]["received_bytes"] and r["stats"]["cache_bytes"] == 0
    assert r["stats"]["duplicate_bytes"] == m["total_bytes"] - uniq == 128
    assert r["sent"]["chunks"] == r["stats"]["chunks"] > len(m["slices"]) and r["stats"]["hash_checks"] == r["stats"]["received_slices"]
    d = tmp_path / m["model_sha256"]
    for s in m["slices"]:
        assert hashlib.sha256((d / s["sha256"]).read_bytes()).hexdigest() == s["sha256"]
    assert not list(d.glob("*.part*"))
    r2 = ship_pair(src, m, tmp_path)
    assert r2["sent"]["shipped_bytes"] == 0 and r2["sent"]["chunks"] == 0 and r2["stats"]["cache_bytes"] == uniq
    r3 = ship_pair(src, ms[2], tmp_path)                         # another rank of the same model: only the ones vectors are there already
    assert r3["sent"]["shipped_bytes"] == shipping.unique_bytes(ms[2]) - 128 and r3["stats"]["cache_bytes"] == 128


def test_a_truncated_cache_file_is_requested_again(tiny, tmp_path):
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    m = ms[1]
    ship_pair(src, m, tmp_path)
    big = max(m["slices"], key=lambda s: s["nbytes"])
    p = tmp_path / m["model_sha256"] / big["sha256"]
    p.write_bytes(p.read_bytes()[:-1])
    r = ship_pair(src, m, tmp_path)
    assert r["sent"]["shipped_slices"] == 1 and r["sent"]["shipped_bytes"] == big["nbytes"]


def test_a_column_slice_is_never_copied_whole(tiny):
    """Codex F3: a column range of a safetensors matrix is not contiguous; hashing and sending must not copy all of it."""
    import numpy as np
    src = shipping.SafetensorsSource(tiny)
    t, _ = src.read("model.layers.0.mlp.down_proj.weight", None, slice(0, 64))
    assert not t.is_contiguous() and tuple(t.shape) == (64, 64)
    whole = bytes(shipping.tensor_bytes(t))
    pieces = list(shipping.iter_bytes(t, 1000))
    assert b"".join(bytes(p) for p in pieces) == whole and shipping.tensor_nbytes(t) == len(whole) == 64 * 64 * 2
    assert max(p.obj.nbytes for p in pieces) <= 1000 < len(whole)             # every piece is backed by a few rows only
    assert shipping._sha(t) == hashlib.sha256(whole).hexdigest()
    c = t.contiguous()
    first = next(shipping.iter_bytes(c, 1000))
    assert np.frombuffer(first, dtype=np.uint8).__array_interface__["data"][0] == c.data_ptr()      # contiguous: no copy at all
    one_row = list(shipping.iter_bytes(t, 50))                                # a piece limit below one row still tiles the bytes
    assert b"".join(bytes(p) for p in one_row) == whole and max(len(p) for p in one_row) <= 50
    assert list(shipping.iter_bytes(t[:0])) == []


def test_a_changed_cache_file_is_trusted_by_name_and_size_and_caught_with_verify(tiny, tmp_path):
    """Codex: the cache trusts name and size. `--verify-cache` reads and hashes every cached slice instead."""
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    m = ms[1]
    ship_pair(src, m, tmp_path)
    big = max(m["slices"], key=lambda s: s["nbytes"])
    p = tmp_path / m["model_sha256"] / big["sha256"]
    data = bytearray(p.read_bytes())
    data[3] ^= 1
    p.write_bytes(bytes(data))                                                 # same name, same size, other content
    r = ship_pair(src, m, tmp_path)
    assert r["sent"]["shipped_bytes"] == 0 and r["stats"]["cache_verified_slices"] == 0       # the documented limit
    r = ship_pair(src, m, tmp_path, verify=True)
    assert r["sent"]["shipped_slices"] == 1 and r["sent"]["shipped_bytes"] == big["nbytes"]
    assert r["stats"]["cache_verified_slices"] == len({s["sha256"] for s in m["slices"]}) - 1
    assert hashlib.sha256(p.read_bytes()).hexdigest() == big["sha256"]
    r = ship_pair(src, m, tmp_path, verify=True)
    assert r["sent"]["shipped_bytes"] == 0 and r["stats"]["cache_verified_slices"] == len({s["sha256"] for s in m["slices"]})


def corrupt_first_chunk(monkeypatch, flip_crc=False):
    """The sender's frames pass through this: the first weights chunk is changed after its CRC32 was computed."""
    real = transport.Link.send
    state = {"done": False}

    def send(self, kind, payload=b"", dtype=None, shape=(), aux0=0, aux1=0):
        if kind == transport.KIND_WEIGHTS and aux0 & 0xFF == shipping.OP_CHUNK and not state["done"] and len(payload) > 8:
            state["done"] = True
            if flip_crc:
                aux1 ^= 1
            else:
                payload = bytearray(payload)
                payload[5] ^= 0x40
                payload = bytes(payload)
        return real(self, kind, payload, dtype, shape, aux0, aux1)
    monkeypatch.setattr(transport.Link, "send", send)
    return state


@pytest.mark.parametrize("flip_crc", [False, True])
def test_a_corrupted_chunk_is_rejected_and_nothing_enters_the_cache(tiny, tmp_path, monkeypatch, flip_crc):
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    state = corrupt_first_chunk(monkeypatch, flip_crc)
    r = ship_pair(src, ms[1], tmp_path)
    assert state["done"]
    assert "rejected" in str(r["send_err"]) and "CRC32" in str(r["send_err"]) and "s0/1" in str(r["send_err"])
    assert isinstance(r["err"], shipping.ShipRejected) and "CRC32" in str(r["err"])
    assert list((tmp_path / ms[1]["model_sha256"]).iterdir()) == []


def test_a_slice_whose_sha256_differs_from_the_manifest_is_rejected(tiny, tmp_path):
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    m = json.loads(json.dumps(ms[1]))
    m["slices"][0]["sha256"] = "f" * 64                          # every CRC32 is right, the content is not what was promised
    r = ship_pair(src, m, tmp_path)
    assert "sha256 differs from the manifest" in str(r["send_err"]) and isinstance(r["err"], shipping.ShipRejected)
    assert not (tmp_path / m["model_sha256"] / ("f" * 64)).exists() and not list((tmp_path / m["model_sha256"]).glob("*.part*"))


def test_a_transfer_that_ends_early_is_rejected_and_leaves_no_partial_file(tiny, tmp_path):
    """Review F5 and F6: DONE before every needed slice arrived, and a link that drops inside a slice."""
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    m = ms[1]
    d = tmp_path / m["model_sha256"]
    for ending in ("done", "close"):
        a, b = tcp_pair()
        res = {}

        def recv():
            try:
                shipping.receive_slices(b, tmp_path, lambda *x: None)
            except Exception as e:      # noqa: BLE001
                res["err"] = e
        t = threading.Thread(target=recv, daemon=True)
        t.start()
        a.send(transport.KIND_WEIGHTS, json.dumps(m).encode(), "uint8", aux0=shipping.OP_MANIFEST)
        need = json.loads(bytes(a.recv()[1]).decode())
        a.send(transport.KIND_WEIGHTS, b"12345678", "uint8", aux0=shipping.OP_CHUNK | need[0] << 8, aux1=zlib.crc32(b"12345678"))
        if ending == "done":
            a.send(transport.KIND_WEIGHTS, aux0=shipping.OP_DONE)
        else:
            a.close()
        t.join(timeout=10)
        assert not t.is_alive()
        if ending == "done":
            assert isinstance(res["err"], shipping.ShipRejected) and "before" in str(res["err"])
        else:
            assert isinstance(res["err"], (ConnectionError, OSError))
        assert list(d.iterdir()) == [], ending
        b.close()
        if ending == "done":
            a.close()


def test_hash_memo_is_dropped_when_the_slicing_code_changes(tmp_path, monkeypatch):
    """Review F4: the memo maps a range to a sha256; it is only valid for the code that cut the slice."""
    memo, path = shipping.load_memo(tmp_path, "a" * 64)
    assert memo == {}
    memo["k"] = {"sha256": "b" * 64, "nbytes": 1}
    shipping.save_memo(memo, path)
    assert shipping.load_memo(tmp_path, "a" * 64)[0] == memo
    monkeypatch.setattr(shipping, "code_stamp", lambda: "another version of the slicing code")
    assert shipping.load_memo(tmp_path, "a" * 64)[0] == {}


def test_manifest_hashes_must_be_hex_before_they_name_a_file(tmp_path):
    ok = {"model_sha256": "a" * 64, "kind": "gguf", "total_bytes": 0, "slices": [{"name": "x", "sha256": "b" * 64, "nbytes": 0}]}
    shipping.check_manifest(ok)
    for bad in (dict(ok, model_sha256="../../etc"), dict(ok, slices=[{"name": "x", "sha256": "../evil", "nbytes": 0}]),
                dict(ok, model_sha256="a" * 64 + "\n"), dict(ok, slices=[{"name": "x", "sha256": "b" * 64 + "\n", "nbytes": 0}]),
                dict(ok, slices=[{"name": "x", "sha256": "b" * 64, "nbytes": -1}])):
        with pytest.raises(shipping.ShipRejected):
            shipping.check_manifest(bad)


def test_rank_refuses_when_the_cache_disk_is_too_small(tiny, tmp_path, monkeypatch):
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    monkeypatch.setattr(shipping.shutil, "disk_usage", lambda p: type("U", (), {"free": 1000})())
    a, b = tcp_pair()
    a.send(transport.KIND_WEIGHTS, json.dumps(ms[1]).encode(), "uint8", aux0=shipping.OP_MANIFEST)
    with pytest.raises(shipping.ShipRejected, match="GB free, the missing slices need"):
        shipping.receive_slices(b, tmp_path, lambda *x: None)
    a.close(); b.close()


# ---- loading from the cache ------------------------------------------------------------------------------------------------
def test_loading_from_the_cache_gives_the_tensors_of_the_local_file(tiny, tmp_path):
    cfg, specs, src, ms, _ = manifests_for(tiny, tp_pp(free_ports(3)))
    for spec, m in zip(specs, ms):
        ship_pair(src, m, tmp_path)
        ls = shipping.load_spec_of(spec, cfg)
        cs = shipping.CacheSource(m, tmp_path)
        assert cs.plain
        for dtype in (torch.bfloat16, torch.float32):
            a, na = load_slices(tiny, cfg, ls, "cpu", dtype=dtype, log=lambda *x: None)
            b, nb = load_from_source(cs, cfg, ls, "cpu", dtype, None, plain=True)
            assert na == nb and set(a) == set(b)
            for k in a:
                assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    with pytest.raises(KeyError, match="not in the shipped manifest"):
        cs.read("model.layers.0.mlp.gate_proj.weight", slice(0, 7), None)


# ---- end to end: ranks without the model file ---------------------------------------------------------------------------------
def test_ranks_without_the_model_give_the_same_tokens_and_a_second_run_ships_nothing(tiny, tmp_path):
    cache = tmp_path / "cache"
    _, local, _, _ = run(tiny, tp_pp(free_ports(3)))
    specs, shipped, co, m = run(tiny, tp_pp(free_ports(3)), cache, ship=True)
    same(local, shipped)
    sh = co.shipping
    assert sh["kind"] == "safetensors" and set(sh["ranks"]) == {"s0/1", "s1/0"} and sh["model_sha256"] == shipping.model_identity(tiny)[0]
    for s in specs[1:]:
        r = sh["ranks"][s.name]
        assert r["manifest_bytes"] == s.planned["weights"]                                 # the manifest is the planned bytes
        assert r["shipped_bytes"] == r["received_bytes"] and r["shipped_bytes"] + r["cache_bytes"] + r["duplicate_bytes"] == r["manifest_bytes"]
        assert r["duplicate_bytes"] % 128 == 0 and r["duplicate_bytes"] <= 256 and r["cache_bytes"] in (0, 128)     # the ones vectors
        assert r["hash_checks"] == r["received_slices"] and r["ship_s"] >= 0
        assert m["ranks"][s.name]["loaded_tensor_bytes"] == 2 * r["manifest_bytes"]      # bf16 file loaded as fp32 here
        assert m["ranks"][s.name]["shipping"]["received_bytes"] == r["received_bytes"]
    assert m["ranks"]["s0/0"]["shipping"] is None                                          # the coordinator reads its own file
    _, again, co2, _ = run(tiny, tp_pp(free_ports(3)), cache, ship=True)
    same(local, again)
    for r in co2.shipping["ranks"].values():
        assert r["shipped_bytes"] == 0 and r["received_bytes"] == 0 and r["hash_checks"] == 0
        assert r["cache_bytes"] + r["duplicate_bytes"] == r["manifest_bytes"]
    assert co2.shipping["slices_hashed"] == 0                                              # the coordinator's hash memo was reused


def test_another_cut_of_the_same_model_ships_only_the_slices_that_changed(tiny, tmp_path):
    cache = tmp_path / "cache"
    run(tiny, tp_pp(free_ports(3), heads=(2, 2)), cache, ship=True)
    _, local, _, _ = run(tiny, tp_pp(free_ports(3), heads=(3, 1)))
    specs, out, co, _ = run(tiny, tp_pp(free_ports(3), heads=(3, 1)), cache, ship=True)
    same(local, out)
    r = co.shipping["ranks"]
    assert 0 < r["s0/1"]["shipped_bytes"] < r["s0/1"]["manifest_bytes"]          # norms and vocab rows were already there
    assert r["s1/0"]["shipped_bytes"] == 0 and r["s1/0"]["cache_bytes"] + r["s1/0"]["duplicate_bytes"] == r["s1/0"]["manifest_bytes"]


def test_a_corrupted_chunk_ends_the_run_with_the_rank_and_slice_named(tiny, tmp_path, monkeypatch):
    from nocturne.engines.torch_tp.hybrid import RankFailed
    corrupt_first_chunk(monkeypatch)
    with pytest.raises(RankFailed, match=r"(s0/1|s1/0): slice \S+ rejected: chunk \d+ failed its CRC32"):
        run(tiny, tp_pp(free_ports(3)), tmp_path / "cache", ship=True)
    for d in (tmp_path / "cache").iterdir():
        if d.is_dir():
            assert not list(d.glob("*.part*"))


def test_a_rank_without_model_and_without_shipping_reports_it(tiny, tmp_path):
    from nocturne.engines.torch_tp.hybrid import Coordinator, RankFailed, rank_main
    pl = tp_pp(free_ports(3))
    specs = plan(Dims.of(load_config(tiny)), pl)
    for s in specs[1:]:
        threading.Thread(target=rank_main, args=(["127.0.0.1"], s.port, None, "cpu"), kwargs={"log": lambda *a: None}, daemon=True).start()
    with pytest.raises(RankFailed, match="s0/1: RuntimeError: started without --model, and the plan ships no weights"):
        Coordinator(specs, str(tiny), {"h": "127.0.0.1"}, ["127.0.0.1"], 64, "float32", 256, device="cpu", log=lambda *a: None,
                    dtype="float32", ship_weights=False)


# ---- GGUF: quantized blocks keep the block alignment rule ----------------------------------------------------------------------
gguf = pytest.mark.skipif(not gguf_tiny.available(), reason="llama.cpp converter or tokenizer files missing")


def gguf_tp(pts):
    r = lambda i: RankIn("h", "cpu", pts[i])
    return Placement("m", (StageIn("s0", (0, 2), (r(0), r(1)), embed=True, lm_head=True),), max_seq_len=64, granule=64)


@gguf
@pytest.mark.parametrize("arch,quant", [("qwen2", "q8_0"), ("qwen2", "q4_k_m"), ("llama", "q8_0"), ("qwen3", "q4_k_m"), ("llama", "bf16")])
def test_gguf_slices_ship_as_stored_blocks_and_give_the_same_tokens(tmp_path, arch, quant):
    pytest.importorskip("gguf")
    model = gguf_tiny.build(arch)[quant]
    cfg, specs, src, ms, _ = manifests_for(model, gguf_tp(free_ports(2)))
    hd = cfg.head_dim
    for spec, m in zip(specs, ms):
        assert m["kind"] == "gguf"
        one_d_extra = 0
        for s in m["slices"]:
            if len(s["shape"]) == 1:
                one_d_extra += s["nbytes"] - 2 * s["shape"][0]          # stored F32, held as bf16
                continue
            if s["qtype"] in DENSE:
                continue
            bs, ts = BLOCK[s["qtype"]]
            assert s["dtype"] == "torch.uint8" and s["shape"][2] == ts and s["nbytes"] == s["shape"][0] * s["shape"][1] * ts
            if s["cols"] is not None:                                    # a row split layer: cut on block boundaries only
                assert s["cols"][0] % bs == 0 and s["cols"][1] % bs == 0 and s["shape"][1] == (s["cols"][1] - s["cols"][0]) // bs
        _, loaded = load_slices(model, cfg, shipping.load_spec_of(spec, cfg), "cpu", dtype=torch.bfloat16, log=lambda *a: None)
        assert m["total_bytes"] - one_d_extra == loaded                  # shipped bytes are the stored bytes of what is loaded
        assert abs(m["total_bytes"] - one_d_extra - spec.planned["weights"]) <= max(64, 0.001 * spec.planned["weights"])
    cache = tmp_path / "cache"
    kw = dict(dtype="float32", prompts=((151644, 872, 198, 9707),), n_tokens=3)     # fp32: bf16 kernels on the CPU are slow
    _, local, _, _ = run(model, gguf_tp(free_ports(2)), **kw)
    _, shipped, co, m = run(model, gguf_tp(free_ports(2)), cache, ship=True, **kw)
    same(local, shipped)
    r = co.shipping["ranks"]["s0/1"]
    assert r["manifest_bytes"] == ms[1]["total_bytes"] and r["shipped_bytes"] == shipping.unique_bytes(ms[1]) == r["manifest_bytes"] - r["duplicate_bytes"]
    _, again, co2, _ = run(model, gguf_tp(free_ports(2)), cache, ship=True, **kw)
    same(local, again)
    assert co2.shipping["ranks"]["s0/1"]["shipped_bytes"] == 0


@gguf
def test_gguf_cache_source_returns_the_blocks_of_the_file(tmp_path):
    pytest.importorskip("gguf")
    model = gguf_tiny.build("llama")["q8_0"]               # llama: the rows of q and k are un-permuted before they are shipped
    cfg, specs, src, ms, _ = manifests_for(model, gguf_tp(free_ports(2)))
    ship_pair(src, ms[1], tmp_path)
    cs = shipping.CacheSource(ms[1], tmp_path)
    assert not cs.plain
    ls = shipping.load_spec_of(specs[1], cfg)
    a, na = load_slices(model, cfg, ls, "cpu", dtype=torch.bfloat16, log=lambda *x: None)
    b, nb = load_from_source(cs, cfg, ls, "cpu", torch.bfloat16, None, plain=False)
    assert na == nb and set(a) == set(b)
    for k in a:
        if isinstance(a[k], QuantLinear):
            assert a[k].qtype == b[k].qtype and torch.equal(a[k].data, b[k].data), k
        else:
            assert torch.equal(a[k], b[k]), k
