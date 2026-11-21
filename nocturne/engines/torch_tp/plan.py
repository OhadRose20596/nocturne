"""Placement planner (session 6). plan(cfg, placement) -> list[RankSpec] is pure: no I/O, no torch.
A placement is a list of PP stages; each stage is a TP group. Inside a group every rank gets a contiguous range of query
heads, a contiguous range of MLP rows (in granules) and, where the stage holds the embedding or the output head, a
contiguous range of vocab rows (in granules). Shares follow `weight`, else `budget_gb`, else equal. The planner
balances total bytes per rank: heads first, then vocab rows by share, then MLP granules fill each rank up to its share.
Bytes assume bf16 unless the model carries quantization info (GGUF): then the stored block bytes of every tensor type are
counted, per layer, and the workspace adds the largest transient dequantized chunk. Row split layers (o_proj, down_proj)
may be cut only at block boundaries; heads and MLP rows are allocated in units that respect it. Workspace is an estimate.
Global rank order is stage order, then rank order; gid 0 (stage 0, rank 0) is also the coordinator."""
import math
from dataclasses import asdict, dataclass, field

from .quant import BLOCK, CHUNK_ELEMS, CHUNK_WORKING_BYTES_PER_ELEM, DENSE
from .shard import SliceSpec, kv_range

BF16 = 2
MPS_HEAP = 1 << 30           # session 6: any tensor of 10 MiB or more on MPS lands in a 1 GiB resident heap
MPS_BIG_TENSOR = 10 << 20


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class RankIn:
    host: str
    device: str = "mps"
    port: int = 0
    weight: float | None = None
    budget_gb: float | None = None
    heads: int | None = None


@dataclass(frozen=True)
class StageIn:
    name: str
    layers: tuple[int, int] | None
    ranks: tuple[RankIn, ...]
    embed: bool = False
    lm_head: bool = False


@dataclass(frozen=True)
class Placement:
    model: str
    stages: tuple[StageIn, ...]
    coordinator_host: str | None = None
    reduce_dtype: str = "float32"
    max_seq_len: int = 4096
    prefill_max: int = 256
    granule: int = 128
    allow_mixed_devices: bool = False


@dataclass(frozen=True)
class QuantInfo:
    """Quantization type per layer and role (names of gguf quant types), and of the embedding and the output head."""
    q: tuple
    k: tuple
    v: tuple
    o: tuple
    gate: tuple
    up: tuple
    down: tuple
    embed: str
    head: str

    @classmethod
    def from_meta(cls, meta: dict, n_layers: int) -> "QuantInfo":
        def role(suffix):
            return tuple(meta[f"model.layers.{i}.{suffix}.weight"].qtype for i in range(n_layers))
        emb = meta["model.embed_tokens.weight"].qtype
        return cls(role("self_attn.q_proj"), role("self_attn.k_proj"), role("self_attn.v_proj"), role("self_attn.o_proj"),
                   role("mlp.gate_proj"), role("mlp.up_proj"), role("mlp.down_proj"), emb,
                   meta["lm_head.weight"].qtype if "lm_head.weight" in meta else emb)


def _bpe(qtype: str) -> float:
    """Stored bytes per weight element."""
    bs, ts = BLOCK[qtype]
    return ts / bs


@dataclass(frozen=True)
class Units:
    """Bytes of one unit, averaged over the layers of a stage (so that layers * unit is the exact stage total)."""
    q_head: float
    kv_head: float
    mlp_row: float
    embed_row: float
    head_row: float
    norms: float
    o_block: int            # input columns per block of o_proj (largest over the layers); 1 for dense types
    down_block: int


@dataclass(frozen=True)
class Dims:
    """The model numbers the planner needs (from ModelConfig)."""
    hidden: int
    intermediate: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab: int
    tied: bool
    hd: int = 0                      # head size; 0 means hidden // n_heads
    qkv_bias: bool = True
    qk_norm: bool = False
    quant: QuantInfo | None = None

    @property
    def head_dim(self) -> int:
        return self.hd or self.hidden // self.n_heads

    @classmethod
    def of(cls, cfg, meta: dict | None = None) -> "Dims":
        return cls(cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.num_attention_heads,
                   cfg.num_key_value_heads, cfg.vocab_size, cfg.tie_word_embeddings, hd=cfg.head_dim,
                   qkv_bias=cfg.arch.qkv_bias, qk_norm=cfg.arch.qk_norm,
                   quant=QuantInfo.from_meta(meta, cfg.num_hidden_layers) if meta else None)

    def units(self, l0: int, l1: int) -> Units:
        hd, H = self.head_dim, self.hidden
        bias = hd if self.qkv_bias else 0
        norms = BF16 * (2 * H + (2 * hd if self.qk_norm else 0))
        if self.quant is None:
            return Units(BF16 * (2 * hd * H + bias), BF16 * 2 * (hd * H + bias), BF16 * 3 * H, BF16 * H, BF16 * H, norms, 1, 1)
        q = self.quant
        L = max(1, l1 - l0)
        rng = range(l0, l1) if l1 > l0 else range(0, 1)
        mean = lambda f: sum(f(i) for i in rng) / len(rng)
        row = lambda t: H * _bpe(t)                      # bytes of one weight row with `hidden` input columns
        q_head = mean(lambda i: hd * row(q.q[i]) + H * hd * _bpe(q.o[i])) + BF16 * bias
        kv_head = mean(lambda i: hd * (row(q.k[i]) + row(q.v[i]))) + BF16 * 2 * bias
        mlp_row = mean(lambda i: 2 * row(q.gate[i]) + H * _bpe(q.down[i]))
        blk = lambda roles: max(BLOCK[roles[i]][0] if roles[i] not in DENSE else 1 for i in rng)
        return Units(q_head, kv_head, mlp_row, row(q.embed), row(q.head), norms, blk(q.o), blk(q.down))

    # bytes of one unit, bf16 (the unquantized default); kept for callers that ask a Dims directly
    def q_head(self) -> int:
        return int(self.units(0, self.n_layers).q_head)

    def kv_head(self) -> int:
        return int(self.units(0, self.n_layers).kv_head)

    def mlp_row(self) -> int:
        return int(self.units(0, self.n_layers).mlp_row)

    def layer_norms(self) -> int:
        return int(self.units(0, self.n_layers).norms)

    def vocab_row(self) -> int:
        return BF16 * self.hidden

    def kv_cache_per_head(self, max_seq: int) -> int:
        return 2 * max_seq * self.head_dim * BF16

    def matrix_elems(self, nh: int, nkv: int, rows: int, vocab_rows: int) -> int:
        """Elements of the largest dense matrix a rank expands in one go (the dequant chunk caps it)."""
        hd, H = self.head_dim, self.hidden
        return max(nh * hd * H, nkv * hd * H, rows * H, vocab_rows * H, 1)


@dataclass(frozen=True)
class RankSpec:
    gid: int
    name: str
    stage: int
    stage_name: str
    group_rank: int
    group_size: int
    host: str
    device: str
    port: int
    layers: tuple[int, int]
    q0: int
    q1: int
    m0: int
    m1: int
    kv0: int
    kv1: int
    embed_rows: tuple[int, int] | None
    head_rows: tuple[int, int] | None
    final_norm: bool
    budget: int | None
    planned: dict = field(default_factory=dict, compare=False)

    @property
    def n_layers(self) -> int:
        return self.layers[1] - self.layers[0]

    def slice_spec(self, d: Dims) -> SliceSpec:
        return SliceSpec(d.n_heads, d.n_kv_heads, d.head_dim, d.intermediate, self.q0, self.q1, self.m0, self.m1,
                         row_bias=self.group_rank == 0)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "RankSpec":
        d = dict(d)
        for k in ("layers", "embed_rows", "head_rows"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)


# ---- parsing (pure: takes the already loaded YAML dict) --------------------------------------------------------------
def placement_from_dict(y: dict) -> Placement:
    stages = []
    for i, s in enumerate(y.get("stages") or []):
        lay = s.get("layers")
        ranks = tuple(RankIn(host=r["host"], device=r.get("device", "mps"), port=int(r.get("port", 0)),
                             weight=r.get("weight"), budget_gb=r.get("budget_gb"), heads=r.get("heads"))
                      for r in s.get("ranks") or [])
        stages.append(StageIn(name=s.get("name", f"s{i}"), layers=tuple(lay) if lay else None, ranks=ranks,
                              embed=bool(s.get("embed", False)), lm_head=bool(s.get("lm_head", False))))
    c = y.get("coordinator") or {}
    return Placement(model=y.get("model", ""), stages=tuple(stages), coordinator_host=c.get("host"),
                     reduce_dtype=y.get("reduce_dtype", "float32"), max_seq_len=int(y.get("max_seq_len", 4096)),
                     prefill_max=int(y.get("prefill_max", 256)), granule=int(y.get("granule", 128)),
                     allow_mixed_devices=bool(y.get("allow_mixed_devices", False)))


# ---- helpers -----------------------------------------------------------------------------------------------------------
def largest_remainder(x: list[float], total: int) -> list[int]:
    """Integers summing to `total`, each the floor of x[i] plus one for the largest remainders (ties to lower i)."""
    s = sum(x)
    x = [v * total / s for v in x] if s > 0 else [total / len(x)] * len(x)
    f = [int(v) for v in x]
    order = sorted(range(len(x)), key=lambda i: (-(x[i] - f[i]), i))
    for i in order[:total - sum(f)]:
        f[i] += 1
    return f


def _ranges(counts: list[int], unit: int, limit: int) -> list[tuple[int, int]]:
    out, c = [], 0
    for n in counts:
        out.append((min(limit, c * unit), min(limit, (c + n) * unit)))
        c += n
    return out


def _shares(ranks: tuple[RankIn, ...]) -> list[float]:
    if any(r.weight is not None for r in ranks):
        sh = [float(r.weight if r.weight is not None else 1.0) for r in ranks]
    elif all(r.budget_gb is not None for r in ranks):
        sh = [float(r.budget_gb) for r in ranks]
    else:
        sh = [1.0] * len(ranks)
    if any(v <= 0 for v in sh):
        raise PlanError(f"weights and budgets must be positive: {sh}")
    return sh


def workspace_bytes(d: Dims, n_layers: int, nh: int, nkv: int, rows: int, vocab_rows: int, head: bool,
                    T: int, max_seq: int) -> int:
    """Estimate of transient tensors for a prefill of T tokens and a decode at max_seq (bf16 unless stated):
    activations 8 T H; q, k, v and rotary temporaries 2 T (nh + 2 nkv) hd; K and V gathered per query head
    2 nh max_seq hd; attention scores fp32 nh (T^2 + max_seq); MLP 3 T rows; logits bf16 plus fp32 per vocab row."""
    hd = d.head_dim
    ws = 8 * T * d.hidden * BF16
    if n_layers:
        ws += 2 * T * (nh + 2 * nkv) * hd * BF16 + 2 * nh * max_seq * hd * BF16 + nh * (T * T + max_seq) * 4 + 3 * T * rows * BF16
    if head:
        ws += vocab_rows * (BF16 + 4)
    if d.quant is not None:                 # one dequantized chunk at a time: bf16 result plus fp32 working copies
        ws += min(d.matrix_elems(nh, nkv, rows, vocab_rows), CHUNK_ELEMS) * CHUNK_WORKING_BYTES_PER_ELEM
    return ws


# ---- the planner -------------------------------------------------------------------------------------------------------
def validate(d: Dims, p: Placement) -> None:
    if not p.stages:
        raise PlanError("no stages")
    if p.reduce_dtype not in ("float32", "bfloat16"):
        raise PlanError(f"reduce_dtype {p.reduce_dtype!r}: float32 or bfloat16")
    nxt = 0
    for i, s in enumerate(p.stages):
        if not s.ranks:
            raise PlanError(f"stage {s.name}: no ranks")
        if s.layers:
            a, b = s.layers
            if a != nxt or b <= a:
                raise PlanError(f"stage {s.name}: layers [{a}, {b}) do not continue at layer {nxt}")
            nxt = b
        elif not s.lm_head and not s.embed:
            raise PlanError(f"stage {s.name} holds no layers, no embedding and no head")
        if len({r.device.split(":")[0] for r in s.ranks}) > 1 and not p.allow_mixed_devices:
            raise PlanError(f"stage {s.name}: mixed device types in one group; activations would drift apart "
                            f"(set allow_mixed_devices: true to accept this)")
        hs = [r.heads for r in s.ranks]
        if any(h is not None for h in hs):
            if any(h is None for h in hs) or sum(hs) != d.n_heads or min(hs) < 0:
                raise PlanError(f"stage {s.name}: heads override {hs} must be given for every rank and sum to {d.n_heads}")
    if nxt != d.n_layers:
        raise PlanError(f"layers cover [0, {nxt}) of {d.n_layers}")
    emb = [i for i, s in enumerate(p.stages) if s.embed]
    head = [i for i, s in enumerate(p.stages) if s.lm_head]
    if emb and emb != [0]:
        raise PlanError("the embedding belongs to the first stage, once")
    if head and head != [len(p.stages) - 1]:
        raise PlanError("the output head belongs to the last stage, once")
    seen = set()
    for s in p.stages:
        for r in s.ranks:
            if (r.host, r.port) in seen:
                raise PlanError(f"port {r.port} used twice on {r.host}")
            seen.add((r.host, r.port))
    if p.coordinator_host and p.coordinator_host != p.stages[0].ranks[0].host:
        raise PlanError(f"the coordinator runs in rank 0 of the first stage, on {p.stages[0].ranks[0].host}, "
                        f"not on {p.coordinator_host}")


def plan(cfg, p: Placement, check_budget: bool = True) -> list[RankSpec]:
    """RankSpecs in global order. Raises PlanError naming every rank over its budget (unless check_budget is False,
    for a forced run whose failure is the point)."""
    d = cfg if isinstance(cfg, Dims) else Dims.of(cfg)
    validate(d, p)
    G = p.granule
    last = len(p.stages) - 1
    has_embed = [s.embed or (i == 0 and not any(t.embed for t in p.stages)) for i, s in enumerate(p.stages)]
    has_head = [s.lm_head or (i == last and not any(t.lm_head for t in p.stages)) for i, s in enumerate(p.stages)]
    specs, over, gid = [], [], 0
    for si, s in enumerate(p.stages):
        n, sh = len(s.ranks), _shares(s.ranks)
        l0, l1 = s.layers if s.layers else (0, 0)
        L = l1 - l0
        u = d.units(l0, l1)
        hu = u.o_block // math.gcd(u.o_block, d.head_dim)         # heads per o_proj block boundary
        Gm = G * u.down_block // math.gcd(G, u.down_block)          # MLP granule, multiple of the down_proj block
        if s.ranks[0].heads is not None:
            heads, hu_ = [r.heads for r in s.ranks], 1
        elif L:
            heads, hu_ = largest_remainder(sh, -(-d.n_heads // hu)), hu
        else:
            heads, hu_ = [0] * n, 1
        if not L:
            heads = [0] * n
        qr = _ranges(heads, hu_, d.n_heads)
        for a_, b_ in qr:
            for hb in (a_, b_):
                if L and 0 < hb < d.n_heads and (hb * d.head_dim) % u.o_block:
                    raise PlanError(f"stage {s.name}: a head boundary at head {hb} is column {hb * d.head_dim} of o_proj, not a "
                                    f"multiple of its block of {u.o_block} columns. Row split layers (o_proj, down_proj) can be "
                                    f"cut only at block boundaries (32 for Q8_0, Q4_0, Q5_0; 256 for K quants)")
        kvr = [kv_range(a, b, d.n_heads, d.n_kv_heads) for a, b in qr]
        emb, hd_ = has_embed[si], has_head[si]
        if emb or hd_:
            vr = _ranges(largest_remainder(sh, -(-d.vocab // G)), G, d.vocab)
        else:
            vr = [(0, 0)] * n
        # a rank holding both with the same tied rows loads one tensor: count it once
        row_b = (u.embed_row if emb else 0) + (u.head_row if hd_ and not (emb and d.tied) else 0)
        vb = [(b - a) * row_b for a, b in vr]
        fn = d.hidden * BF16 if hd_ else 0
        attn = [(b - a) * u.q_head + (kb - ka) * u.kv_head for (a, b), (ka, kb) in zip(qr, kvr)]
        if L:
            ng = -(-d.intermediate // Gm)
            total = L * (sum(attn) + d.intermediate * u.mlp_row + n * u.norms) + sum(vb) + n * fn
            raw = [max(0.0, (sh[i] / sum(sh) * total - L * (attn[i] + u.norms) - vb[i] - fn) / (L * Gm * u.mlp_row))
                   for i in range(n)]
            mr = _ranges(largest_remainder(raw if sum(raw) > 0 else sh, ng), Gm, d.intermediate)
        else:
            mr = [(0, 0)] * n
        for i, r in enumerate(s.ranks):
            (q0, q1), (kv0, kv1), (m0, m1), (v0, v1) = qr[i], kvr[i], mr[i], vr[i]
            if L and q0 == q1 and m0 == m1 and v0 == v1:
                raise PlanError(f"rank {s.name}/{i} on {r.host} holds no heads, no MLP rows and no vocab rows")
            w = int(L * (attn[i] + (m1 - m0) * u.mlp_row + u.norms) + vb[i] + fn)
            kv = L * (kv1 - kv0) * d.kv_cache_per_head(p.max_seq_len)
            ws = workspace_bytes(d, L, q1 - q0, kv1 - kv0, m1 - m0, v1 - v0, hd_, p.prefill_max, p.max_seq_len)
            budget = int(r.budget_gb * 1e9) if r.budget_gb is not None else None
            big = d.quant is not None or d.matrix_elems(q1 - q0, kv1 - kv0, m1 - m0, v1 - v0) * BF16 >= MPS_BIG_TENSOR
            planned = {"weights": w, "kv": kv, "workspace": ws, "total": w + kv + ws,
                       "layer_bytes": int(attn[i] + (m1 - m0) * u.mlp_row + u.norms) if L else 0,
                       "vocab_bytes": int(vb[i]), "kv_heads_replicated": 0}
            planned["mps_floor"] = MPS_HEAP if r.device.startswith("mps") and big else 0     # informational, not in total
            planned["footprint"] = max(planned["total"], planned["mps_floor"])
            spec = RankSpec(gid=gid, name=f"{s.name}/{i}", stage=si, stage_name=s.name, group_rank=i, group_size=n,
                            host=r.host, device=r.device, port=r.port, layers=(l0, l1), q0=q0, q1=q1, m0=m0, m1=m1,
                            kv0=kv0, kv1=kv1, embed_rows=(v0, v1) if emb else None, head_rows=(v0, v1) if hd_ else None,
                            final_norm=hd_, budget=budget, planned=planned)
            if budget is not None and planned["total"] > budget:
                over.append(f"{spec.name} on {r.host}: planned {planned['total'] / 1e9:.3f} GB (weights "
                            f"{w / 1e9:.3f}, KV {kv / 1e9:.3f}, workspace {ws / 1e9:.3f}) > budget {budget / 1e9:.3f} GB")
            specs.append(spec)
            gid += 1
        if L:
            copies = sum(kb - ka for ka, kb in kvr)
            for sp in specs[-n:]:
                sp.planned["kv_heads_replicated"] = copies - d.n_kv_heads
    if over and check_budget:
        raise PlanError("over budget: " + "; ".join(over))
    return specs


def needed_peers(spec: RankSpec, specs: list[RankSpec]) -> dict[int, str]:
    """gid -> role ("grp", "up", "down") of every data link a rank needs: its group, the previous stage's leader,
    and, for a leader, every rank of the next stage."""
    p = {s.gid: "grp" for s in specs if s.stage == spec.stage and s.gid != spec.gid}
    up = next((s for s in specs if s.stage == spec.stage - 1 and s.group_rank == 0), None)
    if up:
        p[up.gid] = "up"
    if spec.group_rank == 0:
        for s in specs:
            if s.stage == spec.stage + 1:
                p[s.gid] = "down"
    return p


def listen_addrs(specs: list[RankSpec], hosts: dict[str, str]) -> dict[int, list[str]]:
    """Per rank, the addresses its inbound peers need: 127.0.0.1 for a peer on the same host, the host's chosen
    address for a peer on another host. Inbound peers are the coordinator (ctl link, gid 0's host) and every needed
    peer with a higher gid."""
    out = {}
    for s in specs:
        inbound = [specs[g].host for g in needed_peers(s, specs) if g > s.gid] + ([specs[0].host] if s.gid else [])
        addrs = (["127.0.0.1"] if s.host in inbound else []) + ([hosts[s.host]] if any(h != s.host for h in inbound) else [])
        out[s.gid] = addrs or ["127.0.0.1"]
    return out


# ---- frames per decode step (section 4 of the session 6 prompt) --------------------------------------------------------
def frames_per_step(specs: list[RankSpec], embed_split: bool | None = None) -> dict:
    """Predicted frames sent by all processes in one decode step, by term. The coordinator is gid 0."""
    stages: dict[int, list[RankSpec]] = {}
    for s in specs:
        stages.setdefault(s.stage, []).append(s)
    order = sorted(stages)
    n0 = len(stages[order[0]])
    head = [s for s in specs if s.head_rows is not None]
    emb = [s for s in specs if s.embed_rows is not None]
    split = (len(emb) > 1) if embed_split is None else embed_split
    t = {"token": n0 - 1, "embed": n0 * (n0 - 1) if split else 0,
         "layers": sum(stages[k][0].n_layers * 2 * len(stages[k]) * (len(stages[k]) - 1) for k in order),
         "hops": sum(len(stages[k]) for k in order[1:]),
         "head": len(head) - (1 if any(s.gid == 0 for s in head) else 0)}
    t["total"] = sum(t.values())
    return t


def bytes_per_step(specs: list[RankSpec], hidden: int, header: int = 24) -> int:
    """Hidden state frames times (2 hidden + header); token and head frames are small and left out."""
    f = frames_per_step(specs)
    return (f["embed"] + f["layers"] + f["hops"]) * (BF16 * hidden + header)


def format_plan(specs: list[RankSpec], d: Dims) -> str:
    rows = ["| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in specs:
        p = s.planned
        v = s.embed_rows or s.head_rows
        what = ("E" if s.embed_rows else "") + ("H" if s.head_rows else "")
        vocab = f"{v[0]}-{v[1]} {what}" if v else "-"
        budget = f"{s.budget / 1e6:.0f}" if s.budget else "-"
        rows.append(f"| {s.name} | {s.host} | {s.device} | {s.layers[0]}-{s.layers[1]} | {s.q0}-{s.q1} ({s.q1 - s.q0}) | "
                    f"{s.kv0}-{s.kv1} | {s.m0}-{s.m1} ({s.m1 - s.m0}) | {vocab} | {p['weights'] / 1e6:.1f} | {p['kv'] / 1e6:.1f} | "
                    f"{p['workspace'] / 1e6:.1f} | {p['total'] / 1e6:.1f} | {budget} |")
    f = frames_per_step(specs)
    rows.append("")
    rows.append(f"Frames per decode step: {f['total']} (token {f['token']}, embedding {f['embed']}, layers {f['layers']}, "
                f"hops {f['hops']}, head {f['head']}); bytes per step about {bytes_per_step(specs, d.hidden) / 1e6:.3f} MB.")
    return "\n".join(rows)
