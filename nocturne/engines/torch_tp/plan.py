"""Placement planner (session 6). plan(cfg, placement) -> list[RankSpec] is pure: no I/O, no torch.
A placement is a list of PP stages; each stage is a TP group. Inside a group every rank gets a contiguous range of query
heads, a contiguous range of MLP rows (in granules) and, where the stage holds the embedding or the output head, a
contiguous range of vocab rows (in granules). Shares follow `weight`, else `budget_gb`, else equal. The planner
balances total bytes per rank: heads first, then vocab rows by share, then MLP granules fill each rank up to its share.
Bytes assume bf16 and the Qwen2 block (q, k, v with bias; o, gate, up, down without). Workspace is an estimate.
Global rank order is stage order, then rank order; gid 0 (stage 0, rank 0) is also the coordinator."""
from dataclasses import asdict, dataclass, field

from .shard import SliceSpec, kv_range

BF16 = 2


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
class Dims:
    """The model numbers the planner needs (from Qwen2Config)."""
    hidden: int
    intermediate: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab: int
    tied: bool

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @classmethod
    def of(cls, cfg) -> "Dims":
        return cls(cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.num_attention_heads,
                   cfg.num_key_value_heads, cfg.vocab_size, cfg.tie_word_embeddings)

    # bytes of one unit, bf16
    def q_head(self) -> int:
        return BF16 * (2 * self.head_dim * self.hidden + self.head_dim)

    def kv_head(self) -> int:
        return BF16 * 2 * (self.head_dim * self.hidden + self.head_dim)

    def mlp_row(self) -> int:
        return BF16 * 3 * self.hidden

    def layer_norms(self) -> int:
        return BF16 * 2 * self.hidden

    def vocab_row(self) -> int:
        return BF16 * self.hidden

    def kv_cache_per_head(self, max_seq: int) -> int:
        return 2 * max_seq * self.head_dim * BF16


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
        heads = [r.heads for r in s.ranks] if s.ranks[0].heads is not None else largest_remainder(sh, d.n_heads)
        if not L:
            heads = [0] * n
        qr = _ranges(heads, 1, d.n_heads)
        kvr = [kv_range(a, b, d.n_heads, d.n_kv_heads) for a, b in qr]
        emb, hd_ = has_embed[si], has_head[si]
        if emb or hd_:
            vr = _ranges(largest_remainder(sh, -(-d.vocab // G)), G, d.vocab)
        else:
            vr = [(0, 0)] * n
        vocab_bytes = [(b - a) * d.vocab_row() for a, b in vr]
        # a rank holding both with the same tied rows loads one tensor: count it once
        vocab_cnt = (1 if emb else 0) + (1 if hd_ else 0)
        if emb and hd_ and d.tied:
            vocab_cnt = 1
        vb = [vocab_cnt * x for x in vocab_bytes]
        fn = d.hidden * BF16 if hd_ else 0
        attn = [(b - a) * d.q_head() + (kb - ka) * d.kv_head() for (a, b), (ka, kb) in zip(qr, kvr)]
        if L:
            ng = -(-d.intermediate // G)
            total = L * (sum(attn) + d.intermediate * d.mlp_row() + n * d.layer_norms()) + sum(vb) + n * fn
            raw = [max(0.0, (sh[i] / sum(sh) * total - L * (attn[i] + d.layer_norms()) - vb[i] - fn) / (L * G * d.mlp_row()))
                   for i in range(n)]
            mr = _ranges(largest_remainder(raw if sum(raw) > 0 else sh, ng), G, d.intermediate)
        else:
            mr = [(0, 0)] * n
        for i, r in enumerate(s.ranks):
            (q0, q1), (kv0, kv1), (m0, m1), (v0, v1) = qr[i], kvr[i], mr[i], vr[i]
            if L and q0 == q1 and m0 == m1 and v0 == v1:
                raise PlanError(f"rank {s.name}/{i} on {r.host} holds no heads, no MLP rows and no vocab rows")
            w = L * (attn[i] + (m1 - m0) * d.mlp_row() + d.layer_norms()) + vb[i] + fn
            kv = L * (kv1 - kv0) * d.kv_cache_per_head(p.max_seq_len)
            ws = workspace_bytes(d, L, q1 - q0, kv1 - kv0, m1 - m0, v1 - v0, hd_, p.prefill_max, p.max_seq_len)
            budget = int(r.budget_gb * 1e9) if r.budget_gb is not None else None
            planned = {"weights": w, "kv": kv, "workspace": ws, "total": w + kv + ws,
                       "layer_bytes": attn[i] + (m1 - m0) * d.mlp_row() + d.layer_norms() if L else 0,
                       "vocab_bytes": vb[i], "kv_heads_replicated": 0}
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
