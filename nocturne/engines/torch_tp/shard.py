"""Megatron style sharding of one Qwen2 block, pure functions on tensors and slices (no I/O).
Column split (by output rows of the weight, PyTorch Linear stores [out, in]): q_proj, k_proj, v_proj, gate_proj, up_proj.
Row split (by input columns): o_proj, down_proj. Heads are split GQA aware: rank r gets heads [r*H/W, (r+1)*H/W) and
kv heads [r*KV/W, (r+1)*KV/W), so q heads on a rank share the kv heads on the same rank. Each rank adds a partial sum
for o_proj and down_proj; the all-reduce completes it. Bias of a row split layer is added on rank 0 only (o_proj and
down_proj have no bias in Qwen2; q, k, v have bias, column split with the rows).
SliceSpec (session 6) is the general form: any contiguous query head range and MLP row range, zero allowed. The KV heads
a rank holds follow from GQA; a KV head used by two ranks is held by both. Shard is the even special case."""
from dataclasses import dataclass


class _SliceMath:
    """Row ranges and the GQA map from q0, q1, kv0, kv1, m0, m1 and head_dim."""

    @property
    def heads_local(self) -> int:
        return self.q1 - self.q0

    @property
    def kv_heads_local(self) -> int:
        return self.kv1 - self.kv0

    @property
    def inter_local(self) -> int:
        return self.m1 - self.m0

    def q_rows(self) -> slice:
        return slice(self.q0 * self.head_dim, self.q1 * self.head_dim)

    def kv_rows(self) -> slice:
        return slice(self.kv0 * self.head_dim, self.kv1 * self.head_dim)

    def mlp_rows(self) -> slice:
        return slice(self.m0, self.m1)

    def kv_index(self) -> list[int]:
        """Local KV head of every local query head."""
        g = self.n_heads // self.n_kv_heads
        return [h // g - self.kv0 for h in range(self.q0, self.q1)]

    def slice_for(self, name: str) -> tuple[int, slice] | None:
        """(dim, slice) of the weight `name` (the part after 'model.layers.N.'), or None when it is replicated.
        dim 0 = rows (output features, column parallel), dim 1 = columns (input features, row parallel)."""
        if name in ("self_attn.q_proj.weight", "self_attn.q_proj.bias"):
            return 0, self.q_rows()
        if name in ("self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias"):
            return 0, self.kv_rows()
        if name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            return 0, self.mlp_rows()
        if name == "self_attn.o_proj.weight":
            return 1, self.q_rows()
        if name == "mlp.down_proj.weight":
            return 1, self.mlp_rows()
        return None       # input_layernorm, post_attention_layernorm: replicated; row split biases: see row_bias


def kv_range(q0: int, q1: int, n_heads: int, n_kv_heads: int) -> tuple[int, int]:
    """KV heads [kv0, kv1) used by query heads [q0, q1); empty when q0 == q1."""
    if q1 <= q0:
        return q0 * n_kv_heads // n_heads, q0 * n_kv_heads // n_heads
    g = n_heads // n_kv_heads
    return q0 // g, (q1 - 1) // g + 1


@dataclass(frozen=True)
class SliceSpec(_SliceMath):
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    q0: int
    q1: int
    m0: int
    m1: int
    row_bias: bool = True      # this rank adds the bias of row split layers (one rank per group)

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"heads {self.n_heads} not a multiple of kv heads {self.n_kv_heads}")
        if not (0 <= self.q0 <= self.q1 <= self.n_heads and 0 <= self.m0 <= self.m1 <= self.intermediate):
            raise ValueError(f"bad ranges q [{self.q0}, {self.q1}) of {self.n_heads}, mlp [{self.m0}, {self.m1}) of {self.intermediate}")

    @property
    def kv0(self) -> int:
        return kv_range(self.q0, self.q1, self.n_heads, self.n_kv_heads)[0]

    @property
    def kv1(self) -> int:
        return kv_range(self.q0, self.q1, self.n_heads, self.n_kv_heads)[1]


@dataclass(frozen=True)
class Shard(_SliceMath):
    """Even split over `world` ranks (the session 3 engine). Rank 0 adds row split biases."""
    rank: int
    world: int
    n_heads: int          # total attention heads
    n_kv_heads: int       # total kv heads
    head_dim: int
    intermediate: int     # total intermediate size

    def __post_init__(self):
        if self.n_heads % self.world or self.n_kv_heads % self.world or self.intermediate % self.world:
            raise ValueError(f"heads {self.n_heads}, kv heads {self.n_kv_heads}, intermediate {self.intermediate} "
                             f"must divide by world {self.world}")

    q0 = property(lambda s: s.rank * s.n_heads // s.world)
    q1 = property(lambda s: (s.rank + 1) * s.n_heads // s.world)
    kv0 = property(lambda s: s.rank * s.n_kv_heads // s.world)
    kv1 = property(lambda s: (s.rank + 1) * s.n_kv_heads // s.world)
    m0 = property(lambda s: s.rank * s.intermediate // s.world)
    m1 = property(lambda s: (s.rank + 1) * s.intermediate // s.world)
    row_bias = property(lambda s: s.rank == 0)


def shard_tensor(t, spec: tuple[int, slice] | None):
    """Slice a full tensor as slice_for says. Works on torch tensors and safetensors slices alike (both index)."""
    if spec is None:
        return t[...] if hasattr(t, "get_shape") else t
    dim, sl = spec
    return t[sl] if dim == 0 else t[:, sl]
