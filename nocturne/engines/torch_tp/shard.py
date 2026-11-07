"""Megatron style sharding of one Qwen2 block, pure functions on tensors and slices (no I/O).
Column split (by output rows of the weight, PyTorch Linear stores [out, in]): q_proj, k_proj, v_proj, gate_proj, up_proj.
Row split (by input columns): o_proj, down_proj. Heads are split GQA aware: rank r gets heads [r*H/W, (r+1)*H/W) and
kv heads [r*KV/W, (r+1)*KV/W), so q heads on a rank share the kv heads on the same rank. Each rank adds a partial sum
for o_proj and down_proj; the all-reduce completes it. Bias of a row split layer is added on rank 0 only (o_proj and
down_proj have no bias in Qwen2; q, k, v have bias, column split with the rows)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Shard:
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

    @property
    def heads_local(self) -> int:
        return self.n_heads // self.world

    @property
    def kv_heads_local(self) -> int:
        return self.n_kv_heads // self.world

    @property
    def inter_local(self) -> int:
        return self.intermediate // self.world

    def q_rows(self) -> slice:
        n = self.heads_local * self.head_dim
        return slice(self.rank * n, (self.rank + 1) * n)

    def kv_rows(self) -> slice:
        n = self.kv_heads_local * self.head_dim
        return slice(self.rank * n, (self.rank + 1) * n)

    def mlp_rows(self) -> slice:
        return slice(self.rank * self.inter_local, (self.rank + 1) * self.inter_local)

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
        return None       # input_layernorm, post_attention_layernorm: replicated


def shard_tensor(t, spec: tuple[int, slice] | None):
    """Slice a full tensor as slice_for says. Works on torch tensors and safetensors slices alike (both index)."""
    if spec is None:
        return t[...] if hasattr(t, "get_shape") else t
    dim, sl = spec
    return t[sl] if dim == 0 else t[:, sl]
