"""Qwen2 decoder, tensor parallel over `world` ranks, one block per layer, KV cache per rank.
Numerics follow transformers' Qwen2 in bf16 so that world 1 reproduces `generate` greedy output: RMSNorm in fp32 and
cast back, rotary cos/sin computed in fp32 then cast to the activation dtype, attention through
scaled_dot_product_attention with kv heads repeated, MLP down(silu(gate(x)) * up(x)), residual adds in bf16.
Two all-reduces per layer (after o_proj and after down_proj), done by the Comm object: identity for world 1, one frame
each way plus a local add for two ranks. The add is commutative, so both ranks hold bit identical sums.
Session 6: a rank may hold any contiguous head range (KV heads gathered per query head, so an uneven split and a
replicated KV head work), zero heads or zero MLP rows (its partial is zeros), a layer range, and vocab rows of the
embedding (lookup of the ids in range, zeros elsewhere, summed by the caller's all-reduce) and of the output head
(local argmax, combined by the coordinator).
Session 10 part 3 (overlap): the KV cache has a slot dimension, one cache per request in flight (slot 0 alone is the old
cache). A layer is two partial sums (attention, MLP), each followed by an all-reduce; the all-reduce is two halves,
send_partial and recv_sum, so two requests can share a group in lockstep (forward_pair): A's partial travels while B's
is computed. Each request uses the same operations on the same shapes as when it runs alone."""
import math

import torch
import torch.nn.functional as F

from .config import ModelConfig as Qwen2Config
from .quant import QuantLinear, linear
from .shard import Shard


class Comm:
    """One rank: nothing to exchange. send_partial hands a partial sum over and returns a ticket; recv_sum turns the
    ticket into the group's sum. all_reduce is the two in a row."""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def send_partial(self, x: torch.Tensor, tag: int | None = None):
        return x

    def recv_sum(self, ticket) -> torch.Tensor:
        return ticket


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rotary(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype, device, factors=None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin of shape [T, head_dim], as transformers computes them (fp32, duplicated halves, cast to dtype).
    `factors` (Llama 3 scaling) divide the frequencies, as ggml's freq_factors do."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    if factors is not None:
        inv = inv / torch.tensor(factors, dtype=torch.float32, device=device)
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Layer:
    def __init__(self, i: int, w: dict[str, torch.Tensor], cfg: Qwen2Config, shard: Shard, device, dtype, max_seq: int,
                 slots: int = 1):
        p = f"model.layers.{i}."
        g = lambda n: w[p + n]
        self.ln1, self.ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        opt = lambda n: w.get(p + n)
        self.wq, self.bq = g("self_attn.q_proj.weight"), opt("self_attn.q_proj.bias")
        self.wk, self.bk = g("self_attn.k_proj.weight"), opt("self_attn.k_proj.bias")
        self.wv, self.bv = g("self_attn.v_proj.weight"), opt("self_attn.v_proj.bias")
        self.q_norm, self.k_norm = opt("self_attn.q_norm.weight"), opt("self_attn.k_norm.weight")
        if cfg.arch.qkv_bias and self.bq is None:
            raise KeyError(f"{cfg.arch_name} expects q, k, v bias in layer {i}")
        if cfg.arch.qk_norm and self.q_norm is None:
            raise KeyError(f"{cfg.arch_name} expects q_norm and k_norm in layer {i}")
        self.wo = g("self_attn.o_proj.weight")
        self.wg, self.wu, self.wd = g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"), g("mlp.down_proj.weight")
        self.cfg, self.shard = cfg, shard
        self.hd = cfg.head_dim
        self.nh, self.nkv = shard.heads_local, shard.kv_heads_local
        kvi = shard.kv_index() if self.nh else []
        self.kv_idx = None if kvi == list(range(self.nkv)) else torch.tensor(kvi, dtype=torch.long, device=device)
        self.hidden = cfg.hidden_size
        self.k_cache = torch.zeros(slots, self.nkv, max_seq, self.hd, dtype=dtype, device=device)      # one cache per slot
        self.v_cache = torch.zeros(slots, self.nkv, max_seq, self.hd, dtype=dtype, device=device)

    def attn_partial(self, h: torch.Tensor, pos0: int, cos: torch.Tensor, sin: torch.Tensor, slot: int = 0) -> torch.Tensor:
        """This rank's part of the attention output, to be summed over the group."""
        x = rms_norm(h, self.ln1, self.cfg.rms_norm_eps)
        return self._attention(x, pos0, cos, sin, slot) if self.nh else torch.zeros(h.shape[0], self.hidden, dtype=h.dtype, device=h.device)

    def mlp_partial(self, h: torch.Tensor) -> torch.Tensor:
        """This rank's part of the MLP output, to be summed over the group."""
        x = rms_norm(h, self.ln2, self.cfg.rms_norm_eps)
        if self.wg.shape[0]:
            return linear(F.silu(linear(x, self.wg)) * linear(x, self.wu), self.wd)     # partial over this rank's slice
        return torch.zeros_like(h)

    def forward(self, h: torch.Tensor, pos0: int, cos: torch.Tensor, sin: torch.Tensor, comm: Comm, slot: int = 0) -> torch.Tensor:
        h = h + comm.all_reduce(self.attn_partial(h, pos0, cos, sin, slot))
        return h + comm.all_reduce(self.mlp_partial(h))

    def forward_pair(self, ha, a, hb, b, comm: Comm) -> tuple[torch.Tensor, torch.Tensor]:
        """Two requests in lockstep. a and b are (pos0, cos, sin, slot). Each all-reduce: A's partial is sent, then B's
        is computed and sent, then A's sum is received, then B's. Every rank of the group keeps this order."""
        ta = comm.send_partial(self.attn_partial(ha, *a), 0)
        tb = comm.send_partial(self.attn_partial(hb, *b), 1)
        ha = ha + comm.recv_sum(ta)
        hb = hb + comm.recv_sum(tb)
        ta = comm.send_partial(self.mlp_partial(ha), 0)
        tb = comm.send_partial(self.mlp_partial(hb), 1)
        return ha + comm.recv_sum(ta), hb + comm.recv_sum(tb)

    def _attention(self, x: torch.Tensor, pos0: int, cos: torch.Tensor, sin: torch.Tensor, slot: int = 0) -> torch.Tensor:
        T = x.shape[0]
        q = linear(x, self.wq, self.bq).view(T, self.nh, self.hd).transpose(0, 1)      # [nh, T, hd]
        k = linear(x, self.wk, self.bk).view(T, self.nkv, self.hd).transpose(0, 1)
        v = linear(x, self.wv, self.bv).view(T, self.nkv, self.hd).transpose(0, 1)
        if self.q_norm is not None:                    # Qwen3: per head RMSNorm before rope
            q = rms_norm(q, self.q_norm, self.cfg.rms_norm_eps)
            k = rms_norm(k, self.k_norm, self.cfg.rms_norm_eps)
        q = q * cos[None] + rotate_half(q) * sin[None]
        k = k * cos[None] + rotate_half(k) * sin[None]
        self.k_cache[slot, :, pos0:pos0 + T] = k
        self.v_cache[slot, :, pos0:pos0 + T] = v
        K = self.k_cache[slot, :, :pos0 + T]
        V = self.v_cache[slot, :, :pos0 + T]
        if self.kv_idx is not None:
            K = K.index_select(0, self.kv_idx)
            V = V.index_select(0, self.kv_idx)
        if T > 1:
            # causal over the new tokens; cached prefix (pos0 > 0 only when a prompt extends a cache) is fully visible
            mask = torch.ones(T, pos0 + T, dtype=torch.bool, device=x.device).tril(diagonal=pos0)
            a = F.scaled_dot_product_attention(q[None], K[None], V[None], attn_mask=mask[None, None], scale=1.0 / math.sqrt(self.hd))
        else:
            a = F.scaled_dot_product_attention(q[None], K[None], V[None], scale=1.0 / math.sqrt(self.hd))
        a = a[0].transpose(0, 1).reshape(T, self.nh * self.hd)
        return linear(a, self.wo)                      # partial sum over this rank's heads

    def reset(self):
        pass        # the cache is overwritten from position 0 on the next prompt; nothing to clear


class Qwen2TP:
    def __init__(self, cfg: Qwen2Config, weights: dict[str, torch.Tensor], shard: Shard, device: str, max_seq: int,
                 dtype: torch.dtype = torch.bfloat16, layers: tuple[int, int] | None = None,
                 embed_rows: tuple[int, int] | None = None, head_rows: tuple[int, int] | None = None, slots: int = 1):
        self.cfg, self.shard, self.device, self.dtype, self.max_seq, self.slots = cfg, shard, device, dtype, max_seq, slots
        l0, l1 = layers if layers else (0, cfg.num_hidden_layers)
        self.layers = [Layer(i, weights, cfg, shard, device, dtype, max_seq, slots) for i in range(l0, l1)]
        self.embed = weights.get("model.embed_tokens.weight")
        self.norm = weights.get("model.norm.weight")
        self.lm_head = weights.get("lm_head.weight")
        self.embed_rows = embed_rows or ((0, cfg.vocab_size) if self.embed is not None else None)
        self.head_rows = head_rows or ((0, cfg.vocab_size) if self.lm_head is not None else None)

    def embed_tokens(self, ids: list[int]) -> torch.Tensor:
        """Embedding of `ids`. With a vocab slice, rows outside it are exact zeros; the group's all-reduce completes it."""
        t = torch.tensor(ids, device=self.device)
        v0, v1 = self.embed_rows
        if isinstance(self.embed, QuantLinear):
            lookup = lambda ix: self.embed.rows(ix, self.dtype)
        else:
            lookup = lambda ix: F.embedding(ix, self.embed)
        if (v0, v1) == (0, self.cfg.vocab_size):
            return lookup(t)
        inside = (t >= v0) & (t < v1)
        e = lookup((t - v0).clamp(0, max(0, v1 - v0 - 1)))
        return torch.where(inside[:, None], e, torch.zeros_like(e))

    def _rotary(self, h: torch.Tensor, pos0: int, slot: int):
        T = h.shape[0]
        if pos0 + T > self.max_seq:
            raise ValueError(f"sequence {pos0 + T} exceeds max_seq_len {self.max_seq}")
        if not 0 <= slot < self.slots:
            raise ValueError(f"slot {slot} of {self.slots}")
        pos = torch.arange(pos0, pos0 + T, device=self.device)
        return rotary(pos, self.cfg.head_dim, self.cfg.rope_theta, h.dtype, self.device, self.cfg.rope_factors)

    @torch.inference_mode()
    def layers_forward(self, h: torch.Tensor, pos0: int, comm: Comm, slot: int = 0) -> torch.Tensor:
        cos, sin = self._rotary(h, pos0, slot)
        for layer in self.layers:
            h = layer.forward(h, pos0, cos, sin, comm, slot)
        return h

    @torch.inference_mode()
    def layers_forward_pair(self, ha: torch.Tensor, pos_a: int, slot_a: int, hb: torch.Tensor, pos_b: int, slot_b: int,
                            comm: Comm) -> tuple[torch.Tensor, torch.Tensor]:
        """Two requests through every layer in lockstep (Layer.forward_pair)."""
        if slot_a == slot_b:
            raise ValueError(f"a pair needs two slots, got {slot_a} twice")
        a = (pos_a, *self._rotary(ha, pos_a, slot_a), slot_a)
        b = (pos_b, *self._rotary(hb, pos_b, slot_b), slot_b)
        for layer in self.layers:
            ha, hb = layer.forward_pair(ha, a, hb, b, comm)
        return ha, hb

    @torch.inference_mode()
    def logits_last(self, h: torch.Tensor) -> torch.Tensor:
        """Logits of the last position over this rank's head rows (all rows for a whole head)."""
        x = rms_norm(h[-1:], self.norm, self.cfg.rms_norm_eps)
        return linear(x, self.lm_head)[0]

    def head_argmax(self, h: torch.Tensor) -> tuple[float, int, torch.Tensor]:
        """(max logit, global index of its first occurrence, local logits). Greedy over this rank's vocab rows."""
        lg = self.logits_last(h)
        m = lg.max()
        i = int(torch.nonzero(lg == m)[0, 0].item())
        return float(m.float().item()), self.head_rows[0] + i, lg


    def head_device(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """head_argmax without reading the device (--async-send): ([max logit, local index of its first occurrence] as
        two float32 values on the device, local logits). The index is exact in float32 below 2**24 rows."""
        lg = self.logits_last(h)
        n = lg.shape[0]
        if n >= 1 << 24:
            raise ValueError(f"{n} head rows do not fit a float32 index")
        m = lg.max()
        i = torch.where(lg == m, torch.arange(n, device=lg.device), n).min()
        return torch.stack([m.float(), i.float()]), lg


def pick_global(cands: list[tuple[float, int]]) -> int:
    """Greedy token from (value, global index) per head rank: the largest value, ties to the lowest index."""
    return min(cands, key=lambda c: (-c[0], c[1]))[1]
