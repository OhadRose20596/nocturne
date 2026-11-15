"""Qwen2 decoder, tensor parallel over `world` ranks, one block per layer, KV cache per rank.
Numerics follow transformers' Qwen2 in bf16 so that world 1 reproduces `generate` greedy output: RMSNorm in fp32 and
cast back, rotary cos/sin computed in fp32 then cast to the activation dtype, attention through
scaled_dot_product_attention with kv heads repeated, MLP down(silu(gate(x)) * up(x)), residual adds in bf16.
Two all-reduces per layer (after o_proj and after down_proj), done by the Comm object: identity for world 1, one frame
each way plus a local add for two ranks. The add is commutative, so both ranks hold bit identical sums.
Session 6: a rank may hold any contiguous head range (KV heads gathered per query head, so an uneven split and a
replicated KV head work), zero heads or zero MLP rows (its partial is zeros), a layer range, and vocab rows of the
embedding (lookup of the ids in range, zeros elsewhere, summed by the caller's all-reduce) and of the output head
(local argmax, combined by the coordinator)."""
import math

import torch
import torch.nn.functional as F

from .config import Qwen2Config
from .shard import Shard


class Comm:
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rotary(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin of shape [T, head_dim], as transformers computes them (fp32, duplicated halves, cast to dtype)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Layer:
    def __init__(self, i: int, w: dict[str, torch.Tensor], cfg: Qwen2Config, shard: Shard, device, dtype, max_seq: int):
        p = f"model.layers.{i}."
        g = lambda n: w[p + n]
        self.ln1, self.ln2 = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.wq, self.bq = g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")
        self.wk, self.bk = g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")
        self.wv, self.bv = g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")
        self.wo = g("self_attn.o_proj.weight")
        self.wg, self.wu, self.wd = g("mlp.gate_proj.weight"), g("mlp.up_proj.weight"), g("mlp.down_proj.weight")
        self.cfg, self.shard = cfg, shard
        self.hd = cfg.head_dim
        self.nh, self.nkv = shard.heads_local, shard.kv_heads_local
        kvi = shard.kv_index() if self.nh else []
        self.kv_idx = None if kvi == list(range(self.nkv)) else torch.tensor(kvi, dtype=torch.long, device=device)
        self.hidden = cfg.hidden_size
        self.k_cache = torch.zeros(1, self.nkv, max_seq, self.hd, dtype=dtype, device=device)
        self.v_cache = torch.zeros(1, self.nkv, max_seq, self.hd, dtype=dtype, device=device)

    def forward(self, h: torch.Tensor, pos0: int, cos: torch.Tensor, sin: torch.Tensor, comm: Comm) -> torch.Tensor:
        x = rms_norm(h, self.ln1, self.cfg.rms_norm_eps)
        o = self._attention(x, pos0, cos, sin) if self.nh else torch.zeros(h.shape[0], self.hidden, dtype=h.dtype, device=h.device)
        h = h + comm.all_reduce(o)
        x = rms_norm(h, self.ln2, self.cfg.rms_norm_eps)
        if self.wg.shape[0]:
            m = F.linear(F.silu(F.linear(x, self.wg)) * F.linear(x, self.wu), self.wd)     # partial over this rank's slice
        else:
            m = torch.zeros_like(h)
        return h + comm.all_reduce(m)

    def _attention(self, x: torch.Tensor, pos0: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        q = F.linear(x, self.wq, self.bq).view(T, self.nh, self.hd).transpose(0, 1)      # [nh, T, hd]
        k = F.linear(x, self.wk, self.bk).view(T, self.nkv, self.hd).transpose(0, 1)
        v = F.linear(x, self.wv, self.bv).view(T, self.nkv, self.hd).transpose(0, 1)
        q = q * cos[None] + rotate_half(q) * sin[None]
        k = k * cos[None] + rotate_half(k) * sin[None]
        self.k_cache[0, :, pos0:pos0 + T] = k
        self.v_cache[0, :, pos0:pos0 + T] = v
        K = self.k_cache[0, :, :pos0 + T]
        V = self.v_cache[0, :, :pos0 + T]
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
        return F.linear(a, self.wo)                    # partial sum over this rank's heads

    def reset(self):
        pass        # the cache is overwritten from position 0 on the next prompt; nothing to clear


class Qwen2TP:
    def __init__(self, cfg: Qwen2Config, weights: dict[str, torch.Tensor], shard: Shard, device: str, max_seq: int,
                 dtype: torch.dtype = torch.bfloat16, layers: tuple[int, int] | None = None,
                 embed_rows: tuple[int, int] | None = None, head_rows: tuple[int, int] | None = None):
        self.cfg, self.shard, self.device, self.dtype, self.max_seq = cfg, shard, device, dtype, max_seq
        l0, l1 = layers if layers else (0, cfg.num_hidden_layers)
        self.layers = [Layer(i, weights, cfg, shard, device, dtype, max_seq) for i in range(l0, l1)]
        self.embed = weights.get("model.embed_tokens.weight")
        self.norm = weights.get("model.norm.weight")
        self.lm_head = weights.get("lm_head.weight")
        self.embed_rows = embed_rows or ((0, cfg.vocab_size) if self.embed is not None else None)
        self.head_rows = head_rows or ((0, cfg.vocab_size) if self.lm_head is not None else None)

    def embed_tokens(self, ids: list[int]) -> torch.Tensor:
        """Embedding of `ids`. With a vocab slice, rows outside it are exact zeros; the group's all-reduce completes it."""
        t = torch.tensor(ids, device=self.device)
        v0, v1 = self.embed_rows
        if (v0, v1) == (0, self.cfg.vocab_size):
            return F.embedding(t, self.embed)
        inside = (t >= v0) & (t < v1)
        e = F.embedding((t - v0).clamp(0, max(0, v1 - v0 - 1)), self.embed)
        return torch.where(inside[:, None], e, torch.zeros_like(e))

    @torch.inference_mode()
    def layers_forward(self, h: torch.Tensor, pos0: int, comm: Comm) -> torch.Tensor:
        T = h.shape[0]
        if pos0 + T > self.max_seq:
            raise ValueError(f"sequence {pos0 + T} exceeds max_seq_len {self.max_seq}")
        pos = torch.arange(pos0, pos0 + T, device=self.device)
        cos, sin = rotary(pos, self.cfg.head_dim, self.cfg.rope_theta, h.dtype, self.device)
        for layer in self.layers:
            h = layer.forward(h, pos0, cos, sin, comm)
        return h

    @torch.inference_mode()
    def logits_last(self, h: torch.Tensor) -> torch.Tensor:
        """Logits of the last position over this rank's head rows (all rows for a whole head)."""
        x = rms_norm(h[-1:], self.norm, self.cfg.rms_norm_eps)
        return F.linear(x, self.lm_head)[0]

    def head_argmax(self, h: torch.Tensor) -> tuple[float, int, torch.Tensor]:
        """(max logit, global index of its first occurrence, local logits). Greedy over this rank's vocab rows."""
        lg = self.logits_last(h)
        m = lg.max()
        i = int(torch.nonzero(lg == m)[0, 0].item())
        return float(m.float().item()), self.head_rows[0] + i, lg


def pick_global(cands: list[tuple[float, int]]) -> int:
    """Greedy token from (value, global index) per head rank: the largest value, ties to the lowest index."""
    return min(cands, key=lambda c: (-c[0], c[1]))[1]
