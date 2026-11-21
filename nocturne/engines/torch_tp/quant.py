"""Quantized weights kept on the device (engine GGUF). Torch ops only, no custom kernels.
A QuantLinear holds the raw GGUF blocks as uint8 [rows, blocks_per_row, type_size] and expands to the compute dtype just
before its matmul, in row chunks, so only one chunk is ever dense. Unquantized types (F32, F16, BF16) keep their own dtype.
Layouts follow ggml (gguf-py `gguf.quants` is the independent reference in the tests). Row split layers may be cut only at
block boundaries (BLOCK[qtype] input columns); column split layers may be cut at any row."""
import torch
import torch.nn.functional as F

QK_K = 256
# qtype name -> (values per block, bytes per block)
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q4_0": (32, 18), "Q5_0": (32, 22),
         "Q4_K": (QK_K, 144), "Q5_K": (QK_K, 176), "Q6_K": (QK_K, 210)}
DENSE = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
CHUNK_ELEMS = 1 << 24              # dense elements expanded at once (bf16 chunk 32 MiB); the planner uses the same number
CHUNK_WORKING_BYTES_PER_ELEM = 24           # measured peak per element incl. the bf16 result: Q8_0 10, Q6_K 12, Q4_K 14, Q5_K 15, Q5_0 22


class UnsupportedQuant(ValueError):
    pass


def check_supported(qtype: str) -> None:
    if qtype not in BLOCK:
        raise UnsupportedQuant(f"quantization type {qtype} is not supported; supported: {', '.join(BLOCK)}")


def _f16(b: torch.Tensor) -> torch.Tensor:
    """[..., 2] uint8 -> [..., 1] float32 (little endian float16)."""
    return b.clone(memory_format=torch.contiguous_format).view(torch.float16).to(torch.float32)


def _nibbles(qs: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """[n, groups*width] bytes -> [n, groups*2, width]: per group the low nibbles, then the high nibbles."""
    n = qs.shape[0]
    q = qs.reshape(n, groups, 1, width)
    both = torch.cat([q & 0x0F, q >> 4], dim=2)
    return both.reshape(n, groups * 2, width)


def _scale_min(sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 12 packed bytes of Q4_K and Q5_K -> 8 scales and 8 mins (6 bits each)."""
    n = sc.shape[0]
    s = sc.reshape(n, 3, 4)
    d, m, md = s[:, 0], s[:, 1], s[:, 2]
    scales = torch.cat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mins = torch.cat([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return scales, mins


def dequant_blocks(blocks: torch.Tensor, qtype: str) -> torch.Tensor:
    """[n, type_size] uint8 -> [n, block] float32."""
    n = blocks.shape[0]
    if qtype == "Q8_0":
        return blocks[:, 2:].view(torch.int8).to(torch.float32) * _f16(blocks[:, :2])
    if qtype == "Q4_0":
        q = _nibbles(blocks[:, 2:], 1, 16).reshape(n, 32).to(torch.int8) - 8
        return q.to(torch.float32) * _f16(blocks[:, :2])
    if qtype == "Q5_0":
        qh = blocks[:, 2:6].clone(memory_format=torch.contiguous_format).view(torch.int32).reshape(n, 1)       # clone: a one block slice keeps its byte offset
        bits = ((qh >> torch.arange(32, device=blocks.device, dtype=torch.int32).reshape(1, 32)) & 1)
        ql = _nibbles(blocks[:, 6:], 1, 16).reshape(n, 32).to(torch.int32)
        q = (ql | (bits << 4)) - 16
        return q.to(torch.float32) * _f16(blocks[:, :2])
    if qtype in ("Q4_K", "Q5_K"):
        d, dmin = _f16(blocks[:, 0:2]), _f16(blocks[:, 2:4])
        sc, mn = _scale_min(blocks[:, 4:16])
        dd = (d * sc.to(torch.float32)).reshape(n, 8, 1)
        dm = (dmin * mn.to(torch.float32)).reshape(n, 8, 1)
        if qtype == "Q4_K":
            q = _nibbles(blocks[:, 16:144], 4, 32).to(torch.float32)
        else:
            qh = blocks[:, 16:48].reshape(n, 1, 32)
            hi = (qh >> torch.arange(8, device=blocks.device, dtype=torch.uint8).reshape(1, 8, 1)) & 1
            q = (_nibbles(blocks[:, 48:176], 4, 32) | (hi << 4)).to(torch.float32)
        return (dd * q - dm).reshape(n, QK_K)
    if qtype == "Q6_K":
        ql = _nibbles(blocks[:, 0:128], 2, 64).reshape(n, 8, 32)            # per half: low nibbles of 64, high nibbles of 64
        qh = blocks[:, 128:192].reshape(n, 2, 1, 32)
        hi = ((qh >> torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 8, 32)
        q = ((ql | (hi << 4)).to(torch.int16) - 32).reshape(n, 16, 16).to(torch.float32)
        d = _f16(blocks[:, 208:210]) * blocks[:, 192:208].view(torch.int8).to(torch.float32)
        return (d.reshape(n, 16, 1) * q).reshape(n, QK_K)
    raise UnsupportedQuant(f"no block dequantizer for {qtype}")


def dequant(raw: torch.Tensor, qtype: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """raw [rows, blocks, type_size] uint8 -> [rows, blocks*block] in `dtype`, chunked by rows."""
    rows, nb, ts = raw.shape
    bs = BLOCK[qtype][0]
    out = torch.empty(rows, nb * bs, dtype=dtype, device=raw.device)
    step = max(1, CHUNK_ELEMS // (nb * bs))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        out[r0:r1] = dequant_blocks(raw[r0:r1].reshape(-1, ts), qtype).reshape(r1 - r0, nb * bs).to(dtype)
    return out


class QuantLinear:
    """Weight [out, in] stored as quantized blocks (or a dense tensor for F32, F16, BF16)."""

    def __init__(self, data: torch.Tensor, qtype: str):
        check_supported(qtype)
        self.qtype, self.data = qtype, data
        if qtype in DENSE:
            self.shape = tuple(data.shape)
        else:
            self.shape = (data.shape[0], data.shape[1] * BLOCK[qtype][0])

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def rows(self, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dense rows `idx` (embedding lookup): only those rows are expanded."""
        sel = self.data[idx]
        return sel.to(dtype) if self.qtype in DENSE else dequant(sel, self.qtype, dtype)

    def weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.data.to(dtype) if self.qtype in DENSE else dequant(self.data, self.qtype, dtype)

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out_f, in_f = self.shape
        if self.data.dtype == x.dtype and self.qtype in DENSE:
            return F.linear(x, self.data, bias)
        if out_f * in_f <= CHUNK_ELEMS:
            return F.linear(x, self.weight(x.dtype), bias)
        step = max(1, CHUNK_ELEMS // in_f)
        parts = []
        for r0 in range(0, out_f, step):
            r1 = min(out_f, r0 + step)
            w = self.data[r0:r1].to(x.dtype) if self.qtype in DENSE else dequant(self.data[r0:r1], self.qtype, x.dtype)
            parts.append(F.linear(x, w, None if bias is None else bias[r0:r1]))
        return torch.cat(parts, dim=-1)


def linear(x: torch.Tensor, w, b: torch.Tensor | None = None) -> torch.Tensor:
    return w.linear(x, b) if isinstance(w, QuantLinear) else F.linear(x, w, b)
