# Review request for Codex: engine GGUF, part 1

You did not write this code. Please look for errors, not for style. Rate each finding high, medium or low, and say how to check it.
Put your answer in `experiments/engine_gguf/REVIEW_CODEX.md` (Ohad pastes it). Short sentences. No em dash, no en dash.

## What changed

The torch engine (`nocturne/engines/torch_tp`) read only Qwen2 bf16 safetensors. It now also reads GGUF files, keeps the weights quantized on the device, and supports three architectures (qwen2, llama, qwen3) from one table.
Start SHA: `ef50523`. The diff is `git diff ef50523 -- nocturne scripts tests docs pyproject.toml`.

## Rules the code must follow (design constraints)

1. One engine. GGUF is a second weight source next to safetensors. Rank, transport, frame format, planner and hybrid code stay shared. No change to the transport or the frame format.
2. Weights stay quantized on the device. `QuantLinear` keeps the raw blocks and expands to bf16 just before its matmul, in row chunks. The transient buffer counts in the planner's workspace.
3. Row split layers (`o_proj`, `down_proj`, cut by input columns) may be cut only at block boundaries: 32 for Q8_0, Q4_0, Q5_0 and 256 for K quants. Column split layers (q, k, v, gate, up) can be cut at any row. The planner enforces it and names the rule in its message.
4. Byte accounting counts the stored blocks, plus the largest transient bf16 matrix per rank, plus the MPS rule (a tensor of 10 MiB or more lands in a 1 GiB heap; reported as `mps_floor` and `footprint`, not added to `total`).
5. Numerics as in session 3: RMSNorm in fp32, rotary in fp32 cast to the activation dtype, `scaled_dot_product_attention`, reduce dtype float32.
6. Tokenizer and EOS ids come from the GGUF.
7. The loader takes a tensor source interface (a local file today), so a later session can stream blocks.

## Files to read, in this order

| file | what to check |
|---|---|
| `nocturne/engines/torch_tp/quant.py` | Dequant math of Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K against ggml's block layouts (`ggml/src/ggml-quants.c`, `dequantize_row_*`). Nibble order, the 6 bit scale and min packing of Q4_K and Q5_K, the Q6_K high bits. Chunking. |
| `nocturne/engines/torch_tp/gguf_source.py` | Name and shape handling (GGUF stores `ne = [in, out]`). The un-permute of llama Q and K rows against `conversion/llama.py` (`LlamaModel.permute`). Block aligned column slices. Metadata to config: head size, rope dimension, rope scaling, EOS ids, tie. |
| `nocturne/engines/torch_tp/arch.py` | Name map. Which architectures the converter permutes. |
| `nocturne/engines/torch_tp/config.py` | HF config and GGUF metadata give the same `ModelConfig`. The Llama 3 rope factors against `generate_extra_tensors` in the converter. |
| `nocturne/engines/torch_tp/model.py` | Optional bias, Qwen3 q_norm and k_norm (per head, before rope), rope with factors, embedding lookup from quantized rows, head over quantized rows. Anything that changes the old Qwen2 path. |
| `nocturne/engines/torch_tp/loader.py` | `_wanted` shared by both sources, `_load_gguf`, byte budget, tied head. |
| `nocturne/engines/torch_tp/plan.py` | `Dims.units` (per stage average bytes), head units for block alignment, MLP granule as a multiple of the down block, workspace transient, mps floor. Any change to the numbers of an unquantized model. |
| `nocturne/engines/torch_tp/launch.py`, `hybrid_run.py` | Registry (`torch_model`, or a `.gguf` `path`), hashes of a single GGUF file. |
| `scripts/15_join.sh`, `15_join.ps1`, `16_join_apply.py` | llama.cpp check, models list, link type, networks by type, `lan_ip` on Linux. Quoting. bash 3.2 compatibility. |

## Questions

1. Dequant math: is any quant type wrong for some block content that the random block tests would not reach? Is the fp32 intermediate path the same as ggml's (`d * q - m`) in every type?
2. Name and permute map: does every tensor of a llama, qwen2 and qwen3 GGUF map to exactly one engine name? Is the Q and K un-permute right for K heads (GQA) and for biases? Is qwen3 really not permuted?
3. Block alignment: can a placement still cut `o_proj` or `down_proj` inside a block? Think about uneven weights, the `heads:` override, zero heads, a rank with the last partial granule, and layers that mix Q4_K and Q6_K.
4. Planner bytes: do the planned weight bytes equal the loaded bytes for every rank, for mixed type files? Is the transient estimate an upper bound for the largest matrix a rank expands (embedding rows, head rows, o_proj columns)?
5. Tokenizer and EOS: are the EOS ids from GGUF metadata enough for Qwen and Llama 3 chat? What does `eot` and `eom` change?
6. Old behavior: does any change alter the result of a bf16 safetensors Qwen2 run (same ids, same frames, same planned bytes)? Check `Dims` defaults, `plan()` rounding (`int()` of float sums), and the removal of the `model_type == qwen2` check.
7. Anything unsafe: a port bound to `0.0.0.0`, a path or shell string built from untrusted input, a report field used without validation.

## How to run the tests

Python is the torch venv `~/dev/nocturne-torch/bin/python` (torch 2.14.1, MPS and CPU, `gguf` from `~/dev/llama.cpp/gguf-py`). Do not install or upgrade torch.

```bash
cd nocturne
~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_engine_gguf.py -q          # dequant against gguf-py and transformers, name map
~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_engine_gguf_models.py -q    # tiny qwen2, llama, qwen3 made by llama.cpp (about 4 minutes, first run builds 1.1 GB under ~/.cache/nocturne-tests)
~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_join_tools.py tests/bench/test_hybrid.py tests/bench/test_torch_tp.py -q
```

`experiments/engine_gguf/VERIFY.md` lists what each test is checked against and which deliberate break each test caught (`mutate.py`). Think of a break that the tests would miss.
