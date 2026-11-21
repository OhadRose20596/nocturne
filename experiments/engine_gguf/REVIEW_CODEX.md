# Codex review: engine GGUF, part 1

Reviewed HEAD `7015256` against `ef50523`, including the working tree.
Review only. No engine or script code was changed.

## Findings

### 1. High: GGUF filenames are executed as shell syntax

Locations: `nocturne/engines/torch_tp/hybrid_run.py:65-74` and `scripts/16_join_apply.py:417-426`.

The new GGUF path is inserted into shell commands without quoting.
A filename with spaces breaks preflight hashing.
A filename containing a semicolon can execute a command locally or on a peer.
The printed `scp` and remote checksum commands have the same problem.
The report validation does not protect these filenames.
They come from the model path or local directory listing.

How to check: create a temporary file named `model space.gguf`.
Run `hash_cmd('Linux', str(path))` through Bash.
It fails because `sha256sum` receives two filenames.
I also used the harmless filename `model; printf CODEX_REPRO; #.gguf`.
The hash command printed `CODEX_REPRO` and returned success without hashing the file.
For `gguf_commands`, use that filename and replace `scp` with a shell function that does nothing.
The generated copy command still runs the `printf`.

Quote paths for each shell boundary, including the remote shell.
Keep the safetensors glob separate from literal GGUF filenames.
Also fix `model_hashes` at line 90 when adding space support.
Its `split()[1]` truncates filenames at the first space.

### 2. Medium: a single Q5_0 block crashes

Location: `nocturne/engines/torch_tp/quant.py:59`.

The high bits start at byte offset 2.
For one block, `blocks[:, 2:6]` is already contiguous.
Calling `contiguous()` keeps that offset.
The int32 view then fails because the offset is not divisible by four.
This affects a 32-column embedding lookup of one row.
It also affects any dequant chunk containing exactly one block.

How to check, with the supplied venv:

```python
import torch
from nocturne.engines.torch_tp.quant import dequant_blocks
dequant_blocks(torch.zeros(1, 22, dtype=torch.uint8), "Q5_0")
```

Observed: `RuntimeError: self.storage_offset() must be divisible by 4 ... got 2`.
Two blocks succeed.
The random tests use 300 blocks.
The chunk test uses four blocks per row.
Neither reaches this case.
Copy into aligned storage or unpack the four bytes without an int32 view.

### 3. Medium: accepted Llama bias settings give wrong results and bytes

Locations: `nocturne/engines/torch_tp/config.py:72-102`, `model.py:56-65,110`, `arch.py:37`, and `plan.py:118-129`.

Removing the Qwen2 restriction admits Llama configs with `attention_bias=True`.
The loader reads their q, k, v and o biases.
The model applies only the q, k and v biases.
It silently ignores `self_attn.o_proj.bias`.
The planner counts none of these biases because the architecture table says Llama has no qkv bias.
The corresponding GGUF output bias also has no name-map entry.
It raises instead of loading.

How to check: make a tiny transformers `LlamaForCausalLM` with `attention_bias=True`.
Use hidden size 32, intermediate size 64, one layer, two query heads, one KV head, head size 16 and vocab size 64.
Use bf16 and set its output projection bias to `torch.arange(32) / 8`.
Save it with `save_pretrained`, then use `load_config`, `load_slices` and `Qwen2TP`.
Compare last logits for ids `[1, 2, 3]` with transformers.
With seed 9, I measured max absolute error `0.484375`.
The loader reported 27,008 bytes.
The planner reported 26,816 bytes.
The missing 192 bytes are the four attention biases.

Honor the optional biases and count the tensors that exist.
Alternatively, reject unsupported bias configurations before loading.
The existing tiny Llama fixture uses the default `attention_bias=False`.

### 4. Low: every GGUF rank gets the MPS floor, even below 10 MiB

Location: `nocturne/engines/torch_tp/plan.py:377-382`.

`d.quant is not None` makes `big` true for every GGUF.
This includes a BF16 GGUF and tiny quantized ranks.
The stated rule requires an individual tensor of at least 10 MiB.
The reported footprint can therefore jump to 1 GiB with no qualifying tensor.
This does not change `total` or the budget check.

How to check: plan one MPS rank with hidden size 32, intermediate size 64, one layer, two query heads, one KV head, head size 16 and vocab size 64.
Use Q8_0 for all matrices, no biases, `prefill_max=4` and `max_seq_len=16`.
I obtained `total=71808` and `mps_floor=footprint=1073741824`.
All stored tensors and all dequant intermediates are far below 10 MiB.
Apply the threshold to actual stored tensors and transient allocations.

## Checks and limits

The requested five test files passed with `~/dev/nocturne-torch/bin/python`.
Result: **126 passed, 1 warning in 288.93 seconds**.
Files: `test_engine_gguf.py`, `test_engine_gguf_models.py`, `test_join_tools.py`, `test_hybrid.py` and `test_torch_tp.py`, all under `tests/bench`.
The warning concerns a non-writable NumPy array passed to `torch.from_numpy` in `gguf_source.py:163`.
Torch was not installed or upgraded.

`/bin/bash -n scripts/15_join.sh` passed under Bash 3.2.57.
PowerShell was inspected but not executed.
The targeted reproductions above ran locally on CPU.
No remote commands, model copies or join configuration changes were made.

I inspected the six `dequantize_row_*` functions in `~/dev/llama.cpp/ggml/src/ggml-quants.c`.
The nibble order, K scale packing, Q6 high bits and fp32 arithmetic order agree with the implementation.
An extra CPU check used 2,048 random blocks per type with scales sampled across finite fp16 bit patterns.
All six types matched gguf-py exactly in fp32.
The Q5_0 finding concerns storage alignment.
I also checked the permutation and Llama 3 factors against `~/dev/llama.cpp/conversion/llama.py`.
The inverse uses the KV head count for K and also covers Q and K biases.
The Qwen2 and Qwen3 paths do not require this permutation.

I found no additional block boundary error for supported block sizes 1, 32 and 256.
The maximum block size is a common multiple of these sizes.
The head override is checked at internal boundaries.
The MLP granule is aligned to the down block.
The loader also rejects unaligned column slices.
The mixed gate and up accounting and rounding fixes are present.
The EOS set includes GGUF EOS, EOT and EOM metadata and the five listed token texts.
No transport or frame-format change appears in the requested diff.

The transient formula test compares the planner with its own constants.
It does not independently prove a peak-memory bound on every backend.
I did not measure such a bound or repeat the recorded hardware runs in `VERIFY.md`.

## Disposition (author)

All four findings were reproduced as tests, then fixed. Full `tests/bench`: 260 passed.

| # | rating | what was done |
|---|---|---|
| 1 | high | Fixed. `hash_cmd` and `stat_cmd` quote the directory and the file name (a leading `~/` stays expandable); the safetensors glob stays a glob. The cache parser keeps names with spaces (`split(None, 1)`). `gguf_commands` quotes every path and the remote command; the copy target is `host:models/<dir>/` relative to the remote home. Tests run the generated lines in bash with a hostile file name and a name with a space. |
| 2 | medium | Fixed. The int32 and fp16 views are taken from a contiguous clone, so a one block slice no longer keeps a bad byte offset. Test: 1, 2 and 3 blocks of every type against gguf-py. |
| 3 | medium | Fixed by rejection. A config with `attention_bias` or `mlp_bias` raises a clear error at `load_config`; the engine applies only the qwen2 q, k, v bias. Supporting the o_proj bias is a separate feature. |
| 4 | low | Fixed. The 1 GiB MPS floor needs a tensor of 10 MiB or more: the largest matrix, capped at one dequant chunk for quantized weights. A tiny rank reports no floor. |
