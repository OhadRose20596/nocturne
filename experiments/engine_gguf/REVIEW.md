# Review (pass 1)

Reviewer: independent, read the diff ef50523..HEAD against the reference code. No source file edited. Snippets run on CPU with tiny tensors only.

## Findings

Count: high 0, medium 3, low 7.

### F1 (medium) EOS ids are only the single GGUF key
File: nocturne/engines/torch_tp/gguf_source.py, lines 422 to 426 (`config()`).
What is wrong: the EOS set is `tokenizer.ggml.eos_token_id` plus `eot` and `eom` if present. The converter (gguf-py `vocab.py`, `special_token_types`) writes eot or eom only for unusual TemplateProcessing, not for Qwen or Llama 3. So the set has one id. The safetensors path used config.json plus generation_config.json (Qwen2.5 Instruct: 151645 and 151643; Llama 3.1 Instruct: 128001, 128008, 128009).
Concrete failing cases: a Llama 3.0 Instruct GGUF converted from a tokenizer_config whose eos_token is `<|end_of_text|>` (128001) gives eos {128001}. The engine never stops at `<|eot_id|>` (128009) and runs to max_tokens. Qwen Instruct stops at `<|im_end|>` but not at `<|endoftext|>`. llama.cpp itself stops on EOG tokens found by token text, the engine does not.
How to check: `GgufSource(path).config().eos_token_ids` on a Llama 3 Instruct GGUF, compare with `generation_config.json` of the HF repo. Fix idea: also add token ids found by text in `tokenizer.ggml.tokens` for `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|eom_id|>`, `<|end_of_text|>`.

### F2 (medium) `python -m nocturne.engines.torch_tp plan` ignores the GGUF quant info
File: nocturne/engines/torch_tp/__main__.py, lines 55 to 65.
What is wrong: `launch.py` builds `Dims.of(cfg, weight_meta(path))`, but the `plan` subcommand calls `plan(cfg, p)`, which does `Dims.of(cfg)` with no meta. For a `.gguf` model the plan is then computed as bf16: wrong weight bytes, wrong budget verdict, no block alignment check (`o_block` and `down_block` stay 1). A placement accepted by `plan` can be rejected by `launch`, or the other way round. `format_plan(specs, Dims.of(cfg))` is fine, it needs only `hidden`.
How to check: run `plan` and `launch --force`-less on the same placement with a Q4_K_M GGUF and a heads override that is not a multiple of 2 for hd=128. `plan` prints a table, `launch` raises PlanError.

### F3 (medium) Transient estimate does not bound the dense (F16, F32) path
Files: nocturne/engines/torch_tp/quant.py lines 1195 to 1201 (`weight`, `linear`), plan.py `workspace_bytes` (the `d.quant is not None` branch).
What is wrong: for DENSE types `linear` does `F.linear(x, self.data.to(x.dtype))`, one full copy of the matrix, never chunked. The planner counts `min(matrix_elems, CHUNK_ELEMS) * 14`, at most 235 MB. For an F16 GGUF the head matrix of a rank with all vocab rows (152064 x 3584 = 545M elements) expands to 1.09 GB bf16. Same for an F32 2-D tensor (copy to bf16). BF16 files do not copy.
How to check: planned workspace of a one-rank F16 placement (Qwen2.5-7B shapes) against `vocab_rows * hidden * 2`.

### F4 (low) CHUNK_WORKING_BYTES_PER_ELEM (14) is below the measured peak for Q5_0, Q5_K, Q4_K
File: nocturne/engines/torch_tp/quant.py line 1092.
Measured on CPU with torch.profiler memory events, one `dequant` of 2M elements to bf16, peak live bytes per element including the bf16 output: Q8_0 10.1, Q4_0 11.1, Q6_K 12.3, Q4_K 14.3, Q5_K 15.3, Q5_0 22.3. Q5_0 holds int32 `bits` and `ql` (4 bytes each). At the 16M element cap the miss is about 138 MB for Q5_0 and 22 MB for Q5_K.
How to check: script in the scratchpad `rv/t5.py` (profile_memory, sum of `[memory]` events); run with the qtype as argument.

### F5 (low) Planned weight bytes can be 1 byte lower than loaded bytes for mixed type files
File: plan.py, `Dims.units` (float means over layers) and `plan()` `w = int(...)`.
What is wrong: `mean()` divides by the number of layers and `L * mean` is multiplied back in float, then `int()` truncates. With mixed layers (Q4_K and Q6_K) and L not a power of two the result is sometimes `x - 1e-9`, so the plan says one byte less than the loader counts. In 300 random trials with L in {7, 13, 24, 28, 32, 36} about 20 rank rows were off by exactly 1. The README says planned equals loaded.
How to check: scratchpad `rv/t1.py` (random QuantInfo, compares `planned["weights"]` with an independent per-layer sum). A test that asserts equality on a file with mixed per-layer types and 28 layers fails. Fix idea: sum exact integers per layer (keep per-layer bytes as ints, or round instead of truncate).

### F6 (low) `mlp_row` assumes ffn_up has the type of ffn_gate
File: plan.py line 955 (`mlp_row = mean(2 * row(q.gate[i]) + ...)`). `QuantInfo.up` is stored and never used.
For a file where ffn_up and ffn_gate differ in one layer (custom `--tensor-type`, dynamic quant files), planned bytes differ from loaded bytes by percent (up to 2.3 percent per rank in my random test), silently.
How to check: `rv/t1.py` with the `g_` line removed (up independent of gate). Fix: use `row(q.gate[i]) + row(q.up[i])`.

### F7 (low) `hybrid_run.transformers_check` with a GGUF
File: hybrid_run.py, lines 377 to 379.
`AutoTokenizer.from_pretrained(a.tokenizer_dir or tdir)` is plain transformers, not `load_tokenizer`. If `--tokenizer-dir` is a `.gguf` file, or if neither `--transformers-dir` nor `--tokenizer-dir` is given for a GGUF placement, `from_pretrained` receives a file path and fails with an unclear message. The main run path uses `load_tokenizer` and accepts the same argument, so the two paths disagree.

### F8 (low) CUDA OOM handler dropped on the GGUF load path
File: loader.py `_load_gguf` (diff lines 698 to 702).
The safetensors loop catches `torch.cuda.OutOfMemoryError` and sets `e.tensor_name`. `_load_gguf` does not. `RankRuntime.load` then reports "weights or KV cache" instead of the tensor name. Only the message changes.

### F9 (low) New peer with unknown link type is recorded as wifi
File: scripts/16_join_apply.py, `update_peers` new peer block (`link_key(f) or "wifi"`) and the empty `networks: {}` case.
When the report says `link_type=unknown`, a new peer gets `networks: {wifi: <ip>}`. That is a guess stored as a fact. A later wired report then adds `wired:` next to a possibly wrong `wifi:`. Existing non empty `networks` are left alone with a warning (correct). Constraint B6 says never overwrite one type with another; this is not an overwrite, but it mislabels.

### F10 (low) Behavior change for safetensors Qwen2 with a rope_scaling block
File: config.py `load_config` (diff lines 305 to 313).
Before, any `rope_scaling` was ignored. Now a Qwen2 config with `rope_scaling` of type yarn or linear raises ValueError. This is right numerically (the old run silently ignored the scaling), but it changes what a bf16 Qwen2 run does for such configs. Official Qwen2.5 configs have `rope_scaling: null`, so the standard runs are not affected.

## Checked and found correct

- Dequant math (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K): compared `dequant_blocks` with `gguf.quants.dequantize` on 64 blocks each for random bytes, all 0xFF, all zero, with finite fp16 scales: max abs difference 0.0 for all six. Also read against `dequantize_row_*` in ggml-quants.c for nibble order, the Q5_0 bit index (bit equals element index), Q5_K high bit per sub-block, Q6_K layout (ql/qh/scales, 16 element scale groups), get_scale_min_k4 packing.
- Row chunking in `dequant`: chunked result with CHUNK_ELEMS=300 on a 5x3 block tensor equals the reference exactly. `step` is at least 1.
- Q and K un-permutation: `_unpermute_index` checked against `LlamaModel.permute` for (nh, nkv, hd) in (8,2,16), (4,4,8), (32,8,128), for q with nh heads and k with nkv heads, for row slices starting at 0, mid head and exactly at a head start. All equal. Biases use the same index. K uses n_kv_heads (the converter's `permute(w, n_head, n_kv_head)`).
- Qwen2 and Qwen3 are not permuted: `undo_permute = True` only on LlamaModel (conversion/llama.py line 35); Qwen2Model and Qwen3Model(Qwen2Model) have no permute; Llama4 and Apertus set it False and have their own arch names.
- Name map: every tensor name of llama, qwen2, qwen3 (token_embd, output_norm, output, blk.N.* including attn_q_norm and attn_k_norm, biases, rope_freqs ignored) maps to one engine name; duplicate and unknown names raise. Tied embeddings: missing `output.weight` gives `tie_word_embeddings=True` and the head uses the embedding tensor in both `plan` (head type = embed type, counted once) and the loader.
- rope_freqs and `llama3_factors`: formula equals `generate_extra_tensors` in conversion/llama.py (branch order, smooth term, 1/((1-s)/f+s)). Dividing `inv` by the factors equals ggml freq_factors and HF llama3 rope up to fp32 rounding. Partial rotary and unsupported scaling types raise.
- Qwen3 q_norm and k_norm: per head RMSNorm on the head_dim axis, before rope, same `rms_norm` as the layer norms (fp32 compute, weight times bf16 result), matches modeling_qwen3. Scale is 1/sqrt(head_dim). Weights are replicated (slice_for returns None). `head_dim` is used for rotary, caches, o_proj reshape, planner units and `SliceSpec`.
- Block alignment (question 3): `hu = o_block / gcd(o_block, hd)` head units, MLP granule `lcm(G, down_block)`, heads override check raises PlanError with the rule named. Boundaries inside `_ranges` are multiples of the unit or clamp to the total, and the total in_f is a multiple of the block (the reader raises otherwise). Blocks are 1, 32 or 256, so the max over mixed layers is a multiple of every block. Zero heads and zero MLP rows: boundaries 0 or the total are not checked and are aligned; model code skips the matmul (`if self.nh`, `if self.wg.shape[0]`), so the zero column `dequant` (division by zero for nb=0) is never reached. Random trials (300, mixed types, uneven weights): no misaligned boundary found in realistic shapes.
- Bf16 Qwen2 safetensors (question 6): `Units` for `quant is None` equals the old formulas term by term (q_head, kv_head, mlp_row, norms, vocab row), `Gm == G`, `hu == 1`, `largest_remainder(sh, n_heads)` as before, vocab counting (embed plus head, once when tied) same, workspace unchanged (the quant term is guarded), `planned` only gains `mps_floor` and `footprint` (not in `total`). `Dims` new fields have defaults and come after the old ones; `ModelConfig` fields appended after the old ones. rotary with `factors=None` and the F.embedding path are unchanged. Layer constructor: bias optional, Qwen2 still checked by `arch.qkv_bias`.
- Tied or separate head, embedding lookup with QuantLinear: `rows(idx)` dequantizes only the selected rows; the slice offset logic is the same as the F.embedding path.
- MPS rule: `mps_floor` and `footprint` are informational and documented in the README as not in `total`; consistent with the README text.
- Join tool: `bash -n` passes; constructs used (`${x:+,}`, `case`, `$(...)`, `awk -v`) are bash 3.2 safe. The `--version` sed pattern matches this build's output (`commit 1e7bcf3da)`), and the pin compare is a prefix match. macOS hardware port awk (`substr($0,16)`) is right. Report values used later: `link_type` is accepted only if it equals wifi or wired (`link_key`); `host`, `user`, `ip` are validated before use; `REMOTE_LIST` goes through `shlex.quote`; ssh and scp lines are printed, not executed, with only local file names and a validated target. `merge_networks` never touches other types and keeps order; an existing non empty `networks` with unknown link type is left alone with a warning. `lan_ip()` on Linux uses a list argument for `ip addr` and falls back to `hostname -I`.

## Not checked

- Execution on MPS or CUDA: whether `uint8 >> uint8`, `view(torch.int32)` on a uint8 slice, and `view(torch.float16)` work on MPS for the dequant code. Only CPU was run.
- Real tokenizer behavior: `AutoTokenizer.from_pretrained(dir, gguf_file=...)` on a Llama 3 or Qwen GGUF (transformers 5.21 has qwen2, qwen3, llama mappings including chat_template; not run because no model file was loaded).
- `GgufSource` on a real GGUF (memmap behavior, `contents()` return types for scalar fields); only code reading plus gguf-py source.
- Q5_0 and Q6_K checked by reading C and gguf-py, not by running the C code.
- Test files were only scanned for coverage; docs and README claims other than the byte accounting sentence were not checked.
- Windows `15_join.ps1` was read, not run (regex for media type looks right for 802.11 and 802.3 names).

## Disposition (author, after pass 1)

Commit 5f3f255 fixes F1 to F9. A second fresh reviewer checked the fixes only (`REVIEW_pass2.md`): F1 to F9 all fixed, no new high or medium finding, five low findings.

| id | rating | what was done |
|---|---|---|
| F1 | medium | Fixed. End tokens found by text in the vocabulary (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|eom_id|>`, `<|end_of_text|>`) are added to the EOS set. Test: Qwen 1.5B Q8_0 gives {151643, 151645}. |
| F2 | medium | Fixed. The `plan` subcommand builds `Dims` with the GGUF tensor metadata, as `launch` does. |
| F3 | medium | Fixed. Dense F16 and F32 matrices are expanded in row chunks like quantized ones; the constant is 24 bytes per element (F4). |
| F4 | low | Fixed. `CHUNK_WORKING_BYTES_PER_ELEM` is 24, above the measured peak of every type (Q5_0 22.3). |
| F5 | low | Fixed. `round()` instead of `int()`. Test with mixed layer types and 7, 13, 28, 36 layers. |
| F6 | low | Fixed. `mlp_row` uses the gate and the up type separately. |
| F7 | low | Fixed. `transformers_check` uses `load_tokenizer`. |
| F8 | low | Fixed. The GGUF load path names the tensor on a CUDA out of memory. |
| F9 | low | Fixed. A new peer with an unknown link type gets empty `networks`, with a warning. |
| F10 | low | Not a bug, left on purpose. A Qwen2 config with yarn or linear rope scaling now raises; before, the scaling was silently ignored. Official Qwen2.5 configs have `rope_scaling: null`. |

Pass 2 low findings, left open: a wrong EOS id on a base model whose vocabulary holds a token with an EOS text that is not an end token (the five texts are all end tokens in the families supported); the F1 test is weak on a model without the file (it needs `~/models`); the doc line for unknown link on a new peer (added in `docs/JOIN.md`); the 24 byte constant is conservative; the byte test tolerates 0.5. None changes a result.
Gate: no open high finding.
