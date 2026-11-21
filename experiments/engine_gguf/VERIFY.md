# Engine GGUF, part 1: verification record

Machine: M5 (24 GB, MPS), torch 2.14.1, transformers 5.21.0, `gguf` 0.19.0 from `~/dev/llama.cpp/gguf-py` (the one that ships with llama.cpp 1e7bcf3da; installed editable into `~/dev/nocturne-torch` with uv, torch untouched). Start SHA of the change: `ef50523`.
Predictions for V3 were committed before any V3 run: `PREDICTIONS.md` (commit 702a215).

## V1. Unit tests, each against an implementation we did not write

Run: `~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_engine_gguf.py tests/bench/test_engine_gguf_models.py tests/bench/test_join_tools.py`.

| what | reference (not ours) | result | file |
|---|---|---|---|
| Dequant of Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K on 300 random blocks each (valid fp16 scales, random payload) | `gguf.quants.dequantize` (numpy) | exact (bit equal in float32) for all six | `test_engine_gguf.py::test_dequant_random_blocks_exact` |
| Same, chunked and unchunked | itself | equal | `test_dequant_chunked_matches_unchunked` |
| Real tensors, 4 rows of every tensor kind and type of the GGUF files under `~/models` smaller than 2 GB (0.5B Q8_0, 0.5B and 1.5B Q4_K_M) | `gguf.quants.dequantize` | exact | `test_real_tensors_match_gguf_py` |
| Names and un-permute, 40 tensors of the 0.5B files (Q8_0 and Q4_K_M) | transformers' own GGUF name map and reverse permute (`load_gguf_checkpoint`). Its values come from gguf-py's dequantizer too, so for values it is not a third reading | exact | `test_real_tensors_match_transformers_loader`. `from_pretrained(gguf_file=)` needs `accelerate`, a dependency we may not add |
| Name map covers every tensor of every GGUF under `~/models` (10 files, 0.5B to 32B, qwen2 and qwen3) exactly once, none left over, every layer has the expected set | the file's own tensor list, plus `gguf.get_tensor_name_map` for the names | pass | `test_name_map_covers_every_tensor_exactly_once`, `test_name_map_agrees_with_gguf_py_tensor_name_map` |
| Llama Q and K un-permute: a tiny random llama (Llama 3 rope scaling) written as safetensors and converted with `convert_hf_to_gguf.py --outtype bf16`. The engine's tensors after load against the safetensors tensors | the safetensors file, the converter | every tensor bit equal (all of q, k, v, o, gate, up, down, norms, embedding, head), for qwen2, llama and qwen3. A guard test shows the stored rows differ from HF order for llama and not for qwen2 | `test_bf16_gguf_weights_equal_safetensors_exactly`, `test_llama_permute_really_happens_in_the_file` |
| Un-permute of a Q8_0 llama and qwen2 file | transformers' `_reverse_permute_weights` through `load_gguf_checkpoint` | exact | `test_unpermute_agrees_with_transformers_own_reverse_permute` |
| Config from GGUF metadata against config from the HF `config.json` (dims, head size, theta, tie, rope factors) | the HF config | equal. The Llama 3 factors equal the converter's `rope_freqs.weight` to 1e-6 | `test_config_from_gguf_matches_hf_config`, `test_llama3_factors_equal_the_converters_rope_freqs` |
| Rotary cos and sin at positions 0 to 8000 for Llama 3 scaling | `LlamaRotaryEmbedding` of transformers | within 2 bf16 ulps, and the unscaled rope is far off (the check has teeth) | `test_llama3_rotary_matches_transformers_at_long_positions` |
| Head size from `attention.key_length` (head size 64, hidden 256, 8 heads) | the HF config | pass | `test_qwen3_head_size_comes_from_key_length_not_hidden_over_heads` |
| Engine, world 1, bf16, safetensors, last position logits | transformers forward on the same files | same argmax, max abs error within 5 percent of the max logit, qwen2, llama, qwen3 | `test_engine_on_safetensors_matches_transformers` |
| Engine on the bf16 GGUF against engine on the safetensors | itself | equal logits, same argmax, all three | `test_engine_on_bf16_gguf_equals_engine_on_safetensors` |
| Engine on the Q8_0 and Q4_K_M GGUF (made by llama.cpp's converter and `llama-quantize`, mixed Q4_K, Q6_K, F32) against the same weights expanded by gguf-py into dense bf16 | gguf-py dequantizer | logits bit equal, six cases | `test_quantized_path_equals_dense_path_on_gguf_py_dequantized_weights` |
| Block aligned and unaligned reads of a row split tensor | slices of the full read | equal, and an unaligned column range is refused with the block size | `test_sharded_reads_equal_slices_of_the_full_read` |
| Planned weight bytes against loaded bytes for every rank of a two rank split, all three architectures, Q8_0 and Q4_K_M files | the loader | within 0.1 percent (0 to 64 bytes) | `test_planned_weight_bytes_equal_loaded_bytes_for_every_rank` |
| Block alignment: the planner rejects a head split inside a 256 column block and names the rule; Q8_0 allows any split on 32; the planner's own split moves whole blocks; the MLP granule is a multiple of the down block | the rule (32 for Q8_0, Q4_0, Q5_0; 256 for K quants) | pass | `test_planner_*` |
| Transient dequant bytes and MPS floor in the plan | the formula | exact difference to the dense plan | `test_transient_and_mps_floor_in_the_plan` |
| Join tool: `update_peers` with two network types (wired report keeps wifi and hotspot, wifi report keeps wired, unknown leaves networks alone, new peer gets its type), report parsing with the new fact lines and a real M5 report, Linux `lan_ip` parsing, GGUF listing and copy commands with sha256, a dry run and a real apply on copies of the real `peers.yaml` files and an ssh config | hand written expectations | 16 pass | `test_join_tools.py` |
| Old tests (bf16 engine, planner, hybrid, bench) | themselves | all pass (238 in the full `tests/bench` run after the fixes) | `tests/bench` |

Known limit of V1: the tiny models are random. They test the math and the plumbing, not the quality of a real file. V3 does that.

## V2. Mutation check

17 deliberate breaks, each applied to a scratch branch (thrown away), each run against a test selection. `mutate.py` does it. Round 1 missed three breaks. A test was written for each (rope scaling at long positions, head size from `key_length`, exact transient bytes) and round 2 ran those three again. Raw results: `mutations_round1.json`, `mutations_round2.json`.

| break | file | first test that failed |
|---|---|---|
| Q4_K: swap the two halves of the scale and min unpacking | `quant.py` | gguf::test_dequant_random_blocks_exact[Q4_K] |
| Q8_0: drop the block scale | `quant.py` | gguf::test_dequant_random_blocks_exact[Q8_0] |
| Q4_0: offset 7 instead of 8 | `quant.py` | gguf::test_dequant_random_blocks_exact[Q4_0] |
| Q5_0: high bit shifted by 3 | `quant.py` | gguf::test_dequant_random_blocks_exact[Q5_0] |
| Q5_K: high bit shifted by 3 | `quant.py` | gguf::test_dequant_random_blocks_exact[Q5_K] |
| Q4_K: add the min instead of subtracting it | `quant.py` | gguf::test_dequant_random_blocks_exact[Q4_K] |
| Q6_K: bias 31 instead of 32 | `quant.py` | gguf::test_dequant_random_blocks_exact[Q6_K] |
| llama: skip the Q and K un-permute | `gguf_source.py` | models::test_bf16_gguf_weights_equal_safetensors_exactly[llama] |
| column slice of a row split layer: one block too many | `gguf_source.py` | models::test_sharded_reads_equal_slices_of_the_full_read[qwen2] |
| planner: block alignment check off by one head | `plan.py` | models::test_planner_rejects_unaligned_head_split_and_names_the_rule |
| planner: MLP granule ignores the down_proj block | `plan.py` | models::test_planner_aligns_head_splits_to_k_quant_blocks |
| planner: no transient dequant buffer in the workspace | `plan.py` | MISSED in round 1. Round 2 (after a new test): models::test_transient_and_mps_floor_in_the_plan |
| planner: count every layer at layer 0 quant type | `plan.py` | models::test_planned_weight_bytes_equal_loaded_bytes_for_every_rank |
| qwen3: skip the per head q_norm and k_norm | `model.py` | models::test_engine_on_safetensors_matches_transformers[qwen3] |
| llama: ignore the Llama 3 rope factors | `model.py` | MISSED in round 1. Round 2 (after a new test): models::test_llama3_rotary_matches_transformers_at_long_positions[hf] |
| qwen2: drop the q, k, v bias | `model.py` | models::test_engine_on_safetensors_matches_transformers[qwen2] |
| GGUF config: head size from hidden // heads, ignoring attention.key_length | `gguf_source.py` | MISSED in round 1. Round 2 (after a new test): models::test_qwen3_head_size_comes_from_key_length_not_hidden_over_heads |

## V3. End to end on the M5 (MPS, loopback), run folders in `runs/`

Commands: `python -m nocturne.engines.torch_tp.hybrid_run --placement experiments/engine_gguf/placements/<p>.yaml --session-dir experiments/engine_gguf --run-id <id> [--reference <run>] [--transformers --transformers-dir <hf dir> --tokenizer-dir <hf dir>]`.
GGUF files were made with `convert_hf_to_gguf.py <hf dir> --outtype q8_0` (llama.cpp 1e7bcf3da, a stub `sentencepiece` module on PYTHONPATH for the converter only): Qwen2.5-1.5B Q8_0 sha256 5926a692b27bc90f12815ce8326593becea3dae459cf5c250579d7ce0ffff8d7, Qwen2.5-7B Q8_0 862574978752413e9d54e4d104c455797f9c7e477ee256a46e2d8a31f9242cd9, Llama-3.2-1B-Instruct Q8_0 8496a0324097b36ee86c4eb66abe04dad0f60611c404d8f892687be76018251e, Qwen3-0.6B Q8_0 0aa8ab391383f6d9efbd4066894be2a826dee7a5c998b0f34c89b98b257e5081. Llama 3.2 1B Instruct comes from the ungated mirror `unsloth/Llama-3.2-1B-Instruct` (Llama 3.2 Community License, 2.5 GB), Qwen3 0.6B from `Qwen/Qwen3-0.6B` (Apache 2.0, 1.5 GB), wikitext-2 raw from the llama.cpp CI dataset `ggml-org/ci` (CC BY-SA 3.0).

### Regression, one rank, loopback and engine against engine

| run | placement | quant | file sha256 | tokens (eval prompts, 8; long prompt) | frames per step measured / predicted | loaded bytes vs planned weights | process footprint / planned footprint per rank |
|---|---|---|---|---|---|---|---|
| A1_P3_loopback_qwen2.5-1.5b-bf16_r1 | 1.5B TP3 loopback | bf16 safetensors | - | 8/8 vs recorded P3_loopback r1; long: identical | 346.0 / 346 | equal | 1.38, 1.21, 1.23 |
| A2_P1_loopback_qwen2.5-7b-bf16_r1 | 7B TP2 3:2 loopback | bf16 safetensors | - | 8/8 vs recorded P1_loopback r1; long: identical | 116.0 / 116 | equal | 1.09, 1.07 |
| B1_q8_1.5b_1rank_r1 | 1.5B, one rank | Q8_0 | 5926a692 | 8/8 vs bf16 R0; long: diverges at 54 | 0.0 / 0 | equal | 1.76 |
| B2_q8_7b_1rank_r1 | 7B, one rank | Q8_0 | 86257497 | 8/8 vs bf16 R0; long: diverges at 49 | 0.0 / 0 | equal | 1.17 |
| B4_q4km_7b_1rank_r1 | 7B, one rank | Q4_K_M | 65b8fcd9 (full 65b8fcd92af6b4fe...) | 8/8 vs bf16 R0; long: diverges at 5 | 0.0 / 0 | equal | 1.30 |
| C1_q8_7b_pp3_r1 | 7B PP 3 stages | Q8_0 | 86257497 | 8/8 vs one rank run B2; long: identical | 3.0 / 3 | equal | 1.28, 1.21, 1.27 |
| C2_q8_7b_tp3_r1 | 7B TP 3 ranks | Q8_0 | 86257497 | 8/8 vs one rank run B2; long: diverges at 162 | 346.0 / 346 | equal | 1.53, 1.49, 1.52 |
| C3_q8_7b_hybrid_r1 | 7B hybrid TP2 + 1 | Q8_0 | 86257497 | 8/8 vs one rank run B2; long: diverges at 49 | 61.0 / 61 | equal | 1.55, 1.51, 1.15 |

Reading: every row meets its criterion. The bf16 path reproduces the recorded runs token for token (A1, A2), planned bytes unchanged. C1 (PP) is bit identical to the one rank run (logit difference 0). Loaded bytes equal planned weight bytes on every rank of every run. Footprint ratios: 1.07 to 1.76. The small 1.5B rank has a fixed overhead (1.76); on 7B ranks the ratio is 1.07 to 1.55. My prediction C6 (0.9 to 1.3) missed for TP ranks (about 1.5): the TP ranks also hold the KV cache, the dequant chunk heaps and the MPS 1 GiB heaps. Not a gate.

### Engine against llama.cpp on the same file (greedy, token ids, llama-server Metal)

| file | eval prompts identical | long prompt first divergence |
|---|---|---|
| 1.5B Q8_0 | 8/8 | 54 |
| 7B Q8_0 | 8/8 | 49 |
| 7B Q4_K_M | 8/8 | 5 |

llama.cpp keeps its own tokens close to the bf16 reference longer than our engine does (7B Q8_0: llama.cpp against bf16 diverges at token 73, our engine at 49; 7B Q4_K_M: 45 against 5). We expand weights to bf16 and multiply bf16 activations; llama.cpp keeps fp32 activations. The perplexity rows below show the gap is small.

### Perplexity, 4 chunks of 128 tokens of wikitext-2 test (offset 200000 characters), 252 scored tokens, same ids from llama-tokenize

| file | engine | llama-perplexity | difference |
|---|---|---|---|
| 7B Q8_0 | 10.869 | 10.939 | 0.6 percent |
| 7B Q4_K_M | 11.383 | 11.505 | 1.1 percent |

First chunk only (128 tokens): Q8_0 engine 10.908; Q4_K_M engine 12.273. llama.cpp's own estimate has an error bar of +/- 2.5 on this short sample; the two programs agree far inside it.

### Architectures, one rank, against transformers `generate` (bf16, MPS)

| run | model | eval prompts identical | divergences (token) |
|---|---|---|---|
| E1_llama32_1b_st_r1 | Llama 3.2 1B Instruct, safetensors | 8/8 | none |
| E2_llama32_1b_q8_r1 | Llama 3.2 1B Instruct, Q8_0 GGUF (sha 8496a032) | 8/8 | {'longopen_01': 0} |
| E3_qwen3_06b_st_r1 | Qwen3 0.6B, safetensors | 7/8 | {'extract_02': 26, 'longopen_01': 36} |
| E4_qwen3_06b_q8_r1 | Qwen3 0.6B, Q8_0 GGUF (sha 0aa8ab39) | 6/8 | {'arith_04': 4, 'extract_02': 26, 'longopen_01': 36} |

E2 long prompt: the first generated token of an open ended prompt differs (a near tie on a 1B model under Q8_0). The 8 eval prompts are identical. The tokenizer built from each GGUF gives the ids of the Hugging Face tokenizer on all 43 chat prompts for all four models (`records/tokenizer_check.jsonl`). The EOS id sets differ: a GGUF carries one id (Qwen: 151645; Llama 3.2: 128009), the Hugging Face generation config has 2 or 3. Fixed after review pass 1 by adding the end tokens found by text in the vocabulary.

### Join tool on the M5

`bash scripts/15_join.sh --check --self` passes (one FAIL, `awdl`, as before: the hold-down loop only runs during runs). The report has `llama_cpp PASS commit 1e7bcf3da (same as the pinned 1e7bcf3), backend metal`, `models PASS hf: 5 directories; gguf: 14 file(s)` and `link PASS wifi on en0`.

### After review fixes

Review fixes were applied after V3 (commit 5f3f255). Full `tests/bench`: 238 passed. B1 was run again after the fixes (`B1post_q8_1.5b_1rank_r1`).
