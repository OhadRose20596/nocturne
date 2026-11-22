# Engine GGUF, part 1: pre-registered predictions (V3)

Written 2026-11-21 on the M5, before any end-to-end run of the GGUF path. The unit tests (V1) and the mutation check (V2) ran before this file. No V3 row has run.
Setup: M5, 24 GB, MPS, torch 2.14.1, llama.cpp 1e7bcf3da (Metal), engine `nocturne/engines/torch_tp`. Time per token is not predicted (T1).
The outcome column is filled after the runs, failures included. A row that misses its prediction stays in the table with a diagnosis.

Terms. "Identical" means the same token ids over the whole generation of a prompt (32 tokens for eval prompts, 256 for the long one). A "first divergence" is the index of the first differing token. The eval set is the 8 prompts of session 6 plus one long prompt, the same ids the recorded runs used.

## Why these numbers

The bf16 safetensors path is not changed in its math: same loader order, same model code. Only the architecture table, the optional bias and the `linear` helper were added. So the old runs must reproduce token for token.
A GGUF run differs from the bf16 reference in two ways. The weights carry quantization noise. llama.cpp also quantizes its activations to Q8_0 inside the dot product, and we do not (we expand the weights to bf16 and multiply bf16 activations). So engine against llama.cpp on one file is a noise comparison, not an identity test. Short answers (3 to 6 tokens) are robust to that noise. A 256 token free text drifts.
PP only moves layers between processes. It changes no arithmetic. TP and hybrid change the order of one sum (two bf16 partials against one fp32 accumulation), which is the session 3 and 6 effect.

## Predictions

| id | run | prediction (Claude) | confidence |
|---|---|---|---|
| A1 | bf16 regression, `P3_loopback` (1.5B, TP 3, M5 loopback) against recorded `P3_loopback_qwen2.5-1.5b-bf16_r1` | 9 of 9 prompts identical to the recorded run, long prompt included. Frames per step equal the formula. Loaded bytes equal planned bytes on all ranks. | high |
| A2 | bf16 regression, `P1_loopback` (7B, TP 2 at 3:2) against recorded `P1_loopback_qwen2.5-7b-bf16_r1` | 9 of 9 identical. Same frames and bytes statement. Needs the usual apps closed (15 GB on the M5). | high |
| B1 | 1.5B Q8_0, one rank on MPS, against bf16 reference `R0_qwen2.5-1.5b-bf16_mps_r1` | 5 to 8 of 8 eval prompts identical. First divergence of the long prompt between token 10 and 100. | medium |
| B2 | 7B Q8_0, one rank on MPS, against `R0_qwen2.5-7b-bf16_mps_r1` | 6 to 8 of 8 identical. Long prompt first divergence between 15 and 120. | medium |
| B3 | engine against llama.cpp on the same 1.5B and 7B Q8_0 files (llama-server, greedy, token ids) | 5 to 8 of 8 identical on eval prompts. Long prompt diverges before token 120. | medium |
| B4 | 7B Q4_K_M (A1 file, sha256 starts 65b8fcd9), engine one rank against llama.cpp | 5 to 8 of 8 identical. K quant noise is larger than Q8_0, so this row is the least likely to reach 8. | medium |
| C1 | 7B Q8_0 loopback, PP 3 stages, against the one rank run of the same file | 9 of 9 identical. PP is exact. | high |
| C2 | 7B Q8_0 loopback, TP 3 ranks, against the one rank run | 7 or 8 of 8 identical (criterion: at least 7 of 8). Long prompt may diverge late. | medium |
| C3 | 7B Q8_0 loopback, one hybrid placement (stage of TP 2 plus a PP stage), against the one rank run | 7 or 8 of 8 identical. | medium |
| C4 | C1 to C3: frames per step | equal the formula on every placement. | high |
| C5 | C1 to C3: memory per rank. Loaded weight bytes against the plan | within 0.1 percent (session 6 measured equality). The 5 percent criterion of the prompt is met by this number. | high |
| C6 | C1 to C3: process footprint on MPS against the planner's `footprint` | ratio between 0.9 and 1.3. The 1 GiB heap floor and the dequant chunk make the plan a lower bound for small ranks. Reported, not a gate. | low |
| D1 | perplexity, 128 tokens of wikitext-2 test at a fixed offset, 7B Q8_0: engine against `llama-perplexity` on the same file | within 2 percent of each other. A bug would show as a gap of 10 percent or more. | medium |
| D2 | same for 7B Q4_K_M | within 3 percent. | medium |
| E1 | llama 3.2 1B Instruct, safetensors, one rank, against transformers `generate` | at least 7 of 8 eval prompts identical. | medium |
| E2 | llama 3.2 1B Q8_0 GGUF made locally, one rank, against transformers `generate` on the safetensors | at least 5 of 8 identical. | medium |
| E3 | Qwen3 0.6B, safetensors, one rank, against transformers `generate` | at least 7 of 8 identical. Qwen3 chat prompts open a think block; the same template is used on both sides. | medium |
| E4 | Qwen3 0.6B Q8_0 GGUF made locally, against transformers on the safetensors | at least 5 of 8 identical. | medium |
| F1 | `scripts/15_join.sh --check --self` on the M5 | all checks PASS except `awdl` (the loop is not running outside runs). The report has the llama.cpp line (commit 1e7bcf3da, Metal), the models line (hf and gguf) and the link line (wifi). | high |

## Possible causes of a miss (written now, checked after)

1. A row B or C misses by one or two prompts: quantization noise on a prompt where two tokens were close. Check the logit gap at the first divergence in the first logits file.
2. A row A misses: the architecture table or the loader changed an old path. This is a bug, not noise. Stop and bisect against the start SHA `ef50523`.
3. Row D gaps above 3 percent: the engine expands weights to bf16 while llama.cpp keeps f32 activations. Check by running the engine's reduce in float32 on the final norm and head.
4. A Q4_K_M row far from llama.cpp: suspect the Q6_K path (the file uses Q6_K for half of `attn_v` and `ffn_down`). The unit tests cover it against gguf-py; the file level test is this row.

## Outcomes (filled 2026-11-21, after the runs)

| id | outcome | met |
|---|---|---|
| A1 | 9 of 9 identical to the recorded run, frames 346 of 346, loaded bytes equal planned | yes |
| A2 | 9 of 9 identical to the recorded run, frames 116 of 116 | yes |
| B1 | 8 of 8 identical to bf16 R0; long prompt diverges at 54 | yes |
| B2 | 8 of 8; long prompt at 49 | yes |
| B3 | 8 of 8 on 1.5B and 7B Q8_0; long at 54 and 49 | yes |
| B4 | 8 of 8; long at 5 | yes (long prompt earlier than hoped) |
| C1 | 8 of 8, long prompt identical, logit difference 0 | yes |
| C2 | 8 of 8; long diverges at 162 | yes |
| C3 | 8 of 8; long diverges at 49 | yes |
| C4 | frames 3, 346, 61 equal the formula | yes |
| C5 | loaded weight bytes equal planned on every rank | yes |
| C6 | ratio 1.07 to 1.76, TP ranks about 1.5 | no (reported, not a gate) |
| D1 | 0.6 percent | yes |
| D2 | 1.1 percent | yes |
| E1 | 8 of 8 | yes |
| E2 | 8 of 8 (long prompt differs at the first token) | yes |
| E3 | 7 of 8 | yes |
| E4 | 6 of 8 | yes |
| F1 | all PASS except awdl; llama.cpp, models and link lines present | yes |

Cause check for the one miss: C6 is the footprint, as listed under possible causes. The cause is real memory held beyond tensors (KV cache, dequant chunk buffers, MPS heaps). It does not affect results. The long prompt of B4 diverging at token 5 follows the larger Q4_K noise and bf16 activations; the perplexity gap to llama.cpp is 1.1 percent.

# Part 2: three machines (written 2026-11-22 before any part 2 run)

Hosts: m5 (MPS, torch 2.14.1), air (MPS, torch 2.15.0, gguf 0.19.0 added with uv, torch untouched), gtx (GTX 1070, 8 GB, CUDA, torch 2.14.1+cu126, WSL2). Home Wi-Fi, repo SHA bbbfd94 on all three (the Air's repo is the iCloud copy of the M5 folder). Files on all three hosts, sha256 checked on both ends: Qwen2.5-7B Q8_0 862574978752413e and Q4_K_M 65b8fcd92af6b4fe. A3 passed before this section: dequant of all six types on CUDA is bit exact against gguf-py (34 tests).
Criterion of every row: at least 7 of 8 eval prompts identical against the one rank run of the same file from part 1 (`B2` for Q8_0, `B4` for Q4_K_M). Time per token is not predicted (T1).

| id | run | prediction | confidence |
|---|---|---|---|
| R1 | 7B Q8_0, PP 3 stages: layers 0 to 10 and embedding on m5, 10 to 19 on air, 19 to 28 and head on gtx (budget 6.0 GB) | 7 or 8 of 8 identical. Frames per step 3. Loaded bytes equal planned. gtx loaded about 3.5 GB, VRAM free at ready above 1 GB. Session 8b did the same split in bf16 and gave 8 of 8, but the Air runs torch 2.15 against 2.14.1 and CUDA matmul rounds differently from MPS | medium |
| R2 | 7B Q8_0, TP over three machines, heads 14 (m5), 7 (air), 7 (gtx, budget 5.5 GB), mixed devices accepted | 7 or 8 of 8. Frames 346. gtx loaded about 3.9 GB | medium |
| R3 | 7B Q8_0 hybrid: TP pair (m5, air) for layers 0 to 13 with the embedding, one gtx stage for layers 14 to 27 and the head (budget 6.0 GB) | 7 or 8 of 8. Frames equal the planner formula. gtx loaded about 4.7 GB (14 layers and the head) and may be the tightest rank | medium |
| R4a | 7B Q4_K_M, PP 3 stages as R1 | 6 to 8 of 8 against `B4`. Frames 3. K quant noise is larger but PP adds none | medium |
| R4b | 7B Q4_K_M, TP with the heads override 14, 7, 7 | the planner rejects it, naming the block rule: head 21 is column 2688 of `o_proj`, not a multiple of 256. This is a prediction about the planner, not the run | high |
| R4c | 7B Q4_K_M, TP over three machines with planner chosen heads (equal shares, heads in units of 2) | the planner accepts, every head boundary is even, 6 to 8 of 8 identical against `B4`. Frames 346 | medium |
| R4d | R4a and R4c against llama.cpp on the same file (token ids from part 1) | same count of identical prompts as against `B4` plus or minus one | medium |
| M | memory, every rank of R1 to R4c | loaded bytes equal planned weight bytes (within 0.1 percent). MPS footprint 1.1 to 1.6 times the plan (part 1). gtx peak allocated within its budget | high |
| L | link record before each run | Air TCP connect median near 100 ms on an idle link (session 8), gtx near 10 ms. Both on Wi-Fi | high |

Possible causes of a miss, written now: one or two prompts differing is a near tie moved by cross device rounding (check the logit gap in the first logits file); a rank failing to load is memory (gtx VRAM already holds 0.9 GB for Windows); a plan rejection is the alignment rule working, not a bug.

## Part 2 outcomes (filled 2026-11-22)

| id | outcome | met |
|---|---|---|
| R1 | 8 of 8 against `B2`. Frames 3 of 3. Loaded bytes equal planned. gtx loaded 2.81 GB, peak 3.12 GB, VRAM free 4.25 GB (I predicted about 3.5 GB loaded: the 9 layers and the head are smaller than my estimate) | yes |
| R2 | 8 of 8. Frames 346 of 346. gtx loaded 2.02 GB (predicted about 3.9 GB: I forgot that 7 of 28 heads is a quarter of the attention, and a quarter of the MLP) | yes (number miss on gtx bytes) |
| R3 | 8 of 8. Frames 61 of 61. gtx loaded 4.05 GB (predicted about 4.7), tightest rank with 2.94 GB VRAM free | yes |
| R4a | 8 of 8 against `B4` | yes |
| R4b | plan_rejected with the predicted rule message | yes |
| R4c | planner chose heads 14, 8, 6 (all boundaries even); 8 of 8 against `B4`; frames 346 | yes |
| R4d | against llama.cpp: R4a 8 of 8, R4c 8 of 8, the same count as against `B4` | yes |
| M | loaded bytes equal planned on every rank (1.0000). MPS footprints 1.4 to 2.0 times the planned weights (the plan footprint is the larger of total and 1 GiB); gtx peak allocated within its budget everywhere | yes |
| L | Air TCP connect median 8.2 to 9.8 ms (min about 7, max about 105), gtx 7.4 to 8.5 ms. I predicted about 100 ms for the Air median: that was the session 8 idle link figure, taken while the Air was asleep-prone; with caffeinate and the AWDL hold-down on, the median is about 9 ms | no (Air median), the link was better than predicted |

## R5 (written 2026-11-22 before the run)

Qwen2.5-32B Q8_0 (34.8 GB), PP over three machines, context 512, M5 GPU limit at the macOS default (`iogpu.wired_limit_mb` 0, about 17.76 GiB of Metal working set). Placement `R5_pp3_q8_32b.yaml`: m5 layers 0 to 31 with the embedding, air 31 to 53, gtx 53 to 64 with the head. The llama.cpp row of session 8c (`-ts 21,11,32` air, gtx, m5) carried 16.76, 10.84, 5.52 GiB on the same hosts, so the tight part here is gtx: our planner puts 7.04 GB on it, with about 7.3 GB of VRAM free after Windows (0.9 GB).

| id | prediction | confidence |
|---|---|---|
| R5a | the planner accepts the placement (done: it does) with planned totals m5 17.45 GB, air 11.94 GB, gtx 7.04 GB | high |
| R5b | the m5 and air ranks load. gtx loads or fails with CUDA out of memory (planned 7.04 GB of 7.3 GB free, plus the CUDA context). I give gtx 55 percent to load. A gtx failure is reported with the planner's numbers and the gap, then I try the split the planner proposes with one layer fewer on gtx | low |
| R5c | if it completes: frames per step 3; task accuracy on the 8 prompts equal to the llama.cpp row (1.0); eval prompts identical to the llama.cpp answers in text for at least 6 of 8 (no token reference exists for 32B, since llama.cpp cannot hold it on the M5 alone); loaded bytes equal planned | medium |
| R5d | M5 memory pressure: the M5 rank holds 17.4 GB of weights on a 24 GB machine with the usual apps. Swap grows. The run still completes if the apps are closed | medium |
| R6 | only if R5 passes and the planner accepts a hybrid or TP split of 32B. TP of 32B over three machines would send 64 layers times two reductions per token over Wi-Fi: 128 round trips per token. I expect the planner to accept it only with unequal shares and I expect it to be slow, not wrong | low |

## R5 outcome, run 1 (2026-11-22)

`R5_gguf_pp3_q8_32b_c512_r1`: all three ranks loaded their planned bytes exactly (m5 16.888 GB, air 11.398 GB, gtx 6.526 GB; gtx at 7.3 of 8 GiB). R5b met for gtx, which I gave 55 percent. The run then failed in the first prefill: the Air rank hit `MPS command buffer execution failed: Insufficient Memory` and closed its link (`LinkLost` on the coordinator). No prompt completed.
Diagnosis: the planner's total for the Air was 11.94 GB, under its budget of 12.5 GB, but the MPS process carries about 1.3 GB above the plan (part 1, C6 miss: heaps and allocator cache), so the Air rank needed about 13.2 GB against a Metal working set of about 12.1 GB. This is the known footprint gap of the plan, not a new bug. R5b missed (I predicted the Air would load and gtx was the risk).
One diagnosed retry, registered before it runs: `R5b_pp3_q8_32b_m5raised.yaml`, layers m5 0 to 34, air 34 to 53, gtx 53 to 64, with the M5 GPU limit raised to 20 GiB (a separate policy, as in session 8c). Planned totals m5 19.01 GB, air 10.38 GB, gtx 7.04 GB. With about 1.3 to 1.5 GB of MPS overhead: m5 about 20.5 GB of 21.47 GB, air about 11.7 GB of 12.1 GB. Prediction: loads and completes, frames 3, task accuracy 1.0 as the llama.cpp row. Confidence low: the M5 will swap heavily (7.9 GB of swap during run 1).

## R6 (written 2026-11-22 while R5b runs, before R6 runs)

Placement `R6_hybrid_q8_32b_m5raised.yaml`, M5 GPU limit 20 GiB. Planner: m5 heads 0 to 26 and MLP rows 0 to 18176 of layers 0 to 53 with the embedding (planned 19.18 GB), air heads 26 to 40 and rows 18176 to 27648 (10.22 GB), gtx layers 53 to 64 and the head (7.04 GB). Frames per step 217. Q8_0 blocks of 32: every boundary aligned.
R6 runs only if R5b completes. Prediction: loads and completes (medium, the M5 is at about 20.7 of 21.47 GB with overhead and swaps), frames 217, loaded bytes equal planned, task accuracy equal to R5b and to the llama.cpp row (1.0), eval prompts identical to R5b in at least 6 of 8 (the TP pair changes one sum order per layer). Time per token is background (T1), but I expect it to be slower than R5b: 106 all-reduces per token cross the Wi-Fi to the Air.

## R5 outcome, retry (2026-11-22)

`R5b_gguf_pp3_q8_32b_c512_m5raised_r1` completed. 32B Q8_0 runs on three machines with the engine reading GGUF.

| id | outcome | met |
|---|---|---|
| R5a | planner accepted (retry plan: m5 19.01 GB, air 10.38 GB, gtx 7.04 GB) | yes |
| R5b | all three ranks loaded on the retry. Run 1 failed on the Air (MPS out of memory), not on gtx | no on run 1, yes on the retry |
| R5c | frames 3.0 of 3 over 279 decode steps. Task accuracy 1.0 (8 of 8), the same as the llama.cpp row `A1_wifi_qwen2.5-32b-q8_m5_c512ts3m_r1`. Answer text identical to llama.cpp in 8 of 8 (predicted at least 6). Loaded bytes equal planned on every rank (m5 18.443 GB, air 9.844 GB, gtx 6.526 GB) | yes |
| R5d | the M5 swapped (about 9 to 10 GB during the eval prompts, 4.1 GB at the end) and the run completed | yes |

Memory, planned total against measured: m5 19.01 GB planned, MPS footprint peak 20.67 GB (gap 1.66 GB). air 10.38 GB planned, footprint peak 11.46 GB (gap 1.08 GB). gtx 7.04 GB planned, CUDA peak allocated 6.79 GB, VRAM free at the end 0.36 GB. The MPS gap is the same 1 to 1.7 GB seen in part 1 and in R5 run 1.
Time per token (background, T1): about 45 s per decode step against 0.35 s for llama.cpp on the same split. Not diagnosed. The likely cause is M5 swap: the rank footprint (20.7 GB) plus the coordinator on a 24 GB machine. This is a measurement for session 9, not a result about PP.
Note on L: the Air link median depends on load. 102 ms on an idle link, 9 to 12 ms while copies or runs were active. Both figures were seen today.

## R6 outcome (2026-11-23)

`R6_gguf_hybrid_q8_32b_c512_m5raised_r1`: the planner's split loaded exactly (m5 18.627 GB, air 9.735 GB, gtx 6.526 GB, equal to planned weights). The 8 eval prompts finished between 22:18 and 22:45 with the same token counts as R5b (4, 4, 4, 3, 4, 3, 4, 6). The long prompt then ran for 80 minutes until gtx left the network at 00:06 (its ssh session reported "Operation timed out" and the host answers no ping). The coordinator raised LinkLost at 00:36. The record keeps prompts and frames only on success, so the answer text and the 217 frames are not measured.
Prediction "loads": met. "Completes", "frames 217", "accuracy 1.0", "at least 6 of 8 identical to R5b": not tested. The failure is the host, not the engine or the plan.
Time per token (background): about 50 s, close to R5b's 45 s. The 106 all-reduces per token to the Air did not dominate; the M5 swap is the likely common cost in both.
Two engine changes for a separate approved step: write the finished prompts into the record when a later prompt fails, and log the frame counters per prompt.
