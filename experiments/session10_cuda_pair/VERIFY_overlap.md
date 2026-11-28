# Session 10, part 3 (overlap): verification record

Date: 2026-11-28. Operator: the M5 (24 GB, MPS, torch 2.14.1, Python 3.12). Start SHA of the change: `965385f`.
The compute path is final at `c352d39`. After it: `511fd68` adds one placement file, `be73eed` changes tests and one document, `dd3f8f5` changes comments and the early check of a reference. The column "commit" says which code a run used.
gtx (GTX 1070, WSL2, torch 2.14.1+cu126) ran the CUDA checks. Lior's laptop (`pc`, RTX 2000 Ada, WSL2) ran the pair checks of the first round and was then switched off.
No experiment rows were run. Every run below verifies the build. Time is background (T1).
Design: `docs/ENGINE_DESIGN.md` section 11. Numbers: `docs/MEASUREMENT.md` section 3.1. Reviews: `REVIEW_overlap.md`, `REVIEW_overlap_pass2.md`, `REVIEW_overlap_pass3.md`.

## Gate

| condition | result |
|---|---|
| correctness rules met | yes. O1, O3, O4, O5: every request gives exactly the tokens of the run with the options off, on the 8 eval prompts and on the long prompts, with bit equal first step logits. O2: 8 of 8 eval prompts identical, the difference is reported and explained below |
| options off give the old behaviour | yes. Same tokens, frames and bytes per link as the recorded part 1 runs (table "Options off"), and the frame structure is pinned against the engine of `965385f` by a test. Three reviewers ran the old engine against the new one frame by frame: identical |
| frame formula holds | yes, in every run: frames per decode step and prefill frames, measured equal predicted |
| mutations caught | yes: 88 of 88 on the final compute path, 53 of them for part 3, the four named breaks included (round 1 had two that were caught only by a hung test; both now fail fast) |
| tests pass | yes: 420 passed and 3 skipped (CUDA only) on the M5, `tests/bench`, 5.4 minutes, on the last commit. 89 tests are new in part 3. On gtx (Linux, CUDA) the overlap test file gives 88 passed, the pinned copy tests and the socket tests on a Linux kernel included; one run there did not return (see Open) |
| no open high finding | yes. Three review passes: the first found one high finding (socket buffers on Linux), the second found one high finding in that fix, the third found both closed and no new high finding |

## What was built

| option | what it does | off value |
|---|---|---|
| `--slots K` | up to K requests in flight; a KV cache per slot; the slot id in TOKEN, HIDDEN and HEAD frames | 1 |
| `--prefill-chunk C` | the prompt travels in chunks of C tokens; only the last chunk gets a head result | 0 |
| `--tp-interleave 2` | two ready requests go through a TP group in lockstep; the add order of the all-reduce is unchanged | 1 |
| `--async-send` | a sender thread per rank; CUDA: non blocking copy into pinned memory; MPS and CPU: the copy stays on the compute thread | off |
| a host in several stages | interleaved pipeline stages; the planner sums all rank processes of a host against `host_budget_gb` | a placement, not a flag |

Not built, described in the design (section 11.9): speculative decoding, keeping a conversation's KV cache.

## V1. Correctness

Every run with options names two references: the one rank run of the model, and the options-off run of the same placement. The second is the rule of this part.
Columns: "identical" counts the 8 eval prompts; then the long prompts (first token that differs, or the full length when none differs); then the largest difference of the first step logits.
Counted over all runs of this part that name their options-off run: 30 runs fall under the exact rule (slots, pairs, async send, interleaved stages, and their combinations without chunks), and all 30 have 8 of 8 eval prompts identical, every long prompt identical, and a first step logit difference of exactly 0. 18 runs use chunked prefill, and all 18 have 8 of 8 eval prompts identical. 77 runs were made in all: 76 completed, one ended with an error that is a finding (MPS, below).
Runs with `l3` in their name use three long prompts of 96 tokens, so that several long requests are in flight together. The other runs use the prompt set of part 1 (one long prompt of 256 tokens).

### 1.5B Q8_0 on the M5 loopback (MPS), every feature

Run on `cfab15e` (after the fixes of review pass 1). The fixes of pass 2 came after it; the subset below repeats the affected runs on the final code.

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OV_15b_hybrid_all_l3_r1 | S10_q8_15b_hybrid.yaml | slots 3, chunk 16, pair, async (inline) | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; first divergence [20, 88, 96]; 0.4375 | 61.0 / 61 | 2591 / 2591 | 182708 | 2.37 | 32 / 320; 3 | m5 mps: 7 | s0/0 4.3 (4.2), s0/1 0.6 (0.1), s1/0 1.3 (0.0) | none |
| OV_15b_hybrid_c16_l3_r1 | S10_q8_15b_hybrid.yaml | chunk 16 | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; first divergence [20, 88, 96]; 0.4375 | 61.0 / 61 | 2591 / 2591 | 182708 | 2.14 | 0 / 352; 1 | m5 mps: 97 | s0/0 2.6 (2.0), s0/1 0.7 (0.0), s1/0 3.2 (0.0) | none |
| OV_15b_hybrid_off_l3_r1 | S10_q8_15b_hybrid.yaml | off | cfab15e | 8/8; 20 (2 prompts not in the reference) | - | 61.0 / 61 | 671 / 671 | 182708 | 2.29 | - | m5 mps: 97 | s0/0 0.4 (0.0), s0/1 0.4 (0.0), s1/0 0.0 (0.0) | none |
| OV_15b_hybrid_off_r1 | S10_q8_15b_hybrid.yaml | off | cfab15e | 8/8; 20 | - | 61.0 / 61 | 549 / 549 | 182708 | 2.33 | - | m5 mps: 97 | s0/0 0.4 (0.0), s0/1 0.4 (0.0), s1/0 0.0 (0.0) | none |
| OV_15b_hybrid_s2_l3_r1 | S10_q8_15b_hybrid.yaml | slots 2 | cfab15e | 8/8; 20 (2 prompts not in the reference) | 8/8; identical; 0 | 61.0 / 61 | 671 / 671 | 182708 | 2.47 | 0 / 320; 2 | m5 mps: 99 | s0/0 2.8 (0.0), s0/1 2.6 (0.0), s1/0 20.1 (0.0) | none |
| OV_15b_hybrid_s2_pair_l3_r1 | S10_q8_15b_hybrid.yaml | slots 2, pair | cfab15e | 8/8; 20 (2 prompts not in the reference) | 8/8; identical; 0 | 61.0 / 61 | 671 / 671 | 182708 | 2.51 | 2 / 318; 2 | m5 mps: 99 | s0/0 2.7 (0.2), s0/1 2.8 (0.1), s1/0 20.7 (0.0) | none |
| OV_15b_hybrid_s3_l3_r1 | S10_q8_15b_hybrid.yaml | slots 3 | cfab15e | 8/8; 20 (2 prompts not in the reference) | 8/8; identical; 0 | 61.0 / 61 | 671 / 671 | 182708 | 2.53 | 0 / 320; 3 | m5 mps: 100 | s0/0 57.3 (57.2), s0/1 3.7 (0.0), s1/0 97.5 (0.0) | none |
| OV_15b_pp2_contig_off_l3_r1 | O5_q8_15b_pp2_contiguous.yaml | off | cfab15e | 8/8; 96 (2 prompts not in the reference) | - | 2.0 / 2 | 22 / 22 | 3112 | 2.36 | - | m5 mps: 100 | a0/0 0.0 (0.0), b0/0 0.0 (0.0) | none |
| OV_15b_pp3_async_l3_r1 | S10_q8_15b_pp3.yaml | async (inline) | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 3.0 / 3 | 33 / 33 | 6208 | 2.41 | - | m5 mps: 59 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_c16_l3_r1 | S10_q8_15b_pp3.yaml | chunk 16 | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; first divergence [77, 88, 96]; 0.3516 | 3.0 / 3 | 97 / 97 | 6208 | 2.21 | 0 / 352; 1 | m5 mps: 100 | s0/0 0.4 (0.4), s1/0 0.8 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_f32_c16_l3_r1 | S10_q8_15b_pp3.yaml | chunk 16 | cfab15e | 8/8; 54 (2 prompts not in the reference) | 8/8; identical; 9.155e-05 | 3.0 / 3 | 97 / 97 | 12352 | 2.20 | 0 / 352; 1 | m5 mps: 100 | s0/0 0.5 (0.5), s1/0 0.7 (0.1), s2/0 0.1 (0.0) | none |
| OV_15b_pp3_f32_off_l3_r1 | S10_q8_15b_pp3.yaml | off | cfab15e | 8/8; 54 (2 prompts not in the reference) | - | 3.0 / 3 | 33 / 33 | 12352 | 2.36 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_off_l3_r1 | S10_q8_15b_pp3.yaml | off | cfab15e | 8/8; 96 (2 prompts not in the reference) | - | 3.0 / 3 | 33 / 33 | 6208 | 2.37 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_off_r1 | S10_q8_15b_pp3.yaml | off | 6ddcf01 | 8/8; 256 | - | 3.0 / 3 | 27 / 27 | 6208 | 2.39 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_off_r2 | S10_q8_15b_pp3.yaml | off | cfab15e | 8/8; 256 | - | 3.0 / 3 | 27 / 27 | 6208 | 2.37 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_s2_async_l3_r1 | S10_q8_15b_pp3.yaml | slots 2, async (inline) | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 3.0 / 3 | 33 / 33 | 6208 | 2.57 | 0 / 320; 2 | m5 mps: 87 | s0/0 0.0 (0.0), s1/0 0.1 (0.0), s2/0 0.0 (0.0) | none |
| OV_15b_pp3_s2_c16_l3_r1 | S10_q8_15b_pp3.yaml | slots 2, chunk 16 | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; first divergence [77, 88, 96]; 0.3516 | 3.0 / 3 | 97 / 97 | 6208 | 2.37 | 0 / 352; 2 | m5 mps: 100 | s0/0 4.0 (3.2), s1/0 4.5 (0.7), s2/0 1.9 (0.8) | none |
| OV_15b_pp3_s2_l3_r1 | S10_q8_15b_pp3.yaml | slots 2 | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 3.0 / 3 | 33 / 33 | 6208 | 2.56 | 0 / 320; 2 | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.1 (0.0), s2/0 0.1 (0.0) | none |
| OV_15b_pp3_s3_l3_r1 | S10_q8_15b_pp3.yaml | slots 3 | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 3.0 / 3 | 33 / 33 | 6208 | 2.64 | 0 / 320; 3 | m5 mps: 100 | s0/0 0.1 (0.1), s1/0 29.2 (29.0), s2/0 52.4 (0.0) | none |
| OV_15b_pp4_il_off_l3_r1 | O5_q8_15b_pp4_interleaved.yaml | off | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 2.40 | - | m5 mps: 99 | a0/0 0.0 (0.0), b0/0 0.0 (0.0), a1/0 0.0 (0.0), b1/0 0.0 (0.0) | none |
| OV_15b_pp4_il_s2_c16_l3_r1 | O5_q8_15b_pp4_interleaved.yaml | slots 2, chunk 16 | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; first divergence [77, 88, 96]; 0.3516 | 4.0 / 4 | 140 / 140 | 9304 | 2.37 | 0 / 352; 2 | m5 mps: 100 | a0/0 1.8 (1.3), b0/0 1.9 (0.4), a1/0 1.8 (1.6), b1/0 3.4 (0.0) | none |
| OV_15b_tp3_async_l3_r1 | S10_q8_15b_tp3.yaml | async (inline) | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.37 | - | m5 mps: 17 | s0/0 0.5 (0.1), s0/1 0.0 (0.0), s0/2 0.0 (0.0) | none |
| OV_15b_tp3_off_l3_r1 | S10_q8_15b_tp3.yaml | off | cfab15e | 8/8; 77 (2 prompts not in the reference) | - | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.28 | - | m5 mps: 94 | s0/0 4.1 (0.0), s0/1 5.3 (0.0), s0/2 6.1 (0.0) | none |
| OV_15b_tp3_off_notl_r1 | S10_q8_15b_tp3.yaml | off | cfab15e | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.36 | - | - | - | - |
| OV_15b_tp3_off_r1 | S10_q8_15b_tp3.yaml | off | 6ddcf01 | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.30 | - | m5 mps: 94 | s0/0 3.9 (0.1), s0/1 5.5 (0.1), s0/2 6.6 (0.0) | none |
| OV_15b_tp3_off_r2 | S10_q8_15b_tp3.yaml | off | cfab15e | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.30 | - | m5 mps: 94 | s0/0 4.2 (0.0), s0/1 5.9 (0.0), s0/2 5.5 (0.0) | none |
| OV_15b_tp3_s2_pair_async_l3_r1 | S10_q8_15b_tp3.yaml | slots 2, pair, async (inline) | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.43 | 172 / 148; 2 | m5 mps: 46 | s0/0 8.8 (0.9), s0/1 5.7 (1.0), s0/2 7.1 (1.2) | none |
| OV_15b_tp3_s2_pair_l3_r1 | S10_q8_15b_tp3.yaml | slots 2, pair | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.38 | 214 / 106; 2 | m5 mps: 96 | s0/0 27.6 (19.4), s0/1 27.6 (21.1), s0/2 28.6 (21.0) | none |
| OV_15b_tp3_s3_pair_l3_r1 | S10_q8_15b_tp3.yaml | slots 3, pair | cfab15e | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.39 | 232 / 88; 3 | m5 mps: 96 | s0/0 61.0 (22.1), s0/1 64.3 (59.3), s0/2 64.5 (59.6) | none |

### The same on the final code (a subset)

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVf_15b_hybrid_all_l3_r1 | S10_q8_15b_hybrid.yaml | slots 4, chunk 16, pair, async (inline) | 511fd68 | 8/8; 96 (2 prompts not in the reference) | 8/8; first divergence [20, 88, 96]; 0.4375 | 61.0 / 61 | 2591 / 2591 | 182708 | 2.35 | 36 / 316; 4 | m5 mps: 8 | s0/0 5.2 (5.2), s0/1 0.6 (0.1), s1/0 1.4 (0.0) | none |
| OVf_15b_hybrid_off_r1 | S10_q8_15b_hybrid.yaml | off | 511fd68 | 8/8; 20 | - | 61.0 / 61 | 549 / 549 | 182708 | 2.29 | - | m5 mps: 97 | s0/0 0.4 (0.0), s0/1 0.4 (0.0), s1/0 0.0 (0.0) | none |
| OVf_15b_pp3_off_r1 | S10_q8_15b_pp3.yaml | off | 511fd68 | 8/8; 256 | - | 3.0 / 3 | 27 / 27 | 6208 | 2.37 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OVf_15b_pp3_s3_l3_r1 | S10_q8_15b_pp3.yaml | slots 3 | 511fd68 | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 3.0 / 3 | 33 / 33 | 6208 | 2.62 | 0 / 320; 3 | m5 mps: 100 | s0/0 0.1 (0.1), s1/0 29.8 (29.7), s2/0 53.4 (0.0) | none |
| OVf_15b_tp3_off_r1 | S10_q8_15b_tp3.yaml | off | 511fd68 | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.29 | - | m5 mps: 94 | s0/0 4.0 (0.0), s0/1 6.0 (0.0), s0/2 5.4 (0.0) | none |
| OVf_15b_tp3_s2_pair_async_l3_r1 | S10_q8_15b_tp3.yaml | slots 2, pair, async (inline) | 511fd68 | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.41 | 164 / 156; 2 | m5 mps: 45 | s0/0 9.2 (1.0), s0/1 6.7 (1.4), s0/2 6.0 (1.3) | none |
| OVf_15b_tp3_s2_pair_l3_r1 | S10_q8_15b_tp3.yaml | slots 2, pair | 511fd68 | 8/8; 77 (2 prompts not in the reference) | 8/8; identical; 0 | 346.0 / 346 | 3806 / 3806 | 1058920 | 2.37 | 214 / 106; 2 | m5 mps: 95 | s0/0 27.7 (19.0), s0/1 26.9 (20.6), s0/2 28.2 (20.9) | none |

### 7B Q8_0 on the M5 loopback, final code

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OV_7b_hybrid_all_r1 | Q8_7b_hybrid.yaml | slots 3, chunk 16, pair, async (inline) | be73eed | 8/8; 246 | 8/8; first divergence [49]; 0.375 | 61.0 / 61 | 2049 / 2049 | 424372 | 0.49 | 18 / 295; 3 | m5 mps: 3 | s0/0 1.7 (1.7), s0/1 0.2 (0.0), s1/0 0.1 (0.0) | none |
| OV_7b_hybrid_off_r1 | Q8_7b_hybrid.yaml | off | 511fd68 | 8/8; 49 | - | 61.0 / 61 | 549 / 549 | 424372 | 0.53 | - | m5 mps: 99 | s0/0 0.8 (0.0), s0/1 0.6 (0.0), s1/0 0.0 (0.0) | none |
| OV_7b_pp3_off_r1 | Q8_7b_pp3.yaml | off | 511fd68 | 8/8; 256 | - | 3.0 / 3 | 27 / 27 | 14400 | 0.49 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OV_7b_pp3_s2_c16_async_r1 | Q8_7b_pp3.yaml | slots 2, chunk 16, async (inline) | 511fd68 | 8/8; 49 | 8/8; first divergence [49]; 0.375 | 3.0 / 3 | 77 / 77 | 14400 | 0.49 | 0 / 313; 2 | m5 mps: 69 | s0/0 2.6 (2.1), s1/0 3.3 (0.9), s2/0 0.2 (0.0) | none |
| OV_7b_pp4_il_s2_r1 | O5_q8_7b_pp4_interleaved.yaml | slots 2 | be73eed | 8/8; 256 | 8/8; identical; 0 | 4.0 / 4 | 36 / 36 | 21592 | 0.50 | 0 / 288; 2 | m5 mps: 100 | a0/0 0.1 (0.0), b0/0 0.2 (0.0), a1/0 0.2 (0.0), b1/0 0.8 (0.0) | none |
| OV_7b_tp3_off_r1 | Q8_7b_tp3.yaml | off | 511fd68 | 8/8; 162 | - | 346.0 / 346 | 3114 / 3114 | 2459752 | 0.54 | - | m5 mps: 98 | s0/0 3.5 (0.0), s0/1 4.0 (0.0), s0/2 4.1 (0.0) | none |
| OV_7b_tp3_s2_pair_r1 | Q8_7b_tp3.yaml | slots 2, pair | be73eed | 8/8; 162 | 8/8; identical; 0 | 346.0 / 346 | 3114 / 3114 | 2459752 | 0.54 | 34 / 254; 2 | m5 mps: 99 | s0/0 8.0 (4.6), s0/1 8.7 (4.5), s0/2 8.7 (4.6) | none |

### CUDA, final code: gtx alone (GTX 1070, Linux socket buffers, pinned copies)

`C5` is one TP group of two rank processes on gtx: the path of review findings F6 and P1 with real frames. `C1` and `C2` are two and four pipeline processes on the one GPU.

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVc4_gtx_pp2_off_l3_r1 | C1_gtx_15b_bf16_pp2.yaml | off | 511fd68 | 8/8; 96 (2 prompts not in the reference) | - | 2.0 / 2 | 22 / 22 | 3112 | 12.76 | - | gtx cuda: 98 | a0/0 0.0 (0.0), b0/0 0.0 (0.0) | none |
| OVc4_gtx_pp4_off_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | off | 511fd68 | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 12.46 | - | gtx cuda: 95 | a0/0 0.0 (0.0), b0/0 0.0 (0.0), a1/0 0.0 (0.0), b1/0 0.0 (0.0) | none |
| OVc4_gtx_pp4_s3_c16_async_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, chunk 16, async (pinned) | 511fd68 | 8/8; 21 (2 prompts not in the reference) | 8/8; first divergence [21, 96, 96]; 0.4844 | 4.0 / 4 | 140 / 140 | 9304 | 13.07 | 0 / 352; 3 | gtx cuda: 86 | a0/0 7.6 (2.6), b0/0 6.6 (1.5), a1/0 7.2 (2.3), b1/0 11.5 (1.8) | none |
| OVc4_gtx_tp2_off_l3_r1 | C5_gtx_15b_bf16_tp2.yaml | off | 511fd68 | 8/8; 21 (2 prompts not in the reference) | - | 116.0 / 116 | 1276 / 1276 | 352988 | 6.00 | - | gtx cuda: 82 | s0/0 13.9 (1.2), s0/1 14.4 (1.3) | none |
| OVc4_gtx_tp2_s2_pair_async_l3_r1 | C5_gtx_15b_bf16_tp2.yaml | slots 2, pair, async (pinned) | 511fd68 | 8/8; 21 (2 prompts not in the reference) | 8/8; identical; 0 | 116.0 / 116 | 1276 / 1276 | 352988 | 4.81 | 226 / 94; 2 | gtx cuda: 75 | s0/0 34.9 (9.4), s0/1 17.6 (10.7) | none |
| OVc4_gtx_tp2_s2_pair_l3_r1 | C5_gtx_15b_bf16_tp2.yaml | slots 2, pair | 511fd68 | 8/8; 21 (2 prompts not in the reference) | 8/8; identical; 0 | 116.0 / 116 | 1276 / 1276 | 352988 | 5.42 | 174 / 146; 2 | gtx cuda: 82 | s0/0 37.2 (9.9), s0/1 30.4 (19.8) | none |
| OVc4_gtx_tp2_s4_pair_c16_l3_r1 | C5_gtx_15b_bf16_tp2.yaml | slots 4, chunk 16, pair | 511fd68 | 8/8; 21 (2 prompts not in the reference) | 8/8; first divergence [96, 45, 31]; 0.5 | 116.0 / 116 | 4956 / 4956 | 352988 | 5.24 | 264 / 88; 4 | gtx cuda: 84 | s0/0 55.8 (19.7), s0/1 59.9 (56.4) | none |

### After the last commit (`dd3f8f5`): one run per machine and path

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVz_15b_hybrid_all_l3_r1 | S10_q8_15b_hybrid.yaml | slots 4, chunk 16, pair, async (inline) | dd3f8f5 | 8/8; 96 (2 prompts not in the reference) | 8/8; first divergence [20, 88, 96]; 0.4375 | 61.0 / 61 | 2591 / 2591 | 182708 | 2.39 | 34 / 318; 4 | m5 mps: 7 | s0/0 4.5 (4.5), s0/1 0.5 (0.1), s1/0 1.3 (0.0) | none |
| OVz_15b_tp3_off_r1 | S10_q8_15b_tp3.yaml | off | dd3f8f5 | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.34 | - | m5 mps: 94 | s0/0 4.0 (0.1), s0/1 5.6 (0.1), s0/2 7.1 (0.0) | none |
| OVz_gtx_tp2_s2_pair_l3_r1 | C5_gtx_15b_bf16_tp2.yaml | slots 2, pair | dd3f8f5 | 8/8; 21 (2 prompts not in the reference) | 8/8; identical; 0 | 116.0 / 116 | 1276 / 1276 | 352988 | 5.74 | 136 / 184; 2 | gtx cuda: 83 | s0/0 35.3 (8.8), s0/1 38.1 (27.9) | none |

### CUDA, first round: gtx and the pc together (Wi-Fi), on `1d39b45`

These ran before the review fixes, while the pc was still on. The runs with options completed, but their records are incomplete: the driver stopped at the comparison step, because a plain run folder was not yet accepted as a reference. The comparisons in this table are computed from the raw files of the runs.

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVc_pair_il_off_r1 | C3_gtx_pc_15b_bf16_interleaved.yaml | off | 1d39b45 | 8/8; 34 | - | 4.0 / 4 | 36 / 36 | 9304 | 9.78 | - | gtx cuda: 34, pc cuda: 43 | a0/0 0.0 (0.0), b0/0 0.0 (0.0), a1/0 0.0 (0.0), b1/0 0.0 (0.0) | none |
| OVc_pair_il_s2_c16_async_r1 | C3_gtx_pc_15b_bf16_interleaved.yaml | slots 2, chunk 16, async (pinned) | 1d39b45 | 8/8; 21 | 8/8; first divergence [21]; 0.5 | 4.0 / 4 | 111 / 111 | 9304 | 9.72 | - | - | - | - |
| OVc_pair_tp2_off_r1 | C4_gtx_pc_15b_bf16_tp2.yaml | off | 1d39b45 | 8/8; 54 | - | 116.0 / 116 | 1044 / 1044 | 352988 | 1.84 | - | gtx cuda: 31, pc cuda: 37 | s0/0 23.9 (11.3), s0/1 30.1 (8.6) | none |
| OVc_pair_tp2_pair_async_r1 | C4_gtx_pc_15b_bf16_tp2.yaml | slots 2, pair, async (pinned) | 1d39b45 | 8/8; 54 | 8/8; identical; 0 | 116.0 / 116 | 1044 / 1044 | 352988 | 1.91 | - | - | - | - |
| OVc_pair_tp2_pair_r1 | C4_gtx_pc_15b_bf16_tp2.yaml | slots 2, pair | 1d39b45 | 8/8; 54 | 8/8; identical; 0 | 116.0 / 116 | 1044 / 1044 | 352988 | 1.65 | - | - | - | - |

### Earlier rounds: gtx alone on `1d39b45`, `6ddcf01` and `cfab15e`, and the smoke runs on the M5

| run | placement | options | commit | identical to the one rank run (8; long first divergence) | against the options-off run of the placement (8; long; largest first step logit difference) | frames per step measured / predicted | prefill frames measured / predicted | bytes per decode step | tokens per s | steps in pairs / alone; most requests in flight | busy % per device | overlap % per rank (sent part) | links where sends and receives do not match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OVc_gtx_pp2_off_r1 | C1_gtx_15b_bf16_pp2.yaml | off | 1d39b45 | 8/8; 256 | - | 2.0 / 2 | 18 / 18 | 3112 | 14.84 | - | gtx cuda: 98 | a0/0 0.0 (0.0), b0/0 0.0 (0.0) | none |
| OVc_gtx_pp4_off_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | off | 1d39b45 | 8/8; 256 | 8/8; identical; 0 | 4.0 / 4 | 36 / 36 | 9304 | 12.31 | - | - | - | - |
| OVc_gtx_pp4_s3_c16_async_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, chunk 16, async (pinned) | 1d39b45 | 8/8; 21 | 8/8; first divergence [21]; 0.4844 | 4.0 / 4 | 111 / 111 | 9304 | 10.68 | - | - | - | - |
| OVc2_gtx_pp2_off_l3_r1 | C1_gtx_15b_bf16_pp2.yaml | off | 6ddcf01 | 8/8; 96 (2 prompts not in the reference) | - | 2.0 / 2 | 22 / 22 | 3112 | 13.99 | - | gtx cuda: 98 | a0/0 0.0 (0.0), b0/0 0.0 (0.0) | none |
| OVc2_gtx_pp4_off_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | off | 6ddcf01 | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 14.74 | - | gtx cuda: 95 | a0/0 0.0 (0.0), b0/0 0.0 (0.0), a1/0 0.0 (0.0), b1/0 0.0 (0.0) | none |
| OVc2_gtx_pp4_s3_async_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, async (pinned) | 6ddcf01 | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 19.13 | - | gtx cuda: 85 | a0/0 7.1 (3.1), b0/0 13.4 (1.2), a1/0 8.1 (2.5), b1/0 13.6 (2.3) | none |
| OVc2_gtx_pp4_s3_c16_async_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, chunk 16, async (pinned) | 6ddcf01 | 8/8; 21 (2 prompts not in the reference) | 8/8; first divergence [21, 96, 96]; 0.4844 | 4.0 / 4 | 140 / 140 | 9304 | 14.20 | - | gtx cuda: 89 | a0/0 7.0 (2.6), b0/0 9.6 (2.7), a1/0 12.7 (3.5), b1/0 12.5 (1.9) | none |
| OVc3_gtx_pp2_off_l3_r1 | C1_gtx_15b_bf16_pp2.yaml | off | cfab15e | 8/8; 96 (2 prompts not in the reference) | - | 2.0 / 2 | 22 / 22 | 3112 | 16.48 | - | gtx cuda: 98 | a0/0 0.0 (0.0), b0/0 0.0 (0.0) | none |
| OVc3_gtx_pp4_off_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | off | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 12.33 | - | gtx cuda: 96 | a0/0 0.0 (0.0), b0/0 0.0 (0.0), a1/0 0.0 (0.0), b1/0 0.0 (0.0) | none |
| OVc3_gtx_pp4_s3_async_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, async (pinned) | cfab15e | 8/8; 96 (2 prompts not in the reference) | 8/8; identical; 0 | 4.0 / 4 | 44 / 44 | 9304 | 18.61 | 0 / 320; 3 | gtx cuda: 88 | a0/0 6.5 (1.7), b0/0 9.9 (1.8), a1/0 11.1 (2.5), b1/0 14.3 (2.2) | none |
| OVc3_gtx_pp4_s3_c16_async_l3_r1 | C2_gtx_15b_bf16_pp4_interleaved.yaml | slots 3, chunk 16, async (pinned) | cfab15e | 8/8; 21 (2 prompts not in the reference) | 8/8; first divergence [21, 96, 96]; 0.4844 | 4.0 / 4 | 140 / 140 | 9304 | 14.36 | 0 / 352; 3 | gtx cuda: 89 | a0/0 9.1 (2.9), b0/0 8.1 (2.0), a1/0 9.9 (2.3), b1/0 11.3 (3.0) | none |
| OVs_15b_hybrid_all_r1 | S10_q8_15b_hybrid.yaml | slots 3, chunk 16, pair, async (inline) | 1d39b45 | 8/8; 112 | 8/8; first divergence [20]; 0.4375 | 61.0 / 61 | 2049 / 2049 | 182708 | 2.12 | - | - | - | - |
| OVs_15b_hybrid_off_r1 | S10_q8_15b_hybrid.yaml | off | 1d39b45 | 8/8; 20 | - | 61.0 / 61 | 549 / 549 | 182708 | 2.26 | - | m5 mps: 97 | s0/0 0.6 (0.0), s0/1 0.7 (0.0), s1/0 0.0 (0.0) | none |
| OVs_15b_pp3_off_r1 | S10_q8_15b_pp3.yaml | off | 1d39b45 | 8/8; 256 | - | 3.0 / 3 | 27 / 27 | 6208 | 2.32 | - | m5 mps: 100 | s0/0 0.0 (0.0), s1/0 0.0 (0.0), s2/0 0.0 (0.0) | none |
| OVs_15b_pp3_s2_c16_async_r1 | S10_q8_15b_pp3.yaml | slots 2, chunk 16, async () | 1d39b45 | outcome: error | | | | | | | | | |
| OVs_15b_pp3_s2_c16_async_r2 | S10_q8_15b_pp3.yaml | slots 2, chunk 16, async (inline) | 1d39b45 | 8/8; 77 | 8/8; first divergence [77]; 0.3516 | 3.0 / 3 | 77 / 77 | 6208 | 2.17 | - | - | - | - |
| OVs_15b_pp4_il_s2_r1 | O5_q8_15b_pp4_interleaved.yaml | slots 2 | 1d39b45 | 8/8; 256 | 8/8; identical; 0 | 4.0 / 4 | 36 / 36 | 9304 | 2.32 | - | - | - | - |
| OVs_15b_tp3_off_r1 | S10_q8_15b_tp3.yaml | off | 1d39b45 | 8/8; 77 | - | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.22 | - | m5 mps: 94 | s0/0 3.6 (0.0), s0/1 5.7 (0.0), s0/2 6.1 (0.0) | none |
| OVs_15b_tp3_s2_pair_async_r1 | S10_q8_15b_tp3.yaml | slots 2, pair, async (inline) | 1d39b45 | 8/8; 77 | 8/8; identical; 0 | 346.0 / 346 | 3114 / 3114 | 1058920 | 2.30 | - | - | - | - |

### Options off: the gate against the recorded part 1 runs

| options-off run | recorded part 1 run | same tokens, 9 prompts | frames per step | bytes per step | frames and bytes per link per step |
|---|---|---|---|---|---|
| OV_15b_hybrid_off_r1 | V3final_q8_15b_hybrid_tl_r1 | yes | 61.0 = 61.0 | 182708 = 182708 | identical |
| OV_15b_pp3_off_r1 | V3codex_q8_15b_pp3_tl_r1 | yes | 3.0 = 3.0 | 6208 = 6208 | identical |
| OV_15b_pp3_off_r2 | V3codex_q8_15b_pp3_tl_r1 | yes | 3.0 = 3.0 | 6208 = 6208 | identical |
| OV_15b_tp3_off_notl_r1 | V3codex_q8_15b_tp3_notl_r1 | yes | 346.0 = 346.0 | 1058920 = 1058920 | identical |
| OV_15b_tp3_off_r1 | V3post_q8_15b_tp3_tl_r4 | yes | 346.0 = 346.0 | 1058920 = 1058920 | identical |
| OV_15b_tp3_off_r2 | V3post_q8_15b_tp3_tl_r4 | yes | 346.0 = 346.0 | 1058920 = 1058920 | identical |
| OVf_15b_hybrid_off_r1 | V3final_q8_15b_hybrid_tl_r1 | yes | 61.0 = 61.0 | 182708 = 182708 | identical |
| OVf_15b_pp3_off_r1 | V3codex_q8_15b_pp3_tl_r1 | yes | 3.0 = 3.0 | 6208 = 6208 | identical |
| OVf_15b_tp3_off_r1 | V3post_q8_15b_tp3_tl_r4 | yes | 346.0 = 346.0 | 1058920 = 1058920 | identical |
| OV_7b_hybrid_off_r1 | V3codex_q8_7b_hybrid_tl_r1 | yes | 61.0 = 61.0 | 424372 = 424372 | identical |
| OV_7b_pp3_off_r1 | V3final_q8_7b_pp3_tl_r1 | yes | 3.0 = 3.0 | 14400 = 14400 | identical |
| OV_7b_tp3_off_r1 | V3final_q8_7b_tp3_tl_r1 | yes | 346.0 = 346.0 | 2459752 = 2459752 | identical |
| OVz_15b_tp3_off_r1 | V3post_q8_15b_tp3_tl_r4 | yes | 346.0 = 346.0 | 1058920 = 1058920 | identical |

Also with every option off: `tests/bench/golden_frames.py` ran on the engine of `965385f`, and its six hashes (three placements, with and without the recorder) are pinned in `test_options_off_send_the_frames_of_the_engine_before_part_3`. It passes on the M5 and on gtx.

### O2: the difference of chunked prefill, and why

| pair of runs | activations | eval prompts identical | long prompts | largest first step logit difference |
|---|---|---|---|---|
| `OV_15b_pp3_c16_l3_r1` against `OV_15b_pp3_off_l3_r1` | bf16 (the default) | 8 of 8 | one identical, two differ from token 77 and 88 of 96 | 0.352 |
| `OV_15b_pp3_f32_c16_l3_r1` against `OV_15b_pp3_f32_off_l3_r1` | float32 | 8 of 8 | all three identical | 0.0000916 |

The math of a chunk is the math of the whole prompt. The attention kernel runs on other shapes, so the rounding differs. In float32 that is a difference in the fifth decimal of logits of size 33, and no token changes. In bf16 every operation rounds to 8 bits, the difference grows to a few tenths over 28 layers, and a long generation takes another branch late. The same size of difference separates two placements of one model in part 1 (0.69 between the one rank run and the hybrid run).
So chunked prefill meets its criterion (at least 7 of 8), and it is not bit identical in bf16. Slots, pairs, async send and interleaved stages are bit identical.

## V2. Mutation check

`experiments/session10_cuda_pair/mutate.py`, run in a separate checkout so that the runs were not disturbed.

| round | code | result | file |
|---|---|---|---|
| 1 | `1d39b45` | 59 of 59 caught, two of them only because a test hung until the time limit | `mutations_overlap_round1.json` |
| 2 | `cfab15e` (after pass 1) | 73 of 73 caught, no hang. One entry no longer applied and was corrected and rerun | `mutations_overlap_round2.json`, `round2b.json` |
| 3 | `c352d39` (final compute path) | 88 of 88 caught. The last six entries of the round were disturbed by me (I switched the separate checkout while they ran), so they were run again with the tests of `be73eed`: 13 of 13 | `mutations_overlap_round3.json`, `round3b.json` |

Of the 88 mutations 53 are for part 3: 24 written with the code, 14 after review pass 1, 15 after pass 2. The other 35 are the part 1 mutations, still caught.

The four breaks the task names, and the test that catches each:

| break | mutations | caught by |
|---|---|---|
| the slot id on the wire | the coordinator drops it; a rank ignores it; a head frame travels without it (compute thread, sender thread) | `test_each_request_in_flight_gives_the_tokens_it_gives_alone`, `test_async_send_with_slots_chunks_and_pairs` |
| the add order under interleave | the second request is summed in another order; a pair reads the second sum first | `test_interleaved_requests_give_the_same_tokens_and_the_same_sums` |
| the chunk position offset | every chunk starts at position 0 | `test_chunked_prefill_gives_the_same_tokens_and_the_predicted_frames` |
| the per host memory sum of the planner | the largest rank instead of the sum; the process cost left out; a host over its budget accepted | `test_planner_sums_all_rank_processes_of_a_host_against_the_host_budget` |

## V3. Recorder

- Every span carries its request, its slot and its channel (group link, pipeline link, control link).
- A frame is in flight from the start of its send until the receiver has read it. Sends and receives are matched by order per socket. In every run above with a timeline the list of unmatched links is empty.
- `timeline.html` has a new figure "Busy share and overlap share" per rank, a table per rank and per device, and a row "in flight" under every rank (frames it sent in the upper half, frames addressed to it in the lower half). A run with `--async-send` has a row for the sender thread. The tooltip of every span names the request and the slot. Checked in the browser on `OVc_pair_tp2_off_r1` and `OVc_gtx_pp4_s3_c16_async_r1`, no console error.
- What the numbers show already, as background: on the pair over Wi-Fi with the options off, each GPU computed 31 and 37 percent of the time (`OVc_pair_tp2_off_r1`). On gtx alone, three slots with async send raised the throughput of four pipeline processes from 12.3 to 18.6 tokens per second (`OVc3_gtx_pp4_off_l3_r1`, `OVc3_gtx_pp4_s3_async_l3_r1`).

Two limits a reader must know (both in `docs/MEASUREMENT.md` 3.1):

1. "In flight" ends when the receiver reads the frame. A frame that waits in the buffer of a busy receiver counts. So a TP group shows an overlap share with every option off. The part that answers "does a machine compute while its own data travels" is `overlap_out_share`.
2. With `--async-send` the compute thread does no device sync, so a `compute` span is thread time. The busy share of such a run is not the busy share of the device: on the M5 it reads 7 to 59 percent for runs whose device was busy all the time. For a CUDA run with the option, read the GPU utilization of the sampler. A Mac has no GPU sampler.

## V4. Reviews

| pass | reviewer | result |
|---|---|---|
| 1 | fresh subagent, on the whole change | 0 high, 7 medium, 8 low. One medium (F6, socket buffers) became **high** when checked on gtx. All addressed (`REVIEW_overlap.md`, disposition at the end) |
| 2 | fresh subagent, on the fixes | of pass 1: 10 closed, 5 partly, 0 not closed. New: **1 high** (P1), 2 medium, 6 low. All addressed (`REVIEW_overlap_pass2.md`, disposition at the end) |
| 3 | fresh subagent, on the fix of pass 2 | P1 closed. Of pass 2: 7 closed, 2 partly, 0 not closed. New: 0 high, 1 medium (the test of the fix passed without the fix), 5 low. All addressed (`REVIEW_overlap_pass3.md`, disposition at the end) |
| Codex | for Ohad to run | `REVIEW_REQUEST_CODEX_overlap.md` |

All three reviewers ran the engine of `965385f` against the new engine with every option off: every data frame on every link identical and in the same order, tokens and logits identical (24, 24 and 12 cases; the later two also with the helper thread path forced, which is the path Linux takes).

## Findings worth knowing

1. **Linux grants small socket buffers (high, fixed).** The engine asks for 8 MiB per direction. The Macs grant it. gtx grants 425984 bytes (`net.core.wmem_max` is 212992). Every rank of a TP group sends before it receives, so a frame sent on the compute thread must fit the buffer without the peer reading. A pair doubles that. Before the fix a pair of 7B prefill frames on the CUDA machines could have blocked both ranks for good, and so could one frame between about 0.4 and 1 MiB in the old path. Now the granted size is read back, and a frame goes inline only when it fits together with everything sent on that link and not yet answered, a frame on a helper thread included. The granted sizes are in every record (`ranks[].socket_buffers`). The pc was off when this was measured; the rule does not depend on the value.
2. **On MPS two threads may not use the device (found by a run, not by a probe).** The first design let the sender thread copy the tensor. A probe passed 12,000 copies. The first engine run died after seconds with a Metal assertion (`OVs_15b_pp3_s2_c16_async_r1`, kept). On a Mac `--async-send` now moves only the socket send.
3. **With two slots a pair seldom forms behind a later stage.** In a pure TP group 214 of 320 steps ran as pairs. In the hybrid placement 2 of 320 did. The scheduler does not wait for a partner, because the first stage would idle. The record counts pairs. This is a policy question for part 4 (open, Ohad).
4. **Large frames of a pair had no fixed order.** Two frames over the inline limit were handed to two helper threads for the same link. Fixed before the first run: the second waits for the first.
5. **A sender thread that failed left the run waiting for 30 minutes.** Fixed: it sends its error to the coordinator and closes the rank's links.
6. **The overlap figure was wrong for a pure TP run with slots.** Two ranks can share two sockets. Fixed: spans carry the channel and the match is per socket.

## Memory cost of a host in several stages

| device | two rank processes | four rank processes | per extra process | planned by the planner |
|---|---|---|---|---|
| GTX 1070 (gtx), 1.5B bf16, `nvidia-smi` maximum minus the 734 MiB the desktop uses | 3821 MiB (`OVc4_gtx_pp2_off_l3_r1`) | 4130 MiB (`OVc4_gtx_pp4_off_l3_r1`) | about 155 MiB | 3899 and 4296 MiB: above the measurement |
| M5 (MPS), 1.5B Q8_0, sum of the processes' largest footprint | 5293 MiB (`OV_15b_pp2_contig_off_l3_r1`) | 7007 MiB (`OV_15b_pp4_il_off_l3_r1`) | about 860 MiB | 3178 and 4906 MiB: about 2.1 GB below the measurement |

On CUDA a second process per GPU costs under 2 percent of an 8 GB card, so separate processes are affordable and nothing else was built. On a Mac each process brings its own 1 GiB heap floor, and the planner's sum is not an upper bound there. That matters only for loopback checks: the pair of part 4 is CUDA.

## Runs that did not complete, and why

| run | what happened |
|---|---|
| `OVs_15b_pp3_s2_c16_async_r1` | The MPS finding above: a rank died with a Metal assertion. Outcome `error`, as it should be. Kept as evidence |
| `OVs_15b_pp3_s2_c16_async_r2`, `OVs_15b_tp3_s2_pair_async_r1`, `OVs_15b_hybrid_all_r1`, `OVs_15b_pp4_il_s2_r1`, `OVc_pair_tp2_pair_r1`, `OVc_pair_tp2_pair_async_r1`, `OVc_pair_il_s2_c16_async_r1`, `OVc_gtx_pp4_off_r1`, `OVc_gtx_pp4_s3_c16_async_r1` | The run completed. The driver then stopped in the comparison, because a run folder without `reference.json` was not yet accepted as a reference. Their `record.json` has the tokens and frames but no `correctness` and no `timeline` block. Fixed in `137f921`; the comparison is in the tables above |

## Open

- The pairing policy with two slots (finding 3): Ohad's call before part 4.
- A host with several rank processes and no `host_budget_gb` is accepted and marked "not checked", because every loopback placement from before part 3 is of that kind.
- The pc did not run the final code. It ran the first round, and gtx ran everything after it.
- When a sender thread fails, the record names its error only if the failed frame was the head frame of a head rank. Otherwise the record says a link was lost and the rank's log has the error (review pass 3, Q2).
- On a Mac the planner's per host sum is below the real footprint (table above).
- The Codex review of this part has not run yet.
- **One run of the overlap test file on gtx did not return, and it is not explained.** It was the first run on the last commit. gtx was idle, and the output with the stack dump was lost, because I had piped it through `tail`. The same file then passed there 9 of 9 times (one verbose run, then a loop of 8 with the stack dump kept), and the tests that shrink the socket buffers passed 14 of 14 in a loop. The three real check runs on that commit completed. I could not make it happen again, so I cannot say whether it was a test or the engine. To catch it: run the file on gtx with `timeout` and `-v` into a file, as `/tmp/ov_hunt.sh` there does.
