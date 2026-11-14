# Session 5, memory edge: pre-registered predictions

Written 2026-11-07 before any run. Setup: M5 (24 GB, Metal line 18.2 GB) requester, Air (16 GB, Metal line 12.1 GB) worker, home Wi-Fi, llama.cpp `1e7bcf3`.
The Ohad column is empty on purpose. Ohad fills it before the first run. The outcome column is filled after the runs, failures included.

## Hypothesis (Claude)

On two machines, TP does not fit any model that PP cannot fit. Prediction: the memory edge of PP is at or above the edge of TP for every context tested.
One run where TP completes and PP fails (same model, same context, same engine) falsifies it.

## Derivation for 32B Q4_K_M

Source: the engine's own log of `experiments/r3/capacity/20261010-164354_MMF4FWRG4R_cap_split_Qwen2.5-32B-Instruct-Q4_K_M_P1` (PP, M5 requester, ctx 2048). The GGUF header was not read separately (no `gguf` package here); the `print_info` lines of that log come from the same file.

- Structure: 64 layers, hidden 5120, 40 heads, 8 KV heads, head dim 128, vocab 152064.
- Weights (MiB, measured): M5 10939 (37 layers plus output layer), Air 7569 (27 layers), CPU 417.66 (embedding).
- Per layer: about 280 MiB. Output layer: about 608 MiB.
- KV, f16: 4096 B per layer per token, 262144 B per token over 64 layers. That is 256 MiB per 1k tokens in total. Measured at ctx 2048: 296 MiB (37 layers) plus 216 MiB (27 layers). It matches.
- Compute buffer at ctx 2048: 184 MiB on each GPU node, 22 MiB on CPU. I assume it is flat in ctx (flash attention). Not verified.
- Other models: 14B has 48 layers, 8 KV heads, KV 192 KiB per token. 7B has 28 layers, 4 KV heads, KV 56 KiB per token.

## Group 1. 32B Q4_K_M, ctx sweep (per node, ctx 4096, MiB)

| cell | quantity | Ohad | Claude | outcome |
|---|---|---|---|---|
| PP default (about 37 / 27 layers) | M5 model / KV / compute | | 10939 / 592 / 184 (plus CPU 418 and 22) | 10939 / 592 / 184 MiB, CPU 418 and 24. Holds. `A1_wifi_qwen2.5-32b-q4km_m5_c4096_r1` |
| PP default | Air model / KV / compute | | 7569 / 432 / 184 | 7569 / 432 / 184 MiB. Holds (same 37/27 split as 2026-10-10). Same run |
| PP default | totals vs line | | M5 about 11.7 GiB of 18.2 GB. Air about 8.0 GiB of 12.1 GB. Headroom Air about 3 GB | M5 11.44 GiB, headroom 6.32 GiB. Air 8.00 GiB, headroom 3.84 GiB (line 12124 MiB). Holds |
| PP balanced `-ts 1,1` (32 / 32) | M5 model / KV / compute | | 9560 / 512 / 184 | 9297 / 492 / 184 MiB. Weights 260 MiB lower than predicted: 1:1 gave the M5 31 layers plus the output layer, not 32 plus output. `A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r1` |
| PP balanced | Air model / KV / compute | | 8950 / 512 / 184 | 9211 / 532 / 184 MiB (33 layers). Holds within 3 percent. Same run |
| TP `-ts 1,1 -fit off` | M5 model / KV / compute | | 9500 to 9900 / 512 / 250 to 600 (low confidence, I have no TP number) | 9255 / 512 / 186 MiB. Fails: weights below the range and compute buffer equal to PP, not larger. `B1_wifi_qwen2.5-32b-q4km_m5_c4096_r1` |
| TP | Air model / KV / compute | | 8700 to 9000 / 512 / 250 to 600 | 9255 / 512 / 186 MiB (requester share applied to the worker; checked against 9.9 GB of set_tensor on a cold load). Weights fail high, compute fails low |
| TP vs balanced PP | requester total | | TP requester 0.3 to 0.6 GB heavier. Both per node totals within 10 percent of balanced PP (Part 6 of session 4, kept) | Within 10 percent: holds (9.72 against 9.74 and 9.69 GiB). Requester heavier: fails. TP splits the output layer too and keeps only the 418 MiB embedding on CPU, as PP does; the requester is 0.02 GiB lighter in TP |

| ctx | BASELINE M5 | A1 default | A1 balanced | B1 TP |
|---|---|---|---|---|
| 512 | Ohad: / Claude: OOM at first compute (as 2026-10-10 at 2048; KV saving is only 0.4 GiB) / outcome: oom_requester_compute, holds (`..._m5_c512_r2`) | Ohad: / Claude: completes / outcome: completes, 8/8 correct | Ohad: / Claude: completes / outcome: completes | Ohad: / Claude: completes, or tool_unsupported with one error line (I give 35 percent to tool_unsupported) / outcome: completes; tensor mode works over RPC on 1e7bcf3 for 32B (the 2026-10-10 failure is not reproduced) |
| 4096 | Ohad: / Claude: OOM / outcome: oom_requester_compute, holds | Ohad: / Claude: completes / outcome: completes | Ohad: / Claude: completes / outcome: completes | Ohad: / Claude: as ctx 512 / outcome: completes |
| 16384 | Ohad: / Claude: OOM / outcome: oom_requester_compute, holds | Ohad: / Claude: completes (KV 4 GiB total; Air about 9.4 GiB) / outcome: completes, Air 9.27 GiB, holds | Ohad: / Claude: completes (Air about 10.3 GiB, headroom 1.8 GB) / outcome: completes, Air 11.25 GiB, headroom 0.59 GiB; total within 1 GiB, headroom overestimated (unit mix) | Ohad: / Claude: completes if ctx 512 did / outcome: completes, Air 11.23 GiB |
| 32768 | Ohad: / Claude: OOM / outcome: oom_requester_compute, holds | Ohad: / Claude: completes with Air headroom under 1.5 GB (about 11.0 GiB used, KV 8 GiB total) / outcome: completes, Air 10.99 GiB, headroom 0.85 GiB, holds | Ohad: / Claude: fails on the Air, oom_worker (about 12.9 GiB against 12.1 GB) / outcome: oom_worker_load, Air 13.35 GiB, holds | Ohad: / Claude: fails on the Air, same arithmetic plus replication / outcome: oom_worker_load, Air 13.27 GiB; fails as predicted but there is no replication term. Added cell TP 3:2: completes, Air 10.46 GiB, headroom 1.37 GiB (`B1_..._c32768prop_r1`) |

Note: units of the 12.1 GB line against GiB figures are mixed above. The EDGE_TABLE will use one unit.
Falsifier to watch: balanced PP failing at 32768 while default PP completes shows that the split choice matters more than PP versus TP.

## Group 2. Crossing the edge, ctx 4096 (weights: Q5_K_M about 23.3 GB, Q6_K about 26.9 GB, Q8_0 about 34.8 GB)

| model | A1 default | A1 balanced | B1 TP | Ohad | outcome |
|---|---|---|---|---|---|
| L4 32B Q5_K_M | completes (Air about 9 GiB, M5 about 13 GiB) | completes | completes if TP works at all | | Holds. All five cells complete. PP default M5 13.24 / Air 9.29 GiB. Balanced PP Air 11.29 GiB, headroom 0.55. TP 1:1 Air 11.26 GiB. 8/8 correct, identical to PP default. `*_q5km_m5_c4096*` |
| L5 32B Q6_K | completes with under 1.5 GB headroom on the Air with a split near 27 / 37 layers; I give 60 percent to completes, 40 percent to OOM on one node | likely fails or marginal, since 1,1 puts about 12.6 GiB on the Air | fails or marginal, same arithmetic as balanced PP | | Holds. PP default completes, Air 10.66 GiB, headroom 1.18. Balanced PP fails, oom_worker_load, Air 12.99 GiB. TP 1:1 fails the same way, Air 12.90 GiB. Added cells PP 3:2 and TP 3:2 both complete (Air headroom 1.56 and 1.49). |
| L6 32B Q8_0 | fails at load (34.8 GB against 30.3 GB combined) | fails | fails | | Holds. Every cell fails at ctx 4096 and at ctx 512: oom_worker_load in PP and TP 1:1, oom_requester_compute in TP 3:2 (M5 weight share alone 18.8 GiB against a 17.76 GiB line). |

Q-C prediction: no. No (model, ctx) exists where TP completes and PP fails. Confidence about 85 percent. The 15 percent is mostly that PP default packs the Air too tightly in a case where a TP split would not.

Outcome: holds, with one qualification. At 32B Q6_K ctx 16384, TP 3:2 completed 3/3 while PP default failed 2/2 (Air 0.10 GiB over the line) and PP 3:2 failed 3/3 (M5 out of memory at the warm-up run, 0.44 GiB under the line by the buffer sums). PP 3:2 with `--load-mode none` completed 2/2 with the same buffers and outputs identical to TP 3:2 (8/8). The failure is the default memory mapped load on the requester, not the partition. The 15 percent case (PP default packs the Air too tightly) did happen at this cell; a one layer shift fixes it.
Context sweep outcome: PP edge for Q6_K is ctx 16384 (no mmap), TP 3:2 edge is ctx 16384. Both fail at 32768. Equal, as predicted.
Context sweep for the largest PP model that completed: PP edge in ctx at or above TP edge. Stop rule applies to a model failing at ctx 512 in both.

## Group 3. Real long prompts at the edge

| item | Ohad | Claude | outcome |
|---|---|---|---|
| Populated KV (eval_long and about 4000 token prompt) against reserved KV | | llama.cpp reserves the whole KV at load, so peak memory does not move by more than 0.2 GB from empty to populated. Compute buffers can grow by a few hundred MiB at the first long prefill | Partly checked. Every long prompt run completed at the edge of its partition (Q6_K ctx 4096 PP default and PP 3:2, Q6_K ctx 16384 TP 3:2). The engine's buffer lines are identical to the short prompt runs, compute buffers included: no growth. Peak resident memory during the long prefill was not measured, so the 0.2 GB bound is not tested. |
| Output identical to BASELINE where BASELINE exists | | 14B and 7B: identical on the graded subset. 32B has no BASELINE on the M5, so correctness for 32B is graded against the expected answers only, not against the engine | Holds. 14B and 7B, PP and TP, both ctx: exact match with BASELINE 1.0. 32B: every completed cell answers 8/8 and is identical to PP default at the same ctx (8/8); long prompts 3/3 and 1/1 identical. |

## Group 4. 14B and 7B, ctx 4096 and 32768

| item | Ohad | Claude | outcome |
|---|---|---|---|
| 14B BASELINE M5, both ctx | | completes (weights 9 GB plus KV 6 GiB at 32768 is about 16 GiB, under 18.2 GB) | Holds. 8.81 and 14.16 GiB |
| 14B BASELINE Air, ctx 4096 | | completes (about 10 GiB, under 12.1 GB) | Holds. 8.81 GiB, headroom 3.01 (I overestimated by about 1 GiB) |
| 14B BASELINE Air, ctx 32768 | | fails (about 15 GiB) | Holds. 14.16 GiB, out of memory at first compute |
| 14B PP and TP, both ctx | | complete; PP and TP per node totals within 10 percent | Completes: holds. Within 10 percent: fails, because PP default is not balanced (M5 5.43 against Air 3.48 GiB at 4096, TP 4.46 on each). The sums agree (8.91 against 8.92 GiB) |
| 7B all cells, both ctx | | complete (KV 1.75 GiB at 32768) | Holds. KV 1.75 GiB at 32768 exactly |

## Group 5. torch_tp, bf16

| item | Ohad | Claude | outcome |
|---|---|---|---|
| T1 1.5B ctx 4096 | | completes | Holds, 8/8 correct (`B1t_wifi_qwen2.5-1.5b-bf16_m5_c4096_r1`) |
| T2 7B ctx 4096 | | completes (B1t exists), about 7.6 GB sharded plus about 2.2 GB replicated per node | Completes: holds, 8/8. Memory per node not measured (torch_tp writes no buffer lines), so the split is untested |
| T3 14B | | OOM on the Air side at load (29.5 GB over two nodes, replicated embedding and head about 2.2 GB each leaves under 1 GB on the Air) | Not run (disk; optional in the brief) |

## Not predicted, stated up front

- TP compute buffer size in llama.cpp tensor mode over RPC. I have no number. The range above is a guess.
- Whether `--split-mode tensor` works over RPC on `1e7bcf3` at all (it failed on 32B on 2026-10-10 for an unknown reason). Under ADR-007 a TP failure is not evidence about PP.
- Time per token. It is recorded, not predicted (T1).
