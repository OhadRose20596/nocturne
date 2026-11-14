# Session 5 report: the memory edge of the Air and M5 pair

Runs: 2026-11-07 to 2026-11-14. llama.cpp `1e7bcf3`. Requester M5 (24 GB, Metal line 18186 MiB = 17.76 GiB). Worker Air M3 (16 GB, Metal line 12124 MiB = 11.84 GiB). Home Wi-Fi, VPN off.
Models: Qwen2.5 Instruct GGUF (bartowski), 7B and 14B Q4_K_M, 32B Q4_K_M, Q5_K_M, Q6_K, Q8_0. torch_tp: 1.5B and 7B bf16.
Source of every number: `EDGE_TABLE.md` and `EDGE_TABLE.csv` (built by `python -m nocturne.bench.derive_edge` from the run records and the engine's own buffer lines). Log excerpts per run are in `evidence/`.
Units: GiB. "Line" is the Metal working set the engine reports. "Headroom" is line minus the node's GPU buffers (weights, KV, compute). CPU buffers are excluded.
Partition names: PP is the layer split (`--split-mode layer`), TP is the in-layer split (`--split-mode tensor`). The ratio after the name is M5 to Air. PP default is the engine's own split (37 or 38 layers on the M5 for 32B).
Time is recorded, not a result (T1).

## Answers in short

- Q-A. With the right split, the pair runs 32B Q4_K_M at 32768 tokens and 32B Q6_K at 16384 tokens. Neither machine alone runs any 32B. 32B Q8_0 runs nowhere.
- Q-B. Weights fill 80 to 95 percent of each node at the edge. KV is the rest. Compute buffers are 0.18 to 0.23 GiB and the same in PP and TP. TP replicates nothing measurable.
- Q-C. No. No (model, context) on this pair needs TP. One cell looked like it (32B Q6_K, ctx 16384). PP completes there when the requester loads the weights without mmap.

## Q-A. Largest model and context per partition

| partition | largest that completes | first that fails | deciding runs |
|---|---|---|---|
| M5 alone | 14B Q4_K_M, ctx 32768 | 32B Q4_K_M at ctx 512 (weights alone 18.08 GiB against 17.76) | `BASELINE_local_qwen2.5-14b-q4km_m5_c32768_r1`, `BASELINE_local_qwen2.5-32b-q4km_m5_c512_r2` |
| Air alone | 14B Q4_K_M ctx 4096; 7B ctx 32768 | 14B at ctx 32768 (14.16 GiB against 11.84) | `BASELINE_local_qwen2.5-14b-q4km_air_c32768_r1` |
| PP default | 32B Q4_K_M ctx 32768; 32B Q6_K ctx 4096 | 32B Q6_K ctx 16384 (Air 0.10 GiB over) | `A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1`, `_r2` |
| PP 3:2 | 32B Q4_K_M ctx 32768; 32B Q6_K ctx 4096 | 32B Q6_K ctx 16384 with mmap (M5 at warm-up) | `A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1` to `_r3` |
| PP 3:2, no mmap | 32B Q6_K ctx 16384 | not probed higher | `A1_wifi_qwen2.5-32b-q6k_m5_c16384propnommap_r1`, `_r2` |
| PP 1:1 | 32B Q4_K_M ctx 16384; 32B Q5_K_M ctx 4096 | 32B Q4_K_M ctx 32768 (Air 1.51 over); Q6_K at ctx 512 | `A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r1` to `_r3` |
| TP 1:1 | 32B Q4_K_M ctx 16384; 32B Q5_K_M ctx 4096 | 32B Q4_K_M ctx 32768 (Air 1.43 over); Q6_K at ctx 4096 | `B1_wifi_qwen2.5-32b-q4km_m5_c32768_r1` |
| TP 3:2 | 32B Q4_K_M ctx 32768; 32B Q6_K ctx 16384 | 32B Q6_K ctx 32768 (both nodes over); Q8_0 at ctx 512 | `B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1` to `_r3` |
| TP, torch_tp bf16 | 7B ctx 4096 (largest tried) | not probed (14B T3 skipped for disk) | `B1t_wifi_qwen2.5-7b-bf16_m5_c4096_r1` |

Observed. Every completed cell answered the graded subset correctly (8/8 for 32B; 7/8 for 14B, the same answer as BASELINE).
Where a BASELINE exists (7B and 14B), PP and TP outputs match it exactly (exact match 1.0).
For 32B no BASELINE fits, so outputs are compared with PP default at the same context: 8/8 identical in every completed cell. At Q6_K ctx 16384, where PP default fails, PP 3:2 without mmap and TP 3:2 are 8/8 identical to each other and to Q6_K PP default at ctx 4096.
Repeats: the Group 1 cells ran 2 or 3 times with the same outcome every time. Most Group 2 and 4 cells ran once.

Limitation. Fit is tested at four context points (512, 4096, 16384, 32768). The true edge lies between the last success and the first failure.

## Q-B. What fills each node

At the edge (per node, GiB):

| cell | node | weights | KV | compute | total | line | headroom |
|---|---|---|---|---|---|---|---|
| 32B Q4_K_M ctx 32768, PP default | M5 | 10.68 | 4.63 | 0.21 | 15.52 | 17.76 | 2.24 |
| | Air | 7.39 | 3.38 | 0.23 | 10.99 | 11.84 | 0.85 |
| 32B Q4_K_M ctx 32768, TP 3:2 | M5 | 10.86 | 5.00 | 0.23 | 16.08 | 17.76 | 1.68 |
| | Air | 7.24 | 3.00 | 0.23 | 10.47 | 11.84 | 1.37 |
| 32B Q6_K ctx 16384, PP 3:2 | M5 | 14.75 | 2.38 | 0.19 | 17.32 | 17.76 | 0.44 |
| | Air | 9.69 | 1.63 | 0.19 | 11.51 | 11.84 | 0.33 |
| 32B Q6_K ctx 16384, TP 3:2 | M5 | 14.68 | 2.50 | 0.19 | 17.38 | 17.76 | 0.38 |
| | Air | 9.79 | 1.50 | 0.19 | 11.48 | 11.84 | 0.36 |

Observed.
- Weights are the largest item everywhere. KV is exactly 256 MiB per 1000 tokens for 32B (64 layers, 8 KV heads, f16), split by layers in PP and by KV heads in TP (8 heads, so a 3:2 ratio becomes 5:3).
- Compute buffers grow slowly with context: 178 MiB at 512, 227 MiB at 32768 for 32B. They are the same in PP and TP (`sched_reserve` lines).
- TP replicates nothing in the Metal buffers. At ctx 4096, 32B Q4_K_M: TP 1:1 is 9.72 GiB per node, balanced PP is 9.75 and 9.69. The output layer is split in TP, as it is placed whole in PP.
- The embedding stays on the M5's CPU in both modes: 418 MiB (Q4_K_M) and 609 MiB (Q6_K). It is outside the Metal line.
- Failure classes follow the per-node sums: in 60 of 61 attempted llama.cpp cells, "every node total under its line" predicts fit or fail correctly. The one exception is Q-C's cell below.
- The line is not an allocation limit. In Q8_0 TP 3:2 at ctx 512 the Air accepted buffers 0.95 GiB above its line without a load error. Failures appear at the first command buffer (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), at warm-up or at the first request.

Limitation. These are the engine's buffer reservations, not peak resident memory. TP per node values are derived: llama.cpp prints one `Meta()` device line (the requester's share), and the worker's share is computed from the `-ts` ratio. The ratio was checked once against the worker's received bytes (2026-11-07), not in every run. torch_tp prints no buffer lines, so its memory per node is unknown.
Hypothesis. KV populated by real long prompts does not change the reservation: the long prompt runs (G3) show identical buffer lines and all completed. Peak resident memory during prefill was not measured.

## Q-C. Is there a (model, context) that TP completes and PP does not?

Answer: no.

The deciding row is 32B Q6_K at ctx 16384:

| partition | runs | outcome | M5 headroom | Air headroom |
|---|---|---|---|---|
| PP default | 0/2 | Air out of memory | 0.87 | -0.10 |
| PP 3:2 (mmap, default load) | 0/3 | M5 out of memory at warm-up | 0.44 | 0.33 |
| PP 3:2, `--load-mode none` | 2/2 | completes, 8/8 | 0.44 | 0.33 |
| TP 3:2 | 3/3 | completes, 8/8 | 0.38 | 0.36 |

Observed. With the default load, TP 3:2 completed where both PP splits failed. This looked like a falsifier.
PP default fails for a plain reason: one layer too many on the Air (0.10 GiB over).
PP 3:2 moves that layer to the M5. The buffer sums fit on both nodes, yet the M5 failed at the warm-up run every time.
The only other difference between the PP 3:2 and TP 3:2 logs is the load mode of the requester's weights: PP maps the GGUF file into a Metal buffer (`MTL0_Mapped`, mmap); TP copies them into its own buffer.
PP 3:2 with `--load-mode none` (flag present on `1e7bcf3`) has identical buffer sizes and completed 2/2.
So the partition does not decide this cell. The load mode does.

Limitation. Two runs without mmap; one cell. The mmap cost itself was not measured. It is above 0.44 GiB on the M5 at this load and below the 2.23 GiB of headroom where PP 3:2 with mmap still works (Q6_K ctx 4096).
Hypothesis. A memory mapped Metal buffer on unified memory carries the file's page cache residency on top of the GPU working set, so it costs more than a private buffer near the line. Not checked.
Consequence for the placement code (O5): when the requester's headroom is under about 2 GiB, launch llama-server with `--load-mode none`.

## The crossover rule, from the measurements

Notation, per node i: `C_i` line, `W_i` weights, `K_i(ctx)` KV, `B_i(ctx)` compute buffer. Total weights `W`, total KV `K(ctx)`.
Per node feasibility: `T_i = W_i + K_i(ctx) + B_i(ctx) <= C_i` (and, on the requester under PP with mmap, enough extra headroom for the mapping; see Q-C).

Measured facts that shape the rule:
1. `T_m5 + T_air = W + K(ctx) + 2 B(ctx)` in both PP and TP, within 0.06 GiB. So `R_tp,i = 0` and `B_tp,i = B_pp,i` on this engine.
2. PP can move weight and KV between nodes only in whole layers. One layer costs `u(ctx) = w_layer + k_layer(ctx)`: 0.28 to 0.41 GiB for 32B Q4_K_M, 0.38 to 0.51 for Q6_K.
3. TP moves them in much finer steps (rows and KV heads), set by the `-ts` ratio.

Define the pair's slack `S(ctx) = C_m5 + C_air - (W + K(ctx) + 2 B(ctx))`.

- Nothing fits if `S < 0`.
- PP fits if `S >= u(ctx)`. The feasible window for the M5's share has width `S`, so a window at least one layer wide always contains a layer boundary.
- TP fits if `S >= 0` and the ratio is chosen inside the window.
- TP is necessary on memory grounds only if `0 <= S < u(ctx)` and no layer boundary falls inside the window. That is, `(R_tp + B_tp - B_pp) = 0` and TP's only advantage is granularity, worth less than one layer.

Check against every (model, context) with both nodes in use:

| model | ctx | S | u | rule says | PP observed | TP observed |
|---|---|---|---|---|---|---|
| 7B Q4_K_M | 4096 | 25.06 | 0.17 | PP fits | fits | fits |
| 7B Q4_K_M | 32768 | 23.47 | 0.23 | PP fits | fits | fits |
| 14B Q4_K_M | 4096 | 20.69 | 0.18 | PP fits | fits | fits |
| 14B Q4_K_M | 32768 | 15.21 | 0.29 | PP fits | fits | fits |
| 32B Q4_K_M | 512 | 11.04 | 0.28 | PP fits | fits | fits |
| 32B Q4_K_M | 4096 | 10.16 | 0.30 | PP fits | fits | fits |
| 32B Q4_K_M | 16384 | 7.14 | 0.35 | PP fits | fits | fits |
| 32B Q4_K_M | 32768 | 3.09 | 0.41 | PP fits | fits (default, 3:2) | fits (3:2) |
| 32B Q5_K_M | 4096 | 7.08 | 0.35 | PP fits | fits | fits |
| 32B Q6_K | 512 | 4.68 | 0.38 | PP fits | fits | fits |
| 32B Q6_K | 4096 | 3.80 | 0.40 | PP fits | fits | fits |
| 32B Q6_K | 16384 | 0.77 | 0.45 | PP fits | fits (3:2, no mmap) | fits (3:2) |
| 32B Q6_K | 32768 | -3.33 | 0.51 | nothing fits | not run (fails at 16384 with default) | fails |
| 32B Q8_0 | 512 | -2.27 | 0.50 | nothing fits | fails | fails |
| 32B Q8_0 | 4096 | -3.15 | 0.51 | nothing fits | fails | fails |

Observed. The rule holds on every row. "Fits" means the best split tried; badly matched splits (1:1) fail where `S` is small, in PP and TP alike.
The rule is never tested in its TP-only zone (`0 <= S < u`): no row has slack below one layer. The closest is Q6_K at ctx 16384 (S 0.77, u 0.45).

Arithmetic for larger dense models (not measured). A 70B class dense model (80 layers, hidden 8192) has about 0.86 billion parameters per layer, about 0.5 GiB at Q4_K_M. Its output layer is about 1 GiB. Both are far below the Air's 11.84 GiB. A single layer never forces TP on nodes of this class. The whole model, about 40 GiB at Q4_K_M, exceeds the pair's 29.6 GiB, so it needs a third node, not a different split.
MoE caveat (not tested). An expert layer holds all its experts. A layer of a 600B class MoE model is about 11 billion parameters, about 6 GiB at Q4. That still fits the Air alone, but with little room for KV. Whether llama.cpp's split handles experts as whole layers or rows was not checked.

Answer to the supervisor's question for this pair: memory never forced TP for dense models up to 32B. TP is a granularity tool. It can matter only when the pair's slack is under one layer, and no such case occurred. The cases that would force it are a single layer larger than the small node's headroom (not reached by dense models up to 70B at Q4 on 16 GB) or MoE expert layers (not tested).

## Hybrid, from the numbers (not measured)

Every failure here is either `S < 0` (not enough total memory: Q8_0, Q6_K at 32768) or a split ratio that does not match the node sizes (1:1 on a 24 GB and a 16 GB machine). None is a layer too big for a node. So a third node helps as one more PP stage that adds capacity. A TP pair inside a stage would pay off only if one layer did not fit a node's headroom, which these models never approach. For the three machine plan: PP stages with ratios proportional to each node's line, TP only as a last resort for the last fraction of a layer.

## What was not checked

- Peak resident memory. All numbers are the engine's buffer reservations.
- The size of the mmap overhead on the requester, and whether it also affects BASELINE runs near the line.
- The worker as requester (Air to M5 direction). The M5 was always the requester.
- Wired networking, Tailscale, two homes. Home Wi-Fi only.
- torch_tp memory per node, and T3 (14B bf16, skipped for disk).
- MoE models.
- Context points between the four tested ones.
- Most Group 2 and Group 4 cells ran once.

## Incidents that cost runs (no data lost, listed for honesty)

- 2026-11-08: the Air slept mid Group 2 (macOS sleep after one minute). 12 runs had no worker. They are excluded in `EDGE_TABLE.md`. Fixed by `caffeinate` on both Macs and an ssh gate before every run.
- 2026-11-14: the worker's tensor cache (`ggml-rpc-server -c`, `~/Library/Caches/llama.cpp/rpc`) grew to 63 GB and filled the Air's disk. Preflight stopped the run. The orchestrator now clears the cache when free space falls under 25 GB. This affects load time only, never memory.
- One harness error (remote binary check for the Air BASELINE), fixed and rerun.
