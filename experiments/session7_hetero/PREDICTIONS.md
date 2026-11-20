# Session 7a: pre-registered predictions

Written 2026-11-16 on the M5, before the L1 run (run by Ohad by hand, 2026-11-16) and before any run of the mixed device option. Time per token is not predicted (T1).

## L1_tp_mixed_0.5b_loopback_r1

Placement: one TP group, rank s0/0 on MPS and rank s0/1 on CPU, both on the M5 over loopback, 0.5B bf16, `allow_mixed_devices: true`.
References: `R0_qwen2.5-0.5b-bf16_mps_r2` and `R0_qwen2.5-0.5b-bf16_cpu_r1` (session 6 folder).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 7 heads and 2432 MLP rows per rank | equal shares, 14 heads | Planner: accepted, 7 heads and 2432 MLP rows per rank (PLANS.md). |
| Frame formula | holds: 100 frames per decode step (token 1, embedding 2, layers 96, head 1) | same group size and layer count as the formula; device type does not change the frame count | Held: 100 measured, 100 predicted (280 decode steps). |
| Run outcome | completes | the all-reduce goes through the host in fp32 for both device types | Completed. |
| Identical vs MPS reference | 7 or 8 of 8, most likely 8 | half of the heads run on MPS; the other half differ by kernel rounding only | 8 of 8 (reference `R0_qwen2.5-0.5b-bf16_mps_r2`). Within the prediction. |
| Identical vs CPU reference | 7 or 8 of 8, most likely 7 | the two references already differ from each other on the long prompt (token 10) | 8 of 8 (reference `R0_qwen2.5-0.5b-bf16_cpu_r1`). Within the prediction, one better than the most likely value. |
| Criterion (at least 7 of 8 identical, same graded accuracy) | met against both, confidence about 65 percent | accuracy is a coarse grade; the risk is one extra flip in a category with a threshold | Met against both. Accuracy 0.875, equal to both references. |
| First step logit diff | nonzero, same order as MPS against CPU in P9 (about 1e-1 in bf16 logits) | different kernels per rank | Wrong order of magnitude: 0.99 against MPS, 0.73 against CPU, not about 0.1. Tokens still identical. |
| What would count as a failure of the reasoning | fewer than 7 identical against both references | would mean the mixed sum is not the same on both ranks, not only kernel rounding | Not triggered. Long prompt first divergence: token 10 against MPS (the two references already differ there), token 110 against CPU. |


## H1_pp_1.5b_r1

Written 2026-11-16 before the run. Placement: PP, stage s0 layers 0 to 14 with the embedding on the M5 (MPS), stage s1 layers 14 to 28 with the head on the PC (CUDA), over the phone hotspot. Coordinator on the M5.
References: `R0_qwen2.5-1.5b-bf16_mps_r1` (session 6) and `R0_qwen2.5-1.5b-bf16_cuda_r1` (session 7).

| item | prediction | why | outcome |
|---|---|---|---|
| Preflight | passes: ssh, same engine and model sha256, nvidia-smi line for pc, Mac items "n/a" on pc | checked by hand before the run | Passed. m5 Darwin (caffeinate on, AWDL up), pc Linux with the Mac items "n/a", nvidia-smi recorded, engine and model sha256 equal. |
| Run outcome | completes | each stage runs whole layers on one device; only the hidden state crosses the link | Completed. |
| Frame formula | holds: 2 frames per decode step (hop 1, head 1) | one hop from s0 to s1, the head result back to the coordinator | Held: 2 measured, 2 predicted (279 decode steps, about 3.1 KB per step). |
| Identical vs MPS reference | 8 of 8 | first half of the layers is the same MPS computation; both R0s already agree 8/8 with each other | 8 of 8 (`R0_qwen2.5-1.5b-bf16_mps_r1`). |
| Identical vs CUDA reference | 8 of 8 | same argument for the second half | 8 of 8 (`R0_qwen2.5-1.5b-bf16_cuda_r1`). |
| Criterion | met against both | | Met against both. Accuracy 1.0, equal to both references. |
| First step logit diff | below 1.0 against both, nonzero | the logits come from CUDA but half the residual stream was computed on MPS | Held: 0.44 against MPS, 0.45 against CUDA. |
| Long prompt first divergence | after token 50 against both | the R0s diverge from each other at tokens 54 to 97 | Held: token 146 against MPS, token 97 against CUDA. |

## H2_tp_1.5b_r1

Written 2026-11-16 before the run. Placement: one TP group over all 28 layers, 6 heads each, rank 0 on the M5 (MPS), rank 1 on the PC (CUDA), embedding and head split by vocab rows, `allow_mixed_devices: true`, phone hotspot. Coordinator on the M5.
References: `R0_qwen2.5-1.5b-bf16_mps_r1` (session 6) and `R0_qwen2.5-1.5b-bf16_cuda_r1` (session 7).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 4480 MLP rows each, vocab split at row 76032 | PLANS.md | Accepted as in PLANS.md. |
| Run outcome | completes | L1 showed the mixed group works on one host; the link only adds latency | Completed. |
| Frame formula | holds: 116 frames per decode step (token 1, embedding 2, layers 112, head 1) | 28 layers, 2 all-reduces each, 2 ranks | Held: 116 measured, 116 predicted (279 decode steps, about 353 KB per step). |
| Identical vs MPS reference | 7 or 8 of 8, most likely 8 | as in L1, half of every layer runs on each device and the sum is the same on both ranks | 8 of 8 (`R0_qwen2.5-1.5b-bf16_mps_r1`). |
| Identical vs CUDA reference | 7 or 8 of 8, most likely 8 | same | 8 of 8 (`R0_qwen2.5-1.5b-bf16_cuda_r1`). |
| Criterion | met against both | | Met against both. Accuracy 1.0, equal to both references. |
| First step logit diff | between 0.3 and 1.1 against both | L1 gave 0.73 to 0.99 on 0.5B; H1 gave 0.44 on 1.5B | Held: 0.56 against MPS, 0.50 against CUDA. |
| What would count as a failure of the reasoning | fewer than 7 identical against both, or a rank disconnect on the hotspot | | Not triggered. No disconnect: 16416 all-reduces per rank. Long prompt first divergence token 76 against both. |

## H3_pp_7b_r1

Written 2026-11-16 before the run, while the 7B files were copied to the PC. Placement: PP in three stages, s0 layers 0 to 22 with the embedding on the M5 (MPS), s1 layers 22 to 28 on the PC (CUDA, budget 3.5 GB), stage `head` with the output head and final norm on the M5 (MPS, port 50071). Phone hotspot, coordinator on the M5.
References: `R0_qwen2.5-7b-bf16_mps_r3` and `R0_qwen2.5-7b-bf16_cpu_r1` (session 6). There is no CUDA reference for 7B: it does not fit the 8 GB card (about 6.2 GB usable).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts; s1 planned 2.962 GB against the 3.5 GB budget | PLANS.md | |
| PC memory | loads within the budget; peak allocated about 3.0 GB, below the 6.2 GB usable | 6 layers of 466 MB plus KV 50 MB plus workspace | |
| M5 memory | both M5 ranks load (11.3 GB plus 1.1 GB of tensors) | R0 7B on MPS (15.2 GB) already ran on this machine | |
| Run outcome | completes | as H1, whole layers per device | |
| Frame formula | holds: 3 frames per decode step (hops 2, head 1) | s0 to s1, s1 to head, the head result back | |
| Identical vs MPS reference | 8 of 8 | 22 of 28 layers and the head are the same MPS computation | |
| Identical vs CPU reference | 7 or 8 of 8 | the MPS and CPU 7B references are not identical to each other everywhere | |
| Criterion | met against both | | |
| First step logit diff | below 1.0 against both, nonzero | | |

# Session 8 (2026-11-20): home Wi-Fi, CUDA peer `gtx`

The CUDA machine of session 7 (`pc`, RTX 2000 Ada) was not available. The CUDA peer is `gtx`: GTX 1070 8 GB, WSL2 Ubuntu 26.04, torch 2.14.1+cu126, Python 3.14.4.
The placements are copies with `host: gtx` (`*_gtx.yaml`). Run ids carry `homewifi_gtx`. All blocks below were written before any run of this session.
The prompt asked for run ids without `gtx`. The suffix was added so nobody reads these as runs on the session 7 card.

## R0_qwen2.5-1.5b-bf16_cuda_gtx_r1

Written 2026-11-20 before the run. Placement `R0_1.5b_cuda_gtx.yaml`: one rank, all 28 layers on gtx (CUDA). Coordinator on gtx. No network.
Compared with `R0_qwen2.5-1.5b-bf16_cuda_r1` (RTX 2000 Ada) and `R0_qwen2.5-1.5b-bf16_mps_r1`.

| item | prediction | why | outcome |
|---|---|---|---|
| Run outcome | completes, planned 3256 MB fits the 8 GB card | PLANS.md; about 1.2 GB is in use by Windows | Held. Completed. Peak allocated 3.223 GB, VRAM free at ready 4.14 of 8.59 GB. |
| transformers generate | identical on every prompt | same weights, same device, same greedy decode; held for the Ada R0 | Wrong in part. 8 of 8 short prompts identical. The long prompt differs at token 5. |
| Identical vs Ada CUDA reference | 7 or 8 of 8 | same wheel family but a Pascal card has no native bf16 path, so kernels can round differently | Held: 8 of 8. |
| Identical vs MPS reference | 7 or 8 of 8 | the Ada and MPS R0s agree 8 of 8 on the short prompts | Held: 8 of 8. |
| First step logit diff vs Ada | nonzero, below 1.0 | different GPU generation, same arithmetic in principle | Held: 0.531 (0.500 against MPS). Long prompt first divergence token 21 against both. |

## H1_pp_qwen2.5-1.5b-bf16_homewifi_gtx_r1

Written 2026-11-20 before the run. Placement `H1_pp_1.5b_gtx.yaml`: layers 0 to 14 and the embedding on the M5 (MPS), layers 14 to 28 and the head on gtx (CUDA). Home Wi-Fi. Coordinator on the M5.
References: `R0_qwen2.5-1.5b-bf16_mps_r1`, `R0_qwen2.5-1.5b-bf16_cuda_r1` (Ada) and `R0_qwen2.5-1.5b-bf16_cuda_gtx_r1`.

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 1886.5 MB on m5 and 1887.4 MB on gtx | PLANS.md | Held. Accepted as in PLANS.md. |
| Run outcome | completes | same placement shape as H1 on the hotspot, which completed | Held. Completed, first attempt. |
| Frame formula | holds: 2 frames per decode step | one hop and the head | Held: 2 measured, 2 predicted (279 decode steps, 3112 bytes per step). |
| Identical vs MPS reference | 8 of 8 | H1 on the hotspot gave 8 of 8 | Held: 8 of 8. |
| Identical vs CUDA references | 8 of 8 against the Ada reference, 7 or 8 of 8 against the gtx reference | H1 gave 8 of 8 against Ada; the gtx reference is new | Held: 8 of 8 against Ada, 8 of 8 against gtx. |
| Criterion | met (at least 7 of 8 against at least one reference, same graded accuracy) | | Held. Met against all three. Accuracy 1.0, equal to every reference. |
| First step logit diff | below 1.0 against every reference | H1 gave 0.44 and 0.45 | Held: 0.375 MPS, 0.500 Ada, 0.508 gtx. |

## H4_pp3_qwen2.5-1.5b-bf16_homewifi_gtx_r1

Written 2026-11-20 before the run. Placement `H4_pp3_1.5b_gtx.yaml`: layers 0 to 10 and the embedding on the M5 (MPS), 10 to 19 on the Air (MPS, torch 2.15.0), 19 to 28 and the head on gtx (CUDA). Home Wi-Fi. Coordinator on the M5.
References: the same three as H1.

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 1495.3 MB on m5, 930.8 MB on air, 1398.4 MB on gtx | PLANS.md | Held. Accepted without edits. |
| Run outcome | completes, no rank disconnect | whole layers per device, three ranks | Held. Completed, first attempt, 0 reconnects. |
| Frame formula | holds: 3 frames per decode step (2 hops and the head) | H1 gave 2 with one hop | Held: 3 measured, 3 predicted (279 decode steps, 6208 bytes per step). |
| Identical vs MPS reference | 7 or 8 of 8 | 19 of 28 layers on MPS; the Air differs from the M5 at the rounding edge (session 6) | Held: 8 of 8. |
| Identical vs CUDA references | 7 or 8 of 8 against each | 9 layers and the head on CUDA | Held: 8 of 8 against Ada, 8 of 8 against gtx. |
| Criterion | met | | Held. Met against all three. Accuracy 1.0. |
| Per rank memory | as in the planner table, within 15 percent | | Held. Allocated after the run: m5 1456 MB (planned 1495), air 887 MB (planned 931), gtx 1358 MB (planned 1398), peak on gtx 1366 MB. |

## H5_tp3_qwen2.5-1.5b-bf16_homewifi_gtx_r1

Written 2026-11-20 before the run. Placement `H5_tp3_1.5b_gtx.yaml`: one TP group over all 28 layers, 4 heads each on the M5 (MPS), the Air (MPS) and gtx (CUDA), `allow_mixed_devices: true`. Home Wi-Fi. Coordinator on the M5.
References: the same three as H1.

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 1105.9 MB on m5, 1186.9 MB on air, 1138.7 MB on gtx | PLANS.md | Held. Accepted without edits. |
| Run outcome | completes, no rank disconnect | H2 held 16416 all-reduces on the hotspot with two ranks | Held. Completed, first attempt. 16416 all-reduces per rank, 0 reconnects. |
| Frame formula | holds: 346 frames per decode step (token 2, embedding 6, layers 336, head 2) | the `plan` line for this placement | Held: 346 measured, 346 predicted (279 decode steps, 1058920 bytes per step). |
| Identical vs MPS reference | 7 or 8 of 8 | partial sums are added in fp32 on the host in a fixed order; three ranks were only checked on two Macs (P3, session 6) | Held: 8 of 8. The long prompt is identical for all 256 tokens. |
| Identical vs CUDA references | 7 or 8 of 8 against each | same | Held: 8 of 8 against Ada, 8 of 8 against gtx. |
| Criterion | met | | Held. Met against all three. Accuracy 1.0. |
| What would count as a failure of the reasoning | garbled text from the first token (transport or dtype, not rounding), or a rank disconnect | | Not triggered. Text is correct from the first token. First step logit diff 0.469 MPS, 0.418 Ada, 0.511 gtx. |

# Session 8b (2026-11-20): 7B on three machines, CUDA peer `gtx`

Same hosts and link as session 8. All blocks were written before any 7B run. References: `R0_qwen2.5-7b-bf16_mps_r3` and `R0_qwen2.5-7b-bf16_cpu_r1` (session 6).
There is no CUDA reference for 7B: the model (15.2 GB) does not fit the 8 GB card. The criterion is at least 7 of 8 identical against at least one reference, with the same graded accuracy.

## H3_pp_qwen2.5-7b-bf16_homewifi_gtx_r1

Placement `H3_pp_7b_gtx.yaml`: layers 0 to 18 and the embedding on the M5 (MPS), layers 18 to 28 on gtx (CUDA, budget 5.5 GB), the head and final norm back on the M5 (port 50071).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts; gtx planned 4860 MB against the 5500 MB budget | PLANS.md | Held. Accepted without edits. |
| gtx memory | loads; peak allocated about 4.8 to 5.0 GB, below the free VRAM (about 7.0 GB) | 10 layers of 466 MB plus KV and workspace | Held. Loaded 4.661 GB, peak allocated 4.789 GB, VRAM free at ready 2.68 GB. |
| Run outcome | completes | whole layers per device, as H1 on 1.5B | Held. Completed, first attempt, 0 reconnects. |
| Frame formula | holds: 3 frames per decode step (hops 2, head 1) | s0 to s1, s1 to head, head result back | Held: 3 measured, 3 predicted (279 decode steps, 14400 bytes per step). |
| Identical vs MPS reference | 7 or 8 of 8 | 18 of 28 layers and the head are the same MPS computation; 10 layers run on a Pascal card | Held: 8 of 8. |
| Identical vs CPU reference | 7 or 8 of 8 | the MPS and CPU 7B references do not agree everywhere | Held: 8 of 8. |
| Criterion | met against at least one | | Held. Met against both. Accuracy 1.0. |
| First step logit diff | below 1.0 against both, nonzero | | Held: 0.312 MPS, 0.375 CPU. Long prompt first divergence token 72 and 109. |

## H4_pp3_qwen2.5-7b-bf16_homewifi_gtx_r1

Placement `H4_pp3_7b_gtx.yaml`: layers 0 to 10 and the embedding on the M5, 10 to 19 on the Air (torch 2.15.0), 19 to 28 and the head on gtx (budget 6.0 GB).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts; gtx planned 5476.5 MB against 6000 MB, air 4385.6 MB, m5 5950.1 MB | PLANS.md | Held. Accepted without edits. |
| gtx memory | loads; this is the tightest rank of the session, peak allocated about 5.4 to 5.6 GB of about 7.0 GB free | 9 layers plus the 1.09 GB head | Held. Loaded 5.285 GB, peak allocated 5.404 GB. VRAM free at ready 1.41 GB, less than the 1.6 GB the prediction implied. |
| Run outcome | completes, no rank disconnect | | Held. Completed, first attempt, 0 reconnects. |
| Frame formula | holds: 3 frames per decode step (2 hops and the head) | same shape as H4 on 1.5B | Held: 3 measured, 3 predicted (279 decode steps, 14400 bytes per step). |
| Identical vs MPS reference | 7 or 8 of 8 | three devices, the Air differs from the M5 at the rounding edge | Held: 8 of 8. |
| Identical vs CPU reference | 7 or 8 of 8 | | Held: 8 of 8. |
| Criterion | met against at least one | | Held. Met against both. Accuracy 1.0. First step logit diff 0.406 MPS, 0.500 CPU. |

## H5_tp3_qwen2.5-7b-bf16_homewifi_gtx_r1

Placement `H5_tp3_7b_gtx.yaml`: one TP group over all 28 layers, KV aligned heads 14 (M5), 7 (Air), 7 (gtx, budget 5.5 GB), `allow_mixed_devices: true`.

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts; m5 7798.6 MB, air 3906.8 MB, gtx 3906.8 MB | PLANS.md | Held. Accepted without edits. |
| Run outcome | completes, no rank disconnect | H5 on 1.5B held 16416 all-reduces per rank | Held. Completed, first attempt. 16416 all-reduces per rank, 0 reconnects. |
| Frame formula | holds: 346 frames per decode step, about 2.46 MB per step | the `plan` line | Held: 346 measured, 346 predicted (279 decode steps, 2459752 bytes per step). |
| Identical vs MPS reference | 7 or 8 of 8 | fp32 sums on the host in a fixed order; uneven shares were checked on two Macs only (session 6) | Held: 8 of 8. |
| Identical vs CPU reference | 7 or 8 of 8 | | Held: 8 of 8. |
| Criterion | met against at least one | | Held. Met against both. Accuracy 1.0. First step logit diff 0.438 MPS, 0.406 CPU. |
| What would count as a failure of the reasoning | garbled text from the first token, a rank disconnect, or CUDA out of memory on gtx | | Not triggered. gtx peak allocated 3.882 GB, VRAM free at ready 3.45 GB. |

