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
