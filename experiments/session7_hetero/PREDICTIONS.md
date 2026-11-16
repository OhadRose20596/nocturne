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

