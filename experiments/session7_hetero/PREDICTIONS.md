# Session 7a: pre-registered predictions

Written 2026-11-16 on the M5, before the L1 run (which Ohad runs by hand) and before any run of the mixed device option. Time per token is not predicted (T1).

## L1_tp_mixed_0.5b_loopback_r1

Placement: one TP group, rank s0/0 on MPS and rank s0/1 on CPU, both on the M5 over loopback, 0.5B bf16, `allow_mixed_devices: true`.
References: `R0_qwen2.5-0.5b-bf16_mps_r2` and `R0_qwen2.5-0.5b-bf16_cpu_r1` (session 6 folder).

| item | prediction | why | outcome |
|---|---|---|---|
| Planner | accepts, 7 heads and 2432 MLP rows per rank | equal shares, 14 heads |  |
| Frame formula | holds: 100 frames per decode step (token 1, embedding 2, layers 96, head 1) | same group size and layer count as the formula; device type does not change the frame count |  |
| Run outcome | completes | the all-reduce goes through the host in fp32 for both device types |  |
| Identical vs MPS reference | 7 or 8 of 8, most likely 8 | half of the heads run on MPS; the other half differ by kernel rounding only |  |
| Identical vs CPU reference | 7 or 8 of 8, most likely 7 | the two references already differ from each other on the long prompt (token 10) |  |
| Criterion (at least 7 of 8 identical, same graded accuracy) | met against both, confidence about 65 percent | accuracy is a coarse grade; the risk is one extra flip in a category with a threshold |  |
| First step logit diff | nonzero, same order as MPS against CPU in P9 (about 1e-1 in bf16 logits) | different kernels per rank |  |
| What would count as a failure of the reasoning | fewer than 7 identical against both references | would mean the mixed sum is not the same on both ranks, not only kernel rounding |  |
