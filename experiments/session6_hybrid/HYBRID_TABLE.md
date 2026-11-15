# Session 6 hybrid table

Built by `python -m nocturne.engines.torch_tp.derive_hybrid` from `runs/*/record.json`. One row per run.
Correctness is against the first R0 reference listed in the record. ms per step is background only (T1).

| run_id | placement | model | reduce | outcome | ranks | hosts | reference | identical | accuracy | ref_accuracy | criterion | logit_maxdiff | long_first_div | frames_predicted | frames_measured | formula | bytes_per_step | max_rank_planned_MB | max_rank_footprint_MB | ms_per_step_background | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_2mac_qwen2.5-7b-bf16_r1 | P1_2mac | qwen2.5-7b-bf16 | float32 | completed | 2 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 105 | 116 | 116.0 | holds | 819932 | 9448 | 10365 | 843 |  |
| P1_2mac_qwen2.5-7b-bf16_r2 | P1_2mac | qwen2.5-7b-bf16 | float32 | completed | 2 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 105 | 116 | 116.0 | holds | 819932 | 9448 | 10365 | 769 |  |
| P1_2mac_qwen2.5-7b-bf16_r3 | P1_2mac | qwen2.5-7b-bf16 | float32 | completed | 2 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 105 | 116 | 116.0 | holds | 819932 | 9448 | 10364 | 751 |  |
| P1_loopback_bf16ctl_qwen2.5-7b-bf16_r1 | P1_loopback | qwen2.5-7b-bf16 | bfloat16 | completed | 2 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.5 | 49 | 116 | 116.0 | holds | 819932 | 9448 | 10259 | 146 |  |
| P1_loopback_qwen2.5-7b-bf16_r1 | P1_loopback | qwen2.5-7b-bf16 | float32 | completed | 2 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.5 | 49 | 116 | 116.0 | holds | 819932 | 9448 | 10258 | 147 |  |
| P2_2mac_qwen2.5-7b-bf16_r1 | P2_2mac | qwen2.5-7b-bf16 | float32 | completed | 3 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.5 | 49 | 346 | 346.0 | holds | 2459752 | 5272 | 5799 | 791 |  |
| P2_loopback_qwen2.5-7b-bf16_r1 | P2_loopback | qwen2.5-7b-bf16 | float32 | completed | 3 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 72 | 346 | 346.0 | holds | 2459752 | 5272 | 5732 | 147 |  |
| P3_2mac_qwen2.5-1.5b-bf16_r1 | P3_2mac | qwen2.5-1.5b-bf16 | float32 | completed | 3 | air+m5 | R0_qwen2.5-1.5b-bf16_mps_r1 | 8/8 | 1.0 | 1.0 | met | 0.5859 | 97 | 346 | 346.0 | holds | 1058920 | 1187 | 1533 | 695 |  |
| P3_loopback_qwen2.5-1.5b-bf16_r1 | P3_loopback | qwen2.5-1.5b-bf16 | float32 | completed | 3 | m5 | R0_qwen2.5-1.5b-bf16_mps_r1 | 8/8 | 1.0 | 1.0 | met | 0.5625 | 97 | 346 | 346.0 | holds | 1058920 | 1187 | 1535 | 67 |  |
| P4_loopback_qwen2.5-7b-bf16_r1 | P4_loopback | qwen2.5-7b-bf16 | float32 | completed | 5 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.5 | 127 | 1148 | 1148.0 | holds | 8199056 | 3258 | 3742 | 169 |  |
| P5_2mac_qwen2.5-7b-bf16_r1 | P5_2mac | qwen2.5-7b-bf16 | float32 | completed | 2 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 72 | 2 | 2.0 | holds | 7208 | 8798 | 9934 | 201 |  |
| P5_loopback_qwen2.5-7b-bf16_r1 | P5_loopback | qwen2.5-7b-bf16 | float32 | completed | 2 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0 | 256 | 2 | 2.0 | holds | 7208 | 8798 | 9928 | 134 |  |
| P6_2mac_qwen2.5-7b-bf16_r1 | P6_2mac | qwen2.5-7b-bf16 | float32 | completed | 5 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 29 | 29.0 | holds | 201392 | 7849 | 8859 | 243 |  |
| P6_loopback_qwen2.5-7b-bf16_r1 | P6_loopback | qwen2.5-7b-bf16 | float32 | completed | 5 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 29 | 29.0 | holds | 201392 | 7849 | 8851 | 3940 |  |
| P7_2mac_mpscap_qwen2.5-7b-bf16_r1 | P7_2mac | qwen2.5-7b-bf16 | float32 | rank_failed | 8 | air+m5 |  |  |  |  |  |  |  | 34 |  |  |  | 6899 |  |  | s1/0: RuntimeError: MPS backend out of memory (MPS allocated: 1.01 GiB, other allocations: 464.00 KiB, max allowed: 381.47 MiB). Tried to allocate 43.00 MiB on  |
| P7_2mac_qwen2.5-7b-bf16_r1 | P7_2mac | qwen2.5-7b-bf16 | float32 | completed | 8 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 34 | 34.0 | holds | 223000 | 6899 | 7920 | 251 |  |
| P7_2mac_qwen2.5-7b-bf16_r2 | P7_2mac | qwen2.5-7b-bf16 | float32 | completed | 8 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 34 | 34.0 | holds | 223000 | 6899 | 7920 | 225 |  |
| P7_2mac_qwen2.5-7b-bf16_r3 | P7_2mac | qwen2.5-7b-bf16 | float32 | completed | 8 | air+m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 34 | 34.0 | holds | 223000 | 6899 | 7920 | 245 |  |
| P7_loopback_qwen2.5-7b-bf16_r1 | P7_loopback | qwen2.5-7b-bf16 | float32 | completed | 8 | m5 | R0_qwen2.5-7b-bf16_mps_r3 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 34 | 34.0 | holds | 223000 | 6899 | 7854 | 10250 |  |
| P7ctl_2mac_forced_qwen2.5-7b-bf16_r1 | P7ctl_2mac | qwen2.5-7b-bf16 | float32 | rank_failed | 5 | air+m5 |  |  |  |  |  |  |  | 5 |  |  |  | 6899 |  |  | s1a/0: BudgetExceeded: load: model.layers.12.mlp.up_proj.weight (135.8 MB) would bring tensor bytes to 0.407 GB, over the budget of 0.400 GB; s1b/0: BudgetExcee |
| P7ctl_2mac_qwen2.5-7b-bf16_r1 | P7ctl_2mac |  |  | plan_rejected | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | over budget: s1a/0 on m5: planned 0.590 GB (weights 0.466, KV 0.008, workspace 0.115) > budget 0.400 GB; s1b/0 on m5: planned 0.590 GB (weights 0.466, KV 0.008, |
| P8_local_qwen2.5-0.5b-bf16_r1 | P8_local | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.7441 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1449 | 249 |  |
| P8_loopback_qwen2.5-0.5b-bf16_r1 | P8_loopback | qwen2.5-0.5b-bf16 | float32 | completed | 16 | m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.8906 | 60 | 115 | 115.0 | holds | 203452 | 202 | 1447 | 43 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r1 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1449 | 477 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r2 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1450 | 484 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r3 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1450 | 556 |  |
| P9_2mac_qwen2.5-7b-bf16_r1 | P9_2mac | qwen2.5-7b-bf16 | float32 | completed | 8 | air+m5 | R0_qwen2.5-7b-bf16_cpu_r1 | 8/8 | 1.0 | 1.0 | met | 0.375 | 49 | 34 | 34.0 | holds | 223000 | 6899 | 7920 | 248 |  |
| R0_qwen2.5-0.5b-bf16_cpu_r1 | R0_0.5b_cpu | qwen2.5-0.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 326 | 17 |  |
| R0_qwen2.5-0.5b-bf16_mps_r1 | R0_0.5b_mps | qwen2.5-0.5b-bf16 | float32 | error | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 1501 |  | KeyError: 'links' |
| R0_qwen2.5-0.5b-bf16_mps_r2 | R0_0.5b_mps | qwen2.5-0.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 1499 | 14 |  |
| R0_qwen2.5-1.5b-bf16_cpu_r1 | R0_1.5b_cpu | qwen2.5-1.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 3256 | 423 | 44 |  |
| R0_qwen2.5-1.5b-bf16_mps_r1 | R0_1.5b_mps | qwen2.5-1.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 3256 | 3632 | 37 |  |
| R0_qwen2.5-7b-bf16_cpu_r1 | R0_7b_cpu | qwen2.5-7b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 15582 | 577 | 183 |  |
| R0_qwen2.5-7b-bf16_mps_r1 | R0_7b_mps | qwen2.5-7b-bf16 | float32 | aborted | 1 | m5 |  |  |  |  |  |  |  | 0 |  |  |  | 15582 |  |  | Stopped by hand at 19:08 on 2026-11-15. The M5 was swapping (17.2 of 18.4 GB swap used, about 5 GB swapped in per 30 s);
the first prompt took 81 s for 4 tokens |
| R0_qwen2.5-7b-bf16_mps_r2 | R0_7b_mps | qwen2.5-7b-bf16 | float32 | aborted | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 15582 | 16558 | 134 | Stopped by hand at 19:39 on 2026-11-15 during the transformers check: the driver kept the engine's 15 GB of weights in the same process while transformers loade |
| R0_qwen2.5-7b-bf16_mps_r3 | R0_7b_mps | qwen2.5-7b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 15582 | 16555 | 136 |  |
