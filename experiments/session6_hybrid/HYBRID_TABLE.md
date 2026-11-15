# Session 6 hybrid table

Built by `python -m nocturne.engines.torch_tp.derive_hybrid` from `runs/*/record.json`. One row per run.
Correctness is against the first R0 reference listed in the record. ms per step is background only (T1).

| run_id | placement | model | reduce | outcome | ranks | hosts | reference | identical | accuracy | ref_accuracy | criterion | logit_maxdiff | long_first_div | frames_predicted | frames_measured | formula | bytes_per_step | max_rank_planned_MB | max_rank_footprint_MB | ms_per_step_background | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P3_2mac_qwen2.5-1.5b-bf16_r1 | P3_2mac | qwen2.5-1.5b-bf16 | float32 | completed | 3 | air+m5 | R0_qwen2.5-1.5b-bf16_mps_r1 | 8/8 | 1.0 | 1.0 | met | 0.5859 | 97 | 346 | 346.0 | holds | 1058920 | 1187 | 1533 | 695 |  |
| P3_loopback_qwen2.5-1.5b-bf16_r1 | P3_loopback | qwen2.5-1.5b-bf16 | float32 | completed | 3 | m5 | R0_qwen2.5-1.5b-bf16_mps_r1 | 8/8 | 1.0 | 1.0 | met | 0.5625 | 97 | 346 | 346.0 | holds | 1058920 | 1187 | 1535 | 67 |  |
| P8_local_qwen2.5-0.5b-bf16_r1 | P8_local | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.7441 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1449 | 249 |  |
| P8_loopback_qwen2.5-0.5b-bf16_r1 | P8_loopback | qwen2.5-0.5b-bf16 | float32 | completed | 16 | m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.8906 | 60 | 115 | 115.0 | holds | 203452 | 202 | 1447 | 43 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r1 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1449 | 477 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r2 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1450 | 484 |  |
| P8_wifi_qwen2.5-0.5b-bf16_r3 | P8_wifi | qwen2.5-0.5b-bf16 | float32 | completed | 16 | air+m5 | R0_qwen2.5-0.5b-bf16_mps_r2 | 8/8 | 0.875 | 0.875 | met | 0.9297 | 10 | 115 | 115.0 | holds | 203452 | 202 | 1450 | 556 |  |
| R0_qwen2.5-0.5b-bf16_cpu_r1 | R0_0.5b_cpu | qwen2.5-0.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 326 | 17 |  |
| R0_qwen2.5-0.5b-bf16_mps_r1 | R0_0.5b_mps | qwen2.5-0.5b-bf16 | float32 | error | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 1501 |  | KeyError: 'links' |
| R0_qwen2.5-0.5b-bf16_mps_r2 | R0_0.5b_mps | qwen2.5-0.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 1070 | 1499 | 14 |  |
| R0_qwen2.5-1.5b-bf16_cpu_r1 | R0_1.5b_cpu | qwen2.5-1.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 3256 | 423 | 44 |  |
| R0_qwen2.5-1.5b-bf16_mps_r1 | R0_1.5b_mps | qwen2.5-1.5b-bf16 | float32 | completed | 1 | m5 |  |  |  |  |  |  |  | 0 | 0.0 | holds |  | 3256 | 3632 | 37 |  |
