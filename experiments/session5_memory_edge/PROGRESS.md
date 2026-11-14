# Session 5 progress

Updated 2026-11-14 15:17:44. Steps known so far: 19, finished: 16. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

| group | step | status | records | start | end |
|---|---|---|---|---|---|
| G0 | qwen2.5-32b-q4km ctx 32768 PP 3:2 | done earlier: ok | A1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1 |  |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP default | done earlier: ok | A1_wifi_qwen2.5-32b-q5km_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP 3:2 | done earlier: ok | A1_wifi_qwen2.5-32b-q5km_m5_c4096prop_r1 |  |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP balanced | done earlier: ok | A1_wifi_qwen2.5-32b-q5km_m5_c4096bal_r2 |  |  |
| G2a | qwen2.5-32b-q5km ctx 4096 TP | done earlier: ok | B1_wifi_qwen2.5-32b-q5km_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q5km ctx 4096 TP 3:2 | done earlier: ok | B1_wifi_qwen2.5-32b-q5km_m5_c4096prop_r2 |  |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP default | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP 3:2 | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096prop_r1 |  |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP balanced | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q6k_m5_c4096bal_r2 |  |  |
| G2a | qwen2.5-32b-q6k ctx 4096 TP | done earlier: oom_worker_load | B1_wifi_qwen2.5-32b-q6k_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q6k ctx 4096 TP 3:2 | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c4096prop_r2 |  |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP default | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q8_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP 3:2 | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q8_m5_c4096prop_r1 |  |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP balanced | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q8_m5_c4096bal_r2 |  |  |
| G2a | qwen2.5-32b-q8 ctx 4096 TP | done earlier: oom_worker_load | B1_wifi_qwen2.5-32b-q8_m5_c4096_r2 |  |  |
| G2a | qwen2.5-32b-q8 ctx 4096 TP 3:2 | done earlier: oom_requester_compute | B1_wifi_qwen2.5-32b-q8_m5_c4096prop_r2 |  |  |
| G2edge | qwen2.5-32b-q8 ctx 512 PP default | pending |  |  |  |
| G2edge | qwen2.5-32b-q8 ctx 512 PP 3:2 | pending |  |  |  |
| G2edge | qwen2.5-32b-q8 ctx 512 TP 3:2 | pending |  |  |  |

## Log (last 40 lines)

```
[2026-11-14 15:17:44] == G0: layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2) (1 steps)
[2026-11-14 15:17:44] commit G0: ok
[2026-11-14 15:17:44] == G2a: 32B Q5_K_M, Q6_K, Q8_0 at ctx 4096, all partitions (15 steps)
[2026-11-14 15:17:44] commit G2a: ok
[2026-11-14 15:17:44] == G2edge: models that fit nowhere at 4096, checked at ctx 512 (3 steps)
```
