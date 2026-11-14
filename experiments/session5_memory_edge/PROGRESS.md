# Session 5 progress

Updated 2026-11-14 15:17:44. Steps known so far: 37, finished: 27. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G2edge | qwen2.5-32b-q8 ctx 512 PP default | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q8_m5_c512_r1 |  |  |
| G2edge | qwen2.5-32b-q8 ctx 512 PP 3:2 | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q8_m5_c512prop_r1 |  |  |
| G2edge | qwen2.5-32b-q8 ctx 512 TP 3:2 | done earlier: oom_requester_compute | B1_wifi_qwen2.5-32b-q8_m5_c512prop_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 512 PP default | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c512_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 512 PP 3:2 | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 512 PP balanced | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q6k_m5_c512bal_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 512 TP 3:2 | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 16384 PP default | done earlier: oom_worker_load | A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 16384 PP 3:2 | done earlier: oom_requester_compute | A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 16384 TP 3:2 | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 |  |  |
| G2b | qwen2.5-32b-q6k ctx 32768 TP 3:2 | done earlier: oom_worker_compute | B1_wifi_qwen2.5-32b-q6k_m5_c32768prop_r1 |  |  |
| G1r | qwen2.5-32b-q4km ctx 512 PP default | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 512 PP balanced | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP default | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP balanced | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP default | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP balanced | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP default | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP balanced | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 TP | pending |  |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 TP 3:2 | pending |  |  |  |

## Log (last 40 lines)

```
[2026-11-14 15:17:44] == G0: layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2) (1 steps)
[2026-11-14 15:17:44] commit G0: ok
[2026-11-14 15:17:44] == G2a: 32B Q5_K_M, Q6_K, Q8_0 at ctx 4096, all partitions (15 steps)
[2026-11-14 15:17:44] commit G2a: ok
[2026-11-14 15:17:44] == G2edge: models that fit nowhere at 4096, checked at ctx 512 (3 steps)
[2026-11-14 15:17:44] commit G2edge: ok
[2026-11-14 15:17:44] largest model fitting at 4096: PP ['qwen2.5-32b-q6k'], TP ['qwen2.5-32b-q6k']; context sweep on ['qwen2.5-32b-q6k']
[2026-11-14 15:17:44] == G2b: context sweep of the largest fitting models
[2026-11-14 15:17:44] skip qwen2.5-32b-q6k ctx 16384 PP balanced: already fails at a smaller context
[2026-11-14 15:17:44] skip qwen2.5-32b-q6k ctx 32768 PP default: already fails at a smaller context
[2026-11-14 15:17:44] skip qwen2.5-32b-q6k ctx 32768 PP 3:2: already fails at a smaller context
[2026-11-14 15:17:44] skip qwen2.5-32b-q6k ctx 32768 PP balanced: already fails at a smaller context
[2026-11-14 15:17:44] commit G2b: On branch main
Your branch is ahead of 'origin/main' by 13 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git
[2026-11-14 15:17:44] == G1r: repeats of the 32B Q4_K_M cells (10 steps)
```
