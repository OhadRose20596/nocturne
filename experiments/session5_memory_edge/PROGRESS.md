# Session 5 progress

Updated 2026-11-14 15:17:44. Steps known so far: 59, finished: 53. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G1r | qwen2.5-32b-q4km ctx 512 PP default | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c512_r1 A1_wifi_qwen2.5-32b-q4km_m5_c512_r2 A1_wifi_qwen2.5-32b-q4km_m5_c512_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 512 PP balanced | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP default | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 A1_wifi_qwen2.5-32b-q4km_m5_c4096_r2 A1_wifi_qwen2.5-32b-q4km_m5_c4096_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP balanced | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP default | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c16384_r1 A1_wifi_qwen2.5-32b-q4km_m5_c16384_r2 A1_wifi_qwen2.5-32b-q4km_m5_c16384_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP balanced | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP default | done earlier: ok, ok, ok | A1_wifi_qwen2.5-32b-q4km_m5_c32768_r1 A1_wifi_qwen2.5-32b-q4km_m5_c32768_r2 A1_wifi_qwen2.5-32b-q4km_m5_c32768_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP balanced | done earlier: oom_worker_load, oom_worker_load, oom_worker_load | A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r3 |  |  |
| G1r | qwen2.5-32b-q4km ctx 4096 TP | done earlier: ok, ok | B1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 B1_wifi_qwen2.5-32b-q4km_m5_c4096_r2 |  |  |
| G1r | qwen2.5-32b-q4km ctx 32768 TP 3:2 | done earlier: oom, ok | B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1 B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r2 |  |  |
| G4 | qwen2.5-14b-q4km ctx 4096 M5 alone | done earlier: ok | BASELINE_local_qwen2.5-14b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 4096 Air alone | done earlier: ok | BASELINE_local_qwen2.5-14b-q4km_air_c4096_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 4096 PP default | done earlier: ok | A1_wifi_qwen2.5-14b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 4096 TP | done earlier: ok | B1_wifi_qwen2.5-14b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 32768 M5 alone | done earlier: ok | BASELINE_local_qwen2.5-14b-q4km_m5_c32768_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 32768 Air alone | done earlier: oom_requester_compute | BASELINE_local_qwen2.5-14b-q4km_air_c32768_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 32768 PP default | done earlier: ok | A1_wifi_qwen2.5-14b-q4km_m5_c32768_r1 |  |  |
| G4 | qwen2.5-14b-q4km ctx 32768 TP | done earlier: ok | B1_wifi_qwen2.5-14b-q4km_m5_c32768_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 M5 alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 Air alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_air_c4096_r2 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 PP default | done earlier: ok | A1_wifi_qwen2.5-7b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 TP | done earlier: ok | B1_wifi_qwen2.5-7b-q4km_m5_c4096_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 32768 M5 alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_m5_c32768_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 32768 Air alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_air_c32768_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 32768 PP default | done earlier: ok | A1_wifi_qwen2.5-7b-q4km_m5_c32768_r1 |  |  |
| G4 | qwen2.5-7b-q4km ctx 32768 TP | done earlier: ok | B1_wifi_qwen2.5-7b-q4km_m5_c32768_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_long | pending |  |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_kv4k | pending |  |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_long | pending |  |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_kv4k | pending |  |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_long | pending |  |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_kv4k | pending |  |  |  |

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
[2026-11-14 15:17:44] commit G1r: ok
[2026-11-14 15:17:44] == G4: 14B and 7B Q4_K_M, every partition, ctx 4096 and 32768 (16 steps)
[2026-11-14 15:17:44] commit G4: ok
[2026-11-14 15:17:44] == G3: real long prompts at the edge of each partition (6 steps)
```
