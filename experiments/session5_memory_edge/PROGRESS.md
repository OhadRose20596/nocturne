# Session 5 progress

Updated 2026-11-14 16:11:50. Steps known so far: 65, finished: 65. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G2b | qwen2.5-32b-q6k ctx 16384 PP default | done earlier: oom_worker_load, oom_worker_load | A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384_r2 |  |  |
| G2b | qwen2.5-32b-q6k ctx 16384 PP 3:2 | done earlier: oom_requester_compute, oom_requester_compute, oom_requester_compute | A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 |  |  |
| G2b | qwen2.5-32b-q6k ctx 16384 TP 3:2 | done earlier: ok, ok, ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 |  |  |
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
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_long | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096long_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_kv4k | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096kv4k_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_long | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096proplong_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_kv4k | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096propkv4k_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_long | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384proplong_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_kv4k | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384propkv4k_r1 |  |  |
| G2c | qwen2.5-32b-q6k ctx 16384 PP 3:2 | done earlier: oom_requester_compute, oom_requester_compute, oom_requester_compute | A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 |  |  |
| G2c | qwen2.5-32b-q6k ctx 16384 TP 3:2 | done earlier: ok, ok, ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 |  |  |
| G2c | qwen2.5-32b-q6k ctx 16384 PP default | done earlier: oom_worker_load, oom_worker_load | A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384_r2 |  |  |
| G2d | qwen2.5-32b-q6k ctx 16384 PP 3:2 no-mmap | ok | A1_wifi_qwen2.5-32b-q6k_m5_c16384propnommap_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384propnommap_r2 | 2026-11-14 16:09:29 | 2026-11-14 16:10:49 |
| G5 | qwen2.5-1.5b-bf16 ctx 4096 TP torch | done earlier: ok | B1t_wifi_qwen2.5-1.5b-bf16_m5_c4096_r1 |  |  |
| G5 | qwen2.5-7b-bf16 ctx 4096 TP torch | done earlier: ok | B1t_wifi_qwen2.5-7b-bf16_m5_c4096_r1 |  |  |

## Log (last 40 lines)

```
[2026-11-14 16:06:14] == G0: layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2) (1 steps)
[2026-11-14 16:06:14] commit G0: ok
[2026-11-14 16:06:14] == G2a: 32B Q5_K_M, Q6_K, Q8_0 at ctx 4096, all partitions (15 steps)
[2026-11-14 16:06:14] commit G2a: ok
[2026-11-14 16:06:14] == G2edge: models that fit nowhere at 4096, checked at ctx 512 (3 steps)
[2026-11-14 16:06:14] commit G2edge: ok
[2026-11-14 16:06:14] largest model fitting at 4096: PP ['qwen2.5-32b-q6k'], TP ['qwen2.5-32b-q6k']; context sweep on ['qwen2.5-32b-q6k']
[2026-11-14 16:06:14] == G2b: context sweep of the largest fitting models
[2026-11-14 16:06:14] skip qwen2.5-32b-q6k ctx 16384 PP balanced: already fails at a smaller context
[2026-11-14 16:06:14] skip qwen2.5-32b-q6k ctx 32768 PP default: already fails at a smaller context
[2026-11-14 16:06:14] skip qwen2.5-32b-q6k ctx 32768 PP 3:2: already fails at a smaller context
[2026-11-14 16:06:14] skip qwen2.5-32b-q6k ctx 32768 PP balanced: already fails at a smaller context
[2026-11-14 16:06:14] commit G2b: On branch main
Your branch is ahead of 'origin/main' by 21 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git
[2026-11-14 16:06:14] == G1r: repeats of the 32B Q4_K_M cells (10 steps)
[2026-11-14 16:06:14] commit G1r: ok
[2026-11-14 16:06:14] == G4: 14B and 7B Q4_K_M, every partition, ctx 4096 and 32768 (16 steps)
[2026-11-14 16:06:14] commit G4: ok
[2026-11-14 16:06:14] == G3: real long prompts at the edge of each partition (6 steps)
[2026-11-14 16:06:14] commit G3: ok
[2026-11-14 16:06:14] == G2c: Q-C candidate: 32B Q6_K ctx 16384, TP 3:2 completed where both PP splits failed; repeats (3 steps)
[2026-11-14 16:06:14] commit G2c: ok
[2026-11-14 16:06:14] == G2d: Q-C mechanism: 32B Q6_K ctx 16384 PP 3:2 with the weights not memory-mapped (--load-mode none) (1 steps)
[2026-11-14 16:06:15] preflight qwen2.5-32b-q6k
[2026-11-14 16:08:04]   [m5] model sha 3393e05b1239, free 484.0 GB
[2026-11-14 16:08:04]   [air] model sha 3393e05b1239, free 47.0 GB
[2026-11-14 16:08:04]   vpn_state disconnected, awdl up
[2026-11-14 16:08:04]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 16:08:04] run qwen2.5-32b-q6k ctx 16384 PP 3:2 no-mmap (G2d): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384propnommap --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3 --override-flag=--load-mode=none
[2026-11-14 16:09:29]   A1_wifi_qwen2.5-32b-q6k_m5_c16384propnommap_r1: OK, quality unknown  decode 4.035424060871512  cmds/step 7.0  cmds/tok 8.3125  B/tok 104894.5625  acc 1.0 
[2026-11-14 16:09:29] run qwen2.5-32b-q6k ctx 16384 PP 3:2 no-mmap (G2d): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384propnommap --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3 --override-flag=--load-mode=none
[2026-11-14 16:10:49]   A1_wifi_qwen2.5-32b-q6k_m5_c16384propnommap_r2: OK, quality unknown  decode 4.285937958237989  cmds/step 7.0  cmds/tok 8.3125  B/tok 102892.15625  acc 1.0 
[2026-11-14 16:11:50] commit G2d: ok
[2026-11-14 16:11:50] == G5: torch_tp engine, bf16, ctx 4096 (2 steps)
[2026-11-14 16:11:50] commit G5: ok
[2026-11-14 16:11:50] all groups finished
```
