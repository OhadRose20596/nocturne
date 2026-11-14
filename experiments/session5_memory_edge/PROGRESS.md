# Session 5 progress

Updated 2026-11-14 16:00:15. Steps known so far: 62, finished: 62. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_long | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096long_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_kv4k | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096kv4k_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_long | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096proplong_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_kv4k | done earlier: ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096propkv4k_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_long | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384proplong_r1 |  |  |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_kv4k | done earlier: ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384propkv4k_r1 |  |  |
| G2c | qwen2.5-32b-q6k ctx 16384 PP 3:2 | fail (oom_requester_compute) | A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 | 2026-11-14 15:25:38 | 2026-11-14 15:27:04 |
| G2c | qwen2.5-32b-q6k ctx 16384 TP 3:2 | ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2 B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3 | 2026-11-14 15:43:51 | 2026-11-14 15:56:50 |
| G2c | qwen2.5-32b-q6k ctx 16384 PP default | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1 A1_wifi_qwen2.5-32b-q6k_m5_c16384_r2 | 2026-11-14 15:57:52 | 2026-11-14 15:59:16 |

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
[2026-11-14 15:17:44] commit G3: ok
[2026-11-14 15:17:44] == G2c: Q-C candidate: 32B Q6_K ctx 16384, TP 3:2 completed where both PP splits failed; repeats (3 steps)
[2026-11-14 15:17:44] preflight qwen2.5-32b-q6k
[2026-11-14 15:19:33]   [m5] model sha 3393e05b1239, free 485.0 GB
[2026-11-14 15:19:33]   [air] model sha 3393e05b1239, free 63.0 GB
[2026-11-14 15:19:33]   vpn_state disconnected, awdl up
[2026-11-14 15:19:33]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 15:19:33] run qwen2.5-32b-q6k ctx 16384 PP 3:2 (G2c): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 15:25:37]   A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2: FAIL (oom_requester_compute), quality unknown  decode None  cmds/step 11  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 15:25:38] run qwen2.5-32b-q6k ctx 16384 PP 3:2 (G2c): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 15:27:04]   A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3: FAIL (oom_requester_compute), quality unknown  decode None  cmds/step 11  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 15:27:59] run qwen2.5-32b-q6k ctx 16384 TP 3:2 (G2c): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 15:43:51]   B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r2: OK, quality unknown  decode 0.2811667920865608  cmds/step 1.5  cmds/tok 457.78125  B/tok 65750137.21875  acc 1.0 
[2026-11-14 15:43:51] run qwen2.5-32b-q6k ctx 16384 TP 3:2 (G2c): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 15:56:50]   B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r3: OK, quality unknown  decode 0.28014895569505893  cmds/step 1.5  cmds/tok 457.78125  B/tok 65751237.84375  acc 1.0 
[2026-11-14 15:57:52] run qwen2.5-32b-q6k ctx 16384 PP default (G2c): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 15:59:16]   A1_wifi_qwen2.5-32b-q6k_m5_c16384_r2: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
```
