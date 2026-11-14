# Session 5 progress

Updated 2026-11-14 10:12:15. Steps known so far: 19, finished: 19. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G2a | qwen2.5-32b-q8 ctx 4096 PP default | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q8_m5_c4096_r2 | 2026-11-14 09:19:59 | 2026-11-14 09:28:11 |
| G2a | qwen2.5-32b-q8 ctx 4096 PP 3:2 | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q8_m5_c4096prop_r1 | 2026-11-14 09:28:36 | 2026-11-14 09:30:02 |
| G2a | qwen2.5-32b-q8 ctx 4096 PP balanced | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q8_m5_c4096bal_r2 | 2026-11-14 09:30:26 | 2026-11-14 09:33:37 |
| G2a | qwen2.5-32b-q8 ctx 4096 TP | fail (oom_worker_load) | B1_wifi_qwen2.5-32b-q8_m5_c4096_r2 | 2026-11-14 09:34:02 | 2026-11-14 09:45:44 |
| G2a | qwen2.5-32b-q8 ctx 4096 TP 3:2 | fail (oom_requester_compute) | B1_wifi_qwen2.5-32b-q8_m5_c4096prop_r2 | 2026-11-14 09:46:12 | 2026-11-14 09:59:19 |
| G2edge | qwen2.5-32b-q8 ctx 512 PP default | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q8_m5_c512_r1 | 2026-11-14 09:59:55 | 2026-11-14 10:01:21 |
| G2edge | qwen2.5-32b-q8 ctx 512 PP 3:2 | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q8_m5_c512prop_r1 | 2026-11-14 10:01:51 | 2026-11-14 10:03:05 |
| G2edge | qwen2.5-32b-q8 ctx 512 TP 3:2 | fail (oom_requester_compute) | B1_wifi_qwen2.5-32b-q8_m5_c512prop_r1 | 2026-11-14 10:03:34 | 2026-11-14 10:11:39 |

## Log (last 40 lines)

```
[2026-11-14 09:18:47] == G0: layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2) (1 steps)
[2026-11-14 09:18:47] commit G0: ok
[2026-11-14 09:18:47] == G2a: 32B Q5_K_M, Q6_K, Q8_0 at ctx 4096, all partitions (15 steps)
[2026-11-14 09:18:51] Air free disk 11 GB: cleared the RPC tensor cache (63G /Users/ohadrosentswaig/Library/Caches/llama.cpp/rpc 74 GB free after)
[2026-11-14 09:18:51] preflight qwen2.5-32b-q8
[2026-11-14 09:19:59]   [m5] model sha fc5103335689, free 488.0 GB
[2026-11-14 09:19:59]   [air] model sha directory, free 74.0 GB
[2026-11-14 09:19:59]   vpn_state disconnected, awdl up
[2026-11-14 09:19:59]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 09:19:59] run qwen2.5-32b-q8 ctx 4096 PP default (G2a): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 09:28:11]   A1_wifi_qwen2.5-32b-q8_m5_c4096_r2: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 09:28:36] run qwen2.5-32b-q8 ctx 4096 PP 3:2 (G2a): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 09:30:02]   A1_wifi_qwen2.5-32b-q8_m5_c4096prop_r1: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 09:30:26] run qwen2.5-32b-q8 ctx 4096 PP balanced (G2a): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 09:33:37]   A1_wifi_qwen2.5-32b-q8_m5_c4096bal_r2: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 09:34:02] run qwen2.5-32b-q8 ctx 4096 TP (G2a): nocturne.bench.run --cell B1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 09:45:44]   B1_wifi_qwen2.5-32b-q8_m5_c4096_r2: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 2.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 09:46:12] run qwen2.5-32b-q8 ctx 4096 TP 3:2 (G2a): nocturne.bench.run --cell B1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 09:59:19]   B1_wifi_qwen2.5-32b-q8_m5_c4096prop_r2: FAIL (oom_requester_compute), quality unknown  decode None  cmds/step 4  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 09:59:54] commit G2a: ok
[2026-11-14 09:59:54] == G2edge: models that fit nowhere at 4096, checked at ctx 512 (3 steps)
[2026-11-14 09:59:55] run qwen2.5-32b-q8 ctx 512 PP default (G2edge): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 10:01:21]   A1_wifi_qwen2.5-32b-q8_m5_c512_r1: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 10:01:51] run qwen2.5-32b-q8 ctx 512 PP 3:2 (G2edge): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 10:03:05]   A1_wifi_qwen2.5-32b-q8_m5_c512prop_r1: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 10:03:34] run qwen2.5-32b-q8 ctx 512 TP 3:2 (G2edge): nocturne.bench.run --cell B1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 10:11:39]   B1_wifi_qwen2.5-32b-q8_m5_c512prop_r1: FAIL (oom_requester_compute), quality unknown  decode None  cmds/step 4  cmds/tok None  B/tok None  acc 0.0 
```
