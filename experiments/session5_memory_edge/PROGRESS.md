# Session 5 progress

Updated 2026-11-14 11:04:07. Steps known so far: 27, finished: 27. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G2b | qwen2.5-32b-q6k ctx 512 PP default | ok | A1_wifi_qwen2.5-32b-q6k_m5_c512_r1 | 2026-11-14 10:14:04 | 2026-11-14 10:20:04 |
| G2b | qwen2.5-32b-q6k ctx 512 PP 3:2 | ok | A1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1 | 2026-11-14 10:20:35 | 2026-11-14 10:22:05 |
| G2b | qwen2.5-32b-q6k ctx 512 PP balanced | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q6k_m5_c512bal_r1 | 2026-11-14 10:22:36 | 2026-11-14 10:24:53 |
| G2b | qwen2.5-32b-q6k ctx 512 TP 3:2 | ok | B1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1 | 2026-11-14 10:25:24 | 2026-11-14 10:41:48 |
| G2b | qwen2.5-32b-q6k ctx 16384 PP default | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1 | 2026-11-14 10:42:23 | 2026-11-14 10:43:33 |
| G2b | qwen2.5-32b-q6k ctx 16384 PP 3:2 | fail (oom_requester_compute) | A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 | 2026-11-14 10:44:07 | 2026-11-14 10:45:25 |
| G2b | qwen2.5-32b-q6k ctx 16384 TP 3:2 | ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1 | 2026-11-14 10:45:59 | 2026-11-14 10:58:20 |
| G2b | qwen2.5-32b-q6k ctx 32768 TP 3:2 | fail (oom_worker_compute) | B1_wifi_qwen2.5-32b-q6k_m5_c32768prop_r1 | 2026-11-14 10:58:56 | 2026-11-14 11:03:30 |

## Log (last 40 lines)

```
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
[2026-11-14 10:12:15] commit G2edge: ok
[2026-11-14 10:12:15] largest model fitting at 4096: PP ['qwen2.5-32b-q6k'], TP ['qwen2.5-32b-q6k']; context sweep on ['qwen2.5-32b-q6k']
[2026-11-14 10:12:15] == G2b: context sweep of the largest fitting models
[2026-11-14 10:12:15] preflight qwen2.5-32b-q6k
[2026-11-14 10:14:03]   [m5] model sha 3393e05b1239, free 487.0 GB
[2026-11-14 10:14:03]   [air] model sha 3393e05b1239, free 41.0 GB
[2026-11-14 10:14:03]   vpn_state disconnected, awdl up
[2026-11-14 10:14:03]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 10:14:04] run qwen2.5-32b-q6k ctx 512 PP default (G2b): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 10:20:04]   A1_wifi_qwen2.5-32b-q6k_m5_c512_r1: OK, quality unknown  decode 4.566521047896057  cmds/step 7.0  cmds/tok 8.3125  B/tok 114180.0  acc 1.0 
[2026-11-14 10:20:35] run qwen2.5-32b-q6k ctx 512 PP 3:2 (G2b): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 10:22:05]   A1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1: OK, quality unknown  decode 4.464605770493309  cmds/step 7.0  cmds/tok 8.3125  B/tok 115474.46875  acc 1.0 
[2026-11-14 10:22:36] run qwen2.5-32b-q6k ctx 512 PP balanced (G2b): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 10:24:53]   A1_wifi_qwen2.5-32b-q6k_m5_c512bal_r1: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 10:25:24] run qwen2.5-32b-q6k ctx 512 TP 3:2 (G2b): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 10:41:48]   B1_wifi_qwen2.5-32b-q6k_m5_c512prop_r1: OK, quality unknown  decode 0.27390871277849704  cmds/step 1.5  cmds/tok 457.78125  B/tok 65754403.90625  acc 1.0 
[2026-11-14 10:42:23] skip qwen2.5-32b-q6k ctx 16384 PP balanced: already fails at a smaller context
[2026-11-14 10:42:23] run qwen2.5-32b-q6k ctx 16384 PP default (G2b): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 10:43:33]   A1_wifi_qwen2.5-32b-q6k_m5_c16384_r1: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 10:44:07] run qwen2.5-32b-q6k ctx 16384 PP 3:2 (G2b): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 10:45:25]   A1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1: FAIL (oom_requester_compute), quality unknown  decode None  cmds/step 11  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 10:45:59] run qwen2.5-32b-q6k ctx 16384 TP 3:2 (G2b): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 10:58:20]   B1_wifi_qwen2.5-32b-q6k_m5_c16384prop_r1: OK, quality unknown  decode 0.28941333805763547  cmds/step 1.5  cmds/tok 457.78125  B/tok 65750983.34375  acc 1.0 
[2026-11-14 10:58:56] skip qwen2.5-32b-q6k ctx 32768 PP default: already fails at a smaller context
[2026-11-14 10:58:56] skip qwen2.5-32b-q6k ctx 32768 PP 3:2: already fails at a smaller context
[2026-11-14 10:58:56] skip qwen2.5-32b-q6k ctx 32768 PP balanced: already fails at a smaller context
[2026-11-14 10:58:56] run qwen2.5-32b-q6k ctx 32768 TP 3:2 (G2b): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 11:03:30]   B1_wifi_qwen2.5-32b-q6k_m5_c32768prop_r1: FAIL (oom_worker_compute), quality unknown  decode None  cmds/step 105  cmds/tok None  B/tok None  acc 0.0 
```
