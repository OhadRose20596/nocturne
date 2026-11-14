# Session 5 progress

Updated 2026-11-14 12:06:31. Steps known so far: 37, finished: 37. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G1r | qwen2.5-32b-q4km ctx 512 PP default | ok | A1_wifi_qwen2.5-32b-q4km_m5_c512_r1 A1_wifi_qwen2.5-32b-q4km_m5_c512_r2 A1_wifi_qwen2.5-32b-q4km_m5_c512_r3 | 2026-11-14 11:09:58 | 2026-11-14 11:11:15 |
| G1r | qwen2.5-32b-q4km ctx 512 PP balanced | ok | A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r3 | 2026-11-14 11:13:59 | 2026-11-14 11:15:32 |
| G1r | qwen2.5-32b-q4km ctx 4096 PP default | ok | A1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 A1_wifi_qwen2.5-32b-q4km_m5_c4096_r2 A1_wifi_qwen2.5-32b-q4km_m5_c4096_r3 | 2026-11-14 11:17:31 | 2026-11-14 11:18:53 |
| G1r | qwen2.5-32b-q4km ctx 4096 PP balanced | ok | A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r3 | 2026-11-14 11:20:55 | 2026-11-14 11:22:20 |
| G1r | qwen2.5-32b-q4km ctx 16384 PP default | ok | A1_wifi_qwen2.5-32b-q4km_m5_c16384_r1 A1_wifi_qwen2.5-32b-q4km_m5_c16384_r2 A1_wifi_qwen2.5-32b-q4km_m5_c16384_r3 | 2026-11-14 11:24:27 | 2026-11-14 11:25:44 |
| G1r | qwen2.5-32b-q4km ctx 16384 PP balanced | ok | A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r3 | 2026-11-14 11:27:46 | 2026-11-14 11:29:07 |
| G1r | qwen2.5-32b-q4km ctx 32768 PP default | ok | A1_wifi_qwen2.5-32b-q4km_m5_c32768_r1 A1_wifi_qwen2.5-32b-q4km_m5_c32768_r2 A1_wifi_qwen2.5-32b-q4km_m5_c32768_r3 | 2026-11-14 11:31:01 | 2026-11-14 11:32:18 |
| G1r | qwen2.5-32b-q4km ctx 32768 PP balanced | fail (oom_worker_load) | A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r1 A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r2 A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r3 | 2026-11-14 11:34:05 | 2026-11-14 11:35:07 |
| G1r | qwen2.5-32b-q4km ctx 4096 TP | ok | B1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 B1_wifi_qwen2.5-32b-q4km_m5_c4096_r2 | 2026-11-14 11:35:45 | 2026-11-14 11:51:14 |
| G1r | qwen2.5-32b-q4km ctx 32768 TP 3:2 | ok | B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1 B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r2 | 2026-11-14 11:51:56 | 2026-11-14 12:05:47 |

## Log (last 40 lines)

```
[2026-11-14 11:05:32]   [m5] model sha 2e5f6daea180, free 486.0 GB
[2026-11-14 11:05:32]   [air] model sha 2e5f6daea180, free 72.0 GB
[2026-11-14 11:05:32]   vpn_state disconnected, awdl up
[2026-11-14 11:05:32]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 11:05:32] run qwen2.5-32b-q4km ctx 512 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:09:58]   A1_wifi_qwen2.5-32b-q4km_m5_c512_r2: OK, quality unknown  decode 5.784037707260447  cmds/step 7.0  cmds/tok 8.3125  B/tok 118797.90625  acc 1.0 
[2026-11-14 11:09:58] run qwen2.5-32b-q4km ctx 512 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:11:15]   A1_wifi_qwen2.5-32b-q4km_m5_c512_r3: OK, quality unknown  decode 6.086560783035343  cmds/step 7.0  cmds/tok 8.3125  B/tok 118491.15625  acc 1.0 
[2026-11-14 11:11:55] run qwen2.5-32b-q4km ctx 512 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:13:59]   A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r2: OK, quality unknown  decode 5.584003844710109  cmds/step 7.0  cmds/tok 8.3125  B/tok 137476.46875  acc 1.0 
[2026-11-14 11:13:59] run qwen2.5-32b-q4km ctx 512 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:15:32]   A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r3: OK, quality unknown  decode 5.415121049452683  cmds/step 7.0  cmds/tok 8.3125  B/tok 137284.28125  acc 1.0 
[2026-11-14 11:16:10] run qwen2.5-32b-q4km ctx 4096 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:17:31]   A1_wifi_qwen2.5-32b-q4km_m5_c4096_r2: OK, quality unknown  decode 5.343947234073295  cmds/step 7.0  cmds/tok 8.3125  B/tok 105375.4375  acc 1.0 
[2026-11-14 11:17:31] run qwen2.5-32b-q4km ctx 4096 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:18:53]   A1_wifi_qwen2.5-32b-q4km_m5_c4096_r3: OK, quality unknown  decode 5.47314736830819  cmds/step 7.0  cmds/tok 8.3125  B/tok 118451.53125  acc 1.0 
[2026-11-14 11:19:31] run qwen2.5-32b-q4km ctx 4096 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:20:54]   A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r2: OK, quality unknown  decode 5.552055864665677  cmds/step 7.0  cmds/tok 8.3125  B/tok 138398.84375  acc 1.0 
[2026-11-14 11:20:55] run qwen2.5-32b-q4km ctx 4096 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:22:20]   A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r3: OK, quality unknown  decode 5.567285637031572  cmds/step 7.0  cmds/tok 8.3125  B/tok 136942.0  acc 1.0 
[2026-11-14 11:23:03] run qwen2.5-32b-q4km ctx 16384 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:24:27]   A1_wifi_qwen2.5-32b-q4km_m5_c16384_r2: OK, quality unknown  decode 5.515784700109913  cmds/step 7.0  cmds/tok 8.3125  B/tok 118563.5625  acc 1.0 
[2026-11-14 11:24:27] run qwen2.5-32b-q4km ctx 16384 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:25:44]   A1_wifi_qwen2.5-32b-q4km_m5_c16384_r3: OK, quality unknown  decode 5.281253406420705  cmds/step 7.0  cmds/tok 8.3125  B/tok 105137.53125  acc 1.0 
[2026-11-14 11:26:23] run qwen2.5-32b-q4km ctx 16384 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:27:45]   A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r2: OK, quality unknown  decode 5.667345702705356  cmds/step 7.0  cmds/tok 8.3125  B/tok 137512.9375  acc 1.0 
[2026-11-14 11:27:46] run qwen2.5-32b-q4km ctx 16384 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:29:07]   A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r3: OK, quality unknown  decode 5.528212941626631  cmds/step 7.0  cmds/tok 8.3125  B/tok 137558.25  acc 1.0 
[2026-11-14 11:29:45] run qwen2.5-32b-q4km ctx 32768 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:31:01]   A1_wifi_qwen2.5-32b-q4km_m5_c32768_r2: OK, quality unknown  decode 5.456562470570866  cmds/step 7.0  cmds/tok 8.3125  B/tok 118994.5  acc 1.0 
[2026-11-14 11:31:01] run qwen2.5-32b-q4km ctx 32768 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:32:18]   A1_wifi_qwen2.5-32b-q4km_m5_c32768_r3: OK, quality unknown  decode 5.638838348957358  cmds/step 7.0  cmds/tok 8.3125  B/tok 117956.21875  acc 1.0 
[2026-11-14 11:32:57] run qwen2.5-32b-q4km ctx 32768 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:34:04]   A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r2: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 11:34:05] run qwen2.5-32b-q4km ctx 32768 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 11:35:07]   A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r3: FAIL (oom_worker_load), quality unknown  decode None  cmds/step 5.0  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 11:35:45] run qwen2.5-32b-q4km ctx 4096 TP (G1r): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 11:51:14]   B1_wifi_qwen2.5-32b-q4km_m5_c4096_r2: OK, quality unknown  decode 0.2555365456407366  cmds/step 1.5  cmds/tok 457.78125  B/tok 65806272.90625  acc 1.0 
[2026-11-14 11:51:56] run qwen2.5-32b-q4km ctx 32768 TP 3:2 (G1r): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 12:05:47]   B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r2: OK, quality unknown  decode 0.2954926445736928  cmds/step 1.5  cmds/tok 457.78125  B/tok 65744991.90625  acc 1.0 
```
