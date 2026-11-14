# Session 5 progress

Updated 2026-11-14 15:16:11. Steps known so far: 59, finished: 59. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

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
| G4 | qwen2.5-14b-q4km ctx 4096 M5 alone | ok | BASELINE_local_qwen2.5-14b-q4km_m5_c4096_r1 | 2026-11-14 12:07:10 | 2026-11-14 12:07:20 |
| G4 | qwen2.5-14b-q4km ctx 4096 Air alone | ok | BASELINE_local_qwen2.5-14b-q4km_air_c4096_r1 | 2026-11-14 12:08:05 | 2026-11-14 12:08:17 |
| G4 | qwen2.5-14b-q4km ctx 4096 PP default | ok | A1_wifi_qwen2.5-14b-q4km_m5_c4096_r1 | 2026-11-14 12:09:00 | 2026-11-14 12:11:16 |
| G4 | qwen2.5-14b-q4km ctx 4096 TP | ok | B1_wifi_qwen2.5-14b-q4km_m5_c4096_r1 | 2026-11-14 12:12:00 | 2026-11-14 12:21:11 |
| G4 | qwen2.5-14b-q4km ctx 32768 M5 alone | ok | BASELINE_local_qwen2.5-14b-q4km_m5_c32768_r1 | 2026-11-14 12:21:56 | 2026-11-14 12:22:04 |
| G4 | qwen2.5-14b-q4km ctx 32768 Air alone | fail (oom_requester_compute) | BASELINE_local_qwen2.5-14b-q4km_air_c32768_r1 | 2026-11-14 12:22:49 | 2026-11-14 12:23:02 |
| G4 | qwen2.5-14b-q4km ctx 32768 PP default | ok | A1_wifi_qwen2.5-14b-q4km_m5_c32768_r1 | 2026-11-14 12:23:47 | 2026-11-14 12:24:37 |
| G4 | qwen2.5-14b-q4km ctx 32768 TP | ok | B1_wifi_qwen2.5-14b-q4km_m5_c32768_r1 | 2026-11-14 12:25:22 | 2026-11-14 12:34:00 |
| G4 | qwen2.5-7b-q4km ctx 4096 M5 alone | ok | BASELINE_local_qwen2.5-7b-q4km_m5_c4096_r1 | 2026-11-14 12:35:07 | 2026-11-14 12:35:11 |
| G4 | qwen2.5-7b-q4km ctx 4096 Air alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_air_c4096_r2 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 PP default | ok | A1_wifi_qwen2.5-7b-q4km_m5_c4096_r1 | 2026-11-14 12:35:58 | 2026-11-14 12:37:17 |
| G4 | qwen2.5-7b-q4km ctx 4096 TP | ok | B1_wifi_qwen2.5-7b-q4km_m5_c4096_r1 | 2026-11-14 12:38:04 | 2026-11-14 12:43:20 |
| G4 | qwen2.5-7b-q4km ctx 32768 M5 alone | ok | BASELINE_local_qwen2.5-7b-q4km_m5_c32768_r1 | 2026-11-14 12:44:07 | 2026-11-14 12:44:11 |
| G4 | qwen2.5-7b-q4km ctx 32768 Air alone | ok | BASELINE_local_qwen2.5-7b-q4km_air_c32768_r1 | 2026-11-14 12:44:59 | 2026-11-14 12:45:08 |
| G4 | qwen2.5-7b-q4km ctx 32768 PP default | ok | A1_wifi_qwen2.5-7b-q4km_m5_c32768_r1 | 2026-11-14 12:45:54 | 2026-11-14 12:46:35 |
| G4 | qwen2.5-7b-q4km ctx 32768 TP | ok | B1_wifi_qwen2.5-7b-q4km_m5_c32768_r1 | 2026-11-14 12:47:22 | 2026-11-14 12:52:07 |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_long | ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096long_r1 | 2026-11-14 12:52:57 | 2026-11-14 13:02:33 |
| G3 | qwen2.5-32b-q6k ctx 4096 PP default, eval_kv4k | ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096kv4k_r1 | 2026-11-14 13:03:23 | 2026-11-14 13:05:55 |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_long | ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096proplong_r1 | 2026-11-14 13:06:45 | 2026-11-14 13:12:21 |
| G3 | qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_kv4k | ok | A1_wifi_qwen2.5-32b-q6k_m5_c4096propkv4k_r1 | 2026-11-14 13:13:14 | 2026-11-14 13:15:45 |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_long | ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384proplong_r1 | 2026-11-14 13:16:41 | 2026-11-14 14:52:59 |
| G3 | qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_kv4k | ok | B1_wifi_qwen2.5-32b-q6k_m5_c16384propkv4k_r1 | 2026-11-14 14:53:53 | 2026-11-14 15:15:15 |

## Log (last 40 lines)

```
[2026-11-14 12:22:04]   BASELINE_local_qwen2.5-14b-q4km_m5_c32768_r1: OK, quality_pass  decode 17.20039515812254  cmds/step None  cmds/tok None  B/tok None  acc 0.875 
[2026-11-14 12:22:49] run qwen2.5-14b-q4km ctx 32768 Air alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host air --subset 8
[2026-11-14 12:23:02]   BASELINE_local_qwen2.5-14b-q4km_air_c32768_r1: FAIL (oom_requester_compute), quality_pass  decode None  cmds/step None  cmds/tok None  B/tok None  acc 0.0 
[2026-11-14 12:23:47] run qwen2.5-14b-q4km ctx 32768 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:24:37]   A1_wifi_qwen2.5-14b-q4km_m5_c32768_r1: OK, quality_pass  decode 8.847183854986273  cmds/step 7.0  cmds/tok 8.4  B/tok 89154.43333333333  acc 0.875 
[2026-11-14 12:25:22] run qwen2.5-14b-q4km ctx 32768 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:34:00]   B1_wifi_qwen2.5-14b-q4km_m5_c32768_r1: OK, quality_pass  decode 0.46209686342558076  cmds/step 1.5  cmds/tok 347.4  B/tok 37310976.333333336  acc 0.875 
[2026-11-14 12:34:46] preflight qwen2.5-7b-q4km
[2026-11-14 12:35:07]   [m5] model sha 65b8fcd92af6, free 487.0 GB
[2026-11-14 12:35:07]   [air] model sha 65b8fcd92af6, free 40.0 GB
[2026-11-14 12:35:07]   vpn_state disconnected, awdl up
[2026-11-14 12:35:07]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 12:35:07] run qwen2.5-7b-q4km ctx 4096 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 12:35:11]   BASELINE_local_qwen2.5-7b-q4km_m5_c4096_r1: OK, quality_pass  decode 34.126309119092724  cmds/step None  cmds/tok None  B/tok None  acc 1.0 
[2026-11-14 12:35:58] run qwen2.5-7b-q4km ctx 4096 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:37:17]   A1_wifi_qwen2.5-7b-q4km_m5_c4096_r1: OK, quality_pass  decode 20.780713038733634  cmds/step 7.0  cmds/tok 8.3125  B/tok 61371.03125  acc 1.0 
[2026-11-14 12:38:04] run qwen2.5-7b-q4km ctx 4096 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:43:20]   B1_wifi_qwen2.5-7b-q4km_m5_c4096_r1: OK, quality_pass  decode 0.9546785522086902  cmds/step 1.5  cmds/tok 201.28125  B/tok 13636239.71875  acc 1.0 
[2026-11-14 12:44:07] run qwen2.5-7b-q4km ctx 32768 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 12:44:11]   BASELINE_local_qwen2.5-7b-q4km_m5_c32768_r1: OK, quality_pass  decode 33.92523005493803  cmds/step None  cmds/tok None  B/tok None  acc 1.0 
[2026-11-14 12:44:59] run qwen2.5-7b-q4km ctx 32768 Air alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host air --subset 8
[2026-11-14 12:45:08]   BASELINE_local_qwen2.5-7b-q4km_air_c32768_r1: OK, quality_pass  decode 26.23695827000108  cmds/step None  cmds/tok None  B/tok None  acc 1.0 
[2026-11-14 12:45:54] run qwen2.5-7b-q4km ctx 32768 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:46:35]   A1_wifi_qwen2.5-7b-q4km_m5_c32768_r1: OK, quality_pass  decode 21.487384809745834  cmds/step 7.0  cmds/tok 8.3125  B/tok 61325.78125  acc 1.0 
[2026-11-14 12:47:22] run qwen2.5-7b-q4km ctx 32768 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 12:52:07]   B1_wifi_qwen2.5-7b-q4km_m5_c32768_r1: OK, quality_pass  decode 0.7805886813577941  cmds/step 1.5  cmds/tok 201.28125  B/tok 13638019.65625  acc 1.0 
[2026-11-14 12:52:57] commit G4: ok
[2026-11-14 12:52:57] == G3: real long prompts at the edge of each partition (6 steps)
[2026-11-14 12:52:57] run qwen2.5-32b-q6k ctx 4096 PP default, eval_long (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096long --load-timeout 2400 --request-timeout 3600 --network wifi --subset 3 --prompt-set eval_long --fixed-tokens 256
[2026-11-14 13:02:33]   A1_wifi_qwen2.5-32b-q6k_m5_c4096long_r1: OK, quality unknown  decode 3.3061331559592895  cmds/step 7.0  cmds/tok 7.091145833333333  B/tok 45631.979166666664  acc None 
[2026-11-14 13:03:23] run qwen2.5-32b-q6k ctx 4096 PP default, eval_kv4k (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096kv4k --load-timeout 2400 --request-timeout 3600 --network wifi --prompt-set eval_kv4k --fixed-tokens 64
[2026-11-14 13:05:55]   A1_wifi_qwen2.5-32b-q6k_m5_c4096kv4k_r1: OK, quality unknown  decode 3.1291499434381698  cmds/step 7.0  cmds/tok 8.859375  B/tok 50799.921875  acc None 
[2026-11-14 13:06:45] run qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_long (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096proplong --load-timeout 2400 --request-timeout 3600 --network wifi --subset 3 --prompt-set eval_long --fixed-tokens 256 --override-flag=-ts=2,3
[2026-11-14 13:12:21]   A1_wifi_qwen2.5-32b-q6k_m5_c4096proplong_r1: OK, quality unknown  decode 3.1272564193412404  cmds/step 7.0  cmds/tok 7.091145833333333  B/tok 46078.9609375  acc None 
[2026-11-14 13:13:14] run qwen2.5-32b-q6k ctx 4096 PP 3:2, eval_kv4k (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096propkv4k --load-timeout 2400 --request-timeout 3600 --network wifi --prompt-set eval_kv4k --fixed-tokens 64 --override-flag=-ts=2,3
[2026-11-14 13:15:45]   A1_wifi_qwen2.5-32b-q6k_m5_c4096propkv4k_r1: OK, quality unknown  decode 3.196500471284038  cmds/step 7.0  cmds/tok 8.859375  B/tok 50777.15625  acc None 
[2026-11-14 13:16:41] run qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_long (G3): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384proplong --load-timeout 2400 --request-timeout 3600 --network wifi --subset 3 --prompt-set eval_long --fixed-tokens 256 --override-flag=-ts=3,2
[2026-11-14 14:52:59]   B1_wifi_qwen2.5-32b-q6k_m5_c16384proplong_r1: OK, quality unknown  decode 0.19443213445993143  cmds/step 1.5  cmds/tok 390.51953125  B/tok 84804915.51432292  acc None 
[2026-11-14 14:53:53] run qwen2.5-32b-q6k ctx 16384 TP 3:2, eval_kv4k (G3): nocturne.bench.run --cell B1 --model qwen2.5-32b-q6k --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384propkv4k --load-timeout 2400 --request-timeout 3600 --network wifi --prompt-set eval_kv4k --fixed-tokens 64 --override-flag=-ts=3,2
[2026-11-14 15:15:15]   B1_wifi_qwen2.5-32b-q6k_m5_c16384propkv4k_r1: OK, quality unknown  decode 0.1826653478951599  cmds/step 1.5  cmds/tok 487.5703125  B/tok 84986448.125  acc None 
```
