# Session 5 progress

Updated 2026-11-14 07:48:31. Steps known so far: 1, finished: 1. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

| group | step | status | records | start | end |
|---|---|---|---|---|---|
| G0 | qwen2.5-32b-q4km ctx 32768 PP 3:2 | ok | A1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1 | 2026-11-14 07:46:49 | 2026-11-14 07:48:15 |

## Log (last 40 lines)

```
[2026-11-14 07:45:27] == G0: layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2) (1 steps)
[2026-11-14 07:45:27] preflight qwen2.5-32b-q4km
[2026-11-14 07:46:49]   [m5] model sha 2e5f6daea180, free 496.0 GB
[2026-11-14 07:46:49]   [air] model sha 2e5f6daea180, free 52.0 GB
[2026-11-14 07:46:49]   vpn_state disconnected, awdl up
[2026-11-14 07:46:49]   worker 192.168.1.219:50052 closed (the bench starts it over ssh)
[2026-11-14 07:46:49] run qwen2.5-32b-q4km ctx 32768 PP 3:2 (G0): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 07:48:15]   A1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1: OK, quality unknown  decode 5.433245800059083  cmds/step 7.0  cmds/tok 8.3125  B/tok 100866.1875  acc 1.0 
```
