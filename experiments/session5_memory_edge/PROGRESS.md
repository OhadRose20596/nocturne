# Session 5 progress

Updated 2026-11-14 07:06:59. Steps known so far: 57, finished: 57. Later groups add steps from earlier results (the largest model that fits, the edge per partition).

| group | step | status | records | start | end |
|---|---|---|---|---|---|
| G0 | qwen2.5-32b-q4km ctx 32768 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q5km ctx 4096 PP balanced | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q5km ctx 4096 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q5km ctx 4096 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q6k ctx 4096 PP balanced | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q6k ctx 4096 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q6k ctx 4096 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q8 ctx 4096 PP balanced | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q8 ctx 4096 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G2a | qwen2.5-32b-q8 ctx 4096 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q5km ctx 512 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q5km ctx 512 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q5km ctx 512 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q6k ctx 512 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q6k ctx 512 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q6k ctx 512 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q8 ctx 512 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q8 ctx 512 PP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G2edge | qwen2.5-32b-q8 ctx 512 TP 3:2 | dry run |  | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 512 PP default | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c512_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 512 PP balanced | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c512bal_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP default | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 4096 PP balanced | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c4096bal_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP default | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c16384_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 16384 PP balanced | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c16384bal_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP default | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c32768_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 32768 PP balanced | dry run | A1_wifi_qwen2.5-32b-q4km_m5_c32768bal_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 4096 TP | dry run | B1_wifi_qwen2.5-32b-q4km_m5_c4096_r1 | 2026-11-14 07:06:59 |  |
| G1r | qwen2.5-32b-q4km ctx 32768 TP 3:2 | dry run | B1_wifi_qwen2.5-32b-q4km_m5_c32768prop_r1 | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 4096 M5 alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 4096 Air alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 4096 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 4096 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 32768 M5 alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 32768 Air alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 32768 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-14b-q4km ctx 32768 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 4096 M5 alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 4096 Air alone | done earlier: ok | BASELINE_local_qwen2.5-7b-q4km_air_c4096_r2 |  |  |
| G4 | qwen2.5-7b-q4km ctx 4096 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 4096 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 32768 M5 alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 32768 Air alone | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 32768 PP default | dry run |  | 2026-11-14 07:06:59 |  |
| G4 | qwen2.5-7b-q4km ctx 32768 TP | dry run |  | 2026-11-14 07:06:59 |  |
| G3 | qwen2.5-32b-q4km ctx 32768 PP default, eval_long | dry run |  | 2026-11-14 07:06:59 |  |
| G3 | qwen2.5-32b-q4km ctx 32768 PP default, eval_kv4k | dry run |  | 2026-11-14 07:06:59 |  |
| G3 | qwen2.5-32b-q4km ctx 32768 TP 3:2, eval_long | dry run |  | 2026-11-14 07:06:59 |  |
| G3 | qwen2.5-32b-q4km ctx 32768 TP 3:2, eval_kv4k | dry run |  | 2026-11-14 07:06:59 |  |
| G5 | qwen2.5-1.5b-bf16 ctx 4096 TP torch | dry run |  | 2026-11-14 07:06:59 |  |
| G5 | qwen2.5-7b-bf16 ctx 4096 TP torch | dry run |  | 2026-11-14 07:06:59 |  |

## Log (last 40 lines)

```
[2026-11-14 07:06:59] run qwen2.5-32b-q8 ctx 512 PP 3:2 (G2edge): nocturne.bench.run --cell A1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=2,3
[2026-11-14 07:06:59] run qwen2.5-32b-q8 ctx 512 TP 3:2 (G2edge): nocturne.bench.run --cell B1 --model qwen2.5-32b-q8 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 07:06:59] largest model fitting at 4096: PP none, TP none; context sweep on []
[2026-11-14 07:06:59] == G2b: context sweep of the largest fitting models
[2026-11-14 07:06:59] == G1r: repeats of the 32B Q4_K_M cells (10 steps)
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 512 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 512 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 512 --runs 1 --no-results-md --allow-unmeasured-memory --tag c512bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 4096 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 4096 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 16384 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 16384 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 16384 --runs 1 --no-results-md --allow-unmeasured-memory --tag c16384bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 PP default (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 PP balanced (G1r): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768bal --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=1,1
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 4096 TP (G1r): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 TP 3:2 (G1r): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768prop --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8 --override-flag=-ts=3,2
[2026-11-14 07:06:59] == G4: 14B and 7B Q4_K_M, every partition, ctx 4096 and 32768 (16 steps)
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 4096 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 4096 Air alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --host air --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 4096 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 4096 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 32768 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 32768 Air alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host air --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 32768 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-14b-q4km ctx 32768 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-14b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 4096 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 4096 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 4096 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 32768 M5 alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host m5 --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 32768 Air alone (G4): nocturne.bench.run --cell BASELINE --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --host air --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 32768 PP default (G4): nocturne.bench.run --cell A1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-q4km ctx 32768 TP (G4): nocturne.bench.run --cell B1 --model qwen2.5-7b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] == G3: real long prompts at the edge of each partition (4 steps)
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 PP default, eval_long (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768long --load-timeout 2400 --request-timeout 3600 --network wifi --subset 3 --prompt-set eval_long --fixed-tokens 256
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 PP default, eval_kv4k (G3): nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768kv4k --load-timeout 2400 --request-timeout 3600 --network wifi --prompt-set eval_kv4k --fixed-tokens 64
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 TP 3:2, eval_long (G3): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768proplong --load-timeout 2400 --request-timeout 3600 --network wifi --subset 3 --prompt-set eval_long --fixed-tokens 256 --override-flag=-ts=3,2
[2026-11-14 07:06:59] run qwen2.5-32b-q4km ctx 32768 TP 3:2, eval_kv4k (G3): nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 32768 --runs 1 --no-results-md --allow-unmeasured-memory --tag c32768propkv4k --load-timeout 2400 --request-timeout 3600 --network wifi --prompt-set eval_kv4k --fixed-tokens 64 --override-flag=-ts=3,2
[2026-11-14 07:06:59] == G5: torch_tp engine, bf16, ctx 4096 (2 steps)
[2026-11-14 07:06:59] run qwen2.5-1.5b-bf16 ctx 4096 TP torch (G5): nocturne.bench.run --cell B1t --model qwen2.5-1.5b-bf16 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] run qwen2.5-7b-bf16 ctx 4096 TP torch (G5): nocturne.bench.run --cell B1t --model qwen2.5-7b-bf16 --peers /Users/orosentswaig/.nocturne/peers.yaml --ctx 4096 --runs 1 --no-results-md --allow-unmeasured-memory --tag c4096 --load-timeout 2400 --request-timeout 3600 --network wifi --subset 8
[2026-11-14 07:06:59] all groups finished
```
