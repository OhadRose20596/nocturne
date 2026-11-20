# Session 7 planner tables

Made with `python -m nocturne.engines.torch_tp plan --placement <file>` on the M5, 2026-11-16. Static: nothing was run on the PC.

## R0_0.5b_cuda

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | pc | cuda | 0-24 | 0-14 (14) | 0-2 | 0-4864 (4864) | 0-151936 EH | 988.1 | 50.3 | 31.8 | 1070.2 | - |

Frames per decode step: 0 (token 0, embedding 0, layers 0, hops 0, head 0); bytes per step about 0.000 MB.

## R0_1.5b_cuda

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | pc | cuda | 0-28 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 EH | 3087.4 | 117.4 | 51.6 | 3256.4 | - |

Frames per decode step: 0 (token 0, embedding 0, layers 0, hops 0, head 0); bytes per step about 0.000 MB.

## L1_tp_mixed_0.5b_loopback

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-24 | 0-7 (7) | 0-1 | 0-2432 (2432) | 0-76032 EH | 494.2 | 25.2 | 17.7 | 537.1 | - |
| s0/1 | m5 | cpu | 0-24 | 7-14 (7) | 1-2 | 2432-4864 (2432) | 76032-151936 EH | 494.0 | 25.2 | 17.7 | 536.9 | - |

Frames per decode step: 100 (token 1, embedding 2, layers 96, hops 0, head 1); bytes per step about 0.178 MB.

## H1_pp_1.5b

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-14 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 E | 1777.1 | 58.7 | 50.7 | 1886.5 | - |
| s1/0 | pc | cuda | 14-28 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 H | 1777.1 | 58.7 | 51.6 | 1887.4 | - |

Frames per decode step: 2 (token 0, embedding 0, layers 0, hops 1, head 1); bytes per step about 0.003 MB.

## H2_tp_1.5b

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-28 | 0-6 (6) | 0-1 | 0-4480 (4480) | 0-76032 EH | 1544.0 | 58.7 | 28.9 | 1631.7 | - |
| s0/1 | pc | cuda | 0-28 | 6-12 (6) | 1-2 | 4480-8960 (4480) | 76032-151936 EH | 1543.6 | 58.7 | 28.9 | 1631.3 | - |

Frames per decode step: 116 (token 1, embedding 2, layers 112, hops 0, head 1); bytes per step about 0.353 MB.

## H3_pp_7b

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-22 | 0-28 (28) | 0-4 | 0-18944 (18944) | 0-152064 E | 11344.5 | 184.5 | 115.0 | 11644.1 | - |
| s1/0 | pc | cuda | 22-28 | 0-28 (28) | 0-4 | 0-18944 (18944) | - | 2796.7 | 50.3 | 115.0 | 2962.0 | 3500 |
| head/0 | m5 | mps | 0-0 | 0-0 (0) | 0-0 | 0-0 (0) | 0-152064 H | 1090.0 | 0.0 | 15.6 | 1105.6 | - |

Frames per decode step: 3 (token 0, embedding 0, layers 0, hops 2, head 1); bytes per step about 0.014 MB.

# Session 8 planner tables (2026-11-20)

Made with `python -m nocturne.engines.torch_tp plan --placement <file>` on the M5 at 6196396, before any run.
`H4_pp3_1.5b.yaml` and `H5_tp3_1.5b.yaml` (host `pc`) were written on 2026-11-20 and never run. The `_gtx` files are copies with `host: gtx` and a new first comment line. Nothing else changed, and the planner accepted all four without edits.
The `pc` and `gtx` tables are equal apart from the host name, because the planner does not know the card.

## R0_1.5b_cuda_gtx

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | gtx | cuda | 0-28 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 EH | 3087.4 | 117.4 | 51.6 | 3256.4 | - |

Frames per decode step: 0 (token 0, embedding 0, layers 0, hops 0, head 0); bytes per step about 0.000 MB.

## H1_pp_1.5b_gtx

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-14 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 E | 1777.1 | 58.7 | 50.7 | 1886.5 | - |
| s1/0 | gtx | cuda | 14-28 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 H | 1777.1 | 58.7 | 51.6 | 1887.4 | - |

Frames per decode step: 2 (token 0, embedding 0, layers 0, hops 1, head 1); bytes per step about 0.003 MB.

## H4_pp3_1.5b_gtx (and H4_pp3_1.5b with host pc)

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-10 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 E | 1402.7 | 41.9 | 50.7 | 1495.3 | - |
| s1/0 | air | mps | 10-19 | 0-12 (12) | 0-2 | 0-8960 (8960) | - | 842.4 | 37.7 | 50.7 | 930.8 | - |
| s2/0 | gtx | cuda | 19-28 | 0-12 (12) | 0-2 | 0-8960 (8960) | 0-151936 H | 1309.1 | 37.7 | 51.6 | 1398.4 | - |

Frames per decode step: 3 (token 0, embedding 0, layers 0, hops 2, head 1); bytes per step about 0.006 MB.

## H5_tp3_1.5b_gtx (and H5_tp3_1.5b with host pc)

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-28 | 0-4 (4) | 0-1 | 0-2944 (2944) | 0-50688 EH | 1025.7 | 58.7 | 21.4 | 1105.9 | - |
| s0/1 | air | mps | 0-28 | 4-8 (4) | 0-2 | 2944-5888 (2944) | 50688-101376 EH | 1047.8 | 117.4 | 21.7 | 1186.9 | - |
| s0/2 | gtx | cuda | 0-28 | 8-12 (4) | 1-2 | 5888-8960 (3072) | 101376-151936 EH | 1058.4 | 58.7 | 21.6 | 1138.7 | - |

Frames per decode step: 346 (token 2, embedding 6, layers 336, hops 0, head 2); bytes per step about 1.059 MB.

# Session 8b planner tables: 7B on three machines (2026-11-20)

Made on the M5 before any 7B run. New files, written for gtx (GTX 1070, about 7.0 GB of VRAM free before a run). The planner accepted all three without edits.

## H3_pp_7b_gtx

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-18 | 0-28 (28) | 0-4 | 0-18944 (18944) | 0-152064 E | 9480.1 | 151.0 | 115.0 | 9746.1 | - |
| s1/0 | gtx | cuda | 18-28 | 0-28 (28) | 0-4 | 0-18944 (18944) | - | 4661.2 | 83.9 | 115.0 | 4860.1 | 5500 |
| head/0 | m5 | mps | 0-0 | 0-0 (0) | 0-0 | 0-0 (0) | 0-152064 H | 1090.0 | 0.0 | 15.6 | 1105.6 | - |

Frames per decode step: 3 (token 0, embedding 0, layers 0, hops 2, head 1); bytes per step about 0.014 MB.

## H4_pp3_7b_gtx

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-10 | 0-28 (28) | 0-4 | 0-18944 (18944) | 0-152064 E | 5751.2 | 83.9 | 115.0 | 5950.1 | - |
| s1/0 | air | mps | 10-19 | 0-28 (28) | 0-4 | 0-18944 (18944) | - | 4195.0 | 75.5 | 115.0 | 4385.6 | - |
| s2/0 | gtx | cuda | 19-28 | 0-28 (28) | 0-4 | 0-18944 (18944) | 0-152064 H | 5285.0 | 75.5 | 115.9 | 5476.5 | 6000 |

Frames per decode step: 3 (token 0, embedding 0, layers 0, hops 2, head 1); bytes per step about 0.014 MB.

## H5_tp3_7b_gtx

| rank | host | device | layers | heads | KV heads | MLP rows | vocab rows | weights MB | KV MB | workspace MB | total MB | budget MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0/0 | m5 | mps | 0-28 | 0-14 (14) | 0-2 | 0-9472 (9472) | 0-76032 EH | 7615.8 | 117.4 | 65.3 | 7798.6 | - |
| s0/1 | air | mps | 0-28 | 14-21 (7) | 2-3 | 9472-14208 (4736) | 76032-114048 EH | 3808.1 | 58.7 | 40.0 | 3906.8 | - |
| s0/2 | gtx | cuda | 0-28 | 21-28 (7) | 3-4 | 14208-18944 (4736) | 114048-152064 EH | 3808.1 | 58.7 | 40.0 | 3906.8 | 5500 |

Frames per decode step: 346 (token 2, embedding 6, layers 336, hops 0, head 2); bytes per step about 2.460 MB.

