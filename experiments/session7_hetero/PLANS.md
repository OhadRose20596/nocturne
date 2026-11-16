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
