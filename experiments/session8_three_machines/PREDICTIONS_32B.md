# Session 8c: 32B Q8_0 with llama.cpp on three machines, predictions

Written 2026-11-20 before any run. Engine llama.cpp at 1e7bcf3 on the three hosts (gtx built today with CUDA 12.4, architecture 61).
Bench cell A1 (PP, layer split over RPC), model `qwen2.5-32b-q8` (34.8 GB file, sha256 fc510333...), home Wi-Fi, requester m5, workers air and gtx.
Peers file `~/.nocturne/peers_3m.yaml`: the usual file with `workers: [air, gtx]` and `rpc_port: 50071` (the Windows firewall rule on gtx covers 50070 to 50079, not 50052).
The comparison is with the session 5 rows of the same model, where every split on the two Macs failed (`A1_wifi_qwen2.5-32b-q8_m5_c512_r1` and the rows next to it in experiments/RESULTS.md).

## Run 1: default memory policy

Record id `A1_wifi_qwen2.5-32b-q8_m5_c512ts3m_r1`. Context 512, `-ts 21,11,32` (air, gtx, m5: the `--rpc` devices come first, the local device last), 8 prompts.
Memory policy as in session 5: `-ngl 99`, macOS default GPU limit on both Macs (`iogpu.wired_limit_mb` 0), budgets 17.76 GiB (m5) and 11.84 GiB (air). For gtx the budget is the VRAM that is free before the run, about 7.05 GiB.

| item | prediction | why | outcome |
|---|---|---|---|
| Load | loads on the three devices, no OOM class | session 5 needed 32.1 GiB of device buffers at ctx 512; the three budgets sum to about 36.6 GiB | Held. Loaded on the three devices, 65 of 65 layers offloaded, no OOM. |
| m5 buffers | about 16.4 GiB, headroom about 1.3 GiB | 32 layers at about 0.484 GiB, the output layer 0.73 GiB, compute and KV | Wrong by 0.3 GiB. 16.759 GiB, headroom 1.00 GiB. A layer is 0.5055 GiB, not 0.484. |
| air buffers | about 10.3 GiB, headroom about 1.5 GiB | 21 layers | Wrong by 0.5 GiB. 10.837 GiB, headroom 1.00 GiB. |
| gtx buffers | about 5.45 GiB, headroom about 1.6 GiB | 11 layers | Held. 5.518 GiB, headroom 1.54 GiB (7.06 GiB free before the run). |
| Outcome | success: every request finishes with stop or length | | Held. success true, 8 of 8 requests finished with stop. |
| Accuracy | equal to the Q6_K rows of session 5 or better on the 8 prompts | same model family at a higher precision | Held. Task accuracy 1.0 on the 8 prompts. |
| What would count as a failure of the reasoning | an OOM on any device although the planned buffers are under its budget, or wrong text from a CUDA 12.4 build on a Pascal card | | Not triggered. |

## Run 1b: TP (llama.cpp tensor mode), default memory policy

Written 2026-11-20 after run 1 and before this run. Record id `B1_wifi_qwen2.5-32b-q8_m5_c512ts3m_r1`. Cell B1: `--split-mode tensor`, `-fit off`, context 512, 8 prompts.
`-ts 31,20,10` in the order m5, air, gtx (in tensor mode the Meta device lists the local device first). Same memory policy as run 1.
Session 5 reference: `B1_wifi_qwen2.5-32b-q8_m5_c512prop_r1` loaded on both Macs and failed with `oom_requester_compute` (m5 model share 18.831 GiB, air 12.554 GiB, 31.385 GiB in total).
Tensor mode was never run with three devices, and never with a CUDA device behind RPC. Upstream documents the Meta backend for CUDA only.

| item | prediction | why | outcome |
|---|---|---|---|
| Tool | outcome (a): a Meta device over three devices, each holding its share | it worked with two devices on 7B and on 32B Q6_K; three devices is untested, so (c) assert or crash is possible | Held. Outcome (a): Meta device from 3 devices, each holds its share. |
| m5 buffers | about 16.2 GiB on the GPU, headroom about 1.5 GiB | 31 of 61 shares of 31.385 GiB plus KV and compute | Held: 16.450 GiB on the GPU. The policy did not hold: the M5 limit was 20480 MiB during this run (raised by Ohad just before the server started), so the headroom was 3.55 GiB. Against the default limit it would be 1.31. |
| air buffers | about 10.5 GiB, headroom about 1.3 GiB | 20 of 61 | Held: 10.670 GiB, headroom 1.17 GiB. |
| gtx buffers | about 5.4 GiB, headroom about 1.6 GiB | 10 of 61 | Held: 5.424 GiB, headroom 1.64 GiB. |
| Outcome | success if the tool accepts three devices; every request finishes | memory is under each budget on paper | Held. success true, 8 of 8 requests finished with stop. |
| Accuracy | 1.0 on the 8 prompts, as in run 1 | same weights | Held: 1.0. |
| Background | decode far slower than run 1 (session 5 Q6_K TP gave 0.27 tok/s against PP at several tok/s) | every layer crosses the links | Held: 0.198 tok/s median against 3.79 in run 1. |

## Run 2: PP with the M5 GPU limit raised to 20 GiB, context 4096

Written 2026-11-20 after runs 1 and 1b, before this run. Record id `A1_wifi_qwen2.5-32b-q8_m5_c4096ts3mwl20_r1`. Cell A1, context 4096, 8 prompts.
Memory policy: `iogpu.wired_limit_mb=20480` on the M5 (set by Ohad with sudo, resets at reboot). Air and gtx as before. This is a different policy from session 5 and from run 1, hence the `wl20` tag.
`-ts 18,10,36` (air, gtx, m5). The session message proposed 37, 18, 9; 36 on the M5 leaves about 1 GiB there with the measured layer size.
Session 5 reference at this context: every Q8 split on two Macs failed (`A1_wifi_qwen2.5-32b-q8_m5_c4096_r2` and the rows next to it).

| item | prediction | why | outcome |
|---|---|---|---|
| Load | loads, no OOM class | measured layer 517.7 MiB; KV grows 8 times against ctx 512 (about 1.0 GiB in total) | Held. Loaded, 65 of 65 layers offloaded, no OOM. KV 1024 MiB in total. |
| m5 GPU buffers | about 19.0 GiB, headroom about 1.0 GiB under the 20.0 GiB limit | 36 layers 18.20, KV about 0.56, compute about 0.2 to 0.3 | Wrong by 0.6 GiB. 18.389 GiB, headroom 1.61 GiB. llama.cpp placed 18083 MiB on the M5, about 35 layers, not 36. |
| air buffers | about 9.6 GiB, headroom about 2.2 GiB | 18 layers 9.10, KV about 0.28, compute about 0.2 | Held: 9.647 GiB, headroom 2.19 GiB. |
| gtx buffers | about 5.4 GiB, headroom about 1.6 GiB | 10 layers 5.06, KV about 0.16, compute about 0.2 | Close: 5.173 GiB, headroom 1.87 GiB. |
| Outcome | success, 8 of 8 finish | | Held. success true, 8 of 8 finished with stop. |
| Accuracy | 1.0 on the 8 prompts | as runs 1 and 1b | Held: 1.0. |
| What would count as a failure of the reasoning | an allocation failure on the M5 below 20 GiB: the raised limit is then not usable in full with other applications open | | Not triggered. The M5 allocated 18.39 GiB on the GPU. |
