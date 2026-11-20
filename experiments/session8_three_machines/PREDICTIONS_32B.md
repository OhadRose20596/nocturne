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
| Load | loads on the three devices, no OOM class | session 5 needed 32.1 GiB of device buffers at ctx 512; the three budgets sum to about 36.6 GiB | |
| m5 buffers | about 16.4 GiB, headroom about 1.3 GiB | 32 layers at about 0.484 GiB, the output layer 0.73 GiB, compute and KV | |
| air buffers | about 10.3 GiB, headroom about 1.5 GiB | 21 layers | |
| gtx buffers | about 5.45 GiB, headroom about 1.6 GiB | 11 layers | |
| Outcome | success: every request finishes with stop or length | | |
| Accuracy | equal to the Q6_K rows of session 5 or better on the 8 prompts | same model family at a higher precision | |
| What would count as a failure of the reasoning | an OOM on any device although the planned buffers are under its budget, or wrong text from a CUDA 12.4 build on a Pascal card | | |

## Run 2: M5 GPU limit raised

Only after run 1. Ohad sets `sudo sysctl iogpu.wired_limit_mb=20480` on the M5 (20 GiB). This is a different memory policy and gets its own row and tag (`wl20`).
Split and context are chosen after run 1 from the measured per layer size. Prediction to register then, before the run.
