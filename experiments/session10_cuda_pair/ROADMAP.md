# Roadmap from 28 November 2026

Written by Claude (Cowork) with Ohad after the 23.11 meeting with Chen. Sessions run in this order. Each has a build part (code, tests, independent review) and an experiment part (machines on). Only session 10 has prompts so far.

| session | goal | Chen's points | prompts |
|---|---|---|---|
| 10 | Infrastructure on the CUDA pair (gtx and Lior's laptop `pc`): clock sync, a recorder that logs compute, waits and bytes per device, any machine can be the requester and planner, the requester ships each worker its part of the model. Then CUDA baseline and the largest models the pair can hold, with repeated prompts | time sync, per-device compute and communication, CUDA baseline, repeat prompts | `01_BUILD.md`, `02_EXPERIMENTS.md` |
| 11 | Four machines (M5, Air, gtx, pc): large models (32B 8 bit, larger if the planner allows), PP, TP, hybrid, repeated prompts, every run with the recorder and synced clocks | scale, charts by device, total bytes | not written |
| 12 | Efficiency: K requests in flight in PP, interleaved requests in TP, chunked prefill, speculative decoding with the draft model run while waiting. Measured against session 11 | overlap of compute and communication | not written |
| 13 | MoE: a current MoE model on llama.cpp, then experts as a split axis in our engine | MoE | not written |
| later | The medium (wired, Wi-Fi, internet), pipeline order by bandwidth, tracker based joining | medium, ordering, joining | not written |

The design walkthrough slides for the next meeting (about 30.11) are made separately in Cowork. Session 10 part 1 writes `docs/ENGINE_DESIGN.md` and `docs/MEASUREMENT.md`, which the slides use.
