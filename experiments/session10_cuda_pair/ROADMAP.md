# Roadmap from 28 November 2026

Written by Claude (Cowork) with Ohad after the 23.11 meeting with Chen. Sessions run in this order. Each has a build part (code, tests, independent review) and an experiment part (machines on). Only session 10 has prompts so far.

| session | goal | Chen's points | prompts |
|---|---|---|---|
| 10 | Infrastructure on the CUDA pair (gtx and Lior's laptop `pc`): clock sync, a recorder that logs compute, waits and bytes per device, any machine can be the requester and planner, the requester ships each worker its part of the model. Then CUDA baseline and the largest models the pair can hold, with repeated prompts. Then (added 28.11) compute while data is on the network: requests in flight, chunked prefill, interleaved TP requests, non-blocking sends, interleaved pipeline stages, measured before and after | time sync, per-device compute and communication, CUDA baseline, repeat prompts, overlap | order: `01_BUILD.md` (done), then `03_BUILD_OVERLAP.md` (M5 only), then one session on the pair with `02_EXPERIMENTS.md` (options off) followed by `04_EXPERIMENTS_OVERLAP.md` |
| 11 | Four machines (M5, Air, gtx, pc): large models (32B 8 bit, larger if the planner allows), PP, TP, hybrid, repeated prompts, every run with the recorder, synced clocks and the overlap features that part 4 found best | scale, charts by device, total bytes | not written |
| 12 | Speculative decoding (the draft model runs while waiting), keeping a conversation's KV cache between requests | single-request speed, follow-ups | not written |
| 13 | MoE: a current MoE model on llama.cpp, then experts as a split axis in our engine | MoE | not written |
| later | The medium (wired, Wi-Fi, internet), pipeline order by bandwidth, tracker based joining | medium, ordering, joining | not written |

The design walkthrough slides for the next meeting (about 30.11) are made separately in Cowork. Session 10 part 1 writes `docs/ENGINE_DESIGN.md` and `docs/MEASUREMENT.md`, which the slides use.
