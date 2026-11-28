# Session 10, part 2 of 2: the CUDA pair

Work prompt for Claude Code. Written 2026-11-28 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo after part 1 passed its gate and Ohad has `REVIEW_CODEX.md`.
Hosts: `gtx` (GTX 1070 8 GB, WSL2) and `pc` (Lior's Lenovo P16v, RTX 2000 Ada 8 GB, WSL2). The M5 is only the operator, not a rank. Ohad is nearby for human steps (Windows prompts, sudo, VPN).

## 0. Read first

`01_BUILD.md`, `VERIFY.md`, `REVIEW.md`, `REVIEW_CODEX.md`, `docs/MEASUREMENT.md`, `docs/ENGINE_DESIGN.md`, the engine GGUF part 2 section of `experiments/RESULTS.md`. If `REVIEW_CODEX.md` has an unfixed high finding, stop.

## 1. Phase A: machines

A1. Join tool on `pc` (its address is stale): `15_join.ps1 -Check`, then `16_join_apply.py apply --peer pc`. Same check on `gtx`. Record in `experiments/session10_cuda_pair/connectivity.md`.
A2. Same repo SHA and engine hash on both. Torch versions recorded (gtx 2.14.1+cu126, pc 2.14.1+cu130 last time).
A3. Clock sync with `scripts/17_timesync.sh` on both, then the engine offset check. Record both.
A4. Link record: ping and TCP connect between gtx and pc in both directions, interface type.
A5. Unit tests with `NOCTURNE_TEST_DEVICE=cuda` on both.

## 2. Phase B: runs

Every run: `--timeline`, `--ship-weights`, clocks synced just before, coordinator on gtx unless the row says pc. Pre-register each row in `experiments/session10_cuda_pair/PREDICTIONS.md`. Stop at the first failing rung, one diagnosed retry.

| rung | what | criterion |
|---|---|---|
| B1 | CUDA baseline, one rank: 1.5B and 7B Q4_K_M on each card alone, our engine and llama.cpp (CUDA build) on the same file | tokens against the Metal one-rank runs of the same file (report the count, not a gate across devices), and engine against llama.cpp on the same card |
| B2 | 7B Q8_0, PP over the pair, coordinator gtx | 7 of 8 identical against the one rank run. Timeline complete |
| B3 | 7B Q8_0, TP over the pair | same |
| B4 | B2 again with the coordinator on pc | same tokens as B2. Shows the requester can be any machine |
| B5 | Repeated prompts: 3 consecutive prompts and 1 follow-up in one session, PP and TP, our engine and llama.cpp | the setup phase (weight shipping, cache saves) per prompt, from the recorder |
| B6 | The largest model the pair can hold, chosen by the planner (candidates: 14B Q4_K_M, 14B Q5_K_M, 7B bf16). PP, then TP if the planner accepts | 7 of 8 against llama.cpp on the same file where a one-rank reference cannot exist |
| B7 | Charts for Chen from the recorder: per device compute and communication rows, total bytes per link, B2 and B3 side by side, CUDA pair against the earlier Mac pair (23.11 numbers) | files exist and parse |

## 3. Records and push

`experiments/RESULTS.md` section "Session 10: CUDA pair", the same columns as engine GGUF part 2 plus clock offsets, shipped bytes and the coordinator host. Fill predictions. `docs/CURRENT_CONTEXT.md` two lines. Shared log entry (private). Small commits, push, pull on both hosts.
Final message under 25 lines: rungs, the CUDA against Metal comparison in one line, whether the compute-then-send pattern is the same on CUDA, the largest model that ran, shipped bytes, what session 11 (four machines) should do first.

## 4. Do not

Change engine code (report bugs, fix in a separate approved step). Bind `0.0.0.0`. Use hotspot addresses. Install or upgrade torch. Edit old run folders. Commit model files. Leave processes running.
