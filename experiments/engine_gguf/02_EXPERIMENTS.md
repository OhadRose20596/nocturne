# Engine GGUF, part 2 of 2: runs on three machines

Work prompt for Claude Code. Written 2026-11-21 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo, after part 1 (`01_BUILD.md`) passed its gate and Ohad has the Codex review (`REVIEW_CODEX.md`).
Ohad powers on the Air and gtx before this part. He is nearby for the human steps listed in session 8 section 3 (VPN, AWDL loop, Windows prompts, sudo).

## 0. Read first

1. `CLAUDE.md`.
2. `experiments/engine_gguf/01_BUILD.md`, `VERIFY.md`, `REVIEW.md`, `REVIEW_CODEX.md`, `PREDICTIONS.md`, and the part 1 section of `experiments/RESULTS.md`.
3. `docs/JOIN.md` (with the part 1 additions).
4. The session 8c rows of `experiments/RESULTS.md` (32B Q8_0 with llama.cpp on three machines).

Rules as in part 1. If `REVIEW_CODEX.md` has a high finding that part 1 did not fix, stop and report before any run.

## 1. Phase A: machines back

A1. Join tool on each host: `--check --self` on the M5, `--check` on the Air and on gtx (through `15_join.ps1 -Check` if a Windows step is needed, a human checkpoint). Then `16_join_apply.py apply` for each report. Record the reports and the diffs in `experiments/engine_gguf/connectivity.md`.
A2. Same repo SHA on the three hosts. Same engine sha256.
A3. On gtx: run the part 1 unit tests (dequant, name map, alignment) on CUDA. They ran only on MPS and CPU so far. Any failure stops the session.
A4. Copy the GGUF files that the runs below need to the hosts that lack them, with the `scp` commands `apply` prints. Check sha256 on both ends.
A5. Link record before every run: ping and TCP connect from the M5 to each peer, interface type of each host (the new `networks` field).

## 2. Phase B: the runs

Pre-register each row in `experiments/engine_gguf/PREDICTIONS.md` before it runs. Session dir `experiments/engine_gguf`. Run ids carry `gguf` and `_r1`. Stop at the first failing rung, one diagnosed retry, never edit old run folders.

| rung | what | criterion |
|---|---|---|
| R1 | 7B Q8_0 GGUF, `H4_pp3` placement (m5, air, gtx) | at least 7 of 8 identical against the part 1 one rank run of the same file. Frames per step as planned |
| R2 | 7B Q8_0 GGUF, `H5_tp3` placement | same |
| R3 | 7B Q8_0 GGUF, one hybrid placement over the three machines (TP pair m5 and air, PP stage on gtx) | same |
| R4 | 7B Q4_K_M (sha256 65b8fcd9), `H4_pp3` and `H5_tp3`, against the one rank run of the same file and against llama.cpp on the same file. Watch the K quant alignment rule on the TP head split | same |
| R5 | 32B Q8_0 (sha256 fc510333, the session 8c file), PP over the three machines, layer shares as close to `A1_wifi_qwen2.5-32b-q8_m5_c512ts3m_r1` as the planner accepts. If it does not fit, report the planner's numbers and the gap, then try the placement it proposes | completes, task accuracy on the 8 prompts equal to the llama.cpp row. There is no one rank reference for 32B, so compare tokens with the llama.cpp row and report the match count |
| R6 | 32B Q8_0, hybrid or TP, only if R5 passed and the planner accepts one | as R5 |

R5 is the first run where both engines load the same 32B file on the same three machines. Record the M5 GPU limit policy (`iogpu.wired_limit_mb`) as session 8c did. Raising it is a human step and a separate policy row.

## 3. Records and push

- `experiments/RESULTS.md`: section "Engine GGUF, part 2 (three machines)", same columns as session 8b and 8c, plus quant type, file sha256 and the link fields.
- Fill the outcome column of each prediction.
- `docs/CURRENT_CONTEXT.md`: two lines.
- One entry in `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` (private, never commit).
- Small commits, push, pull on air and gtx, same SHA on three hosts.

Final message under 25 lines: rungs passed or failed with cause, R5 against llama.cpp in one line, memory planned against measured per host, what session 9 (timeline) should run first.

## 4. Do not

Bind `0.0.0.0`. Use hotspot addresses. Install or upgrade torch. Change engine code (bugs found here are reported, fixed in a separate approved step, and re-verified as in part 1). Edit old run folders. Commit model files. Leave test processes running.
