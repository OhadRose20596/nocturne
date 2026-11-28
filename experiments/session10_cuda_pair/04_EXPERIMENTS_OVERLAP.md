# Session 10, part 4 of 4: overlap on the CUDA pair (experiments)

Work prompt for Claude Code. Written 2026-11-28 by Claude (Cowork) with Ohad. Run it on the M5 after part 3 passed its gate and Ohad has the Codex review of part 3.
Hosts: gtx and pc, coordinator gtx. The M5 operates. Same human steps as part 2.

## 0. Read first

`03_BUILD_OVERLAP.md`, `VERIFY_overlap.md`, the review files, the part 2 rows in `experiments/RESULTS.md` (the "before"), `docs/MEASUREMENT.md`.

## 1. Phase A

Same as part 2 phase A: same SHA on both hosts, clock sync, link record, CUDA tests.

## 2. Phase B: before and after

Same model and file as part 2 rung B2 and B3 (7B Q8_0), the 8 eval prompts plus the long prompt, `--timeline` on every run. Pre-register predictions in `PREDICTIONS.md`. The 23.11 predictions: PP with 2 requests in flight raises each machine's busy share from about 40% to about 80%, TP with 2 interleaved requests from about 47% to about 94%, chunked prefill shortens prefill by about a third. Correctness rules of part 3 apply to every run.

| rung | what | compare with |
|---|---|---|
| C1 | PP, `--slots 1, 2, 3` | part 2 B2 |
| C2 | PP, `--prefill-chunk 16` and `32`, slots 1 | C1 slots 1 |
| C3 | PP, slots 2 and chunked prefill together | C1, C2 |
| C4 | TP, `--tp-interleave 2` | part 2 B3 |
| C5 | O4 on and off, PP slots 2 and TP interleave 2 | C1, C4 |
| C6 | Interleaved stages: gtx 0 to 6, pc 7 to 13, gtx 14 to 20, pc 21 to 27, with slots 1, 2, 3 and chunked prefill | C1, C3 on the contiguous placement |
| C7 | The best combination for 3 prompts in a row and a follow-up (as part 2 B5) | part 2 B5 |

Report per run: tokens per second of the whole system, time per request, busy share and overlap share per device, bytes per link, and the frame count against the formula. Time is background for correctness (T1), but here it is the object of the study: report it carefully, with the clock bounds.

## 3. Charts for Chen

For C1, C4 and C6: the before and after timelines one above the other on the same time scale, per device compute and communication rows, colored by peer. One table with all rungs: busy share per device, overlap share, tokens per second, time per request, bytes.

## 4. Records and push

`experiments/RESULTS.md` section "Session 10, part 4: overlap", predictions filled, `docs/CURRENT_CONTEXT.md`, private shared log. Small commits, push, same SHA on both hosts.
Final message under 25 lines: what each feature did against its prediction, the best combination, what the interleaved stages added, correctness, and what session 11 (four machines) should use by default.

## 4b. Do not

Change engine code. Bind `0.0.0.0`. Install or upgrade torch. Edit old run folders. Leave processes running.
