# Session 10, part 1: verification record

Date: 2026-11-28. Operator: the M5 (24 GB, MPS, torch 2.14.1, Python 3.12). Start SHA of the change: `0e96d48`.
gtx (GTX 1070, WSL2, torch 2.14.1+cu126, Python 3.14) and the Air (16 GB, MPS, torch 2.15.0) came on during the session and were used for short checks. Lior's laptop (`pc`) joined the Wi-Fi late and its ssh server did not answer, so nothing ran on it.
No experiment rows were run. The runs below verify the build.

## Gate

| condition | result |
|---|---|
| tests pass | yes: 323 passed on the M5 (`tests/bench`, 3.5 minutes), 61 of them new |
| mutations caught | yes: 24 of 24 after two added assertions |
| V3 identical, 8 of 8 | yes, every run |
| no open high finding | yes: pass 1 had 0 high, 2 medium, 15 low; pass 2 (fixes only): 0 high, 0 medium, 13 low, and of the 17 pass 1 findings 12 fixed, 4 partly fixed, 1 left on purpose |

## V1. Tests

Run: `~/dev/nocturne-torch/bin/python -m pytest tests/bench -q`.

| where | result |
|---|---|
| M5, CPU and MPS | 323 passed on the M5 (`tests/bench`, 3.5 minutes), 61 of them new |
| gtx, Linux, `NOCTURNE_TEST_DEVICE=cuda`: `test_engine_gguf.py`, `test_engine_gguf_models.py`, the three `test_session10_*.py` files | 136 passed, 2 skipped, 1 failed. The failure was a test of mine, not the engine: `test_sampler_writes_cumulative_counters` burned too little CPU for the 10 ms tick of Linux. Fixed (the test now burns 250 ms) |
| gtx, the dequant and model tests on CUDA | all passed. `llama-quantize` was built on gtx for this (one target, two jobs, no sudo) |

New test files (61 tests on the M5):

| file | what it checks |
|---|---|
| `tests/bench/test_session10_timeline.py` | SNTP formula and OS tool parsers, the clock estimate and its sign, conversion to the coordinator clock, the recorder cursor, the sampler, loopback: identical tokens and logits with the timeline on, spans tile every step, recorder bytes equal the link counters, the four outputs |
| `tests/bench/test_session10_coordinator.py` | `plan.slots` against the full plan, the coordinator host lookup, who is started where (dry run with gtx as coordinator), preflight on a Linux coordinator, the tar collection, a loopback run with the coordinator under a second peer name in its own process, the same with shipped weights, `hybrid_run.main` end to end on a tiny GGUF, cleanup of leftover processes |
| `tests/bench/test_session10_shipping.py` | manifest against what the loader loads, slice bounds, chunks and CRC32, sha256, the cache, a truncated cache file, a corrupted chunk, an early end, the hash memo, loading from the cache against the local file, ranks without a model file (safetensors and five GGUF cases), a second run ships 0 bytes |

One finding from running on Linux: file time stamps on WSL2 have a coarse grain, so two writes of the same size inside a few ms keep the same mtime. A test assumed otherwise and was corrected. The hash memo of model files is keyed by size and mtime, as the operator's own model hash cache always was.

## V2. Mutation check

`experiments/session10_cuda_pair/mutate.py`. Each break edits one file, runs the session 10 tests, records the first failing test, and writes the file back.
Raw results: `mutations_round1.json` (17 breaks, 1 missed), `mutations_round2.json` (the missed one after a new assertion), `mutations_round3.json` (24 breaks on the code after the review fixes; one entry there is void, see the last row), `mutations_round3b.json` (that entry, corrected).

| break | file | first test that failed |
|---|---|---|
| clock: offset sign (local minus remote) | `clock.py` | timeline::test_estimate_is_remote_minus_local_and_uses_the_smallest_round_trip |
| clock: event times shifted the wrong way | `clock.py` | timeline::test_event_times_go_to_the_coordinator_clock_with_the_interpolated_offset |
| clock: keep the first round, not the smallest round trip | `clock.py` | timeline::test_estimate_is_remote_minus_local_and_uses_the_smallest_round_trip |
| clock: OS comparison with the hosts swapped | `clock.py` | timeline::test_compare_with_os_sync_flags_a_gap_over_two_ms |
| timesync: SNTP offset sign | `timesync.py` | timeline::test_sntp_offset_is_server_minus_local_and_delay_excludes_the_server_hold |
| recorder: the cursor does not advance | `recorder.py` | timeline::test_recorder_cursor_leaves_no_gap_inside_a_step |
| timeline: sent and received bytes swapped | `timeline.py` | MISSED in round 1 (the test checked symmetric totals only). Round 2, after an assertion on a one way link: timeline::test_recorded_spans_tile_every_step_and_bytes_equal_the_transport_counters |
| shipping: sha256 of a received slice not checked | `shipping.py` | shipping::test_a_slice_whose_sha256_differs_from_the_manifest_is_rejected |
| shipping: CRC32 of a chunk not checked | `shipping.py` | shipping::test_a_corrupted_chunk_is_rejected_and_nothing_enters_the_cache |
| shipping: a cached file accepted whatever its size | `shipping.py` | shipping::test_a_truncated_cache_file_is_requested_again |
| shipping: manifest slice bounds one short | `shipping.py` | shipping::test_manifest_lists_exactly_what_the_rank_loads_with_hashes_of_the_slices |
| shipping: row range used as a column range | `shipping.py` | same test |
| shipping: manifest hashes not checked before they name a file | `shipping.py` | shipping::test_manifest_hashes_must_be_hex_before_they_name_a_file |
| shipping: a transfer that ends early is accepted | `shipping.py` | shipping::test_a_transfer_that_ends_early_is_rejected_and_leaves_no_partial_file |
| shipping: the temporary file of a broken transfer is left | `shipping.py` | shipping::test_a_corrupted_chunk_is_rejected_and_nothing_enters_the_cache |
| shipping: hash memo kept across a change of the slicing code | `shipping.py` | shipping::test_hash_memo_is_dropped_when_the_slicing_code_changes |
| planner: coordinator host taken from the last stage | `plan.py` | coordinator::test_slots_need_no_model_and_agree_with_the_full_plan |
| operator: coordinator process started on the host of rank 1 | `operate.py` | coordinator::test_operator_starts_the_coordinator_on_the_placement_host_and_ranks_on_theirs |
| coordinator: the host check accepts any machine | `coordinate.py` | coordinator::test_coordinator_process_refuses_another_host |
| operator: model path of the operator, not of the coordinator host | `operate.py` | coordinator::test_operator_starts_the_coordinator_on_the_placement_host_and_ranks_on_theirs |
| operator: `pkill` without the option terminator | `operate.py` | same test |
| operator: `pkill` pattern not anchored at the end of the tag | `operate.py` | same test |
| rank: run tag from the wire used as a path | `hybrid.py` | timeline::test_run_tag_from_the_plan_cannot_leave_the_runs_folder |
| recorder: no compute span at the end of a step of a plain group rank | `hybrid.py` | The entry in round 3 was void: my replacement text made a syntax error, so every test failed. Corrected and run again (round 3b): timeline::test_recorded_spans_tile_every_step_and_bytes_equal_the_transport_counters |

The four breaks the prompt names are in the table: the offset sign, the hash check, the manifest slice bounds, the coordinator host lookup.

## V3. End to end on the M5 (MPS, loopback)

Command: `python -m nocturne.engines.torch_tp.hybrid_run --placement <p> --session-dir experiments/session10_cuda_pair --run-id <id> --reference <one rank run> --cache-dir <scratch cache> [--timeline] [--no-ship-weights]`.
Weights are shipped unless the row says local. The ranks got no `--model`. The slice cache was a scratch folder, deleted after the session.
References: the one rank runs of engine GGUF part 1 (`B1_q8_1.5b_1rank_r1`, `B2_q8_7b_1rank_r1`). For the 7B placements the tokens are also compared with the recorded part 1 run of the same placement file (`C1`, `C2`, `C3`).

| run | placement | weights | timeline | identical to the one rank run (8; long first divergence) | same tokens as the recorded part 1 run | frames per step measured / predicted | bytes per step, recorder / counters | shipped GB (from cache GB) | mean ms per step |
|---|---|---|---|---|---|---|---|---|---|
| V3_q8_15b_hybrid_noship_r1 | S10_q8_15b_hybrid.yaml | local | on | 8/8; 20 | - | 61.0 / 61 | 182708 / 182708 | - | 433.5 |
| V3_q8_15b_hybrid_tl_r1 | S10_q8_15b_hybrid.yaml | shipped | on | 8/8; 20 | - | 61.0 / 61 | 182708 / 182708 | 0.466 (0.950) | 434.7 |
| V3_q8_15b_pp3_tl_r1 | S10_q8_15b_pp3.yaml | shipped | on | 8/8; 256 | - | 3.0 / 3 | 6208 / 6208 | 1.177 (0.016) | 421.3 |
| V3_q8_15b_tp3_notl_r1 | S10_q8_15b_tp3.yaml | shipped | off | 8/8; 77 | - | 346.0 / 346 | - / 1058920 | 1.120 (0.000) | 420.7 |
| V3_q8_15b_tp3_notl_r2 | S10_q8_15b_tp3.yaml | shipped | off | 8/8; 77 | - | 346.0 / 346 | - / 1058920 | 0.000 (1.120) | 417.7 |
| V3_q8_15b_tp3_tl_r1 | S10_q8_15b_tp3.yaml | | | outcome: None | | | | | |
| V3_q8_15b_tp3_tl_r2 | S10_q8_15b_tp3.yaml | shipped | on | 8/8; 77 | - | 346.0 / 346 | 1058920 / 1058920 | 0.000 (1.120) | 430.6 |
| V3_q8_7b_hybrid_tl_r1 | Q8_7b_hybrid.yaml | shipped | on | 8/8; 49 | 8/8, long identical | 61.0 / 61 | 424372 / 424372 | 2.023 (4.046) | 1910.6 |
| V3_q8_7b_pp3_tl_r1 | Q8_7b_pp3.yaml | shipped | on | 8/8; 256 | 8/8, long identical | 3.0 / 3 | 14400 / 14400 | 5.285 (0.000) | 2095.8 |
| V3_q8_7b_tp3_tl_r1 | Q8_7b_tp3.yaml | shipped | on | 8/8; 162 | 8/8, long identical | 346.0 / 346 | 2459752 / 2459752 | 5.399 (0.029) | 1836.1 |
| V3post_q8_15b_tp3_notl_r3 | S10_q8_15b_tp3.yaml | shipped | off | 8/8; 77 | - | 346.0 / 346 | - / 1058920 | 0.000 (1.120) | 417.0 |
| V3post_q8_15b_tp3_notl_r4 | S10_q8_15b_tp3.yaml | shipped | off | 8/8; 77 | - | 346.0 / 346 | - / 1058920 | 0.000 (1.120) | 414.5 |
| V3post_q8_15b_tp3_tl_r3 | S10_q8_15b_tp3.yaml | shipped | on | 8/8; 77 | - | 346.0 / 346 | 1058920 / 1058920 | 0.000 (1.120) | 424.5 |
| V3post_q8_15b_tp3_tl_r4 | S10_q8_15b_tp3.yaml | shipped | on | 8/8; 77 | - | 346.0 / 346 | 1058920 / 1058920 | 0.000 (1.120) | 425.0 |
| V3post_q8_7b_hybrid_tl_r2 | Q8_7b_hybrid.yaml | shipped | on | 8/8; 49 | 8/8, long identical | 61.0 / 61 | 424372 / 424372 | 0.000 (6.070) | 1905.7 |

Reading:

- Every run: 8 of 8 eval prompts identical to the one rank run, accuracy 1.0, frames per step as predicted.
- The three 7B runs give the same tokens as the recorded part 1 runs of the same placements, the 256 token prompt included. So shipped weights, the clock exchange and the recorder changed no token.
- Recorder bytes per decode step equal the link counters in every timeline run.
- A second run of a placement ships 0 bytes. Another placement of the same model ships only what its cut changed: the 7B hybrid run found the whole second stage in the cache left by the PP run.
- Shipped bytes: the manifest bytes of a rank minus `stored_minus_loaded_bytes` equal its loaded bytes exactly, and loaded bytes equal the planned weight bytes, in every run (review finding F2 explains the difference for GGUF).
- The model sha256 computed by the coordinator equals the one from the operator's shell command, in every run (`shipping.agrees_with_preflight`).
- `V3_q8_15b_tp3_tl_r1` has no outcome. I killed it myself when I restarted the series under `nohup`. It is kept as it is.
- The `V3post` rows ran on the code after the pass 1 fixes (commit `efad24d`). The pass 2 fixes (commit `82e6b25`) touch the operator, error paths and one trace label, not the compute path; after them the full test suite and the check `K1_gtxcoord_air_pp2_7b_q8_r3` ran.

### Recorder overhead (1.5B, TP over three ranks, M5 loopback, weights from the cache)

| recorder | runs | mean ms per decode step, each run | mean |
|---|---|---|---|
| off | `V3_q8_15b_tp3_notl_r1`, `V3_q8_15b_tp3_notl_r2`, `V3post_q8_15b_tp3_notl_r3`, `V3post_q8_15b_tp3_notl_r4` | 420.7, 417.7, 417.0, 414.5 | 417.5 |
| on | `V3_q8_15b_tp3_tl_r2`, `V3post_q8_15b_tp3_tl_r3`, `V3post_q8_15b_tp3_tl_r4` | 430.6, 424.5, 425.0 | 426.7 |

The recorder costs 9.3 ms per decode step here, 2.2 percent. The spread between runs of one kind is 6.2 ms (off) and 6.1 ms (on).

The same placement with local weights before the shipping code existed (`A3_*` runs): 416.6 and 423.5 ms without, 427.2 ms with.
7B, against the part 1 runs of a week earlier (local weights, no recorder): PP 2096 against 2002 ms, TP 1836 against 1814 ms, hybrid 1911 against 1970 ms. That is inside the day to day spread, so it shows no cost that the 1.5B pairs do not show.
Tokens are identical with the recorder on and off in all pairs. Time is background (T1).

## Checks on real machines (not experiment rows)

| run | what | result |
|---|---|---|
| `K1_gtxcoord_air_pp2_7b_q8_r1` | **gtx is the coordinator and the planner** (Linux, CUDA, WSL2). The Air is a worker with shipped weights. The M5 only operates. 7B Q8_0, PP over two stages, 2 prompts | Completed. 2 of 2 identical to the MPS one rank run. The plan was made on gtx (`planner_host: Ohad`). 5.04 GB shipped over Wi-Fi in 218 s, 218 hash checks. Loaded bytes equal planned bytes. Nothing left running on gtx or the Air |
| `K1_gtxcoord_air_pp2_7b_q8_r2` | the same again | 0 bytes shipped, 5.04 GB from the Air's cache. Load time 7 s instead of 261 s |
| `K2_m5coord_gtx_air_pp3_7b_q8_r1` | The M5 is the coordinator. gtx and the Air are workers with shipped weights. PP over three stages | Completed. 2 of 2 identical. gtx received 2.48 GB in 162 s. The Air received 0 bytes: every slice of its new cut was already in its cache |
| `K1_gtxcoord_air_pp2_7b_q8_r3` | the K1 placement on the final code (after the pass 2 fixes, commit `82e6b25`) | Completed. 2 of 2 identical. 0 bytes shipped. The coordinator hashed its 218 slices again, because its hash memo is dropped when the slicing code changes; the Air's cache is keyed by content and stayed valid. Files of both hosts collected after the ranks ended. Nothing left running |

Clocks measured in these runs (engine level, control link):

| run | rank, host | wall clock minus the coordinator's | round trip | drift during the run | gap to the OS level prediction | bound of the OS level |
|---|---|---|---|---|---|---|
| K1 r1 | s1/0, Air against gtx | +24.2 ms | 5.3 ms | 14.2 ppm (0.5 ms in 35 s) | 7.7 ms, flagged (over 2 ms) | 67 ms |
| K2 r1 | s1/0, gtx against the M5 | +54.2 ms | 5.3 ms | -6.0 ppm | -2.6 ms, flagged | 58 ms |
| K2 r1 | s2/0, Air against the M5 | +82.2 ms | 7.7 ms | +6.0 ppm | 1.8 ms | 67 ms |

So between machines the engine level offset is good to about 3 to 4 ms (half the round trip) on this Wi-Fi, and the clocks really differ by tens of ms. The OS level measurement over the internet agrees inside its own bound, which is ten times wider. The M5 is the machine that is off: 59 to 147 ms behind `time.apple.com` during the morning.
The `nvidia-smi` sampler ran on gtx under WSL2 (2943 samples in K1). The `timeline.html` of K2 shows the three machines computing one after the other on one clock.

Other things first seen on real machines:

- A stale rank process from the failed R6 run (2026-11-23) was still alive on gtx and held port 50070. The cleanup `pkill` had never worked (review F1). It is fixed, and the leftover was stopped by hand.
- The Air calls itself `Mac.lan` now; peers.yaml says `ohads-MacBook-Air-2`. The operator now asks the coordinator host for its own name.

## V4. Review

- Pass 1 (`REVIEW.md`): a fresh reviewer with the diff, sections 2 to 4 of the prompt and the design document. 0 high, 2 medium, 15 low.
- Disposition is at the end of `REVIEW.md`.
- Pass 2 (`REVIEW_pass2.md`): a second fresh reviewer, the fixes only. Its disposition is at the end of that file. 0 high, 0 medium, 13 low, and of the 17 pass 1 findings 12 fixed, 4 partly fixed, 1 left on purpose.
- `REVIEW_REQUEST_CODEX.md` is ready for Codex.

## Not verified

- A CUDA pair. gtx was the only CUDA machine that answered. The pc was on the Wi-Fi at 192.168.1.239 with every port closed from the M5 and from gtx, and its join report did not reach the M5, after the join script ran there as administrator. The cause is on the pc (its firewall policy or a security client) and is open.
- `scripts/17_timesync.sh --set` and `17_timesync.ps1`: they need sudo or an administrator and were not run. The measuring mode ran on the M5, the Air and gtx.
- `powermetrics`: not used (sudo).
- A coordinator that dies or hangs on another host in the middle of a run: read, not tried.
- A rank host whose cache disk fills up during shipping: tested with a fake free space number only.
