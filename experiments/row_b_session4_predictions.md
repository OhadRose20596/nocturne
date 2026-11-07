# Row B, session 4: pre-registered predictions

Written 2026-11-02 before any Part 1 run. Ohad's predictions come from the session 4 brief. Claude's come from the
Part 0 analysis of the session 3 records (`nocturne/bench/results/derived_v3.json`). Outcomes are filled in after
each part, with the record ids. Setup: M5 requester, Air worker, home Wi-Fi, about 8 ms RTT, Qwen2.5-7B unless said.

## Part 1. Diagnostics

| item | measurement | Ohad | Claude | outcome |
|---|---|---|---|---|
| 1.1 | exchange microbenchmark, 56 frames of 7168 B per step, 300 steps, Wi-Fi, thread mode | 8 to 10 ms per exchange; then the medium is the floor | 6 to 9 ms per exchange. Loopback is 0.06 ms, so the thread per send costs nothing measurable; the gap to the 14.6 ms per sync of B1t is local work around the exchange (MPS sync, copies) plus the 30 to 80 ms stalls | 2026-11-02, AWDL up, the torch BASELINE was running on the M5 at the same time. thread: mean 7.55 ms, p50 6.13, p95 14.0, p99 94; step of 56 exchanges 423 ms mean; 284 of 16800 exchanges over 30 ms holding 19.6 percent of the time. inline: mean 7.56, p50 6.03, p95 14.6; 334 stalls, 22.7 percent. Both predictions hold on the mean (Ohad's on p95, Claude's on p50 and mean). Thread against inline differ by 0.01 ms: the thread per send costs nothing. The stall share of the bare transport equals the 22 percent seen in B1t and B1 decode, so the stalls are the medium, not the engine. The bare exchange explains 423 of the 830 ms per B1t step; the other 400 ms is compute (about 70) plus local work around each exchange (about 330, 6 ms per all-reduce), to be split by the trace of item 1.2. Files: experiments/row_b_session4/microbench_wifi_thread.json, _inline.json |
| 1.2 | all-reduce trace in B1t, fixed 256 tokens, eval_long | not stated | to_cpu (the MPS sync) 1 to 3 ms, recv wait 7 to 10 ms, to_device plus add under 1 ms; total per all-reduce 11 to 14 ms | 2026-11-02, `B1t_wifi_qwen2.5-7b-bf16_m5_r7`, 3 prompts of 256 tokens (the full run was cut short for time), AWDL up, 43400 all-reduces over 775 steps. Per all-reduce, mean: to_cpu 2.47 ms, send_queued 0.14, recv_done 9.33, to_device 0.74, add 0.49 (with a device sync), total 13.16. Prediction holds on every stage. 737 of the 779 ms per step are inside all_reduce, so compute outside it is about 42 ms per step. Local work around the wire is 3.9 ms per all-reduce (218 ms per step); the recv wait of 9.3 ms against the bare exchange of 7.6 ms leaves about 1.7 ms of peer skew per all-reduce. Part 2 targets: to_cpu (2.5 ms, the MPS sync plus the numpy copy) first, then to_device. Side finding: with 256 tokens the two-rank text diverges from the one-rank BASELINE on 3 of 3 prompts, at words 19, 107 and 151, inside normal prose (session 3 saw 31 of 33 identical on short prompts). This is the bf16 partial sum question of Part 7; identity over long decodes is now a measured failure, not a pass. Ping during decode: p50 9.5 ms, p95 84, p99 103, loss 8.6 percent at 10 ms interval. |
| 1.3 | AWDL down on both Macs, B1t and A1, fixed 256, 3 reps each | the 524 ms silences disappear; ms per step drops 15 to 25 percent in both | the 524.3 ms phase lock of the stall starts disappears; B1t ms per step drops 15 to 20 percent, A1 drops 5 to 15 percent (A1 has one blocking wait per step, so fewer chances to be hit) |A1 part only, 2026-11-02, on the phone hotspot (not home Wi-Fi; the network changed mid-session, ids `_hotspot_`). Controls with AWDL up on the hotspot: A1 r5 to r7 at 65.0, 65.5, 65.0 ms per step, ping in decode p50 7.7, p95 11.6, p99 14 ms. AWDL down on both Macs: r8 63.9 (p95 10.5, p99 12.7), r9 67.0 (p95 12.4, p99 36.5). macOS brought awdl0 back up on both Macs by itself within about 10 minutes, so r10 (66.1) ran with AWDL up again and the record says so. Outcome for A1: no change (65.5 against 65.2, inside rep noise), both predictions fail on this network. Caveat: the hotspot tail is already short (p95 12 ms against 75 to 84 at home), so the test has little to detect here; the home Wi-Fi test is still open. B1t with AWDL down (3 reps, about 108 min) not run for time. Phase lock of stall starts not analysed yet. For the next attempt keep AWDL down with a loop (`while true; do sudo ifconfig awdl0 down; sleep 5; done`) or disable AirDrop and Handoff first. B1t with AWDL up on the hotspot for reference: `B1t_hotspot_qwen2.5-7b-bf16_m5_r2` 702.7 ms per step (home 779 to 865), ping p50 9.0, p95 12.8, p99 15.3; r1 excluded (9 minute link drop). B1t part, 2026-11-07, home Wi-Fi, AWDL held down on both Macs by a root loop (`sudo sh -c 'while true; do ifconfig awdl0 down; sleep 5; done'`), sampled inactive on the M5 every 10 minutes for the whole block. `B1t_wifi_qwen2.5-7b-bf16_m5_r8`, `r9`, `r10`: 602.0, 586.8, 622.2 ms per step pooled (mean 603.7), per prompt 560 to 769. Against the AWDL up home reference r7 (779.4, 2026-11-02) that is a 22.5 percent drop; against session 3 (830 to 865) 27 to 30 percent. Ohad's 15 to 25 percent holds; Claude's 15 to 20 percent fails low. Ping in decode: p50 8.1 (r7 9.5), p95 11.0 to 17.6 (r7 83.5), p99 32 to 51 (r7 103). Share of pings above 30 ms: 1.1 to 2.0 percent against 17.6 percent with AWDL up; above 100 ms: 0.16 to 0.25 percent against 1.6 percent; pings above 450 ms: none in 442000 against 3 in 58000. Ping loss did not change (6.5 to 9.2 percent against 8.65), so the loss is not AWDL. The 524 ms phase lock was not re-measured from stall starts; the ping tail that carried it is gone. Exact match against the one-rank BASELINE 0 of 10 in all three reps, first divergence median 70.5 in all three (hotspot gave 70): divergence is deterministic and independent of the network, which is the Part 7 question. Combined with the hotspot A1 result, the AWDL cost falls on the engine with 56 blocking waits per step, not on the one with one. |
| 1.4 | B1 graph metadata on Qwen2.5-1.5B Q4_K_M (28 layers, hidden 1536) and 0.5B (24 layers, hidden 896), fixed 64 | grows with layer count, not hidden size; faster than linear | 1.5B gives the same n_tensors sequence and the same 16.6 MB per step as 7B (same depth); 0.5B gives about 24/28 squared, about 12.4 MB, so near quadratic in layers; hidden size changes only the payload bytes |2026-11-07, home Wi-Fi, AWDL up, `B1_wifi_qwen2.5-1.5b-q4km_m5_r1` and `B1_wifi_qwen2.5-0.5b-q4km_m5_r1`, 10 prompts of 64 tokens each. graph_meta_bytes_per_step: 1.5B 16.64 MB (232 messages, 57 blocking syncs, 1480 ms per step); 0.5B 13.87 MB (212 messages, 49 syncs, 1221 ms per step); 7B from 2026-11-02 16.6 MB, 232 messages. Depth, not hidden size: holds exactly (1.5B equals 7B to three digits). Growth with layers: 0.5B is 0.834 of the 28 layer value, between linear (0.857) and quadratic (0.735); a two point fit gives about 480 KB per layer plus an ancestor-walk term of 3.2 MB at 28 layers. Ohad's 'faster than linear' holds weakly; Claude's 'near quadratic' fails, the per-layer constant dominates. Side result for Q7: a 0.5B model decodes at 1.2 s per step in this mode, so B1 step time is metadata-bound and nearly independent of model size. Ping in decode p50 12.3 to 12.6 ms, p95 85 to 87 ms. No one-rank baseline for these models, so exact match is null. |

## Part 2. Transport and engine cleanup

| item | Ohad | Claude | outcome |
|---|---|---|---|
| 1.5B loopback ms per step | from about 53 to under 40 | to 40 to 45: loopback is 56 times (to_cpu plus to_device) and the exchange is already 0.06 ms | 2026-11-07, loopback on the M5, eval_long, fixed 64 tokens, 3 reps each. Before (engine as in session 3): `B1t_wifi_qwen2.5-1.5b-bf16_m5_loopback_r2` to `_r4` 66.7, 65.9, 66.7 ms per step pooled (the 53 of session 3 was eval_set without fixed tokens, so the before point was remeasured). After (zero copy wire view, receive into a reused buffer, inline send for frames under 1 MiB, host side add and one upload): `_r5` to `_r7` 64.1, 61.7, 61.6; mean 62.5 against 66.4, a 6 percent gain, outputs identical on 10 of 10 prompts. Both predictions fail: the copies were never the cost. Traced rep `_r8` (61.9 ms per step): 0.92 ms per all-reduce, of which to_cpu 0.65 (the download, which waits for the device), send 0.03, recv 0.09, host add plus upload 0.15, device sync 0.003; 51.7 of the 62 ms are inside all_reduce. A standalone probe on the idle M5 gives 0.4 ms for a small matmul plus one synchronize and 0.2 ms for a 7 KB upload plus synchronize, so about 0.6 ms of every all-reduce is the MPS command buffer round trip (commit, wait, download, upload), not data movement. 56 of those per step put the loopback floor near 34 ms plus compute; the remaining lever is fewer device waits per step, not cheaper ones. |
| 7B Wi-Fi ms per step | down by 2 to 3 ms per all-reduce, so about 100 to 170 ms per step | down 2 to 4 ms per all-reduce, so 600 to 700 ms per step with AWDL up; the stalls are the larger term and belong to item 1.3 | |

## Part 3. Tokens per pass (B4t)

| item | Ohad | Claude | outcome |
|---|---|---|---|
| step time for T = 1, 2, 4, 8 rows | flat within 10 percent | flat within 10 percent up to T = 4; T = 8 within 20 percent (the frame grows to 57 KB, still small) | |
| k = 8 aggregate | 6 to 8 times B1t | 6 to 7.5 times | |

## Part 4. llama.cpp graph metadata

| item | Ohad | Claude | outcome |
|---|---|---|---|
| patched build, Wi-Fi | metadata 16.6 MB to under 2 MB per step; 1500 to 900 or 1100 ms per step | metadata under 1 MB if the cache keys on uid; the step time falls less than the bytes, to 1100 to 1300 ms, because the 57 blocking gets stay | |
| loopback | from about 86 to under 60 ms per step | 55 to 70 ms | |

## Part 5. Delay sweep

| engine | Ohad slope (ms per step per ms one way) | Claude | outcome |
|---|---|---|---|
| B1t | about 57 | 57 to 60 | |
| B1 | about 114 | 114 plus the sets that wait behind a full socket buffer, up to 120 | |
| A1 | about 2 | 2 | |

## Part 6. 32B memory

Both: weights split about equally; per node totals within 10 percent between TP and balanced PP. Claude adds: TP holds
the full output layer plus the 292 MiB CPU buffer on the requester, so the requester side is 0.3 to 0.6 GB heavier in TP.

## Part 7. Precision control

Both: fp32 reduce gives 33 of 33 identical; the simulated world 1 split reproduces the same two divergent prompts.
