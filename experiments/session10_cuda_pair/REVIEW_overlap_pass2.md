# Review of part 3 (overlap), pass 2: the fixes

Reviewer: a fresh reviewer. I had the two diffs, `REVIEW_overlap.md` with its disposition, `03_BUILD_OVERLAP.md`, `docs/ENGINE_DESIGN.md` sections 6 and 11, `docs/MEASUREMENT.md` section 3.1 and `CLAUDE.md`. Date: 2026-11-28.

Reviewed:

- `git diff 6ddcf01..4c0ef0f` (one commit, the fixes after pass 1).
- `git diff 1d39b45..137f921` (the MPS copy mode, a plain run folder as a reference, `overlap_table.py`).

While I reviewed, commit `cfab15e` arrived. It changes one test and one `mutate.py` entry. The engine files of `cfab15e` and `4c0ef0f` are identical, and I used the test file of `cfab15e`.
Later the working tree got uncommitted edits in `mutate.py`, `overlap_table.py`, `test_session10_overlap.py` and `test_session10_coordinator.py`. I did not review those edits. Every result below is for the files of `cfab15e`, taken with `git archive`.

How:

- Reading the diffs and the code around them.
- `tests/bench/test_session10_overlap.py` in the repository at `cfab15e`: 77 passed, 3 skipped (the CUDA test).
- Throwaway scripts on real loopback sockets and on the tiny CPU models of the tests. They are in the session scratch directory, folder `review_overlap2`.
- A differential run of my own: the engine of commit `965385f` against the engine of `4c0ef0f`, every option off, every data frame of every link compared in order with its payload.
- 36 mutations in scratch copies made with `git archive`: the 14 "review F" entries of `mutate.py` and 22 of mine.
- The recorded timelines under `experiments/session10_cuda_pair/runs/` were loaded read only with the current `timeline.py`.

Nothing in the repository was edited except this file. No `hybrid_run`, no ssh, no GPU work, no port from 50070 to 50083, no git command that changes state.

Line numbers are of the files at commit `cfab15e`.
Each line says how it was checked: **ran** (confirmed by running something) or **read** (by reading only).

## Findings of pass 1: closed or not

Count: closed 10, partly closed 5, not closed 0.

| id | verdict | checked | reason |
|---|---|---|---|
| F1 | closed | ran | `send_partial` closes a `compute` span before the hand over (`hybrid.py:332-333`). With that line removed, `test_async_send_closes_a_compute_span_before_every_hand_over` fails. The test counts spans, not time. That is enough for this defect |
| F2 | closed | ran | Every send and wait span carries a channel, and `in_flight` matches per channel. I reverted the fix faithfully (sends and waits both keyed without the channel): `test_frames_in_flight_are_matched_per_socket_in_a_tp_group_with_slots[1-False]` fails. The `mutate.py` entry for F2 keys only the sends without a channel. That breaks every link, so it does not show that the test can see the old defect. The channels are consistent for all four frame kinds on all three paths (see "Checked and found correct"). Test gap on the old path: P6 |
| F3 | closed | ran | With the `on_fail` call removed the test fails after its 20 s wait. 16 runs of the scenario on four placements, with one request and with slots: the coordinator gets `LinkLost` at once, no rank thread is left, no uncaught error in a rank thread. A failure of the coordinator's own sender also ends at once. What the record then says: P7 |
| F4 | closed as a record, policy open | ran | `pair_steps`, `single_steps` and `max_in_flight` are counted and reach the record (seen in the real record `OVc3_gtx_pp4_s3_async_l3_r1`: 0, 320, 3). Section 11.4 says how seldom two slots pair. The policy question stays with Ohad. Two notes: no test pins that the counts reach the record (P6), and `max_in_flight` counts requests, not units (P5) |
| F5 | partly | ran | The placement is still accepted, by decision. It is now marked in the plan table, in the operator's log and in the record (`planned_hosts.<host>.unchecked`), and the test fails when the mark is removed. The failure scenario of pass 1 (a hand written placement without the key, CUDA out of memory at load) is still possible, with a note in the log. The `plan` command prints the mark also for a checked host: P4 |
| F6 | partly | ran | The rule closes the two cases pass 1 named: two inline frames of a pair, and one inline frame between half the granted buffer and 1 MiB. Both `mutate.py` entries are caught. Real records confirm the grant on gtx: `socket_buffers` is `[425984, 425984]` on every link. Not closed: a frame that left on a helper thread is not counted for its pair partner, and both ranks can still block for good (P1, high). The count can also go wrong in a group of three (P2) |
| F7 | closed | ran | Both mutations of pass 1 now fail: the scheduler that waits for every head result (`test_the_scheduler_counts_pairs_singles_and_units_in_flight`), and partials sent on the compute thread although the sender is on (`test_async_send_gives_the_same_tokens_logits_and_frames[tp3-inline]`). The chunk test observes real frames (two hidden states leave before a head result comes back) |
| F8 | closed | ran | I ran `golden_frames.py` on the engine of `965385f`: all six hashes are the pinned ones. So the test compares with the old code. What it cannot see is listed under "Checked and found correct". My differential run covers that part today |
| F9 | closed | ran | `_step` masks the request number (`hybrid.py:996`). Without the mask the test fails. For request numbers under 65536 the frame is the old one |
| F10 | closed | ran | `check_overlap` rejects a chunk of 1 and a negative chunk. Without the rule the test fails |
| F11 | closed | ran | The values are checked in the argument parser of `hybrid_run` (before any folder is made), in the planner's answer (`plan_job`), in `run_job` and in `generate_many`. Removing the check from `overlap_options` or from the parser fails the test |
| F12 | closed | read | The decode rule checks the slot against the table first (`hybrid.py:1209-1210`). No test: with the guard removed all 77 tests pass (P6) |
| F13 | partly | read | The CUDA test exists and is skipped on this machine, so I could not run it. The disposition says it ran on gtx and points to `VERIFY_overlap.md`. That file does not exist in the repository. The unit test of the pinned buffer pool with a fake pinned mode, which pass 1 suggested, was not added. From the records of the real CUDA runs: `OVc3_gtx_pp4_s3_async_l3_r1` (copy mode `pinned`, 3 slots) completed and is identical to its options-off run on 8 of 8 prompts and on the three long prompts. The chunked run `OVc3_gtx_pp4_s3_c16_async_l3_r1` is identical on 8 of 8 and differs on a long prompt, which the rule of O2 allows |
| F14 | partly | read | Five of the six places are fixed. `VERIFY_overlap.md` is still missing, and now three places point to it. New mismatches: P8 |
| F15 | partly | ran | As the disposition says. The work space of group ranks doubles for a pair, and the test fails without it. The cost per process still counts only against a host budget, and the sum stays per host name. No test pins that `plan_job` and `run_job` pass the pair switch to the planner (P6) |

## New findings

Count: high 1, medium 2, low 6.

| id | severity | file and line | statement |
|---|---|---|---|
| P1 | high | `hybrid.py:349-370`, `393-394`, `transport.py:193-197` | In a pair, a frame that left on a helper thread is not counted. Its small partner then goes inline on top of it, and both ranks of a group block for good. Reproduced on real sockets with 212992 byte buffers |
| P2 | medium | `hybrid.py:348`, `355-358`, `393-394` | The ticket holds one inline amount for all links. In a group of three it is subtracted from a link that sent nothing inline. The count goes negative and keeps falling, and the rule of F6 stops working on that link |
| P3 | medium | `hybrid_run.py:495-512`, `471-475`, `overlap_table.py:82` | `compare` passes without comparing: a reference without the long prompt gives `long_identical: true`, a reference that shares no prompt gives `criterion_met: true` with 0 of 0, and then the log line raises `TypeError` after the run |
| P4 | low | `__main__.py:80`, `plan.py:570-590` | The `plan` command prints "none: the sum is not checked" for a host whose sum the planner did check |
| P5 | low | `hybrid.py:1140-1142`, `docs/ENGINE_DESIGN.md` 11.4 and 11.8 | `max_in_flight` counts requests that wait for a head result. Chunks in the pipeline are not counted, so a chunked run with one slot records 1 |
| P6 | low | `tests/bench/test_session10_overlap.py` | Twelve changes of the fixes that no test pins, and one test that writes into the repository when it fails |
| P7 | low | `hybrid.py:455-460` | After a sender failure the record says "peer closed the connection". The real error is only in the log of the rank |
| P8 | low | `docs/ENGINE_DESIGN.md`, `docs/MEASUREMENT.md`, `timeline.py` | Seven places where the documents and the code differ after the fixes |
| P9 | low | `experiments/session10_cuda_pair/overlap_table.py:99` | The gate table prints "=" between the two frames per step values whatever they are |

### P1 (high) A threaded frame is not counted for its pair partner, and both ranks can block for good

Checked: **ran** for the mechanism, on this Mac with the buffer size Linux grants. **Read** for which frame sizes hit it on gtx and pc: I could not measure the Linux kernel.

File: `nocturne/engines/torch_tp/hybrid.py:349-370` (`GroupComm.send_partial`), `393-394` (`recv_sum`), `transport.py:193-197` (`Link.fits_inline`), `docs/ENGINE_DESIGN.md` 11.4 "Socket buffers".

What is wrong: `self._out[r]` counts only bytes that were sent inline (line 357). A frame that was too large and left on a helper thread is not counted. But once that thread has handed the frame to the kernel, the frame sits unread in the same socket buffers.
Take a pair without the sender thread. A is larger than half the granted buffer, so it leaves on a thread. If A fits what the two kernels take unread, the thread ends at once. Then B is sent: `prev` is not alive (line 351), `_out` is zero, B is small, so B goes inline (line 355). If A plus B is more than the kernels take, the inline send of B blocks the compute thread. Every rank of the group sends frames of the same sizes in the same order, so every rank is blocked in a send and nobody reads. `DATA_TIMEOUT_S` is None, so the run hangs without an error.
The rule in the design text is "what it sends on its own thread must fit the kernel's send buffer without the peer reading". The inline frame fits an empty buffer. The buffer is not empty.

Evidence. Two ranks, real loopback sockets, `GroupComm` of commit `4c0ef0f`, `SOCK_BUF` set to 212992 as in the test of F6 (granted 212992, so the rule allows 106496 bytes inline). On this Mac the two kernels take 946176 bytes unread with that setting (measured three times with a non blocking sender).
Each rank: `send_partial(A)`, a pause for the compute of B, `send_partial(B)`, `recv_sum(A)`, `recv_sum(B)`.

| A bytes | B bytes | pause | A's thread when B is sent | B | result |
|---|---|---|---|---|---|
| 880000 | 100000 | 50 ms | ended | inline | both ranks blocked |
| 880000 | 100000 | 0 | alive | behind A, on a thread | passes |
| 600000 | 100000 | 50 ms | ended | inline | passes (fits) |
| 944000 | 7168 | 50 ms | ended | inline | both ranks blocked |
| 600000 | 600000 | 50 ms | ended | thread | passes (this is the case of the existing test) |

The fourth row is a prefill frame paired with one decode frame of 7B. It blocked for one of ten sizes between 942000 and 960000 bytes: the window is as wide as the small frame.

Failure scenario: gtx and pc, one TP group, `--slots 2 --tp-interleave 2` without `--async-send` (rung C4, and the "off" half of C5). Call C the bytes the two Linux kernels take unread on one link. C is at least the 425984 the socket reports. I could not measure it.
By the severity rule this is high: a hang in a supported configuration. Whether part 4 as planned meets it depends on C and on the prompt lengths, as follows.

- 7B, whole prompts. A hidden state is 7168 bytes per token, so every prompt of 30 tokens or more leaves on a thread. Each time a slot takes a new prompt while the other slot decodes, the pair is "prefill A, then a decode step B of 7 kB". It blocks when A is at most C and A plus 7 kB is more than C. The nine prompts (44 to 78 tokens, two of the same length) give eight windows of 7 kB between 315 and 566 kB. If C is above 566 kB, this case cannot happen with these prompts.
- Chunks, or a short prompt. A chunk of 32 or 48 tokens of 7B is 229 or 344 kB (thread). A remainder chunk or a prompt of 10 to 29 tokens is 72 to 208 kB (inline). The pair blocks when the sum, 301 to 552 kB, is more than C and the first frame alone is not.
- Over Wi-Fi the thread of A needs longer, so more often B finds it alive and waits behind it. That makes the hang rarer and harder to reproduce, not impossible.

With `--async-send` there is no such risk: the compute thread reads while the sender thread sends.

Suggested fix: count every frame of an all-reduce against its link until that all-reduce is received, inline or not, and keep the amount per link. In `send_partial`, add `len(wire)` to `self._out[r]` in all three branches. Put `len(wire)` into the ticket. In `recv_sum`, subtract it for every link. Then a partner of a threaded frame never goes inline: it leaves on its own thread, or behind the first one when that is still alive. With one request `_out` is zero at every send, so the default path decides as today.
I tried this in a scratch copy: all rows of the table pass, the scenario of P2 keeps every count at zero, and the test file gives 76 passed and 1 failed. The failing line is `assert comm._out == {1: 120_000}` (`test_session10_overlap.py:846`), which must become 240000 under the new count. I did not rerun the differential run on that copy.
Add the first row of the table as a test next to `test_a_pair_of_large_frames_cannot_block_both_ranks_with_small_socket_buffers`. The first frame must be sized from a measured capacity, because the capacity differs per kernel.
To know how exposed part 4 is before the fix: on gtx and pc, fill a tuned loopback socket and a socket between the two machines with a non blocking sender while the peer does not read, and note the byte count.

### P2 (medium) The inline amount of a ticket is one number for all links, so the count drifts in a group of three

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:348`, `355-358` (the count rises only on a link that sent inline), `393-394` (the amount is subtracted on every link).

What is wrong: `inline` is set when any link sent the frame inline. `recv_sum` then subtracts it from `_out[r]` for every peer, also for a peer whose frame left on a thread and added nothing. The decision is per link (`prev`, `fits_inline`), so in a group of three or more the links can differ. The count of the other link goes below zero and never comes back. `fits_inline(nbytes, outstanding)` with a negative `outstanding` allows more than half the buffer, so the protection of F6 fades on that link.
It cannot grow without bound: every add has its subtract. It can only fall.

Evidence: three ranks, real loopback sockets, 212992 byte buffers. A is 1.0 MB (more than the kernels take, so its thread lives until the peer reads), B is 100 kB. Rank 1 is fast and reads A early. Rank 2 is slow. So when rank 0 sends B, its thread to rank 1 has ended and its thread to rank 2 is alive: B goes inline to rank 1 and behind A to rank 2.

| pairs run | `_out` of rank 0 | 
|---|---|
| 1 | `{1: 0, 2: -100000}` |
| 4 | `{1: 0, 2: -400000}` |

After that `l02.fits_inline(150000, _out[2])` is True. With a correct count it is False (150000 is more than 106496).

Failure scenario: a TP group of three machines of different speed (session 11), pairs without `--async-send`, a long run. It needs A's thread to have ended towards one peer and to be alive towards the other: A is more than the kernels take unread, or it is still on its way over Wi-Fi, and one peer reads earlier than the other. The usual pair "prefill A, decode step B" then lowers the count of the slow peer's link by one decode frame, 7168 bytes for 7B. After about 15 such pairs a 315 kB prefill frame (44 tokens) goes inline on that link again, after about 50 a 559 kB frame. That is the state before the fix of F6. With two ranks there is one link and this cannot happen.

Suggested fix: the fix of P1 removes it, because the amount is then added and subtracted on every link. If P1 is fixed another way, keep the amount per link in the ticket (a dict by peer).

### P3 (medium) `compare` reports a pass for comparisons that did not happen

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid_run.py:495-512` (`compare`, commit `137f921`), `471-475` (the log line), `experiments/session10_cuda_pair/overlap_table.py:82`.

What is wrong: since `137f921` a prompt that the reference does not hold is dropped and named in `not_in_reference`. The pass values are then computed over what is left, and an empty rest passes.

- `long_identical` is `all(...)` over the long prompts that the reference holds. With none it is True.
- `criterion_met` is `sum(same) >= min(7, len(ev)) and acc == racc`. With no common eval prompt that is `0 >= 0 and None == None`: True.
- `logit_maxdiff` is then None, and `main` formats it with `:.4g` (line 474). That raises `TypeError` after the run, before the last `save()`. The record on disk then has no `correctness` at all.

Evidence, calling `compare` directly:

| reference | result |
|---|---|
| shares no prompt with the run (8 eval, 1 long) | `identical: 0, of: 0, long_identical: True, criterion_met: True, logit_maxdiff: None`; the format of line 474 raises `TypeError: unsupported format string passed to NoneType.__format__` |
| holds the 8 eval prompts, not the long one | `identical: 8, of: 8, long_identical: True, long_first_div: {}, criterion_met: True, not_in_reference: ['L']` |

The partial case is in real records already: `OVc3_gtx_pp4_off_l3_r1` against the R0 reference has `not_in_reference: ['longopen_02', 'longopen_03']`.

Failure scenario: a part 4 run is compared with an options-off run that was recorded with `--long 0`, or with fewer long prompts. `overlap_table.py:82` prints "identical" in the column "against the options-off run" for a long prompt that nobody compared. The rule of V1 is "the 8 eval prompts and the long prompt, any difference fails", so this cell is the evidence for O1, O3 and O4.

Suggested fix: when no long prompt was compared, set `long_identical` to None and print "not compared". Set `criterion_met` to False when `of` is 0 or when an eval prompt is missing. Guard the log line against None. Load every `--reference` before the run starts, so a wrong reference costs seconds and not a run. Extend `test_a_plain_run_folder_is_a_reference_and_missing_prompts_are_named` with the two rows above.

### P4 (low) The `plan` command says "not checked" for a checked host

Checked: **ran**.

File: `nocturne/engines/torch_tp/__main__.py:80` calls `format_plan(specs, cfg)` without the host budgets. `plan.py:582` then builds the host table with no budget, and line 588 prints "none: the sum is not checked".

Evidence: the interleaved placement of the tests with `host_budget_gb` for both hosts. `plan()` checks both sums (`budget: 1000000000, unchecked: False`). `format_plan(specs, d)` prints "none: the sum is not checked" for both.
The planner's answer for a real run is right: `coordinate.plan_job` passes the budgets.

Suggested fix: pass `p.host_budget_gb` in `__main__.py:80`. Better, make the argument of `format_plan` required, so the next caller cannot forget it.

### P5 (low) `max_in_flight` counts requests, not units

Checked: **read**, and the tests show it.

File: `nocturne/engines/torch_tp/hybrid.py:1140-1142`. A unit counts as flying only when it ends in a head result. A prefill chunk that is not the last is never counted.
`test_the_scheduler_counts_pairs_singles_and_units_in_flight` pins `max_in_flight == 1` for one slot with chunks of 3. `test_a_chunk_leaves_the_first_stage_before_the_head_result_of_the_prompt` shows for the same configuration that two chunks are in the pipeline at once.
`docs/ENGINE_DESIGN.md` 11.4 and 11.8 call the field "the most units in flight".

Failure scenario: a part 4 row for O2 with one slot reads `max_in_flight: 1` as "nothing overlapped".

Suggested fix: rename it in the documents to "the most requests in flight", or count chunks too (a chunk is in flight until the next unit of its slot has left the last stage, which the coordinator cannot see, so the rename is the honest choice).

### P6 (low) Changes of the fixes that no test pins

Checked: **ran**. Each line is a mutation that passes the whole of `test_session10_overlap.py`, plus `test_session10_timeline.py` or `test_session10_coordinator.py` where named.

| change removed or broken | files run | result |
|---|---|---|
| The old path (`Coordinator._step`) records the head wait on the wrong channel (`hybrid.py:1019`) | overlap, timeline | 99 passed |
| The old path records the token send on the wrong channel (`hybrid.py:1005`) | overlap, timeline | 99 passed |
| The guard of F12 in the decode rule (`hybrid.py:1210`) | overlap | 77 passed |
| The record gets the scheduler counts (`coordinate.py:133`) | overlap, coordinator | 103 passed |
| `run_job` passes the pair switch to the planner (`coordinate.py:94`) | overlap, coordinator | 103 passed |
| `plan_job` passes the pair switch to the planner (`coordinate.py:65`) | overlap, coordinator | 103 passed |
| `plan_job` gives the host budgets to the plan table (`coordinate.py:69`). Without it the planner's answer has the defect of P4 | overlap, coordinator | 103 passed |
| `recv_sum` lowers the inline count (`hybrid.py:393-394`). Without it the count only grows, and after 4 MiB on a Mac or 213 kB on Linux every frame leaves on a thread | overlap | 77 passed |
| `in_flight` reports a count mismatch on one channel (`timeline.py:188`, the term `len(s) == len(r)`) | overlap, timeline | 99 passed |
| In `inline` mode a tensor is ready at once (`hybrid.py:212-213`). With the queued copy job back, the results are the same, so only a test of when the compute thread returns could see it | overlap | 77 passed |
| The sender failure handler closes every link (`hybrid.py:459-460`). Closing only the control link passes | overlap | 77 passed |
| The two rank engine uses the granted buffer (`engine.py:103`). The old rule back passes | overlap, `test_torch_tp.py` | 86 passed |

The first two and the count matter most. No test looks at `_out` after a receive, which is why the drift of P2 is not seen either. Every options-off run with `--timeline` uses the old path, and those are the "before" runs of part 4. A wrong channel there would not be silent (`unmatched_links` would list every link), but no test would stop it. The channels are right today: the recorded options-off runs with the new column have an empty list.

One more: `test_overlap_options_are_checked_before_anything_starts` calls `hybrid_run.main` with `--run-id x` and the default session directory. Today the parser exits first. If the check is ever lost, `main` goes on and creates `experiments/session6_hybrid/runs/x/` with a log file in the repository before it fails on the missing placement.

Suggested fix: one recorded run on the old path (`generate`, `timeline=True`, `tp3` and `hybrid_tp_first`) with `unmatched_links == []`. One call of `coordinate.run_job` with slots that asserts `overlap.pair_steps` in the coordinator record and a doubled work space in `planned`. Pass `--session-dir` with a temporary folder in the options test.

### P7 (low) After a sender failure the record names a lost link, not the error

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:455-460` (`RankRuntime._sender_failed`).

The handler logs the real error on the rank and closes every link, the control link too. The coordinator then sees "peer closed the connection" and the record gets `rpc_disconnect`. In my 16 runs of the scenario of F3 that was the text every time. When the coordinator's own sender fails, the text is "[Errno 9] Bad file descriptor", "peer closed the connection" or the sender's text, by timing.
The design says so ("the rank's log names the real error"), so this is a note on the record, not a broken promise. Section 8 has a better pattern for a load failure: the rank sends its error text first.

Suggested fix: in `_sender_failed`, send one ERROR frame with the rank and the error on the control link before the links are closed, inside a `try` that ignores a failure. `Coordinator._step` and `drain` already turn an ERROR frame into `RankFailed` with the text. Wrap the body of the handler in a `try`, so that a log call that raises cannot keep the links open.

### P8 (low) Documents and code differ in seven places

Checked: **read**.

1. `experiments/session10_cuda_pair/VERIFY_overlap.md` does not exist. `docs/ENGINE_DESIGN.md:288`, `docs/MEASUREMENT.md:111` and the disposition of F13 point to it. MEASUREMENT says the verification runs and their `unmatched_links` lists are in it.
2. `docs/MEASUREMENT.md:93`: "A frame larger than 1 MiB ... is sent by a helper thread. Its `send` span is the hand over to that thread." Since the fix of F6 this holds for every frame over half the granted buffer: 212992 bytes on Linux, which is every 7B prefill of 30 tokens or more. So on gtx and pc the `send` and `allreduce_wait` spans of a TP prefill mean something else than in the part 1 recordings. The bytes on the wire are the same. A reader who compares prefill timelines across the two must know it.
3. `docs/ENGINE_DESIGN.md:393`: "A frame over 1 MiB leaves on a helper thread (section 5)". Section 5 does not say that, and the limit is no longer 1 MiB alone.
4. `docs/ENGINE_DESIGN.md:436`: "The two are matched by their order on the link ... That needs no new field." The paragraph before it introduces the channel column, which is a new field, and the match is per socket.
5. `nocturne/engines/torch_tp/timeline.py:5-6` (module text): "the k-th send on a directed link belongs to the k-th receive". The text of `in_flight` is right, the module text is the old rule.
6. `docs/ENGINE_DESIGN.md` 11.4 and 11.8: "the most units in flight" (see P5).
7. `docs/ENGINE_DESIGN.md:279` (known limits): "One request at a time per stage". That is the default since part 3, not a limit.

The text on socket buffers in 11.4 and in the engine README matches the code. It is the code that has the gap (P1).

### P9 (low) The gate table prints "=" for frames per step in every case

Checked: **read**.

File: `experiments/session10_cuda_pair/overlap_table.py:99`. The bytes column chooses between "=" and "!=". The frames column is `{a} = {b}` with a fixed sign. Two different values would print as "61.0 = 60.0".
The last column of the same row compares frames and bytes per link and prints "DIFFERENT", so the gate itself is not weakened.

Suggested fix: the same choice as in the bytes column.

## Checked and found correct

**Off means the old behaviour (the hard gate).** **Ran.** The package of commit `965385f` and the package of `4c0ef0f`, the tiny model of `golden_frames.py`, the same ports, six prompts of 2 to 40 tokens, 5 tokens each, through `generate`. Placements: PP of 3, TP of 3, hybrid. Each with float32 and with bfloat16, each with and without the recorder: 12 cases. In every case every data frame of every link is identical in kind, dtype, shape, both header fields and payload, in the same order per link. Tokens, first step logits, the decode counters per link, the frames by kind per link, the steps and the all-reduce calls are identical.
The same 12 cases again with `SOCK_BUF = 4096` in the new engine, so that the new rule sends frames on helper threads (162 frames in the TP case, 30 in the hybrid case, none in the old engine): still identical per link and in order. This is the path gtx and pc take for frames between 213 kB and 1 MiB, and it is the part of the default path that the fixes changed.
So nothing on the wire changed for the default path: `_tune` only reads two socket options more, `fits_inline` only chooses the thread, the recorder column is in the timeline file only, and the mask in `_step` changes nothing under request number 65536.
Control frames are not part of that claim. The READY and metrics replies have one more key (`socket_buffers`), as PLAN has two more since part 3.

**The golden test.** **Ran.** It is sound in what it claims: the six hashes are reproduced by the engine of `965385f`, and a bit set in a header field fails it. What it cannot see:

- The order of frames on a link, and which of two links of the same kind a frame took. It hashes the sorted frames and the frame counts by kind per link.
- Payload bytes, the index and the value of a HEAD frame.
- Control frames.
- The helper thread path. It runs on the Mac, where the buffers are 8 MiB, with frames of a few hundred bytes.
- bfloat16 as the dtype of the run, CUDA and MPS.

My differential run covers the first, second, fourth and fifth point today, for CPU. It is not a test in the repository. A second golden record made with `SOCK_BUF = 4096` would pin the thread path.

**The inline count in the other paths.** **Read**, and **ran** where named.

- One request. `_out` is zero at every send, because the receive of an all-reduce comes before the send of the next. Every link decides the same way (the links of a rank are sockets of one kernel, so they are granted the same size). The count returns to zero.
- A pair in a group of two, both frames inline, or the first inline and the second on a thread (the test of F6), or both on threads: correct. **Ran.**
- `--async-send`: the ticket carries 0 and `_out` is not touched. The compute thread reads while the sender thread sends, so nothing must fit a buffer.
- Order on a link. A frame goes inline only when no earlier thread for that link is alive. A frame behind a live thread waits for it (`_send_after`). The threads of an all-reduce are joined in its `recv_sum`. So a link never has two writers out of order. The differential run with forced threads confirms it for one request, and `test_small_socket_buffers_do_not_change_the_tokens` for pairs.
- Why "the peer answered" is a safe moment to lower the count, although the peer may not have read yet: a rank that is blocked in a send has finished its previous receive, so it has read everything of earlier all-reduces. A link towards a blocked rank holds frames of the current all-reduce only. This argument needs every frame of the current all-reduce to be counted. That is exactly what P1 breaks.
- `engine.py` (the two rank engine): one frame per all-reduce, `fits_inline(len(wire))` with nothing outstanding, the thread is joined after the receive. Correct. The mutation that puts the old rule back passes its tests (P6 table), so no test pins it.

**The sender thread and its failure.** **Ran.**

- F3 scenario, 16 runs (PP of 3 with one request and with two slots, hybrid, TP of 3): no hang, the coordinator gets `LinkLost`, every rank thread ends, no uncaught error in a rank thread.
- The coordinator's own sender fails (PP with `generate`, PP and TP with `generate_many`, three runs each): `LinkLost` at once, shutdown in under a second, no rank thread left.
- By reading: `Link.close` twice is harmless (the shutdown error is caught, `close` is idempotent). `flush`, `host` and `copy` raise after a failure and cannot wait for ever. `stop` joins for 3 s at most. A rank that waits in `select` wakes, because the socket is shut down before it is closed.
- `copy()` in `inline` mode sets `ready` at once and queues no copy job. The tensor is on the host before the call returns, the frames that carry it are queued after it, and `free` is queued after the compute thread has used it. I found no race. No `tx_wait` span is written in that mode, which matches MEASUREMENT 3.1. Putting the queued copy job back passes every test (P6 table), as it must: the results are the same.

**Channels.** **Read**, and **ran** on recorded files.

| frame | sender | receiver |
|---|---|---|
| TOKEN | `c` in `_step` and in `launch` | `c` in `serve` (stage 0, the upstream link is the control link) |
| ALLREDUCE | `g` on the compute thread and on the sender thread | `g` in `recv_sum` |
| HIDDEN | `p` on the compute thread and on the sender thread | `p` in `serve` (later stages) |
| HEAD | `c` in `_send_head` and in the sender's head job | `c` in `_step` and in `drain` |

No two sockets of one pair of ranks share a channel in one direction.
Recorded runs loaded with the current `timeline.py`, read only: 25 runs with 8 columns, 22 with 9 and 9 with 10 load. One more folder (`V3codex_q8_7b_pp3_tl_r1`) could not be read here: a file of it is empty on this disk, probably held in iCloud only. Every run with the channel column has an empty `unmatched_links`, the same list as in its record. Among them are options-off runs on the old path (`OV_15b_tp3_off_r2`, `OV_15b_hybrid_off_r1`) and CUDA runs with slots and the sender thread (`OVc3_gtx_pp4_s3_async_l3_r1`). Runs from before the column with pairs in a TP group show the unmatched link that F2 described, as they must.

**Planner and option checks.** **Ran** through the tests and mutations, **read** for the call sites. `check_overlap` runs before any folder or process in `hybrid_run`, and inside the `try` of `plan_job`, so a bad value becomes a rejected plan. `host_totals` marks a host only when it has several processes and no budget. The record fields `overlap.pair_steps`, `single_steps` and `max_in_flight` are written when the scheduler ran, and `hybrid_run` copies them.

**`load_reference`.** **Read**, and the test. A plain folder of a completed run is accepted, a run that did not complete is refused with its outcome. The defect is in `compare` (P3).

**Mutations.** **Ran**, in scratch copies only.

The 14 "review F" entries of `mutate.py`: all 14 are caught. For eight of them I ran a selection of tests, so the test named is the first failing one of that selection.

| entry | first failing test |
|---|---|
| F1 | `test_async_send_closes_a_compute_span_before_every_hand_over` |
| F2, both entries | `test_a_recorded_run_with_slots_matches_every_send_with_a_receive[False]` |
| F3 | `test_a_failure_of_the_sender_thread_ends_the_run_at_once` |
| F4, F7 (the scheduler waits) | `test_the_scheduler_counts_pairs_singles_and_units_in_flight` |
| F7 (partials on the compute thread) | `test_async_send_gives_the_same_tokens_logits_and_frames[tp3-inline]` |
| F6, both entries | `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted` |
| F8 | `test_options_off_send_the_frames_of_the_engine_before_part_3` |
| F9 | `test_the_old_path_masks_the_request_number_so_it_never_reaches_the_slot_bits` |
| F10 | `test_overlap_options_are_checked_before_anything_starts` |
| F5 | `test_a_host_with_several_rank_processes_and_no_host_budget_is_marked_unchecked` |
| F15 | `test_a_pair_plans_the_work_space_of_two_requests_for_group_ranks` |

My 22 mutations: 10 are caught, 12 are not. The 12 are the table of P6.

| mutation of mine that is caught | first failing test |
|---|---|
| F2 reverted faithfully: sends and waits both matched without the channel | `test_frames_in_flight_are_matched_per_socket_in_a_tp_group_with_slots[1-False]` |
| The sender thread records a head frame on the group channel | `test_a_recorded_run_with_slots_matches_every_send_with_a_receive[True]` |
| The sender thread records a hidden state frame on the group channel | the same test, `[True]` |
| A later stage records its upstream wait on the control channel | the same test, `[False]` |
| The scheduler (`launch`) records a token send on the group channel | the same test, `[False]` |
| Inline up to the whole granted buffer, not half | `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted` |
| Steps alone are not counted | `test_the_scheduler_counts_pairs_singles_and_units_in_flight` |
| `max_in_flight` is never raised | the same test |
| `overlap_options` does not check the values | `test_overlap_options_are_checked_before_anything_starts` |
| The argument parser of `hybrid_run` does not reject a bad value | the same test |

A mutant that survived a targeted selection of tests was run again against the whole files before I counted it as not caught.

One entry of `mutate.py` at `cfab15e` no longer applied: "Codex: the request number on the wire with the recorder off" still held the text from before the mask of F9. The uncommitted edit of `mutate.py` in the working tree corrects it.

## Not checked

- Anything on CUDA or MPS. The pinned path, the CUDA test of F13 and the MPS copy mode are judged by reading and by the records of the real runs.
- What the Linux kernels of gtx and pc take unread on one link (the number C of P1). The granted sizes are from the records, not from a measurement of mine.
- The real runs. I read records for prompt lengths, options, socket buffers, the scheduler counts and the correctness fields, and loaded the timelines. Nothing else.
- `timeline_page.html` in a browser.
- `VERIFY_overlap.md`: it does not exist yet.
- The uncommitted edits of `mutate.py`, `overlap_table.py` and the two test files that appeared during the review, and everything outside the two diffs (the dashboard, `nocturne/placement`).

## Disposition (by the author of the code, 2026-11-28)

The reviewer's text above is unchanged, except that two placeholder lines it left were replaced by what its final message said. Fixes are in the commit after `cfab15e`. Tests named here are in `tests/bench/test_session10_overlap.py` unless a file is named.

| id | decision | what was done |
|---|---|---|
| P1 | **fixed (high)** | Every frame of an all-reduce is counted against its link until that all-reduce is answered, inline or on a helper thread. A partner of a threaded frame therefore never goes inline on top of it. Test: `test_a_frame_on_a_helper_thread_counts_for_its_pair_partner` (real sockets, the capacity of the two kernels is measured first, then A just under it and B of 100 kB). `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted` now expects both frames in the count |
| P2 | fixed | The amount is added and subtracted on every link, so no link can drift. Tests: the counts are zero after the exchange in the P1 test, and `test_the_count_of_unanswered_bytes_returns_to_zero_on_every_link_of_a_group_of_three` |
| P3 | fixed | `compare`: `long_identical` is None when no long prompt was compared, `long_compared` says how many were, `criterion_met` is False when nothing was compared or an eval prompt is missing in the reference. The log line prints "not compared". `hybrid_run` loads every `--reference` before the run starts. `overlap_table.py` prints "no long prompt compared" and the number of prompts the reference lacks. Tests: `test_a_plain_run_folder_is_a_reference_and_missing_prompts_are_named`, `test_overlap_options_are_checked_before_anything_starts` |
| P4 | fixed | The `plan` command passes the host budgets. Test: `test_the_plan_command_shows_the_host_budget_it_checked` |
| P5 | renamed in the documents | "the most requests that waited for a head result at the same time". Chunks in the pipeline are not in that number, and the design says so |
| P6 | fixed, three rows left | New tests: `test_the_old_path_records_channels_that_match_on_every_link`, `test_a_head_frame_for_a_slot_outside_the_table_is_named`, `test_the_coordinator_record_holds_the_scheduler_counts_and_the_pair_work_space` (`test_session10_coordinator.py`, real processes). The options test now uses a temporary session folder. Not pinned by a test: the inline rule in the old two rank engine (`engine.py`), the redundant flush, and the queued copy job of `inline` mode |
| P7 | fixed | The failing sender sends an ERROR frame with its text on the control link before the links are closed, inside a `try`. The F3 test now asserts that the coordinator's error names `sender thread failed: ValueError` |
| P8 | fixed | All seven places. `VERIFY_overlap.md` now exists |
| P9 | fixed | The frames column chooses between "=" and "!=" |

Open after pass 2, by decision: F5 (a host with several rank processes and no host budget is marked, not rejected), F13 (the pinned pool has no unit test with a fake mode; the CUDA test and the CUDA runs cover it on a real device), F15 (the cost per process counts only against a host budget), and the pairing policy of F4, which is Ohad's call.
