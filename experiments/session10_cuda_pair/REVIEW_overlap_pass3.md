# Review of part 3 (overlap), pass 3: the fixes of pass 2

Reviewer: a fresh reviewer. Date: 2026-11-28.

Reviewed: `git diff cfab15e..c352d39`, one commit, the fixes for P1 to P9 of `REVIEW_overlap_pass2.md`.
Read first: `REVIEW_overlap_pass2.md` with its disposition, `docs/ENGINE_DESIGN.md` 11.4 and 11.5.

How:

- Reading the diff and the code around it.
- `tests/bench/test_session10_overlap.py` in the repository: 83 passed, 3 skipped (the CUDA test). The repository was at `511fd68`. Its engine and test files are those of `c352d39`.
- Throwaway scripts on real loopback sockets and on the tiny CPU model. They are in the session scratch directory, folder `review_overlap3`.
- A differential run of three packages taken with `git archive`: `965385f`, `cfab15e`, `c352d39`.
- Mutations in scratch copies only: the 15 entries of `mutate.py` named "pass 2", and 8 of mine.

Nothing in the repository was edited except this file. No `hybrid_run`, no ssh, no GPU work, no port from 50070 to 50083, no git command that changes state.
During the review the working tree got uncommitted edits in `overlap_table.py` and `docs/ENGINE_DESIGN.md`. I did not review them.

Line numbers are of the files at `c352d39`. **ran** means confirmed by running something. **read** means by reading only.

## Findings of pass 2: closed or not

Count: closed 7, partly closed 2, not closed 0.

| id | verdict | checked | reason |
|---|---|---|---|
| P1 | closed | ran, read | Every all-reduce frame is counted on its link from its send until this rank receives the answer (`hybrid.py:355-357`, `393-394`). A partner of a threaded frame no longer goes inline on top of it. No block in 331 runs on real sockets, see "Checked and found correct". The test that was added for it does not test it: Q1 |
| P2 | closed | ran | The amount is added and subtracted on every link. Three ranks of different speed, the scenario of pass 2: every count is zero after 1 and after 4 pairs. In 10 random runs with three ranks no count was negative or left above zero |
| P3 | closed | ran | `compare` gives `long_identical: None` and `long_compared: 0` when no long prompt was compared, and `criterion_met: False` when nothing was compared or an eval prompt is missing. The log line prints "not compared". Every `--reference` is loaded before the run. Three mutations of the author are caught. Edges: Q4, Q5 |
| P4 | closed | ran | `__main__.py:80` passes the host budgets. The mutation is caught by `test_the_plan_command_shows_the_host_budget_it_checked` |
| P5 | closed | read | Sections 11.4 and 11.8 now say "the most requests that waited for a head result at the same time", and that chunks are not in that number |
| P6 | closed, three rows left by decision | ran | The three new tests can fail: six mutations of the author are caught by them, three of these through real processes in `test_session10_coordinator.py`. The options test now gets a temporary session folder and asserts that it stays empty. Left without a test, as the disposition says: the inline rule of `engine.py`, the redundant flush, the queued copy job |
| P7 | partly | ran | A failing sender now sends an ERROR frame first. The record names the error only when the coordinator reads that frame before it sees a lost link: 9 of 30 runs. No hang and no frame damage. See Q2 |
| P8 | partly | read | Six of the seven places are fixed. `VERIFY_overlap.md` still does not exist, although the disposition says it does. See Q3 |
| P9 | closed | read, ran | The frames column chooses between "=" and "!=". I ran the script of `c352d39` on the recorded runs with the prefix `OV_15b`: six gate rows, all "=" and "identical" |

## New findings

Count: high 0, medium 1, low 5.

| id | severity | file and line | statement |
|---|---|---|---|
| Q1 | medium | `tests/bench/test_session10_overlap.py:989-1030`, `1033-1041` | The real socket test of P1 passes on the engine before the fix. No test pins the decision that P1 is about. The test of P2 also passes on the old engine |
| Q2 | low | `hybrid.py:455-466`, `1070-1071`, `1266-1267`, `docs/ENGINE_DESIGN.md:386` | The error text of a failing sender reaches the record only when the failed frame is the head frame of a remote head rank. In every other case the record still says "peer closed the connection" |
| Q3 | low | `docs/ENGINE_DESIGN.md:288`, the disposition of P8 | `experiments/session10_cuda_pair/VERIFY_overlap.md` is not in `c352d39`, not in `511fd68` and not on the disk |
| Q4 | low | `hybrid_run.py:309-313`, `478`, `522` | The early reference check misses two cases, and four parts of the fixes have no test |
| Q5 | low | `experiments/session10_cuda_pair/overlap_table.py:46-48` | The first comparison column still drops long prompts that the one rank reference lacks, without a word |
| Q6 | low | `transport.py:4-5`, `195`, `hybrid.py:393` | Three texts still describe the count as "sent inline", and the comment on lowering it gives a reason that is not true |

### Q1 (medium) The test of P1 passes without the fix

Checked: **ran**.

File: `tests/bench/test_session10_overlap.py:989-1001` (`_capacity`), `1004-1030` (`test_a_frame_on_a_helper_thread_counts_for_its_pair_partner`), `1033-1041` (the group of three test).

What is wrong: `_capacity` fills a socket with a non blocking sender and stops at the first refusal. On this Mac that measures the send buffer only. The receive side takes much more a moment later.

| how the capacity is measured, `SOCK_BUF` 212992 | bytes |
|---|---|
| stop at the first refusal (the `_capacity` of the test), 10 measurements | 212992 to 262144 |
| wait until nothing more goes in for 0.3 s, 11 measurements | 942080 to 946176 |

So the test sends A of about 153 kB and B of 100 kB. Together they are about 253 kB. The link takes 946 kB. On the engine of `cfab15e` B goes inline on top of A and nothing blocks.

Evidence:

- The engine of `cfab15e` with the tests of `c352d39`: the P1 test passed 8 of 8 times. A probe with the body of the test printed, for that engine: "A thread alive at B: False", "B inline", "blocked False", five of five.
- The whole fix of P1 and P2 reverted (`send_partial` and `recv_sum` of `cfab15e`): one test fails, `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted`, at the line `comm._out == {1: 240_000}`. That is the fake link with an inline first frame. With that test deselected, 82 pass. So both new socket tests pass on the old engine. The group of three test sees a count that is never lowered (a mutation of mine: lowered on the first link only, caught). It does not see the drift of P2.
- A mutation that is exactly the defect of P1 with a count that still balances (a threaded frame is counted, but the inline decision ignores it): 83 passed. Nothing pins "B does not go inline on top of a threaded A that has ended".
- The two "pass 2 P1" entries of `mutate.py` are caught, but by the count assertion of the fake link test, not by a block.

Failure scenario: someone changes the decision line (`hybrid.py:357`) or passes another amount to `fits_inline`. The count still balances, every test passes, and the hang of P1 is back on gtx and pc.

Suggested fix: test the decision without a kernel. This fails on `cfab15e` and passes on `c352d39` (ran both):

```python
link = _FakeLink(sndbuf=425984)
comm = hybrid.GroupComm(0, 2, {1: link}, "cpu")
ta = comm.send_partial(torch.ones(1, 75_000), 0)      # 300 kB: leaves on a thread
ta[2][0].join(timeout=5)                             # the thread has ended
tb = comm.send_partial(torch.ones(1, 25_000), 1)      # 100 kB
assert len(tb[2]) == 1, "the small partner of a threaded frame went inline"
```

If the socket test stays, let `_capacity` wait until nothing more goes in for a few tenths of a second. Even then the number is not sharp on macOS: on the old engine A = capacity minus 60000 blocked with a pause of 50 ms and did not block with 200 ms. So keep the fake link test as the one that must fail.

### Q2 (low) The record names the sender's error in one case only

Checked: **ran**.

File: `hybrid.py:455-466` (`_sender_failed`), `1070-1071` and `1266-1267` (a lost link becomes `LinkLost`), `docs/ENGINE_DESIGN.md:386`.

What is wrong: the ERROR frame goes to the control link of the failing rank. The coordinator reads control links only of head ranks, and only while it waits for a head result. When the failing rank is not a head rank, nobody reads the frame. When the failing rank shares a group with the coordinator's own rank, the coordinator's all-reduce sees the closed group link first.

Evidence: ranks in threads, tiny model, `--async-send`, the sender made to raise on one chosen frame, three runs each. A rank's links were closed when its main function returned, as a process exit does.

| placement, failing frame | runs in which the error was named |
|---|---|
| PP of 3, head frame of the head rank (the case of the test) | 3 of 3 |
| PP of 3, hidden state of the middle rank, one request and two slots | 0 of 6 |
| TP of 3, a partial of a remote rank, towards rank 0 and towards rank 2 | 0 of 6 |
| TP of 3 with slots and pairs, a partial of a remote rank | 0 of 3 |
| TP of 3, head frame of a remote rank | 3 of 3 |
| hybrid, a partial of a group rank | 0 of 3 |
| hybrid, hidden state of the middle rank | 0 of 3 |
| hybrid, head frame of a head rank | 3 of 3 |

In the 21 other runs the coordinator's error was "peer closed the connection" 19 times, "Connection reset by peer" once and "Bad file descriptor" once. All 30 runs ended in half a second and no rank thread was left.

Failure scenario: gtx and pc in one TP group with `--async-send`, a sender fails while it sends a partial. The record says `rpc_disconnect: peer closed the connection`, as before the fix. Section 11.5 now says "the record and the rank's log name the real error".

Suggested fix: when `generate` or `generate_many` catch a lost link, wait up to about a second on every control link, and if an ERROR frame is there, raise `RankFailed` with its text. Or change the sentence in 11.5 to say when the text arrives. Add one of the rows above to the test of F3.

### Q3 (low) `VERIFY_overlap.md` still does not exist

Checked: **read** (`git ls-tree -r c352d39`, the same for `511fd68`, and a search of the disk).

The disposition of P8 says "`VERIFY_overlap.md` now exists". It does not. `docs/ENGINE_DESIGN.md:288`, `03_BUILD_OVERLAP.md:66`, `04_EXPERIMENTS_OVERLAP.md:8` and the disposition of F13 in `REVIEW_overlap.md` point to it.
The other six places of P8 are fixed.
The results of the "pass 2" mutations are not recorded either: `mutations_overlap_round2.json` holds none of them. I ran them (see below).

Suggested fix: commit the file, or remove the pointers and correct the disposition line.

### Q4 (low) Holes in the early reference check, and parts without a test

Checked: **ran**.

File: `hybrid_run.py:309-313`, `478`, `522`.

- An empty `first_logits.npz` raises `EOFError`, which is not in `(OSError, ValueError, KeyError)`. The run still stops before it starts, with a traceback and not with the message. Pass 2 met such an empty file in a run folder on this disk.
- A `reference.json` without `prompts` passes the early check. `compare` then raises `KeyError` after the run.
- The default reference (no `--reference`) is still loaded only after the run. The placement is not known earlier, so this is a note.
- No test fails when these are removed: the term `bool(ev)` of `criterion_met` (it decides only for a run without eval prompts), the classes `ValueError` and `KeyError` of the early check, the guard of the log line (`hybrid_run.py:478`), and the `try` in `_sender_failed`.

Suggested fix: catch `Exception` in the early loop, and check there that the reference has `prompts`. Extend the two tests by one line each.

### Q5 (low) The first column of the table still compares fewer long prompts than ran, silently

Checked: **ran**, on recorded runs, read only.

File: `experiments/session10_cuda_pair/overlap_table.py:46-48` (`against`).

The fix of P3 marks missing prompts in the column "against the options-off run". The column "identical to the one rank run" does not.
`OV_15b_hybrid_all_l3_r1` ran three long prompts. Against `B1_q8_1.5b_1rank_r1` the comparison has `long_compared: 1` and `not_in_reference: ['longopen_02', 'longopen_03']`. The cell prints "8/8; 96".

Suggested fix: append the count of prompts the reference lacks, as the second column does.

### Q6 (low) Three texts are stale

Checked: **read**.

- `transport.py:195`: "`outstanding`: bytes already sent inline on this link". It now holds every unanswered frame, threaded ones too. The module text (`transport.py:4-5`) says "everything sent inline and not yet answered".
- `hybrid.py:393`: "the peer answered this all-reduce: it has read, or is reading, what was sent for it". The peer sends its answer before it reads. Lowering the count there is safe for another reason, given below. The next reader of this line should not rely on the comment.

## Checked and found correct

**P1, every path.** **Ran** and **read**.

Why it holds. Every rank of a group runs the same sequence of sends and receives. Look at the rank that is stuck earliest in that sequence.
A receive cannot be stuck for good: the peer is at the same place or later, so it has sent that frame, inline or on a thread, and a thread moves while its receiver reads.
An inline send cannot be stuck for good: the peer has finished every receive that comes before this send in the sequence. So the link holds only frames of this rank whose receive comes later. `_out` is exactly their sum, because it is raised at the send and lowered at this rank's own receive of the same all-reduce, which is the same place in the sequence for both sides. The rule keeps that sum within half the granted buffer.
So the moment of lowering is safe by position, in a group of any size. The count cannot go negative or drift: every add of `len(wire)` on a link has one subtract of the same amount on the same link. With `--async-send` the ticket carries 0 and `_out` stays empty.
Order on a link: a frame goes inline only when no earlier thread of that link is alive, a threaded frame behind a live one waits for it, and the threads of an all-reduce are joined in its receive.

| path | what happens at `c352d39` |
|---|---|
| one request | The count is zero at every send, so the decision is the one of `cfab15e` |
| a pair, both frames within half the buffer together | both inline |
| first inline, second too much | the second on a thread |
| first on a thread that has ended, second small (P1) | the second on a thread. With 8 MiB buffers it goes inline only when both fit 4 MiB |
| first on a thread that is alive | the second behind it |
| group of three, links differ | every link keeps its own count, all return to zero |
| `--async-send` | no count, the compute thread reads while the sender sends |

Runs on real loopback sockets with `GroupComm`. Each rank: send A, pause, send B, receive A, receive B. A and B carry different values and shapes, so a frame out of order fails the sum. The capacity was measured with the 0.3 s wait: 528384 bytes for `SOCK_BUF` 4096, 651264 for 65536, 946176 for 212992, 1372160 for 425984, 16763348 for 8 MiB.

| runs | result |
|---|---|
| 14 size pairs around half the buffer and around the capacity, pauses 0, 50 and 200 ms, `SOCK_BUF` 212992: 42 runs | no block, sums right, counts zero. The same script on `cfab15e`: 4 of 42 blocked, each of them the case of P1 |
| B of 7168, 28672 and 100000 bytes and of half the granted buffer, A swept from capacity minus 3 B to capacity plus 2 B, `SOCK_BUF` 65536, 212992 and 425984: 240 runs | no block, sums right, counts zero |
| 11 size pairs with 8 MiB buffers, two pauses: 22 runs | no block. B went inline on top of an ended thread only with A plus B at most 4 MiB |
| random sequences of 40 steps, pairs and single requests mixed, sizes around the limits, random pauses per rank. Two and three ranks, five buffer sizes: 23 runs, 3746 calls of `send_partial` | no block, sums right, no count negative, every count zero after every step |
| `--async-send` with real `Sender` threads, two and three ranks, `SOCK_BUF` 4096 and 212992, 16 steps: 4 runs | no block, `_out` empty |

**Off means the old behaviour (the hard gate).** **Ran.**

- The packages of `965385f` and `c352d39`, tiny model, six prompts of 2 to 40 tokens, 5 tokens each, through `generate`. PP of 3, TP of 3, hybrid, each with float32 and bfloat16, each with and without the recorder: 12 cases, 9240 data frames. Every data frame of every link is identical in kind, dtype, shape, both header fields and payload, in the same order per link. Tokens, first step logits, the counters per link, the steps and the all-reduce calls are identical.
- The same with `SOCK_BUF` 4096 and 212992 in the new engine: still identical to `965385f`.
- `cfab15e` against `c352d39`, both with `SOCK_BUF` 4096: identical frames, and the same frames leave on helper threads on the same links (216 in TP float32, 162 in TP bfloat16, 40 and 30 in the hybrid).
- `GroupComm` alone, one request at a time, 19 frame sizes across every limit, five buffer sizes, groups of two and three: 760 sends. Inline or thread, the bytes and the sums are identical between `cfab15e` and `c352d39`.
- `test_options_off_send_the_frames_of_the_engine_before_part_3` passes.

**P7, the handler.** **Read** and **ran**. The ERROR frame goes through `Link.send`, which holds the send lock of the link for the whole frame. It cannot land inside a frame that another thread writes on the control link. It is about 150 bytes on a link that carries small frames which the coordinator reads, so it does not wait. An exception from the log call or the send is caught, and the links are closed after it in every case. 30 runs: no hang, no rank thread left.

**P3.** **Ran.** Seven cases of `compare` and ten reference folders, see Q4 for the two that slip.

**Mutations.** **Ran**, in scratch copies.

The 15 entries of `mutate.py` named "pass 2": 15 caught.

| entry | first failing test |
|---|---|
| frames in flight matched per pair of ranks | `test_frames_in_flight_are_matched_per_socket_in_a_tp_group_with_slots[1-False]` |
| P1, both entries | `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted` |
| P2 | `test_a_frame_on_a_helper_thread_counts_for_its_pair_partner` |
| P3, compare, both entries | `test_a_plain_run_folder_is_a_reference_and_missing_prompts_are_named` |
| P3, the early check | `test_overlap_options_are_checked_before_anything_starts` |
| P4 | `test_the_plan_command_shows_the_host_budget_it_checked` |
| P6, channels of the old path, both entries | `test_the_old_path_records_channels_that_match_on_every_link[tp3]` |
| P6, the slot guard | `test_a_head_frame_for_a_slot_outside_the_table_is_named` |
| P6, three entries in `coordinate.py` | `test_the_coordinator_record_holds_the_scheduler_counts_and_the_pair_work_space` |
| P7 | `test_a_failure_of_the_sender_thread_ends_the_run_at_once` |

Mine: 3 caught, 5 not caught.

| mutation | result |
|---|---|
| The fix of P1 and P2 reverted in full | caught, by the fake link test only (Q1) |
| A threaded frame is counted, but the inline decision ignores it | not caught, 83 passed (Q1) |
| The count is lowered on the first link only | caught, the group of three test |
| `criterion_met` without the missing eval prompt term | caught |
| `criterion_met` without `bool(ev)` | not caught (Q4) |
| The early check catches `OSError` only | not caught (Q4) |
| The log line formats None | not caught (Q4) |
| No `try` around the log call and the ERROR frame | not caught (Q4) |

## Not checked

- What the Linux kernels of gtx and pc take unread on one link. The rule assumes that half of the size Linux reports is always usable. It counts payload bytes, and a frame has 24 more bytes of header. On this Mac the margin is large. On Linux I could not measure it.
- Anything on CUDA or MPS, and the real runs.
- `test_session10_coordinator.py` as a whole. It ran inside the three mutation runs up to the new test, which failed there as it should.
- The uncommitted edits of `overlap_table.py` and `docs/ENGINE_DESIGN.md` that appeared during the review, commit `511fd68` (a placement file), and everything outside the diff.

## Disposition (by the author of the code, 2026-11-28)

The reviewer's text above is unchanged. The test, table and document changes are in `be73eed`; the comments and the early reference check are in the commit after it.

| id | decision | what was done |
|---|---|---|
| Q1 | fixed | Two tests without a kernel: `test_the_small_partner_of_a_threaded_frame_that_has_ended_does_not_go_inline` (the reviewer's test) and `test_in_a_group_of_three_every_link_keeps_its_own_count`. The socket test now measures the capacity until nothing more goes in for 0.4 s. All three fail on the engine of `cfab15e` and pass on `c352d39` (ran both) |
| Q2 | documented | Section 11.5 of the design now says when the record names the sender's error (the head frame of a head rank) and that otherwise the record says a link was lost, with the error in the rank's log. Waiting on every control link after a lost link was not built |
| Q3 | fixed | `VERIFY_overlap.md` is in the repository, and the results of the "pass 2" mutations are in `mutations_overlap_round3.json` and `round3b.json` |
| Q4 | fixed | The early check catches every exception and refuses a reference without prompts. The options test has the two cases. Not pinned by a test: `bool(ev)` in `criterion_met`, the guard of the log line, the `try` in `_sender_failed` |
| Q5 | fixed | The first comparison column names the prompts the one rank reference lacks |
| Q6 | fixed | The three texts say what the count holds, and the comment gives the reviewer's reason for the moment of lowering |
