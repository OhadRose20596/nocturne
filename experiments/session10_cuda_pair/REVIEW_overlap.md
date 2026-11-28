# Review of part 3 (overlap), pass 1

Reviewer: a fresh reviewer with the diff, the task (`03_BUILD_OVERLAP.md`), `docs/ENGINE_DESIGN.md` sections 6 and 11, `docs/MEASUREMENT.md` section 3.1 and `CLAUDE.md`. Date: 2026-11-28.

Reviewed: `git diff 965385f..1d39b45` (commits fd9b098, ad58799, 1d39b45).

How:

- Reading the diff and the code around it.
- `tests/bench/test_session10_overlap.py` at the commit: 58 passed.
- Throwaway scripts on the tiny CPU models of the tests, loopback, ephemeral ports. They are in the session scratch directory, folder `review_overlap`.
- A differential run: the engine of commit 965385f against the engine of 1d39b45, every option off, every frame of every link compared in order.
- The 24 part 3 entries of `mutate.py`, and eight mutations of mine, run in a scratch copy of the commit. Nothing was mutated in the repository.

Nothing in the repository was edited except this file. No `hybrid_run`, no ssh, no GPU work, no port from 50070 to 50083.

Line numbers are of the files at commit 1d39b45. The working tree changed while I reviewed (about 12:12 and 12:22): the MPS copy mode became `inline`, `docs/MEASUREMENT.md` 3.1 got a list of limits, `hybrid_run` learned to use a plain run folder as a reference. In `hybrid.py` the working tree was then one line further down after line 167. Where a finding is touched by those edits I say so.
By the end of the review more files had uncommitted edits (`coordinate.py`, `plan.py`, `recorder.py`, `timeline.py`, `transport.py`, `engine.py`, a new `tests/bench/golden_frames.py`). I did not review those edits. Some findings may already be answered by them. Those edits came at 12:38 and later. Every run of mine that is quoted below is older or used a scratch copy made with `git archive 1d39b45`. So it ran the code of the commit, or the commit plus the two small edits of 12:22 named above, which do not touch a CPU run.

Each finding says how it was checked: **ran** (confirmed by running something) or **read** (by reading only).

## Findings

Count: high 0, medium 7, low 8.

Summary. The hard gate holds: with every option off the new code sends the frames of the old code, byte for byte, per link and in order (24 cases, see the last section). I found no path that changes tokens, mixes slots, or changes the add order of the all-reduce. Tokens were identical to the one request path in every configuration I ran, also with EOS, three and four slots, pairs, chunks, the sender thread and helper threads for every frame.
The medium findings are in the measurement (two), in one error path, in what `--tp-interleave 2` does with two slots, in two missing checks, and in one test gap. F6 can become high: it depends on a fact about the Linux hosts that I could not measure from here.

| id | severity | file and line | statement |
|---|---|---|---|
| F1 | medium | `hybrid.py:323-327` | With `--async-send` a rank of a TP group records its compute time as `allreduce_wait`. Its busy share drops from about 0.5 to 0.04 for the same work |
| F2 | medium | `timeline.py:162-183` | `in_flight` matches sends and receives per pair of ranks, but two ranks can have two sockets. In a pure TP placement with slots the frames to the coordinator are matched wrongly |
| F3 | medium | `hybrid.py:184-186`, `291-292`, `644-658` | A failure of the sender thread that is not a closed link is never seen by a rank that waits on its upstream link. The run stalls until the 1800 s control timeout |
| F4 | medium | `hybrid.py:1160-1222` | With `--tp-interleave 2` a pair forms only when two slots are ready at the same instant. With two slots in a hybrid placement almost every step runs alone, and the record does not show it |
| F5 | medium | `plan.py:366-380`, `465-472` | The per host sum is checked only for a host that `host_budget_gb` names. A host in several stages without it is accepted with no check and no warning |
| F6 | medium, high if confirmed on gtx or pc | `hybrid.py:338-346`, `transport.py:43-44`, `97-103` | A pair sends two inline frames each way before any receive. That is safe only if the kernel granted the 8 MiB socket buffers. The grant is never read back, and Linux caps the request silently |
| F7 | medium | `tests/bench/test_session10_overlap.py:143-153`, `242-255` | No test shows that two requests are in flight at once. A scheduler that runs one request at a time passes every test |
| F8 | low | `tests/bench/test_session10_overlap.py:156-179` | The gate "off means the old bytes" is tested against `generate` of the same code, not against the old code, and without link or order |
| F9 | low | `hybrid.py:969` | The old path writes the request number without a mask. With `--timeline`, request 65536 sets the slot bits and the step fails with "slot 1 of 1" |
| F10 | low | `plan.py:523-533`, `hybrid_run.py:296` | `--prefill-chunk 1` is accepted and gives one token chunks, which the counters take for decode steps |
| F11 | low | `hybrid.py:1056-1059`, `408-409`, `coordinate.py:194-199` | Wrong option values are rejected late: after every rank has loaded, or inside a rank |
| F12 | low | `hybrid.py:1177`, `1182-1185` | A head frame with a slot beyond the table raises `IndexError` in the decode rule, before the check that names the rank and the slot |
| F13 | low | `tests/bench/test_session10_overlap.py:395-430`, `hybrid.py:194-209` | The `pinned` copy path (CUDA) and a copy on a real device are not run by any test. The MPS failure of `thread` mode could only be found by a real run |
| F14 | low | `docs/ENGINE_DESIGN.md` 11, `docs/MEASUREMENT.md` 3.1 | Six places where the documents and the code differ |
| F15 | low | `plan.py:26`, `309-322`, `366-380` | Planner details: a pair holds two prefill work spaces at once, the cost per process counts only against a host budget, and the sum is per host name, not per device |

### F1 (medium) With `--async-send` a TP rank's compute is recorded as all-reduce wait

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:323-327` (`GroupComm.send_partial`, the branch for the sender thread).

What is wrong: the branch without the sender closes the compute span before the copy (`rec.lap("compute")`, line 330). The branch with the sender hands the tensor over and returns without any lap. The next lap is `allreduce_wait` in `recv_sum` (line 376). So everything since the previous lap goes into `allreduce_wait`: the attention or MLP partial of this request, and in a pair also the partial of the other request.
A pipeline rank is not affected: `_finish` closes its compute span (line 513).

Evidence: `tp3` on the tiny model, CPU, loopback, 5 prompts, 12 tokens, the same work twice.

| rank | option | compute ms | allreduce_wait ms | busy share |
|---|---|---|---|---|
| s0/0 | off | 122.0 | 63.4 | 0.519 |
| s0/0 | `--async-send` | 12.9 | 281.1 | 0.040 |
| s0/1 | off | 106.1 | 76.0 | 0.452 |
| s0/1 | `--async-send` | 12.1 | 275.8 | 0.038 |

On a CPU the compute runs on the compute thread, so the true busy share did not change. The record says it fell by a factor of 13.

Failure scenario: part 4 rung C5 ("O4 on and off, TP interleave 2", with `--timeline`). The table for Chen would show the busy share of both GPUs falling to a few percent when `--async-send` is turned on. The cause is the missing lap, not the option. On MPS (now `inline`) the copy waits for the device on the compute thread, and that device time also lands in `allreduce_wait`. `docs/MEASUREMENT.md` 3.1 (working tree) says it is inside `compute`. That is true for a pipeline rank only.
The test `test_a_recorded_run_with_slots_matches_every_send_with_a_receive` runs this case but only asserts `0 <= overlap_share <= busy_share <= 1`.

Suggested fix: at the top of the sender branch, `if rec: rec.lap("compute")`. I tried this line in a scratch copy of the commit: the busy share of s0/0 is 0.461 with the option against 0.478 without, and the whole test file still passes (mutation list at the end). Add a test that compares the compute time of a TP rank with and without the option on CPU.

### F2 (medium) Frames in flight are matched per pair of ranks, not per socket

Checked: **ran**.

File: `nocturne/engines/torch_tp/timeline.py:162-183` (`in_flight`), `docs/MEASUREMENT.md` 3.1 ("The k-th send on a directed link is matched with the k-th receive on it").

What is wrong: the rule is true for one socket. The code applies it to a pair of rank names. A rank of the first stage that also holds head rows has two sockets to the coordinator: the group link (all-reduce frames) and the control link (head frames). That is every remote rank of a pure TP placement. With one request both sides use the two sockets in the same order. With slots they do not: the rank sends the head result of request A and then the partials of request B, while the coordinator reads the partials of B inside its step and the head of A only after it.

Evidence: `tp3`, slots 2, no interleave, 12 tokens: `unmatched_links` is `['s0/1 -> s0/0', 's0/2 -> s0/0']`. Counts agree (600 and 600), the bytes do not: 58 of 600 frames of `s0/1 -> s0/0` are matched to a receive of another size, the first one a head frame of 212 bytes against an all-reduce receive of 536 bytes. The same with slots 3 and interleave 2 (54 of 600), with and without the sender thread. `hybrid_tp_first`, `hybrid_pp_first` and `pp3` give an empty list, because there the head ranks are not group peers of the coordinator.

Why it matters: the intervals of those frames are wrong, so `overlap_share`, `overlap_out_share` and `overlap_in_share` of the ranks of a pure TP run with slots are computed from wrong pairs. This is rung C4 of part 4. The record does say so (`unmatched_links` is not empty), but the numbers stay in the record and in `timeline.html`.
MEASUREMENT says the list was "empty in every verified run". No recorded check run could have shown it: at the time of the review the runs with slots (`OVc_pair_tp2_pair_r1`, `OVc_pair_tp2_pair_async_r1`, `OVs_15b_tp3_s2_pair_async_r1`) have no timeline in their record.
The only test of the list uses `hybrid_tp_first` (line 623).

Suggested fix: match per socket. The receive side already tells the two apart (`allreduce_wait` against `recv_wait`). The send side does not. Give the send span the channel, for example a tenth column with `grp`, `pipe` or `ctl` (old files load without it, as with the slot column), and key `in_flight` by (from, to, channel). Add `tp3` with slots 2 to the test. Until then, treat a run with a non empty `unmatched_links` as having no overlap share.

### F3 (medium) A sender failure that is not a link failure stalls the run for 30 minutes

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:291-292` (the sender stores the error), `184-186` (`check`), `644-658` (`serve` blocks on the upstream link), `docs/ENGINE_DESIGN.md` 11.5 ("A failure in the sender ... is raised on the compute thread at its next call").

What is wrong: the compute thread learns of the error only when it calls the sender again (`copy`, `host`, `flush`). A rank that has handed over its last frame goes back to `up.recv()` and makes no such call. When the link itself broke, the other ranks notice and the run ends. When the error is something else, nothing breaks: the frame never leaves, the rank waits for its next input, and the coordinator waits for the head result.

Evidence: `pp3`, a head rank whose logits are all NaN (no maximum, so the sender raises `ValueError` at line 270). I set `CTL_TIMEOUT_S` to 8 s for the check.

| option | what the coordinator got | after | head rank |
|---|---|---|---|
| off | `LinkLost: peer closed the connection` (the rank raised `IndexError` in `head_argmax`) | 0.0 s | gone |
| `--async-send` | `LinkLost: timed out` | 8.0 s, which is `CTL_TIMEOUT_S`: 1800 s in a real run | still alive, blocked on its upstream link |

With several slots the wait is in `drain(block=True)` and ends with `TimeoutError` after the same 1800 s. The 300 s stall check of `hybrid_run` does not help, because the coordinator keeps its heartbeat.
Other errors of this kind: a CUDA error in `event.synchronize()`, a device error in the copy of `thread` mode, an index over 16.7 million.
A second effect: `check` turns every sender error into `ConnectionError`. So `rank_main` logs "link lost" and the coordinator records `rpc_disconnect` for what was a `ValueError`.

Suggested fix: when the sender thread fails, let it close the rank's links (at least the upstream link and the control link) after it stored the error. The blocked `recv` then raises at once, on this rank and on its peers. Keep the original exception type in the log. Add the NaN case to the tests with a short timeout.

### F4 (medium) With two slots a pair seldom forms after the first step, and nothing records it

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:1199-1222` (the scheduler loop), `1160-1194` (`drain`, which returns after the first head results), `coordinate.py:93-97` (the record).

What is wrong: a pair needs two slots in `ready` at the same moment (line 1201). A slot becomes ready when its head result is read. `drain(block=True)` returns as soon as one result came and nothing else is waiting on the sockets (line 1194). Then one slot is ready and its step starts alone (line 1217). From then on the two requests are out of phase and stay so. Nothing waits for the partner.
This follows the design text (11.4: "With one slot ready, the step runs alone"). The effect on what the option measures is not in the design.

Evidence: loopback, CPU, 40 tokens per request, counting the coordinator's `step_pair` and `step` calls.

| placement | slots | sender thread | steps in pairs | steps alone |
|---|---|---|---|---|
| TP of 2 | 2 | off | 76 | 4 |
| TP of 3 | 2 | off | 78 | 2 |
| TP of 3 | 2 | on | 54 | 26 |
| hybrid (TP of 2, one rank, head group) | 2 | off | 12 | 68 |
| hybrid (the same) | 2 | on | 2 | 78 |
| all of the above | 4 | off | 160 | 0 |

In a pure TP group the two head results arrive together, so pairs mostly hold. Behind a later stage they arrive one step apart, and the pair is lost after the first step. With four slots two are always ready.

Failure scenario: a part 4 row "hybrid, `--slots 2 --tp-interleave 2`" is read as "two requests in lockstep". In fact about one step in seven ran as a pair. The record has `overlap.tp_interleave: 2` and no count of pairs. The timing in real runs will differ from loopback, so the share must be measured, not assumed.
The test `test_a_pair_is_sent_before_it_is_received...` asserts only that the pattern "ssrr" occurs once (line 312). The first step always gives it.

Suggested fix: count paired and single steps in `generate_many` and write both to `record.json` (`overlap.pairs`, `overlap.singles`). Then decide the policy with Ohad: either keep it and say in 11.4 that pairs need more slots than two, or let the scheduler wait for the partner when interleave is on, exactly one slot is ready and another slot has a head result pending.

### F5 (medium) Without `host_budget_gb` the per host sum is not checked

Checked: **ran**.

File: `nocturne/engines/torch_tp/plan.py:366-380` (`host_totals`), `465-472` (the check), `docs/ENGINE_DESIGN.md` 11.6.

What is wrong: the task says the planner "must count the memory of all rank processes of one host against that host's budget". The sum is always computed. It is compared only when the placement has `host_budget_gb` for that host (line 466). A placement that puts a host into several stages and leaves the key out passes with no check, no warning, and no line in the plan table that says "unchecked". The cost per process (150 MiB on CUDA) then counts against nothing, because rank budgets do not include it.

Evidence: the four stage placement of the tests on `cuda`, two hosts, no `host_budget_gb`: `plan` accepts it, `host_totals` shows `budget: None` for both hosts. The same placement with a rank budget of 1.76 MB per rank (each rank fits alone) and a host total of 318 MB: accepted. With a host budget one byte under the total: rejected, as it should be.
The test asserts `ht["h"]["budget"] is None` for this case (line 371), so the gap is pinned as intended behaviour.
All six placements of part 3 with a host in several stages do set the key. So no run so far is affected.

Failure scenario: session 11, four machines, a new interleaved placement written by hand without the key. Two rank processes of 4 GB each go to one 8 GB card. The planner says yes. The second process fails at load with CUDA out of memory, after the first one has loaded.

Suggested fix: raise `PlanError` when a host holds ranks of more than one stage and has no `host_budget_gb` (old TP placements with several ranks of one stage on one host keep working). Or take the sum of the rank budgets of the host as the default host budget. Print the per host table with the word "unchecked" when no budget applies.

### F6 (medium, high if confirmed on gtx or pc) A pair doubles what must fit into the socket buffers, and the buffer size is assumed

Checked: **ran** for the mechanism and for the Macs. **Read** for Linux: I could not measure gtx or pc.

File: `nocturne/engines/torch_tp/hybrid.py:338-346` (frames up to `SMALL_FRAME` are sent inline), `transport.py:43-44` (`SOCK_BUF = 8 << 20`, `SMALL_FRAME = 1 << 20`), `transport.py:97-103` (`_tune` ignores a refusal and never reads the result back), the module text of `transport.py` ("both kernels buffer SOCK_BUF, so two such frames in flight cannot deadlock").

What is wrong: without the sender thread, a rank in a pair sends A's partial inline, computes B's, sends B's inline, and only then receives. Every rank of the group does the same. So up to two frames of up to 1 MiB each must sit in the kernel buffers of each direction before anyone reads. Before part 3 it was one frame. This is safe when the kernel really gives 8 MiB. Nothing checks that.

Evidence:

- This Mac: after `_tune`, `SO_SNDBUF` and `SO_RCVBUF` read back as 8388608. With that, pairs of frames up to 1000 KiB pass. The Macs are safe.
- The mechanism, on this Mac with `SOCK_BUF` set to 212992 for the test: two ranks, real loopback sockets, `GroupComm`. Frames of 500, 600, 800 and 850 KiB: one all-reduce at a time passes, a pair blocks both ranks for good. Both are inside `link.send` of B's partial. The exact limit depends on the kernel: 750 KiB passed as a pair in one trial, and at 900 and 1000 KiB a single all-reduce blocked too. So the old code has the same exposure for one frame near 1 MiB, and a pair halves the frame size at which it starts.
- Linux, from knowledge of the kernel and not measured here: `setsockopt(SO_SNDBUF)` is cut to `net.core.wmem_max` without an error, and the default is 212992 (the kernel then doubles it to 425984). The same for `SO_RCVBUF` and `rmem_max`. Setting the option also turns the kernel's own buffer growth off. I found no `wmem_max` setting in the repository.
- The recorded check `OVc_pair_tp2_pair_r1` (gtx and pc, TP of 2, pair, no sender thread, 1.5B) completed. Its prompts have 44 to 78 tokens, so a frame is at most 240 kB and a pair 480 kB. That fits even a capped buffer. It does not settle the question for 7B.

Failure scenario: part 4 rung C4, 7B on gtx and pc, `--slots 2 --tp-interleave 2` without `--async-send`. The first step is a pair of two prompts. A hidden state of 7B is 7168 bytes per token, so the same prompts give frames of 315 to 559 kB and a pair of 630 kB to 1.1 MB each way. If the buffers are capped near 420 kB per side, both ranks block in the send of the second partial and the run hangs with no timeout (`DATA_TIMEOUT_S` is None). Chunks do not help unless they are small: two chunks of 64 tokens are 917 kB.
With `--async-send` there is no such risk: the compute thread receives while the sender thread sends.

To settle it, on gtx and on pc: `sysctl net.core.wmem_max net.core.rmem_max`, and read `SO_SNDBUF` back after setting 8 MiB (a three line Python check). If the answer is 8 MiB or more, this finding is low. If it is about 425984, it is high for rung C4.

Suggested fix: in `_tune`, read both sizes back, keep them on the `Link`, and write them to the rank metrics. In `send_partial`, send inline only when the frames that can be outstanding on the link (two in a pair) fit into the granted size with a margin; otherwise use the thread path, which exists and keeps the order (`_send_after`). Add the `sysctl` lines for Linux to the setup guide if the larger buffers are wanted.

### F7 (medium) No test shows that two requests are ever in flight at once

Checked: **ran** (a mutation in a scratch copy survives the whole test file).

File: `tests/bench/test_session10_overlap.py:143-153` (O1), `242-255` (O2).

What is wrong: the tests of O1 and O2 assert tokens, logits, frames and steps. Those are the right checks for the rule "scheduling does not change tokens". None asserts the feature itself, that a stage works on one request while another stage works on another.

The mutation: after every single step the scheduler waits for the head result of that slot before it starts another one (a loop of `drain(block=True)` after `hybrid.py:1220`). Then no two requests are ever in flight in a pipeline. That is the behaviour before part 3 with extra bookkeeping. Result: 58 passed. The only trace of concurrency in the O1 test is "every slot was used" (line 149), and that stays true, because all slots are filled before the loop starts.
For O3 the lockstep has a test (`ssrr`, see F4 for its limit). For O4 the frames that leave through the sender are counted (line 419), and replacing the sender branch of `send_partial` by the old path fails that test.

Failure scenario: a later change of `generate_many` serialises the requests by accident, for example a blocking read where a non blocking one was meant. Every test stays green. Part 4 then measures no gain from `--slots` and the cause is looked for in the hardware.

Suggested fix: in the O1 test for `pp3` with slots 2, spy on `Link.send` and `Link.recv` of the coordinator and assert that at some moment two slots have a token frame sent and no head result read yet. For O2, assert that the second chunk of a prompt leaves the first stage before the head result of that prompt exists and before the last stage has finished the first chunk (two units of one request in flight).

### F8 (low) The "off means old bytes" test compares the new code with itself

Checked: **ran** (the differential run below replaces what the test does not do).

File: `tests/bench/test_session10_overlap.py:156-179`.

`test_one_slot_through_the_scheduler_is_the_old_path_frame_for_frame` compares `generate_many` with one slot against `generate`, both of the current code. It cannot see a change of `generate` itself against commit 965385f. It also compares sorted lists of frames over all links together, so a frame on the wrong link or in another order would pass.
The gate of this part is the stronger claim. Today it holds: I ran the engine of 965385f and of 1d39b45 side by side (last section). But no test will catch the next change that breaks it.

Suggested fix: a golden test. For one tiny model and two placements, hash the stream of data frames per link (kind, payload, dtype, shape, both header fields) and compare with hashes stored in the test file. They were produced by the old code, so the test pins the old bytes.

### F9 (low) The old path does not mask the request number

Checked: **ran**.

File: `nocturne/engines/torch_tp/hybrid.py:969`. `pack_aux` (line 60) masks the request number to 16 bits. `_step` writes `self.req_seq << REQ_SHIFT` unmasked. With `--timeline`, request number 65536 puts a 1 into bit 24, which is now the slot. I set `req_seq` to 65535 on a recorded `pp3` run with every option off: the next `generate` fails with `ValueError: slot 1 of 1`. Before part 3 the limit was 16.7 million requests. No run comes near it, so this is low. It is also the one place where "off" differs from the old behaviour.
The recorder of a rank masks to 16 bits (line 662), so request numbers in a timeline wrap at 65536. Section 6 of the design says bits 8 to 23, which matches the mask and not line 969.

No test pins the mask: removing `& REQ_MASK` from `pack_aux` passes every test.

Suggested fix: `(self.req_seq & REQ_MASK) << REQ_SHIFT` in `_step`, and one assertion on `pack_aux(0, 65536, 0)`.

### F10 (low) `--prefill-chunk 1` is accepted

Checked: **ran** (`chunk_ranges(7, 1)` gives `[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7)]`).

File: `nocturne/engines/torch_tp/plan.py:523-533`, `hybrid.py:1098`, `1108-1111`, `hybrid_run.py:296`.

The rule "a chunk never has one token" joins only the last chunk. With C = 1 every other chunk has one token. `launch` then counts it as a decode step, the links count its frames as decode frames, and it has no head frame. So `decode_steps`, `measured_per_step` and `formula_holds` of that run are wrong. Tokens are not affected.

Suggested fix: reject a chunk size of 1 in `overlap_options` and in the argument parser (0, or 2 and more).

### F11 (low) Option values are checked late

Checked: **read**.

- `--tp-interleave 2` with `--slots 1` is rejected in `generate_many` (`hybrid.py:1058`). By then every rank has loaded its weights, which takes minutes for 7B.
- `--slots 300` passes the planner (`plan.py:389` checks only the lower bound) and fails in `RankRuntime` (`hybrid.py:408`), on the coordinator after the plan and inside every rank.
- A negative `--prefill-chunk` is taken as 0 without a word (`coordinate.py:196-198`).

Suggested fix: validate all four in `overlap_options` and call it in `plan_job` too, so the operator gets the error before any process starts.

### F12 (low) A head frame with a slot beyond the table raises the wrong error

Checked: **read**.

File: `nocturne/engines/torch_tp/hybrid.py:1177` and `1182-1185`. The `decode` rule passed to `recv` indexes `table[a0 >> SLOT_SHIFT]` before line 1183 checks `s < len(table)`. A slot of K or more gives `IndexError: list index out of range` from inside `recv`, not the message that names the rank and the slot. The frame is rejected either way.
An ERROR frame has zero in that field, so it is not affected.

Suggested fix: guard the index in the rule (`s < len(table) and table[s] and table[s]["decode"]`).

### F13 (low) The copy modes that matter on a device are not run by any test

Checked: **read**, and the working tree confirms it.

File: `tests/bench/test_session10_overlap.py:395-430`, `nocturne/engines/torch_tp/hybrid.py:194-209`, `280-286`.

The tests run `inline` and `thread` on CPU tensors. `pinned` (the CUDA default) is never run: the pool of pinned buffers, the event, and the rule that a buffer returns only after its frames left (`free`, line 220) have no test. By reading I found the order correct: every `free` is queued after the sends of its tensor, and the compute thread has finished its add before it queues the `free` (lines 378-395).
`thread` on a real device was the MPS default at the commit. The first engine run with it died with a Metal assertion (`OVs_15b_pp3_s2_c16_async_r1`), and the working tree changed the default to `inline`. The probe and the tests could not have found this.

Suggested fix: a CUDA only test (skipped elsewhere) that runs `tp3` and `pp3` with `pinned` and slots, and a unit test of the pool with a fake pinned mode: hand over, free, hand over again with the same shape, and check that the first frame's bytes were not overwritten before it left. Record in `VERIFY_overlap.md` which CUDA runs stand in for the missing test.

### F14 (low) Documents and code differ in six places

Checked: **read**, except where another finding ran it.

1. `docs/ENGINE_DESIGN.md:288` names `experiments/session10_cuda_pair/VERIFY_overlap.md`. The file is not in the repository at this commit.
2. `docs/MEASUREMENT.md` 3.1: `unmatched_links` is "empty in every verified run". It is not empty for a pure TP run with slots (F2), and no recorded run with slots has a timeline.
3. `docs/ENGINE_DESIGN.md` 11.5 and `docs/MEASUREMENT.md` 3.1: with the option, `compute` is the time of the compute thread. For a TP rank that time is recorded as `allreduce_wait` (F1).
4. `docs/ENGINE_DESIGN.md` 11.5: the second half of an all-reduce "waits for the copy of that partial, not for its send". There is one queue, so the copy job stands behind every send handed over earlier, and `ready` is set only when the sender reaches the job, also in `inline` mode where the bytes are already on the host (`hybrid.py:208-210`, `248-255`). So the compute thread can wait for earlier sends. I found no deadlock from this: every rank receives before it waits for its own copy, and all ranks send in the same order.
5. `docs/ENGINE_DESIGN.md` 11.4 says how a pair forms, and not that with two slots it mostly does not (F4).
6. `docs/ENGINE_DESIGN.md` 11.7: "a run under 256 slots". The code allows 256 (slot ids 0 to 255, `hybrid.py:56`, `408`).

One more, for the record and not a defect: the PLAN message and the metrics reply have two new keys each (`slots`, `async_send`). So the byte totals of a control link differ from the old code by a few dozen bytes per message. 11.7 says so for PLAN. The comparison "same bytes per link" must use the data frames, which is what the link counters of a decode step do.

### F15 (low) Planner details

Checked: **read**.

- A pair holds the activations of two requests at once. `workspace_bytes` (`plan.py:309-322`) counts one prefill. The planner does not know about `--tp-interleave`, because the option stays on the coordinator. Two prefills of `prefill_max` tokens in one pair need about twice the planned work space on every rank of the group.
- The cost per rank process (`PROCESS_OVERHEAD`, `plan.py:26`) enters only the host total. A placement with rank budgets and no host budget never counts it (see F5).
- `host_totals` sums by host name. A host with two GPUs would get one sum for both. No machine of the project has two, so this is a note.

Suggested fix: pass the interleave value to `plan` as `kv_slots` is passed, and double the work space for group ranks when it is 2. Key the sum by host and device.

## Checked and found correct

**Off means the old behaviour (the hard gate).** Differential run, **ran**: the package of commit 965385f and the package of 1d39b45, the same tiny model, the same ports, six prompts of 2 to 40 tokens, 6 tokens each. Placements: PP of 3, TP of 3, hybrid with the TP group first, hybrid with the TP group last. Each with float32, with float32 and the recorder, and with bfloat16. All of it twice: with the normal `SMALL_FRAME`, and with `SMALL_FRAME = 64` so that every all-reduce frame leaves on a helper thread. 24 cases. In every case: every data frame on every link is identical in kind, payload, dtype, shape and both header fields, in the same order; tokens and first step logits are identical; the decode counters, the frames by kind, the bytes of every data link, the steps and the all-reduce calls are identical. The control frames have the same kinds in the same order. The PLAN message differs by the two new keys and nothing else.

**Tokens do not depend on scheduling**, **ran**: 14 random prompts of 2 to 29 tokens, up to 30 tokens, EOS not ignored (one request ends by EOS), on the four placements. Slots 3 and 4, with and without pairs, with and without the sender thread, and with `SMALL_FRAME = 16`: tokens, finish reason and first step logits equal the one request path bit for bit in all 16 runs. With chunks of 4 and pairs: tokens equal in 14 of 14 on every placement, first step logits within 1e-5.

**Read and found correct:**

- Slots. The KV cache has the slot as its first dimension, every read and write uses the frame's slot (`model.py:126-129`), and `_rotary` rejects a slot outside the cache. With one slot the tensors and the index are the old ones.
- Bit packing. `pack_aux` (flags, 16 bit request number, slot), the head frame (24 bit index, slot), the masks on the receiving side (`hybrid.py:662`, `1189`), the index check on both sending paths (`270`, `632`), slots limited to 256 in every rank.
- The all-reduce in two halves. `recv_sum` does the old operations in the old order: all partials in group rank order, in float32, rounded once. `all_reduce` is `send_partial` followed by `recv_sum`. The old code joined its helper threads at the same point.
- The lockstep. Every rank sends A then B and receives A then B, for the embedding and for both sums of every layer. A pair never has one slot twice (`model.py:193`), and the scheduler cannot produce that: a slot is in the ready list at most once.
- The pair flag. It is set on the first frame only, travels in the hidden state frames, and a stage of one rank takes the two frames in turn. A control frame cannot come between the two frames of a pair, because only the coordinator sends them and it is one thread.
- The scheduler. Head results are collected per slot from the slot in the frame; a slot has one unit in flight at a time; a chunk that is not the last asks for no head result and the head ranks send none; chunk positions are the token offsets; a finished slot takes the next waiting request; the stale cache of the previous request in a slot is never read.
- The sender thread. One queue, so frames leave every link in hand over order. No tensor that was handed over is changed in place afterwards. The compute thread waits for the copy before it adds. The sender is flushed before metrics, clock answers, reset and shutdown on every rank. `stop` cannot hang (3 s join, daemon thread). With frames larger than the socket buffers I found no deadlock for two or three ranks, with or without pairs: every rank receives in group rank order and sends in group rank order.
- Large frames of a pair on helper threads. The second frame waits for the first on its link (`_send_after`), also when the second is small.
- The planner. `kv_slots` multiplies the KV bytes; the host total is the sum of the rank footprints plus the cost per process; a host over its budget is named; a budget for a host without ranks is an error; `--force` skips both checks as before.
- `chunk_ranges` for every chunk size of 2 or more.
- The interval helpers, the request window, and the shares of `timeline.overlap` for one socket per pair of ranks.
- `coordinate.run_job` with every option off calls `generate` per prompt as before.
- This Mac grants the 8 MiB socket buffers (read back).

**Mutations**, **ran** in a scratch copy of the commit, never in the repository.

The 24 part 3 entries of `mutate.py`: all 24 are caught.

| break named in V2 | mutations | first failing test |
|---|---|---|
| the slot id on the wire | 4 | `test_each_request_in_flight_gives_the_tokens_it_gives_alone[2-pp3]` (three), `test_async_send_with_slots_chunks_and_pairs[pp3-3-3-1]` (the head result of the sender thread) |
| the add order under interleave | 2 | `test_interleaved_requests_give_the_same_tokens_and_the_same_sums[2-tp3]` |
| the chunk position offset | 1 | `test_chunked_prefill_gives_the_same_tokens_and_the_predicted_frames[1-2-pp3]` |
| the per host memory sum | 3 | `test_planner_sums_all_rank_processes_of_a_host_against_the_host_budget` |

The other 14 are caught by the tests their names suggest.
Two are caught by a hang and not by a failing assertion: "the compute thread does not wait for its partial to reach the host" and "a failure of the sender thread is not reported". I ended each after six to seven minutes. `mutate.py` counts its 900 s timeout as caught, so the result stands, but each costs 15 minutes per round. A join with a timeout in the test helper would turn both into failures.

Eight mutations of mine:

| mutation | result |
|---|---|
| the scheduler waits for the head result after every single step (never two requests in flight) | not caught, 58 passed. F7 |
| the sender branch of `send_partial` replaced by the old path (partials copied and sent on the compute thread) | caught, `test_async_send_gives_the_same_tokens_logits_and_frames[tp3-inline]` |
| the v cache of slot 0 is written whatever the slot | caught, `test_each_request_in_flight_gives_the_tokens_it_gives_alone[2-pp3]` |
| the ranks of a group ignore the pair flag | caught, `test_interleaved_requests_give_the_same_tokens_and_the_same_sums[2-tp3]` |
| `pack_aux` without the mask of the request number | not caught, 58 passed. F9 |
| `RankRuntime.metrics` without its own flush | not caught, 58 passed. Not a gap: every caller flushes first (`forward_control`, `Coordinator.metrics`), so that line is redundant |
| the host total counts the first rank process of a host as free | caught, `test_planner_sums_all_rank_processes_of_a_host_against_the_host_budget` |
| the fix of F1 (a `compute` lap in the sender branch of `send_partial`) | 58 passed. So no test pins the span kinds of a TP rank with the option, in either direction |

## Not checked

- Anything on CUDA or MPS. The `pinned` path, the device events and the MPS behaviour are judged by reading only.
- The socket buffer sizes on gtx and pc (F6).
- The real runs under `experiments/session10_cuda_pair/runs/`. I read the records of the `OVc` and `OVs` check runs for prompt lengths, options and the presence of a timeline, and nothing else.
- `timeline_page.html` in a browser. I read the diff only.

## Disposition (by the author of the code, 2026-11-28)

The reviewer's text above is unchanged. Fixes are in the commit after `6ddcf01`. Tests named here are in `tests/bench/test_session10_overlap.py`.

| id | decision | what was done |
|---|---|---|
| F1 | fixed | `GroupComm.send_partial` closes a `compute` span before the hand over. Test: `test_async_send_closes_a_compute_span_before_every_hand_over` |
| F2 | fixed | Every send and wait span carries its channel (group, pipeline, control). `timeline.in_flight` matches per channel. Tests: `test_frames_in_flight_are_matched_per_socket_in_a_tp_group_with_slots`, `test_in_flight_keeps_two_sockets_of_one_pair_of_ranks_apart` |
| F3 | fixed | A failing sender thread closes every link of its rank and logs the real error. Test: `test_a_failure_of_the_sender_thread_ends_the_run_at_once` |
| F4 | recorded, policy kept, open for Ohad | The scheduler counts `pair_steps`, `single_steps` and `max_in_flight`, and the record holds them. The policy stays "no waiting for a partner": waiting would idle the first stage while a later stage works. Section 11.4 of the design now says that two slots seldom pair behind a later stage and four do |
| F5 | marked, not an error | Every loopback placement from before part 3 has one host in several stages and no host budget, so an error would break the old placements. The plan table and the operator's log now say "the sum is not checked", and the record has `planned_hosts.<host>.unchecked`. Test: `test_a_host_with_several_rank_processes_and_no_host_budget_is_marked_unchecked` |
| F6 | **confirmed high on gtx, fixed** | gtx: `net.core.wmem_max = 212992`, and a socket reports 425984 after the request for 8 MiB. `transport._tune` now reads the granted sizes back, `Link.fits_inline` allows an inline frame only when it and the outstanding inline frames fit half the granted send buffer, and `GroupComm` counts the inline frames of a pair partner. The granted sizes are in the rank metrics (`socket_buffers`). The old two rank engine (`engine.py`) uses the same rule. Tests: `test_a_frame_goes_inline_only_when_it_fits_the_buffer_the_kernel_granted`, `test_a_pair_of_large_frames_cannot_block_both_ranks_with_small_socket_buffers` (real sockets, 600 kB frames, 212992 byte buffers), `test_small_socket_buffers_do_not_change_the_tokens`. The pc was off when this was checked; it runs the same WSL2 kernel defaults, and the rule does not depend on the value |
| F7 | fixed | `test_the_scheduler_counts_pairs_singles_and_units_in_flight` (two and three units in flight), `test_a_chunk_leaves_the_first_stage_before_the_head_result_of_the_prompt`, `test_async_send_returns_before_the_frame_has_left`. The reviewer's two mutations are now in `mutate.py` |
| F8 | fixed | `tests/bench/golden_frames.py` ran on the engine of commit `965385f` and its hashes are pinned in `test_options_off_send_the_frames_of_the_engine_before_part_3` (three placements, with and without the recorder) |
| F9 | fixed | `_step` masks the request number. Test: `test_the_old_path_masks_the_request_number_so_it_never_reaches_the_slot_bits` |
| F10, F11 | fixed | `plan.check_overlap` checks all option values. It runs in the argument parser of `hybrid_run`, in the planner's answer and in the scheduler. Test: `test_overlap_options_are_checked_before_anything_starts` |
| F12 | fixed | The decode rule checks the slot against the table first |
| F13 | fixed, CUDA only | `test_pinned_copy_on_cuda_gives_the_tokens_and_logits_of_the_plain_path` (skipped without CUDA; run on gtx, see `VERIFY_overlap.md`). The CUDA runs `OVc_*` and `OVc2_*` also use the pinned path |
| F14 | fixed | All six places. In `inline` mode a partial is now ready at once, so the compute thread no longer waits behind earlier sends there |
| F15 | partly fixed | The planner doubles the work space of group ranks for a pair. The cost per process still counts only against a host budget (see F5). The sum per host name stays; no machine of the project has two GPUs |
