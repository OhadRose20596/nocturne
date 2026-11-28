# Session 10, part 3. Independent overlap review

Reviewed `965385f` through `3616976`. Two slots can deadlock during ordinary generation with first logits enabled. The strict all-options-off wire gate also fails. Short default runs matched the base on tokens and ordered data frames. Their control payloads changed.

## Findings

### 1. High. A large HEAD reply can block the next slot's all-reduce

`hybrid.py:1201-1218,1241-1266`. The coordinator polls for HEAD between units. If the poll finds nothing, it can enter slot B's all-reduce while a peer is sending slot A's logits. Once the control socket fills, the peer cannot finish HEAD and the coordinator cannot finish the collective. Async send has the same cycle. The peer queues B's partial behind A's blocked HEAD on its single sender. The normal job requests first logits for every prompt.

**Check.** I reproduced both modes on CPU loopback with two TP ranks, two slots and interleave 1. Use one layer, hidden size 8, intermediate size 32, vocabulary 1,048,576 and float32 activations. Set `transport.SOCK_BUF=4096` before boot. Generate two three-token prompts with first logits and two output tokens each. Hold the first remote HEAD until the coordinator starts sending slot B's TOKEN, then release it. The 2,097,152-byte HEAD remained in `_send_all`. The coordinator was in slot B's `recv_sum`. With async enabled, the peer's sender was in `_send_all` and its compute thread was also in `recv_sum`. The coordinator needs to keep reading HEAD while its local collective waits.

### 2. Medium. Default control frames carry extra bytes

`hybrid.py:953-960,630-640`. The default PLAN serializes `slots=1` and `async_send=false`. READY and METRICS_REPLY also carry new metrics fields. The golden test covers data frames. It cannot establish byte equality for every link across the full run.

**Check.** Capture PLAN at `Link.send` on both revisions with the same TP3 fixture and the recorder off. Each worker's PLAN payload grew from 2,074 to 2,107 bytes. The same 33-byte increase occurred with the recorder on, in PP3 and in the hybrid placement. The next session can reuse the checked decode baselines for those cases. A claim of identical full wire traffic would be incorrect.

### 3. Medium. Recorded request IDs alias after 65,535 requests

`hybrid.py:53-60,698,1005-1008` and `timeline.py:60-69`. The wire now carries 16 request bits. The coordinator records the full sequence number, but remote ranks record the masked value. At request 65,536 even the default recorded data frame differs from the base. After the wrap, remote spans can be attributed to an earlier request.

**Check.** In a PP3 run with the recorder on and all options off, set `co.req_seq=65535` before generating once. I saw request 65,536 on the coordinator and request 0 on the remote rank. On a TOKEN with the logits flag clear, the base encodes `aux1=0x01000000`. The current code encodes zero. Check both the wire field and the request IDs in the breakdown.

### 4. Medium. A failed large-frame helper can release its successor

`hybrid.py:359-375,386-403`. On the successful path, `_send_after` orders two large frames. On failure, `Thread.join()` establishes only that the helper exited. Neither the next helper nor `recv_sum` receives its exception. B can therefore be sent after A failed. The peer identifies those all-reduce frames by order and has no request tag to detect the substitution.

**Check.** Force both sends onto helpers. Make the first fake-link send raise `OSError` before writing. Let the second succeed. Supply two valid zero partials from the fake peer. The thread exception hook saw the error. The caller returned both local sums, 1 and 2, although only B left the link. Propagate a failed predecessor to the collective and wake any blocked receiver.

### 5. Medium. A scheduling regression can pass the recorded correctness check

`hybrid_run.py:477-485,519-524`. A same-placement run with chunking off requires exact tokens for each request. The comparison uses the seven-of-eight threshold for this case too. It records long-prompt divergence separately and ignores it in `criterion_met`.

**Check.** Compare eight scored evaluation prompts to a same-placement reference. Change one output token and keep all scores equal. I got `identical=7`, `of=8` and `criterion_met=True`. Also try changing only the long prompt. Require exact token lists for scheduling-only comparisons. Keep the relaxed criterion for chunked prefill.

### 6. Medium. A one-token prompt corrupts prefill and decode counts

`hybrid.py:1135-1148` and `coordinate.py:136-145`. The scheduler identifies decode from the length of the unit. A one-token prompt is a valid prefill. Its frames, including the first logits, are charged to decode. The prediction still charges one prefill.

**Check.** On the tiny TP3 placement, generate from `[5]` with two slots, first logits and three output tokens. I got three decode steps, zero prefill steps, zero measured prefill frames and 58 predicted prefill frames. The request executed one prefill and two decode steps. Carry the phase from the scheduled unit into the counters and recorder.

### 7. Medium. A large slotted KV tensor can escape the MPS heap floor

`plan.py:461-472` and `model.py:86`. The planner scales KV bytes by the slot count. Its 1 GiB MPS heap test only examines weight matrices. A K or V allocation can cross the documented 10 MiB threshold even when every weight matrix is small. The resulting host total can pass a budget below the heap floor.

**Check.** Plan the four-layer tiny model on one MPS rank. Use 256 slots, sequence length 1,024, prefill maximum 16, two KV heads and head size 16. One BF16 K tensor is 16 MiB. The planner returned a zero floor and accepted a 650,000,000-byte host budget. Its total was 344,583,040 bytes, including process overhead. This reproduces the planner decision. The threshold comes from the project's MPS measurements.

### 8. Medium. An unmatched trace still publishes an overlap share

`timeline.py:174-187,206-233` and `timeline_page.html:147-165`. Dropping a receive shifts every later ordinal match on that socket. The code flags the mismatch but still uses the truncated pairs to compute and display percentages. Measurement section 3.1 says a run with unmatched links has no overlap share.

**Check.** Use a request window from 0 to 10. Send equal-size frames from A at 2 and 3. Keep only the receive ending at 8. Put A's compute from 4 to 7. I got `unmatched_links=["a -> b"]` and 30% overlap for A. Suppress the invalid share in the record and chart. The busy share can still be calculated.

### 9. Low. The sender can retain more than its pinned-buffer limit

`hybrid.py:285-291`. The pool checks its current size before adding the returned buffer. One large buffer can take it over the 256 MiB limit. With an empty pool, a single oversized buffer is kept in full.

**Check.** Set `POOL_MAX_BYTES=16`. Pass a 64-byte tensor marked pooled through `Sender.free`, then flush. I measured `pool_bytes=64`. This exercises the pool accounting on CPU. Include the returning buffer's size in the capacity check.

### 10. Low. Coordinator CLOCK and SHUTDOWN bypass the local sender flush

`hybrid.py:1288-1293,1304-1315`. The coordinator sends CLOCK directly. It also sends SHUTDOWN to first-stage peers before reaching the flush in `forward_control`. The normal job's preceding metrics call drains the sender. A direct clock call or an early shutdown lacks that ordering guarantee from design section 11.5.

**Check.** Instrument `rank.flush` and the coordinator's control-link send. Calling `clock_sync()` produced CLOCK with no preceding flush. For shutdown, prepopulate the end-clock record and trace the first-stage SHUTDOWN against the flush in `forward_control`.

## Checks that passed and limits

- The requested overlap suite passed 85 tests. Three CUDA pinned-copy cases skipped because CUDA was unavailable. I used the existing torch environment. I did not rerun the remote GPU experiments.
- The requested `overlap_table.py OV_ OVc` command completed. All nine recorded baseline rows matched their part 1 comparisons on nine prompts and on decode frames and bytes per link per step. The table reads saved records. It does not exercise the current engine.
- I compared an archived copy of `965385f` with the reviewed engine on CPU loopback. Five prompts ran in TP3, PP3 and hybrid, each with the recorder on and off. I hashed the header, dimensions and payload in order per sender link for TOKEN, HIDDEN, HEAD and ALLREDUCE. All six comparisons matched, including generated tokens. Each revision contributed 3,080 data frames. The fixed header remains 16 bytes. I found no added frame kind.
- I exercised a single-rank first stage feeding a three-rank TP stage. With two slots and interleave 2, three requests generated 1, 6 and 3 tokens. Slot 0 was reused while slot 1 continued. All three token lists and first-logit tensors matched serial generation. The run used eight paired units and two single units. I found no wrong partner in that case.
- The default reduction visits group ranks in ascending order. It adds in float32 and casts once at the end. On the pinned path, the copy records a CUDA event and the sender waits for it. The free operation follows the queued writes and local use. This was source inspection for CUDA.
- For tied maxima, both head implementations choose the first index. For logits `[1, NaN, 1]` on CPU, the synchronous method raised on its empty index list. The device method produced NaN and sentinel index 3, which the sender rejects. Both paths fail the request. They surface different errors.
- For complete traces with channel tags, matching uses direction and socket channel. The interval runs from send start through receive completion. Device shares use the union across ranks. Older traces without channel tags cannot distinguish two sockets between the same ranks.
- The host sum adds each rank footprint and one fixed process allowance per rank. The MPS floor enters through a maximum. The recorded GTX delta was about 155 MiB per extra process. The CUDA allowance is 150 MiB, so a tight host budget has no guaranteed margin. I found no duplicate addition of that allowance.

## Disposition (by the author of the code, 2026-11-28)

The reviewer's text above is unchanged. All ten findings were confirmed. Code and tests are in `c82ec1c` and the commit after `ed4a8c2`. The text changes are in `ed4a8c2`. Results: `VERIFY_overlap.md`, section "After the Codex review".

| id | severity | decision | what was done |
|---|---|---|---|
| 1 | high | fixed | Reproduced on CPU loopback as described, with and without async send. The same cycle exists on a plain pipeline, which the review did not name: the coordinator blocks in the prefill send of a third request while the last stage blocks in the head frame of the second. Reproduced too. Fix: a thread of the coordinator reads head frames at any time (`hybrid.HeadReader`), only inside `generate_many`. The old path is untouched. Tests: `test_a_large_head_frame_does_not_block_the_work_of_the_next_slot` (both modes, hangs on the old engine), `test_head_frames_are_read_while_the_scheduler_is_busy_and_the_reader_ends_with_the_call`, `test_a_lost_control_link_ends_a_run_with_slots`. Four real runs with slots on the M5 (MPS) after the fix |
| 2 | medium | fixed, and the claim corrected | The PLAN frame names `slots` and `async_send` only when they are set, so with the options off it has the fields of `965385f` in their order (`test_the_plan_frame_names_an_overlap_option_only_when_it_is_set`). READY and the METRICS reply keep their three new fields: they are JSON reports, and the granted socket buffers belong in every record. The design, the README and the verification record no longer say that every byte on every link is the same |
| 3 | medium | fixed in the record, documented on the wire | A rank rebuilds the full request number from the 16 bits of the frame and the last number it saw (`Recorder.full_req`), so spans past request 65535 are not given to an earlier request. The frame itself still carries 16 bits, because the slot took the upper 8. So with `--timeline`, from request 65536 of one run on, a recorded frame differs from the old engine's in that field. This is stated in the design (11.1) and the README |
| 4 | medium | fixed | `hybrid.FrameThread`: a frame behind one that failed is not sent, the failure closes the link, the all-reduce raises it, and no later all-reduce starts on that link. Test: `test_a_frame_behind_one_that_failed_is_not_sent_and_the_all_reduce_raises` |
| 5 | medium | fixed | A reference with the same placement file and hash, dtype and prefill chunk differs in scheduling only. Against it every token of every prompt must match (`rule: every token`). Other references keep seven of eight. No earlier result changes: all 30 runs under the exact rule had 8 of 8 and identical long prompts |
| 6 | medium | partly fixed | The step counts now follow the scheduled unit (one prefill step for a one token prompt, as on the old path). The frames of such a prompt are still counted as decode frames, because every rank tells a decode frame by its one token, on the old path as well. A full fix needs a phase flag in the frame. Documented as a known limit; no prompt of the project has one token |
| 7 | medium | fixed | The K tensor of one layer holds every slot. The planner now applies the 1 GiB MPS floor when that tensor reaches 10 MiB. Test with the reviewer's case |
| 8 | medium | fixed | A run with an unmatched link has a busy share and no overlap share: `null` in the record, "n/a" on the page, the links named |
| 9 | low | fixed | The pool limit counts the buffer that returns |
| 10 | low | fixed | The coordinator empties its own sender before CLOCK and before SHUTDOWN. At shutdown the wait gives up after 20 s |

From "Checks that passed and limits", not changed:

- The two head implementations fail a request with NaN logits through different errors. Both fail it.
- The CUDA allowance per process is 150 MiB and the measured cost was about 155 MiB. The pair session measures it on both cards before any tight host budget is used.

Not checked after the fixes: anything on Linux or CUDA. gtx and the Air were closed for the day before the review arrived. The first step of the next session is `git pull` on gtx and the overlap test file there.
