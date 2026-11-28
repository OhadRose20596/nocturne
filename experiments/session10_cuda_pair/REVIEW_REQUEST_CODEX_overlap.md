# Review request for Codex: session 10, part 3 (overlap)

You did not write this code. Please look for errors, not for style. Rate each finding high, medium or low, and say how to check it.
Put your answer in `experiments/session10_cuda_pair/REVIEW_CODEX_overlap.md` (Ohad pastes it). Short sentences. No em dash, no en dash.

## What changed

The torch engine (`nocturne/engines/torch_tp`) got five options, so that a machine computes other work while its data is on the network.
Start SHA: `965385f`. The diff is `git diff 965385f -- nocturne tests docs experiments/session10_cuda_pair/mutate.py`.

1. **Slots (`--slots K`).** The KV cache has a slot dimension. The slot id travels in TOKEN, HIDDEN and HEAD frames. The coordinator keeps K requests in flight (`Coordinator.generate_many`). A rank never schedules: it handles frames in arrival order.
2. **Chunked prefill (`--prefill-chunk C`).** The prompt travels in chunks. Only the last chunk gets a head result.
3. **Interleaved pair in a TP group (`--tp-interleave 2`).** The all-reduce is split into `send_partial` and `recv_sum`. Two requests send both partials before either sum is read.
4. **Async send (`--async-send`).** One sender thread per rank (`hybrid.Sender`). CUDA: non blocking copy into pinned memory, the sender waits for a device event. MPS and CPU: the copy stays on the compute thread.
5. **Interleaved pipeline stages.** A host may hold several stages, each its own process. The planner sums all rank processes of a host against `host_budget_gb`.

Read `docs/ENGINE_DESIGN.md` section 11 first, then section 6. `docs/MEASUREMENT.md` section 3.1 defines the busy share and the overlap share. The task text is `experiments/session10_cuda_pair/03_BUILD_OVERLAP.md`. My own record is `VERIFY_overlap.md`, and the first review is `REVIEW_overlap.md`.

## Rules the code must follow

1. Every option is off by default. With every option off the engine must send the same frames and the same bytes per link and produce the same tokens as at `965385f`. This is a gate: the "before" runs of the next session depend on it.
2. The tokens of a request must not depend on the scheduling: not on the number of slots, not on the pair, not on async send. Chunked prefill may differ in rounding; the criterion there is 7 of 8 prompts.
3. The numerics and the add order of the all-reduce must not change: all partials in group rank order, in float32, rounded once.
4. No new frame kind and no change of the header size. The new fields use bits that were zero (section 11.7).
5. Never bind `0.0.0.0`. Security is deferred by ADR-009.
6. Time is background (T1). Nothing may gate on a time.
7. No new dependency. Torch is not installed or upgraded.

## Files to read, in this order

1. `nocturne/engines/torch_tp/hybrid.py`: `pack_aux` and the flag constants, `Sender`, `GroupComm.send_partial` and `recv_sum`, `RankRuntime.step`, `_finish`, `step_pair`, `_send_head`, `serve`, `Coordinator.generate_many`.
2. `nocturne/engines/torch_tp/model.py`: the slot dimension of the KV cache, `Layer.forward_pair`, `Qwen2TP.head_device`.
3. `nocturne/engines/torch_tp/plan.py`: `host_totals`, the host budget check in `plan`, `chunk_ranges`, `frames_prefill`.
4. `nocturne/engines/torch_tp/coordinate.py` and `hybrid_run.py`: how the options reach the job, the record, `load_reference`, `compare`.
5. `nocturne/engines/torch_tp/timeline.py`: `in_flight`, `overlap`; `timeline_page.html`: the new figure and rows.
6. `tests/bench/test_session10_overlap.py`, `experiments/session10_cuda_pair/mutate.py` (the entries that start with "overlap").

## Questions I would ask first

1. Can `generate_many` hang or mix up two slots? Look at `drain`, `after`, `complete`, and at a request that ends while its pair partner goes on.
2. A pair in a later stage: the leader of the stage before sends two HIDDEN frames, the first flagged. Can a group rank ever read the partner of the wrong pair, or wait for a partner that is not coming?
3. The sender thread: is there an order of events in which a frame leaves a link out of order, a buffer is reused while it is still needed, or an error is lost? Is every control frame sent only after the sender is empty?
4. A frame over 1 MiB leaves on a helper thread. I found that two such frames of a pair had no fixed order and added `GroupComm._send_after`. Is the fix complete?
5. With `--async-send` on CUDA the head result is computed by `head_device` (first index of the maximum). Is it the same choice as `head_argmax` in every case, NaN included?
6. Is anything on the default path different from `965385f`? I claim the old `generate` and `_step` are untouched and the new header bits are zero.
7. The overlap share counts a frame as in flight until the receiver reads it. Is the definition in `docs/MEASUREMENT.md` what `timeline.in_flight` computes, and is matching by order on the link safe?
8. The planner adds a fixed cost per process (150 MiB CUDA, 200 MiB MPS). The measured cost on the GTX 1070 was about 155 MiB per extra process. Is anything counted twice or not at all?

## How to run

```bash
~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_session10_overlap.py -q
```

```bash
~/dev/nocturne-torch/bin/python experiments/session10_cuda_pair/overlap_table.py OV_ OVc
```
