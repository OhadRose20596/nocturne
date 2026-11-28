# Session 10, part 3 of 4: compute while data is on the network (build)

Work prompt for Claude Code. Written 2026-11-28 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo.
Order changed on 2026-11-28: this part runs before part 2. Part 2 and part 4 then run together in one session on the CUDA pair: first the part 2 rungs with every overlap option off (the "before"), then the part 4 rungs. That is valid only because every option here is off by default and "off" gives the old behaviour byte for byte. So that rule is a gate of this part, not a wish.
Before you start: commit or set aside any uncommitted work from part 1 (the K3 check and the open edits in `clock.py`, `hybrid.py`, `recorder.py`, `transport.py`), and tell Ohad which.
Build and verify only. Loopback on the M5, plus short checks on gtx and pc if they are on. Experiments are parts 2 and 4.

## 0. Why

The 23.11 timelines and part 2 show that every machine computes, sends, then waits. Chen asked why a machine does not compute while it sends. The reason is our design: a blocking all-reduce, a copy to the host with a device sync before every send, and one request at a time per stage (`docs/ENGINE_DESIGN.md` section 1). Ohad decided on 2026-11-28 to change this now, inside session 10.
Two goals:
1. Overlap: a machine computes other work while its data is on the network.
2. Interleaved pipeline stages (the Megatron interleaved schedule): a machine holds several non-contiguous layer ranges, so with several units of work in the pipeline no machine waits long.

The hard rule stays: the tokens of a request must not depend on how the work was scheduled.

## 1. Read first

`CLAUDE.md`, `docs/ENGINE_DESIGN.md`, `docs/MEASUREMENT.md`, `experiments/session10_cuda_pair/01_BUILD.md`, `VERIFY.md`, the part 2 records, `nocturne/engines/torch_tp/` (`hybrid.py`, `model.py`, `plan.py`, `transport.py`, `recorder.py`). The 23.11 meeting page section "For discussion: can the machines compute while data is on the network?" (`Nocturne_progress_meeting_2026-11-23.html` in the project folder) has the predictions.

Rules as in part 1. This part may change the engine and the frame format within the scope below. Document every wire change in `transport.py` and `docs/ENGINE_DESIGN.md` section 6.

## 2. What to build

Write the design into `docs/ENGINE_DESIGN.md` (a new section "Overlap") before the code. Every feature is behind an option, off by default, and an option set to its "off" value gives the old behaviour byte for byte.

O1. Several requests in flight (PP and hybrid): `--slots K`.
- KV cache with a slot dimension on every rank. The slot id travels in TOKEN and HIDDEN frames always, not only with the recorder.
- The coordinator keeps up to K requests in flight. A stage processes frames in arrival order, so stage 1 works on request A while stage 0 works on request B.
- The coordinator collects HEAD frames per slot and feeds the next token of that slot.

O2. Chunked prefill (PP and hybrid): `--prefill-chunk C`.
- The prompt is split into chunks of C tokens. Chunk i goes to the next stage while this stage computes chunk i+1. Each chunk attends to the cached prefix of its own request and causally within itself (the existing `pos0 > 0` path).

O3. Interleaved requests in a TP group: `--tp-interleave 2`.
- Two requests share a group in lockstep. For each block: compute A's partial, send it without waiting, compute B's partial, send it, then receive and add A's, then B's. A's transfer is hidden behind B's compute.
- The add order inside each all-reduce stays exactly as today, so the sums are the same bytes.

O4. Send without blocking the device.
- Start the device to host copy as soon as a partial or a hidden state is ready (non blocking copy on CUDA, the best available on MPS), and send from a sender thread. Record where the device sync now happens.

O5. Interleaved pipeline stages (virtual stages).
- Allowed by the placement: a host may appear in several stages, each a separate rank process on its own port, with its own layer range. Example for two machines: gtx 0 to 6, pc 7 to 13, gtx 14 to 20, pc 21 to 27.
- The planner must count the memory of all rank processes of one host against that host's budget (one GPU), including the CUDA context per process. If a second process per GPU is too costly, report it and propose one process holding several stage slices instead. Do not build that without asking.
- With K = 1 this only adds hops. It is meant to be combined with O1 and O2.

Not in this part: speculative decoding, keeping a conversation's KV cache between requests. Write one paragraph each in the design section on what they would need.

## 3. Verify

V1. Correctness, all on the M5 loopback and, if on, on gtx and pc:
- O1: each request in a K = 2 and K = 3 run gives exactly the tokens it gives alone (K = 1), on the 8 eval prompts and the long prompt. Any difference fails.
- O3: the same rule for each of the two interleaved requests, and the all-reduce sums are byte identical to K = 1 (check with the first-step logits).
- O4: tokens identical with and without.
- O2: at least 7 of 8 identical against unchunked prefill, and the first-step logit difference reported. Explain any difference (attention over different shapes can round differently).
- O5: an interleaved placement gives the same tokens as the contiguous placement of the same layers on the same devices.
- Frames per step: extend the formula for slots, chunks and virtual stages. Measured equals predicted in every run.
- Options off: the part 1 V3 runs again with this code and every option off give the same tokens, the same frames and the same bytes per link as recorded in `VERIFY.md`.
V2. Mutation check: break the slot id on the wire, the add order under interleave, the chunk position offset, the per host memory sum of the planner. Each must fail a test.
V3. Recorder: the timeline shows per request and per slot which work runs while a transfer is in flight. Add an "overlap" figure to `timeline.html`: per device, the share of time computing while it also has data in flight.
V4. Review: a fresh subagent on the diff and the design section, `REVIEW_overlap.md`. Then `REVIEW_REQUEST_CODEX_overlap.md` for Ohad. Fix every high finding, second fresh pass on the fixes.
Gate: correctness rules met, mutations caught, no open high finding.

## 4. Records and push

`VERIFY_overlap.md`, review files, `docs/ENGINE_DESIGN.md`, `docs/MEASUREMENT.md` (the overlap share), the engine README, two lines in `docs/CURRENT_CONTEXT.md`, the private shared log. Small commits, push.
Final message under 25 lines: what was built, correctness per feature, any difference in O2 and why, the memory cost of virtual stages per GPU, open findings, what Ohad must do before part 4.

## 5. Do not

Change the numerics or the add order of the all-reduce. Make any feature on by default. Run experiment rows. Edit old run folders. Bind `0.0.0.0`. Install or upgrade torch.
