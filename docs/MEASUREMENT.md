# Measurement: where every number comes from

Status: session 10, part 1 (2026-11-28). For the supervisor and for anyone who reads a record.
It covers our engine (`nocturne/engines/torch_tp/`). The design of the engine is in `docs/ENGINE_DESIGN.md`.
Rule T1 still holds: time is background, never a gate. The numbers below describe a run. They do not rank engines.

## 1. The files of a run

A run is one folder, `experiments/<session>/runs/<run_id>/`. A run id is never reused.

| file | written by | what it holds |
|---|---|---|
| `record.json` | the operator, `hybrid_run.main` | Everything about the run: placement, git SHA, preflight, plan, prompts and tokens, frames, memory, clocks, shipping, timeline summary |
| `coordinator_record.json` | the coordinator, `coordinate.run_job` | The coordinator's part, copied into `record.json` |
| `job.json`, `plan.json` | the operator; the coordinator (`coordinate --plan-only`) | The job as sent, and the planner's answer |
| `coordinator.log`, `rank_<rank>.log` | the operator | The output of every process, with wall time stamps of the operator |
| `first_logits.npz` | the coordinator | Logits of the first step of every prompt, float32 |
| `timeline_<rank>.jsonl.gz` | each rank, `Recorder.write` | Raw spans of that rank, on that rank's own raw monotonic clock. Only with `--timeline` |
| `samples_rank_<rank>.jsonl.gz` | each rank, `Sampler.write` | Raw counters of that process and its host, same clock |
| `samples_<host>.jsonl` | `timeline.write_host_samples` | The counters of one host, moved to the coordinator clock |
| `trace.json.gz` | `timeline.trace` | All spans in Trace Event Format. Open it at ui.perfetto.dev |
| `breakdown.csv` | `timeline.write_breakdown` | One row per step and rank: ms per kind, bytes per peer |
| `timeline.html` | `timeline.write_html` | The chart: per rank a compute row and a communication row, bytes per link, clock table |

The last four are derived. `python -m nocturne.engines.torch_tp.timeline <run_dir>` builds them again from the raw files and `record.json`.

## 2. Clocks

Two machines never share a clock. We align them in two independent ways and compare.

### 2.1 Operating system level, before every run

Tool: `scripts/17_timesync.sh` (it calls `nocturne/bench/probes/timesync.py`, standard library only).
It asks one NTP server with SNTP (RFC 4330). From the four time stamps, offset = ((t2 - t1) + (t3 - t4)) / 2. That is the server clock minus the host clock.
It sends 8 queries (4 in preflight) and keeps the one with the smallest round trip. It also reads what the OS time service reports: `sntp` on macOS, `chronyc tracking` or `timedatectl timesync-status` on Linux and WSL2, and `w32tm` on the Windows side of a WSL2 host.

- One server for all hosts: `time.apple.com`, or `NOCTURNE_NTP_SERVER`.
- Without flags the script measures and sets nothing. `--set` steps the clock and needs sudo: `sntp -sS` on macOS, `chronyc makestep` or systemd-timesyncd on Linux, `hwclock -s` on WSL2 after `17_timesync.ps1 -Set` on Windows.
- Recorded per run and per host by `hybrid_run.preflight`: `record.json`, `preflight.hosts.<host>.timesync`: the server, the offset, the round trip, the time of the measurement (`at`), the OS tool with its source and its own offset, and the time of the last sync as far as the OS tells it. That is `last_sync_utc` from `chronyc` and `last_sync` from `w32tm`. macOS has no such line. `last_set` is the time, the server and the method of the last `17_timesync.sh --set` on that host.
- The OS services keep their own servers. On gtx, chrony inside WSL2 follows the Hyper-V clock and an Ubuntu NTS server, and Windows follows `time.windows.com`. `--set` on plain Linux adds the common server to chrony before it steps. On WSL2 it only reads the Windows clock, so the common server is set on the Windows side (`17_timesync.ps1 -Set`). The offset we report is always measured by our own SNTP query against the one common server, whatever the OS follows.
- Precision: half the round trip to the server (`bound_s`).

Measured on the M5 on 2026-11-28, home Wi-Fi, a VPN client installed:

| time | offset to `time.apple.com` | bound |
|---|---|---|
| 06:44 | +59 ms | 30 ms |
| 07:09 | +120 ms | 32 ms |
| 07:49 | +82 ms | 44 ms |

So over this path the OS level number is good to about 30 to 45 ms, and the M5 clock moved by tens of ms within an hour.
That is not enough to order events inside a decode step. It is enough to detect a host whose clock is seconds off. A time server on the LAN would make the bound a fraction of a ms.

### 2.2 Engine level, at the start and at the end of every run

Code: `clock.measure` (coordinator), `clock.answer` (rank), called from `Coordinator._bring_up` and `Coordinator.clock_sync`.
The coordinator sends 16 requests on the control link of each rank. The rank answers each with its wall clock and its monotonic clock.
From the round with the smallest round trip: offset = remote minus local minus half the round trip. It is computed twice, for the monotonic clocks and for the wall clocks.

- The monotonic clock is the raw one, `CLOCK_MONOTONIC_RAW` (`clock.now_ns`). Every recorder time stamp uses it. The OS time service does not touch it. Python's `time.monotonic_ns()` is `CLOCK_MONOTONIC` on Linux, and the time service runs that clock faster or slower while it corrects the wall clock. Measured on 2026-11-28 on the pc, whose wall clock was 370 ms off: its `CLOCK_MONOTONIC` ran 0.8 to 1.9 percent fast, and its offset to gtx moved by 110 ms in 6 s. On the raw clock the same pair drifts by 14 ppm. `clock.<rank>.drift.wall_minus_mono_change_ns` shows how far the wall clock moved against the raw clock during a run (66 ms in 27 s on the pc).
- Recorded always, also without `--timeline`: `record.json`, `clock.<rank>.start` and `.end` (`offset_mono_ns`, `offset_wall_ns`, `rtt_ns`, `rtt_median_ns`), and `.drift` (the change between start and end, also in ppm).
- Every recorded event time is converted with it: `clock.to_coordinator`. The offset is interpolated linearly between the start and the end measurement, which removes a constant drift.
- Precision of one exchange: at most half its round trip, when the two directions take the same time. Between the two exchanges the offset is interpolated, which is exact for a constant drift. Two measurements cannot bound a drift that changes during a long run, so for events in the middle the bound is an estimate.
- Every CLOCK frame travels on the control link of its rank.
- Measured: on loopback the round trip is about 0.2 ms. Between machines it is the round trip of the network: 5 to 8 ms on the home Wi-Fi on 2026-11-28.
- The check in the coordinator process that it runs on the right host uses the name the host reports when the operator asks it. It protects a coordinator started by hand on the wrong machine. In an operator run it cannot fail, because the operator starts the process on that host.
- The two levels are compared in `hybrid_run.clock_report`: the OS level predicts (server minus coordinator) minus (server minus rank). `clock.<rank>.os.gap_ns` is the engine's wall offset minus that prediction. `disagree` is true when the gap is over 2 ms. `os_bound_ns` is the sum of the two OS bounds, so the reader can see whether the gap means anything.

Loopback check on 2026-11-28 (1.5B, three ranks on the M5, one physical clock): offsets of -3.5 and +2.0 microseconds at start, +7.8 and -8.5 at end, round trips 0.12 and 0.19 ms. The offsets are inside the bound, as they must be on one host.

## 3. The recorder

Code: `recorder.Recorder`. On with `--timeline`. Off by default, and then the engine's code path is unchanged.
The recorder keeps a cursor on the raw monotonic clock (`clock.now_ns`, section 2.2). `lap(kind)` closes the span from the cursor to now. So every instant of a step belongs to exactly one span.

| kind | what it covers | where it is closed |
|---|---|---|
| `compute` | Device work: embedding, attention, MLP, output head. It ends after a device sync (`torch.mps.synchronize` or `torch.cuda.synchronize`), so queued device work is inside the span | `GroupComm.all_reduce` (start), `RankRuntime.step` (before the hidden state is sent, after the head) |
| `host_copy` | Tensor from the device to the host, the sum of the partials on the host, the result back to the device with a sync | `GroupComm.all_reduce` (twice), `RankRuntime.step`, `RankRuntime.serve` |
| `send` | The call that hands a frame to the kernel. With the destination and the bytes of the frame | after every `Link.send` of a data frame |
| `allreduce_wait` | Blocked until a group peer's partial has arrived. With the source and the bytes | `GroupComm.all_reduce` |
| `recv_wait` | Blocked on the upstream link (a rank), or on a head rank's answer (the coordinator). With the source and the bytes | `RankRuntime.serve`, `Coordinator._step` |
| `other` | The coordinator's Python between steps: choosing the token, the loop | `Coordinator._step`, `Coordinator.generate` |

Every span carries the step number and the request number; the rank and the host are in the header line of the file. The request number travels in the upper bits of an existing header field of the TOKEN and HIDDEN frames, only in runs with `--timeline`.
On a host with more than one GPU, `compute` ends after a sync of the rank's own device (`torch.cuda.synchronize(device)`).

What a reader must know:

- A step of a rank starts when the frame that triggers it arrives. The wait before it is part of that step. For the first step of a request that wait includes the idle time since the previous request.
- `send` is the time to hand the bytes to the kernel. It is not the time on the wire.
- A wait is the time blocked. It contains the peer's own compute and the network in both directions. The recorder does not split these two. They can be separated later by subtracting the peer's busy time in the same window, which needs the aligned clocks of section 2.
- A large all-reduce frame (prefill of a long prompt in a group) is sent by a helper thread. Its `send` span is the hand over to that thread. "Large" was "over 1 MiB" until session 10 part 3. Since then it is "more than fits half the send buffer the kernel granted": still 1 MiB on the Macs, 212992 bytes on Linux with default settings, which is every 7B prefill of 30 tokens or more. So on gtx and pc the `send` and `allreduce_wait` spans of a TP prefill are not comparable with recordings made before part 3. The bytes on the wire are the same.
- The device sync at the end of `compute` exists only with the recorder. Without it the same wait happens inside the copy that follows. So the recorder moves time from `host_copy` to `compute`; it does not add work.

Overhead and identity, measured: section 6.

### 3.1 Several requests, the sender thread, and the overlap share (session 10, part 3)

Every span also carries the slot of its request. With several requests in flight the spans of two steps of one rank interleave; each still belongs to exactly one step.

With `--async-send` a rank has a second thread, the sender. It records two more kinds, which overlap the kinds above and are kept out of the per step bar:

| kind | what it covers |
|---|---|
| `tx_wait` | The sender waits until a tensor has reached the host (CUDA: the event after the non blocking copy; MPS and CPU: nothing to wait for, the compute thread made the copy) |
| `tx_send` | The sender hands a frame to the kernel. With the destination and the bytes |

In such a run the compute thread does no device sync. It closes a `compute` span every time it hands a tensor to the sender. On CUDA `compute` is then the time of the compute thread, which hands work to the device and goes on; the device time shows in `tx_wait` and in the GPU sampler. On MPS the compute thread still makes the copy, which waits for the device; that wait falls into the span that follows the hand over (`allreduce_wait` in a group, the next `compute` in a pipeline rank), so on a Mac the split between compute and wait is less sharp with the option than without it. Compare `compute` between a run with the option and a run without it only with this in mind.

**In flight.** A frame is in flight from the start of its send span (`send` or `tx_send`) to the end of the receiver's wait span for it (`recv_wait` or `allreduce_wait`). The k-th send on a socket is matched with the k-th receive on it. Two ranks can share two sockets (a first stage rank with head rows has a group link and a control link to the coordinator), so every send and wait span carries its channel and the match is per channel. `timeline.in_flight` does this, and `record.json` `timeline.overlap.unmatched_links` lists any link where the counts or the bytes do not agree. Treat a run with a non empty list as having no overlap share. The verification runs and their lists are in `experiments/session10_cuda_pair/VERIFY_overlap.md`.

**Busy share and overlap share**, per rank and per device (`timeline.overlap`):

- Time base: the time in which at least one request is in work (the union of the request intervals, coordinator clock).
- Busy share: the length of the `compute` spans inside the time base, divided by it. For a device, the union over its ranks.
- Overlap share: the length of the `compute` time during which a frame the rank sent, or a frame addressed to it, is in flight, divided by the time base. The record also gives the two parts alone (`overlap_out_share`, `overlap_in_share`).

Limits.

- "In flight" ends when the receiver reads the frame, not when the frame reaches the receiver's machine. The recorder cannot see the arrival. So a frame that waits in the receiver's socket buffer while the receiver computes still counts as in flight. When the receiver is already waiting, the two moments are the same.
- Because of that, a run with every option off already shows an overlap share in a TP group: both ranks compute their partials at the same time, the faster one sends first, and its frame waits until the slower one has finished. In the check `OVc_pair_tp2_off_r1` (gtx and pc, one TP group, Wi-Fi, options off) the overlap share was 24 and 30 percent, most of it the "addressed to it" part. The part that answers "does a machine compute while its own data travels" is `overlap_out_share`.
- The start of a frame is on the sender's clock and its end on the receiver's, so each in flight interval carries the alignment error of section 2.2 (half a control link round trip, a few ms on Wi-Fi, well under 0.1 ms on loopback).
- On a fast link a frame is in flight for a short time, so the overlap share is small even when the busy share is high. The busy share is the number that shows whether a machine still waits.

## 4. The sampler

Code: `recorder.Sampler`, a thread in every rank process, on with `--timeline`.
Every 20 ms it reads cumulative counters with `psutil`: CPU seconds of the rank process, and, in one rank per host, the busy CPU seconds of the host and the bytes in and out of the interfaces the rank listens on. Rates are formed later from differences.
On a host with a CUDA rank the host sampler also runs `nvidia-smi` every 100 ms, for every GPU of the host (index, utilization, memory used). It ran on gtx under WSL2 in the checks of 2026-11-28.
On a Mac there is no GPU sampler: `powermetrics` needs sudo and is not used.

Limits: CPU time is counted by the OS in ticks of 10 ms, so one 20 ms interval is coarse; use windows of a second or more. On loopback the interface counts every byte twice (in and out). `nvidia-smi` utilization is an average over its own window.

## 5. Bytes

Bytes are counted twice, independently.

1. Link counters: `transport.Counters`, per link and per direction, split into prefill and decode. They go into `record.json`, `link_counters` and `frames.bytes_per_step`. They exist in every run.
2. The recorder: the bytes of every `send` span. `timeline.summary` sums them per directed link.

Both count frame bytes: header, dims and payload. They do not count TCP and IP headers. A test requires the two to be equal (`test_recorded_spans_tile_every_step_and_bytes_equal_the_transport_counters`), and every `--timeline` run prints both.
"Total bytes per configuration" in `timeline.html` is the sum over all links: per decode step, over all decode steps, and over the whole run with prefill.
Bytes of shipped weights are separate: `record.json`, `shipping.ranks.<rank>`.

## 6. Table of reported numbers

| number | file and key | written by | clock | precision and limits |
|---|---|---|---|---|
| ms per decode step, per prompt | `record.json` `prompts[].ms_per_step` | `Coordinator.generate` | coordinator wall clock | Whole step, everything included. Background only (T1) |
| ms per step by kind, per rank | `record.json` `timeline.per_rank_ms_per_step`; `breakdown.csv` | `timeline.summary`, `timeline.breakdown` | each rank's monotonic clock; durations need no alignment | Microsecond stamps. With the recorder a step is about 2 percent slower (see below) |
| when a span happened, across ranks | `trace.json.gz`, `timeline.html` | `timeline.load` | coordinator monotonic clock | Half the control link round trip of that rank (section 2.2) |
| frames per decode step | `record.json` `frames.measured_per_step`, `.predicted` | `coordinate.run_job` from `transport.Counters`; `plan.frames_per_step` | none | Exact counts. `formula_holds` says whether they agree |
| bytes per decode step | `record.json` `frames.bytes_per_step`; `timeline.decode_bytes_per_step` | link counters; recorder | none | Exact frame bytes, no TCP or IP overhead |
| bytes per link | `record.json` `frames.per_link_per_step`, `timeline.links`; table in `timeline.html` | same | none | Sender side counts |
| clock offset per rank | `record.json` `clock.<rank>` | `Coordinator.clock_record`, `hybrid_run.clock_report` | both | Half the round trip |
| host clock against NTP | `record.json` `preflight.hosts.<host>.timesync` | `hybrid_run.preflight` | host wall clock | Half the round trip to the server, tens of ms from home |
| weights shipped, cached, time | `record.json` `shipping.ranks.<rank>` | `shipping.send_slices`, `shipping.receive_slices` | wall clock of each side | Exact bytes. Time is background |
| planned and loaded bytes, memory | `record.json` `planned[]`, `ranks[]` | `plan.plan`; `RankRuntime.metrics` | none | As in engine GGUF part 1 |
| CPU busy, process CPU, network bytes per host | `record.json` `timeline.hosts`; `samples_<host>.jsonl` | `timeline.write_host_samples` | coordinator monotonic clock | OS tick of 10 ms; whole run averages in the record |
| GPU utilization (CUDA) | same | same | same | `nvidia-smi`, 100 ms. Not on Metal |
| busy share and overlap share, per rank and per device | `record.json` `timeline.overlap`; figure and table in `timeline.html` | `timeline.overlap` | coordinator monotonic clock | Section 3.1. In flight intervals carry the alignment error; with `--async-send` compute is thread time |
| tokens per second of the whole system | `record.json` `throughput` | `coordinate.run_job` | coordinator wall clock | All prompts, prefill included. Background (T1) |
| prefill frames | `record.json` `frames.prefill_measured`, `.prefill_predicted` | link counters; `plan.frames_prefill` | none | Exact counts. `prefill_holds` says whether they agree |
| tokens, identical prompts, accuracy | `record.json` `prompts[]`, `correctness` | the coordinator; `hybrid_run.compare` | none | Exact |

Recorder overhead and identity, measured on 2026-11-28 (`experiments/session10_cuda_pair/VERIFY.md`): on the M5, 1.5B with TP over three ranks, a decode step took 417.5 ms without the recorder and 426.7 ms with it (four and three runs), so the recorder costs 9.3 ms per step, 2.2 percent. On 7B the difference is inside the day to day spread. Tokens were identical with the recorder on and off in every pair.

Measured between machines on the home Wi-Fi on 2026-11-28 (same file): the control link round trip was 5 to 8 ms, so the alignment between machines is good to about 3 to 4 ms. The wall clocks differed by 24 to 82 ms. The offset drifted by 6 to 14 ppm during a run, which is why it is measured twice and interpolated.

## 7. The method of 23 November, and what the recorder replaces

The charts shown on 2026-11-23 were made from packet captures (`experiments/session9_timeline/existing_data/`, `tl.py`, `fig23.py`).

- `tcpdump` ran on the requester and on one worker, on the engine's port, with sudo.
- States were inferred from gaps between packets: a gap before an outgoing packet was called compute, a gap before an incoming packet was called waiting, gaps under 1 ms were called transfer.
- The two captures were aligned with ICMP echo packets seen by both (`capture.clock_offset`: the median over the echoes, each corrected by half its round trip).
- GPU activity came from `powermetrics` on the Macs, at 100 ms, with sudo.

Limits of that method:

- It infers. A gap before a send can be compute, a copy to the host, or Python. It cannot tell them apart.
- It sees two machines, the requester and one worker. Links between workers were not captured.
- It cannot see loopback runs, and it needs sudo on every host.
- The M5 and Windows drop ICMP, so the alignment does not work with gtx or with the M5 as the far side.
- Its bytes include TCP and IP headers and acknowledgements, so they are not the bytes the engine sends.

What the recorder replaces: the state inference (now measured inside each process, for any number of ranks), the alignment (now on the control link, no ICMP, no sudo), and the per link bytes (now exact frame bytes for every link, worker to worker included).
What a packet capture is still good for: an independent check of bytes on the wire, and llama.cpp runs, where we cannot put a recorder inside the engine.

## 8. Reproduce

```bash
bash scripts/17_timesync.sh
```

```bash
python -m nocturne.engines.torch_tp.hybrid_run --placement <placement.yaml> --session-dir experiments/<session> --run-id <id> --timeline
```

```bash
python -m nocturne.engines.torch_tp.timeline experiments/<session>/runs/<id> --req 9 --from-step 100 --steps 3
```
