# Review request for Codex: session 10, part 1

You did not write this code. Please look for errors, not for style. Rate each finding high, medium or low, and say how to check it.
Put your answer in `experiments/session10_cuda_pair/REVIEW_CODEX.md` (Ohad pastes it). Short sentences. No em dash, no en dash.

## What changed

The torch engine (`nocturne/engines/torch_tp`) got four things.
Start SHA: `0e96d48`. The diff is `git diff 0e96d48 -- nocturne scripts tests docs`.

1. **Clock sync.** `scripts/17_timesync.sh` measures each host against one NTP server. The engine measures every rank's clock offset on the control link, at the start and at the end of a run.
2. **Recorder.** With `--timeline` every rank records spans (compute, copy, send, waits) and a sampler reads CPU and network counters. `timeline.py` aligns them and writes a chart and tables.
3. **Any host as coordinator.** `hybrid_run` is now only the operator (ssh, files). The coordinator process plans and runs on the host the placement names.
4. **Shipped weights.** The coordinator sends every rank its slices. A rank needs no model file. It keeps a cache by sha256.

Read `docs/ENGINE_DESIGN.md` first. It is the design the code claims to follow. `docs/MEASUREMENT.md` says where every number comes from.

## Rules the code must follow

1. Transport: only two new frame kinds are allowed, CLOCK and WEIGHTS, both on the control link. No change to the header or to any data frame. (The request number uses the upper bits of `aux1` in TOKEN and HIDDEN frames. Is that a change you accept?)
2. Numerics must not change. Tokens must be identical with the timeline on or off, and with shipped or local weights.
3. The recorder is off by default. Off must mean the old code path.
4. Never bind `0.0.0.0`. Security is deferred by ADR-009, but a hash from the wire must not name a file outside the cache.
5. Streaming: a slice is never held twice in memory. A quantized row split layer is cut only at block boundaries.
6. Time is background (T1). Nothing may gate on a time.
7. No new dependency except `psutil`. Torch is not installed or upgraded.
8. Scripts run on bash 3.2 and bash 4+.

## Files to read, in this order

| file | what to check |
|---|---|
| `nocturne/engines/torch_tp/transport.py` | The two new kinds and their documentation. `last_sent`, `last_recv`. Nothing else changed |
| `nocturne/engines/torch_tp/clock.py` | The offset formula and its sign. Smallest round trip. Interpolation in `to_coordinator`. `compare_with_os` |
| `nocturne/engines/torch_tp/hybrid.py` | Where `clock.answer` and `clock.measure` run, at start and at end, for ranks of every stage. Recorder laps in `GroupComm.all_reduce`, `RankRuntime.step`, `RankRuntime.serve`, `Coordinator._step`. The shipping order in `rank_main` and `Coordinator._bring_up`. Error paths |
| `nocturne/engines/torch_tp/recorder.py` | The cursor model. The sampler thread and the `nvidia-smi` process: can either be left running? |
| `nocturne/engines/torch_tp/timeline.py`, `timeline_page.html` | Alignment, breakdown per step, bytes per link, the window choice, the trace |
| `nocturne/engines/torch_tp/shipping.py` | Manifest, slice keys, chunks, CRC32, sha256, the cache, the temporary files, `CacheSource`, `ship_all` threads |
| `nocturne/engines/torch_tp/loader.py`, `config.py`, `gguf_source.py` | `load_from_source` with `plain`. The config through JSON. `open_gguf` keeps one source per file: is that safe across threads? |
| `nocturne/engines/torch_tp/plan.py` | `slots` and `coordinator_host` against `plan` and `validate` |
| `nocturne/engines/torch_tp/operate.py`, `coordinate.py`, `hybrid_run.py` | Who runs where. Shell quoting of paths and tags. The tar collection (`safe_extract`). Preflight. What the record holds when the coordinator process dies |
| `nocturne/bench/probes/timesync.py`, `scripts/17_timesync.sh`, `17_timesync.ps1` | SNTP math, the parsers of `chronyc`, `timedatectl`, `sntp`, `w32tm`. What `--set` does on each OS |

## Questions

1. Clock: is the sign of every offset right all the way to `timeline.html`? Is half the round trip a fair bound when a rank is busy answering? Is linear interpolation between two measurements right when the run is long?
2. Recorder: does any path with the recorder on change a tensor or the order of frames? Is there any instant of a step that belongs to no span, or to two? Is the device sync placed so that `compute` is real device time on MPS and on CUDA?
3. Bytes: the recorder and the link counters must agree. Is there a frame that one counts and the other does not (async sends of large frames, control frames, the HEAD frame)?
4. Shipping protocol: can the coordinator and a rank deadlock (both sending, kernel buffers full)? What happens when a rank dies in the middle of a slice? When two ranks on one host fetch the same slice at once? When the cache disk fills up?
5. Shipping content: are the shipped bytes exactly what the local loader would read, for safetensors and GGUF, for tied embeddings, for a rank with zero heads, for llama (un-permuted Q and K rows)? Can a manifest from the wire make the rank write or read outside its cache directory?
6. Cache: a cached file is trusted by its name and size. Is that acceptable? What would you check instead, and at what cost for 20 GB?
7. Operator split: is there still a place where the operator must hold the model, or must be a Mac? Can a run id or a path with spaces break a command? Can `pkill -f "--tag <id>"` kill the wrong process?
8. Failure: when the coordinator process on another host dies or hangs, does the operator end, and does `record.json` say why?
9. Tests: think of a break that the tests in `tests/bench/test_session10_*.py` would miss. `experiments/session10_cuda_pair/VERIFY.md` lists the breaks that were tried (`mutate.py`).

## How to run the tests

Python is the torch venv `~/dev/nocturne-torch/bin/python` (torch 2.14.1, MPS and CPU). Do not install or upgrade torch.

```bash
~/dev/nocturne-torch/bin/python -m pytest tests/bench/test_session10_timeline.py tests/bench/test_session10_coordinator.py tests/bench/test_session10_shipping.py -q
```

```bash
~/dev/nocturne-torch/bin/python -m pytest tests/bench -q
```

The first command takes about a minute. The second takes about three minutes. The GGUF cases need the llama.cpp converter at `~/dev/llama.cpp` and are skipped without it.

## Not run yet

gtx and Lior's laptop were off on 2026-11-28. So nothing in this change ran on CUDA, over a real network, or with a Linux coordinator. The `nvidia-smi` sampler and the Linux branches of `timesync.py` are covered only by parser tests.
