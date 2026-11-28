# Codex review: session 10, part 1

Reviewed on 2026-11-28. Base: `0e96d48`. Reviewed HEAD: `499689f0df1679731d178f3d3e02773cf0e4bc79`, plus the working tree.
Read `docs/ENGINE_DESIGN.md` first, then `docs/MEASUREMENT.md`.
Scope: `git diff 0e96d48 -- nocturne scripts tests docs` and the requested supporting files.
No implementation changes were made.

## Findings

### F1. High: a hung coordinator leaves the operator waiting forever

Location: `nocturne/engines/torch_tp/operate.py:201-209`.

`run_coordinator` blocks while iterating over stdout. It has no deadline or independent liveness check. The SSH connection has a connect timeout, but no configured keepalive failure limit. A stopped coordinator can keep the connection and stdout open forever. A lost host can also leave SSH waiting on a half-open connection.

The operator never reaches collection or cleanup. Its last `record.json` still has `outcome: null`. The coordinator writes its record only when `run_job` leaves its `finally` block. That does not help a hung process.

Check: start a disposable tiny-model run. Send SIGSTOP to its coordinator after startup. Leave the operator and ranks running long enough to observe the stall, then send SIGCONT and clean up. Require a bounded failure result and a saved reason. Also test a dropped SSH connection. Locally, I substituted a silent sleeping child for the spawned coordinator. The operator remained blocked until I terminated that child. This checks the blocking path, not a real host failure.

Fix direction: monitor the child independently of log reading. Provide a configurable failure deadline or watchdog. Save the failure before cleanup. Keep this separate from timing-based performance acceptance under T1.

### F2. Medium: run-id validation happens after filesystem writes and accepts parent directory names

Locations: `nocturne/engines/torch_tp/hybrid_run.py:293-298`, `nocturne/engines/torch_tp/operate.py:84-86`.

`hybrid_run` constructs the output path, creates it, and opens its log before `Operator` validates the tag. An invalid run id such as `../outside` therefore writes outside `runs/`, even though setup later rejects it. An absolute run id also replaces the intended base path.

The operator's regex additionally accepts `.` and `..`. Those names reach remote `host_dir` paths and collapse the per-run directory. The regex uses `$` and `match`, so it also accepts a trailing newline. `default_run_dir` already handles these cases more strictly, but this operator path bypasses it.

Check: in a temporary session directory, call `hybrid_run.main` with valid disposable peers and placement files and `--run-id ../outside`. I reproduced return code 4 and a `setup_failed` record at `<session>/outside/record.json`. I also confirmed that constructing `Operator` accepts `.`, `..`, and `x\n`. Reject these before any directory or log is created. Use one shared full-match validator.

### F3. Medium: safetensors column shipping holds two representations of the slice

Locations: `nocturne/engines/torch_tp/shipping.py:147-153`, `175-177`, `227-228`.

`SafetensorsSource.read` can return a noncontiguous column view. `tensor_bytes` makes a contiguous copy of the entire view. The caller retains the original tensor as `t` while the returned memoryview retains the new storage. This happens both during manifest hashing and during sending. Four MiB chunks limit frame size, but do not limit this allocation. Parallel shipping repeats it for each active rank.

This violates the stated rule that a slice is never held twice in memory. Large `o_proj` and `down_proj` cuts can cause substantial extra host memory use.

Check: create the model with `tests.bench.test_torch_tp.write_tiny_model`. Read `model.layers.0.mlp.down_proj.weight` through `SafetensorsSource` with `cols=slice(0, 64)`. Call `tensor_bytes(t)`. I confirmed that `t.is_contiguous()` is false and that `t.data_ptr()` differs from `mv.obj.__array_interface__['data'][0]`. Both objects remain live. Add a large column-slice peak-memory check. Stream bounded row segments, or transfer ownership of one contiguous representation before hashing and sending.

### F4. Medium: recorder synchronization can wait for the wrong CUDA device

Location: `nocturne/engines/torch_tp/hybrid.py:139-142`. Also the new recorder sync calls in `RankRuntime.step`.

The recorder calls `sync(self.device)` before closing compute spans. That helper calls `torch.cuda.synchronize()` without a device argument. The process never sets its current CUDA device to the placement's device. A rank placed on `cuda:1` can therefore synchronize GPU 0. Pending GPU 1 work then appears inside the subsequent host copy instead of compute. GPU 0 activity can also inflate the span.

The helper predates this diff. Its new use as the recorder's measurement boundary introduces this reporting error. This is not evidence of changed tokens.

Check: patch `torch.cuda.synchronize` with a mock and call `hybrid.sync('cuda:1')`. I confirmed the call has no arguments. On a two-GPU host, queue work on GPU 1 and verify that the recorder waits for that device. Pass the explicit device to synchronization. Existing CUDA memory metric helpers also omit the device, but that older issue is outside this finding.

### F5. Medium: end clock synchronization sends CLOCK on pipeline data links

Locations: `nocturne/engines/torch_tp/hybrid.py:349-351`, `720-727`.

`Coordinator.clock_sync` calls `rank.forward_control(KIND_CLOCK, ...)`. That sends CLOCK on the pipeline links. Each downstream leader forwards it again. Only the measurement replies and requests then use the control link.

This directly violates the supplied rule that CLOCK and WEIGHTS are control-link kinds only. It occurs in ordinary PP and hybrid runs, including runs without a recorder. Successful token and byte-count tests do not detect the violation because CLOCK is excluded from data-frame counters.

Check: instrument `Link.send` with link-role labels in a three-stage loopback run. Assert that every CLOCK uses a control link. The current end measurement fails that assertion. Keep rank control reception independent of its upstream data receive if the control-only rule remains mandatory.

## Other requested decisions

- I would not accept the aux1 change under the stated strict rule without an explicit exception. TOKEN and HIDDEN now carry different field values in every run, including recorder-off runs. Header size and payload layout remain unchanged. Current receivers mask `FLAG_LOGITS`, so I found no token error from these upper bits. Either approve this protocol extension explicitly or preserve the old fields when recording is off.
- The clock offset signs are consistent. `estimate` returns remote minus coordinator. Timeline conversion subtracts that offset. The OS comparison uses the same sign. Interpolation is correct for an affine monotonic-clock drift. Two samples cannot establish a bound on nonlinear drift during a long run. Half the measured round trip bounds the midpoint error for that exchange even when rank scheduling delays its answer. It is not a guaranteed error bound for every event throughout the run.
- Cached files are trusted by name and size, as documented. Same-size corruption therefore survives a warm-cache run. This is an explicit integrity tradeoff, not a new hidden guarantee. For stronger integrity, hash each unique cached file before use and request it again on mismatch. For 20 GB, this adds a sequential 20 GB read and SHA256 pass per validation. Measure that cost on the target disk. Size and mtime memoization does not detect every same-size corruption.
- Manifest hashes are checked before the normal receive path uses them as path components. I found no direct `../` escape through those hash strings. Concurrent receivers use distinct temporary names and atomic replacement. This does not establish containment against pre-existing local symlinks.
- The reviewed numerical paths use the same source slicing and loader logic for shipped and local weights. I found no tensor mutation added by the recorder. The focused tests passed for the supplied tied-weight, zero-head, and GGUF cases. I did not independently run a real CUDA pair or a remote failure test.
- `VERIFY.md` now reports real CUDA, network, and Linux-coordinator checks. This is newer evidence than the pasted request's "Not run yet" paragraph. Those are reported prior checks, not checks I performed.

## Validation

- Requested focused command: **63 passed**, one read-only NumPy warning, 69.61 seconds.
- Full `~/dev/nocturne-torch/bin/python -m pytest tests/bench -q`: **323 passed**, one read-only NumPy warning, 209.35 seconds. No skips were reported.
- `bash -n scripts/17_timesync.sh` and `/bin/bash -n scripts/17_timesync.sh`: passed. `/bin/bash` is version 3.2.57. No clock was set. PowerShell was not executed.
- Extra checks used temporary models and disposable local child processes. No dependencies were installed or upgraded.

The existing mutation checks do not cover a live silent coordinator, output creation before tag validation, duplicate column-slice storage, explicit CUDA device selection, or the link role carrying CLOCK. Passing those mutations does not rule out F1 to F5.

## Disposition (Claude Code, after the Codex review)

Commit `8fa7bbd`. All five findings were reproduced as described and fixed. Tests: 332 pass. Mutations: 33 of 33 caught (`mutations_round4.json`, `mutations_round4b.json`).

| finding | rating | disposition |
|---|---|---|
| F1 a hung coordinator leaves the operator waiting for ever | high | Fixed. `Operator.run_coordinator` watches the process apart from its log. The coordinator prints a sign of life every 20 s from its own thread (`coordinate.start_heartbeat`). Without one for `--stall-timeout` (300 s, 0 waits for ever) the operator ends the process, saves `outcome: coordinator_stalled` with the reason before cleanup, then collects and cleans up. ssh now uses keepalive, so a host that left the network ends the connection after about a minute. The final `pkill` is followed by SIGKILL for what ignores SIGTERM. It is a liveness check, not a time limit (T1). Tests: a silent child, a child that only beats, a child stopped with SIGSTOP, and `hybrid_run.main` with a silent coordinator |
| F2 run id checked after files were written | medium | Fixed. `operate.valid_tag` (full match, no `.` or `..`) is called in `hybrid_run.main` before any folder or log is created, and by `Operator`. Test: `test_run_id_is_checked_before_anything_is_written` |
| F3 a column slice held twice | medium | Fixed. `shipping.iter_bytes` hashes and sends in pieces; a tensor that is not contiguous is copied a few rows at a time. Test: `test_a_column_slice_is_never_copied_whole`. What remains: safetensors itself reads the whole matrix to give a column range, so the full matrix is in memory once while its slice is sent. GGUF slices come from a memory map and do not have this |
| F4 device sync on the wrong CUDA device | medium | Fixed. `torch.cuda.synchronize(device)`. Test with a mock. Not run on a host with two GPUs: we have none |
| F5 CLOCK on pipeline links | medium | Fixed. The end measurement sends its begin frame on each rank's ctl link. A rank of a later stage waits on its upstream link and its ctl link with `select`. Test: `test_clock_frames_use_only_ctl_links_and_the_request_number_only_with_the_recorder` records the link of every CLOCK frame in a three stage run |
| aux1 request number | decision | Taken as proposed: the upper bits are set only in runs with the recorder. Without it TOKEN and HIDDEN frames are what they were. The same test checks both cases. Ohad still has to approve the extension for recorder runs |
| clock bound | decision | Wording corrected in `clock.py` and `docs/MEASUREMENT.md` 2.2: half the round trip bounds one exchange; between two exchanges the interpolation is exact only for a constant drift |
| cache trusted by name and size | decision | Kept as the default and documented. New option `--verify-cache`: every rank reads and hashes its cached slices before use and asks again for one that differs. Test: `test_a_changed_cache_file_is_trusted_by_name_and_size_and_caught_with_verify` |

Gate: no open high finding.
