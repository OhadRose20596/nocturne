# Review (pass 2, fixes only)

Reviewer: a second fresh reviewer, with `REVIEW_pass1.md`, `fixes_pass1.diff`, `ENGINE_DESIGN.md` and the requirements (sections 2 to 4) only. Date: 2026-11-28.

Scope: are the fixes for F1 to F17 right, and do they add defects. Line numbers are of the files at commit `efad24d`. Nothing was edited except this file.
Evidence used besides reading: one targeted test, two small shell and Python checks on this Mac, and the records of the two runs made with the fixed code (`runs/K1_gtxcoord_air_pp2_7b_q8_r1` and `_r2`, coordinator on gtx, read only).

## Status of the pass 1 findings

Totals: fixed 12, partly fixed 4, not fixed 0, left on purpose 1.

| finding | status | how checked |
|---|---|---|
| F1 `pkill` by run tag | fixed | Read `operate.kill_argv`, `Operator.stop`, `launch.py:137`. Ran `pgrep -f --` with the pattern for tag `T.a-b_r1` against four sleepers on this Mac: the exact tag matched (with and without a following argument), `T.a-b_r13` and `TXa-b_r1` did not, and the `bash -lc` wrapper did not match itself. The K1 records list the new command on air (macOS) and gtx (Linux), and their logs hold no `pkill on ... failed` line, so the return code was 0 or 1 on both. The operator cannot match itself: `hybrid_run` takes `--run-id`, `launch` takes no `--tag`. `launch.py` does not log the return code. See N1. |
| F2 shipped bytes and planned bytes for GGUF | partly fixed | Read `hybrid_run.py:390-395` and design section 3.8. The record now holds manifest, loaded and planned bytes per rank, and the design states the rule. K1 (7B Q8_0, rank s1/0): loaded = planned = 5036495872, manifest minus loaded = 431104. So the explanation holds for that run. Still open: equality is not made true for GGUF, no test pins the gap for a real model, and the new field cannot show an unexplained gap. See N6. |
| F3 loopback test that could not fail | fixed | Read `test_session10_coordinator.py:292-293`: `pgrep -f --`, return code 1 asserted, the empty assertion removed. |
| F4 cache and memo keys | fixed | Read `code_stamp`, `load_memo`, `save_memo`. A memo in the old flat format has no `code` key and is dropped. The cache is addressed by content, so a dropped memo gives new hashes and a changed slice is shipped again. `fsync` is before the rename. All six `CODE_FILES` exist. The optional re-hash flag was not added. |
| F5 DONE before every slice arrived | fixed | Read `shipping.py:298-304`. `missing_now` is never empty when `cur` is set, so the message cannot raise. Ran `test_a_transfer_that_ends_early_is_rejected_and_leaves_no_partial_file`: 1 passed. |
| F6 `.part` files left behind | partly fixed | Read the `try/finally`. An exception or a SIGTERM (new handler) now removes the file. A SIGKILL or a power loss still leaves it, and nothing sweeps old `.part` files. Two ranks of one host still check free space alone. One hole in the `finally`: N5. |
| F7 two wire values | fixed | Read `default_run_dir` and `check_manifest` (`fullmatch` in both). The rank name is also safe: `recorder.file_name` replaces the slash. Remark: `operate.TAG_OK` still uses `$` with `match` and accepts `X\n` (checked). That value is the operator's own `--run-id`, not a wire value. |
| F8 a completed run recorded as failed | fixed | Read `Operator.collect` (per host `try`) and `run_job` (results first, then the end clock in its own `try`). No test for either part. When the coordinator host's own collect fails the text still says the process "left no record". See N4. |
| F9 shipping failure | partly fixed | Fixed: the ERROR frame is read after a send error (`shipping.py:390-399`), the constructor closes the control links and the listener (`hybrid.py:553-555`). Not fixed, and no reason given in code or docs: `_bring_up` still joins every ship thread before it raises, and `rec["shipping"]` is still not written when the constructor fails (`co` is `None` in `run_job`). See N7. |
| F10 design text on wire changes | fixed | Read design section 6 against `REQ_SHIFT` in `hybrid.py` and the `transport.py` docstring. |
| F11 time of the last sync, common server | partly fixed | Fixed: `Ref time (UTC)` is parsed (K1 holds `last_sync_utc` for gtx), `last_set` is reported, the macOS limit is stated in the probe docstring. Not fixed: the `chronyc add server` line is not reached on WSL2, the timesyncd branch is unchanged, `docs/MEASUREMENT.md` is not updated. See N10. |
| F12 output names | left on purpose | Design section 10 now names the gzipped files and gives the size as the reason. Acceptable. The note does not mention that rank and host are in the header line only. |
| F13 GPU samples | fixed | Read `start_sampler`, `Sampler.__init__`, `smi_argv`, `parse_smi`, `write_host_samples`. K1: `timeline.hosts.gtx.gpu` is `{"0": ...}` with 2943 samples, so the three column line parses on the real host. See N9. |
| F14 preflight quoting, setup errors | fixed | Read `hybrid_run.py:174` and `:325-329`, and `test_setup_errors_leave_a_record`. All five `python` entries of the real `peers.yaml` are single tokens, so quoting does not break them. |
| F15 per host model path | fixed | Read `launch.py:68`, `:104` and `__main__.py:65-69`. The `plan` command finds its peer by the `host` name of `peers.yaml` and falls back to the shared entry. No test. |
| F16 test gaps | fixed | Two new tests (operator with shipping in processes, `hybrid_run.main` on a tiny GGUF). The GGUF test can run on this machine (converter, `gguf` and `psutil` present). Part 3 (device sync on MPS and CUDA) cannot be a unit test and stays with the recorded runs. See N11. |
| F17 last compute of a step | fixed | Read `hybrid.py:289-291`. The head path returns before it and the forward path is excluded, so no step gets two compute spans in a row. The lap is inside the step that `begin` opened. The new assertion fails without the fix (the last span would be `host_copy`). |

## New findings

Count: high 0, medium 0, low 13.

None gives wrong tokens, corrupts the cache or hangs a run in a path I could reach. The slice cache is safe under the new SIGTERM handler: a file gets its cache name only by `os.replace` after `fsync` and the hash check, so an exit at any point leaves at most a `.part` file.

### N1 (low) After the SIGTERM handler, `pkill` alone no longer stops a rank that is inside a C call, and a stopped coordinator leaves a record with no outcome

File: `nocturne/engines/torch_tp/coordinate.py:147-150`, `__main__.py:52-54`, `operate.py:234-237`, `coordinate.py:135-143`.

What is wrong:
1. SIGTERM is now a Python handler that raises `SystemExit` in the main thread. It runs only when the main thread returns to Python. Before the fix SIGTERM ended the process at once. `Operator.stop` sends SIGTERM once per host and never SIGKILL to a remote process (only local wrappers get SIGKILL from `Runner.stop_all`).
2. A coordinator ended by SIGTERM runs its `finally`, shuts the ranks down and saves `coordinator_record.json` with `outcome: null` and no `error`. The operator then records `outcome: error` with the text `None`.

Checked: a small script with the same handler. A main thread blocked in `queue.get()`, `socket.recv()` or `Thread.join()` exits with code 143 and runs its `finally`. So the F1 scenarios (rank blocked on a link or on the listener queue) do end.

Failure scenario: a rank on gtx hangs inside a driver call (`torch.cuda.synchronize()` that never returns). The operator's `pkill` sets the flag, the handler never runs, the process stays and keeps its port and VRAM. I did not see this happen. It is the one case the old default action covered.

Suggested fix: after the `pkill`, wait a few seconds and send `pkill -KILL -f --` with the same pattern. In `run_job`, catch `SystemExit` and `KeyboardInterrupt`, set `outcome: terminated`, and raise again.

### N2 (low) Two operator shell calls can raise `TimeoutExpired` where nothing handles it

File: `nocturne/engines/torch_tp/operate.py:121-125` with `hybrid_run.py:334`, and `operate.py:234-237`.

What is wrong:
1. `Operator.hostname` promises the `peers.yaml` name "when it does not answer". A failed ssh (return code 255) does fall back. A host that accepts the connection and then stalls for 20 s raises `subprocess.TimeoutExpired`. `op.job(...)` is called outside any `try` in `hybrid_run.main`, so the run ends with a traceback and no `record.json`. This is new: before the fix `job()` ran no command.
2. In `Operator.stop` the `pkill` loop has one timeout of 30 s per host and no `try`. A host that stalls raises out of `stop`, and the hosts after it in sorted order get no `pkill`. The structure is older, but it matters now that the `pkill` works.

Failure scenario: the Air goes to sleep between preflight and the job (it sleeps after a minute). `ssh air hostname` hangs. The operator dies with a traceback, the run folder holds only `coordinator.log`.

Suggested fix: catch `(OSError, subprocess.SubprocessError)` in `hostname` and fall back. Put the body of the `pkill` loop in a `try` and log the host.

### N3 (low) The coordinator host check can no longer fail in an operator run

File: `nocturne/engines/torch_tp/operate.py:128-131`, `coordinate.py:22-27`, `hybrid_run.py:332`, `launch.py:84`.

What is wrong: this change is in the fix diff but answers no pass 1 finding. The job now carries the name the host reports over ssh. `check_host` on that host compares its own name with it. Both come from the same machine, so the check always passes. Nothing compares the machine behind the ssh alias with the `host` field of `peers.yaml` any more.
`record.json` still stores the `peers.yaml` name in `coordinator.hostname`, so it can differ from `host` in the coordinator record and from `planner_host`. `launch.py:84` still compares with the `peers.yaml` name, case sensitive, so the two launchers now behave differently for a stale name.

Failure scenario: an ssh alias of `peers.yaml` points at another machine after a network change. The run starts there and nothing says so. The only trace is `planner_host` in the record.

Suggested fix: keep the live name for the check, store both names in the record (`hostname_peers`, `hostname_live`), and log a line when they differ.

### N4 (low) One failed end clock exchange drops the end measurement of every rank, and shutdown then runs the exchange a second time

File: `nocturne/engines/torch_tp/coordinate.py:117-122`, `hybrid.py:738-746`, `clock.py:48-56`, `hybrid_run.py:383`.

What is wrong:
1. On a failure `run_job` sets `co.clock["end"] = {}`. Ranks that were measured before the failing rank lose their end value. All ranks then fall back to the start offset, with no drift.
2. Because `clock["end"]` is empty, `Coordinator.shutdown` calls `clock_sync` again. If it works this time, the record holds end values and `clock_end_error` together. A stage 0 rank that is still inside `clock.answer` from the first attempt receives the second BEGIN on its control link and answers it as a request (`answer` replies to everything except DONE). Every reply the coordinator then reads belongs to the round before, so the offset is biased. This needs a failure that passes on the second attempt, which I could not construct from a real fault.
3. `clock_end_error` stays in `coordinator_record.json`. The operator does not copy it into `record.json`.
4. No test drives this branch.

Failure scenario: the rank on the Air drops off Wi-Fi after the last token. The run is `completed`. The end clock of the rank on gtx, which answered, is thrown away, and the shutdown waits another 20 s on the dead link.

Suggested fix: keep what was measured and store the error per rank. Set a flag so `shutdown` does not repeat the exchange. Let `answer` ignore a BEGIN. Add `clock_end_error` to the copied keys.

### N5 (low) The cleanup in `receive_slices` can be skipped when `close()` raises

File: `nocturne/engines/torch_tp/shipping.py:337-340`.

What is wrong: in the `finally`, `cur["f"].close()` runs before `cur["tmp"].unlink(...)`. `close()` flushes the buffer. If that flush fails, `close()` raises and the `unlink` does not run.
Checked with a small script: a buffered file whose last flush fails raises from `close()` inside a `finally`, and the file stays on disk.

Failure scenario: the disk fills while the short tail of a slice (less than one buffer) is still in the buffer. `flush()` raises "no space left", `close()` raises it again, and the `.part` file stays on the full disk. This is the case F6 named. A 4 MiB chunk is written directly, so the window is only the tail of a slice or a slice smaller than the buffer.

Suggested fix: `try: cur["f"].close()` then `finally: cur["tmp"].unlink(missing_ok=True)`.

### N6 (low) `stored_minus_loaded_bytes` is a plain difference, so the stated identity and its test cannot fail

File: `nocturne/engines/torch_tp/hybrid_run.py:390-395`, `docs/ENGINE_DESIGN.md:126-129`, `tests/bench/test_session10_coordinator.py:361`.

What is wrong: the field is computed as `manifest_bytes - loaded_tensor_bytes`. The design then says "Manifest bytes minus `stored_minus_loaded_bytes` equal the loaded bytes exactly", and the new test asserts the same. Both are true by construction.
The field therefore does not name the 1-D float32 part. It holds whatever the difference is, including a missing or extra tensor. In a float32 run of a bf16 safetensors model it is negative (loaded bytes are counted after conversion). The design sentence "the record names both" is not accurate: the planner rounding is not a field, it is `loaded_tensor_bytes - planned_weight_bytes`.
The independent check exists only in the unit test of the tiny models (`test_session10_shipping.py:381-394`), with the same 0.1 % tolerance as before.

Evidence that the story itself is right: K1, 7B Q8_0, loaded = planned exactly, difference 431104 bytes.

Suggested fix: compute the 1-D extra from the manifest on the coordinator (as the unit test does), record it as `one_d_extra_bytes`, and record `unexplained_bytes = manifest_bytes - one_d_extra_bytes - loaded_tensor_bytes`, which must be 0. Assert that in the end to end test.

### N7 (low) `ship_all` can lose the error text when the extra read fails in an unexpected way

File: `nocturne/engines/torch_tp/shipping.py:388-401`.

What is wrong: the extra read of the rank's ERROR frame catches `(OSError, ValueError)` only, and `errors.append(text)` comes after it. Any other exception from `recv()` ends the thread before the error is stored: a `KeyError` from `DTYPES[dcode]` when the stream is out of step, or a `MemoryError` from a garbage length. `_bring_up` then sees no shipping error and waits for READY on that link.
The read cannot block for long: each `recv` has a 3 s timeout and the rank sends at most a few frames. It swallows nothing the coordinator needs, because every error makes the constructor fail. The socket timeout stays at 3 s, which is harmless for the same reason.

Failure scenario: needs a stream that is out of step after the first error. Unlikely. The cost is a wrong or late error message, not a wrong result.

Suggested fix: `except Exception` around the extra read, and append the error in a `finally`.

### N8 (low) `preexec_fn` in a process that has threads

File: `nocturne/engines/torch_tp/recorder.py:91-98` and `:125-126`.

What is wrong: the Python documentation says `preexec_fn` is not safe when the process has threads, because the child can deadlock before `exec`. The rank and the coordinator have threads at that point (listener threads, torch pools). The hook imports two modules and calls `ctypes.CDLL("libc.so.6")`, which takes the loader lock in the child. If another thread held that lock at the fork, the child blocks and `Popen` in the parent never returns.
It is reached only on Linux, with `--timeline`, on the host level sampler of a host with a CUDA rank. K1 on gtx shows it works in the normal case.
Also: `PR_SET_PDEATHSIG` follows the thread that called `Popen`, not the process. Today that is the main thread in both processes. A sampler started from a short lived thread would lose its `nvidia-smi` when that thread ends.

Failure scenario: a rank on gtx stops in `start_sampler` before it links. The coordinator waits for its data link for 600 s. I rate the chance as small and could not test it here.

Suggested fix: load libc once in the parent (`_LIBC = ctypes.CDLL(None)` at import) so the hook is a single `prctl` call with no import and no `dlopen`.

### N9 (low) The trace and the host mean were not adapted to more than one GPU

File: `nocturne/engines/torch_tp/timeline.py:192-193` and `:162`.

What is wrong: the sampler now reports every GPU of the host. `write_host_samples` splits by `gi`. The Perfetto counter does not: all GPUs write into one counter `gpu util % (<host>)`, so the line jumps between GPUs. `gpu_util_mean_percent` is now the mean over all GPUs of the host, including ones no rank uses. Before the fix it was the one GPU of the sampling rank.
No effect on the hosts of today (gtx has one GPU).

Suggested fix: put the index in the counter name. Take the host mean over the GPUs that carry a rank, or drop the field and keep the per GPU dictionary.

### N10 (low) `timesync_set.json` is written even when the clock was not stepped, and the new chrony line is not reached on WSL2

File: `scripts/17_timesync.sh:24-29` and `:38`, `nocturne/bench/probes/timesync.py:143-151`.

What is wrong:
1. Line 38 writes the state file after every `--set`, also when `sudo sntp -sS` or `chronyc makestep` failed (each failure only prints a message). `last_set` in the record then names a sync that did not happen.
2. The `server` in that file is true only for macOS and for chrony. On WSL2 the clock comes from `hwclock -s`, with timesyncd from its own servers.
3. `chronyc add server "$SERVER"` sits in the branch after the WSL2 test. gtx is WSL2 and runs chrony (K1: `tool: chronyc`, `wsl: true`), so on the only Linux host of the project the line is never run. Where it does run, chrony still selects among all its sources.

Failure scenario: `--set` is run on the Air without sudo rights. The record of the next run says `last_set.at` a minute ago against `time.apple.com`, and the offset is still the old one.

Suggested fix: write the file only when the step command returned 0, and write what was done (`"method": "sntp" | "chrony" | "hwclock" | "timesyncd"`). Decide for WSL2 whether chrony inside WSL or the Windows clock is the source, and say it in `docs/MEASUREMENT.md`.

### N11 (low) Two tests depend on things outside the test

File: `tests/bench/test_session10_coordinator.py:333-372`, `tests/bench/test_session10_shipping.py:342-349`.

What is wrong:
1. `test_hybrid_run_main_end_to_end_on_a_tiny_gguf` runs the real preflight, which runs the time sync probe: DNS and four NTP queries to `time.apple.com` per host, twice per run, plus `sntp` on macOS. Offline it still passes when the name lookup fails fast. With a resolver that hangs, the probe passes its 60 s limit, preflight fails, and the test fails.
2. `test_a_corrupted_chunk_ends_the_run_with_the_rank_and_slice_named` asserts that no `.part` file is left right after the constructor raises. The rank threads are not joined on that path. Since the fix the rank sends the rejecting ack first and removes the file in the `finally` after it. The assertion can now run before the removal. The window is a few bytecodes and I did not see it fail. To confirm, run the test a few hundred times.

Suggested fix: patch `hr.TIMESYNC` or the probe's server in the end to end test (`NOCTURNE_NTP_SERVER` pointing at a closed local port fails fast). In `run()` of the shipping tests, join the rank threads when the constructor raises.

### N12 (low) Several fixes have no test

File: `tests/bench/`.

What is wrong: no test reaches these fixed paths. A later change can undo them without a red test.
- F8: a collect that fails for one host, and an end clock that fails after the results.
- F9: the ERROR frame read in `ship_all`, and the closed links after a failed constructor.
- F13: which sampler starts `nvidia-smi` (the `gpu` argument). Only the line parser is tested.
- F15: the per host model path in `launch` and in the `plan` command. B3 asks for a unit test for every fixed assumption.
- The SIGTERM handler: `test_stop_kills_what_is_left...` kills plain sleepers. No test shows that a rank blocked on a link exits on SIGTERM and stops its sampler.
- `last_set` in the probe, and the fallback of `Operator.hostname`.

Suggested fix: one small test each. Most need no process: a fake `sh` for collect and hostname, a socket pair for `ship_all`, a `Sampler` built with `gpu=True` and `host_level` on and off.

### N13 (low) Documents not updated with the fixes

File: `docs/MEASUREMENT.md:38` and `:96`, `nocturne/engines/torch_tp/recorder.py:9`, `docs/ENGINE_DESIGN.md:162-163`.

What is wrong: `docs/MEASUREMENT.md` is not in the fix diff.
- Line 96 and the recorder docstring still say `nvidia-smi` runs "on a CUDA rank". It now runs in the host level sampler of a host with any CUDA rank, for every GPU, with the index in each row (`gi`).
- Line 38 does not mention `chronyc add server`, the state file, or the new fields `last_sync_utc` and `last_set`.
- Design section 3.12 says the operator writes `record.json` and then stops what is left. The code stops first and writes the final record after (`hybrid_run.py:379-404`).

Suggested fix: three short edits.

## Not checked

- The working tree changed while I read: `nocturne/engines/torch_tp/timeline_page.html` and `tests/bench/test_session10_timeline.py` differ from `efad24d`. I reviewed the diff file and the commit, not those later edits.
- A run was in progress on this machine (`K2_m5coord_gtx_air_pp3_7b_q8_r1`, shipping). I did not touch it and kept my checks small.
- Linux: the `pkill` syntax is confirmed on gtx only through the K1 logs, and those tags hold no dash or dot. `re.escape` writes a dash as `\-`. glibc should read that as a plain dash, as macOS does (checked here). To confirm: `ssh gtx "pgrep -f -- '--tag a\-b( |\$)'; echo \$?"` should print 1.
- `preexec_fn` with `prctl`, `chronyc add server`, `hwclock -s` and `17_timesync.ps1` were read, not run. No run on MPS or CUDA by me.
- The test suite was not run. I ran one test (`test_a_transfer_that_ends_early_is_rejected_and_leaves_no_partial_file`, passed). The new end to end tests were read only.
- The behaviour of a socket after a reset (can the coordinator still read the rank's ERROR frame) is from knowledge of the two kernels, not tested.
- `timeline_page.html` and the README text of the diff were only skimmed for consistency with the code.
- Seen while reading, outside the fix diff, not followed up: `hybrid_run.main` calls `op.collect()` right after the coordinator process ends, before the rank processes are known to have finished writing their timeline files. GNU `tar` returns 1 when a file changes while it is read, and `collect` then drops every file of that host. Worth one look in a pass on the operator.

## Disposition (author, after pass 2)

Commit `82e6b25`. No high or medium finding was open, so the gate held before these changes.

| finding | disposition |
|---|---|
| N1 SIGTERM only | Left. A rank inside a long C call ends when the call returns; the operator's own process group still gets SIGKILL after 5 s for local processes. A remote rank that ignores SIGTERM for long would stay; not seen |
| N2 two shell calls could raise | Fixed. The hostname query falls back to the peers.yaml name, and the final `pkill` goes on to the next host. Test: `test_a_host_that_does_not_answer_does_not_end_the_operator` |
| N3 the host check cannot fail in an operator run | Left on purpose, and said so in `docs/MEASUREMENT.md` 2.2. The operator starts the process on that host, so the check is for a coordinator started by hand. The change was made because the Air now calls itself `Mac.lan` while peers.yaml holds an older name |
| N4 one failed end clock exchange drops all end values | Left. Without an end value a rank is aligned with its start offset only, and the record says so (`clock_end_error`) |
| N5 `close()` could skip the `unlink` | Fixed |
| N6 `stored_minus_loaded_bytes` is a plain difference | Accepted. The independent facts are `loaded_tensor_bytes` (from the rank) against `planned_weight_bytes` (from the planner): equal in every run of `VERIFY.md` |
| N7 lost error text | Fixed (any exception in the extra read is ignored) |
| N8 `preexec_fn` with threads | Left. Linux, CUDA host, `--timeline` only; the function makes one `prctl` call. It ran on gtx in three checks |
| N9 one GPU assumed in the trace | Fixed for the trace (one counter per GPU). The host mean stays a mean over GPUs, with the per GPU numbers next to it |
| N10 `timesync_set.json` written after a failed step | Fixed. On WSL2 the common server belongs to the Windows side; `docs/MEASUREMENT.md` 2.1 says so |
| N11 tests that touch the network | Left. The end to end test records a time sync error and still passes when offline |
| N12 fixes without a test | Partly. Tests added for N2 and for the collect path. F9, F13 on two GPUs, F15 and the SIGTERM handler have none |
| N13 documents out of step | Fixed: `docs/MEASUREMENT.md`, the recorder docstring, design section 3.12 |
| collect before the ranks have written (seen outside the diff) | Fixed. `Operator.wait_ranks` before `collect`, and a tar stream that ends with rc 1 is kept. Test: `test_collect_keeps_the_files_when_tar_reports_a_changed_file` |
