# Review (pass 1)

Reviewer: a fresh reviewer with the diff (`session10.diff`), the requirements (sections 2 to 4) and `ENGINE_DESIGN.md` only. Date: 2026-11-28.

Line numbers are of the current files in the repository. Nothing was edited. No server, GPU job or full test suite was started.

## Findings

Count: high 0, medium 2, low 15.

No finding gives wrong tokens, corrupts the cache, or hangs a run in the paths I could reach by reading and by small checks. The two medium findings are a cleanup step that never works and one requirement that is met only approximately for GGUF.

### F1 (medium) The final `pkill` by run tag is rejected by `pkill`, so remote processes are never stopped by it

File: `nocturne/engines/torch_tp/operate.py:219` (the same line exists in `launch.py:135`, older code).

What is wrong: the operator runs `pkill -f '--tag <tag>'`. The pattern starts with `--`, so `pkill` reads it as an option and exits with an error. Nothing is killed.
Checked on this Mac: `pkill -f '--tag zz'` prints `pkill: illegal option -- -` and returns 2. `pgrep -f -- '--tag zz'` works and returns 1.
I expect the same on Linux (procps reads `--tag zz` as an unknown long option). I could not confirm it here. To confirm: `ssh gtx "pkill -f '--tag x'; echo \$?"`.

Why it matters: `Runner.stop_all` only kills the local `ssh` client. A remote rank started without a terminal keeps running when its `ssh` client dies. The `pkill` is the only step that stops it. The design (section 3.12) says the operator "stops what is left".

Failure scenario: the coordinator fails before it opens the control links (for example the manifest step raises, or the coordinator host runs out of memory). Every remote rank is blocked in `listener.q.get()` in `rank_main` and never exits. The operator calls `stop()`. The `pkill` fails silently (the return code is not checked). The rank on `pc` stays, holds port 50070 and its GPU memory. The next run on that host ends with "rank ... exited rc=... at start". The same happens when a host leaves the network without closing its sockets (as in R6): ranks blocked on a data link with no timeout stay forever.

Suggested fix: `["pkill", "-f", "--", f"--tag {tag}( |$)"]` with the dots of the tag escaped. The anchor matters once the command works: without it tag `X_r1` also kills the processes of `X_r13`. Log a return code other than 0 and 1. Fix `launch.py:135` the same way.

### F2 (medium) "Shipped bytes equal the planned bytes" holds exactly for safetensors only; for GGUF the test accepts a tolerance

File: `tests/bench/test_session10_shipping.py:334-347`, `nocturne/engines/torch_tp/shipping.py:162` (`build_manifest`), `nocturne/engines/torch_tp/plan.py:129`.

What is wrong: requirement C3 says shipped bytes equal the planned bytes of the rank. For safetensors the tests assert `manifest_bytes == planned["weights"]` exactly. For GGUF the test first subtracts `one_d_extra` (1-D tensors are stored and shipped as F32, 4 bytes per value, while the planner counts them as bf16, 2 bytes per value), then accepts a gap of `max(64, 0.1 %)` of the planned bytes.
So for a GGUF model the record shows `shipping.ranks.<rank>.manifest_bytes` different from `planned.weights`, and no test pins the size of that gap for a real model.

Failure scenario: a reader compares `manifest_bytes` with `planned.weights` for the 7B Q8_0 run and finds they differ by the norm and bias vectors plus up to 0.1 %. The requirement text says "equal".

Rating note: by the letter this is a requirement gap for GGUF. I rate it medium, not high, because the difference comes from the planner's estimate, not from missing or extra weight bytes: the same test asserts that manifest bytes minus the 1-D extra equal the loaded bytes exactly (line 346). Raise it if the requirement is meant literally.

Suggested fix: either make the planner count 1-D tensors of a GGUF file at their stored size and assert equality, or record both numbers and the named difference (`one_d_extra_bytes`) per rank in `record.json` and state the rule in `ENGINE_DESIGN.md` 3.8. I did not trace where the remaining 0.1 % comes from (the planner uses fractional per row byte costs and rounds once, `plan.py:413`).

### F3 (low) The "no process left" check of the loopback test cannot fail

File: `tests/bench/test_session10_coordinator.py:292` (and `:283`).

What is wrong: `subprocess.run(["pgrep", "-f", "--tag B3_loop"])` fails with "illegal option" (same cause as F1), prints nothing on stdout, so `left == []` is always true. The test would pass with every rank still running.
Line 283 (`assert not exists or name in got`) is implied by the next assertion and adds nothing.

Suggested fix: `["pgrep", "-f", "--", "--tag B3_loop"]`, and assert the return code is 1.

### F4 (low) The slice cache and the hash memo are keyed by model hash and range only

File: `nocturne/engines/torch_tp/shipping.py:252` (`cached`), `:184` (`load_memo`), `:134` (`slice_key`).

What is wrong: a cached slice is accepted by name and size (the design lists this as a known limit). The coordinator's memo `slice_hashes.json` is accepted by `source|rows|cols` under the model hash. Neither key holds anything about how a slice is cut.
Failure scenario: a later commit changes what bytes a key means (for example a fix in `GgufSource._unpermute_index` or in the block cut). The model file is unchanged, so the model hash is unchanged. The memo still gives the old sha256. The rank still has a file of that name and size. Nothing is shipped, nothing is hashed, and the rank computes with the old bytes while the coordinator computes with the new ones. No error is raised. `hash_checks: 0` is the only trace.
A second, smaller case: the file is renamed into the cache without `fsync`. After a power loss a file with the right size and wrong content is possible on some file systems, and it would be trusted.

Suggested fix: put a format version constant in the cache path and in the memo (`<cache>/<model sha256>/v1/...`), and bump it when `read()` of a source changes. `fsync` the temporary file before `os.replace`. Optionally add a flag that re-hashes cached slices once per run.

### F5 (low) `receive_slices` accepts DONE before every needed slice arrived

File: `nocturne/engines/torch_tp/shipping.py:285`, stats at `:320`.

What is wrong: on `OP_DONE` the loop ends without checking that all slices of the need list were received and that no slice is open.
Checked with a small snippet over a loopback socket pair: manifest with one slice, one chunk, then DONE. `receive_slices` returned normally with `received_slices: 1, received_bytes: 0, hash_checks: 0` and left the `.part` file. The error appears only later, as `FileNotFoundError` from `CacheSource.read`.
`received_slices` is `len(need)`, not a count of what arrived.

Failure scenario: a coordinator bug (or another peer) ends the transfer early. The rank reports a file error at load time instead of "transfer incomplete", and its shipping numbers are wrong.

Suggested fix: on DONE, raise `ShipRejected` when `cur is not None` or when a needed index was not accepted. Count accepted slices.

### F6 (low) Temporary `.part` files are left in the cache on every abnormal end

File: `nocturne/engines/torch_tp/shipping.py:291` and the loop below it.

What is wrong: the temporary file is removed only on the "rejected" path. A lost link, a full disk (`OSError` from `write`), an unexpected frame, or a killed process leave `<sha256>.part<pid>_<thread>`. The name holds the pid, so a later run never reuses or removes it. The file handle is not closed either.
A partial file can never be loaded (it never gets the cache name), so this is disk use only.

Failure scenario: the link drops during an 8 GB slice on the WSL2 host. 8 GB stay in `~/.nocturne/cache/<model>/`. After a few such runs the free space check of `receive_slices` refuses the transfer.
Related: two ranks of one host each pass the free space check alone and can fill the disk together.

Suggested fix: `try/finally` around the receive loop that closes and unlinks the open temporary file. Remove `*.part*` files older than a day when the cache directory is opened.

### F7 (low) Two values from the wire reach a path with a weak check

File: `nocturne/engines/torch_tp/hybrid.py:473` with `:439`; `nocturne/engines/torch_tp/shipping.py:34`.

What is wrong:
1. `run_tag` of the PLAN message is joined to `~/.nocturne/runs` without a check when the rank was started without `--run-dir`. `os.path.join` with an absolute tag or with `..` leaves that folder. With `timeline: true` the rank then creates the directory and writes `timeline_<rank>.jsonl.gz` and `samples_rank_<rank>.jsonl.gz` there. The operator always passes `--run-dir`, so this needs a rank started by hand or by `launch`.
2. `HEX64` uses `$`, which also matches before a final newline. Checked: `HEX64.match("a" * 64 + "\n")` is a match. The name stays inside the cache folder (no slash), so nothing escapes. It is still a file name with a newline.

Security is deferred by ADR-009, but the task statement asks that a value from the wire does not name a file outside its folder.

Suggested fix: check `run_tag` with the operator's `TAG_OK` pattern in `rank_main`. Use `re.fullmatch` for both patterns.

### F8 (low) A completed run can be recorded as failed by a step that only collects or measures

File: `nocturne/engines/torch_tp/hybrid_run.py:362-367`, `nocturne/engines/torch_tp/coordinate.py:103` and `:113`.

What is wrong:
1. In the operator, `op.collect()` is inside the same `try` as the run, before `coordinator_record.json` is read. A timeout of the tar command (600 s) or a tar stream that does not parse (a remote shell profile that prints on stdout) raises, and the record becomes `outcome: error` with no prompts, although the coordinator finished and its record may already be in the run folder.
2. In the coordinator, `co.clock_sync()` runs before `rec["prompts"]`, `rec["ranks"]` and the logits are stored. If the end clock exchange fails (a link that drops after the last token), all results of the run are dropped and the outcome is `rpc_disconnect`.

Suggested fix: read `coordinator_record.json` before collecting from other hosts, and treat a collect failure per host as a warning. Store the results first, then call `clock_sync()` in its own `try`.

### F9 (low) When weight shipping fails, the coordinator waits for all other transfers and then records nothing about shipping

File: `nocturne/engines/torch_tp/hybrid.py:577-583`, `nocturne/engines/torch_tp/coordinate.py:131-137`.

What is wrong: after one rank rejects a slice, `_bring_up` still joins every ship thread (other ranks may need minutes) before it raises. The constructor then fails, `co` stays `None` in `run_job`, so `rec["shipping"]` and `rec["clock"]` are not written. The error text names the rank and the slice, but the per rank numbers and the model hash are lost.
Also: when the rank fails with its own error text while the coordinator thread is still sending (for example disk full), the thread usually sees a broken pipe first. The record then says `BrokenPipeError` and the real reason is only in the rank log.
The constructor's failure path closes neither the control links nor the listener. In a separate process this does not matter (exit closes them). In process users (tests, `launch`) leave rank threads blocked.

Suggested fix: keep the shipping dictionary on the exception (or write it to the record before raising). After a send error, try one `recv` with a short timeout to pick up the rank's ERROR frame. Close links and the listener in the `except BaseException` branch.

### F10 (low) The design says no other frame changed, but TOKEN and HIDDEN frames now carry the request number

File: `docs/ENGINE_DESIGN.md:231`, `nocturne/engines/torch_tp/hybrid.py:612`, `transport.py` docstring.

What is wrong: `aux1` of TOKEN and HIDDEN frames holds the request number in bits 8 to 31. It is sent in every run, also with `--timeline` off. `transport.py` documents it, the design document denies it. Requirement C2 names the weights frame as the only approved transport change; the CLOCK kind and this field are two more wire changes (CLOCK follows from A2).
The frame size is unchanged and no reader misreads the field (`aux1 & FLAG_LOGITS` everywhere; `Coordinator._step` now tests `want_logits`, not `flags`). Tokens are not affected.

Suggested fix: correct section 6 of the design. State the three wire changes in one place.

### F11 (low) A1: the time of the last sync is not recorded on macOS and Linux, and `--set` does not point chrony or timesyncd at the common server

File: `nocturne/bench/probes/timesync.py:126-141`, `scripts/17_timesync.sh:27-30`.

What is wrong: the requirement asks per run and host for the sync source, the offset reported by the OS tool, and the time of the sync. The probe records `at` (the time of the probe). Only the Windows part (`w32tm`) gives a last sync time. On macOS the "OS tool" entry is a fresh `sntp` query, not the state of the system time service.
`17_timesync.sh --set` on Linux steps the clock with whatever servers chrony or timesyncd already use. `--server` is used only for macOS and for the measurement. The requirement says "against the same server".
The measured offset of every host is against one server, which is what the comparison in `clock_report` needs. So the effect is on documentation of the sync, not on alignment.

Suggested fix: parse `Ref time (UTC)` from `chronyc tracking`, and store the time of `--set` in a small state file that the probe reports. Say in `docs/MEASUREMENT.md` that the OS services keep their own servers, or configure them.

### F12 (low) Output names differ from the requirement

File: `nocturne/engines/torch_tp/recorder.py:27`, `nocturne/engines/torch_tp/timeline.py:272`.

What is wrong: the requirement lists `timeline_<rank>.jsonl` and `trace.json`. The code writes `timeline_<rank>.jsonl.gz` and `trace.json.gz`, and an extra `samples_rank_<rank>.jsonl.gz`. Events carry step and request number; rank and host are in the header line of the file, not in every event. `docs/MEASUREMENT.md` describes the real names.

Suggested fix: none needed in code if the supervisor accepts the names. Note the deviation where the requirement is quoted.

### F13 (low) GPU samples only for the device of the lowest gid on a host

File: `nocturne/engines/torch_tp/recorder.py:99`, `nocturne/engines/torch_tp/hybrid.py:443-448`.

What is wrong: `nvidia-smi` is started only by the host level sampler, which is the lowest gid on the host, and only for that rank's device index.
Failure scenario: a host with ranks on `cuda:0` and `cuda:1` gets samples of `cuda:0` only. A host whose lowest gid is a CPU rank gets no GPU samples at all.

Suggested fix: start one `nvidia-smi` per distinct CUDA device of the host, from the first rank that uses it.

### F14 (low) Operator: an unquoted Python path in preflight, and setup errors that leave no record

File: `nocturne/engines/torch_tp/hybrid_run.py:175-187` and `:319-320`.

What is wrong:
1. Preflight builds `f"{py} -c ..."` with the Python path unquoted. `Runner` quotes the same path when it starts ranks. A path with a space works for the processes and fails in preflight with the misleading text "engine files on <host> differ". Since this change the local host goes through this command too.
2. `Operator(...)` can raise `ValueError` (bad run tag, a rank without device and backend, a 0.0.0.0 bind) or `FileNotFoundError`. Only `PlanError` is caught, so the run ends with a traceback and no `record.json`, while the run folder and `coordinator.log` exist.

Suggested fix: `shlex.quote` the path (keep a leading `~/`). Catch `Exception` around the constructor and save `outcome: preflight_failed`.

### F15 (low) `launch` and the `plan` command ignore the per host model path

File: `nocturne/engines/torch_tp/launch.py:67` and `:103`, `nocturne/engines/torch_tp/__main__.py:63`.

What is wrong: `model_dir_of` got a `host` argument, and `hybrid_run` uses it. `Launch` and the `plan` subcommand still call it without a host. `Launch` also passes its own model path to every remote rank.
Failure scenario: `peers.gtx.models.qwen2.5-7b-q8` points to `/mnt/d/...`. `python -m nocturne.engines.torch_tp plan` on gtx, or `launch` with a rank on gtx, looks under `~/models/...` and fails with "file not found". `launch` still must run on the coordinator host and does not ship weights; the README says so.

Suggested fix: pass the local peer name (found by hostname) to `model_dir_of` in both places, and the rank's host in `Launch.start`.

### F16 (low) Test gaps: the default production path is tested only in pieces

File: `tests/bench/test_session10_coordinator.py:265`, `tests/bench/test_session10_shipping.py`, `tests/bench/test_session10_timeline.py:206`.

What is wrong:
1. No test drives `hybrid_run.main`. The merge of the coordinator record, `agrees_with_preflight`, `clock_report` wiring and the timeline call are untested as a whole.
2. The process level loopback test runs with `ship_weights` off. Shipping is tested with rank threads in one process. So "operator, coordinator process, ranks started without `--model`, shipping on" (the default of `hybrid_run`) is never run end to end by a test.
3. Identical tokens with the timeline on is tested on CPU in fp32 only, where `sync()` does nothing. The device sync path on MPS and CUDA needs the recorded runs.

Suggested fix: one loopback test through `Operator` with `ship_weights=True` and a temporary cache directory, and one test of `hybrid_run.main` with a tiny model and a peers file of two local names.

### F17 (low) A non leader rank's last compute of a step is counted as waiting

File: `nocturne/engines/torch_tp/hybrid.py:281-288`.

What is wrong: `step` closes a compute span before a hidden state is sent and after the head. A rank with neither (a non leader rank of a group that is not the last stage) closes nothing after its last all-reduce. The residual add after it, without a device sync, falls into the `recv_wait` span of the next step.
The amount is small (one add per step). It breaks the rule in the recorder docstring that a compute span ends with a device sync.

Suggested fix: at the end of `step`, when `rec` is set and nothing was closed, `sync` and `rec.lap("compute")`.

## Checked and found correct

- Clock offset sign: `clock.estimate` gives remote minus local at the middle of the best round trip. `to_coordinator` subtracts it. The interpolation uses the remote time of both measurements. Read, and the three clock tests state the same signs.
- OS comparison sign: SNTP offset is server minus host, so the prediction for rank minus coordinator is coordinator offset minus rank offset. Read (`clock.compare_with_os`, `hybrid_run.clock_report`). The chrony "fast" sign is negated correctly.
- Control link order: PLAN, then data links on both sides, then manifest, need, chunks, end, ack, done, then load, READY, then CLOCK. The ship threads start after the coordinator's own `build_links` and are joined before the main thread reads READY, so no two threads read one control link. Read.
- End clock: the BEGIN frame travels the data path and every rank receives exactly one (stage 0 ranks on ctl, later stages from the previous leader, a leader forwards before it answers). No deadlock found. Read.
- Error frames during shipping: a rank that fails before NEED or instead of an ACK sends ERROR and `send_slices` reads it at both points. A CRC error keeps the rank reading until END, then it rejects, so the sender is never left blocked. Read.
- Header fields: `HEADER` is `<IBBHII`, so `aux0` and `aux1` are unsigned 32 bit. A CRC32 above 2^31 and `index << 8` fit. Read.
- Decode frame counters: WEIGHTS and CLOCK are not in `DATA_KINDS`, so `frames.measured_per_step` and `bytes_per_step` are not changed by shipping or by clock rounds. Read. (The ctl totals in `link_counters` do include weight bytes.)
- Loaded tensors, shipped against local: `CacheSource.read` returns the manifest's dtype, shape and qtype, and `load_from_source` then runs the same code as for a local GGUF file. The `plain` branch matches the safetensors branch of `load_slices` line by line (convert floats to the run dtype, count bytes after conversion, budget check, device copy, shared head). Read.
- Tied embeddings: `_wanted` on the rank sees only the names of its manifest. For tied and untied models, with equal and different row ranges, it builds the same entries as on the coordinator. Read.
- Empty slices and 1-D tensors: an empty safetensors slice is bf16 with the right shape on both paths; an empty GGUF slice keeps uint8. 1-D tensors keep `dim() == 1` through the cache, so they take the same branch. Read.
- Row order for Llama GGUF files: rows are un-permuted before hashing and shipping, the cache holds the cut slice. Read.
- JSON trip of the model config: all fields are plain values or the two tuples that `config_from_json` restores. Read.
- Concurrent shipping: `GgufSource.read` only slices a memory map; `SafetensorsSource.read` opens the file per call. The coordinator's own load uses the same object without shared mutable state. Read.
- Path safety of the cache: model and slice hashes are checked before they name a file; the need list is checked against the manifest size on the coordinator. Read (see F7 for the `$` detail).
- `safe_extract`: skips links, devices, names with a slash and hidden names. Read.
- No 0.0.0.0: `transport.listen` and `launch.host_addrs` refuse it; the operator takes listen addresses from `plan.listen_addrs`. Read.
- Quoting of rank and coordinator commands: `Runner._q` quotes every argument and keeps `~/` expandable; `shell_path` does the same for `push_job` and `collect`. The run tag is limited to `[A-Za-z0-9._-]`. Read.
- `scripts/17_timesync.sh`: `bash -n` passes with bash 3.2 on this Mac. Two line header present.
- `pkill` and `pgrep` behaviour with a pattern that starts with `--`: ran both on this Mac (F1, F3).
- Early DONE and the `HEX64` newline: ran one small CPU snippet with a cache folder in the scratchpad (F5, F7).

## Not checked

- No run on MPS or CUDA. Identical tokens with the timeline on and with shipped weights on real devices, and the recorder overhead per step (A3), need the recorded runs. `docs/MEASUREMENT.md` points to a `VERIFY.md` that did not exist when I looked.
- Linux and WSL2 behaviour: `pkill` of procps, GNU `tar` member names, `chronyc` and `timedatectl` output of the real hosts, `w32tm` through WSL, `17_timesync.ps1`.
- The test suite was not run. I ran no pytest test, only the two shell commands and the one snippet named above. The GGUF tests are skipped when the converter is missing; I did not check whether they run on this machine.
- `timeline_page.html` was read, not opened in a browser. Ranks beyond the eighth share one color with "unknown peer"; I did not check how that looks.
- `docs/MEASUREMENT.md` (A4) and `COORDINATOR_AUDIT.md` (B1) are not in the diff. I searched `MEASUREMENT.md` only for the recorder limits (the idle time before a request and the helper thread sends are documented there). I did not check the audit's file and line list.
- The real `~/.nocturne/peers.yaml` was not read, so the hostname check of `coordinate.check_host` against the WSL2 host name is not confirmed.
- Thread safety of `gguf-py` readers is inferred from the code (memory map slices), not stress tested.
- Limits of the 32 bit fields (slice index below 2^24, request number below 2^24) are far from real use and were not tested.
