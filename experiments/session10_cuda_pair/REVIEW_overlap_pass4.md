# Session 10, part 3. Review of the fixes after the Codex review (pass 4)

Reviewed `git diff 3616976..6605c15` for `nocturne/engines/torch_tp` and `tests/bench/test_session10_overlap.py`. Main moved to `4711f32` during the review. That commit changes no line under `nocturne` or `tests`. Line numbers below are those of `6605c15`. Everything ran on CPU loopback on one Mac, with ranks as threads of one process.

## Summary

1. I could not break the fix of the high finding. The old engine hangs in the TP form (sync and async) and in the pipeline form (sync). The new engine completes all of them. No high finding.
2. Seven of the ten Codex findings are closed. Three are partly closed: 3 (a jump of the request counter), 5 (two ways to fall back to seven of eight), 6 (frames of a one token prompt).
3. New findings: 0 high, 1 medium, 7 low. Most of the low ones are gaps in the new tests, found by 33 mutations of the fixes.

## How far the evidence goes

Read this before the verdicts. It limits several of them.

- **The plain small buffer stress proved nothing.** 112 runs with `SOCK_BUF=4096` all passed. The old engine passed the same runs. The macOS loopback accepts about 528 KB of one write whatever `SO_SNDBUF` says (measured: 528,384 bytes). I do not count these runs.
- **The tight link stress is my own model.** A user space shim lets a link direction hold at most 4096 or 1024 unread bytes. Bytes and order are unchanged. 336 runs on the new engine: 0 hangs, 0 token differences against `generate` on the same placement. Seven shapes, 2 to 6 slots, interleave 1 and 2, chunk 0, 16 and 64, sync and async, head frames held until the coordinator starts another unit, and random delays.
- **That stress has proven power for one shape only.** Under the shim the old engine hangs in `tp2` (sync and async). It did not hang in the five other shapes I tried on it (`pp2`, `pp3`, both hybrids, two TP stages). So for those shapes the clean result on the new engine does not show that the bad interleaving was reached.
- **Hybrid placements have no failing baseline.** For them the closure of finding 1 rests on my reading of who can wait for whom, plus the stress above. For pipeline with async send the old engine also completed, so there is no failing baseline there either.
- **One slip of mine, corrected.** My first run of the existing reproductions passed `tp async` as one argument (zsh does not split). Those rows were a different run, and for a moment looked like wrong tokens. The table below is from the corrected run.
- **Mutations ran against this one test file.** "Survived" means no test in `test_session10_overlap.py` failed. I did not run the rest of the test suite against the mutants.

## The ten Codex findings

**1. A large HEAD reply can block the next slot. Closed, with the limit above.** `codex_f1_repro.py tp`, `tp async` and `codex_f1_pp.py` hang on `3616976` and complete on `6605c15`. The reader thread reads every remote head rank's control link for the whole call, so a head rank never waits for the coordinator's main thread.

| reproduction | `3616976` | `6605c15` |
|---|---|---|
| `codex_f1_repro.py tp` | hang | completes |
| `codex_f1_repro.py tp async` | hang | completes |
| `codex_f1_pp.py` | hang | completes |
| `codex_f1_pp.py async` | completes | completes |
| `codex_f1_repro.py pp`, `pp async` | completes | completes |

**2. Default control frames carry extra bytes. Closed for the PLAN frame.** I captured PLAN on `965385f`, `3616976` and `6605c15` for PP3, TP3 and a hybrid, with the recorder on and off. With ports made equal, the PLAN text of `6605c15` equals that of `965385f` in all six cases (2057, 2071 and 3004 bytes; `3616976` has 33 more). The data frames of four prompts are equal too, compared as a set per run, not in order across threads. READY and the METRICS reply still differ, as the README now says. I did not measure them.

**3. Request IDs alias after 65,535. Partly closed.** A run that counts up through the wrap keeps its numbers on every rank. The repo test shows it with 3 bits on the old path. My run with 3 bits, 3 slots, 20 requests and pairs shows it for TP3, a hybrid and PP3. The reviewer's own check still fails: set `co.req_seq = 65535` before the first request, and the coordinator records 65536 while both remote ranks record 0. `full_req` cannot rebuild a jump of more than half the modulus. No real run jumps. The frame field from request 65536 on differs from the base by design, and the README says so.

**4. A failed large frame helper can release its successor. Closed.** I made one threaded all-reduce send raise through real links, in six cases (old path, slots, pairs, coordinator or peer, TP3 and a hybrid). The new engine ends each run at once, and names the failure when the frame is the coordinator's own. The old engine hangs in the two old path cases. In one pair case it read a later frame in place of the lost one and only failed because the shapes differed.

**5. A scheduling regression can pass the correctness check. Partly closed.** Against a plain run folder with the same placement path, hash, dtype and chunk, one changed token gives `criterion_met=False` under `rule: every token`. Four recorded comparisons of this session used that rule. Two kinds of same placement reference still get seven of eight, and a missing long prompt passes. See N1 and N2.

**6. A one token prompt corrupts the counts. Partly closed.** The step counts are right now, and equal on both paths: 1 prefill step and 2 decode steps for `[5]` with three tokens. The frames of that prompt are still charged to decode on both paths. Measured on TP3: 87 decode frames per step against 58 predicted, and 0 prefill frames against 58. The frame check fails loud, it does not pass falsely, and the README names the limit. The reviewer asked for the phase to reach the frame counters. That part was not done.

**7. A slotted KV tensor can escape the MPS heap floor. Closed for the K tensor.** The repo test is the reviewer's scenario (256 slots, 16 MiB K tensor, a 0.65 GB host budget rejected). It fails on `3616976`. Two mutations of the new line are caught. I did not look at work space tensors or at a dtype other than bf16.

**8. An unmatched trace still publishes an overlap share. Closed in the record.** The reviewer's case gives `unmatched_links=["a -> b"]`, busy 30 percent and no overlap share. `overlap_table.py` prints `n/a`. I read the page code and did not render it. One number is left: see N7.

**9. The sender can keep more than its pool limit. Closed.** The test fails on `3616976` and two mutations of the condition are caught. This is the accounting on CPU. No pinned memory exists here.

**10. CLOCK and SHUTDOWN bypass the sender flush. Closed for the order.** The test fails on `3616976`. Removing either flush is caught. The two timeouts are not tested: see N7.

## New findings

### N1. Medium. Two same placement references still get the rule of seven of eight

`hybrid_run.py:248` and `hybrid_run.py:267`.

Scenario, first form. An options off run of a placement is saved with `--save-reference`. Its folder holds `reference.json`. `load_reference` prefers that file, and it carries no `numerics`. A later run of the same placement with slots names it as `--reference`. One of eight evaluation prompts differs by one token. Result: `rule: seven of eight`, `criterion_met=True`.

Scenario, second form. The same placement content (equal sha256) is recorded under another path spelling, for example a file outside the repository given once relative and once absolute. `numerics_of` compares the path as well as the hash, so the two runs do not count as the same placement. Same result.

Checked: reproduced both with synthetic run folders through `load_reference`, `scheduling_only` and `compare`.

Why medium and not high. The record and the log name the rule that was used. No run folder of this session holds a `reference.json`, and the dashboard takes placements from inside the repository. By the letter of "a failed check looks passed" this could be rated high. The caller may want to re-rate it.

### N2. Low. Under "every token" a long prompt that the reference lacks is not compared

`hybrid_run.py:539`. Steps: the reference was recorded with `--long 0`. The new run has the same placement and a long prompt. Result: `rule: every token`, `long_compared=0`, `criterion_met=True`. The record does list the prompt under `not_in_reference`. Checked: reproduced.

### N3. Low. The lost control link test does not show that the reader hands its failure over

`test_session10_overlap.py:1219`. The test passes on `3616976` as well. With the reader's failure swallowed (`hybrid.py:953`), the test does not fail. It hangs: my runner stopped it at 420 s, and the engine's own wait is 1800 s. Checked: both reproduced. Unverified reasoning: the test can also pass through the pipeline links, when the dead head rank takes the middle rank with it and the coordinator's next HIDDEN send fails.

### N4. Low. No test covers an ERROR frame or a frame of another kind that arrives through the reader

`hybrid.py:1301` to `1304`. I removed the two kind checks in `drain`. All 98 tests of the file passed. In that mutant an ERROR frame from a head rank is taken as a head result with value 0.0 and token 0. The checks are right at `6605c15`. Nothing guards them. Checked: reproduced by mutation.

### N5. Low. The frame thread test has a real time window of 0.2 s

`test_session10_overlap.py:1305` to `1309`. The first frame fails after `time.sleep(0.2)`. The second `send_partial` must start inside that time. With a stall of 0.25 s between the two calls, the second call raises and the test fails on a correct engine. With 0 and 0.1 s it goes on as written. This is a false failure under load, never a false pass. Checked: reproduced with an injected stall.

### N6. Low. After a failed call the reader can stay on the control socket

`hybrid.py:956` to `963`. Steps: a head rank stops in the middle of a 4 MB HEAD frame. The scheduler fails for another reason. `stop()` gives up its join after 5 s and closes the wake sockets. The thread stays inside `Link.recv`. `shutdown()` then runs the end clock exchange on the same socket. Measured: `generate_many` returned its error after 6.0 s with the reader alive, `shutdown()` took 20.5 s, the end clock was empty, and the thread ended when the links closed. No hang. The cost is time, and two threads reading one socket for 20 s. Checked: reproduced.

### N7. Low. Five more fixes have no test, and one number is still published from shifted pairs

Each of these mutations left every test of the file passing:

- `hybrid.py:742`. The pair partner's request number without `full_req`. My own run shows the code is right.
- `hybrid.py:418` to `420`. The failure of a frame thread not named when the receive raises. The fake link's `recv` never raises. My real link run shows the code is right.
- `hybrid_run.py:494`. `main` passing `exact=False`. Only the four recorded comparisons show the wiring.
- `hybrid.py:1399` and `1394`. The shutdown flush and the end clock flush without their timeout.

A sixth mutant, a `stop()` that does not wake the reader, also survived. The call took 11 s in place of about 1 s. I take that as a slow equivalent, from the time alone.

And `timeline.py:236`: for an unmatched link, `links[...]["in_flight_ms_mean"]` is still the mean over the shifted pairs. The counts beside it show the mismatch. Checked: read only.

### N8. Low. The head frame test relies on how much the loopback accepts

`test_session10_overlap.py:1163`. The peer's head frame is 1 MiB. This Mac's loopback took 528,384 bytes of one write with `SOCK_BUF=4096`. The margin is a factor of two. On a system that accepts 1 MiB the test would pass on the old engine. Checked: the number is measured here. The claim about other systems is reasoning.

## Observations outside the diff

- A coordinator used again after a failed `generate_many` finds old HEAD frames. In three shapes the second call raised "head frame ... which waits for none", and `metrics()` raised too. A silent wrong token is possible in principle. I did not produce one. The old engine has the same state, and the job code never reuses a coordinator.
- `numerics_of` does not hold the reduce dtype. Two runs of one placement with different `--reduce-dtype` are asked for every token. That errs toward a false failure.

## Mutation tally

33 mutations of the fixes. 25 caught by a failing test. 1 detected only by a hang (N3). 7 survived (N4, N7).

## Not checked

- MPS, CUDA, pinned copies, two hosts, any real model.
- Linux socket buffers. The tight link is a shim of mine, not a kernel.
- `docs/ENGINE_DESIGN.md` section 11 and `VERIFY_overlap.md`. I did not open them. I compared the code with the Codex findings and with its own comments.
- The author's Disposition section, beyond its first sentence.
- The timeline page in a browser.
- The bytes of READY and of the METRICS reply.
- The test suite outside `test_session10_overlap.py`, against the mutants.
- `hybrid_run.py` before line 225, `operate.py`, and the dashboard beyond how it names a placement.
- Whether `shutdown()` can block on a full pipeline link of a stuck rank. Reasoning only, and older than this diff.

## What I ran

Scripts are in the scratchpad folder `review_overlap4/x`: `stress4.py` with `tight.py` (the tight link stress), `plancap.py` (finding 2), `f36.py` (findings 3 and 6), `f4.py` (finding 4), `f5.py` (finding 5), `e3.py` (failures and the second call), `t6.py` (N5), `mut.py` with `mut_results.json` (the mutations). The scratchpad can be cleared by a reboot.

## Disposition (by the author of the code, 2026-11-28)

The reviewer's text above is unchanged. Fixes and tests are in the commit that adds this section.

| id | decision | what was done |
|---|---|---|
| N1 | fixed, rated as the reviewer did (medium) | The numerics of a run are the content hash of the placement, the dtype, the reduce dtype and the prefill chunk. The path is out. A reference saved with `--save-reference` now carries them, and one saved before takes them from the `record.json` beside it. Cases added to `test_a_run_that_differs_in_scheduling_only_must_give_every_token` |
| N2 | fixed | Under "every token" a prompt the reference lacks fails the criterion, a long prompt too |
| N3 | fixed | The test cuts the wait for a head frame to 5 s and refuses the time limit as the reason, so a failure that is not handed over fails the test at once |
| N4 | fixed | `test_a_frame_that_is_no_head_result_is_never_taken_for_one`: an ERROR frame and a frame of another kind through the reader |
| N5 | fixed | The frame thread test has no time window: the first frame fails when the test says so, after the second is queued |
| N6 | fixed in part | When the reader did not stop, shutdown skips the end clock exchange and says so in the log, so the two no longer read one socket. The 5 s wait of `stop()` stays |
| N7 | fixed in part | Tests for the pair partner's request number and for the naming of a frame thread failure. An unmatched link has no mean time in flight. Not pinned by a test: `main` passing `exact`, and the two flush timeouts of the shutdown path |
| N8 | fixed | The test measures what the loopback takes unread with its buffer setting and skips when that reaches the size of the head frame |
| Codex 3, partly | documented | A counter that is set to 65535 before the first request still gives 0 on the other ranks. The rule and its condition are in section 11.1 of the design: a rank sees every request, so the number never jumps |
| Codex 6, partly | as before | Known limit, in the README |
| observation: a coordinator used again after a failed `generate_many` | not changed | The job code never reuses a coordinator after a failure, and the old path has the same state |
| observation: reduce dtype | fixed | Part of the numerics now |

Evidence limits the reviewer names, kept as open items in `VERIFY_overlap.md`: no failing baseline for the hybrid shapes, and nothing on MPS, CUDA, Linux buffers or two hosts in this pass. The seven M5 runs after the Codex fixes are MPS runs with slots; Linux and CUDA are the first step of the next session.
