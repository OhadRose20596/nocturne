# Session 10, part 1 of 2: clock sync, recorder, any requester, shipped weights

Work prompt for Claude Code. Written 2026-11-28 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo.
Part 1 builds and verifies. It may use gtx and Lior's laptop (`pc`) for short checks when Ohad has them on, but no experiment rows. Part 2 (`02_EXPERIMENTS.md`) runs the experiments.

## 0. Why

On 2026-11-23 Chen asked for: clocks synchronized before every run, charts that show per device what is compute and what is communication, total bytes per configuration, a CUDA baseline, and a step by step explanation of how our engine works (who plans, who sends what, how a worker gets its part). Ohad adds two engine goals:
1. Any machine can be the requester and the planner, not only the M5.
2. A worker does not hold the model. After the planner decides, the requester sends each worker exactly the part it must compute (PP, TP or hybrid), then the run starts. This is the open requirement from October (`docs/` and memory: "worker must not hold the model"). llama.cpp RPC already does this, our engine does not.

## 1. Read first

1. `CLAUDE.md`.
2. `nocturne/engines/torch_tp/`: `hybrid.py`, `hybrid_run.py`, `plan.py`, `transport.py`, `loader.py`, `gguf_source.py` (the `TensorSource` interface was built for this), `engine.py` (`--trace-allreduce`), `README.md`.
3. `experiments/engine_gguf/` (`01_BUILD.md`, `VERIFY.md`, `REVIEW.md`), `experiments/session9_timeline/existing_data/` (how the 23.11 charts were made from packet captures, and their limits), `experiments/session9_timeline/DRAFT_NEXT_WORK_2026-11-23.md`.
4. `docs/JOIN.md` and the join scripts. `~/.nocturne/peers.yaml`. The `pc` peer still holds a stale hotspot address.
5. `nocturne/bench/probes/` (capture, util samplers, `clock_offset`).

Rules: short sentences, no em dash, no en dash in Markdown. No Hebrew in code or command blocks. T1: time is background, never a gate. Never bind `0.0.0.0`. Do not install or upgrade torch. New dependencies only `psutil` if not present. Ask Ohad before any sudo step, Windows step or download over 10 GB.

## 2. Part A: clock synchronization and what we record

A1. OS level sync, before every run, on every host:
- macOS: `sntp -sS <server>` (sudo), Linux and WSL2: `chrony` or `systemd-timesyncd` against the same server. WSL2 takes its clock from Windows: check `w32tm /query /status` and the drift between Windows and WSL (`date` in both). Pick one server for all hosts (a public pool, or one host on the LAN as server). Write `scripts/17_timesync.sh` (and a PowerShell part for Windows if needed) that syncs and prints the measured offset of the host.
- Record per run, per host: the sync source, the offset reported by the OS tool, the time of the sync.
A2. Application level alignment, in the engine, at the start and the end of every run: 16 round trips on the control link from the coordinator to each rank, keep the smallest round trip, offset = remote minus local minus half the round trip. Store offset, round trip, and drift between start and end in `record.json`. Every event time is converted to the coordinator clock with it. If the OS sync and the engine measure disagree by more than 2 ms, say so in the record.
A3. Recorder (behind `--timeline`, off by default, identical tokens with it on):
- In every rank: `time.monotonic_ns()` start and end of `compute` (with a device sync at the end so the span is real device time), `send` (with bytes and destination), `recv_wait` (blocked on a link, with source), `allreduce_wait`, `host_copy`. Each event carries step, request id, rank, host.
- A sampler per host every 20 ms: CPU total and of the rank process, network bytes in and out of the interface used. GPU: `nvidia-smi` every 100 ms on CUDA hosts, `powermetrics` on Macs only if Ohad approves sudo.
- Per run outputs: `timeline_<rank>.jsonl`, `samples_<host>.jsonl`, `trace.json` (Perfetto), `breakdown.csv` (per step per rank: compute, waits, send, copy, other, bytes sent and received per peer), and `timeline.html`.
- `timeline.html` follows Chen's asks: per device one compute row and one communication row, colors by peer (who talks to whom), plain axis titles and a legend, a table of total bytes per link and per configuration. No hand drawn arrows.
- Overhead check: the same run with and without `--timeline`, tokens identical, overhead per step recorded.
A4. `docs/MEASUREMENT.md` (English, for Chen): for each number we report, which file it comes from, which code writes it, how the clocks are aligned, the precision, and the limits. Include the 23.11 method (packet captures) and say what the recorder replaces.

## 3. Part B: any machine can be requester and planner

B1. Audit every place that assumes the M5 or macOS as the coordinator: `hybrid_run.py` (caffeinate, preflight, paths), `plan.py`, the bench, the join tool (`16_join_apply.py`), `peers.yaml` layout, model registry paths. List them in `experiments/session10_cuda_pair/COORDINATOR_AUDIT.md` with file and line.
B2. Fix them so the coordinator is the host named in the placement (`coordinator: {host: ...}`), on macOS or Linux (WSL2). The planner runs where the coordinator runs. The operator (Claude Code on the M5) may launch it over ssh.
B3. Tests: loopback on the M5 with the coordinator role on a second process name, and unit tests for every fixed assumption. A short check with gtx as coordinator if gtx is on.

## 4. Part C: the requester ships each worker its part

C1. Design first, in `docs/ENGINE_DESIGN.md` (English, the base for Chen's slides). It covers the whole flow step by step: join, peers, planner input and output, placement, launch, links (which rank opens which connection), weight shipping, the decode loop, metrics, shutdown. Include one diagram (Mermaid) of the control plane and one of the data plane. Write it before the code for C2 and keep it in sync.
C2. Build:
- After `plan`, the coordinator builds a manifest per rank: tensor names, slices, dtype or quant type, byte size and sha256 of each slice, and a total.
- A new frame kind on the existing links for weights (chunked, with a per chunk checksum). Document it in `transport.py`. This is the only transport change approved.
- The rank checks a local cache (`~/.nocturne/cache/<model sha256>/<slice sha256>`) and asks only for missing slices. It writes them to disk, verifies the hash, then loads them to the device through the `TensorSource` interface. Streaming: never hold a slice twice in memory.
- Only the coordinator needs the model file. `hybrid_run` gets `--ship-weights` (default on for new runs, off keeps the old behaviour for comparison).
- Record per rank: bytes shipped, bytes from cache, time to ship (background), hash checks.
C3. Tests: a rank on a host with no model file gives the same tokens as with a local file. A second run ships 0 bytes. A corrupted chunk is rejected. Shipped bytes equal the planned bytes of the rank. Works for safetensors and GGUF (quant blocks keep the block alignment rule).

## 5. Verify and review

V1. All existing tests pass on the M5. Run the dequant and model tests on CUDA if gtx or pc is on (`NOCTURNE_TEST_DEVICE=cuda`).
V2. Mutation check as in engine GGUF part 1: break on purpose the offset sign, the hash check, the manifest slice bounds, the coordinator host lookup. Each must make a test fail. Record in `experiments/session10_cuda_pair/VERIFY.md`.
V3. Loopback end to end on the M5: 1.5B and 7B Q8_0 GGUF, PP 3 stages, TP 3 ranks, hybrid, with `--timeline` and `--ship-weights`, against the recorded part 1 runs of engine GGUF. 8 of 8 identical required.
V4. Independent review: a fresh subagent with only the diff, sections 2 to 4 of this prompt and `docs/ENGINE_DESIGN.md`. It writes `REVIEW.md` with findings rated high, medium, low. Write `REVIEW_REQUEST_CODEX.md` for Ohad to give Codex. Fix every high finding, then a second fresh pass on the fixes.
Gate: tests pass, mutations caught, V3 identical, no open high finding.

## 6. Records and push

`VERIFY.md`, `REVIEW.md`, `docs/MEASUREMENT.md`, `docs/ENGINE_DESIGN.md`, `docs/JOIN.md` (if changed), `README.md` of the engine, two lines in `docs/CURRENT_CONTEXT.md`, one entry in `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` (private, never commit). Small commits, push.
Final message under 25 lines: gate result, what the recorder writes, the measured overhead, coordinator audit count, shipping results, open findings, what Ohad must do before part 2 (Codex review, turn on gtx and pc, run the join tool on pc).

## 7. Do not

Run experiment rows. Change the transport beyond the weights frame. Change the numerics. Edit old run folders. Commit model files or caches. Bind `0.0.0.0`. Leave samplers or test processes running.
