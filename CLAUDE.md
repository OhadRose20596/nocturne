# CLAUDE.md — Nocturne

B.Sc. final project (BGU, ECE): **Distributed LLM inference over volunteer consumer GPUs**, BitTorrent-style. Read `docs/CURRENT_CONTEXT.md` first (latest handoff), then `docs/GOAL_AND_OBJECTIVES.md` (v4, five objectives), then `docs/ARCHITECTURE.md`. Decisions that must not be silently reversed are in `docs/DECISIONS.md`. Team decisions T1 to T3 are in `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` section 9.2.

**Writing rule (supervisor, 2026-10-11):** Markdown in short sentences. No long dash (em or en dash) in new or rewritten text. Objectives are referred to by the v4 numbering (O1 to O5); older docs still use O1 to O10, see `docs/archive/`.

## The model in one paragraph

The **requester** (machine with the prompt) is the coordinator of its own job. A tiny **tracker** only answers "who is available for model X". **Workers** run `ggml-rpc-server` exposing one GPU each. The requester computes a layer placement and launches llama.cpp with exactly two derived flags: `--rpc <endpoints>` and `--tensor-split <proportions>`. The placement algorithm is the research contribution; everything else is plumbing.

## Current stage

O1 done on two Macs. O2 done on two Macs on the home LAN (32B runs on both, fails on each alone with full offload). O3 to O5 not started. Hardware: MacBook Air M3 16 GB and Mac M5 24 GB (Metal), both in one home; Lior's NVIDIA machine (CUDA, WSL2) not yet run. llama.cpp pinned at SHA `1e7bcf3`, built outside iCloud (`~/dev/llama.cpp`). Open: why `--split-mode tensor` failed on 32B (repeat with `SPLIT_MODE=tensor scripts/10_capacity.sh ...`).

## Repository layout

```
scripts/      bash, numbered; all source scripts/common.sh. Each file starts with a two-line usage header; keep comments minimal
  00-06                  build, list devices, local smoke test, worker, two-machine smoke test, llama-bench, net baseline
  07_preflight.sh        inventory + binary check -> experiments/inventory/
  08_r0.sh               recorded runs (worker | local | rpc | placement) -> experiments/r0/runs/<run_id>/
  09_r0_report.py        Tables A/B/C + acceptance -> experiments/r0/REPORT.md
  10_capacity.sh         capacity under memory policy P1/P2, SPLIT_MODE=layer|tensor|row -> experiments/r3/capacity/
  11_load.py, 13_serve.sh  throughput under load (split vs replicas) -> experiments/r3/load/
  12_rpc_trace.py        per-step RPC command/bytes trace -> experiments/r3/trace/
  14_build_he_html.sh    docs/he/*.md -> docs/he/html/ (RTL)
nocturne/     Python package (O5): tracker/, worker_agent/, requester/, placement/ (stubs)
experiments/  RESULTS.md (append-only), R0_PROTOCOL.md, r0/, r3/, inventory/. Raw run folders are evidence; never edit them
docs/         CURRENT_CONTEXT, GOAL_AND_OBJECTIVES (v4), FLOW, PLATFORM, BACKGROUND_AND_DESIGN, ARCHITECTURE,
              DECISIONS, SETUP_GUIDE, LEARNING_RESOURCES, BACKENDS (background), archive/ (superseded versions)
docs/he/      Hebrew: MEETING_CHEN_2026-10-11, NEXT_STEPS_2026-10-17, CONNECTIVITY, RUNBOOK_R0, GUIDE_R0_TO_R3,
              STATUS/ONE_PAGER (11.10 snapshots), PARTITIONING_DEEP_DIVE, SHARED_RESEARCH_AND_DECISIONS (private, gitignored)
third_party/  llama.cpp checkout (gitignored)
.env          local config (gitignored); template in env.example; per-machine overrides in ~/.nocturne.env
```

## Rules for working in this repo

1. **Never reverse an ADR silently.** If a change contradicts `docs/DECISIONS.md`, propose a new ADR in the same file and stop for review.
2. **New tool ⇒ new entry in `docs/LEARNING_RESOURCES.md` first.** Supervisor's rule. One link with a tag and a one-line "why".
3. **Every measurement goes to `experiments/RESULTS.md`** with host, date, llama.cpp commit, model, device/split. Append-only; never edit old rows.
4. **Verify llama.cpp flags against the checked-out version** (`$BIN/llama-cli --help`, `$BIN/ggml-rpc-server --help`) before assuming. Scripts were written against llama.cpp 0.4.0 (Sept 2026): `-st` single-turn, `--device`, `--rpc`, `-ts`, `ggml-rpc-server`. When a flag drifts, fix the script **and** docs/SETUP_GUIDE.md §7 in the same commit — people run the guide without an assistant.
5. **Scripts must run on macOS bash 3.2 and Linux bash 4+.** No associative arrays, no `mapfile`, no `${var,,}`. Test with `bash -n` at minimum.
6. **Placement is a pure function**: `place(peers: list[Peer], manifest: Manifest, policy) -> Placement` returning the `--rpc` string and `--tensor-split` list. No I/O inside. Unit-test it with synthetic peers.
7. **Exposure**: workers bind to the address chosen on purpose (LAN, Tailscale, or the relay-facing address per docs/he/CONNECTIVITY.md). Never `0.0.0.0` in committed defaults. Security is deferred by ADR-009, not forgotten: say so wherever a port is opened.
8. **Language**: code, comments, docs in English. Hebrew only in `docs/he/`. Avoid Hebrew in shell files and command blocks; RTL characters have corrupted pasted commands before. `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` is the private working log with Codex: gitignored, never link to it from public docs as if readers can open it.
9. **Python**: 3.11+, `uv` or `pip`, FastAPI + SQLite for the tracker, `httpx` for agents, `pytest`. Keep dependencies minimal; the tracker must run on a Raspberry Pi.
10. **Don't optimize what isn't measured.** Baselines from O1 and O2 come before any placement cleverness.
11. **Comments**: a script carries a two-line header (purpose, usage) and nothing else unless a line is a trap. Explanations go in docs, not in code.

## Common commands

```bash
cp env.example .env && $EDITOR .env          # set MODEL, LLAMA_DIR
bash scripts/00_build_llamacpp.sh             # BACKEND=cuda|metal|cpu to force
bash scripts/01_list_devices.sh
bash scripts/02_run_local.sh MTL0             # use the listed device name; or CUDA0 / CUDA1 / none
bash scripts/03_start_worker.sh CUDA0         # on the worker machine
bash scripts/04_run_two_machines.sh <worker-ip> 0,1   # on the requester machine
bash scripts/05_bench.sh
bash -n scripts/*.sh                          # syntax check
```

## What to do next (in order)

The live list is docs/he/NEXT_STEPS_2026-10-17.md. In short:

1. First git commit from a Mac (iCloud holds some run files only in the cloud).
2. TP on 32B with logs, both directions: `SPLIT_MODE=tensor scripts/10_capacity.sh split ...` (supervisor question Q7).
3. Recollect Table B in the Air to M5 direction (`08_r0.sh rpc`, then `09_r0_report.py`).
4. NVIDIA machine: `07_preflight.sh`, build with `BACKEND=cuda`, Table A. WSL2 needs mirrored networking.
5. 32B between two homes: Tailscale first, then without it (CONNECTIVITY.md section 3). ADR-007 by 15 Nov 2026.
6. O5 code, in this order: `nocturne/manifest.py` (GGUF header -> manifest), `placement/` v1 (memory-feasible, output layer local),
   `requester/cli.py` with a static `peers.toml`, heartbeat + replay, then `tracker/` and `worker_agent/` with the connectivity ladder.
