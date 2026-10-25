# Nocturne

**Distributed LLM Inference over Volunteer Consumer GPUs** — BitTorrent-style.
B.Sc. final project, Electrical and Computer Engineering, Ben-Gurion University of the Negev, 2026–27.

The machine that wants the answer coordinates the work. A tiny tracker only says who is online. Workers expose one GPU each over llama.cpp RPC. The requester decides which transformer layers run where — that decision, under heterogeneity and node churn, is the research question.

| Read this | For |
|---|---|
| **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** | **Start here on a new machine** — install, build, model, PoC-0/1, troubleshooting. No AI tools needed |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | BitTorrent → inference mapping, roles, development ladder |
| [docs/CURRENT_CONTEXT.md](docs/CURRENT_CONTEXT.md) | Latest handoff note. What changed, what is open |
| [docs/GOAL_AND_OBJECTIVES.md](docs/GOAL_AND_OBJECTIVES.md) | Goal (peer-to-peer), five objectives O1 to O5, hypotheses, schedule. v4 after the 2026-10-11 meeting |
| [docs/FLOW.md](docs/FLOW.md) | A model arrives. What happens next. Flow diagram for agreement with the supervisor |
| [docs/PLATFORM.md](docs/PLATFORM.md) | Machines, software, GPU vs CPU, Metal vs CUDA review, what we ask of a participant |
| [docs/BACKGROUND_AND_DESIGN.md](docs/BACKGROUND_AND_DESIGN.md) | Definitions, PP vs TP in practice, model format and manifest, the TP on 32B question |
| [docs/BACKENDS.md](docs/BACKENDS.md) | Older technical note on Metal, CUDA, DOCA. Summarized in PLATFORM.md section 3 |
| [docs/he/](docs/he/) | Hebrew: shared research log with Codex, status, one-pager, meeting notes, next steps. HTML in docs/he/html/ |
| [docs/DECISIONS.md](docs/DECISIONS.md) | ADRs — decisions we will not silently reverse |
| [docs/LEARNING_RESOURCES.md](docs/LEARNING_RESOURCES.md) | Tagged reading list; add before using a new tool |
| [experiments/RESULTS.md](experiments/RESULTS.md) | Every measurement, append-only |
| [CLAUDE.md](CLAUDE.md) | Working rules if an AI coding assistant is used on this repo (optional) |

## Quick start (O1: one machine, chosen device)

```bash
git clone <this repo> && cd nocturne
cp env.example .env            # set MODEL=/path/to/model.gguf
bash scripts/00_build_llamacpp.sh
bash scripts/01_list_devices.sh
bash scripts/02_run_local.sh MTL0       # use the exact Mac GPU name from --list-devices; CUDA0 / CUDA1 on NVIDIA; none = CPU
bash scripts/05_bench.sh                # numbers: GPU vs CPU → experiments/bench.csv + RESULTS.md
```
Full instructions, per platform, in [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md).

## O2: two machines on the LAN, manual split

On the **worker** (e.g. the NVIDIA box):
```bash
bash scripts/03_start_worker.sh CUDA0            # prints: --rpc 192.168.x.y:50052
```
On the **requester** (e.g. the Mac):
```bash
bash scripts/06_net_baseline.sh 192.168.x.y            # RTT + bandwidth (worker runs iperf3 -s)
bash scripts/04_run_two_machines.sh 192.168.x.y 0,1    # smoke: all layers remote
bash scripts/04_run_two_machines.sh 192.168.x.y 1,1    # half/half
bash scripts/04_run_two_machines.sh 192.168.x.y 1,0    # all local (control)
bash scripts/05_bench.sh rpc 192.168.x.y               # numbers for all three splits
```
Then swap roles. Record every row in `experiments/RESULTS.md`. For recorded experiments (raw artefacts per run) use `scripts/08_r0.sh` and `scripts/10_capacity.sh`; see docs/he/RUNBOOK_R0.md. For machines in different homes see docs/he/CONNECTIVITY.md.

## Getting a model

Small, for plumbing tests: `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`. Real, for measurements: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.7 GB) or `gemma-2-9b-it-Q4_K_M.gguf` (~5.8 GB, best Hebrew in our earlier tests). Download from Hugging Face (`bartowski/…-GGUF` or `ggml-org/…`), put the path in `.env`. Keep the same file on every machine — the requester ships weights to workers on load, and the worker cache (`-c`) makes the second load fast.

## Windows note

The NVIDIA machine can run everything from **WSL2 (Ubuntu)** with the NVIDIA WSL driver — the scripts work unchanged. Native PowerShell builds follow llama.cpp's `docs/build.md`; port the scripts when needed.

## Status (2026-10-25)

- [x] Architecture (requester-coordinated, tracker-assisted), decisions, reading list
- [x] Goal and objectives v4 after the supervisor meeting of 2026-10-11
- [x] O1 single machine: Air M3 and Mac M5 measured (Table A). NVIDIA machine pending
- [x] O2 two machines, manual split: 32B runs on two Macs, fails on each alone. Table B valid in one direction; the other to be recollected
- [ ] O3 scale out: third machine, internet between homes, generic models
- [ ] O4 performance study: PP vs TP (TP on 32B to be repeated with logs), saturation, trade-offs
- [ ] O5 application: tracker, worker agent, requester CLI, planner, liveness

Hebrew documents in `docs/he/` (HTML in `docs/he/html/`). The working log between the two AI assistants is kept outside git.
