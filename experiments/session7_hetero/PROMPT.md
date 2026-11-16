# Session 7a: make the torch_tp hybrid engine ready for a Mac plus CUDA pair

Work prompt for Claude Code. Written 2026-11-16 by Claude (Cowork) with Ohad. Run it on the M5 inside the
`nocturne` repo. This is a short, bounded session. It prepares the engine and the files. It does not run anything on
the CUDA machine: Ohad runs the first heterogeneous test by hand with `docs/he/html/GUIDE_HETERO_TORCH.html`.
The long autonomous session comes after that test.

## 0. Read first

1. `CLAUDE.md` (rules 1, 3, 6, 7, 8, 11).
2. `experiments/session6_hybrid/PROMPT.md` and `REPORT.md` (placement file, references, criterion, frame formula).
3. `nocturne/engines/torch_tp/`: `plan.py`, `hybrid.py`, `hybrid_run.py`, `launch.py`, `engine.py`, `model.py`, `README.md`.
4. `tests/bench/test_hybrid.py`.
5. `docs/he/GUIDE_CUDA.md` sections 1 and 2 (WSL2 networking and ssh, written 7.11 for llama.cpp).

Binding decisions: T1 (time is background, never a criterion), T4 (feasibility first), ADR-009 (security deferred,
never bind 0.0.0.0). Same git SHA and same model sha256 on every host.

## 1. The new host

Lenovo ThinkPad P16v Gen 1, Windows 11 with WSL2 Ubuntu, NVIDIA laptop GPU (Ampere or Ada: RTX A500 4 GB,
RTX A1000 6 GB, RTX 2000 or 3000 Ada 8 GB; Ohad will confirm with `nvidia-smi`). bf16 is native on all of them.
WSL2 runs in mirrored networking mode, so the Ubuntu side has the laptop's LAN address. Its sshd listens on port 2222
(the Windows side may own 22). The M5 reaches it as the ssh alias `p16v` from `~/.ssh/config`. peers.yaml key: `pc`.
The coordinator stays on the M5.

## 2. Findings from a static review (Cowork, not run; verify each before fixing)

| id | where | finding | effect |
|---|---|---|---|
| F1 | `plan.py` validate, "mixed device types in one group" | A TP group with MPS and CUDA ranks is rejected | TP across Mac and CUDA is impossible today. PP across them is allowed |
| F2 | `hybrid_run.py` preflight, local branch | `subprocess.Popen(["caffeinate", ...])` raises FileNotFoundError on Linux | Any run coordinated on the PC (the CUDA R0 reference) dies in preflight |
| F3 | `hybrid_run.py` model_hashes | `stat -f '%N %z %m'` is BSD; GNU stat reads `-f` as file system mode. `shasum` may be absent on minimal Ubuntu | Cache key is garbage on Linux; hashing may fail |
| F4 | `hybrid_run.py` preflight, remote branch | `caffeinate`, `ifconfig awdl0`, `scutil` are macOS only | Not fatal, but the record says "off" where it should say "n/a" |
| F5 | `hybrid_run.py` RUNS | Output dir is fixed to `experiments/session6_hybrid/runs` and references are searched only there | Session 7 runs land in the session 6 folder |
| F6 | `hybrid.py` load, `engine.py` | Only MPS is synchronized before `load_s`; CUDA OOM surfaces as a raw torch error | Timing is background only. The OOM message must name the rank and the free VRAM |
| F7 | `hybrid.py` metrics | No VRAM total and free for CUDA ranks | The record cannot show memory against the hard VRAM limit |
| F8 | `pyproject.toml` | `requires-python >= 3.11`; Ubuntu 22.04 ships 3.10 | Install fails on an old WSL image. Guide covers it; no code change |
| F9 | git | Session 6 commits are not pushed | The PC cannot clone the same SHA. Push at the end |

## 3. Tasks

T1. F1. Add a placement key `allow_mixed_devices: bool = false`. When false the rule stays as it is. When true the
planner accepts a group with different device types. Reasoning to write in the engine README: GroupComm sums all
partials on the host in a fixed order (fp32 by default), so every rank of the group receives the same bytes. The
residual stream is then updated by the same elementwise add on each device. Divergence inside one rank's slice is
ordinary kernel rounding, the same class as MPS against CPU in P9. The real check is the run against both references.
This is a small option, not a reversal of an ADR. If you think it needs one, write it as Proposed and continue.

T2. F2 to F4. Make preflight OS aware. Read `os` from the peer entry, else `uname -s`. On Linux: skip caffeinate and
record `"n/a"`, use `stat -c '%n %s %Y'` and `sha256sum`, record awdl and vpn as `"n/a"`. Also record
`nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader` for hosts with a CUDA rank.

T3. F5. Add `--session-dir` to hybrid_run, default `experiments/session6_hybrid` so old commands still work.
`latest_reference` searches the given session dir first, then session 6.

T4. F6 and F7. Call `torch.cuda.synchronize()` where MPS is synchronized. In metrics and in the READY payload of a CUDA
rank add `cuda_name`, `vram_total_bytes`, `vram_free_bytes` (from `torch.cuda.mem_get_info`), `max_allocated_bytes`.
Catch `torch.cuda.OutOfMemoryError` during load and report it through the existing KIND_ERROR path as
`"<rank name>: CUDA out of memory loading <tensor>, free <x> GB of <y> GB"`.

T5. Placements in `experiments/session7_hetero/placements/`. Model keys as in peers.yaml. Ports 50070 to 50079.

| file | content |
|---|---|
| `R0_0.5b_cuda.yaml`, `R0_1.5b_cuda.yaml` | one rank on `pc`, device cuda, coordinator pc |
| `L1_tp_mixed_0.5b_loopback.yaml` | M5 only: one TP group of an MPS rank and a CPU rank, `allow_mixed_devices: true` |
| `H1_pp_1.5b.yaml` | stage s0 layers [0, 14] embed on m5 mps; stage s1 layers [14, 28] lm_head on pc cuda |
| `H2_tp_1.5b.yaml` | one stage [0, 28] embed and lm_head; ranks m5 mps heads 6, pc cuda heads 6; `allow_mixed_devices: true` |
| `H3_pp_7b.yaml` | s0 [0, 22] embed on m5 mps; s1 [22, 28] on pc cuda with `budget_gb: 3.5`; stage head lm_head on m5 mps port 50071 |

Run `python -m nocturne.engines.torch_tp plan --placement <file>` for each and paste the planner tables into
`experiments/session7_hetero/PLANS.md`.

T6. Tests. Planner: mixed group rejected without the key, accepted with it. Preflight helpers: unit tests with a fake
runner for Darwin and Linux outputs. `python -m pytest tests/bench -q` passes on the M5 (same known failures as before).

T7. Do not run any model. Ohad runs every run himself, starting with `L1_tp_mixed_0.5b_loopback` on the M5.
Only check that each placement plans (T5) and that unit tests pass (T6). Print the exact command for L1:
hybrid_run, run id `L1_tp_mixed_0.5b_loopback_r1`, `--session-dir experiments/session7_hetero`, references
`R0_qwen2.5-0.5b-bf16_mps_r1` and `R0_qwen2.5-0.5b-bf16_cpu_r1`. Leave `PREDICTIONS.md` with an empty outcome column.

T8. Commit in small steps, then `git push origin HEAD`. Print the final SHA. Ohad needs it on the PC.

## 4. Do not

Run anything on the PC. Touch llama.cpp scripts. Optimize time. Edit old run folders. Write Hebrew in code or command
blocks. Change the guide in `docs/he/html/GUIDE_HETERO_TORCH.html` except to fix a file name or flag that you changed;
if you do, say so in the summary.

## 5. Done means

1. `allow_mixed_devices` exists, tested, documented in the engine README.
2. hybrid_run runs on Linux as a coordinator (proved by unit tests with fake outputs; the real proof is Ohad's R0).
3. CUDA ranks report VRAM in metrics and a named OOM error.
4. Five placement files exist and plan without error; PLANS.md has their tables.
5. No run was made. The L1 command is printed for Ohad.
6. Pushed. Final message: SHA, what changed per finding, what was not checked.
