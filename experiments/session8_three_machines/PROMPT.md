# Session 8: PC back on home Wi-Fi, then three machines

Work prompt for Claude Code. Written 2026-11-20 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo.
Unlike session 7a, this session runs things. Ohad asked Claude Code to run the connection checks and the three rungs,
and to update every record from the measured results. Ohad is nearby for the few steps that need a human (section 3).

## 0. Read first

1. `CLAUDE.md` (rules on provenance, append-only RESULTS, no edits of old runs).
2. `docs/he/html/GUIDE_PC_HOME_WIFI.html` and `docs/he/html/GUIDE_HETERO_TORCH.html` (text, ignore styling). They hold the exact commands and the reasons.
3. `experiments/session7_hetero/` (`PROMPT.md`, `PLANS.md`, `PREDICTIONS.md`, `placements/`, `runs/`) and its rows in `experiments/RESULTS.md`.
4. `experiments/session6_hybrid/REPORT.md` for the Air runs (frame formula, references, torch 2.15.0 on the Air).
5. `nocturne/engines/torch_tp/README.md` (preflight, `launch`, `hybrid_run`).

Binding decisions: T1 (time is background, never a criterion), T4 (feasibility first), ADR-009 (security deferred, but never bind 0.0.0.0).
Writing rule for every Markdown file you touch: short sentences, no em dash, no en dash. No Hebrew in code or command blocks.

## 1. Machines and what is known

| host key | machine | device | address now | notes |
|---|---|---|---|---|
| m5 | Mac M5 24 GB, coordinator | mps | 192.168.1.242 | runs this session, torch 2.14.1 |
| air | MacBook Air M3 16 GB | mps | 192.168.1.219 | torch 2.15.0, ssh target in `~/.nocturne/peers.yaml`, slept once on 8.11 |
| pc | Lenovo P16v, WSL2 Ubuntu 24.04 | cuda | unknown, Ohad moved it from the phone hotspot | ssh alias `p16v` port 2222, RTX 2000 Ada 8.59 GB (about 6.2 usable), torch 2.14.1+cu130, mirrored networking |

The addresses may have changed. Measure them, do not trust this table. Hotspot leftovers (`172.20.10.x`) in `~/.ssh/config` and `~/.nocturne/peers.yaml` must go.
Known results to compare against: H1 (PP 1.5B m5 plus pc) 8/8 on both references, 2 frames per step. H2 (TP 1.5B m5 plus pc) 8/8, 116 frames per step. Both ran on the hotspot.

## 2. Goal

1. Establish and verify the home Wi-Fi connection of all three machines, and correct every address and setting that is stale.
2. Climb a ladder of runs, stopping at the first rung that fails and diagnosing it.
3. Write every measured fact into the repo records. Ohad should not have to update anything by hand.

## 3. Human checkpoints

Some steps cannot be done from here. When you reach one, print exactly what Ohad must do, on which machine, in a code block, then wait for him to answer. Do not skip and do not guess.

| step | why it needs Ohad |
|---|---|
| Windows side: connect Wi-Fi, set the network profile to Private, firewall rules, sleep settings, router DHCP reservation | Windows PowerShell as administrator, and the router page. Print the commands from the PC guide section 1 |
| `sudo` loops that hold AWDL down on the M5 and the Air | needs a password. Print the loop, ask Ohad to start it in a separate terminal on each Mac and to confirm |
| Opening an Ubuntu window on the PC if `ssh p16v` is refused | WSL stops when no window is open |
| macOS prompts to allow incoming python connections | click on the screen |
| The PC address, if `ssh p16v` fails and the old address is stale | ask Ohad for `ipconfig` output from Windows |

## 4. Phase A: connection and configuration

Do every item. For each, record the command, the raw output and the verdict in `experiments/session8_three_machines/connectivity.md` (new file, append only, one section per check).

A1. Local state on the M5: `ipconfig getifaddr en0`, `networksetup -getairportnetwork en0`, `scutil --nc list`, `ifconfig awdl0 | head -3`, `pmset -g assertions | head -20`. Verdict: address is `192.168.1.x`, no VPN connected, awdl state noted.

A2. Reachability from the M5: ping 20 packets to the Air and to the PC (use the address from `~/.ssh/config` first). If the PC does not answer, run checkpoint "PC address" and "Windows side", then retry. Record loss and the min, avg, p95-ish (use the sorted RTTs) and max.

A3. ssh: `ssh -o BatchMode=yes -o ConnectTimeout=8 p16v 'hostname -s; hostname -I; whoami'` and the same for the Air with its ssh target. On the PC also run `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader`, `~/dev/nocturne-torch/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available())"`, `sudo -n true` is not needed. Check that `hostname -I` on the PC equals the Windows address (mirrored mode). Same on the Air for `python -c "import torch;print(torch.__version__)"` with the `python` path from its peers entry.

A4. Update configuration from what you measured, never from memory. Back up first (`cp ~/.ssh/config ~/.ssh/config.bak_session8`, same for `peers.yaml`). Then:
- `~/.ssh/config`: `HostName` of `p16v` set to the PC address.
- `~/.nocturne/peers.yaml`: `bind` and `networks.wifi` for `m5`, `air`, `pc`. Remove or leave the `hotspot` entries but never use them. `ssh` of `air` points at the home address.
- `nocturne/bench/peers.yaml` in the repo is the gitignored real file for llama.cpp. Update only the addresses there, nothing else.
Show a diff of each file in the final report.

A5. Engine and model parity: `git rev-parse --short HEAD` on all three hosts, `git status -sb` on each, and the engine file sha256 that preflight compares. For each model `qwen2.5-0.5b-bf16` and `qwen2.5-1.5b-bf16`, compare the sha256 of the files on all three hosts (the `model_hashes.json` mechanism). Fix drift by `git pull` on the host that is behind. If a model is missing on a host, copy with `scp` from the M5 and check again. Do not copy 7B in this phase.

A6. Six-way HTTP test. For each host in turn, start `python3 -m http.server 50079 --bind <its-ip>` over ssh (or locally for the M5) with a 60 second timeout, and `curl --max-time 5` it from the other two. Expected 200 in all six cases. Kill the servers afterward and check no process is left (`pgrep -f 'http.server 50079'` on each host). Record a 3 by 3 table.

A7. Stability sample: while nothing else runs, ping the PC and the Air from the M5 for 60 seconds each (`ping -c 60 -i 1`). Record loss and the sorted-RTT percentiles. If p95 is above 40 ms on a Mac pair, check AWDL first and report. Do not start a run until A6 passes.

Phase A is done when A1 to A7 pass, or when a failure is diagnosed and a human checkpoint was reached. Write `PHASE_A_VERDICT` as the last line of `connectivity.md`: `pass`, or `blocked: <reason>`.

## 5. Phase B: the ladder

Only after Phase A passes. Run on the M5, from the repo, with `PY=/Users/orosentswaig/dev/nocturne-torch/bin/python`. Before each rung:

1. `caffeinate -dimsu &` is running on the M5, and the Air answers ssh (a run died once because the Air slept).
2. Add the rung's block to `experiments/session7_hetero/PREDICTIONS.md` (new section named after the run id, written before the run, outcome column empty). Use the predictions in the table below. Planner numbers come from `plan`, not from this file.
3. Run `plan --placement <file>` and keep the table.

Run ids carry `homewifi` and an `_r1` suffix. The references are always `R0_qwen2.5-1.5b-bf16_mps_r1` and `R0_qwen2.5-1.5b-bf16_cuda_r1`. Session dir is `experiments/session7_hetero`. Command form:

```
$PY -m nocturne.engines.torch_tp.hybrid_run --placement <file> --session-dir experiments/session7_hetero --run-id <id> --reference R0_qwen2.5-1.5b-bf16_mps_r1 R0_qwen2.5-1.5b-bf16_cuda_r1
```

| rung | placement | run id | predictions to register |
|---|---|---|---|
| 1 | `H1_pp_1.5b.yaml` | `H1_pp_qwen2.5-1.5b-bf16_homewifi_r1` | completes, 2 frames per step, 8/8 on both references, first step logit diff below 1.0 against both |
| 2 | `H4_pp3_1.5b.yaml` | `H4_pp3_qwen2.5-1.5b-bf16_homewifi_r1` | completes, 3 frames per step (2 hops and the head), 7 or 8 of 8 on each reference, no rank disconnect, per-rank memory as in `plan` |
| 3 | `H5_tp3_1.5b.yaml` | `H5_tp3_qwen2.5-1.5b-bf16_homewifi_r1` | completes, frames per step equal to the `Frames per decode step` line of `plan`, 7 or 8 of 8 on each reference, no disconnect |

H4 and H5 were written on 2026-11-20 and never planned. If `plan` rejects one, fix the file (not the engine), record what you changed in `PLANS.md`, and say so in the report. If the engine itself must change, stop and ask.

Rules for the ladder:

1. Stop at the first rung that fails. A failed run still gets its record, a RESULTS row and a `failure_class`.
2. One diagnosed retry per rung. State the cause you found before retrying. A second failure ends the session at that rung.
3. A rung passes when the run finishes, the frame formula holds, and the criterion is met (at least 7 of 8 identical against at least one reference, same graded accuracy). Time is recorded as background only.
4. Record the measured ping to each peer taken just before the rung, not the Phase A numbers.
5. Never run 7B in this session. Never edit an old run folder. Never delete a failed run.
6. Raw `ssh`, `hybrid_run` logs and `rank_*.log` stay in the run folders. Do not summarise away errors.

If a rung fails with garbled text from the first token, suspect transport or dtype, not rounding, and say that in the diagnosis.

## 6. Phase C: update the records yourself

Do all of this at the end, also when the ladder stopped early.

1. `experiments/RESULTS.md`: append a section "Session 8: home Wi-Fi" with one table row per run, the same columns as the session 7a rows. Add a line for each host with torch version and repo SHA, the link (`wifi`, home, PC wired or wireless as you can tell from `networksetup` and the PC side), AWDL state, VPN state, and the measured ping before the run. Mark every number with its run folder.
2. `experiments/session7_hetero/PREDICTIONS.md`: fill the outcome column of the three new blocks. Write "Held", "Wrong" or "Not run" with the measured value. Do not edit the predictions.
3. `experiments/session7_hetero/PLANS.md`: planner tables for `H4` and `H5`.
4. `experiments/session8_three_machines/connectivity.md`: the Phase A record, finished, with a short summary table on top: check, result, value.
5. Update `docs/he/html/GUIDE_PC_HOME_WIFI.html`: fix any command, flag or expectation that proved wrong, replace placeholders (`<air-ssh>`, `<air-repo>`) with the real values, add a short "מה נמצא בפועל" section with the measured numbers. Keep it RTL and keep commands in English. If you change nothing, say so.
6. `docs/CURRENT_CONTEXT.md`: add or refresh the line on the PC and the three-machine state. Keep it short.
7. Add one entry to the private shared log `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` following the protocol there (append only, finding versus decision). Never commit or link that file.
8. Commit in small steps (config is not committed: `~/.ssh/config`, `~/.nocturne/peers.yaml` are outside the repo), then `git push origin HEAD`. Run the PC and Air `git pull` afterward so all three hosts share the SHA. Print the final SHA.

## 7. Do not

Bind `0.0.0.0`. Change engine code without asking. Optimise anything for time. Run 7B or llama.cpp experiments. Install or upgrade torch anywhere (the Air stays on 2.15.0 and the difference is recorded). Edit raw manifests or old run folders. Use the hotspot addresses. Leave `http.server` or other test processes running. Write secrets or passwords into any file.

## 8. Done means

1. `connectivity.md` is complete and ends with a `PHASE_A_VERDICT` line.
2. `~/.ssh/config` and `~/.nocturne/peers.yaml` hold the home addresses, with backups and diffs reported.
3. Each rung that ran has a run folder, a RESULTS row and a filled outcome block. Rungs that did not run say why.
4. The guide has the real values, nothing is left as a placeholder.
5. Everything is pushed and the three hosts show the same SHA.
6. Final message, under 25 lines: what passed, what failed and its cause, the measured ping and loss per pair, what Ohad must still do, and the first thing the long session should build on.
