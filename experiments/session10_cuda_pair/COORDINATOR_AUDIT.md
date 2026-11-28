# Coordinator audit (session 10, part B1)

Question: where does the code assume that the M5, or macOS, is the coordinator?
Line numbers are at commit `0e96d48`, before any change of this session.
Written 2026-11-28 on the M5.

## Two roles that were one

Before this session one process on the M5 did two jobs.
It was the **operator**: it holds `peers.yaml`, has ssh to every host, starts processes, collects files.
It was also the **coordinator**: rank 0 of the first stage, the planner, the tokenizer, the generation loop.
Most findings below come from that merge. The fix splits them.

- The operator is `hybrid_run` with `operate.py`. It can stay on the M5. It never loads the model.
- The coordinator is `coordinate.py`. It runs on the host named by `coordinator: {host: ...}` in the placement, on macOS or Linux.
- The planner runs inside the coordinator process, with the model file of that host.

The join tool uses the word "coordinator" for the operator machine. `docs/JOIN.md` now says so.

## Findings

Status: **fixed** (code changed, test named), **kept** (already correct or intended, reason given), **open** (not changed, reason given).

| # | file and line | assumption | status |
|---|---|---|---|
| 1 | `launch.py:63-68` | The planner reads the model file on the machine that launches. So the launcher machine must hold the model. | fixed. `coordinate.plan_job` plans on the coordinator host. The operator uses `plan.slots`, which needs no model. Test: `test_slots_need_no_model_and_agree_with_the_full_plan` |
| 2 | `launch.py:77-80` | `socket.gethostname()` must equal the coordinator host. So the launcher is always the coordinator. | fixed. The check moved into the coordinator process (`coordinate.check_host`). Test: `test_coordinator_process_refuses_another_host` |
| 3 | `launch.py:96-100` | The coordinator process starts every rank over ssh. Only the M5 has ssh to the others. | fixed. `Operator.start_ranks`. Test: `test_operator_starts_the_coordinator_on_the_placement_host_and_ranks_on_theirs` |
| 4 | `launch.py:99` | Every rank gets `--model` with one path. Every host must hold the file. | fixed. Per host path (finding 6), and no `--model` when weights are shipped (part C). Test: `test_operator_gives_no_model_to_ranks_when_weights_are_shipped` |
| 5 | `launch.py:114-116` | The coordinator object lives in the launcher process. | fixed. `Operator.run_coordinator` starts `coordinate` on the coordinator host. Test: `test_loopback_run_with_the_coordinator_under_a_second_name` |
| 6 | `launch.py:28-38` | `model_dir_of` gives one path for all hosts, the M5's `~/models/...`. | fixed. `peers.<host>.models.<key>` wins over the shared entry. Test: `test_model_path_may_differ_per_host` |
| 7 | `plan.py:313-315` | The coordinator host check is inline in `validate`, reachable only with the model. | fixed. One function, `plan.coordinator_host`, used by the operator and the planner. Test: `test_coordinator_host_is_rank_0_of_the_first_stage_and_must_match_the_key` |
| 8 | `plan.py:28`, `plan.py:215` | A rank without `device` gets `mps`. Wrong on a CUDA host. | fixed. The operator fills a missing device from the host's `backend` in peers.yaml and stops when there is none. Test: `test_a_rank_without_device_gets_the_backend_of_its_host` |
| 9 | `hybrid_run.py:154-159` | Engine files are compared only on ssh hosts. The local host is assumed to be the coordinator and equal. | fixed. Every host is compared with the operator's files. Test: `test_preflight_stops_when_engine_files_differ` |
| 10 | `hybrid_run.py:161-165`, `:265` | The model hash is taken on every host and must be equal. The record takes it from `specs[0]`, known only after local planning. | fixed. `model_hosts`: the coordinator host only when weights are shipped, else all. Test: `test_preflight_with_a_linux_coordinator_uses_linux_tools_and_only_its_model` |
| 11 | `hybrid_run.py:217`, `:245` | `torch_coordinator` in the record is the torch of the machine that runs `hybrid_run`. | fixed. It comes from the coordinator's record. The operator no longer imports torch for a run |
| 12 | `hybrid_run.py:251` | `Launch` (plan, model, coordinator) is built in the operator process. | fixed. `Operator` builds the job; `plan` is asked from the coordinator host (`coordinate --plan-only`). Test: `test_coordinator_process_reports_a_rejected_plan_and_a_wrong_host` |
| 13 | `hybrid_run.py:277` | The tokenizer is loaded on the operator from the local model path. | fixed. The coordinator tokenizes. Preflight checks that the coordinator host can import `transformers`. Same test as finding 10 |
| 14 | `hybrid_run.py:278-280` | The prompt files are read from the repository of the coordinator. | fixed. The operator reads them and puts the prompts into the job. Test: `test_prompt_list_of_the_job` |
| 15 | `hybrid_run.py:281-309` | The generation loop, the metrics and the frame counts run in the operator. | fixed. `coordinate.run_job` |
| 16 | `hybrid_run.py:335-339` | After the run the operator frees the engine and empties the MPS cache for the transformers check. | fixed by removal. The coordinator is another process and has ended |
| 17 | `hybrid_run.py:387-388` | `--transformers` loads the model on the operator, on the device of rank 0. | kept. It is an operator side check. It needs a local copy, or `--transformers-dir`. Stated in the help text |
| 18 | `hybrid_run.py:113-125` | The local branch of `host_state` starts `caffeinate` and reads AWDL and VPN with macOS tools. | kept. It runs only when the host's OS is Darwin (session 7). Tests: `test_host_state_linux_skips_mac_tools`, `test_preflight_with_a_linux_coordinator_...` |
| 19 | `hybrid_run.py:74`, `:80` | `stat -f` and `shasum` are macOS forms. | kept. Chosen per host OS (session 7). Test: `test_stat_and_hash_commands_per_os` |
| 20 | `hybrid_run.py:19-25`, `:242` | `ROOT`, the session folders and the git SHA are those of the operator's repository. | kept. The record is written by the operator. Finding 9 guarantees the same engine files on every host |
| 21 | `hybrid.py:75-84` | `footprint()` reads `libproc` on Darwin. | kept. Linux takes the `getrusage` branch (session 7) |
| 22 | `hybrid.py:389-390` | `torch.mps.set_per_process_memory_fraction`. | kept. Guarded by `device == "mps"` and the `--mps-fraction-cap` flag |
| 23 | `__main__.py:50-69` | `plan` plans with the local model. | kept. It is the local tool. `coordinate --plan-only` is the form for another host |
| 24 | `__main__.py:70-84`, `launch.py` | `launch` (HTTP server) runs the coordinator in its own process and starts ranks over ssh. | kept. Run it on the coordinator host. It still needs ssh from that host to the others. Not used by recorded runs |
| 25 | `derive_hybrid.py:16-17`, `:25-27` | Colors and names for `m5` and `air` only. Reads the operator's peers.yaml and a local `config.json`. | open. Session 6 chart script, not on the run path. `timeline.py` replaces it for new runs and names no host |
| 26 | `nocturne/bench/run.py:129`, `:531`; `preflight.py:32` | The llama.cpp bench takes the requester from `roles.requester` or `--host`. | kept. Already configurable. Not our engine |
| 27 | `nocturne/bench/derive_edge.py:17`, `:46`, `:53`, `:82`; `plot_edge.py:25`, `:54` | Analysis scripts of session 5 name `m5` and `air`. | open. Old analysis, not on any run path |
| 28 | `scripts/16_join_apply.py:22` | The tool edits `~/.nocturne/peers.yaml` of the machine it runs on. | kept. That machine is the operator. peers.yaml belongs to the operator; the coordinator needs none (finding 30) |
| 29 | `scripts/16_join_apply.py:54-58` | `ipconfig getifaddr` is macOS. | kept. Linux reads `ip route` (engine GGUF part 1) |
| 30 | `~/.nocturne/peers.yaml` | `ssh:` entries are as seen from the M5 (`ssh: null` for the M5 itself). `roles:` is for the bench. `models:` paths assume the same `~` path everywhere. | fixed by design. The coordinator gets host addresses in the job, not from peers.yaml. New optional keys: `peers.<host>.models`, `peers.<host>.run_root` |
| 31 | `~/.nocturne/peers.yaml`, peer `pc` | `bind` and `networks` hold the stale hotspot address 172.20.10.2. | open. Needs the join tool on `pc` when it is on the home network |
| 32 | `scripts/15_join.sh:303` | A machine with no model files fails the `models` check, "the coordinator copies them". | fixed. Now SKIP: a worker of the torch engine needs no model file |
| 33 | `docs/JOIN.md:4` | "The coordinator (the M5)". | fixed. The text names the operator |
| 34 | `nocturne/bench/live/hybrid.py` | Talks to a `launch` server on localhost. | open. Untracked file of Ohad's, not touched |

Counts: 34 findings. 19 fixed, 11 kept, 4 open.

## What a coordinator host needs now

- The repository at the same engine files as the operator (preflight compares a hash).
- The model file. With shipped weights it is the only host that needs it.
- `transformers` and `gguf` in its Python, to build the tokenizer.
- TCP reach to every rank on its port, and from every rank that connects back. No ssh.

## What was run

- Unit tests: `tests/bench/test_session10_coordinator.py` (15 tests).
- Loopback with the coordinator under a second peer name, in its own process and its own run folder, collected by the operator: `test_loopback_run_with_the_coordinator_under_a_second_name`.
- Not run: gtx as coordinator. gtx and pc were off on 2026-11-28. The command for part 2 is a placement with `coordinator: {host: gtx}`, run from the M5.
