# nocturne-bench

One command runs one cell of the experiment matrix on the configured peers, measures everything, checks
correctness, and writes one JSON record. Rows are model partitions, columns are what fills idle workers
during decode, and the third axis is the network. Time is measured but is not a success criterion (T1).

| | 1 single | 2 speculative | 3 outline | 4 independent |
|---|---|---|---|---|
| A PP (llama.cpp RPC layer split) | ready | ready (A2, A2b) | later | ready |
| B1 TP (llama.cpp `--split-mode tensor` over RPC) | ready | later | later | later |
| B1t TP (own PyTorch engine, `nocturne/engines/torch_tp`) | ready | later | later | later |
| B1d TP (distributed-llama) | ready | tool_unsupported (B2d) | later (B3d) | tool_unsupported (B4d) |
| C EP (MoE experts via `-ot`) | later | later | later | later |
| D hybrid (reference engine) | later | later | later | later |

BASELINE is the same model on one machine, no `--rpc`. Every other cell is graded against the BASELINE
of its requester host, same model, same engine and engine commit. The BASELINE cell takes its adapter from
`models.<name>.engine` in peers.yaml (default `llamacpp_rpc`), so each engine has its own BASELINE.

## Run

```bash
source .venv/bin/activate                       # created with: uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
cp nocturne/bench/peers.example.yaml nocturne/bench/peers.yaml   # edit hosts, ssh targets, bind addresses
python -m nocturne.bench.run --cell BASELINE --host m5 --model qwen2.5-7b-q4km
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --runs 3 [--subset 10]
python -m nocturne.bench.run --cell A2 --network wifi --model qwen2.5-7b-q4km      # needs models.<name>.draft
python -m nocturne.bench.run --cell A4 --network wifi --model qwen2.5-7b-q4km      # k = number of long prompts
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --dry-run   # print every command, run nothing
python -m nocturne.bench.run --cell B1t --network wifi --model qwen2.5-7b-bf16 --prompt-set eval_long --fixed-tokens 256 --capture-worker
python -m nocturne.bench.report                 # Markdown matrix from results/, pooled ms per step first
python -m nocturne.bench.derive_v3              # schema v3 values of every record -> results/derived_v3.json (records untouched)
pytest tests/bench
```

Before a networked cell: the worker host runs `iperf3 -s`. RTT and bandwidth are measured before every run.
Utilization on Metal and the traffic capture need passwordless sudo for `powermetrics` and `tcpdump` on
the sampled host. Without it the fields are null and the record says why in `notes`.

## What one run does

1. `probes/net.py`: ping median over 50 packets and iperf3 for 10 s to the first worker.
2. Adapter `prepare()`: binaries, flags (checked against `llama-server --help`), model files, sha256 (cached).
3. `start_workers()`: `ggml-rpc-server -H <bind> -p 50052 -c -d <device>` with `GGML_RPC_DEBUG=1`, over ssh
   for remote peers. Waits for the port.
4. `launch()`: `llama-server` on the requester, bound to 127.0.0.1, with the cell's flag template plus the
   fixed deterministic flags `--temp 0 --seed 42 --samplers temperature --repeat-penalty 1.0`. Waits for `/health`.
   Load time is `setup.weight_transfer_s`.
5. `probes/util.py` samplers start on every peer; `probes/capture.py` starts tcpdump on the worker port.
6. One untimed warm-up request (first prompt, 8 tokens; `--no-warmup` skips it, `timing.warmup_s` records it), then the
   prompts go to `/v1/chat/completions` one at a time (k at a time for column 4). Each response's `timings` gives
   prefill and decode numbers. The decode window is `[t_done - predicted_ms, t_done]`. `predicted_ms` starts after the
   first token, which comes from prefill, so it covers `predicted_n - 1` steps: the record's `decode_ms_per_step_pooled`
   is `sum(predicted_ms) / sum(predicted_n - 1)` over prompts with at least 2 tokens and `decode_tps_pooled` its inverse.
   `decode_tps_median`, the per prompt median of `predicted_n / predicted_ms`, is kept for comparison with v1 and v2
   records and labelled "old" in the report; it overstates speed on short prompts. `--prompt-set eval_long` uses
   `prompts/eval_long.jsonl` (10 open prompts) and `--fixed-tokens N` sends `max_tokens N, ignore_eos true` so every
   prompt decodes exactly N tokens; grading is skipped then and only identity to the BASELINE is checked.
7. Stop everything. `comm` comes from two sources. `rpc_commands_per_step` is the median over the last 32 compute
   steps of the worker's RPC debug log (set_tensor, get_tensor and the graph message; `scripts/12_rpc_trace.py` parse),
   `tokens_per_step` is tokens generated over compute steps, and `rpc_commands_per_token` is derived from the two,
   because a speculative or batched step yields several tokens. `bytes_per_step` and `bytes_per_token` are tcpdump
   payload bytes on the interface that owns the route to the worker (`comm.capture_iface`, from `route get` or
   `ip route get`, never `-i any`), inside the union of the decode windows; the raw packets are saved as `capture.tsv`
   with a direction column (`out` to the worker, `in` from it). Schema v3 adds the per decode step view, divided by
   `sum(predicted_n - 1)` inside the decode windows: `blocking_syncs_per_step` (llama.cpp `get_tensor`, the only
   command the client waits for; torch_tp all-reduces plus the STEP frame), `messages_per_step` (every command, mean),
   `bytes_per_step_decode` with `_out` and `_in`, and `graph_meta_bytes_per_step` (llama.cpp `n_tensors` times 304 B
   per `rpc_tensor` descriptor). `--capture-worker` runs the same tcpdump on the worker over ssh (`capture_worker.tsv`);
   both captures also see the background ping's ICMP echoes (`icmp.tsv`, `icmp_worker.tsv`), and the matched pairs give
   `comm.clock_offset_s` with `clock_offset_unc_s`. A background `ping -i 0.01` to the first worker runs for the whole
   run (`ping.tsv`; 0.1 s when the host refuses the short interval without root), and `network.ping_decode_ms` holds
   p50, p95, p99 inside the decode windows; the pre-run probe stays as `rtt_ms_median` with `rtt_source: probe` and is
   not a decode covariate. `network.awdl` is the awdl0 state at run start, `added_delay_ms` the `--added-delay-ms`
   value. `serial_waits_per_token` is null: it needs a timed RPC trace on the requester, not measured yet. Utilization is the
   mean sample inside the windows. `memory_buffers_gb` is the sum of device buffers from the load log per host;
   `memory_peak_gb` is the process peak sampled at 500 ms (`ps -o rss` of llama-server or ggml-rpc-server on macOS,
   `nvidia-smi memory.used` on CUDA). `setup.load_total_s` is launch to `/health`; `setup.transfer_s` is first to last
   `set_tensor` in the worker log, whose lines the runner prefixes with a wall clock. `ttft_s` is client side, request
   send to the first streamed token; `ttft_server_s` is `prompt_ms`. Records carry each peer's real `hostname -s`.
8. Correctness: graders per category in a subprocess (`python -I`, 5 s, no network through `sandbox-exec` on macOS or
   `unshare -rn` on Linux; without either the code prompts are `skipped_unsandboxed` and excluded), plus
   `exact_match_ratio` and `first_divergence_median` against the BASELINE. The baseline must match model name,
   sha256, quant, llama.cpp commit, prompt set and version, generation flags and requester host, and its accuracy is
   recomputed on the prompt ids that ran (A4 runs only the long prompts). A2 also checks greedy identity with the
   latest matching A1 (`checks.a2_identity`); when no A1 exists the note says "A1 identity check not run" and the field
   is null.
   The code grader runs each test in the sandboxed subprocess and reads one verdict line per test from stderr, tagged
   with a per run nonce; the candidate's stdout cannot score (I024). Score is passed tests over tests, capped at 1.
9. `success` needs: workers loaded, every request finished with `stop` or `length`, no host's buffers or peak above
   its `mem_gb`, memory measured on every peer (or `--allow-unmeasured-memory`, then noted), record validated and
   written. `quality_pass` is `task_accuracy >= baseline - 0.05` on the same prompts, reported separately; a run can
   be `success: true, quality_pass: false`, and null means no baseline.
10. On success one row is appended to `experiments/RESULTS.md` under "Bench matrix" (same columns as before: S is
    `rpc_commands_per_token`, B is `bytes_per_token`, "peak GB" is `memory_buffers_gb`). The engine name goes in the
    llama.cpp column as `<engine> <commit>` for engines other than llama.cpp.
11. Stopping: every spawned command is its own process group and `stop_all` signals the group, so `bash -lc` or `ssh`
    children die with the leader. `eng.stop()` runs with a 90 s bound; if it hangs or raises, the record is still
    written with `failure_class: engine_abort` and the partial data (I024).

`python -m nocturne.bench.report` groups by (model, network, requester host, engine), prints operational success and
`quality_pass` apart, and skips the ids in `results/EXCLUDE.yaml` (reason per id, listed under the tables). Records are
never deleted: smoke runs (`--subset`), the A2 runs without `--spec-type`, and preflight failures are excluded there.

Schema is v3 since 2026-11-02 (`schema/record.schema.json`); v2 stays in `schema/record.schema.v2.json` and v1 in
`schema/record.schema.v1.json`. `record.upgrade()` gives older files a v3 view in memory (the pooled timing is derived
from `per_prompt`, which every version carries; the per step comm fields stay null), so `report.py`, `find_baseline`
and the A2 check read all three. Files are never rewritten. `python -m nocturne.bench.derive_v3` writes
`results/derived_v3.json` with the v3 values of every record that the logs allow (worker log inside the decode windows,
`capture.tsv`), and `report.py` reads it for the older records. `costmodel.py` fits `ms_per_step = C + c1 * S * rtt +
c2 * B / bw` with S = blocking syncs per step, rtt = in-decode ping p50 and C fixed from the engine's one-machine run
(`fit_engine`), reporting the effective latency per sync. `python -m nocturne.bench.recompute --id <record>` recomputes what the logs allow
(rpc commands, bytes, transfer time when stamped, quality against the matching baseline) and prints it without
writing. `python -m nocturne.bench.calibrate_capture [--local]` pushes a known payload through the capture path
and prints measured against expected bytes.

### Capture calibration

| date | host | interface | payload | measured | ratio | by |
|---|---|---|---|---|---|---|
| not yet run | | | | | | run `python -m nocturne.bench.calibrate_capture --local` on each Mac, then over Wi-Fi to the Air, and fill this row |

Records: `results/<cell>_<network>_<model>_r<n>.json`, schema in `schema/record.schema.json`. Logs for one
run (server, workers, placement lines) are under `results/logs/<id>/`, gitignored. The record keeps every
launched command verbatim under `commands`.

## Live view

`python -m nocturne.bench.run ... --live` serves a page on `http://127.0.0.1:8091/` while the cell runs and prints
the URL. `python -m nocturne.bench.live --logs results/logs/<id>` replays a finished run at real speed (`--speed 2`
doubles it); with no `--logs` it attaches to the newest log directory, live if the files are still growing. The page is
one stdlib HTTP server plus Server-Sent Events, no external script, dark and light with `prefers-color-scheme`.
It shows a phase strip (probe, load with cold or warm, prefill, decode, done), one card per peer with state, device
buffers and GPU busy (or `not measured` without sudo), the RPC message stream between the cards with a marker per
decode step and running counters, and a token ticker with the slots in flight.

Sources, all read-only: `server.log` (timestamps, buffers, per request `print_timing`), `worker_<name>.log`
(`GGML_RPC_DEBUG=1` commands and sizes), `util_<peer>.tsv` (the utilization sampler's samples, written only for the
page) and `events.jsonl` (phase and request markers written by `run.py`, also only for the page). None of this enters
the record. Worker log lines carry no clock: live, they are shown as they arrive; in replay they are interleaved by step
count inside each request's span, which is an approximation. Runs before 2026-10-26 have no `events.jsonl` or
`util_*.tsv`, so their replay derives the phases from `server.log` alone and shows `not measured` for GPU busy.
The page binds 127.0.0.1 only; it opens no port to the network.

## Files

```
run.py                one cell, N runs                 report.py          Markdown 4x4 per (model, network)
live/                 parsers, tail (state, replay), server (SSE), page.html; python -m nocturne.bench.live
cells.yaml            the 16 cells + BASELINE + A2b    peers.example.yaml hosts, ssh, backend, bind, bin, dllama_bin
record.py             schema v3 validation, v1/v2 upgrade, writer   config.py  yaml loaders
recompute.py          v2 values from a run's logs, printed       calibrate_capture.py  known payload through tcpdump
derive_v3.py          v3 values of every record -> results/derived_v3.json (records untouched)
runner.py             local/ssh command runner, logs   costmodel.py       ms/step = C + c1*S*rtt + c2*B/bw
adapters/base.py      Engine ABC, Completion           adapters/llamacpp_rpc.py (rows A, B1), torch_tp.py (B1t), distributed_llama.py (B1d)
probes/net.py util.py capture.py mem.py                check/correctness.py, check/graders/<category>.py
prompts/eval_set.jsonl  30 short (6 per category) + 3 long structured;  eval_long.jsonl  10 open prompts for --fixed-tokens
```

## Row B: tensor parallelism, three engines

Gate opened: `docs/DECISIONS.md`, ADR-007 amendment 1 accepted and amendment 2 (team, 2026-10-27). Row B is three
engines in three cells, each with its own BASELINE. The engine name is part of baseline matching.

**B1, llama.cpp `--split-mode tensor` over RPC** (supervisor question Q7). Same adapter and launch path as row A with
`--split-mode tensor -fit off`. `-fit off` is mandatory on `1e7bcf3`: `common/fit.cpp` throws for `SPLIT_MODE_TENSOR`
when fit is on. The build forces flash attention on in tensor mode (`src/llama-context.cpp`). Three outcomes are
possible and all three are recorded: loads with both sides holding about half the weights and decodes; loads but
`load_tensors` shows everything local and the RPC device is ignored; assert or crash. The last two are
`failure_class: tool_unsupported` with the first error line in `notes`. A B1 failure is evidence about this build over
RPC only (ADR-007 symmetric rule).

**B1t, own PyTorch tensor parallel engine** (`nocturne/engines/torch_tp/`, adapter `adapters/torch_tp.py`). The engine
B spike of ADR-007. Two ranks: rank 0 is the requester and serves HTTP on 127.0.0.1; rank 1 is the worker, started over
ssh like `ggml-rpc-server` and bound to the network address (never 0.0.0.0; the port accepts any client, security
deferred by ADR-009). Transport is our own TCP with length-prefixed bf16 frames; two all-reduces per layer, so 56
frames each way per token on Qwen2.5-7B and 1.5B. Qwen2 architecture only, bf16 safetensors, each rank loads its slice
with `safe_open`. `comm.rpc_commands_per_step` comes from the engine's frame counters (source "engine counters:
all-reduce frames"), bytes on the wire from tcpdump on `torch_port` as in row A. Memory per rank comes from the
engine's `/metrics` (`torch.mps` allocations), because `ps -o rss` undercounts Metal. Single-rank mode (`--world 1`)
is this engine's BASELINE. Judged by the ADR-007 mandatory requirements, not by tok/s. Dependencies are the
`[bench-torch]` extra, not the base install.

**B1d, distributed-llama** (renamed from B1; B2d, B3d, B4d likewise). The adapter is `adapters/distributed_llama.py`:
`dllama worker --host <network address> --port 9999 --nthreads N` on each worker over ssh (never 0.0.0.0; the port
accepts any root, security deferred by ADR-009), `dllama-api` on the requester bound to 127.0.0.1, prompts through
`/v1/chat/completions`, then one `dllama inference` pass on the first prompt for the engine's own `Sent`/`Recv` kB per
token (the API prints none). `comm.rpc_commands_*` are null with a note: this engine has no RPC log. Bytes on the wire
come from tcpdump on port 9999 as in row A. Records carry `engine.name: distributed-llama`, its git commit,
`format: dllama-q40` and the tokenizer file. Node count must be a power of two (one root plus one worker here). B4d is
`tool_unsupported`: `dllama-api` serves one request at a time. B2d is `tool_unsupported`: no draft model.
`--dry-run --cell B1d` prints the whole command set. Lowest priority of the three.

Comparability caveat. B1d and B1t are system comparisons (different engine, weight format or dtype, kernels and
backend), not PP vs TP inside one engine. B1t runs bf16 safetensors with ATen kernels against row A's Q4_K_M GGUF
with ggml kernels. The comparable row-A run must use the same model family and each engine has its own BASELINE
(Qwen3-8B GGUF vs `qwen3_8b_q40` for B1d; Qwen2.5-7B-Instruct bf16 vs Q4_K_M for B1t). Only B1 is TP vs PP inside
one engine, and only if it loads.

Models (several GB each; downloaded by hand, see docs/he/GUIDE_BENCH.md section 2.5 and the row B section):

```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models                 # B1d pair, llama.cpp side
python launch.py qwen3_8b_q40     # in ~/dev/distributed-llama; weights from huggingface.co/b4rtaz/Qwen3-8B-Q40-Distributed-Llama
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/hf/Qwen2.5-1.5B-Instruct     # B1t, both Macs
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/hf/Qwen2.5-7B-Instruct         # B1t, both Macs, 15 GB
```

peers.yaml needs `dllama_bin` and `dllama_nthreads` per peer for B1d, `python` (absolute path to the torch venv
interpreter) and `torch_port` (default 50053) for B1t, `models.<name>.torch_model` and `models.<name>.engine: torch_tp`
for the torch models (see `peers.example.yaml`). Not verified by a run for B1d: the stream format of `dllama-api`,
whether `usage` is present, and whether a worker accepts a second root session for the `dllama inference` pass.

## Adding a cell

Add an entry under `cells:` in `cells.yaml` with `partition`, `decode`, `adapter`, `status`, a `flags` list
using the placeholders `{model} {draft} {rpc} {ts} {k} {ot}`, and `notes`. `-ts` is dropped when
`tensor_split` is null in peers.yaml. Nothing else changes.

## Adding an adapter

Subclass `adapters.base.Engine`, implement `prepare, start_workers, launch, complete, stop, counters`, and
register it in `adapters/__init__.py`. Use `self.runner` for every command so dry runs print and records log
it. `counters()` returns engine, setup, memory per host and the worker log texts; `run.py` derives S from
the logs only when they are llama.cpp RPC debug logs, so another engine should fill `comm` itself through
its counters (extend `run.py` at that one point). Row B has three engines (ADR-007 amendment 2); `torch_tp` and
`distributed_llama` fill `comm` through their counters.

## Verified against 1e7bcf3

Checked on 2026-10-26 with `~/dev/llama.cpp` at `1e7bcf3da`, `llama-server --help`, `llama-cli --help`,
`ggml-rpc-server --help`, and the sources.

- **Deterministic flags.** `--temp`, `-s/--seed`, `--samplers`, `--repeat-penalty` exist on llama-cli and
  llama-server. The per request body also pins `temperature 0, seed 42, samplers ["temperature"]`.
- **Speculative decoding is server-only.** `tools/cli` and `tools/completion` have no draft or speculative
  code. `--model-draft` (`-md`), `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`,
  `--device-draft` (`-devd`), `-ngld` are parsed for the server. `--draft-max` and `--draft-min` were removed.
  This is why every cell, including BASELINE, runs through llama-server.
- **`--device-draft` with an RPC device.** `--device-draft` calls `parse_device_list`, and `--rpc` registers
  RPC devices at parse time (`add_rpc_devices` in `common/arg.cpp`). So `RPC[ip:port]` resolves only when
  `--rpc` comes earlier on the command line. Read from source, not yet run. Next session: A2 with the draft
  on a worker uses `--rpc ... --device-draft RPC[ip:port]`.
- **`-ot` to an RPC buffer.** `parse_tensor_buffer_overrides` builds the buffer type list from the devices
  registered so far; the RPC buffer type is named `RPC[<endpoint>]` (`ggml-rpc.cpp`, `ctx->name`). Same rule:
  `--rpc` before `-ot`. Read from source, not yet run. Cell C1's template keeps that order.
- **rpc-server cache.** Flag `-c/--cache`. Directory: `$LLAMA_CACHE/rpc/` if set, else
  `~/Library/Caches/llama.cpp/rpc/` on macOS, `$XDG_CACHE_HOME/llama.cpp/rpc/` or `~/.cache/llama.cpp/rpc/`
  on Linux. File name: 16 lowercase hex digits of the 64-bit FNV-1a hash of the raw tensor bytes
  (`fnv_hash` in `ggml-rpc.cpp`, offset basis `0xcbf29ce484222325`, prime `0x100000001b3`). Only tensors
  larger than `HASH_THRESHOLD` = 10 MiB are hashed and cached. Pre-filling is possible by writing those files.
- **llama-server timings.** Per response `timings`: `cache_n, prompt_n, prompt_ms, prompt_per_token_ms,
  prompt_per_second, predicted_n, predicted_ms, predicted_per_token_ms, predicted_per_second`, plus
  `draft_n` and `draft_n_accepted` when a draft model ran (`server_slot_stats::to_json`). The bench records
  the whole object per prompt and derives decode TPS, prefill TPS and TTFT from it.
- **Other flags used:** `-np/--parallel`, `-cb/--cont-batching`, `--host`, `--port`, `--metrics`, `-c`,
  `-ngl`, `--rpc`, `-ts/--tensor-split`, `-ot/--override-tensor`. `ggml-rpc-server`: `-H -p -d -c -t`.
- **`--spec-type` is mandatory for speculation (run, 2026-10-26).** With `--model-draft` alone the server logs
  `no implementations specified for speculative decoding`, loads the draft and never uses it; `timings` then has no
  `draft_n`. A2 uses `--spec-type draft-simple`. Also, the draft follows `--device` by default and gets split over the
  RPC devices too (166 MiB of the 0.5B draft landed on the worker); `--device-draft {req_device}` keeps it local.
- **6 to 7 RPC commands per decode step (runs, 2026-10-26).** A1 on 7B: one `graph_recompute`, one `get_tensor` and
  five `set_tensor` per step, 7 in total; with `graph_compute` steps the count is 6 to 7. `rpc_commands_per_step` is
  measured, `rpc_commands_per_token` is derived from `tokens_per_step` (A2 verifies about 3 tokens per step).
- **`-lv 4` is needed for the load lines (runs, 2026-10-26).** At the default verbosity the `load_tensors ... buffer size`
  and `offloaded N/N layers` lines are absent and `memory_buffers_gb` would be empty.
- **Stream timings (source, 2026-10-26).** `to_json_oaicompat_chat_stream` puts `timings` on the last chunk when stats are
  set (`tools/server/server-task.cpp`), so the bench streams every request and reads `timings` from the last chunk.
- **`--split-mode tensor` over RPC, cell B1 (source 2026-10-27; runs 2026-10-27 and 2026-10-31).** `llama-server --help` lists
  `{none,layer,row,tensor}` and `-fit [on|off]`. `common/fit.cpp:183` throws "llama_params_fit is not implemented for
  SPLIT_MODE_TENSOR, abort" when fit is on, so `-fit off` is mandatory. `src/llama-context.cpp:3685` turns flash
  attention on when it is `auto` and errors when it is off ("SPLIT_MODE_TENSOR requires flash_attn to be enabled").
  `src/llama.cpp:161` builds one "Meta device" from the device list (default list: every non CPU device, RPC devices
  included, since `ggml-rpc.cpp` reports type GPU). `src/llama-model.cpp:356` rejects the architectures listed in
  `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1123`); Qwen2 is not listed, so the arch check passes. Backend
  sampling falls back to CPU in tensor mode (`llama-context.cpp:1227`). The adapter checks `tensor` in the help text and
  `--fit`, classifies "not implemented", `GGML_ASSERT` and "SPLIT_MODE_TENSOR" lines as `tool_unsupported`, and reports
  the outcome (a, b or c) from the buffer lines; `placement.log` keeps the Meta device and flash_attn lines.
  **Run outcome, loopback on the M5 (2026-10-27):** outcome (a). "creating a Meta device for tensor parallelism from 2
  devices" (MTL0, RPC0), `Meta() model buffer size = 2084.43 MiB` (half of the 4168 MiB layer-mode buffer, held once per
  device; the adapter attributes it to every device of the Meta set), worker received 2.31 GB cold, flash attention
  forced on, decode 15.5 tok/s with output identical to the BASELINE. The worker sees about 63 `graph_compute` per
  token in decode (graphs of 1, 7 or 27 nodes), so the serial round trip count is an order of magnitude above layer
  mode. `--split-mode row` fails at load: "device RPC0 does not support split buffers" (issue 13083 confirmed).
  **Over Wi-Fi (2026-10-31, M5 requester, Air worker, 8 ms RTT, 296 to 316 Mbps):** smoke run `B1_wifi_qwen2.5-7b-q4km_m5_r2`
  and three full runs `_r3` to `_r5`, 33 prompts each. Outcome (a) every time, 2.3 GB transferred per load (warm load
  37 to 41 s), decode 0.74 to 0.77 tok/s, prefill 10.6 to 11.6 tok/s, output identical to the BASELINE on all prompts,
  GPU busy 0.09 to 0.30 on the M5 and 0.11 to 0.14 on the Air. Layer mode (A2) on the same pair gives about 14 tok/s.
  `--split-mode row` between the two Macs fails at load with the same split-buffers line (`_row_r1`, excluded).
  Loopback records are tagged `loopback` and excluded from the matrix.
- **torch_tp engine, cell B1t (runs on the M5, 2026-10-29).** Stage 1: one rank reproduces transformers `generate`
  greedy output on 20 prompts of `eval_set.jsonl` (Qwen2.5-1.5B-Instruct bf16, MPS, 33 tok/s). Stage 2a: two ranks on
  one Mac over loopback, identical output on the same 20 prompts, 56 all-reduce frames per token, rank 1 holds only
  its 1.31 GB slice. Through the bench: BASELINE (world 1) and B1t on loopback both succeed; `comm.rpc_commands_per_step`
  = 56 from the engine counters, tcpdump bytes per token equal 56 all-reduces times two directions of the bf16 hidden
  state. Exact match against the one-rank BASELINE is 30 of 33: the three long prompts diverge after a few words
  (bf16 round-off of split partial sums; the unit test fixes the bound at 8 bf16 ulps of the max hidden magnitude per
  2-layer block, measured 5). 7B on one machine (2026-10-31): BASELINE 8.1 tok/s, 16.2 GB; two ranks on loopback 8.7 tok/s, rank 1 holds 6.5 GB (7.5 GB allocated), output identical on 10 prompts. Stage 2b (2026-10-31, rank 1 on the Air over Wi-Fi, 8 ms RTT): 20 of 20 identical, 1.9 tok/s, 56 all-reduces per step, about 0.95 MB per token on the wire. Stage 3 (2026-10-31, `B1t_wifi_qwen2.5-7b-bf16_m5_r4` to `_r6`, 7B bf16, rank 1 on the Air): 1.32 to 1.47 tok/s decode, prefill 23 to 27 tok/s, task accuracy 1.0, exact match 31 of 33 (same two long prompts as the bf16 round-off above), 56 all-reduces and 782 KB per token on the wire, rank 0 9.75 GB and rank 1 7.5 GB allocated, GPU busy 0.4 and 0.24. `torch_model` stays `~/...` in peers.yaml: each rank expands it on its own machine (fixed 2026-10-31 after three preflight failures `_r1` to `_r3`, excluded). The worker's
  "listening on" line is the readiness signal; `python -m nocturne.engines.torch_tp` must be importable in the torch venv
  on both machines (`uv pip install -e ".[dev,bench-torch]"`).
- **Not verified here:** A2b and the streaming request path have not run yet; `--spec-draft-n-max 4 --spec-draft-p-min 0.75`
  exist in `llama-server --help` on 1e7bcf3.
