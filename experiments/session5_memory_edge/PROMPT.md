# Session 5: the memory edge of two machines, and where PP stops being enough

Work prompt for Claude Code. Written 2026-11-07 by Claude (Cowork) with Ohad. Run it on the M5 inside the
`nocturne` repo. The Air is the worker. Nothing else is in scope.

## 0. Read first, in this order

1. `CLAUDE.md` (rules 1 to 11; rule 3, 4, 8 and 10 matter most here).
2. `experiments/RESULTS.md`, the "Bench matrix" section and everything after it.
3. `experiments/row_b_session4_predictions.md` (Part 0 metric fix, Part 6 memory predictions).
4. `nocturne/bench/README.md`, `nocturne/bench/cells.yaml`, `nocturne/bench/adapters/llamacpp_rpc.py`
   (the `BUF` regex and `memory_buffers_gb`), `nocturne/bench/probes/mem.py`, `nocturne/bench/schema/`.
5. `~/.nocturne/peers.yaml` on the M5 (models, ssh to the Air, bind addresses).
6. `experiments/r3/capacity/` folders from 2026-10-10 (the only 32B evidence so far).

Team decisions that bind this session:

- T1: per token time is not a metric. Record it, never optimize it, never rank by it.
- T4 (supervisor meeting 2026-11-02): feasibility first on two machines; success is "it works and we know its
  limits"; performance is a separate later phase; automation and planners wait.
- ADR-007 symmetric rule: a failure of one engine is never evidence for the other engine.
- Measurement norms: every number links to a run id; same llama.cpp SHA `1e7bcf3` on both Macs; same model
  hash on both; placement read from the measured run's log; pre-register predictions; say what was not checked.

## 1. The question

The supervisor asked: when is layer split (PP) enough, and when does memory or model structure force an
in-layer split (TP)? Where is the crossover on our pair of machines?

Translate it into three measurable questions:

- Q-A. For each partition (local, PP, TP) what is the largest model and context that loads, decodes and
  returns output identical to the single-engine BASELINE on the Air (16 GB) plus M5 (24 GB) pair?
- Q-B. At the edge, what fills the memory on each node: weights, KV cache, compute buffers, replicated
  parts (output layer, embeddings), runtime overhead? Numbers per node, from the engine's own log.
- Q-C. Is there any (model, context) that TP completes and PP does not, on the same pair and the same
  engine? That is the only thing that would make TP necessary on memory grounds for two machines.

Not in scope: speed comparisons, three machines, internet, planners, graph cache patches, AWDL work,
Parts 2 to 5 and 7 of session 4. If you find yourself optimizing time, stop.

## 2. Pre-registered hypothesis and predictions (write before running)

Create `experiments/session5_memory_edge/PREDICTIONS.md` with a table per experiment group below, three
columns: `Ohad | Claude | outcome`. Fill the Claude column. Leave Ohad's column empty. Then STOP and tell
Ohad to fill his column. Do not start a run before his column has at least the Group 1 and Group 2 rows.

Claude's hypothesis to be written there and tested:

> On two machines, TP does not fit any model that PP cannot fit. PP splits at layer granularity
> (about 0.3 GB per layer for 32B Q4_K_M), so its only waste is one layer of slack plus the output layer on
> one node. TP splits weights roughly in half but replicates the output layer and the embedding on the
> requester, keeps a CPU fallback buffer, and in llama.cpp tensor mode over RPC also carries the graph
> metadata buffers. So TP's per node memory is never lower than balanced PP's by more than that slack.
> Prediction: the memory edge of PP is at or above the edge of TP for every context tested.
> A single falsifying run (TP completes, PP fails, same model, same context, same engine) kills it.

Also predict per node numbers for 32B Q4_K_M at ctx 4096: weights on each node, KV on each node, compute
buffers on each node, for PP (default proportional split) and for TP (`--split-mode tensor -ts 1,1`).
Derive them from the GGUF metadata (layer count, hidden size, KV heads, vocab) before running. The
session 4 Part 6 prediction already says "within 10 percent between TP and balanced PP, TP requester 0.3
to 0.6 GB heavier". Keep or revise it, in writing.

## 3. The ladder

### Machines and capacity lines

| node | RAM | Metal recommended working set | role |
|---|---|---|---|
| Air M3 | 16 GB | 12.1 GB | worker (ggml-rpc-server, MTL0) |
| M5 | 24 GB | 18.2 GB | requester (llama-server), also local BASELINE |

The 18.2 GB line was practically binding on 2026-10-10 (32B local on M5: loads 65/65, first compute
`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Treat both lines as the capacity, and report how close
each run gets to them.

### Model ladder (llama.cpp, Q4_K_M first, then quantization steps to cross the edge)

| step | model file | weights GB (approx) | purpose |
|---|---|---|---|
| L1 | Qwen2.5-7B-Instruct-Q4_K_M | 4.7 | already measured; sanity and control |
| L2 | Qwen2.5-14B-Instruct-Q4_K_M | 9.0 | fits on each node alone; mid point |
| L3 | Qwen2.5-32B-Instruct-Q4_K_M | 19.85 | fits only split; known PP result 4 tok/s |
| L4 | Qwen2.5-32B-Instruct-Q5_K_M | about 23.3 | first step past L3 |
| L5 | Qwen2.5-32B-Instruct-Q6_K | about 26.9 | expected near the combined 30.3 GB line |
| L6 | Qwen2.5-32B-Instruct-Q8_0 | about 34.8 | expected to fail in every partition; proves the edge |

Same family on purpose: layer count 64 and hidden size stay fixed from L3 to L6, so only bytes per weight
move. That isolates "memory" from "structure". Verify the file sizes and sha256 on both Macs after
download and record them in `experiments/RESULTS.md` "Model hashes".

Download with `hf download Qwen/Qwen2.5-32B-Instruct-GGUF <file> --local-dir ~/models` on the M5, then
`scp` to the Air. Before any download: `df -h ~` on both Macs, and STOP to ask Ohad if free space after
the download would be under 30 GB on either machine. L4 to L6 are about 85 GB together. If space is
short, do L4 and L5 only; L6 can be replaced by a prediction with a one line justification.

### torch_tp ladder (engine B, bf16; TP only, the engine has no PP yet)

| step | model | weights GB bf16 | purpose |
|---|---|---|---|
| T1 | Qwen2.5-1.5B-Instruct | 3.1 | control |
| T2 | Qwen2.5-7B-Instruct | 15.2 | known to work (B1t); per node about 7.6 plus replicated 2.2 |
| T3 | Qwen2.5-14B-Instruct | 29.5 | optional edge probe; expected OOM on the Air side |

T3 only if disk and time allow. It answers "largest TP model in engine B on this pair", nothing else.

### Context ladder

`--ctx` 512, 4096, 16384, 32768. KV cache grows linearly with context; it is the knob that moves a model
that fits at 512 over the edge at 32768. At each context also send one real long prompt (Group 3) so the
KV is actually populated, not only reserved.

### Partitions (cells)

| cell | partition | engine | flags note |
|---|---|---|---|
| BASELINE on M5 | none | llama.cpp | local, `-ngl 99`; reference for correctness and for "fits alone" |
| BASELINE on Air | none | llama.cpp | same, `--host air`; shows which models fit the small node alone |
| A1 | PP | llama.cpp RPC | default proportional split (`-ts` from peers.yaml, null means default) |
| A1 balanced | PP | llama.cpp RPC | `--override-flag` or `--tag balanced` with `-ts 1,1`; the TP comparable split |
| B1 | TP | llama.cpp RPC tensor | `--split-mode tensor -ts 1,1 -fit off`; `-fit off` is mandatory on 1e7bcf3 |
| B1t | TP | torch_tp | `--world 2`, bf16 |

Remember: llama.cpp lists RPC devices first, so `-ts 1,0` puts everything on the worker. Check
`placement.log` of every run, never assume.

## 4. Run plan, in priority order, with stop rules

Total budget: about 60 runs. A 32B load takes 80 to 450 s depending on the worker cache. Order the groups
so that the question is answered even if the session stops early.

### Group 1. Edge in the known model (L3, 32B Q4_K_M), all partitions, context sweep

For ctx in 512, 4096, 16384, 32768: BASELINE M5, A1, A1 balanced, B1. 1 run each at first; repeat 3 times
only the cells that completed. Use `--prompt-set eval_set --subset 8` for speed; correctness is graded.
Expected outcomes to classify: completes; loads then OOM at first compute; OOM at load; RPC `recv failed`;
timeout. This group alone answers Part 6 of session 4 and gives the first PP versus TP memory table.

Stop rule: if B1 fails at ctx 512 with a tool error (not OOM), record `tool_unsupported` with the first
error line and move on; do not debug llama.cpp tensor mode in this session.

### Group 2. Crossing the edge (L4, L5, L6), PP and TP

For each of L4, L5, L6 at ctx 4096: A1, A1 balanced, B1. Then for the largest one that completed in PP,
sweep ctx 512, 16384, 32768 in PP and in TP. This is where Q-C gets its answer.

Stop rule: a model that fails at ctx 512 in both PP and TP is the edge; do not try larger contexts for it.

### Group 3. Real long prompts at the edge

Take the largest (model, ctx) that completed per partition. Run `--prompt-set eval_long --fixed-tokens 256`
and one synthetic prompt of about 4000 tokens (add it as `prompts/eval_kv4k.jsonl`, one prompt, expected
null, header line per the loader convention). Memory peak during a populated KV is the number that matters
for a planner later; record it.

### Group 4. Mid points for the picture (L2 14B and L1 7B)

At ctx 4096 and 32768: BASELINE on both nodes, A1, B1. Shows the regime where every partition fits and the
memory table is about distribution, not feasibility.

### Group 5. torch_tp (T1, T2, T3 optional)

B1t at ctx 4096 for T1 and T2 (1 run each; they exist already, reuse records if identical config). T3 only
if space allows; expect OOM and record where.

### Group 6. Derive, plot, write (section 6)

## 5. Tooling changes needed before Group 1 (small, tested)

1. `--ctx` already reaches llama-server (`llamacpp_rpc.py` line ~280 passes `-c ctx*k`). Make sure the
   value used is written into the record (`setup.ctx`) so EDGE_TABLE can group by it; add a unit test in
   `tests/bench/` if it is not.
2. OOM classification. Look at `failure_class` in the schema. Make sure these map cleanly, by node:
   `oom_requester_load`, `oom_requester_compute`, `oom_worker_load`, `oom_worker_compute`,
   `rpc_disconnect`, `timeout`, `tool_unsupported`. Detect from the requester stderr and the worker log
   (Metal: `kIOGPUCommandBufferCallbackErrorOutOfMemory`, `failed to allocate`, `ggml_backend_..._buffer`;
   RPC: `recv failed`, `connection reset`). Schema change means version bump and tests; keep it minimal.
3. Per node memory from the engine log. `memory_buffers_gb` already sums model, KV and compute buffers per
   host from the `BUF` regex. Add the breakdown next to it: `memory_breakdown_gb: {host: {model, kv,
   compute, output, cpu}}`. Also save the raw matched log lines in the record so a reviewer can recompute.
4. Capacity lines in the record: `peers.<name>.metal_working_set_gb` (12.1 and 18.2) so `derive` can
   compute headroom per node. Put them in peers.yaml, not in code.
5. Preflight before every group: both binaries report SHA 1e7bcf3; model sha256 identical on both nodes;
   `scutil --nc list` shows no connected VPN on the M5 (record `network.vpn_state`); `awdl0` state
   recorded; worker reachable on 50052; `df -h` both nodes. Fail loudly, do not run.
6. A derive script `nocturne/bench/derive_edge.py` that reads `results/*.json` and writes
   `experiments/session5_memory_edge/EDGE_TABLE.md` and `.csv`: rows (model, quant, ctx, partition),
   columns (fits yes/no, failure_class, per node model/kv/compute GB, per node total and headroom, bytes per
   token, correctness vs BASELINE, ms per step as background, run ids).
7. Plots with matplotlib to PNG under `experiments/session5_memory_edge/fig/`:
   (a) stacked bars per node (model, KV, compute) for PP vs TP at each model step, with the capacity line;
   (b) fit map: model step on x, ctx on y, one panel per partition, green/red;
   (c) KV growth with ctx per partition at 32B. English labels, no codenames in titles.

Keep comments minimal (rule 11). Run `pytest tests/bench -q` after each change.

## 6. Writing the results

1. `experiments/RESULTS.md`: one appended block "Session 5, memory edge, 2026-11" with the table and the
   list of run ids. Append only.
2. `experiments/session5_memory_edge/PREDICTIONS.md`: fill the `outcome` column, every row, including
   failed predictions in plain words.
3. `experiments/session5_memory_edge/REPORT.md` (English): the three questions Q-A, Q-B, Q-C with the
   answer, the evidence (run ids), what was not checked, and the proposed crossover rule (section 7).
   Separate observed, limitation, hypothesis in every paragraph.
4. `docs/he/MEMORY_EDGE_2026-11.md` (Hebrew, for the supervisor): short sentences, no em or en dash
   (`grep -P '[\x{2013}\x{2014}]'` must return nothing), no internal codenames (write "חלוקה לפי שכבות" and
   "חלוקה בתוך שכבה", not A1 or B1; run ids only in a collapsed appendix), the three figures embedded, one
   table of "what fits where". Build HTML with `bash scripts/14_build_he_html.sh`. This is the action item
   from the 2.11 meeting: results, difficulties, measurements, with graphs.
5. `experiments/session5_memory_edge/I027_draft.md`: a draft entry for the shared log in the I-template
   (Hebrew), for Ohad to paste after he checks the latest I-number. Do not write into
   `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` yourself.
6. Git: commit at the end of each group with a message that names the group; do not push unless Ohad asks.

## 7. The crossover rule to derive from the data

Write the rule as inequalities with measured quantities, then check it against every row of the table.

Let `C_i` be the capacity line of node i, `W` total weights, `w_max` the largest single layer, `K(ctx)`
total KV at that context, `B_pp,i(ctx)` and `B_tp,i(ctx)` the compute buffers per node, `R_tp,i` the parts TP
replicates or keeps on node i (output layer, embeddings, CPU fallback, RPC graph buffers).

- PP on two nodes fits when the layers can be assigned so that each node's weights plus its KV plus its
  compute buffer (plus output on the last node) stays under `C_i`. Measure the best balanced split's
  headroom per node.
- TP on two nodes fits when `W/2 + K/2 + R_tp,i + B_tp,i(ctx) <= C_i` on both nodes.
- TP is necessary on memory grounds only when the PP inequality fails and the TP one holds. From the data,
  report the sign of `(R_tp,i + B_tp,i) - (PP slack_i)` at each model step. If it is positive everywhere,
  the answer to the supervisor is: on two machines of this class, memory never forces TP for dense models;
  the cases that force it are a single layer larger than the small node's headroom (not reached by dense
  models up to 70B at Q4 on 16 GB; show the arithmetic) or expert layers in MoE (not tested here).
- If any row falsifies this, say so first, show the row, and revise the rule.

Hybrid is out of scope for measurement, but the report should state in one paragraph, from the numbers,
which direction a third node would help: more total capacity (PP stage) versus splitting a node that is
too small for one layer (TP pair). That is the bridge to the three machine plan, nothing more.

## 8. How to run (reference)

```bash
# M5, requester. Worker must already be up on the Air.
cd ~/path/to/nocturne && source .venv/bin/activate
python -m nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --network wifi --ctx 4096 --runs 1 --subset 8
python -m nocturne.bench.run --cell A1 --model qwen2.5-32b-q4km --network wifi --ctx 4096 --runs 1 --subset 8 --tag balanced --override-flag=-ts=1,1
python -m nocturne.bench.run --cell B1 --model qwen2.5-32b-q4km --network wifi --ctx 4096 --runs 1 --subset 8
python -m nocturne.bench.run --cell BASELINE --model qwen2.5-32b-q4km --host m5 --ctx 4096 --runs 1 --subset 8
```

```bash
# Air, worker. Started by Ohad or over ssh from the bench; check the bind address in peers.yaml first.
cd ~/path/to/nocturne && bash scripts/03_start_worker.sh MTL0
```

`--override-flag` needs the `=` form (argparse), per the help text in `run.py`; a `--tag` record does not
append to RESULTS.md by default, so add the balanced rows to the session 5 block yourself. Add new models to `~/.nocturne/peers.yaml` under `models:` with `quant` set.

## 9. Stop points (ask Ohad, do not guess)

1. After PREDICTIONS.md is written, before the first run.
2. Before any download that leaves under 30 GB free on either Mac.
3. If a run needs a llama.cpp patch or a flag not present on 1e7bcf3.
4. If a result contradicts the Group 1 predictions by more than 2x in memory on any node: report, then ask
   whether to continue the ladder or stop and investigate.
5. Before writing the Hebrew supervisor document, show the EDGE_TABLE and the three figures.

## 10. Definition of done

- EDGE_TABLE has a row for every (model step, ctx, partition) attempted, with a run id or a failure class.
- Q-C has a yes or no answer with the row that decides it.
- PREDICTIONS.md has every outcome filled, failures included.
- The Hebrew document for the supervisor exists with three figures and passes the dash check.
- RESULTS.md appended, tests green, commits per group.
