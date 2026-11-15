# Session 6: generic placement. Uneven TP, any group size, PP stages of TP groups, nodes smaller than one layer

Work prompt for Claude Code. Written 2026-11-15 by Claude (Cowork) with Ohad. Run it on the M5 inside the
`nocturne` repo. The Air is the second host. Several engine processes per host are expected in this session.

## 0. Read first, in this order

1. `CLAUDE.md` (rules 1 to 11; rules 1, 2, 3, 6, 7, 8 and 11 matter most here).
2. `experiments/session5_memory_edge/REPORT.md` and `PREDICTIONS.md` (what session 5 proved and how it was written).
3. `experiments/RESULTS.md`, the "Bench matrix" section and everything after it (row B results, session 4 notes on
   torch_tp: bf16 partial sums diverge after about 256 tokens on long prompts, `--reduce-dtype` exists).
4. `nocturne/engines/torch_tp/` all files: `shard.py`, `loader.py`, `model.py`, `engine.py`, `transport.py`,
   `__main__.py`, `compare.py`. Also `tests/bench/test_torch_tp.py` and `nocturne/bench/adapters/torch_tp.py`.
5. `docs/DECISIONS.md` ADR-007 and amendments 1 and 2 (torch_tp is the engine B spike).
6. `~/.nocturne/peers.yaml` on the M5 (hosts, ssh to the Air, bind addresses, model paths).

Team decisions that bind this session:

- T1: per token time is not a metric. Record it as background. Never optimize it, never rank by it.
- T4: feasibility first. Success is "it works, the output is right, and we know its limits".
- ADR-007 symmetric rule. Nothing here is evidence about llama.cpp, and nothing here picks the product engine.
  This session builds research capability for O4 inside engine B. If a design choice needs a new ADR
  (for example the placement file format), write it in `docs/DECISIONS.md` as Proposed and stop for review.
- Measurement norms: every number links to a run id. Same git SHA and same model files (sha256) on both Macs.
  Pre-register predictions. Say what was not checked.

## 1. Why this session exists

Session 5 showed that on two Macs and dense models up to 32B, memory never forces an in-layer split. That result
is about the pair, not about the system. In a volunteer network the question flips. It is not "does the model need
TP" but "can the weakest node we want to include hold one whole unit". A unit is one transformer layer, or the
embedding, or the output head. Many volunteer machines are laptops with a CPU or an iGPU and a few GB free.
Examples from the architecture arithmetic (not measured): a 7B bf16 layer is 0.47 GB and its output head is
1.09 GB. A 405B dense layer at Q4 is about 1.9 GB. A DeepSeek-V3 class MoE layer is about 11 billion parameters.

Ohad's design goal: a placement is a list of PP stages. Each stage is a group of ranks that split the stage's layers
inside the layer (TP). Group sizes can differ between stages. Shares inside a group can be unequal. Group sizes are
not limited to powers of two. A node smaller than one layer still contributes, as a member of a group.
Example from Ohad: 8 stages, each a TP pair, 16 machines in total.

The current engine cannot express this. `Shard` requires heads, KV heads and the MLP width to divide by `world`.
Shares are equal. There are two ranks only and no stages. Rank 0 holds the whole embedding and output head.

## 2. The questions

- Q-D. Uneven and non power of two TP. Does a group of 2, 3 or 5 ranks with unequal shares produce the same greedy
  output as one rank? Including groups larger than the number of KV heads (KV head replication).
- Q-E. Stages. Does a placement of several PP stages, each a TP group of its own width, produce the same output as
  one rank? Including Ohad's shape: 8 stages of 2 ranks, 16 processes.
- Q-F. Weak nodes. With a memory budget per rank that is smaller than one layer and smaller than the output head,
  does a hybrid placement complete with the right output? Does the planner reject the same nodes under pure PP?
  Does a forced pure PP run fail on the budget, as the planner says?
- Q-G. Accounting. Per rank, how close are the planner's bytes to what the rank actually holds? How many messages
  and bytes per token does each placement send, against a formula written before the runs?

Not in scope: speed, llama.cpp patches, quantization in torch_tp, the NVIDIA node, the internet, MoE models,
weight distribution to workers (each host keeps the model files locally for now). If you find yourself making
anything faster, stop.

## 3. Part A. Engine changes (build and test before any run)

Keep the code small. Pure functions where possible (rule 6 spirit). Run `pytest tests/bench -q` after each step.

A1. Uneven shard spec. Replace "divide by world" with an explicit spec per rank:
  - a contiguous range of query heads `[q0, q1)`, any sizes, zero allowed;
  - the KV heads it needs follow from GQA: `{h // (n_heads / n_kv_heads) for h in [q0, q1)}`. A KV head needed by
    two ranks is held by both (replication). Its K and V weight rows and its KV cache are duplicated. Map local
    query heads to local KV heads correctly; the current `repeat_interleave` assumes an even split;
  - a contiguous range of MLP rows `[m0, m1)`, aligned to a granule (default 128, configurable), zero allowed;
  - `o_proj` and `down_proj` split by the same column ranges. Bias of row split layers on one rank only.
  The only exactness constraint left is whole heads. A rank with zero heads and some MLP rows is legal.
  A rank with zero of both is rejected.

A2. Planner, a pure function `plan(cfg, placement) -> list[RankSpec]`. Input is the placement file (A6). Output
  per rank: layer range, head range, KV heads held, MLP rows, vocab rows, planned bytes split into weights,
  KV at `max_seq_len`, and a workspace estimate. Shares inside a group follow each rank's `weight` (default 1), or
  its budget when `weight` is absent. Round heads to integers and MLP and vocab rows to the granule. Balance the
  total bytes per rank, not heads and rows separately: a rank that gets more attention gets fewer MLP rows.
  Reject the placement, with the rank and the numbers, if any rank exceeds its budget.
  Unit tests: coverage (every head, every MLP row, every vocab row assigned exactly once, KV replication excepted),
  extremes (group larger than the number of heads, so some ranks hold MLP rows only), rounding, rejection.

A3. Vocab parallel embedding and output head (Megatron style).
  - Embedding split by vocab rows: each rank looks up only the ids in its range, writes zeros elsewhere, then one
    all-reduce in the group.
  - Output head split by vocab rows: each rank computes its logits, takes the local max and index, and sends
    `(value, global index)` to the coordinator. The coordinator picks the max; ties go to the lowest index. Greedy
    only. Check the result against the full argmax in a unit test.
  - Tied embeddings (0.5B, 1.5B): the same tensor is needed by the first and the last stage. Load it in both places
    and count it in both. Do not hide the duplication.

A4. N rank group all-reduce. All-to-all is fine for groups up to about 6: every rank sends its partial to every
  other rank. The trap: every rank in a group must end with bitwise identical activations, or the ranks drift
  apart. So sum the partials in rank order on every rank, accumulate in fp32, round to bf16 once. Use
  `float32` reduction as the default this session; keep one bf16 control run (Part B, P1).
  Unit test: three in-process ranks on CPU with a tiny random Qwen2 config, fp32, compared with one rank (tolerance
  1e-5), and identical outputs on all ranks of a group.

A5. Stages and the pipeline. A stage holds a half open layer range. Every rank of a stage holds its row slices of
  every layer in that range. After a stage, its leader sends the hidden state to every rank of the next stage.
  The last group sends `(value, index)` to the coordinator. The coordinator (the process that serves HTTP and owns
  the tokenizer) sends the next token id to the first stage. The coordinator may also be a rank of the first stage.
  Keep decode one token at a time (no micro batching).

A6. Placement file (YAML, under `experiments/session6_hybrid/placements/`). Suggested shape, adjust if needed and
  write the final schema in the README of the engine:

  ```yaml
  model: qwen2.5-7b-bf16
  coordinator: {host: m5}
  reduce_dtype: float32
  max_seq_len: 4096
  granule: 128
  stages:
    - name: s0
      layers: [0, 12]
      embed: true
      ranks:
        - {host: m5, device: mps, port: 50070}
    - name: s1
      layers: [12, 14]
      ranks:
        - {host: m5, device: mps, port: 50071, budget_gb: 0.40}
        - {host: m5, device: mps, port: 50072, budget_gb: 0.40}
        - {host: air, device: mps, port: 50071, budget_gb: 0.40}
    - name: s2
      layers: [14, 28]
      ranks:
        - {host: air, device: mps, port: 50070}
    - name: head
      lm_head: true
      ranks:
        - {host: m5, device: mps, port: 50073, budget_gb: 0.40}
        - {host: air, device: mps, port: 50072, budget_gb: 0.40}
        - {host: air, device: mps, port: 50073, budget_gb: 0.40}
  ```

  A stage with no layers that holds only the head (as above) is allowed.

A7. Memory budget per rank, enforced, not only planned.
  - Planner check (A2) before loading.
  - On MPS, cap the process: `torch.mps.set_per_process_memory_fraction(budget / recommended_max_working_set)`.
    Verify first, in a ten line probe, that allocating above the cap raises in this torch build. If it does not
    raise, STOP and report (stop point 3).
  - On CPU there is no allocator cap. Record peak RSS (`resource.getrusage`) and the tensor bytes, and report both.
  - Define the budget as tensor bytes plus workspace. The Python and torch runtime overhead per process is outside
    the budget. Measure it once per device (an idle rank process) and report it as its own number.

A8. Launcher. `python -m nocturne.engines.torch_tp launch --placement <file> --model <dir> ...` starts every rank
  over ssh (Air) or locally (M5), waits for hello from all, then serves HTTP like `serve`. It stops every process at
  the end. Bind addresses follow rule 7: loopback for same host links, the LAN address from peers.yaml for cross
  host links, never `0.0.0.0`. State in the README that the ports are open without authentication (ADR-009).

A9. Record per run, in `experiments/session6_hybrid/runs/<run_id>/record.json` (or the bench schema if that is
  cheap; your choice, say which): git SHA, torch version, model dir and sha256 of its files, placement file and its
  hash, host and device per rank, per rank planned bytes (weights, KV, workspace), loaded tensor bytes,
  `current_allocated` and `driver_allocated` (MPS), peak RSS, frames and bytes sent and received per link per decode
  step, tokens per prompt, correctness against the reference, `vpn_state` and `awdl` state, ms per step as
  background. Save the launcher and rank logs next to it.

A10. Derive script `nocturne/engines/torch_tp/derive_hybrid.py` writes `experiments/session6_hybrid/HYBRID_TABLE.md`
  and `.csv`, and the figures under `fig/`:
  (a) one diagram per placement, generated from the YAML: stages left to right, ranks as boxes sized by planned
      bytes, host by colour, budget line where set;
  (b) planned versus measured bytes per rank, with budget lines;
  (c) frames per decode step, predicted by the formula of section 4 against measured.
  English labels, no codenames in titles.

Add to `docs/LEARNING_RESOURCES.md` before writing code (rule 2): Megatron-LM tensor parallelism (Shoeybi et al.
2019) for column and row splits and the vocab parallel head; vLLM documentation on tensor parallel size larger than
the number of KV heads (KV head replication); Petals (Borzunov et al.) as related work: volunteer inference with
pipeline stages of whole blocks. One line "why" each.

## 4. Pre-registered predictions (write before running)

Create `experiments/session6_hybrid/PREDICTIONS.md` with three columns: `Ohad | Claude | outcome`. Copy the numbers
below into the Claude column, check them against the real config files, and revise any you disagree with in
writing. Leave Ohad's column empty. Then STOP and ask Ohad to fill his column.

Model facts to verify from `config.json`: 7B: hidden 3584, MLP 18944, 28 heads, 4 KV heads, head dim 128, 28 layers,
vocab 152064, embeddings not tied. 1.5B: hidden 1536, MLP 8960, 12 heads, 2 KV heads, 28 layers, tied.
0.5B: hidden 896, MLP 4864, 14 heads, 2 KV heads, head dim 64, 24 layers, tied.
Bytes in bf16: 7B layer 466 MB, 7B embedding and head 1090 MB each, model 15.2 GB. 1.5B layer 94 MB, tied
embedding 467 MB. 0.5B layer about 30 MB, tied embedding 272 MB. KV per token per layer: 7B 2 KiB, 1.5B 1 KiB,
0.5B 0.5 KiB.

Correctness criterion: greedy tokens identical to the one rank reference for the first 32 tokens on at least 7 of
8 prompts of `eval_set` (`--subset 8`), and the same graded accuracy. Report the max absolute logit difference at the
first step. Mixed device placements (CPU and MPS ranks) are compared with both a CPU and an MPS reference.

Predictions per placement (P numbers are the run plan in section 5):

| id | prediction | confidence |
|---|---|---|
| P1 | 7B TP 2 at 3:2 (heads 17 and 11, MLP rows 11392 and 7552): per layer 282 and 186 MB. Output meets the criterion. On the long prompt the fp32 run drifts from the reference at the same token as the bf16 control or later, never earlier | 90% |
| P2 | 7B TP 3, KV aligned (heads 14, 7, 7). With MLP rows 4608, 7168, 7168 the ranks hold 129, 169, 169 MB per layer: the planner trades attention for MLP rows. With equal weights the planner output is within 5% per rank. Meets the criterion | 90% |
| P3 | 1.5B TP 3 (4, 4, 4 heads; 2 KV heads, so one KV head is replicated). Per layer about 32 MB each. Meets the criterion | 85% |
| P4 | 7B TP 5 (heads 6, 6, 6, 5, 5; two KV heads replicated). Per layer 91 to 97 MB. Meets the criterion | 85% |
| P5 | 7B PP only in torch_tp, 2 stages of 1 rank (12 and 16 layers, as llama.cpp chose in A1). Meets the criterion; frames per step 2 (hidden out, token back) | 95% |
| P6 | 7B hybrid without budgets: s0 on M5 GPU, s1 TP 3, s2 on Air GPU. Meets the criterion | 85% |
| P7 | 7B weak nodes (the YAML in A6). Weak ranks hold 0.32 to 0.33 GB in s1 (two layers at 10, 9, 9 heads, plus 8 MB KV) and 0.36 GB in the head group, under a 0.40 GB budget that no single layer (0.47) and no whole head (1.09) fits. Completes and meets the criterion | 75% |
| P7-ctl | Same nodes, pure PP: the planner rejects it naming the s1 rank (0.47 GB > 0.40). Forced run with the MPS cap: out of memory at load on that rank | 90% |
| P8 | Ohad's shape: 0.5B, 8 stages of 3 layers, each a TP pair, 16 rank processes. Embedding vocab split in s0, head vocab split in s7. Meets the criterion on loopback | 85% |
| P8-wifi | The same with each pair split across the two Macs: completes | 70% |
| P8-local | The same with each pair on one Mac and the stages alternating between Macs: completes | 75% |
| Q-G | Planned weight bytes equal loaded tensor bytes within 1% on every rank. MPS `driver_allocated` is planned bytes plus a per process constant (estimate 100 to 300 MB), not proportional | 90% / 60% |

Frames per decode step (all-to-all all-reduce, group size n sends n(n-1) frames; one stage hop sends one frame per
rank of the next stage; the head group sends one frame per rank to the coordinator; the coordinator sends the token
to the other ranks of the first stage):

- P7: s0 to s1 3, s1 two layers 2 x 2 x 6 = 24, s1 to s2 1, s2 to head 3, head to coordinator 3. Total 34.
- P8: token to the partner in s0 1, embedding all-reduce 2, 24 layers x 2 x 2 = 96, seven hops x 2 = 14,
  head to coordinator 2. Total 115.
Write the formula once as a function in the derive script and check every run against it.

Falsifiers to state up front: a placement that fails the criterion while one rank passes on the same device means a
bug or a numerics problem in the split. Report it first, do not tune it away.

## 5. Run plan, in priority order, with stop rules

Keep both Macs awake for the whole session. On 2026-11-08 the Air slept mid run and 12 runs had no worker.
Before the first run, start `caffeinate -dimsu` on both Macs and keep it alive until the last run ends:
on the M5 in its own terminal or as a background process of the orchestrator, on the Air over ssh
(`ssh air 'nohup caffeinate -dimsu >/dev/null 2>&1 &'`), and stop both at the end. Preflight before every run
group checks that a `caffeinate` process is running on each host (`pgrep -x caffeinate`) and that the Air answers
ssh. If either check fails, restart `caffeinate`, then continue. Record `caffeinate: on` per host in the run record.

Order matters: the questions get an answer even if the session stops early. Start every placement on loopback
(all ranks on the M5) and move to two Macs only after the loopback run passes.

| step | placement | model | hosts | answers |
|---|---|---|---|---|
| R0 | one rank references, MPS and CPU | 0.5B, 1.5B, 7B | M5 | baseline |
| P1 | TP 2 at 3:2, fp32 reduce, plus one bf16 control | 7B | loopback, then M5 and Air | Q-D |
| P2 | TP 3, KV aligned | 7B | loopback, then 2 on M5 and 1 on Air | Q-D |
| P3 | TP 3 with KV replication | 1.5B | loopback, then two Macs | Q-D |
| P4 | TP 5 with KV replication | 7B | loopback only | Q-D |
| P5 | PP only, 2 stages | 7B | M5 and Air | Q-E |
| P6 | hybrid, widths 1, 3, 1 | 7B | M5 and Air | Q-E |
| P7, P7-ctl | weak nodes with budgets | 7B | M5 and Air | Q-F, Q-G |
| P8, P8-wifi, P8-local | 8 stages x TP 2, 16 processes | 0.5B | loopback, then two Macs | Q-E, Q-G |
| P9 (optional) | P7 with the weak ranks on CPU instead of MPS (a CPU only volunteer) | 7B | M5 and Air | Q-F |

Runs: 1 per cell first. Repeat 3 times only P7, P8-wifi and one of P1 to P4. Use `--subset 8` and 32 generated
tokens for the criterion, plus one `eval_long` prompt with `--fixed-tokens 256` per model to see where drift starts.

Stop rules:
- A placement that fails on loopback is not run across the Macs. Record it, find the cause in the split, fix,
  rerun. Report the bug and the fix in the REPORT.
- If 16 processes do not fit in memory on the Macs (runtime overhead times 8 per host), shrink P8 to 4 stages of
  2 and say so. Do not swap to a smaller model silently.
- If cross host runs fail on `recv`/reset (as on 2026-11-02), retry once, then record `rpc_disconnect` and move on.

## 6. Writing the results

1. `experiments/RESULTS.md`: one appended block "Session 6, hybrid placement, 2026-11" with the table and run ids.
2. `PREDICTIONS.md`: fill every outcome, failed ones in plain words.
3. `experiments/session6_hybrid/REPORT.md` (English): Q-D to Q-G, each with answer, evidence (run ids), what was
   not checked. Separate observed, limitation, hypothesis in every paragraph. Add a short section "the general
   rule": when is a group needed (a unit larger than a node), what a group costs per token in messages (the
   formula), and what an unequal or non power of two group costs (replicated KV heads, measured).
4. `docs/he/HYBRID_2026-11.md` (Hebrew, for the supervisor): short sentences, no em or en dash
   (`grep -P '[\x{2013}\x{2014}]'` must return nothing), no internal codenames (write "חלוקה לפי שכבות",
   "חלוקה בתוך שכבה", "שלב", "קבוצה"), run ids only in a collapsed appendix, the figures embedded. Build HTML with
   `bash scripts/14_build_he_html.sh`.
5. `experiments/session6_hybrid/I028_draft.md`: a draft entry for the shared log in the I template (Hebrew), for
   Ohad to paste after he checks the last I number. Do not write into `docs/he/SHARED_RESEARCH_AND_DECISIONS.md`.
6. Git: commit at the end of each part and each run group with a message that names it. Do not push.
   The sandbox may fail to remove `.git/index.lock`; if so, leave the commit commands for Ohad.

## 7. Stop points (ask Ohad, do not guess)

1. After `PREDICTIONS.md` is written, before Part A code.
2. After Part A: tests green and the R0 references done. Show the planner output for P7 and P8 before any run.
3. If the MPS memory cap does not raise above the cap in this torch build.
4. Before any download that leaves under 30 GB free on either Mac (0.5B safetensors is about 1 GB; 1.5B and 7B
   should already be on both Macs from B1t; check).
5. If P1 or P2 fails the criterion after one fix attempt.
6. Before writing the Hebrew supervisor document, show the HYBRID_TABLE and the figures.

## 8. Definition of done

- Every placement in section 5 has a run id or a recorded failure with its cause.
- Q-D, Q-E, Q-F each have a yes or no answer with the run that decides it. Q-G has the planned versus measured
  table and the frame formula checked on every run.
- `Shard` no longer requires divisibility by the group size, and the tests cover groups larger than the number
  of KV heads and ranks with zero heads.
- PREDICTIONS outcomes filled, RESULTS appended, REPORT and the Hebrew document written, tests green, commits per
  part.
