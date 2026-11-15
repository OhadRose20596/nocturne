# Session 6 report: generic placement in the torch_tp engine

Runs: 2026-11-15. Engine: `nocturne/engines/torch_tp` (engine B spike of ADR-007), Qwen2.5 Instruct bf16 safetensors (0.5B, 1.5B, 7B; same sha256 on both Macs, `model_hashes.json`).
Hosts: M5 (24 GB, torch 2.14.1, coordinator) and Air M3 (16 GB, torch 2.15.0), home Wi-Fi, VPN off, AWDL up, caffeinate on both (every record says so).
Source of every number: `HYBRID_TABLE.md` and `HYBRID_TABLE.csv` (built by `python -m nocturne.engines.torch_tp.derive_hybrid` from `runs/*/record.json`). Placement files: `placements/`. Figures: `fig/`.
Nothing here is evidence about llama.cpp, and nothing here picks the product engine (ADR-007 symmetric rule). Time per step is recorded as background only (T1).

Terms. A stage holds a range of layers (PP). Inside a stage, a group of ranks splits every layer (TP). Shares inside a group can be unequal and the group can have any size.
Correctness criterion (pre-registered, adjusted with Ohad's approval): greedy tokens identical to the one rank reference (R0) on at least 7 of the first 8 prompts of `eval_set`, up to 32 tokens or EOS, and the same graded accuracy. Also reported: the largest absolute logit difference at the first step, and the first divergence on one `eval_long` prompt of 256 tokens.

## Answers in short

- Q-D. Yes. Groups of 2, 3 and 5 ranks with unequal shares give the same greedy output as one rank, including groups larger than the number of KV heads (KV heads held by two ranks).
- Q-E. Yes. Pipelines of TP groups of different widths give the same output, including Ohad's shape (8 stages of TP pairs, 16 processes) on one Mac and across two.
- Q-F. Yes on tensor bytes: with a budget of 0.40 GB per rank, below one 7B layer (0.466 GB) and below the head (1.09 GB), the hybrid placement completes with the right output (P7, 4 runs). The planner rejects the same nodes under pure PP, naming each rank, and a forced pure PP run fails at load on exactly those ranks. But on MPS a weak rank's real memory is about 1.3 GB, three times its budget, because of 1 GiB Metal heaps. On CPU it stays under the budget.
- Q-G. The planner's weight bytes equal the loaded tensor bytes exactly on every rank of every run. The frame formula written before the runs equals the measured frames per decode step on every run. On MPS the real memory of a rank is not planned bytes plus a constant: it is quantized by 1 GiB Metal heaps.

## R0. One rank references

| model | device | run | vs transformers `generate` (8 short prompts) | long prompt, first divergence from transformers | accuracy |
|---|---|---|---|---|---|
| 0.5B | MPS | `R0_qwen2.5-0.5b-bf16_mps_r2` | 8/8 identical | token 8 | 0.875 |
| 0.5B | CPU | `R0_qwen2.5-0.5b-bf16_cpu_r1` | 8/8 | token 10 | 0.875 |
| 1.5B | MPS | `R0_qwen2.5-1.5b-bf16_mps_r1` | 8/8 | token 5 | 1.0 |
| 1.5B | CPU | `R0_qwen2.5-1.5b-bf16_cpu_r1` | 8/8 | token 5 | 1.0 |
| 7B | MPS | `R0_qwen2.5-7b-bf16_mps_r3` | 8/8 | token 5 | 1.0 |

Observed. On short prompts the engine equals transformers. On the long prompt every model leaves transformers within 10 tokens. The engine after the session 6 refactor is token identical to the session 3 engine on that prompt (checked for 0.5B MPS). Transformers' own `sdpa` and `eager` attention paths disagree with each other on the same prompt (tokens 8 and 10 against the engine).
Hypothesis. bf16 greedy decoding over 256 tokens is sensitive to the order of additions; any change of kernel moves the first divergence. Not a bug in the engine.
Consequence. Long prompt divergence is reported as a number, not as pass or fail. The CPU and MPS references agree on 8/8 short prompts and part on the long prompt at token 10 (0.5B) and 54 (1.5B).
Harness incidents (no data used): `R0_qwen2.5-0.5b-bf16_mps_r1` (driver bug after a good run), `R0_qwen2.5-7b-bf16_mps_r1` (stopped: the M5 swapped with many apps open), `R0_qwen2.5-7b-bf16_mps_r2` (stopped: the driver held two copies of 7B during the transformers check; fixed in `8083a9d`). Each folder has an `ABORTED.txt` where it applies.

## Q-D. Uneven and non power of two TP

| cell | model | group | heads per rank | KV head copies (model has) | MLP rows per rank | per layer MB | hosts | identical | first step logit diff | long prompt first divergence | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 7B | 2, weights 3:2 | 17, 11 | 5 (4) | 11392, 7552 | 281.7, 186.3 | loopback | 8/8 | 0.5 | 49 | `P1_loopback_qwen2.5-7b-bf16_r1` |
| P1 bf16 control | 7B | same, bf16 reduce | | | | | loopback | 8/8 | 0.5 | 49 | `P1_loopback_bf16ctl_qwen2.5-7b-bf16_r1` |
| P1 | 7B | same | | | | | M5 + Air | 8/8 (3 runs) | 0.375 | 105 (3 runs) | `P1_2mac_qwen2.5-7b-bf16_r1` to `_r3` |
| P2 | 7B | 3, heads forced 14, 7, 7 | 14, 7, 7 | 4 (4) | 5888, 6528, 6528 | 156.0, 155.1, 155.1 | loopback | 8/8 | 0.375 | 72 | `P2_loopback_qwen2.5-7b-bf16_r1` |
| P2 | 7B | same | | | | | 2 M5 + 1 Air | 8/8 | 0.5 | 49 | `P2_2mac_qwen2.5-7b-bf16_r1` |
| P3 | 1.5B | 3, equal | 4, 4, 4 | 4 (2) | planner | 31.1 to 32.3 | loopback | 8/8 | 0.56 | 97 | `P3_loopback_qwen2.5-1.5b-bf16_r1` |
| P3 | 1.5B | same | | | | | 2 M5 + 1 Air | 8/8 | 0.59 | 97 | `P3_2mac_qwen2.5-1.5b-bf16_r1` |
| P4 | 7B | 5, equal | 6, 6, 6, 5, 5 | 8 (4) | 3840, 3712, 3712, 3840, 3840 | 93.6 to 95.4 | loopback | 8/8 | 0.5 | 127 | `P4_loopback_qwen2.5-7b-bf16_r1` |

Accuracy equals the reference in every row (1.0). Frames per decode step equal the formula in every row (116, 346, 346, 1148).

Observed.
- Every group size and every share meets the criterion, on loopback and across the Macs. A group larger than the number of KV heads works (P3: 3 ranks, 2 KV heads; P4: 5 ranks, 4 KV heads).
- All ranks of a group hold bitwise identical activations. This is enforced by construction (every rank sums the partials in group rank order on the host) and checked in the unit tests; no run showed a rank disagreement (it would end the run with a mismatched frame or a wrong token).
- The bf16 control diverges at the same token as fp32 (49). For a pair this is exact, not luck: two bf16 numbers add exactly in fp32, so one rounding of the fp32 sum equals the bf16 add. The reduce dtype can matter only for groups of three or more.
- The three P1 runs across the Macs are identical to each other: the engine is deterministic.

Limitation. One run per cell except P1 across the Macs (3). Eight short prompts end after 3 to 6 tokens (EOS), so the short prompt check is weak; the long prompt is where differences show. One long prompt per model.
Hypothesis. The first divergence on the long prompt (49 to 127 for 7B) reflects where the sum order changes, not the group size. The data do not order it by group size (P4 with 5 ranks holds longer than P1 with 2).

## Q-E. Stages of TP groups

| cell | model | stages (ranks per stage) | hosts | identical | first step logit diff | long prompt first divergence | frames per step (formula = measured) | runs |
|---|---|---|---|---|---|---|---|---|
| P5 | 7B | 1, 1 (12 and 16 layers) | loopback | 8/8 | 0 | none (256 of 256 equal) | 2 | `P5_loopback_qwen2.5-7b-bf16_r1` |
| P5 | 7B | 1, 1 | M5, Air | 8/8 | 0.375 | 72 | 2 | `P5_2mac_qwen2.5-7b-bf16_r1` |
| P6 | 7B | 1, 3, 1 | loopback | 8/8 | 0.375 | 49 | 29 | `P6_loopback_qwen2.5-7b-bf16_r1` |
| P6 | 7B | 1, 3, 1 | M5; 2 M5 + 1 Air; Air | 8/8 | 0.375 | 49 | 29 | `P6_2mac_qwen2.5-7b-bf16_r1` |
| P8 | 0.5B | 8 stages of 2 (16 processes) | loopback | 8/8 | 0.89 | 60 | 115 | `P8_loopback_qwen2.5-0.5b-bf16_r1` |
| P8-wifi | 0.5B | each pair split M5 and Air | 8 + 8 | 8/8 (3 runs) | 0.93 | 10 (3 runs) | 115 | `P8_wifi_qwen2.5-0.5b-bf16_r1` to `_r3` |
| P8-local | 0.5B | pairs on one Mac, stages alternate | 8 + 8 | 8/8 | 0.74 | 10 | 115 | `P8_local_qwen2.5-0.5b-bf16_r1` |

Accuracy equals the reference in every row (1.0 for 7B, 0.875 for 0.5B).

Observed.
- Pure PP on one Mac is bitwise exact: first step logits equal, all 256 long prompt tokens equal (P5 loopback). The stage hop moves bf16 activations without loss.
- The same pure PP split across the Macs is not bitwise exact (P5, 0.375, divergence at 72). There is no in-layer split in P5. So the Air's own computation differs from the M5's.
- Ohad's shape (8 stages of TP pairs, 16 processes) runs on one Mac and across two, in both layouts, and meets the criterion. The tied embedding is loaded twice (stage 0 and stage 7, 272 MB each), counted in both.
- Every run that puts ranks on both Macs (P8-wifi, P8-local) diverges on the long prompt at token 10, as the CPU and MPS references of 0.5B do.

Limitation. The Air has torch 2.15.0 and the M5 torch 2.14.1; the GPUs differ (M3, M5). The runs do not separate the two causes.
Hypothesis. A different Metal kernel or torch version rounds differently. Testing it needs the same torch on both Macs.

## Q-F. Weak nodes: a budget smaller than one layer

Placement P7 (the A6 example of the prompt): s0 holds layers 0 to 11 and the embedding on the M5; s1 holds layers 12 and 13 as a TP group of three weak ranks (2 M5, 1 Air); s2 holds layers 14 to 27 on the Air; a head-only stage holds the output head as a TP group of three weak ranks (1 M5, 2 Air). Every weak rank has `budget_gb: 0.40`. A 7B layer is 0.466 GB and the head 1.09 GB, so no weak rank can hold a whole unit.

| run | outcome | weak ranks: tensors (weights + KV) | weak ranks: footprint | identical | frames |
|---|---|---|---|---|---|
| `P7_loopback_qwen2.5-7b-bf16_r1` | completes | 0.319 to 0.369 GB | 1.26 to 1.29 GB | 8/8, divergence 49 | 34 |
| `P7_2mac_qwen2.5-7b-bf16_r1` to `_r3` | completes, 3/3 | 0.319 to 0.369 GB | 1.28 to 1.37 GB | 8/8, divergence 49 (3 runs) | 34 |
| `P9_2mac_qwen2.5-7b-bf16_r1` (weak ranks on CPU) | completes | 0.319 to 0.369 GB | 0.16 to 0.18 GB (peak RSS 0.44 to 0.62 GB) | 8/8 against MPS and CPU references | 34 |
| `P7ctl_2mac_qwen2.5-7b-bf16_r1` (same nodes, pure PP) | planner rejects | s1a/0 and s1b/0: 0.590 GB planned; head/0: 1.106 GB | | | |
| `P7ctl_2mac_forced_qwen2.5-7b-bf16_r1` (planner check skipped) | fails at load | s1a/0, s1b/0 stop at 0.407 GB on `up_proj`; head/0 at the 1.09 GB head | | | |
| `P7_2mac_mpscap_qwen2.5-7b-bf16_r1` (P7 with torch's MPS cap at the budget) | fails at load | all six weak ranks: "MPS allocated 1.01 GiB, max allowed 381 MiB" | | | |

Observed.
- A node smaller than one layer contributes as a member of a group. The output is correct (8/8, accuracy 1.0, the same first divergence as P6 without budgets), on one Mac and across two, in 4 runs.
- The planner and the ranks agree. The planner names the three ranks that cannot hold a whole unit; the forced run fails on exactly those three, at the first tensor that crosses the budget.
- On MPS the budget holds for tensors but not for the process. Each weak rank holds one tensor of 10 MiB or more (a 45 MB MLP slice, a 363 MB vocab slice), so MPS gives it a 1 GiB heap, and its footprint is 1.26 to 1.37 GB. torch's own cap counts that heap and refuses the legal placement.
- On CPU there is no heap. The weak ranks' footprint is 0.16 to 0.18 GB, below the tensor bytes: the weights stay as file backed pages of the safetensors file (peak RSS 0.44 to 0.62 GB counts them).

Limitation. One budget value (0.40 GB). One model. The workspace is an estimate (the prompts are short). Peak memory inside a step was not sampled. The P7 loopback run swapped on the M5 (eight processes, 23.2 GB of footprint in all, on a 24 GB machine); it completed but slowly.
Hypothesis. With MPS weights split into tensors under 10 MiB (row chunks), a weak MPS rank would cost its share plus about 0.25 GB. Not built: it is a change to the hot path and outside this session.
Answer for the volunteer case. A laptop with a few GB free can join a group and hold a slice of a unit too large for it. On Apple GPUs through torch, its real entry cost is about 1.3 GB per process. As a CPU only volunteer it costs its share plus about 0.15 GB.

## Q-G. Accounting

Planned against loaded. On every rank of every completed run (167 rank instances in 30 runs), loaded tensor bytes equal the planner's weight bytes exactly (`loaded_tensor_bytes == planned.weights`; the unit test checks it too).

Frames. The formula (`plan.frames_per_step`, written in `PREDICTIONS.md` before the runs) equals the measured decode frames per step on all 24 completed placement runs: 2, 29, 34, 115, 116, 346, 1148. Bytes per step match the formula's estimate (for example P1: 819,932 measured against 0.820 MB).

Real memory per rank (MPS). Three numbers per rank in every record: `current_allocated` (tensors), `driver_allocated` (Metal heaps), and the process footprint.

| rank kind | tensors | driver allocated | footprint | example run |
|---|---|---|---|---|
| idle rank (before loading) | 0 | small | 196 to 230 MB | every run (`idle`) |
| P8 middle rank (largest tensor 4.4 MB) | 48.8 MB | 84.6 MB | 294 to 304 MB | `P8_wifi_qwen2.5-0.5b-bf16_r1` |
| P8 first and last rank (136 MB vocab slice) | 187 MB | 1091 MB | 1306 to 1449 MB | same |
| 1.5B TP 3 rank | 1.10 to 1.18 GB | 1.17 to 1.20 GB | 1.40 to 1.53 GB | `P3_2mac_qwen2.5-1.5b-bf16_r1` |
| 7B one rank | 15.72 GB | 16.22 GB | 16.56 GB | `R0_qwen2.5-7b-bf16_mps_r3` |

Observed. `driver_allocated` minus tensor bytes runs from 35 MB to 907 MB. It is not a constant. The probe before the runs explains it: on this torch build, MPS puts a tensor of 10 MiB or more into a heap of 1 GiB, and that heap is resident memory. Smaller tensors use heaps of 8 or 32 MiB.
Consequence. On MPS a rank that holds any tensor of 10 MiB or more costs at least about 1.3 GB of real memory, whatever its planned share. A rank whose tensors are all under 10 MiB costs about its share plus 0.25 GB.
torch's own cap (`set_per_process_memory_fraction`) counts heaps, so it refuses a second large tensor below a cap of about 1.07 GB. Budgets in this session are therefore enforced on tensor bytes (planner, loader, KV), and the footprint is reported next to them.

## The general rule

When is a group needed? When a unit (one layer, the embedding or the head) is larger than the node that should hold it. Then the unit is split across a group, and the node takes a share. Otherwise PP of whole units is enough, and cheaper.

What a group costs per decode step, in messages (exact on every run):

```
frames = (n_0 - 1) + n_0 (n_0 - 1)  +  sum over stages of 2 L_s n_s (n_s - 1)  +  sum over hops of n_next  +  (n_head - 1 or n_head)
         token        embedding        two all-reduces per layer                  stage hops                (value, index)
```

All-to-all in a group of n costs n (n - 1) frames per all-reduce. A pair costs 2 per all-reduce, 4 per layer. A group of 5 costs 40 per layer: 1148 frames per step for 7B in one group of 5, against 2 for pure PP.

What an unequal or non power of two group costs in memory: replicated KV heads. A rank boundary inside a KV group makes that KV head live on two ranks (weights and cache). Measured in the planner output and in the loaded bytes:

| group | KV head copies (model has) | extra weights per layer | extra KV at 4096 over the model |
|---|---|---|---|
| 7B, 17 and 11 heads | 5 (4) | 1.8 MB | 59 MB |
| 7B, 10, 9, 9 | 6 (4) | 3.7 MB | 117 MB |
| 7B, 6, 6, 6, 5, 5 | 8 (4) | 7.3 MB | 235 MB |
| 1.5B, 4, 4, 4 | 4 (2) | 1.6 MB | 117 MB |
| 7B, 14, 7, 7 (KV aligned) | 4 (4) | 0 | 0 |

The cost is small against a 466 MB layer (under 2 percent at 5 ranks), and a KV aligned head split removes it.

## What was not checked

- Speed. Not a metric (T1). ms per step is in the table as background only.
- Long prompts longer than 256 tokens, or prompts longer than 256 tokens of prefill (the workspace plan refuses them).
- More than one long prompt per model. Repeats beyond P1 across the Macs, P7 across the Macs and P8-wifi.
- Peak memory inside a step. Memory is sampled after the runs.
- The same torch version on both Macs.
- The NVIDIA node, the internet, MoE models, quantized weights.
