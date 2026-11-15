# Session 6, hybrid placement: pre-registered predictions

Written 2026-11-15 before any engine code and before any run. Setup: M5 (24 GB, Metal line 18.2 GB) and Air (16 GB, Metal line 12.1 GB), home Wi-Fi, torch 2.14.1 (MPS), engine `nocturne/engines/torch_tp`.
The Ohad column stays empty: Ohad waived it on 2026-11-15 ("irrelevant"). The same day he approved the four open points: the 32 token identity rule below, a per rank `heads:` override, the 0.5B download, and no new ADR for the placement file (it is internal to engine B). The outcome column is filled after the runs, failures included.
The Claude column starts from the numbers in the session prompt (`PROMPT.md` section 4). Where I disagree, the cell says "Revised" and gives the reason.
Time per token is not predicted (T1).

## Model facts, checked against the files

Source: `config.json` and the safetensors headers in `~/models/hf/` on the M5. 0.5B is not on the M5 as safetensors yet. Its shape comes from the GGUF header of `qwen2.5-0.5b-instruct-q4_k_m.gguf` (same model). It is checked again after the download.

| fact | prompt | checked | note |
|---|---|---|---|
| 7B shape | hidden 3584, MLP 18944, 28 heads, 4 KV heads, head dim 128, 28 layers, vocab 152064, not tied | same | |
| 7B bytes, bf16 | layer 466 MB, embedding and head 1090 MB each, model 15.2 GB | layer 466,115,584 B, embedding and head 1,089,994,752 B each, model 15.231 GB | exact |
| 1.5B shape | hidden 1536, MLP 8960, 12 heads, 2 KV heads, 28 layers, tied | same; head dim 128; vocab 151936 | vocab differs from 7B |
| 1.5B bytes | layer 94 MB, tied embedding 467 MB | layer 93,595,648 B, embedding 466,747,392 B, model 3.087 GB | |
| 0.5B shape | hidden 896, MLP 4864, 14 heads, 2 KV heads, head dim 64, 24 layers, tied | same (GGUF); vocab 151936 | the GGUF stores an `output.weight`; tying is checked in `config.json` after the download |
| 0.5B bytes | layer about 30 MB, tied embedding 272 MB | layer 29,824,768 B, embedding 272,269,312 B, model about 0.99 GB | computed from shapes |
| KV per token per layer | 7B 2 KiB, 1.5B 1 KiB, 0.5B 0.5 KiB | same | per KV head: 512 B (7B, 1.5B), 256 B (0.5B) |

Unit sizes the planner works in (bf16, one layer):

| model | one query head (q rows, q bias, o columns) | one KV head (k and v rows, biases) | one MLP row (gate, up, down) | granule of 128 MLP rows |
|---|---|---|---|---|
| 7B | 1,835,264 B | 1,835,520 B | 21,504 B | 2.75 MB (148 granules) |
| 1.5B | 393,344 B | 393,472 B | 9,216 B | 1.18 MB (70 granules) |
| 0.5B | 229,504 B | 229,632 B | 5,376 B | 0.69 MB (38 granules) |

The MLP is 87 percent of a 7B layer. So the planner balances mostly with MLP granules, and heads matter for the KV cache and for KV head replication.

## KV head replication, worked out

A rank holds a contiguous range of query heads. It needs every KV head that one of its query heads uses (GQA: KV head = query head // (heads / KV heads)). A rank boundary that falls inside a KV group makes that KV head live on two ranks.

| split | KV group size | KV heads per rank | KV head copies (model has) |
|---|---|---|---|
| 7B, 17 and 11 heads (P1) | 7 | 3, 2 | 5 (4) |
| 7B, 14, 7, 7 (P2) | 7 | 2, 1, 1 | 4 (4) |
| 7B, 10, 9, 9 (planner default for 3 equal ranks; P7 s1) | 7 | 2, 2, 2 | 6 (4) |
| 1.5B, 4, 4, 4 (P3) | 6 | 1, 2, 1 | 4 (2) |
| 7B, 6, 6, 6, 5, 5 (P4) | 7 | 1, 2, 2, 2, 1 | 8 (4) |
| 0.5B, 7 and 7 (P8 pairs) | 7 | 1, 1 | 2 (2) |

## Frames per decode step: the formula

Notation. Stages s = 0..S-1 in order. n_s ranks in stage s, L_s layers in it. n_h ranks in the group that holds the output head. The coordinator is the process that owns the tokenizer and HTTP.

```
frames = token + embed + layers + hops + head
token  = n_0 - [coordinator is a rank of stage 0]        # next token id to the first stage
embed  = n_0 (n_0 - 1)  if the embedding is vocab split  # all-to-all all-reduce of the embedded row
layers = sum over s of  L_s * 2 * n_s (n_s - 1)          # two all-reduces per layer
hops   = sum over s < S-1 of  n_(s+1)                    # leader sends the hidden state to every rank of the next stage
head   = n_h - [coordinator is a rank of the head group] # (value, index) per rank
```

Bytes per step: every hidden state frame carries `2 * hidden` bytes (bf16) plus a 24 byte header. Token and (value, index) frames are under 50 bytes each.
So `bytes ~= (embed + layers + hops) * (2 * hidden + 24)`.
The formula counts every frame sent by every process. The old two rank engine reported 56 all-reduce frames per step: one direction only (rank 0's sends). Both directions plus the STEP frame make 113, and its 0.81 MB per step already counted both directions.
Prediction: the counters match this formula exactly on every run (frames), and within 1 percent for bytes. Confidence 90 percent. The 10 percent is a frame I forgot (for example a reset or a metrics frame inside the decode window).

## Correctness criterion, and one problem with it

The prompt says "greedy tokens identical to the one rank reference for the first 32 tokens on at least 7 of 8 prompts (`--subset 8`)".
Problem. `--subset 8` takes the first 8 rows of `eval_set.jsonl`: `arith_01` to `arith_06`, `extract_01`, `extract_02`. All 8 have `max_tokens` 16. So no prompt reaches 32 tokens.
Proposal (needs Ohad's choice). For the identity check, generate up to 32 tokens on every prompt (stop at EOS), ignoring the per prompt cap. Grade accuracy with the per prompt cap, as the bench does. Both come from the same run.
The reference is the one rank torch_tp engine on the same device (R0). R0 itself is compared once with transformers `generate`, as in B1t.

## Predictions per placement

Bytes are planned weight bytes per rank (bf16, decimal MB and GB). KV is at max_seq_len 4096. Runtime overhead is outside the budget and is predicted separately below.

| id | Ohad | Claude | confidence | outcome |
|---|---|---|---|---|
| R0 | | One rank references (MPS) for 0.5B, 1.5B and 7B are identical to transformers `generate` on 8 of 8 prompts. The CPU references match the MPS references on at least 7 of 8. | 85% | |
| P1 | | 7B TP 2 at 3:2. Heads 17 and 11, MLP rows 11392 and 7552 (89 and 59 granules). Per layer 281.7 and 186.3 MB. Revised: these numbers include one replicated KV head (head 2), because 17 is not a multiple of 7. The prompt did not say so. Whole rank with the vocab split 3:2 (91264 and 60800 rows of the embedding and the head): about 9.2 and 6.1 GB. Meets the criterion. 116 frames per step (1 + 2 + 112 + 0 + 1), 0.82 MB per step. That equals the old two rank engine's 0.81 MB: the STEP frame of the hidden state becomes a token frame plus an embedding all-reduce. | 90% | |
| P1 long prompt | | On the `eval_long` prompt with 256 tokens, the fp32 reduce run diverges from the reference at the same token as the bf16 control or later. Revised: confidence lowered. Greedy divergence is chaotic. One prompt can go the wrong way even when fp32 is closer on average. Session 4 measured a first divergence median of 70.5 for bf16 over 10 prompts. | 70% (was 90%) | |
| P2 | | 7B TP 3, KV aligned (heads 14, 7, 7). Revised: the prompt's MLP rows (4608, 7168, 7168) give 129, 169 and 169 MB. That is a 40 MB spread (26 percent of the mean), so they cannot also be within 5 percent of each other. A planner that balances total bytes gives MLP rows 5888, 6528 and 6528 (46, 51, 51 granules). Per layer that is 156.0, 155.1 and 155.1 MB. The rank with more heads gets fewer MLP rows. No KV head is replicated. Heads 14, 7, 7 do not come out of the planner by themselves: with equal weights it picks 10, 9, 9. So a KV aligned split needs a per rank `heads:` override in the placement file (proposal). Meets the criterion. 346 frames per step, 2.5 MB per step. | 90% | |
| P2 planner default | | Same group without the override (planner only, no extra run). Heads 10, 9, 9, KV heads 2, 2, 2 (two replicated), MLP rows 6272, 6400 and 6272. Per layer 156.9, 157.8 and 155.1 MB. The replication costs 3.7 MB per layer and 117 MB of extra KV at 4096 over the whole model. | 95% | |
| P3 | | 1.5B TP 3 (4, 4, 4 heads). Revised: both KV heads are replicated, not one. Rank 1 holds KV heads 0 and 1, so there are 4 copies instead of 2. Per layer 32.3, 31.9 and 31.1 MB (MLP rows 3072, 2944, 2944). Meets the criterion. 346 frames per step, 1.06 MB per step. | 85% | |
| P4 | | 7B TP 5 (6, 6, 6, 5, 5 heads). Revised: every KV head is held by two ranks (8 copies instead of 4), not two replicated heads. Any split of 28 heads into 5 near equal parts puts all four boundaries inside KV groups. Per layer 93.6 to 95.4 MB (the prompt said 91 to 97). Meets the criterion. 1148 frames per step, 8.2 MB per step. Risk: on loopback all five ranks run on the M5. That is 15.3 GB of weights plus 5 times the runtime overhead in 24 GB of RAM. I give it 70 percent to run without swapping. If it does not fit, it runs as 3 ranks on the M5 and 2 on the Air, and the report says so. | 85% (output), 70% (fits on loopback) | |
| P5 | | 7B PP only, 2 stages of 1 rank (12 and 16 layers). Weights 6.68 GB (with the embedding) and 8.55 GB (with the head). Meets the criterion. 2 frames per step (hidden out, (value, index) back), 7.2 kB per step. On loopback the output is bitwise identical to R0: the max logit difference at the first step is exactly 0. | 95% (criterion), 80% (exactly 0 on loopback) | |
| P6 | | 7B hybrid without budgets. Defined as the A6 layout with no budgets and the head in s2 (Air, one rank): s0 [0, 12) on the M5 with the embedding, s1 [12, 14) TP 3 (2 on the M5, 1 on the Air), s2 [14, 28) on the Air with the head. Meets the criterion. 29 frames per step (0 + 0 + 24 + 4 + 1), 0.20 MB per step. | 85% | |
| P7 | | 7B weak nodes, the A6 YAML. s1 (2 layers, heads 10, 9, 9, two KV heads replicated): weights 0.314, 0.316 and 0.310 GB, plus 8.4 MB KV each. Head group: 0.363 GB per rank (50688 vocab rows each). The budget is 0.40 GB. No single layer (0.466) and no whole head (1.090) fits it. s0 holds 6.68 GB plus 0.10 GB KV. s2 holds 6.53 GB plus 0.12 GB KV. Completes and meets the criterion. 34 frames per step, 0.22 MB per step. | 75% | |
| P7-ctl | | Same nodes, pure PP. Revised: the planner rejects the placement and names two kinds of rank, not one. It names an s1 rank (0.466 GB layer plus KV against 0.40) and the head rank (1.09 GB against 0.40). The forced run with the MPS cap fails at load, with out of memory on the first over budget rank to cross the cap. | 90% | |
| P8 | | 0.5B, 8 stages of 3 layers, each a TP pair, 16 rank processes on loopback. Heads 7 and 7 per pair, no KV replication. Per rank: middle stages 44.7 MB of weights, s0 and s7 ranks 181 MB (plus half the vocab each: 76032 and 75904 rows). The tied embedding is loaded twice, in s0 and in s7: 272 MB counted twice, 28 percent on top of the 0.99 GB model. KV 3.1 MB per rank. Meets the criterion. 115 frames per step (1 + 2 + 96 + 14 + 2), 0.20 MB per step. Fits in memory without shrinking to 4 stages. | 85% (criterion), 80% (fits) | |
| P8-wifi | | Each pair split across the two Macs (8 processes per host): completes. | 70% | |
| P8-local | | Each pair on one Mac, stages alternating between the Macs: completes. 7 hops cross Wi-Fi. All-reduces stay on loopback. | 75% | |
| P9 (optional) | | P7 with the weak ranks on CPU. Completes. It meets the criterion against the MPS reference on at least 7 of 8 prompts, and against the CPU reference on at least 7 of 8. | 70% | |

## Q-G, accounting

| item | Ohad | Claude | confidence | outcome |
|---|---|---|---|---|
| planned weight bytes against loaded tensor bytes | | Equal on every rank. Both come from the same tensor shapes, so the difference is 0 bytes, not just under 1 percent. | 95% | |
| MPS `driver_allocated` against planned bytes | | Planned bytes plus a constant per process (100 to 300 MB), not proportional. For the small P8 ranks (45 MB of weights) the constant is larger than the weights. | 60% | |
| runtime overhead of an idle rank process | | MPS: 0.3 to 0.6 GB resident (Python, torch, the Metal context). CPU: 0.2 to 0.4 GB. 16 processes (P8) need about 2.5 to 5 GB per host in overhead. | 60% | |
| MPS cap probe (A7) | | `torch.mps.set_per_process_memory_fraction` makes an allocation above the cap raise in torch 2.14.1. | 75% | |

## Falsifiers, stated up front

- A placement fails the criterion while one rank passes on the same device. That is a bug or a numerics problem in the split. It is reported first, not tuned away.
- Different ranks of one group hold activations that are not bitwise identical after an all-reduce. That is a bug in the reduce order.
- A counted frame number differs from the formula. Either the formula or the engine is wrong. The report says which.
- The vocab parallel embedding or argmax changes any token. Both are exact by construction (a sum with zeros, a max with ties to the lowest index), so any change is a bug.

## Not predicted

- Time per step (T1). It is recorded as background only.
- Peak memory during prefill. The prompts are short (eval_set prompts are 44 to 194 characters, eval_long about 250), so the prefill workspace is small. A budget run with a 4000 token prompt would need chunked prefill. That is not in scope.
- Whether the M3 and M5 GPUs give bitwise identical matmul results for the same op. P5 across the Macs shows it.
