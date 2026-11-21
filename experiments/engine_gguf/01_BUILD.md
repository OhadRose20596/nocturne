# Engine GGUF, part 1 of 2: build and verify (M5 only)

Work prompt for Claude Code. Written 2026-11-21 by Claude (Cowork) with Ohad. Run it on the M5 inside the `nocturne` repo.
The Air and gtx are powered off during this part. Everything here runs on the M5 alone. Multi-machine runs belong to part 2 (`02_EXPERIMENTS.md`). Do not try to reach the other hosts.

Goal: when this part ends, the code is written, tested against independent references, reviewed by someone who did not write it, and pushed. Part 2 should only have to run it.

## 0. Why

The torch engine (engine B, `nocturne/engines/torch_tp`) reads only Qwen2 bf16 safetensors. llama.cpp ran 32B Q8_0 on three machines in session 8c and our engine cannot open that file. PP, TP and hybrid comparisons between the engines are fair only on the same file. Bigger and newer models (Chen, 16.11, points 5 and 6) need this too. Ohad decided on 2026-11-21 to do this before session 9 (timeline).

The join tool (`scripts/15_join.sh`, `15_join.ps1`, `16_join_apply.py`, `docs/JOIN.md`) gets three small additions here, because part 2 and session 10 (wired, Wi-Fi, internet) depend on them.

## 1. Read first

1. `CLAUDE.md`.
2. `nocturne/engines/torch_tp/`: `config.py`, `loader.py`, `shard.py`, `model.py`, `plan.py`, `hybrid.py`, `hybrid_run.py`, `README.md`. Tests in `tests/bench/test_torch_tp.py`, `test_hybrid.py`.
3. `experiments/RESULTS.md`: "Session 6", "Session 8b", "Session 8c". `docs/CURRENT_CONTEXT.md` (the open Q8 proposal).
4. The GGUF format and the `gguf` package of llama.cpp (`gguf-py`: `GGUFReader`, `gguf.quants`, `GGMLQuantizationType`). Prefer the version that ships with llama.cpp 1e7bcf3 (`~/dev/llama.cpp/gguf-py`). Record which.
5. `convert_hf_to_gguf.py` at 1e7bcf3: the tensor name map, and which model classes permute Q and K at conversion. Read the code. Do not rely on memory.
6. `docs/JOIN.md` and the three join scripts.

Rules: short sentences, no em dash, no en dash in Markdown. No Hebrew in code or command blocks. T1: time is background, never a gate. Never bind `0.0.0.0`. Do not install or upgrade torch. The only new dependency allowed is `gguf`.

## 2. Design constraints

1. One engine. GGUF is a second weight source next to safetensors. Rank, transport, frame format, planner and hybrid code stay shared. No change to the transport or the frame format.
2. Weights stay quantized on the device. `QuantLinear` keeps the raw blocks and expands to bf16 just before its matmul, one matrix at a time. The transient bf16 buffer counts in the planner's workspace.
3. Row split layers (o_proj, down_proj: input columns) may be cut only at block boundaries: 32 for Q8_0, Q4_0, Q5_0, 256 for K quants. Column split layers (q, k, v, gate, up) can be cut at any row. The planner enforces the rule and names it in its rejection message.
4. Byte accounting counts the stored blocks, plus the largest transient bf16 matrix per rank, plus the MPS rule from session 6 (a tensor of 10 MiB or more lands in a 1 GiB heap).
5. Numerics as in session 3: RMSNorm in fp32, rotary in fp32 cast to the activation dtype, sdpa, reduce dtype float32.
6. Tokenizer and EOS ids from the GGUF (`AutoTokenizer.from_pretrained(dir, gguf_file=...)` or the metadata). Record which.
7. The loader takes a tensor source interface (local file today), so a later session can stream blocks from the coordinator (requirement: the worker must not hold the model).

## 3. Build

B1. `gguf_source.py`: open a GGUF, metadata to config, name map (`blk.N.attn_q.weight` and so on), quant type per tensor, refuse an unknown type with a clear message.
B2. `quant.py`: `QuantLinear` for F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K. Torch ops only, vectorized, on any device. No custom kernels.
B3. Architecture table replacing the Qwen2-only checks: `qwen2` (q, k, v bias), `llama` (no bias, Llama 3 rope scaling, undo the converter's Q and K permutation), `qwen3` (no qkv bias, q_norm and k_norm per head). One table, not one class per model. The safetensors path uses the same table.
B4. Placement YAML: `model: <key>` resolves to a safetensors dir or a `.gguf` file through one registry file. Model hashes for both.
B5. Planner: quantized bytes, transient buffer, block alignment.
B6. Join tool additions:
   - llama.cpp check: `llama-server` and `ggml-rpc-server` exist, `--version` commit, backend (Metal, CUDA, CPU). Fact lines and one check line. No build, only report.
   - Models: list `~/models/hf/*` and `~/models/*.gguf` (and `~/models/gguf/*.gguf` if present). `apply` prints the `scp` commands for missing GGUF files too, with the sha256 to check after the copy.
   - Networks: detect the link type of the LAN interface (macOS `networksetup -listallhardwareports`, Linux `/sys/class/net/<if>/wireless`, Windows adapter media type). Store `networks: {wifi: ..., wired: ...}` by type, never overwrite one type with another. Keep `bind` as today.
   - `16_join_apply.py`: `lan_ip()` works on Linux too.
   - Not in scope: liveness, auth, tracker.

## 4. Verify (all on the M5)

V1. Unit tests, each against an implementation we did not write:
   - Dequant of every quant type against `gguf.quants.dequantize` (numpy) on random blocks and on real tensors from the GGUF files under `~/models`. Q8_0, Q4_0, Q5_0 exact after the cast to float32. K quants: max abs diff 0 or explain why not.
   - Same tensors against transformers' own GGUF loader (`from_pretrained(..., gguf_file=...)` dequantizes with its own code). A third, independent reading.
   - Name map: every tensor of each GGUF under `~/models` maps to exactly one engine name, none left over.
   - Q and K permutation: for a llama GGUF made locally from safetensors with `convert_hf_to_gguf.py`, the engine's Q and K after load equal the safetensors Q and K exactly (after the type cast).
   - Block alignment: the planner rejects a row split that is not aligned, with the rule in the message.
   - Join tool: unit tests for `update_peers` with two network types, for report parsing with the new fact lines, and a dry run of `apply` on a copy of the real files.
V2. Mutation check: break the code on purpose in a scratch branch (swap the two halves of a Q4_K scale, drop the Q8_0 scale, skip the Q and K un-permute, off by one in the block alignment). Each break must make at least one test fail. Record which test caught which break in `experiments/engine_gguf/VERIFY.md`. Throw the branch away.
V3. End to end, one machine. Pre-register expected outcomes in `PREDICTIONS.md` first.
   - bf16 safetensors regression: the session 6 loopback placements on 1.5B and 7B give the same tokens as their recorded runs. Nothing old may change.
   - Q8_0 GGUF of 1.5B and 7B (make them with `convert_hf_to_gguf.py --outtype q8_0`, record command and sha256), one rank on MPS: tokens against the bf16 reference (`R0_*_mps`) and against llama.cpp on the same GGUF on the M5 (`llama-cli` greedy or the bench BASELINE cell). Report identical count and first divergence.
   - 7B Q4_K_M (sha256 65b8fcd9, the A1 file) one rank against llama.cpp on the same file.
   - Loopback on the M5, Q8_0 7B: PP three stages, TP three ranks, and one hybrid placement. Criterion: at least 7 of 8 identical against the one rank run of the same file. Frames per step as planned. Memory per rank within 5% of the plan.
   - Perplexity sanity: 128 tokens of wikitext-2 test at a fixed offset, engine against `llama-perplexity` on the same file, for 7B Q8_0 and Q4_K_M. A bug shows as a large gap. Report both.
   - Architectures: one small llama model (Llama 3.2 1B or 3B Instruct) and one qwen3 model (Qwen3 0.6B to 4B), safetensors and a locally made Q8_0 GGUF, one rank against transformers `generate`. Download size and license first. Ask Ohad before any download over 10 GB.
   - Join tool on the M5: `bash scripts/15_join.sh --check --self` passes, and the report shows the new llama.cpp, models and network lines.
V4. Independent review. Two reviewers who did not write the code:
   - A Claude Code subagent with a fresh context. Give it the diff (`git diff` from the start SHA), this prompt's section 2, the GGUF spec and `gguf-py`. Do not give it your notes or conclusions. Ask it to find errors in: dequant math, name and permute map, block alignment, planner bytes, tokenizer and EOS, and anything that changes old behavior. It writes `experiments/engine_gguf/REVIEW.md` with each finding rated high, medium or low and a way to check it.
   - Codex (Ohad runs it). Write `experiments/engine_gguf/REVIEW_REQUEST_CODEX.md`: the files to read, the same questions, and how to run the tests. Ohad pastes it to Codex and puts the answer in `REVIEW_CODEX.md`. Do not wait for it. Part 2 starts only after Ohad has it.
   - Fix every high finding. For each finding write in `REVIEW.md` what you did or why it is not a bug. Then a second fresh subagent pass on the fixes only.

Gate of part 1: all tests pass on the M5, every mutation is caught, V3 rows meet their criteria or have a diagnosed reason, no open high finding from the subagent review.

## 5. Records and push

- `experiments/engine_gguf/VERIFY.md`: one table per V step: what, reference, result, file.
- `experiments/RESULTS.md`: section "Engine GGUF, part 1 (M5 only)", same columns as session 8b, plus quant type and file sha256.
- `nocturne/engines/torch_tp/README.md`: weight sources, quant types, alignment rule, architecture table.
- `docs/JOIN.md`: the new checks and fields.
- `docs/CURRENT_CONTEXT.md`: close the open Q8 proposal in two lines.
- One entry in `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` (private, never commit).
- Small commits, push. The Air and gtx pull at the start of part 2.

Final message under 25 lines: gate result, the V3 numbers in one line each, the review findings count by rating and what is still open, and what Ohad must do before part 2 (Codex review, power on the machines).

## 6. Do not

Reach the Air or gtx. Bind `0.0.0.0`. Install or upgrade torch. Add a dependency other than `gguf`. Write CUDA or Metal kernels. Change the transport or the frame format. Edit old run folders. Commit model files. Run 32B or larger in this part (it needs the three machines).
