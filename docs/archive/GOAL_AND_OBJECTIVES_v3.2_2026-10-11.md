# Project Goal and Objectives

**Nocturne: Distributed LLM Inference over Volunteer Consumer GPUs**
B.Sc. Final Project, Electrical and Computer Engineering, Ben-Gurion University of the Negev

---

**Revision v3.2 (proposed, 2026-10-10)** — after the PoC-1 review, the Codex/Claude exchange (I001–I009 in docs/he/SHARED_RESEARCH_AND_DECISIONS.md) and **team decision T1**. Folds in the supervisor's notes: *multiple parallelism strategies, tracker can wait, liveness and continuity*. Rationale and work plan in RESEARCH_PLAN.md. Becomes "accepted" on supervisor confirmation.

> **Framing (team decision T1, 2026-10-10).** Latency is **not** the objective and not the interesting factor. The project demonstrates the **feasibility and efficiency of splitting a model across additional machines — and how many** — starting from the premise that the network, and later the internet, **is the bottleneck, as a given**. We do not set out to fix the network; we ask how to split so that the system is possible and efficient despite it. Latency is measured and reported as an *environmental parameter* of each run, never as a target.

## Project Goal

> To design, implement, and experimentally evaluate a **requester-coordinated, tracker-assisted peer-to-peer system** for running large-language-model inference across remote, heterogeneous, and intermittently available consumer machines — and to **demonstrate the feasibility and efficiency of splitting a model across a growing number of such machines**, taking the network as the given bottleneck: which splits are possible, how many machines pay off, at what efficiency, and how robustly — with the choice made automatically from measured resource capabilities.

The goal has two halves on purpose. The first half (*design, implement*) is engineering and yields a working system. The second half (*feasibility and efficiency of the split as machines are added*) is the scientific question and yields measurements that did not exist before. A final project that delivers only the first half is a demo; only the second half is a simulation. We commit to both.

### Research question

*Given volunteer machines with measured compute, memory, network characteristics (taken as slow) and availability history: which model splits are feasible, how does efficiency change as machines are added (1 → 2 → 3 → N), which split strategy and assignment use the machines most efficiently under a slow network — including the cost of recovering from departures — and can a planner choose it automatically from measurements alone?*

**Primary metrics — three separate axes** (I010, adopted; every table reports them next to pp/tg):

1. **Capacity** — does combining machines make another model runnable? Reported as success/failure *under a declared memory policy* (e.g. "entirely on the accelerators within their memory budgets" vs. "runnable at all with CPU offload"), the reason for failure, peak memory per machine, context/concurrency, and throughput. The same checkpoint, quantization and hash on every machine. If the model has no single-machine run, speed-up and any `E(N)` are **N/A**, not 0. A "capacity ratio" uses as denominator the largest model that actually ran on one machine in the same candidate series, not nominal RAM.
2. **Single-request performance** — tok/s and `T_best_single / T_plan` where a single-machine run exists. `E(N) = T(1)/(N·T(N))` may be reported as a *descriptive normalization only*: it is speed-up per machine, it does not separate compute from waiting (a perfect serial pipeline with zero network cost still has `E(2)=0.5`), and on heterogeneous machines `T(1)` must name the machine.
3. **Throughput under load** — total output tokens per second `Q` of completed requests over a shared window, requests/s, completion rate and completion-time spread at 1 / 4 / 8 concurrent requests, with fixed input/output length distributions — **compared against one independent replica per machine with requests routed between them**, same total resources and load. `Q_split / Q_independent` answers *when to split a model and when to use the extra machine for another user*; it is measured under load, never inferred from single-request tg.

Per-token latency, RTT, jitter and bandwidth are recorded as conditions of each run, not optimized.

> **Framing refinement (team decision T2, 2026-10-11).** The project's claim is **the ability to run a huge model — larger than any single machine — across several consumer machines over the internet, and succeed**: it runs, it finishes, it produces a high-quality answer. Network time and traffic are not of interest now. The "task before sleep, answer in the morning" scenario is **one illustration** of why speed does not decide — not a hard deadline and not the only use case. Consequences for the axes above: axis 1 (capacity) is primary and gains two dimensions — **quality** (the distributed large model vs. the small local one on a fixed task set) and **distance** (the same run when the machines are not on one LAN; Tailscale between homes stands in for "the internet"); axis 2 (single-request speed) becomes background data ("~8 h" is an order of magnitude for illustration, not a constraint); axis 3 is secondary. **Overnight robustness becomes a primary objective** (a machine sleeps, Wi-Fi drops; completion rate under injected interruptions, work lost per interruption), and availability history (H2) returns to the planner's core. **Context memory for long reasoning outputs** (tens of thousands of tokens of KV) enters the memory policy. Ethernet points, byte/round-trip traces and split-vs-replica throughput remain background data, not new experiments. "Network share of per-token time" is reported only when a trace supports it; local-vs-remote differences alone do not isolate the network from requester, staging, protocol and scheduling.

### Hypotheses (to be confirmed or refuted by measurement)

- **H1.** Under a slow network, the per-token cost of a split is dominated by its *network interactions* — the number of stage boundaries, the number of request-response waits per boundary, and the **bytes per boundary** (an output layer placed remotely ships `n_vocab × 4` ≈ 608 KB of logits per token for Qwen2.5; a hidden-state boundary ships ~7 KB) — not by compute; hence the **efficient** split minimizes boundaries and keeps the output layer on the requester, even when this leaves faster machines partially idle. (Prefill, weight loading and KV transfer are bandwidth-bound and modelled separately. Which of RTT-waits and bytes dominates on a given link is measured by trace, not assumed.)
- **H2.** Placing layers preferentially on nodes with higher historical availability reduces *expected* time-to-answer more than placing them on nodes with the most free memory, once departure probability during a typical session exceeds a few percent. *(Secondary since v3.)*
- **H3.** Worker-side weight caching (seeder behaviour) reduces session start-up time by more than an order of magnitude on a second run, making short sessions viable on volunteer hardware.
- **H4.** *(demoted to a single characterization by T1)* Under the project's premise of a slow network, tensor parallelism — two synchronizations per layer per token — is expected to be **inefficient** (`E(2)` well below pipeline parallelism's) on every link we have. We measure it **once** per link type to document it and to satisfy the requirement that the planner can *analyse and choose* between strategies; we do not search for an RTT region in which TP wins, since that would be latency optimization. Outcome reported either way.
- **H6.** *(new, T1; hypothesis, not yet measured)* Adding machines under a slow network yields **memory aggregation** (a model that fits on no single machine, under the declared memory policy, runs on N) and may yield **throughput scaling under concurrent requests**, but **no single-request speed-up**. For a model that fits on each machine, independent replicas are expected to beat a split on `Q`; the smallest N that fits the model is expected to be best for a single request but is **not** guaranteed best for multi-request throughput — both are measured, not assumed (overlap of pipeline stages in llama.cpp depends on model and backend conditions, `llama-context.cpp:427–458`).
- **H5.** *(new)* A planner that uses only measured node profiles predicts per-token cost within 15 % and achieves regret `(T_selected − T_best_measured) / T_best_measured ≤ 10 %` against the best plan *within the declared, measured plan space*, on configurations not used for calibration. Thresholds are frozen after a pilot and before the main comparison.

### Non-goals (explicitly out of scope)

Training or fine-tuning; incentive/payment mechanisms; tracker-less (DHT) discovery; cryptographic verification of remote computation; production-grade security beyond a private mesh network; custom GPU kernels; mixture-of-experts and multimodal models (dense decoder-only transformers only).

### "Support every model and the standard packages" — what we commit to

- **Model class:** dense decoder-only transformers (Llama, Qwen, Gemma, Mistral, Phi families). A shared planner and worker protocol, plus a **declared, tested adapter per family** (loading, legal sharding plan incl. GQA/fused-projection constraints, correctness vs a local reference). Families differ in math (e.g. Gemma-2 soft-capping), so "no family-specific code" is not promised. Demonstrated incrementally: one family to correctness, then two more.
- **Model format:** a manifest (hash, layers, hidden size, head counts, bytes/layer, split constraints) consumed by the planner; Hugging Face safetensors and/or GGUF depending on ADR-007. A manifest alone covers new *weights* of a supported architecture; a new *architecture* needs an adapter.
- **Standard packages (proposal, separate axis):** the requester exposes an OpenAI-compatible HTTP API with a declared subset (chat completions, streaming; no tools/vision initially), so the `openai` SDK, LangChain, Open WebUI work unchanged within that subset.

---

## Objectives

Each objective is small, has an owner-agnostic deliverable, a measurable acceptance criterion, and maps to a development stage in ARCHITECTURE.md §6.

### O1 — Controlled single-device execution (PoC-0)
**Deliverable:** Reproducible build of llama.cpp with the Metal backend on Apple Silicon and the CUDA backend on the NVIDIA machine, plus scripts that force inference onto an explicitly chosen device.
**Acceptance:** For each machine, a results table with prefill time (ms) and decode throughput (tok/s) for the same model and prompt on (a) GPU device, (b) CPU only. Difference between runs on the same config < 5 %.
**Why it matters:** Establishes the single-node baseline every distributed number is compared against, and proves we control *where* computation runs.

### O2 — Backend characterization: Metal vs CUDA (PoC-0)
**Deliverable:** A short technical note (docs/BACKENDS.md) explaining what level of control each API gives (device, not core), how llama.cpp dispatches work on each, and a decision on where DOCA/RDMA does and does not fit.
**Acceptance:** Note reviewed by supervisor; one paragraph of it is reusable verbatim in the thesis background chapter.

### O3 — Two-node manual pipeline (PoC-1)
**Deliverable:** Requester on one machine driving a `ggml-rpc-server` worker on a second machine over the home LAN, with hand-set `--tensor-split`.
**Acceptance:** Same model/prompt run at splits 100/0, 50/50, 0/100 (and, for a mixed Mac + CUDA pair, both directions). Table of tok/s and first-token latency. Measured single-hop latency tax reported in ms/token.
**Status (2026-10-10):** one successful `llama-cli` run logged (split `1,1`, remote worker, 7B Q4: 5.9 t/s decode) against a local `llama-bench` figure of 20 t/s — **not a controlled comparison** (different tool and workload; offload per device and network path unrecorded). Table B is still empty. First work item: R0, a fair baseline with the same tool and tokens, per-device offload logged, and network conditions recorded on separate axes (LAN/WAN, Ethernet/Wi-Fi, plain/Tailscale, direct/relay, measured RTT/BW).

### O4 — Feasibility and efficiency of the split as machines are added (PoC-1b) — *new; reframed by T1*
**Deliverable:** Gated steps R0–R3 (RESEARCH_PLAN.md §6) on the network we have (home Wi-Fi first; one Ethernet point only to separate network share from protocol share): (a) **capacity** — a candidate that may fit on no single machine *under the declared memory policy* (first candidate: Qwen2.5-32B-Instruct Q4_K_M, ~19.9 GB file; the Macs have 16 GB / 24 GB unified memory with Metal working-set recommendations of 12.1 / 18.2 GB, which are soft limits — so **the same model is first tried locally on each machine**, and only a documented local failure makes the 2-machine run a capacity result; 14B as a calibration step), run across 2, then 3 machines with memory-proportional and planner-chosen splits, with placement verified from the measured run; (b) **single-request and load** — tok/s, `T_best_single/T_plan` where defined, and `Q` at 1 / 4 / 8 concurrent requests for N = 1, 2, 3 with pipeline parallelism, **versus independent replicas** for a model that fits on each machine; (c) **messages per token** of the substrate (`GGML_RPC_DEBUG=1` count) as the efficiency property of the transport; (d) **one** TP characterization per link type to document that it is inefficient under the premise (H4) and to show the planner can choose. Related-work note (docs/RELATED_WORK.md) positioning against HexGen, Helix, Petals, exo, prima.cpp, TPI-LLM, distributed-llama.
**Acceptance:** Three tables for N = 1, 2, 3 — capacity (success/failure per declared policy, reason, peak memory per machine, throughput), single-request (tok/s, `T_best_single/T_plan` or N/A), and load (`Q`, req/s, completion rate, spread at 1/4/8 requests, `Q_split/Q_independent` where the model fits on each machine) — each row with ≥ 10 repetitions after warm-up, placement evidence from the measured run, the link recorded as a condition; the substrate's per-token command mix (types, bytes per direction, response waits — separated for load / warm-up / decode) from a trace; TP characterized once per link type with repetitions and a correctness check, outcome open. Hybrid only if a third machine makes it meaningful.

### O5 — Engine decision, resource profiler, cost model, planner v1 (PoC-2) — *new*
**Deliverable:** (a) ADR-007 filled in by the **symmetric acceptance rule**: identical mandatory requirements for both engines (correctness on the chosen model/dtype, real use of both accelerators incl. Mac, memory fit, controllable and measurable placement, defined timeout, system-level replay path; TP capability pending team confirmation); failure of one engine is never evidence for the other. Worker interface frozen. (b) A worker self-profile: memory (shared-memory accounting on Apple Silicon), per-layer compute at relevant context/batch, transfer/collective cost to every peer for decode- and prefill-size messages. (c) A planner over a **declared, bounded plan space** (single device included; stage = node subset × TP degree ≤ 2 × layer range; asymmetric shards where the adapter allows; memory incl. KV, activations, buffers, margin), with an island heuristic checked against counter-plans, a cost model with separate decode / prefill / load / KV terms, and reported planning time. Static `peers.toml`; no tracker yet.
**Acceptance:** On ≥ 2 nodes (3 when available), held-out configurations: predicted vs measured per-token cost and regret vs the best measured plan (the objective being efficiency under the given network, not latency per se); H5 reported with the actual distributions, including cases where the planner lost to a simple baseline.

### O6 — Liveness and continuity (PoC-2) — *new*
**Deliverable:** Heartbeat-based membership with configurable timeout; on departure the requester re-plans and recovers the in-flight generation by **full-prefix replay** (prompt + tokens already delivered, with a confirmed-token boundary so nothing is emitted twice; position IDs, cache position, mask and sampling state verified). This v1 is engine-agnostic — a wrapper restarting a llama.cpp process is a valid implementation. Recovering only a lost stage's KV requires the activations that fed it (i.e. recomputation in earlier stages); partial KV retention on surviving stages is a later measurement, conditional on the transport (static process groups require re-init). A defined failure is returned when no legal capacity remains.
**Acceptance:** Failure-detection latency, recovery time and completion rate at context lengths 512 / 2 k / 8 k, ≥ 10 trials each, with failures injected during load, prefill and decode.

### O7 — Model manifest, tracker, and `nocturne run` end-to-end (PoC-3) — *deferred from PoC-2*
**Deliverable:** (a) Manifest format (hash, layers, bytes/layer, hidden size, head counts). (b) Tracker with `announce` / `peers` endpoints storing the worker profile from O5 and last-seen time. (c) Worker agent announcing every N seconds. (d) `nocturne run --model <manifest> --prompt <text>` pulling peers from the tracker and invoking the planner.
**Acceptance:** Three workers announce; peer list reflects a killed worker within 2 × N seconds; zero manual host/port entry; tracker runs on a Raspberry-Pi-class machine.

### O8 — Availability-aware planning (PoC-3) — *formerly the core O6, now one term of the planner*
**Deliverable:** Availability term in the planner's cost: expected recovery cost × departure probability derived from observed uptime.
**Acceptance:** Controlled experiment on ≥ 3 nodes with injected departures; planner with vs without the term compared on mean and p95 time-to-answer over ≥ 30 sessions each. Result reported whether or not H2 holds.

### O9 — Beyond the LAN (PoC-4)
**Deliverable:** The same system running over a Tailscale mesh with at least one node outside the home network.
**Acceptance:** O3/O4-style capacity and efficiency tables repeated over WAN — the premise "the internet is the bottleneck" made concrete: what is still feasible, at what `E(N)`, and whether the planner keeps WAN nodes in separate, large pipeline stages.

### O10 — Documentation, reproducibility, and standard client interface (continuous)
**Deliverable:** Repository with build scripts, experiments log (experiments/RESULTS.md), decisions log (docs/DECISIONS.md), related-work and learning-resources notes, and — if the team adopts it — an OpenAI-compatible `/v1/chat/completions` endpoint with a declared supported subset.
**Acceptance:** A third party can reproduce O1, O3 and O4 from the README alone on their own two machines; within the declared subset, an unmodified `openai` SDK client completes a streamed chat against Nocturne.

### Stretch — RDMA transport (lab-dependent)
If a lab machine with a ConnectX/RoCEv2 NIC is available, repeat O3 with llama.cpp's RDMA transport and report the efficiency difference vs TCP (a data point on how much of the per-token cost is transport, not an optimization goal). This is the only place the NVIDIA networking stack (DOCA / libibverbs) enters the project.

---

## Mapping objectives to the academic year

| Period | Objectives | Milestone |
|---|---|---|
| 12–25 Oct 2026 | O1, O3 (finish), related-work note | Tables A–C filled; RELATED_WORK.md |
| 26 Oct – 15 Nov 2026 | O4 (R0–R4) + Track A | TP-advantage region mapped or its absence recorded; ADR-007 via symmetric rule; Worker interface frozen |
| 16 Nov – 20 Dec 2026 | O5 | Profiler + planner v1; H5 prediction-error table |
| 21 Dec 2026 – 31 Jan 2027 | O6, then O7 | Recovery-time table; `nocturne run` end-to-end with tracker |
| Feb – Apr 2027 | O8, O9 | 3+ nodes incl. WAN; H1–H5 answered |
| May – Jun 2027 | O10, thesis writing | Report + defense |
