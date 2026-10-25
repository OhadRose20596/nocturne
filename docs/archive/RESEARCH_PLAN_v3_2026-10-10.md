# Research Plan after the PoC-1 Review

**Status:** Draft v3 — 2026-10-10. Written after the supervisor meeting on PoC-1; revised the same day after Codex's review (I003 in `docs/he/SHARED_RESEARCH_AND_DECISIONS.md`). Proposes answers to the five points raised, and a work plan to settle them with measurements rather than opinion. Everything here is proposed, not decided.

The five points: (1) is PyTorch the right library; (2) support every model and the standard packages; (3) how to cut the model — pipeline vs tensor parallelism — with a resource-aware decision; (4) sharpen goal and objectives; (5) what C++ gives that Python does not, and whether to build our own system from scratch.

> **Framing — team decision T1 (2026-10-10), applies to everything below.** Latency is **not** the objective and not the interesting factor. The project demonstrates the **feasibility and efficiency of splitting a model across additional machines, and how many**, from the premise that the network — and later the internet — **is the bottleneck, as a given**. We do not set out to fix the network. Wherever this document reasons about per-token latency, read it as reasoning about *where the time goes* so that the split can be made efficient; the metrics are three separate axes (I010): **capacity** under a declared memory policy (speed-up N/A when no single-machine run exists), **single-request performance** (`T_best_single/T_plan`; `E(N)` only as a descriptive normalization — it does not separate compute from waiting), and **throughput under load** `Q` at 1/4/8 requests **versus independent replicas on the same machines**. RTT, jitter and bandwidth are recorded as conditions of each run; a "network share" is claimed only from a trace. Sections 3.1–3.2 and H4 were written before T1 and are kept as the cost reasoning behind the planner; their emphasis on RTT thresholds is now secondary (see §4.4).

---

## 0. Where we actually stand

Before deciding anything, what the repository actually proves (checked 2026-10-10):

- `experiments/bench.csv` has **Table A rows for the Mac only** (Apple M3, Qwen2.5-7B Q4_K_M, commit 1e7bcf3): Metal pp256 = 208 t/s, tg64 = 20.0 t/s; CPU pp256 = 50 t/s, tg64 = 14.7 t/s. No CUDA-machine rows.
- `experiments/logs/20260915-133754_…_rpc_1,1.log` is **one successful PoC-1 run**: requester on the Mac, one remote worker, split `1,1`, 7B Q4, `llama-cli` with a short text prompt: prompt 13.6 t/s, generation **5.9 t/s**. The local figures above come from `llama-bench` with a different workload (`pp256`/`tg64`), so **the two numbers are not a controlled comparison and the gap must not be reported as "the cost of one hop"** until R0 reproduces both with the same tool, prompt tokens and lengths. The log shows neither per-device layer offload nor the worker's identity, so a true 50/50 split is unverified. An *earlier, failed* attempt that day tried `100.106.120.60:50052` — a CGNAT-range address consistent with Tailscale, which can still be a direct LAN path; the successful log records no endpoint. Network conditions of that run are unknown.
- `RESULTS.md` Tables A–C are still empty; the three-split sweep, both directions, the CUDA baseline, and any network characterization are all missing.

So one real data point exists and it is a strong signal that something costs a lot, not a measurement of what. Every "latency tax" number below is still from the literature. **R0 (a fair baseline: same tool, same tokens, per-device offload logged, network path recorded on separate axes — LAN/WAN, Ethernet/Wi-Fi, plain/Tailscale, direct/relay, measured RTT and bandwidth) is week-1 work regardless of which engine we pick.**

---

## 1. PyTorch: what it is, what it gives us, what it does not

"PyTorch" is three different things, and the answer differs for each.

| Layer | What it is | Verdict for Nocturne |
|---|---|---|
| **Tensor compute library** (ATen, CUDA / MPS backends) | Matmul, attention, etc. on CUDA and on Apple Silicon (MPS) | **Usable.** Kernels are C++/CUDA/Metal underneath; Python only dispatches. MPS dtype support, memory fit and decode latency must be measured on the installed OS/PyTorch versions. |
| **Model ecosystem** (Hugging Face `transformers`, safetensors) | A broad model ecosystem with architecture-specific implementations, module layouts and cache semantics | **A major benefit.** Reuse existing model implementations behind the declared adapters in §2; universal model coverage is not assumed. |
| **Distributed toolkit** (`torch.distributed`, NCCL/Gloo, DTensor, `torch.distributed.pipelining`, TorchElastic) | Collectives and sharding for *static* process groups; elastic restarts of the whole group | **Not our runtime, but parts are candidate components.** NCCL is CUDA-only and Gloo/NCCL do not operate on MPS tensors (HF Accelerate docs) — yet since activations are copied to the host before any send anyway, **Gloo on CPU tensors is a legitimate transport candidate** and is compared against our own TCP in R1. For v1 we plan a bounded failure path followed by worker/group restart and full-prefix replay. In-process reinitialization and experimental reconfiguration are version/backend-dependent and require separate validation; rebuilding a communicator is not itself a statement about whether application KV state survives. DTensor has experimental uneven sharding but no hardware-aware asymmetric planner. Heterogeneous placement and generation state stay ours. |

So the honest framing: **PyTorch as the compute-and-model library, possibly supplying the transport, under our own distributed runtime — yes. PyTorch as the distributed runtime — no.** The runtime (placement, worker protocol, membership policy, failure handling, generation state) is ours in either case; that is the thesis.

The competing substrate is the one we already use: llama.cpp / ggml. Same split: excellent compute library (Metal + CUDA + Vulkan + CPU, 4-bit quantization), model coverage through supported GGUF architectures, and an experimental distributed layer (RPC). Control includes device, split mode and split proportions; finer state/placement control must be tested against the pinned build.

What each substrate costs us is in §5. Which one to pick is decided by a two-week spike, not here (§6).

---

## 2. "Support every model and the standard packages"

"Every model" is not a requirement; it is a trap. We propose to define *support* in three concrete tiers and demonstrate each with evidence:

1. **Model class.** Dense decoder-only transformers (Llama, Qwen, Gemma, Mistral, Phi families). A "layer range" and a "tensor shard" are well-defined here, but the families are **not** interchangeable in code or math: Gemma-2 adds logit soft-capping and extra norms; GQA/MQA head counts constrain legal TP splits; some checkpoints fuse QKV projections. MoE (expert routing) and multimodal (vision encoders) are **explicitly out of scope** for the degree project and listed as future work. Acceptance: a shared planner and worker protocol, plus a **declared, tested adapter per family** (loading, legal sharding plan, correctness against a local reference); three families demonstrated incrementally after correctness is proven on one. "Zero family-specific code anywhere" is *not* promised.
2. **Model format.** On the PyTorch path: Hugging Face safetensors. On the llama.cpp path: GGUF (`convert_hf_to_gguf.py`). The planner consumes a **manifest** (hash, layer count, hidden size, bytes per layer, head counts, legal split constraints). A manifest alone suffices only for new *weights* of an architecture that already has an adapter; a new *architecture* needs an adapter.
3. **Standard client interface.** The requester exposes an OpenAI-compatible endpoint with a **declared supported subset** (chat completions, streaming; no tools or vision initially), so common clients — the `openai` SDK, LangChain, Open WebUI — work without modification within that subset. This is a product-side proposal for what "standard packages" could mean (*clients*, not *internals*); it is a separate axis from library/format compatibility and remains for the team to confirm.

Small 1.5–3 B models serve plumbing and correctness experiments. They do not establish the same compute/communication balance or crossover as larger models. Include a larger checkpoint and a configuration that exceeds one device's capacity where hardware permits; record the actual memory condition rather than assuming 7–8 B cannot fit. Memory arithmetic alone does not establish large-model latency.

---

## 3. How to cut the model: pipeline vs tensor parallelism, decided per deployment

### 3.1 The two cuts and a first cost model (single-request decode, no overlap)

Notation: `L` layers, hidden size `h` (one activation vector ≈ `2h` bytes in bf16, ≈ 8 KB for a 7 B model), `S` pipeline stages, `k` tensor-parallel peers.

**Pipeline parallelism (PP) — cut *along* the model.** Each node owns a contiguous layer range. Per token the activation vector crosses each stage boundary once, then the last stage returns logits/token.

```
T_PP(token) ≈ Σ_s stage_compute_s + Σ_b boundary_transfer_b + output_handling
```

- `boundary_transfer` is **measured** for the message size, dtype, copy path (device→host→wire→device) and route actually used — not assumed to be one RTT.
- For decode the message is tiny, so latency dominates. For **prefill** (`seq_len × 2h` per boundary — ~29 MB for a 4 k-token prompt on a 7 B model), **weight loading** (GB) and **KV transfer** (hundreds of MB) bandwidth dominates. These are four different traffic patterns and get four separate terms; "bandwidth is irrelevant" is only true for the first.
- With a single request, **only one stage computes at any instant**: PP aggregates memory, it does not speed up decode. Throughput scales with concurrent requests (micro-batching).
- Tolerant of slow links; this is what llama.cpp RPC, Petals, exo, prima.cpp and EdgeShard do.

**Tensor parallelism (TP) — cut *across* the model.** Each layer is sharded over `k` nodes (attention heads, MLP columns). The common dense-transformer pattern needs two all-reduces per layer per token; the exact count depends on the sharding plan, where embeddings/output live, and overlap.

```
T_TP(token) ≈ Σ_l [ max_i shard_compute_{l,i} + Σ_j collective_time_{l,j} ] + output_handling
```

- `collective_time` is **measured** per message size and participant count; it includes host staging, reduction and scheduling, not just one RTT.
- An *illustration* of why the link matters, with its assumptions stated: if a 28-layer model needs 56 serial collectives per token and each costs 0.3 ms (wired LAN), that is ~17 ms/token of overhead; at 3 ms each (typical home Wi-Fi) ~170 ms; at 20 ms (WAN) > 1 s. Whether the compute saving (which is **not** guaranteed to be `k×` — unequal shards, replicated parts, small kernels, head-divisibility constraints) outweighs this is exactly what R1–R3 measure.
- All `k` nodes compute simultaneously, so TP *can* speed up single-request decode on fast links: TPI-LLM and distributed-llama report this on wired consumer LANs (distributed-llama's figures show gains to 4 nodes on gigabit Ethernet and network time dominating at 8).
- Heterogeneity hurts twice: the slowest shard paces every layer, and standard TP wants equal shards. HexGen's answer is asymmetric shards; ours should be too.

### 3.2 What the arithmetic suggests (and what it does not)

The arithmetic suggests TP pays only where links are fast — the same wired LAN or the same machine — and that PP is the natural choice across slower links. It does **not** prove that Wi-Fi rules out TP, that PP is the only option "between islands", or that a hybrid must beat both pure strategies; those depend on the model, the workload and the measured collective cost. The right question remains the supervisor's: **given the measured capabilities of the available resources, which combination, and where?** — with *single device* always a legal candidate plan.

**Under T1 the practical reading is simpler.** If the network is slow by premise, the efficient split is the one with the fewest network interactions per unit of work: few, large pipeline stages, placed so that the strongest machine carries the most layers, and a transport that spends one message per boundary per token rather than several. Tensor parallelism, with two synchronizations per layer per token, is expected to be inefficient on any link we have; it is characterized once per link to document it and to show the planner can choose, not searched for a sweet spot. Adding machines buys **capacity** (models that fit nowhere alone) and **throughput under concurrent requests** (micro-batching fills the pipeline bubbles), not single-request speed — and `E(N)` quantifies the price.

### 3.3 The decision procedure (what the planner does)

1. **Profile.** Each worker measures itself: free memory (on Apple Silicon: shared, minus system use), per-layer compute time for the target model at the relevant context lengths and batch, and transfer/collective cost to every peer for the decode- and prefill-size messages. This is the "capability analysis of each available resource" and becomes the worker's announcement payload.
2. **Reduce the search space** with an island heuristic: group nodes whose measured pairwise transfer cost is below a threshold. The threshold is **calibrated**, not assumed, and the planner always evaluates a few counter-plans the heuristic would have pruned, so the heuristic is checked rather than trusted.
3. **Enumerate candidate plans** in a **declared, bounded space**: ordered stages; each stage = a node subset with TP degree `k ∈ {1, 2}` initially (asymmetric shards allowed where the adapter's split constraints permit) and a layer range; *single device* included. Memory per node includes weights, KV cache for the target context, activations and buffers plus a safety margin.
4. **Score each plan** with the §3.1 model, plus an availability term (expected recovery cost × departure probability from uptime history — the original O6 idea).
5. **Pick the minimum**, launch, **validate**: predicted vs measured per-token latency and `regret = (T_selected − T_best_measured) / T_best_measured` on configurations not used for calibration, with "best" meaning best *within the measured plan space*. Planning time is reported too.

For a handful of nodes with `k ≤ 2` the declared space is small enough to enumerate; if it grows (subsets × stage order × layer boundaries × TP ratios explode quickly), dynamic programming over (layer index × group) is the standard trick from Alpa/HexGen. Start bounded and explicit.

### 3.4 Prior work we must position against

| System | Cut | Heterogeneous HW | Churn / volunteers | Mac | Auto-decides PP vs TP |
|---|---|---|---|---|---|
| llama.cpp RPC | PP (layer); `--split-mode tensor` is new and experimental, RPC support unverified | yes | no (process dies) | yes | no |
| Petals | PP | partially | yes (DHT, rebalancing) | no | no |
| exo | PP (ring) | yes | partial | yes (MLX) | no |
| prima.cpp (Halda) | PP (piped-ring) + offload | yes, optimizer-driven | no | yes | no |
| TPI-LLM | TP (star all-reduce) | yes (uneven shards) | no | CPU | no |
| distributed-llama | TP over Ethernet | limited (2ⁿ nodes) | no | yes | no |
| HexGen / HexGen-2 | asymmetric PP + TP | yes, constrained optimization | no (datacenter) | no | **yes** |
| Helix | PP via max-flow | yes | no | no | partly |
| **Nocturne (proposed)** | PP + TP hierarchical | yes | **yes, availability-aware** | **yes** | **yes** |

The gap we fill is the last row: HexGen's planning idea moved from datacenters to volunteer consumer hardware, where availability and Apple Silicon are first-class and where a measurement-driven planner must work with noisy home networks. A `docs/RELATED_WORK.md` with this matrix expanded (one paragraph per system, with citations) is a week-2 deliverable.

---

## 4. Goal and objectives — what changed and what we propose

### 4.1 Reading the edits in `GOAL_AND_OBJECTIVES.md`

Two edits were made after the meeting. The goal sentence gained "a P2P system" and the phrase **"requester-coordinated, tracker-assisted"** (inserted mid-sentence; it needs rewriting). Under O3 three notes were added: *"multiple parallelism strategy", "tracker can wait", "liveness and continuity".* We read them as follows, and ask the supervisor to confirm:

- **Parallelism strategy before anything else.** The core question moved from "which node gets which layers" to "which *strategy* (PP/TP/hybrid) and which assignment, chosen from measured resources." Our original O6 (availability-aware placement) is now one term in that planner's cost function, not the whole contribution.
- **Tracker can wait.** Discovery is plumbing. A static `peers.toml` is enough until the planner and continuity work exist. The tracker moves from Nov–Dec to Jan.
- **Liveness and continuity.** Detect loss through bounded heartbeats/timeouts. V1 uses full-prefix replay under either engine, retaining confirmed token IDs and avoiding duplicate output. Partial KV recovery needs compatible retained state and historical activations or recomputation; warm standby is a later extension. Neither engine has demonstrated this recovery path in our experiments yet.

### 4.2 Proposed goal (clean rewrite)

> To design, implement, and experimentally evaluate a requester-coordinated, tracker-assisted peer-to-peer system that runs large-language-model inference across remote, heterogeneous, and intermittently available consumer GPUs — and to characterize how the choice of parallelization strategy (pipeline, tensor, or hybrid) and of layer/shard placement affects end-to-end latency and robustness, so that the choice can be made automatically from measured resource capabilities.

### 4.3 Proposed research question

> Given volunteer nodes with measured compute, memory, pairwise latency/bandwidth, and availability history, which parallelization strategy and assignment minimizes expected time-to-answer (including the cost of recovering from departures), and can a planner choose it automatically from measurements alone?

### 4.4 Hypotheses

- **H1** (reframed by T1) — under a slow network the per-token cost of a split is dominated by the number of network interactions (boundaries × messages per boundary), not bandwidth; the *efficient* split minimizes boundaries — fewer, larger stages — and a substrate that needs several serial messages per token lowers `E(N)` on any link. (Prefill, loading and KV transfer are bandwidth-bound and modelled separately.)
- **H2** (keep, secondary) — once departure probability per session exceeds a few percent, availability-weighted placement beats memory-proportional placement on *expected* time-to-answer.
- **H3** (keep, minor) — worker-side weight caching cuts second-session start-up by > 10×.
- **H4** (demoted to one characterization by T1) — under the premise of a slow network, TP (two synchronizations per layer per token) is expected to be inefficient on every link we have; measured **once per link type** to document it and to show the planner can analyse and choose; no search for an RTT region where TP wins. Reported either way.
- **H6** (new, T1) — adding machines under a slow network yields memory aggregation (a model that fits on no single machine runs on N) and throughput scaling under concurrent requests, but no single-request speed-up; `E(N)` for single-request decode falls with N and the efficient operating point is the smallest N that fits the model, largest stages on the strongest machines.
- **H5** (new) — a planner using only measured profiles predicts per-token cost within 15 % and achieves `regret ≤ 10 %` against the best plan *in the measured plan space*, on configurations not used for calibration. Numeric thresholds are frozen after a pilot and before the main comparison, never adjusted afterwards.

### 4.5 Objectives — proposed re-ordering

| # | Objective | Stage | Change vs. current file |
|---|---|---|---|
| O1 | Single-device control + baseline tables | PoC-0 | unchanged — **still unfinished** |
| O2 | Backend note (Metal/CUDA/DOCA) | PoC-0 | done |
| O3 | Two-node manual pipeline, latency tax | PoC-1 | unchanged — **Table B still empty** |
| **O4 (new)** | **Parallelism characterization**: single device vs PP vs TP (hybrid if resources allow) on 2–3 nodes with injected delay/bandwidth (`tc netem` / dummynet, axis = *measured* RTT); the TP-advantage region mapped, or its absence reported | PoC-1b | replaces old O4 |
| **O5 (new)** | **Engine decision** by symmetric acceptance (ADR-007, §6) + resource profiler + cost model + planner v1 (bounded enumeration) + prediction-error and regret validation | PoC-2 | replaces old O5 |
| **O6 (new)** | **Liveness and continuity**: heartbeat/failure detection, re-plan, **full-prefix replay** as v1 (engine-agnostic); partial KV retention measured later only if feasible | PoC-2 | new |
| O7 | Tracker + `nocturne run` end-to-end (zero manual host entry) | PoC-3 | old O4+O5, deferred |
| O8 | Availability-aware planner v2 (old core O6) | PoC-3 | demoted to "if time permits — but planned" |
| O9 | Beyond the LAN (Tailscale) | PoC-4 | old O7 |
| O10 | Documentation, reproducibility, OpenAI-compatible API | continuous | old O8 + API |

A corresponding edit of `GOAL_AND_OBJECTIVES.md` is proposed in the same commit as this file; the supervisor's confirmation turns "proposed" into "accepted".

---

## 5. C++ vs Python, and "build it all ourselves"

Split the system into the two planes it actually has.

**Control plane** — tracker, worker agent, profiler, planner, requester CLI, HTTP API. Never on the per-token path. **Python, no debate.** Iteration speed is the whole point and nothing here is performance-sensitive.

**Data plane** — per-token compute and the transport between nodes. The comparison is between two *engines*, not two *languages*: ggml's kernels and ATen's kernels are different implementations, GGUF 4-bit vs bf16 is a quantization choice, and the device→host copy path exists in both. Rows marked *measure* are filled by R0–R3, not asserted.

| Criterion | ggml engine (llama.cpp / ggml as a library) | PyTorch engine (ATen + HF transformers) |
|---|---|---|
| Kernel implementation | ggml Metal/CUDA kernels | ATen Metal(MPS)/CUDA kernels — *measure* per-layer time on both machines |
| Dispatch overhead per token | C++ graph execution | Python dispatch; *measure* with `torch.profiler`, separating interpreter/dispatch from copy and wait; `torch.compile`/CUDA graphs reduce it |
| Memory footprint at target config | GGUF 4-bit available | bf16 default; MPS quantization immature — *measure* peak RAM/VRAM for the same model and context |
| Mac GPU | Metal, mature | MPS, adequate for bf16 inference; transport via host staging (own TCP or Gloo-on-CPU) |
| Model coverage | GGUF-converted families; new architectures lag | Hugging Face checkpoints; per-family adapters still needed for TP plans |
| Implementing TP / hybrid / continuity | inside a large C++ codebase; TP exists experimentally (meta backend), RPC support unknown | own code on top of a tensor library; faster to iterate |
| Deployment on a volunteer's machine | single binary (driver/library dependencies remain) — *measure* install time, package size | Python + torch + CUDA environment — *measure* install time, package size (`uv`/conda-pack) |
| Team velocity (2 students, 8 months) | lower for new runtime features | higher |

**"We can build a completely new system and not rely on anything existing."** Agreed — for the *distributed runtime*. Transport, worker protocol, planner, liveness, continuity: all ours, and that is where the novelty is. Not agreed — for the *compute engine*. A transformer inference engine with Metal and CUDA kernels, quantized formats and model loaders is a thesis by itself, and ADR-003 already says no custom kernels. Reusing a kernel library (PyTorch or ggml) beneath our runtime is the same relationship BitTorrent clients have with TCP.

What makes this safe: a narrow **Worker interface** (`load_shard(manifest, layers, tp_rank)`, `forward(activations) → activations`, `collective_hook`, `replay(prefix_tokens)`) so the engine is swappable by an ADR, not a rewrite. Both engines go through the same acceptance (§6); a later move of the data plane to C++ (ggml or libtorch) is justified only by a profile showing the component in question is actually reducible by the move.

---

## 6. Work plan

Academic year calendar; two people; today is 10 Oct 2026.

### Phase A — Finish the baseline and read (12–25 Oct)

| Task | Deliverable | Done when |
|---|---|---|
| **R0 — fair baseline.** Same tool for every row (`llama-bench` locally and with `--rpc`, or `llama-server` + one client with identical token IDs), same model/dtype/lengths/cache settings; `load_tensors` log per device; run `05_bench.sh` on the CUDA machine too | RESULTS.md Tables A and B | GPU/CPU rows for both machines; three splits × two directions; per-device offload verified; < 5 % run-to-run variance |
| **Network characterization on separate axes:** location (LAN/WAN), link (Ethernet/Wi-Fi), overlay (plain/Tailscale), path (direct/relay from `tailscale status`), endpoints, RTT and iperf3 at run time | Table C | one row per configuration actually used in Table B |
| Related-work survey (HexGen, Helix, Petals, exo, prima.cpp, TPI-LLM, distributed-llama, EdgeShard, Galaxy, Alpa) | `docs/RELATED_WORK.md` with the §3.4 matrix expanded | each system: cut, heterogeneity, churn, Mac, planner; one paragraph; citation |

### Phase B — Gated engine evaluation and the TP-advantage map (26 Oct – 15 Nov)

Two tracks in parallel, one per person, same two machines, same model and prompts. **Gates:** R1 and R2 must pass before any full-model work, planner or extra model family; a failure stops expansion and the blocker is analysed first.

| Step | Time box | What is tested | Output that allows moving on |
|---|---|---|---|
| **R1 — transport path** | 3–4 days | Device→host→wire/reduce→device for three message sizes: `2h` (decode), `512×2h` and `4096×2h` (prefill). Three paths: own TCP, CPU-staging + Gloo (if the build supports it), `iperf3` as reference. For each: breakdown (copy-to-host / serialize / wire / reduce / copy-to-device), correctness of the sum, and behaviour when one side disappears (time-to-error, whether the group is reusable) | full-path time per size and path; the recovery path each transport permits |
| **R2 — one TP block** | 4–5 days | One decoder block of Qwen2.5-1.5B split 2-way across Mac (MPS) + NVIDIA (CUDA), with attention/KV and a GQA-aware head split; compared to the unsplit block in bf16 and against an fp32 reference, at context 1 / 128 / 2048. Tolerance fixed **before** the run from reference-vs-reference spread between the two machines; max and relative error reported; never widened afterwards | block-level correctness and the cost of its collectives vs its compute at baseline RTT |
| **R3 — feasibility, efficiency, scaling** | ~1 week | PP on the network we have (Wi-Fi first; one Ethernet point only to separate network share from protocol share). (a) Capacity: candidate Qwen2.5-32B-Instruct Q4_K_M (~19.9 GB) — **first tried locally on each Mac under a declared memory policy** (Metal's 12.1 / 18.2 GB working-set figures are soft; the machines have 16 / 24 GB), then across both, then 3 machines; 14B as calibration; placement from the measured run; a smoke-test correctness check where no reference exists is reported as such; (b) single-request tok/s and `T_best_single/T_plan`, and `Q` at 1/4/8 concurrent requests for N = 1, 2, 3 **versus independent replicas** for a model that fits on each machine; (c) substrate command trace per token (types, bytes per direction, response waits; load/warm-up/decode separated — `GGML_RPC_DEBUG=1` plus requester timing; `GRAPH_RECOMPUTE` reuse rate; logits bytes when the output layer is remote); (d) one time-boxed TP characterization per link type with repetitions and correctness, outcome open (H4). ≥ 10 repetitions after warm-up; link recorded as a condition | capacity, single-request and load tables per N; command trace summary; TP documented; a 7–8 B and a 32 B point |
| **R4 — failure and replay** | ~4 days | Kill a worker during decode; requester re-plans and replays the full prefix; verify no duplicated output tokens and a defined failure when no capacity remains | completion rate, recovery time at several context lengths |
| **Track A — llama.cpp, in parallel** | 1–2 days | Pinned source/build identity; local source exposes `tensor`, but the copied CLI currently fails its loader preflight (RESULTS.md, 2026-10-10). After repairing deployment/build and verifying `--version`/`--help`, use `--split-mode tensor --tensor-split 1,1 -fit off --rpc <worker>`, Qwen2.5-7B Q4_K_M; verify sharding really happens (`load_tensors` per device, memory observed on both sides); multi-token generation; `llama-server` prompt-cache path; worker kill mid-run | one row in the capability matrix **for this exact combination**, with the same measurement meaning as R3/R4 |

**Mandatory requirements, identical for both engines** (ADR-007): correctness on the chosen model and dtype; real use of both accelerators including the Mac; memory fitting the target configuration; measurable and controllable placement; a defined timeout; a system-level replay path. *Proposed for team confirmation:* ability to run a **legal** TP configuration (not necessarily a faster one), because the supervisor's point 3 asks for PP-vs-TP capability analysis and an engine that cannot run TP can only document a limitation.

**Symmetric decision rule** (adopted from I003):

| Acceptance outcome | Action |
|---|---|
| Only A meets the mandatory requirements | choose A, or record an explicit scope change for any requirement not tested |
| Only B meets them | choose B |
| Both meet them | compare measurements plus the pre-declared development, deployment and control criteria |
| Neither | fix a bounded blocker or narrow scope; no automatic winner |
| Evidence missing | decision stays open; the missing test is named |

Two decisions are kept apart: *does the engine enable a correct, controllable experiment* (this rule) and *which strategy pays off* (H4, a research result either way). Failure of one engine is never evidence for the other.

**Deliverables:** R0–R4 and Track A tables; **ADR-007 filled in** from the rule above; the Worker interface frozen; `docs/RELATED_WORK.md`.

### Phase C — Profiler, cost model, planner v1 (16 Nov – 20 Dec)

Worker self-profile (memory incl. shared-memory accounting on Apple Silicon, per-layer compute at relevant context/batch, transfer/collective cost matrix for decode- and prefill-size messages) → island heuristic with counter-plans → bounded, declared plan enumeration (single device included; `k ≤ 2`) with memory + KV constraints → cost model with separate decode / prefill / load / KV terms → launch. Validation: predicted vs measured per-token latency and regret on held-out configurations (H5); planning time reported. Static `peers.toml`; no tracker yet.

### Phase D — Liveness and continuity (21 Dec – 31 Jan)

Heartbeats, failure-detection latency, re-plan on departure, **full-prefix replay** as the v1 recovery (engine-agnostic: a wrapper restarting a llama.cpp process with prompt + generated tokens is a valid implementation), measured for recovery time and completion rate at several context lengths, including the no-capacity case. Partial KV retention is a later experiment that must account separately for communicator recovery, worker-process lifetime, changed sharding and application state. Then the tracker and `nocturne run` end-to-end (O7), since by now the announcement payload (the profile) is known.

### Phase E — Scale-out and the full experiment (Feb – Apr)

3+ nodes with friends' machines; Tailscale (O9); availability-aware term in the planner (O8) and the injected-departure experiment from the original O6; all hypotheses answered, each reported whether or not it holds.

### Phase F — Writing (May – Jun)

---

## 7. Questions for the supervisor (to close at the next meeting)

1. **"multiple parallelism strategy"** — did we read this correctly as "the system must support more than one strategy and choose between them"?
2. **PyTorch as a compute library under our own runtime** (not as the distributed runtime) — acceptable? Or is there a preference for a C++ data plane from day one despite the velocity cost?
3. **Model scale for experiments** — 1.5–3 B models in bf16 for all sweeps, one 7–8 B demonstration run, scaling argued by memory arithmetic. Sufficient?
4. **Scope of "every model"** — dense decoder-only transformers in; MoE and multimodal out. Agreed?
5. **Apple Silicon as a first-class worker** — we treat it as a hard requirement (it is the heterogeneity story). Confirm the exact target machines; direct MPS collectives and CPU-staged Gloo must be distinguished in ADR-007.
6. **Is H4 (the RTT crossover) a result he considers worth a chapter?** It is cheap to obtain and, as far as we found, not reported for consumer hardware with a Mac in the mix.

---

## Sources consulted for this plan

- llama.cpp RPC README and multi-GPU docs; feature request "tensor parallelism over RPC" (row mode is a no-op over RPC); `--split-mode tensor` meta-backend PR #19378 and follow-ups (#23792, #24152), issue #22307 — https://github.com/ggml-org/llama.cpp
- Hugging Face Accelerate, MPS guide: Gloo and NCCL do not work with the MPS device — https://huggingface.co/docs/accelerate/main/en/usage_guides/mps
- HexGen (ICML 2024), asymmetric TP+PP over heterogeneous GPUs — https://arxiv.org/abs/2311.11514
- TPI-LLM, tensor parallelism on low-resource edge devices, star all-reduce — https://arxiv.org/abs/2410.00531
- prima.cpp / Halda scheduler for heterogeneous home clusters — https://arxiv.org/abs/2504.08791
- distributed-llama, TP over Ethernet on consumer devices — https://github.com/b4rtaz/distributed-llama
- Petals — https://github.com/bigscience-workshop/petals ; exo — https://github.com/exo-explore/exo
