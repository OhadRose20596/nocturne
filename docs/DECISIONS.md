# Decisions Log (lightweight ADRs)

One entry per decision that would be expensive to reverse. Newest last. Format: context → decision → consequences.

## ADR-001 — Requester coordinates; central component is a tracker only
**Date:** 2026-09-05 · **Status:** accepted
**Context:** Previous design had a central Coordinator computing placement and launching runs. Supervisor: "it should work like BitTorrent — the one who requests is the coordinator."
**Decision:** The requester (machine with the prompt) discovers peers via a tracker, computes placement locally, and drives workers directly with llama.cpp `--rpc`. The tracker stores only announcements (capabilities + last-seen). It never touches inference data or placement.
**Consequences:** (+) Tracker is stateless, tiny, and its failure does not stop running jobs. (+) Matches llama.cpp's Main-Host model exactly. (+) Placement algorithm becomes a pure function, trivially testable and swappable. (−) No global optimization across concurrent requesters; two requesters may pick the same worker. Accepted for the project scope; noted as future work.

## ADR-002 — llama.cpp RPC as the compute substrate
**Date:** 2026-08 (reaffirmed 2026-09-05) · **Status:** accepted
**Context:** Alternatives: Petals (PyTorch, Linux/CUDA-centric, DHT), exo (MLX-first, Apple-centric), custom torch.distributed.
**Decision:** llama.cpp with `-DGGML_RPC=ON`. Workers run `ggml-rpc-server`; requester runs `llama-server`/`llama-cli`.
**Consequences:** (+) Metal + CUDA + Vulkan + CPU from one code base; our two machines are heterogeneous from day one. (+) Placement output is just `--rpc` + `--tensor-split`. (−) RPC is officially proof-of-concept; no mid-run rebalancing. We treat restart cost as a measured quantity, not a bug.

## ADR-003 — Operate at device + layer-range level; no custom kernels
**Date:** 2026-09-05 · **Status:** accepted
**Context:** Supervisor asked how to "choose the GPU core" and whether to use DOCA or Metal.
**Decision:** We select *devices* (`--device`, `CUDA_VISIBLE_DEVICES`) and assign *layer ranges*. Kernel dispatch is left to ggml's Metal/CUDA backends. Core-level scheduling is not exposed by either API. See BACKENDS.md.
**Consequences:** (+) Keeps the project on the research question (placement), not on kernel engineering. (−) We cannot exploit intra-GPU parallelism beyond what ggml does. Acceptable.

## ADR-004 — DOCA out of scope; RDMA is a lab-conditional stretch
**Date:** 2026-09-05 · **Status:** accepted
**Context:** DOCA is the SDK for BlueField DPUs (network/storage offload), not a GPU compute API. llama.cpp RPC supports RDMA via libibverbs on RoCEv2 NICs.
**Decision:** Primary target is TCP on consumer networks. If lab hardware with ConnectX NICs becomes available, add one RDMA-vs-TCP experiment.
**Consequences:** (+) Thesis stays about volunteer consumer hardware. (+) A clean optional chapter if the hardware appears.

## ADR-005 — Private mesh (Tailscale) instead of exposing RPC ports
**Date:** 2026-08 · **Status:** accepted
**Context:** llama.cpp README: "never run the RPC server on an open network."
**Decision:** All inter-node traffic beyond a single LAN goes over Tailscale. Workers bind to their Tailscale IP.
**Consequences:** (+) NAT traversal and encryption solved for free. (−) Every volunteer must join the tailnet; acceptable at project scale (≤ 100 devices free tier).

## ADR-006 — Model license: prefer Apache-2.0 weights (Qwen / Gemma-class as allowed)
**Date:** 2026-08 · **Status:** accepted
**Context:** Llama weights carry a community license with restrictions; Qwen is Apache-2.0. Earlier Hebrew benchmarks favoured Gemma-2 9B for output quality.
**Decision:** Default experiment model is a Qwen GGUF for licensing simplicity; Gemma-2 9B remains the Hebrew-quality reference. Both recorded in RESULTS.md per run.
**Consequences:** No license discussion needed in the thesis for the primary model.

## ADR-007 — Data-plane engine: decided by spike, behind a fixed Worker interface
**Date:** 2026-10-10 · **Status:** proposed (decision due 15 Nov 2026)
**Context:** PoC-1 review raised PyTorch vs llama.cpp and C++ vs Python. New requirements: more than one parallelism strategy (pipeline / tensor / hybrid) chosen from measured resources; liveness and continuity of in-flight generations; Apple Silicon as a first-class worker. The existing experiment targets layer splitting; experimental tensor splitting over RPC remains unverified for our model/devices/build. Worker loss may terminate the requesting process. Direct Gloo/NCCL operations on MPS are not assumed; CPU-staged Gloo is a transport candidate. Placement, generation state and recovery policy remain system responsibilities, with restart/replay to be validated for either compute engine.
**Decision (framework, not yet the pick; revised 2026-10-10 after I003/I004):** (1) The control plane (tracker, agent, profiler, planner, CLI, API) is Python regardless. (2) The distributed runtime (placement, worker protocol, membership policy, failure handling, generation state) is ours — this is the contribution; transport may be our own TCP or a reused component (Gloo on CPU tensors), decided by measurement in R1. (3) The compute engine is reused, never written from scratch (re-affirms ADR-003), and sits behind a Worker interface: `load_shard(manifest, layers, tp_rank)`, `forward(activations)`, `collective_hook`, `replay(prefix_tokens)`. (4) Two candidates are evaluated against **identical mandatory requirements**: A = llama.cpp at pinned SHA with `--split-mode tensor --rpc`; B = a PyTorch-based runtime (gated steps R0 to R4 in docs/archive/RESEARCH_PLAN_v3_2026-10-10.md section 6). Mandatory requirements: correctness on the chosen model and dtype; real use of both accelerators including the Mac; memory fitting the target configuration; measurable and controllable placement; defined timeout; system-level replay path. *Pending team confirmation:* ability to run a legal TP configuration. **Symmetric rule:** only A passes → A (or an explicit scope change for any untested requirement); only B passes → B; both pass → compare measurements and pre-declared development/deployment/control criteria; neither → fix a bounded blocker or narrow scope, no automatic winner; evidence missing → decision stays open and the missing test is named. Failure of one engine is never evidence for the other. "Which strategy pays off" (H4) is a research result, not an engine acceptance criterion.
**Consequences:** (+) The engine becomes swappable by ADR, not by rewrite. (+) The decision is made on our measurements, with language, engine, quantization and copy path kept as separate factors (archived research plan, section 5). (+) V1 full-prefix replay is a system-level candidate for both engines, subject to implementation and correctness tests; continuity does not pre-decide the engine. (−) Two to three weeks spent before "real" PoC-2 code. (−) If B is chosen, volunteer deployment burden and bf16 memory footprint are measured costs to carry; if A is chosen, partial-KV continuity and asymmetric TP plans are documented limitations.

## ADR-008 — Tracker deferred; static peer list until the planner and continuity exist
**Date:** 2026-10-10 · **Status:** proposed
**Context:** Supervisor: "tracker can wait." The tracker's announcement payload depends on the worker profile that O5 defines.
**Decision:** PoC-2 uses a static `peers.toml`. The tracker (ADR-001 unchanged in role) is built in PoC-3 once the profile format is stable.
**Consequences:** (+) No throw-away tracker schema. (−) Three-node experiments need manual peer files until Jan 2027; acceptable.

## Team decisions (direction, not engineering)

Recorded here so the repository is self-contained. The detailed discussion lives in a private working log.

| Id | Date | Who | Decision |
|---|---|---|---|
| T1 | 2026-10-10 | Team (Ohad) | Latency is not a target and not the interesting factor. The project shows feasibility and efficiency of splitting a model across more machines. The network, later the internet, is the given bottleneck. We do not set out to fix it. Revisit if the supervisor asks for latency optimization. |
| T2 | 2026-10-11 | Team (Ohad) | The claim: run a huge model, larger than any single machine, on several home computers over the internet, and succeed. It runs, finishes and gives a good answer. "A task before sleep, an answer in the morning" is one illustration, not a deadline. Priorities: capability and quality, reliability, automatic planner. Revisit if the supervisor asks for an interactive use case. |
| T3 | 2026-10-11, recorded 2026-10-17 | Supervisor via the team | Git for the project. Markdown in short sentences, no long dash. Goal states only the peer-to-peer idea. Five short objectives first: O1 single machine; O2 split by layers; O3 more than two machines, internet, generic models; O4 performance study incl. PP vs TP; O5 public peer-to-peer application with liveness. Metal vs CUDA becomes a review. Flow diagram, platform doc and formal design doc requested. Open: why TP on 32B failed. See docs/he/MEETING_CHEN_2026-10-11.md. |

## ADR-009 — Open internet connectivity without a mesh VPN; security deferred
**Date:** 2026-10-24 · **Status:** proposed (team: Ohad; supervisor not yet asked)
**Context:** ADR-005 routes all traffic beyond one LAN over Tailscale. The supervisor asks for operation over the internet and a wider network (O3) and for an application the public can use (O5). The team wants joining to be as easy and open as possible. Every participant installing a VPN is friction. The team states that security is not a concern at this stage. llama.cpp warns against running the RPC server on an open network. Workers sit behind home NAT; some Israeli ISPs use CGNAT, where port forwarding fails.
**Decision (proposed):** (1) Connectivity is a ladder tried automatically by the worker agent: same LAN direct; UPnP port mapping; relay through the tracker host (frp or equivalent). (2) The tracker host is also the relay host. (3) Tailscale stays as a fallback and as the first tool for the first cross-home experiment, since it needs no server. (4) Security (encryption, authentication, abuse) is explicitly deferred and recorded as a known limitation in the thesis. (5) Wired links (Thunderbolt Bridge, Ethernet) are diagnostic tools only, not part of the product. Details in docs/he/CONNECTIVITY.md.
**Consequences:** (+) A volunteer joins with one install and no router or VPN setup. (+) Works behind CGNAT. (+) Matches BitTorrent practice (UPnP, then relay). (−) Relay doubles per-token network time and is capped by the host's bandwidth; acceptable under T1. (−) An open RPC port accepts any client; must be fixed before any public release. (−) ADR-005 is narrowed, not reversed: Tailscale remains valid for experiments.

---
*Template:*
```
## ADR-NNN — Title
**Date:** · **Status:** proposed | accepted | superseded by ADR-MMM
**Context:**
**Decision:**
**Consequences:**
```
