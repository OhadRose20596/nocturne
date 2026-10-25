# Architecture: BitTorrent Model Applied to LLM Inference

**Status:** v2.1 (2026-10-25). Requester-coordinated, tracker-assisted design from the Sept 2026 meeting with the supervisor (ADR-001). Sections 6 and 7 updated to the v4 objectives (GOAL_AND_OBJECTIVES.md). The step-by-step flow is in FLOW.md.

## 1. The one-sentence change

> The machine that *wants* the answer is the one that coordinates the work. The central server only tells it who is available.

In the previous design a central Coordinator decided everything: who participates, how layers are split, and it launched the run. That is a master/worker system. BitTorrent does not work that way, and neither should Nocturne.

## 2. How BitTorrent actually works (what we are copying)

BitTorrent (BEP 3) separates three roles that are usually blurred together:

| Role | What it does | What it does *not* do |
|---|---|---|
| **Tracker** | Answers one question: "who is in the swarm for file X right now?" Peers *announce* themselves periodically (`started`, `stopped`, `completed`, and a regular interval). Returns a random subset of peers. | Never touches data. Never decides which piece goes where. Stateless beyond the peer list. |
| **Peer** (seeder / leecher) | Holds some or all pieces of the file. Serves them on request. Announces `have <piece>` when it gets a new piece. | Does not know the global picture. Only knows its direct neighbours. |
| **Downloader** (the leecher that wants the file) | **Is its own coordinator.** Picks which peers to talk to, which piece to request from whom, when to drop a slow peer, when to duplicate a request (endgame). | Does not ask anyone's permission. |

Key mechanisms inside the downloader:

- **Piece selection — rarest first.** Request the piece that the fewest peers hold, so no piece dies with a single peer.
- **Choking / tit-for-tat.** Each peer uploads to the ~4 peers that give it the best download rate, plus one random "optimistic unchoke" to discover better partners. This is the fairness/incentive layer.
- **Endgame mode.** When only a few pieces remain, request them from *every* peer that has them and cancel duplicates. Kills the long tail caused by one slow peer.
- **Piece hashing.** Every piece is verified against the SHA-1 in the `.torrent`. Bad peers are detected and dropped.
- **Metainfo (`.torrent`).** A tiny file: piece size, piece hashes, total length, tracker URL. Everything a downloader needs to start, nothing more.

Later extensions (DHT — BEP 5, Peer Exchange — BEP 11) remove the tracker entirely, but the *client* design did not change: the downloader was always the coordinator.

## 3. The mapping to distributed LLM inference

| BitTorrent | Nocturne | Notes |
|---|---|---|
| The file | The model (one GGUF) | Identified by content hash, exactly like an info-hash |
| `.torrent` metainfo | **Model manifest** — GGUF hash, layer count, bytes per layer, hidden size, tracker URL | Tiny, static, shareable |
| Piece | **A contiguous range of transformer layers** | llama.cpp's `--split-mode layer` (pipeline parallel) — each device owns a slice |
| Tracker | **Tracker** — nodes announce `{device, free VRAM, backend, bandwidth, cached model hashes}` every N seconds; requester asks "who is up for model X?" | The *only* central component. Stateless. Raspberry Pi / €4 VPS class. |
| Seeder | **Worker** — a machine running `ggml-rpc-server` exposing a GPU (or CPU) | Any OS: Metal, CUDA, Vulkan, CPU |
| Downloader / leecher | **Requester** — the machine that has the prompt and wants tokens. Runs `llama-cli` / `llama-server` with `--rpc host:port,...` and `--tensor-split`. **It is the coordinator of its own job.** | This is the "Main Host" in llama.cpp terms |
| Piece selection (rarest first) | **Layer placement** — which worker gets which layer range, how big each range is | Our research contribution lives here (see GOAL_AND_OBJECTIVES.md) |
| `have <piece>` announcements | Worker advertises **which model shards it already has cached** (`ggml-rpc-server -c`) | Cached workers are "seeders" for that model — load is near-instant |
| Choking / tit-for-tat | **Contribution accounting** (future) | Out of scope for the degree project, documented as future work |
| Endgame mode | **Straggler mitigation** — duplicate a slow worker's layer range on a spare node and take whichever answers first | Future / stretch objective |
| Piece hash check | **Output verification** | Hard problem in inference (non-deterministic float math across backends). Documented as a known gap. |
| DHT / PEX | Tracker-less discovery (mDNS on LAN, DHT on WAN) | Explicitly *not* in scope; would be the natural next paper |

### Where the analogy breaks — and why it matters for design

1. **Pieces are independent; layers are not.** In BitTorrent any piece can arrive in any order from anyone. In pipeline inference layer 12 *must* run after layer 11, on every token, and the KV cache for layer 12 lives on whichever machine owns layer 12. If that machine leaves mid-generation the KV cache is gone; the requester must restart the sequence (or later: checkpoint). This makes **node availability a first-class scheduling input**, which BitTorrent never needed.
2. **BitTorrent is bandwidth-bound; decode is bound by what crosses each boundary per token.** Team decision T1: the slow network is the premise, not the problem. Measured (trace, 2026-10-11): per decode step one blocking round trip per boundary, about 14 KB of hidden state for a 7B model, and 608 KB of logits if the output layer sits on a worker. So fewer, larger layer ranges beat many small ones, and the output layer stays on the requester. The opposite intuition from BitTorrent's small pieces.
3. **Weight distribution *is* BitTorrent-shaped.** Loading a model pushes gigabytes from the requester to each worker. That part is bulk transfer, and here the analogy is exact: workers with `-c` cache become seeders, and in a later phase workers could fetch shards from each other instead of from the requester. Two different traffic patterns → two different mechanisms.
4. **No hash verification of results.** We cannot cheaply prove a worker computed its layers correctly. We note it and move on; Petals has the same gap.

## 4. Roles, concretely

```
┌──────────────┐  announce {gpu, vram_free, backend, bw, cached[]}   ┌─────────┐
│  Worker A    │ ─────────────────────────────────────────────────►  │         │
│ ggml-rpc-srv │ ◄───────────── (ack, interval) ─────────────────── │ Tracker │
│  MTL0        │                                                     │  (tiny) │
└──────────────┘                                                     │         │
┌──────────────┐                                                     │         │
│  Worker B    │ ───────────── announce ───────────────────────────► │         │
│ ggml-rpc-srv │                                                     └────┬────┘
│  CUDA0       │                                                          │
└──────────────┘                            "who is up for model X?"      │
       ▲                                    ◄─────────────────────────────┘
       │ TCP :50052                          ┌─────────────────────────────┐
       │  activations per token              │  Requester                  │
       └─────────────────────────────────────│  1. get peer list           │
                                             │  2. PLACE layers (ours)     │
                                             │  3. llama-cli --rpc A,B     │
                                             │       --tensor-split a,b    │
                                             │  4. watch, re-place on loss │
                                             └─────────────────────────────┘
```

**Tracker** — HTTP + SQLite. Endpoints: `POST /announce`, `GET /peers?model=<hash>`. Nothing else. If it dies, in-flight jobs keep running (workers talk directly to the requester); only new jobs are blocked.

**Worker** — `ggml-rpc-server --device <dev> -c -H <bind> -p <port>` plus a ~100-line agent that announces to the tracker and measures its own bandwidth/latency to peers. Reachability beyond the LAN follows the ladder in docs/he/CONNECTIVITY.md: direct, UPnP, relay through the tracker host, Tailscale as fallback (ADR-005, ADR-009). llama.cpp warns against an open RPC port; security is deferred by team decision and recorded as a limitation.

**Requester** — a Python CLI (`nocturne run --model X --prompt ...`) that: pulls peers, runs our placement algorithm, emits exactly two llama.cpp arguments (`--rpc` and `--tensor-split`), launches `llama-server`, and monitors. The placement algorithm is the pluggable heart: v0 = manual, v1 = proportional-to-memory (llama.cpp default), v2 = availability-and-latency-aware (our contribution).

## 5. Why llama.cpp RPC is the right substrate for this (and its limits)

- One binary family covers Metal, CUDA, Vulkan, CPU — we *have* a Mac and an NVIDIA box, so heterogeneity is day one.
- The requester already *is* the coordinator in llama.cpp's model: the "Main Host" holds the GGUF, decides the split, and drives the workers. We are not fighting the tool.
- Worker-side cache (`-c`) gives us the seeder concept for free.
- RDMA transport (RoCEv2 / ConnectX, negotiated automatically when `libibverbs` is present) exists for lab use — see BACKENDS.md.
- **Limits:** RPC is officially "proof-of-concept, fragile and insecure". No mid-run re-placement: if a worker dies the process dies, and we restart with a new split. That is acceptable — measuring the *cost* of that restart under different placements is part of the research.

## 6. Development ladder

Maps to the objectives in GOAL_AND_OBJECTIVES.md v4.

| Stage | Objective | Machines | What is proven | State (2026-10-25) |
|---|---|---|---|---|
| Single device | O1 | 1 | Build, list devices, force a run onto a chosen device, measure | Done on two Macs. NVIDIA machine pending |
| Manual split | O2 | 2 on home LAN | Requester drives a worker over `--rpc` with a manual `--tensor-split` | Done. 32B runs on two Macs, fails on each alone |
| Scale out | O3 | 3+, two homes | Three machines, one outside the home network, a model chosen by name | Not started |
| Performance study | O4 | 1 to 3 | Capacity, saturation, PP vs TP, model to processor | Pilots only |
| Application | O5 | 3+ | Tracker, worker agent, requester CLI, planner, liveness | Not started. Static peer list first (ADR-008) |

Scripts 00 to 14 cover O1, O2 and the O4 pilots. The Python package holds stubs for O5.

## 7. Open design points

Listed with the supervisor's questions in FLOW.md section 4 and in docs/he/NEXT_STEPS_2026-10-17.md.
