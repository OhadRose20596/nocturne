# Platform: hardware, software, and the GPU vs CPU question

Expectation alignment with the supervisor (meeting of 2026-10-11). Lists everything we run on and with. Absorbs the former objective "Metal vs CUDA" as a review, not a goal. Status: proposed.

## 1. Machines

| Machine | Owner | Chip | Memory | Accelerator | OS | Role so far |
|---|---|---|---|---|---|---|
| MacBook Air M3 | Ohad | Apple M3 | 16 GB unified | Metal (`MTL0`), GPU working set about 12.1 GB | macOS | Requester and worker. Tables A, B, 32B runs |
| Mac M5 | Ohad | Apple M5 | 24 GB unified | Metal (`MTL0`), GPU working set about 18.2 GB | macOS 26.7.1 | Requester and worker. Tables A, B, 32B runs |
| NVIDIA machine | Lior | unknown | unknown | CUDA (`CUDA0`) | Windows, WSL2 planned | Not yet run. Spec to be recorded in `experiments/inventory/` |
| Lab machine | BGU, if granted | unknown | unknown | NVIDIA | Linux | Requested. Needed for N=4 or 72B |
| Tracker host | any | tiny | 512 MB is enough | none | any | Raspberry Pi or small VPS. Not built yet |

Network: home Wi-Fi today (ping about 10 to 23 ms average, up to 157 ms max; 46 to 285 Mbit/s). Internet between homes via Tailscale (ADR-005). An Ethernet or USB-C cable is a sensitivity check only.

## 2. Software

| Layer | Choice | Why | Status |
|---|---|---|---|
| Inference engine | llama.cpp, pinned SHA `1e7bcf3` | One code base for Metal, CUDA, Vulkan, CPU. Mature 4-bit quantization. Built-in RPC | In use. ADR-002 |
| Distribution | llama.cpp RPC: `ggml-rpc-server` on workers, `--rpc` and `--tensor-split` on the requester | Already does layer split across machines | In use. Officially proof of concept upstream |
| Alternative engine | PyTorch runtime behind a Worker interface | Generality, partial loading, smarter continuity | Candidate only. ADR-007 open, decision due 15 Nov 2026 |
| Model format | GGUF today. Hugging Face safetensors if PyTorch | See BACKGROUND_AND_DESIGN.md section 3 | GGUF in use |
| Control plane | Python 3.11+, FastAPI, SQLite, httpx, pytest | Small, runs on a Pi | Stubs only |
| Mesh network | Tailscale | NAT traversal and encryption for free | Installed. No cross-home run yet |
| Scripts | bash, macOS 3.2 compatible | Humans run them without an assistant | `scripts/00` to `14` |
| Measurement | `llama-bench`, `GGML_RPC_DEBUG=1`, `iperf3`, `ping`, `/usr/bin/time` | Same tool for every compared number | In use |
| Version control | git, remote on GitHub | Supervisor request | Initialized 2026-10-17 |
| Docs | Markdown, Hebrew in `docs/he/`, HTML via pandoc | Readable RTL for the team | In use |

## 3. Review: GPU or CPU, Metal or CUDA

This was objective O2. It is now background. The questions and the answers we have.

**Do we need a GPU at all?** For a 7B model on the Air: GPU 20 tok/s decode, CPU 14.7. On the M5: 26.3 vs 17.5. Prefill differs more: 208 vs 53 and 683 vs 68. So the CPU is usable for small models and for development. For 32B the CPU path gave 0.04 to 0.31 tok/s with offload to CPU. That is not useful. Answer: GPU for real runs, CPU for plumbing tests and as a fallback.

**What can we choose on the GPU?** The device, and the layer range on it. Not the core. Neither Metal nor CUDA exposes core or SM scheduling. Kernels belong to ggml. ADR-003.

**Metal.** Unified memory. Weights are mmap-ed once. The GPU may use about 70 to 75 percent of RAM by default (the recommended working set). Above that, allocation fails or layers stay on the CPU. We saw a real GPU out-of-memory on the M5 with 32B. So the working set is a practical limit, not only a recommendation.

**CUDA.** Discrete VRAM. Weights are copied on load. `--device CUDA1` or `CUDA_VISIBLE_DEVICES` picks a card. Several cards in one box use the same `--tensor-split` mechanism as several machines. Not yet measured in this project.

**Vulkan.** Fallback for AMD or Intel GPUs of future volunteers. Same flags.

**DOCA.** Not a GPU API. It is the SDK for BlueField DPUs (networking offload). Out of scope. ADR-004. RDMA over libibverbs is a lab-only stretch.

**Where this enters the plan.** The planner needs per-machine numbers: usable accelerator memory, decode speed per layer, prefill speed. These come from the O1 baseline on each machine. The Metal vs CUDA difference is therefore an input to the planner, not a research question.

## 4. What we ask of each participant machine

- A supported accelerator or a CPU with enough RAM.
- llama.cpp built at the pinned SHA, outside iCloud or other synced folders.
- The model file, or disk space for the worker cache.
- Tailscale, if outside the home network.
- Nothing else. No dedicated hardware purchase.

## 5. Open items

- Record the NVIDIA machine: GPU model, VRAM, driver, WSL2 version.
- Decide if a lab machine is available and when.
- Decide the tracker host when the tracker is built.
