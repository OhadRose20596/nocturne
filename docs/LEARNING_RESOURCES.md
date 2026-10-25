# Learning Resources

Rule (from the supervisor): every tool or technology that requires learning gets an entry here *before* we use it. Tag: **[must]** read before touching the tool · **[should]** read during the relevant PoC · **[background]** thesis literature.

Format: `- [tag] Title — why it matters for us. URL`

## 1. BitTorrent (the model we are copying)

- [must] **BEP 3 — The BitTorrent Protocol Specification.** The original spec: tracker, announce, choking, rarest-first. Short; read it whole. https://www.bittorrent.org/beps/bep_0003.html
- [should] **BEP 52 — BitTorrent v2.** Per-file hash trees; relevant to how we might hash model shards. https://www.bittorrent.org/beps/bep_0052.html
- [should] **BEP 5 — DHT Protocol.** How BitTorrent dropped the tracker. We are *not* doing this; read to justify why in the thesis. https://www.bittorrent.org/beps/bep_0005.html
- [background] Cohen, *Incentives Build Robustness in BitTorrent* (2003). The tit-for-tat paper. http://bittorrent.org/bittorrentecon.pdf
- [background] Legout et al., *Rarest First and Choke Algorithms Are Enough* (IMC 2006). Empirical validation of the two core mechanisms. https://arxiv.org/abs/cs/0609026
- [background] *Clustering and Sharing Incentives in BitTorrent Systems.* https://arxiv.org/pdf/cs/0703107

## 2. llama.cpp — RPC, devices, multi-GPU

- [must] **RPC backend README** (`tools/rpc/README.md`). Architecture diagram, `ggml-rpc-server` flags (`--device`, `-c` cache, `-H`, `-p`, `--mem`), RDMA note, the "never on an open network" warning. https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- [must] **Multi-GPU guide** (`docs/multi-gpu.md`). `--split-mode`, `--tensor-split`, `--device`, `--list-devices`, `--main-gpu`; pipeline vs tensor parallel trade-off. https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- [must] **Build guide** (`docs/build.md`). CMake flags for Metal, CUDA, Vulkan, RPC. https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- [should] **llama-server README.** The OpenAI-compatible HTTP API the requester will expose. https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- [should] **llama-bench** (`tools/llama-bench/README.md`). Our measurement instrument; understand `pp` vs `tg`. https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench
- [should] **GGUF format spec.** Needed to write the model manifest (layer count, tensor sizes). https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- [should] Arm Learning Path, *Distributed inference with llama.cpp* — a worked two-node RPC walk-through. https://learn.arm.com/learning-paths/servers-and-cloud-computing/distributed-inference-with-llama-cpp/
- [background] DeepWiki, *Multi-GPU and Distributed Inference* in llama.cpp — code-level explanation. https://deepwiki.com/ggml-org/llama.cpp/8.4-multi-gpu-and-distributed-inference

## 3. GPU backends

- [must] Apple, **Metal overview** — what Metal is, one page. https://developer.apple.com/metal/
- [should] Apple, **Creating threads and threadgroups** — the dispatch model; explains why "choosing a core" is not exposed. https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- [should] DeepWiki, **Metal Backend (Apple)** in llama.cpp — how ggml maps ops to Metal kernels. https://deepwiki.com/ggml-org/llama.cpp/5.2-metal-backend-(apple)
- [should] *Tuning llama.cpp on Apple Silicon: 7 flags.* Practical. https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc
- [must] NVIDIA, **CUDA C++ Programming Guide** — chapters 1–3 only (programming model, device selection). https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- [should] NVIDIA, `CUDA_VISIBLE_DEVICES` and env vars. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
- [background] Apple, **Metal Shading Language Specification** — only if we ever write a kernel. https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

## 4. DOCA / RDMA (stretch only)

- [should] NVIDIA, **DOCA overview** — read once to be able to say precisely why it is out of scope. https://developer.nvidia.com/networking/doca
- [should] NVIDIA Technical Blog, *Developing Applications with BlueField DPU and DOCA Libraries.* https://developer.nvidia.com/blog/developing-applications-with-bluefield-dpu-and-doca-libraries/
- [background] RDMA / libibverbs programming intro (RDMAmojo). https://www.rdmamojo.com/

## 5. Distributed LLM inference — prior art (literature review core)

- [must] Borzunov et al., **Petals: Collaborative Inference and Fine-tuning of Large Models** (2022). The closest system; DHT-based, client routes through servers. https://arxiv.org/abs/2209.01188
- [must] Borzunov et al., **Distributed Inference and Fine-tuning of LLMs Over The Internet** (NeurIPS 2023). Fault tolerance and load balancing under churn — our O6 baseline. https://arxiv.org/abs/2312.08361
- [must] Li et al., **Prima.cpp: Fast 30–70B LLM Inference on Heterogeneous and Low-Resource Home Clusters** (2025). Piped-ring parallelism + Halda layer-to-device assignment — the placement algorithm to compare against. https://arxiv.org/abs/2504.08791
- [should] **exo** (exo-explore). Peer-to-peer, auto-discovery, ring memory-weighted partitioning, no master. The "no coordinator" alternative. https://github.com/exo-explore/exo
- [should] **Petals repository** — read `server/block_selection.py` for how servers pick which blocks to serve (rarest-first analog). https://github.com/bigscience-workshop/petals
- [should] Mei et al., **Helix: Distributed Serving of LLMs via Max-Flow on Heterogeneous GPUs** (2024). Placement as an optimization problem. https://arxiv.org/abs/2406.01566
- [should] **HexGen-2** — heterogeneous serving, placement via graph partitioning. https://arxiv.org/abs/2502.07903
- [background] Kwon et al., **vLLM / PagedAttention** (2023). KV-cache management background. https://arxiv.org/abs/2309.06180
- [background] **hivemind** — the DHT library under Petals. https://learning-at-home.github.io/
- [background] Anderson, **BOINC: A Platform for Volunteer Computing** (2019). The volunteer-computing lineage. https://arxiv.org/abs/1903.01699

## 6. Networking and infrastructure

- [must] **Tailscale docs — quickstart.** Our private mesh for anything beyond one LAN. https://tailscale.com/kb/1017/install
- [should] Tailscale, *How NAT traversal works.* Explains why we do not build this ourselves. https://tailscale.com/blog/how-nat-traversal-works
- [should] **FastAPI** — tracker HTTP service. https://fastapi.tiangolo.com/
- [should] **SQLite** — tracker state. https://www.sqlite.org/docs.html
- [background] **iroh** — if we ever go tracker-less over WAN. https://github.com/n0-computer/iroh
- [should] **Apple, Thunderbolt Bridge** — direct Mac-to-Mac network over a Thunderbolt cable; our fast-link diagnostic point. https://support.apple.com/guide/mac-help/mchlp2712/mac
- [must] **frp (fast reverse proxy)** — relay through the tracker host for workers behind NAT or CGNAT; candidate for the O5 connectivity ladder (ADR-009). https://github.com/fatedier/frp
- [should] **miniupnpc / UPnP IGD** — ask a home router to map a port automatically, as BitTorrent clients do. https://miniupnp.tuxfamily.org/
- [should] **bore** — one-command public TCP tunnel for quick WAN tests without our own server. https://github.com/ekzhang/bore
- [should] **DuckDNS** — free dynamic DNS for a worker behind a home router with port forwarding. https://www.duckdns.org/
- [should] **Microsoft, WSL2 networking** — mirrored mode and port forwarding so a worker inside WSL2 is reachable. https://learn.microsoft.com/windows/wsl/networking

## 7. Tooling

- [must] **Claude Code docs** — how we develop in this repo. https://docs.claude.com/en/docs/claude-code/overview
- [should] `CLAUDE.md` best practices. https://docs.claude.com/en/docs/claude-code/memory
- [should] Hugging Face, **GGUF model hub** (`ggml-org` and `bartowski` quantizations). https://huggingface.co/models?library=gguf
- [must] Hugging Face, **CLI installation and downloads** — install `hf` with Homebrew on macOS or inside a Python virtual environment; download the exact GGUF file. https://huggingface.co/docs/huggingface_hub/en/guides/cli

---

### Adding an entry

Before you run a new tool for the first time: add one `[must]` link here, commit, then run the tool. If you cannot find a source worth linking, that is a signal the tool is not well enough documented to rely on.
