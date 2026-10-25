# Backends: Metal, CUDA, and where DOCA fits

**Status (2026-10-25):** background note from Sept 2026. The current summary is PLATFORM.md section 3. Kept for the thesis background chapter.

Answers the supervisor's question "how do we choose the GPU core, or should we use DOCA or Metal, and what does that mean?"

## 1. Three levels of control — and which one is ours

| Level | What you choose | Who decides | Tool |
|---|---|---|---|
| **Device** | *Which* accelerator: e.g. `MTL0`, `CUDA0`, `CUDA1`, or CPU execution | **Us** | `--device`, `--list-devices`, `CUDA_VISIBLE_DEVICES` |
| **Kernel dispatch** | How a matmul is tiled into threadgroups (Metal) / thread blocks (CUDA) | ggml's backend code | `ggml-metal.m`, `ggml-cuda/` — hand-written per-op kernels |
| **Core / SM scheduling** | Which physical GPU core executes which threadgroup | **The driver and hardware. Not exposed by any public API** on either platform | — |

"Choosing the GPU core inside" is therefore not a thing we can do, on Metal or on CUDA. The unit of *placement* we control is the device; the unit of *work* we control is the layer range assigned to that device. Everything below that is the backend's job, and llama.cpp's backends are already tuned by people who do this full time. **Decision: we operate at the device + layer-range level and do not write kernels.** (Recorded as ADR-003 in DECISIONS.md.)

If we ever need to go one level down, the path is a custom ggml op with a Metal Shading Language or CUDA kernel — resources are in LEARNING_RESOURCES.md §4. It is not on the objectives list.

## 2. Metal (Apple Silicon)

- Apple's GPU API. llama.cpp's `ggml-metal` backend compiles Metal Shading Language kernels at runtime for the exact M-series chip present.
- **Unified memory:** CPU and GPU share one physical pool. Model weights are `mmap`-ed once; no PCIe copy. On a 24 GB Mac the GPU can address roughly 16–18 GB of that pool by default (the "recommended max working set"); the rest falls back to CPU automatically.
- **Apple Silicon GPU.** The inspected build names its first Metal device `MTL0`. Always copy the exact identifier from `--list-devices`; examples are not a discovery mechanism. Forcing CPU is `--device none` (or `-ngl 0`).
- Built by default on macOS; add `-DGGML_RPC=ON` to also get the RPC server.

## 3. CUDA (NVIDIA)

- Discrete VRAM; weights are copied host → device on load. `--list-devices` enumerates `CUDA0`, `CUDA1`, … with free memory.
- Two ways to pick a GPU, equivalent in effect: `CUDA_VISIBLE_DEVICES=1 ./llama-cli …` (hides the others at driver level) or `./llama-cli --device CUDA1 …` (ggml-level filter, works for any backend). The RPC server accepts the same `--device` flag.
- Multi-GPU *within one machine* uses the same `--split-mode layer` / `--tensor-split` mechanism as the network split. Good: our placement code does not care whether two devices are on one box or two.
- Build: `-DGGML_CUDA=ON -DGGML_RPC=ON`. Needs the CUDA toolkit matching the driver.

## 4. Vulkan (fallback, any GPU)

If a friend's machine has an AMD or Intel GPU, llama.cpp's Vulkan backend exposes it the same way (`Vulkan0`). Same flags. Worth knowing for PoC-3 when we recruit volunteers.

## 5. DOCA — what it is, and the honest answer

**DOCA is not a GPU API.** It is NVIDIA's SDK for BlueField DPUs (Data Processing Units) — SmartNICs with Arm cores that offload *networking, storage and security* from the host CPU. NVIDIA's own framing is "DOCA is to DPUs what CUDA is to GPUs". It has nothing to do with selecting or driving GPU compute.

Where NVIDIA's networking stack *does* touch this project:

- llama.cpp's RPC transport can use **RDMA** instead of TCP on Linux hosts with RoCEv2-capable NICs (ConnectX), negotiated automatically when `libibverbs` is found at build time. That is the standard InfiniBand verbs library, not DOCA.
- RDMA cuts per-hop latency from hundreds of microseconds to single digits. Since H1 says hops × RTT is our bottleneck, an RDMA-vs-TCP data point would be a strong result — **but only in a lab**, because consumer home networks have neither the NICs nor the switches.
- A BlueField/DOCA setup would additionally let a DPU host the *tracker* or terminate the mesh VPN without touching the host CPU. Interesting, entirely optional, and a distraction from the research question.

**Decision:** DOCA is out of scope. RDMA via libibverbs is a **stretch objective** conditional on lab hardware. Consumer-hardware TCP is the primary target because that is what the thesis is about.

## 6. Practical implication for PoC-0

On both machines run, in this order:

```
./build/bin/llama-cli --list-devices
./build/bin/llama-bench -m <model.gguf> -ngl 99            # GPU
./build/bin/llama-bench -m <model.gguf> -ngl 0             # CPU
# CUDA box with >1 GPU:
./build/bin/llama-bench -m <model.gguf> -ngl 99 --device CUDA1
```

`llama-bench` reports prefill (pp) and generation (tg) tok/s separately — that separation is the baseline row for every table in the thesis. `scripts/05_bench.sh` automates it.
