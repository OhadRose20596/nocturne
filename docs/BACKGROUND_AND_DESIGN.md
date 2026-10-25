# Formal background and design

Requested by the supervisor on 2026-10-11: define the terms precisely. Explain PP vs TP in practice. Define the model format. Status: proposed. Short sentences by request. The long Hebrew explanation is in docs/he/PARTITIONING_DEEP_DIVE.md.

## 1. Definitions

**Model.** A dense decoder-only transformer. `L` layers. Hidden size `d`. Vocabulary `V`. Each layer has attention and an MLP. An embedding table maps tokens to vectors. An output head maps the last hidden state to `V` logits.

**Inference.** Two phases. *Prefill*: all prompt tokens pass through all layers once. *Decode*: one new token per step. Each step passes one vector of size `d` through all layers and reads the KV cache.

**KV cache.** Per layer, per token: the keys and values of attention. Size per token is `2 × L × n_kv_heads × head_dim × bytes`. For Qwen2.5-32B in 16-bit that is about 256 KB per token. 32k tokens of context is about 8 GB.

**Memory per machine.** `weights_on_machine + kv_on_machine + compute_buffers + headroom`. A plan is *legal* only if this fits the machine's usable accelerator memory. On Apple Silicon that is the Metal working set, about 70 to 75 percent of RAM.

**Requester.** The machine that holds the prompt and wants the answer. It coordinates the job.

**Worker.** A machine that offers one accelerator. It holds a shard and computes on request.

**Tracker.** A directory. It answers "who is online and what do they have". It never touches model data.

**Placement.** The assignment of model parts to machines. In llama.cpp this is `--rpc <endpoints>` plus `--tensor-split <proportions>` and `--split-mode`.

**Pipeline parallelism (PP).** Split by layers. Machine `i` holds layers `[a_i, b_i)`. Data flows machine to machine once per layer boundary.

**Tensor parallelism (TP).** Split inside each layer. Every machine holds a slice of every weight matrix. Machines exchange partial results inside every layer.

**Replicas.** No split. Each machine holds the whole model. Requests are routed between machines.

## 2. PP vs TP in practice

### What crosses the network

| | PP | TP |
|---|---|---|
| Per decode step | One hidden-state vector per boundary. `d × 4` bytes. For Qwen2.5-32B: 5120 × 4 = 20 KB. For 7B: 14,336 B (measured) | Two all-reduce rounds per layer. Each moves a `d`-sized vector per machine. For 64 layers: about 128 exchanges per token |
| Per prefill | `d × 4 × prompt_tokens` per boundary | The same per layer, times 2 per layer |
| Waits per token | One blocking round trip per boundary (measured with llama.cpp RPC) | Roughly `2 × L` blocking rounds |
| Output head remote | Ships `V × 4` logits per token. 152,064 × 4 = 608 KB for Qwen2.5 (measured). Keep the head on the requester | Same rule |

### Memory per machine

| | PP | TP |
|---|---|---|
| Weights | Only its layers. Uneven split possible | About `1/N` of every layer. Even split |
| KV cache | Only for its layers | `1/N` of every layer's KV (heads are split) |
| Embedding and output head | On one machine | Replicated, or split by rows |
| Fit rule for 32B Q4 (19.85 GB) on two machines | 10 + 10 GB of weights plus KV of own layers. Fits two Macs. Shown | About 10 GB of weights per machine plus half the KV. Should fit. Not shown yet |

### When each is preferred

- **Slow network, long distance.** PP. Few exchanges per token. Network time per token is one round trip per boundary.
- **Fast interconnect in one box.** TP. All machines compute every token together. Lower latency per token. Needs NVLink or PCIe class links.
- **Large prompt, short output.** PP suffers less. Prefill crosses each boundary once with a big block.
- **Small model that fits everywhere.** Replicas. Measured: for 7B, replicas beat a split by 1.4 to 2.3 times in throughput.
- **Model that fits on no single machine.** PP first. TP if PP cannot balance memory, or to test the supervisor's expectation.
- **Mixed.** PP between homes, TP inside one home or one box. Not planned before N = 4.

Upstream summary (llama.cpp docs/multi-gpu.md): pipeline parallel minimizes data transfer and maximizes batch throughput. Tensor parallel does multiple cross-GPU reductions per layer and minimizes latency. It is much more bottlenecked by the interconnect.

### What is in llama.cpp at our pinned SHA (1e7bcf3)

`--split-mode` takes `none`, `layer`, `row`, `tensor`.
- `layer` is PP. Default. Used in all our runs.
- `row` splits weight matrices by rows. Older path. CUDA oriented.
- `tensor` is TP. Marked EXPERIMENTAL. Implemented as a "meta device" that wraps several devices. Requires flash attention (enabled automatically). Backend sampling falls back to CPU. Upstream: "performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise". Qwen2 is a supported architecture for it.

### The open question: TP on 32B failed

An attempt with `--split-mode tensor` on Qwen2.5-32B over RPC between the two Macs failed. No log was saved. The supervisor expects it to fit: each machine needs about `19.85 / 2` GB of weights plus overhead.

Candidate causes, to be checked from the log of a repeated run:
1. An operation not implemented in the meta backend for the Metal or RPC device. Symptom: `ggml op not implemented` or `unsupported mul_mat split states`.
2. The meta device cannot wrap an RPC device, or wraps it but every reduction goes through the requester. Symptom: load succeeds, generation is very slow or aborts.
3. Memory. Embedding and head replicated plus compute buffers push one Mac over its working set. Symptom: `failed to allocate` in `load_tensors` or in the context.
4. The run was started without `-fa on` or with a flag the mode rejects. Symptom: an error in the first 50 lines of stderr.

How to repeat with evidence: `SPLIT_MODE=tensor scripts/10_capacity.sh split <32B.gguf> <worker-ip> 1,1 P1`. The run folder keeps the command, stderr and the placement lines. The answer goes into experiments/RESULTS.md and into the shared doc.

## 3. Model format and naming

Goal: name a model the way Hugging Face does, and run it without hand work.

**Name.** `owner/model` plus an optional quantization tag. Example: `Qwen/Qwen2.5-32B-Instruct` with `Q4_K_M`. Resolution: the GGUF repo `bartowski/Qwen2.5-32B-Instruct-GGUF` or the official `ggml-org` repo. The mapping is a small table in the repo, not magic.

**Files.** Today: GGUF. One file, weights plus metadata, quantized. Loaded by llama.cpp directly. If ADR-007 chooses PyTorch: Hugging Face safetensors shards plus `config.json`, with the `weight_map` index to load only the needed shards.

**Manifest.** Written by the requester from the file header. Consumed by the planner.

```
name:            Qwen/Qwen2.5-32B-Instruct
format:          gguf
quant:           Q4_K_M
sha256:          <file hash>
arch:            qwen2
n_layer:         64
hidden_size:     5120
n_heads:         40
n_kv_heads:      8
vocab:           152064
bytes_per_layer: [ ... 64 values ... ]
bytes_embed:     ...
bytes_output:    ...
kv_bytes_per_token: 262144
split_constraints: { tp_divisors: [1, 2, 4, 8] }
```

**Rule.** Same file, same hash, on every machine. The requester refuses to run if a worker reports a different hash. The hash is recorded in every run folder.

**Supported class.** Dense decoder-only. Llama, Qwen, Gemma, Mistral, Phi families. A new checkpoint of a known family needs only a manifest. A new family needs an adapter and a correctness test.

## 4. Correctness and measurement

- Correctness: the split model must produce the same tokens as the single-machine run at temperature 0 on a fixed prompt, or logits within a declared tolerance. Tolerance is fixed before the run.
- Every compared number comes from the same tool, the same SHA, the same hash, with the placement taken from the measured run's log.
- Speed is reported as tokens per second with repeats. Memory as peak per machine. Network as measured ping and iperf3 for that run.
