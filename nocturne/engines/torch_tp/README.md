# torch_tp: the engine B spike (ADR-007, amendment 2)

Our own PyTorch tensor parallel engine for Qwen2, Llama and Qwen3 (bf16 safetensors, or quantized GGUF kept quantized on the device), with our own TCP transport.
Session 3 built the two rank engine (`serve`, `worker`, cell B1t of the bench).
Session 6 added hybrid placement: PP stages, each a TP group of any size with uneven shares.
Session 10 added: any host as coordinator, weights shipped by the coordinator, clock alignment, and a recorder with a timeline.
The whole flow of a run is in `docs/ENGINE_DESIGN.md`. Where every number comes from is in `docs/MEASUREMENT.md`.
Nothing here is evidence about llama.cpp, and nothing here picks the product engine (ADR-007 symmetric rule).

## Files

| file | what |
|---|---|
| `arch.py` | The architecture table (qwen2, llama, qwen3) and the GGUF to engine tensor name map. One table, not one class per model. |
| `config.py` | `ModelConfig` from a Hugging Face `config.json` or from GGUF metadata. Llama 3 rope factors. |
| `gguf_source.py` | GGUF as a weight source: header, name map, raw quantized blocks of a slice, un-permute of llama Q and K. `TensorSource` is the interface the loader uses. |
| `quant.py` | `QuantLinear`: raw blocks on the device, expanded to bf16 in row chunks just before the matmul. Torch ops only. |
| `shard.py` | `SliceSpec`: a contiguous range of query heads and of MLP rows. KV heads follow from GQA; a KV head used by two ranks is held by both. `Shard` is the even special case (session 3). |
| `plan.py` | `plan(cfg, placement) -> list[RankSpec]`, pure. The frame formula `frames_per_step`. |
| `loader.py` | Loads one rank's slice with safetensors slicing; enforces a byte budget tensor by tensor. |
| `model.py` | Qwen2 blocks for a layer range and a slice; vocab parallel embedding and head (local argmax). |
| `transport.py` | Framed TCP links; per link counters, split into prefill and decode frames. |
| `hybrid.py` | N rank all-reduce, the rank runtime, the coordinator, link setup. |
| `launch.py` | Starts every rank (local or over ssh), runs the coordinator, stops everything. |
| `hybrid_run.py` | One recorded run, as the operator: job, preflight, start, collect, correctness against the R0 reference, record. |
| `operate.py` | The operator: starts every rank and the coordinator on their hosts (ssh or local), collects their run files. Never loads the model. |
| `coordinate.py` | The coordinator of a recorded run, on the host the placement names: plans, links, ships, runs the prompts. |
| `shipping.py` | Weight shipping: manifest per rank, chunks with CRC32, the slice cache by sha256, `CacheSource` for the loader. |
| `clock.py` | Clock offset of every rank on the control link, at start and end. |
| `recorder.py` | Per rank spans (compute, copy, send, waits) and the host sampler. |
| `timeline.py`, `timeline_page.html` | Aligns the spans and writes `trace.json.gz`, `breakdown.csv`, `timeline.html`. |
| `engine.py`, `server.py`, `compare.py` | Session 3 two rank engine and the HTTP server (shared by `launch`). |

## Weight sources and architectures (engine GGUF, 2026-11-21)

`model:` in a placement resolves to a safetensors directory or to a `.gguf` file. A key of `peers.yaml` uses `torch_model`, or its `path` when that is a `.gguf` (the llama.cpp file works for this engine too). A bare path works as well. The model hash in a record is the sha256 of the weight files of a directory, or of the one GGUF file.

| | qwen2 | llama | qwen3 |
|---|---|---|---|
| q, k, v bias | yes | no | no |
| per head q_norm, k_norm | no | no | yes |
| rope | plain | plain, or Llama 3 scaling (per frequency factors) | plain |
| head size | hidden / heads | `head_dim` of the config, else hidden / heads | `attention.key_length`, not hidden / heads |
| GGUF Q and K rows | as in HF | permuted by the converter, the loader puts them back | as in HF |

Quantization types: F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K. Any other type is refused with a message that lists the supported ones. 1-D tensors (norms, biases) are held in bf16.
`QuantLinear` keeps the stored blocks. The matmul expands `CHUNK_ELEMS` (16 M) elements at a time to bf16, so only a chunk is dense. This costs time per token (every step expands every matrix), which is background (T1), and it costs transient memory, which the planner counts.

**Block alignment rule.** A row split layer (`o_proj`, `down_proj`, cut by input columns) may be cut only at block boundaries: 32 columns for Q8_0, Q4_0, Q5_0 and 256 for the K quants. Column split layers (q, k, v, gate, up) can be cut at any row, because a block lies inside one row. The planner moves heads in units that respect the rule (for head size 128 and a 256 block: two heads), uses an MLP granule that is a multiple of the `down_proj` block, and rejects a `heads:` override that breaks it, naming the rule. Layers of one file may mix types (Q4_K_M uses Q6_K for half of `ffn_down`); the planner counts each layer with its own type.

**Byte accounting.** Planned weights are the stored block bytes (equal to the loaded bytes on every rank in the tests). The workspace adds one dequantized chunk (bf16 result plus fp32 working copies). On MPS a tensor of 10 MiB or more lands in a 1 GiB heap: the plan reports `mps_floor` (1 GiB) and `footprint` (the larger of total and the floor), not added to `total`.

**Tokenizer and EOS** come from the GGUF (transformers builds the tokenizer from the file; `AutoTokenizer.from_pretrained(dir, gguf_file=...)`). On the four models checked the ids equal those of the Hugging Face tokenizer on all 43 eval prompts. The EOS set from the GGUF is smaller than the one of the HF `generation_config.json` (Qwen: only `<|im_end|>`; Llama 3.2: only `<|eot_id|>`).

**Tests** compare against implementations we did not write: gguf-py's numpy dequantizer, transformers' GGUF loader and its own rotary embedding, and files made by llama.cpp's converter and quantizer. See `experiments/engine_gguf/VERIFY.md`.

## Placement file

YAML. One file per placement, under `experiments/session6_hybrid/placements/`.

```yaml
model: qwen2.5-7b-bf16       # a model key of peers.yaml (torch_model), or a safetensors directory
coordinator: {host: m5}      # must be the host of rank 0 of the first stage; any joined host, macOS or Linux
reduce_dtype: float32        # float32 (default) or bfloat16: where the partials are summed
max_seq_len: 4096            # KV cache length per rank
prefill_max: 256             # longest prompt; sizes the workspace estimate; longer prompts are refused
granule: 128                 # MLP rows and vocab rows are split in granules of this many rows
stages:                      # in order; layer ranges are half open, contiguous, and cover the model
  - name: s0
    layers: [0, 12]
    embed: true              # first stage only (default: the first stage)
    ranks:
      - {host: m5, device: mps, port: 50070}
  - name: s1
    layers: [12, 14]
    ranks:                   # a TP group; any size, not only powers of two
      - {host: m5, device: mps, port: 50071, budget_gb: 0.40}
      - {host: air, device: mps, port: 50070, weight: 2, heads: 14}
  - name: head               # a stage with no layers that holds only the head is allowed
    lm_head: true            # last stage only (default: the last stage)
    ranks:
      - {host: air, device: mps, port: 50071}
```

Per rank fields: `host` (a peers.yaml host), `device` (`mps`, `cpu`, `cuda`; left out, it follows the host's `backend` in peers.yaml), `port`, and optionally:
- `weight`: share of the group's bytes. Without weights, `budget_gb` sets the shares; without both, shares are equal.
- `budget_gb`: tensor bytes plus workspace allowed on this rank (runtime overhead is outside it). The planner rejects a placement with any rank over its budget, naming each one. The rank enforces it again while loading.
- `heads`: number of query heads, to force a split (for example a KV aligned one). Give it for every rank of the group or for none.

All ranks of one group must use the same device type. The planner rejects a mixed group, because the activations of its ranks would no longer be bitwise identical.

## How the planner splits a group

1. Query heads by share (largest remainder), or the `heads` override. Ranges are contiguous; zero heads is allowed.
2. Vocab rows (embedding and head) by share, in granules.
3. MLP granules fill each rank up to its share of the group's total bytes. So a rank with more attention gets fewer MLP rows.

A rank with no heads, no MLP rows and no vocab rows is rejected.
Bytes assume bf16 for safetensors and the stored block bytes for GGUF. Workspace is an estimate (see `workspace_bytes`).

## Mixed devices in one group (session 7)

By default the planner rejects a TP group whose ranks use different device types. The placement key `allow_mixed_devices: true` lifts that rule for the whole placement.

Why it is safe to allow:
- `GroupComm` moves every partial sum to the host and adds them there, in a fixed rank order, in fp32 by default. Every rank of the group receives the same bytes.
- The residual stream is then updated by the same elementwise add on each device, so the shared state does not depend on the device type.
- What can differ is the arithmetic inside one rank's slice (attention and MLP matmuls). That is ordinary kernel rounding, the same class as MPS against CPU in P9.

It is not proof. The check is a run against the references of every device type in the placement (`hybrid_run` does this by default). A Mac plus CUDA TP group sends the whole hidden state through the host twice per layer, so it is meant for feasibility, not for speed (T1, T4).

This is an option inside engine B, not a reversal of an ADR.

## Running

```bash
python -m nocturne.engines.torch_tp plan --placement experiments/session6_hybrid/placements/P7_2mac.yaml
```

```bash
python -m nocturne.engines.torch_tp.hybrid_run --placement experiments/session6_hybrid/placements/P7_2mac.yaml --run-id P7_2mac_qwen2.5-7b-bf16_r1
```

```bash
python -m nocturne.engines.torch_tp launch --placement experiments/session6_hybrid/placements/P7_2mac.yaml --port 8092
```

`hybrid_run --session-dir DIR` writes runs and the model hash cache under DIR (default `experiments/session6_hybrid`); references are looked up in DIR first, then in session 6.

`launch` (the HTTP server) starts the ranks of other hosts over ssh with the `python` of that host in peers.yaml, and each host keeps the model files locally. Run it on the coordinator host.

## Operator, coordinator, shipped weights (session 10)

`hybrid_run` is the operator. It runs on the machine that has `peers.yaml` and ssh to every host. It starts one `rank` process per rank and one `coordinate` process on the coordinator host, then copies every host's run files into the run folder.
The coordinator host is the one the placement names. It needs the model file, `transformers` and `gguf`. It needs no ssh and no peers.yaml.

| option | default | what |
|---|---|---|
| `--ship-weights` / `--no-ship-weights` | on | The coordinator sends every rank its slices. Only the coordinator host holds the model. Off: every host reads its own copy, as before session 10 |
| `--cache-dir DIR` | `~/.nocturne/cache` on each host | Where a rank keeps shipped slices: `<dir>/<model sha256>/<slice sha256>` |
| `--timeline` | off | Record spans and samples, write the timeline files |
| `--verify-cache` | off | Every rank hashes its cached slices before it trusts them (one read of the cached bytes) |
| `--stall-timeout S` | 300 | End the run when the coordinator process gives no sign of life for S seconds (it prints one every 20 s). 0 waits for ever. Not a time limit on the run |

peers.yaml keys added: `peers.<host>.models.<key>` (the model path on that host), `peers.<host>.run_root` (where that host writes its run files; default `~/.nocturne/runs`).
The slice cache is never cleaned. Delete `~/.nocturne/cache/<model sha256>` by hand when a model is no longer used.

## Network exposure

Each rank opens one TCP port.
- It listens on `127.0.0.1` when a peer on its own host connects to it.
- It listens on the host's `bind` address from peers.yaml when a peer on another host connects to it.
- Never on 0.0.0.0 (CLAUDE.md rule 7).

The ports accept any peer without authentication. Security is deferred by ADR-009, not forgotten. Use these ports only on a network you control.

## Records

`hybrid_run` writes its own schema (`session6_hybrid/1`), not the bench schema. The bench schema has one requester and a list of workers, and a placement of many ranks per host does not fit it.
The record holds:
- **Run identity:** git SHA, the torch version per rank, the placement file and its sha256, the model sha256 per host.
- **Host state:** OS, caffeinate, AWDL and VPN state per host ("n/a" on Linux), and `nvidia-smi` name, memory and driver for hosts with a CUDA rank.
- **Memory per rank (CUDA ranks add `cuda_name`, `vram_total_bytes`, `vram_free_bytes`, `max_allocated_bytes`):** planned bytes, loaded tensor bytes, KV bytes, MPS current and driver allocations, the process footprint (idle, current, lifetime maximum) and peak RSS.
- **Frames:** predicted and measured decode frames per step, per link.
- **Clocks (session 10):** per host the offset to one NTP server and the OS tool's report; per rank the offset to the coordinator at start and end, the drift, and the gap between the two methods.
- **Shipping (session 10):** the model sha256, and per rank the manifest bytes, shipped bytes, bytes from the cache, the time and the number of hash checks.
- **Timeline (session 10, with `--timeline`):** per rank the mean ms per decode step by kind, bytes per link, host samples.
- **Correctness:** greedy tokens, graded accuracy, first-step logit differences against the R0 reference.
- **Background only:** ms per step (T1).

## Known limits

- **MPS memory is allocated in heaps (torch 2.14.1).** A tensor of 10 MiB or more opens a 1 GiB heap on Apple Silicon. That 1 GiB is resident. So an MPS rank that holds any tensor of 10 MiB or more costs at least about 1.2 GB of real memory, whatever its budget. `torch.mps.set_per_process_memory_fraction` counts heaps, so it refuses a budget below about 1.07 GB as soon as a second large tensor arrives. Budgets are therefore enforced on tensor bytes (loader and KV). The footprint is reported next to them.
- **Greedy only.** Decode runs one token at a time, with no micro batching.
- **No recovery.** A broken link ends the run.
