# torch_tp: the engine B spike (ADR-007, amendment 2)

Our own PyTorch tensor parallel engine for Qwen2 in bf16, with our own TCP transport.
Session 3 built the two rank engine (`serve`, `worker`, cell B1t of the bench).
Session 6 added hybrid placement: PP stages, each a TP group of any size with uneven shares.
Nothing here is evidence about llama.cpp, and nothing here picks the product engine (ADR-007 symmetric rule).

## Files

| file | what |
|---|---|
| `shard.py` | `SliceSpec`: a contiguous range of query heads and of MLP rows. KV heads follow from GQA; a KV head used by two ranks is held by both. `Shard` is the even special case (session 3). |
| `plan.py` | `plan(cfg, placement) -> list[RankSpec]`, pure. The frame formula `frames_per_step`. |
| `loader.py` | Loads one rank's slice with safetensors slicing; enforces a byte budget tensor by tensor. |
| `model.py` | Qwen2 blocks for a layer range and a slice; vocab parallel embedding and head (local argmax). |
| `transport.py` | Framed TCP links; per link counters, split into prefill and decode frames. |
| `hybrid.py` | N rank all-reduce, the rank runtime, the coordinator, link setup. |
| `launch.py` | Starts every rank (local or over ssh), runs the coordinator, stops everything. |
| `hybrid_run.py` | One recorded run: preflight, prompts, correctness against the R0 reference, frames, memory. |
| `engine.py`, `server.py`, `compare.py` | Session 3 two rank engine and the HTTP server (shared by `launch`). |

## Placement file

YAML. One file per placement, under `experiments/session6_hybrid/placements/`.

```yaml
model: qwen2.5-7b-bf16       # a model key of peers.yaml (torch_model), or a safetensors directory
coordinator: {host: m5}      # must be the host of rank 0 of the first stage; that process is the coordinator
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

Per rank fields: `host` (a peers.yaml host), `device` (`mps`, `cpu`, `cuda`), `port`, and optionally:
- `weight`: share of the group's bytes. Without weights, `budget_gb` sets the shares; without both, shares are equal.
- `budget_gb`: tensor bytes plus workspace allowed on this rank (runtime overhead is outside it). The planner rejects a placement with any rank over its budget, naming each one. The rank enforces it again while loading.
- `heads`: number of query heads, to force a split (for example a KV aligned one). Give it for every rank of the group or for none.

All ranks of one group must use the same device type. The planner rejects a mixed group, because the activations of its ranks would no longer be bitwise identical.

## How the planner splits a group

1. Query heads by share (largest remainder), or the `heads` override. Ranges are contiguous; zero heads is allowed.
2. Vocab rows (embedding and head) by share, in granules.
3. MLP granules fill each rank up to its share of the group's total bytes. So a rank with more attention gets fewer MLP rows.

A rank with no heads, no MLP rows and no vocab rows is rejected.
Bytes assume bf16 and the Qwen2 block. Workspace is an estimate (see `workspace_bytes`).

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

`launch` starts the ranks of other hosts over ssh with the `python` of that host in peers.yaml. Each host keeps the model files locally at the same path.

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
- **Host state:** caffeinate, AWDL and VPN state per host.
- **Memory per rank:** planned bytes, loaded tensor bytes, KV bytes, MPS current and driver allocations, the process footprint (idle, current, lifetime maximum) and peak RSS.
- **Frames:** predicted and measured decode frames per step, per link.
- **Correctness:** greedy tokens, graded accuracy, first-step logit differences against the R0 reference.
- **Background only:** ms per step (T1).

## Known limits

- **MPS memory is allocated in heaps (torch 2.14.1).** A tensor of 10 MiB or more opens a 1 GiB heap on Apple Silicon. That 1 GiB is resident. So an MPS rank that holds any tensor of 10 MiB or more costs at least about 1.2 GB of real memory, whatever its budget. `torch.mps.set_per_process_memory_fraction` counts heaps, so it refuses a budget below about 1.07 GB as soon as a second large tensor arrives. Budgets are therefore enforced on tensor bytes (loader and KV). The footprint is reported next to them.
- **Greedy only.** Decode runs one token at a time, with no micro batching.
- **No recovery.** A broken link ends the run.
