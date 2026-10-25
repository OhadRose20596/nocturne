# Flow: a model arrives. What happens next

Draft for agreement with the supervisor (meeting of 2026-10-11, item "flow diagram"). Status: proposed. Short sentences by request.

## 1. The picture

```mermaid
flowchart TD
    A[User names a model<br/>e.g. Qwen/Qwen2.5-32B-Instruct] --> B[Resolve the model<br/>Hugging Face id to files + manifest]
    B --> C{Is the format supported?<br/>dense decoder-only, known family}
    C -- no --> X[Stop. Report why]
    C -- yes --> D[Read the manifest<br/>layers, hidden size, heads, bytes per layer, hash]
    D --> E[Ask the tracker: who is online<br/>memory, device, speed, network, uptime]
    E --> F[Planner<br/>memory feasibility per machine<br/>weights + context + KV + buffers + headroom]
    F --> G{Fits on one machine?}
    G -- yes --> H[Run locally. Or run replicas if several machines fit]
    G -- no --> I{Fits on N machines with PP?}
    I -- no --> J{Fits with TP or PP+TP?}
    J -- no --> X
    J -- yes --> K[Choose split and placement<br/>output layer on the requester]
    I -- yes --> K
    K --> L[Workers fetch their shard<br/>from cache or from the requester]
    L --> M[Run. Prefill, then decode token by token]
    M --> N{A worker left?}
    N -- yes --> O[Detect by heartbeat. Re-plan.<br/>Replay prompt + tokens so far on a replacement]
    O --> M
    N -- no --> P[Finish. Return the answer.<br/>Save the run record]
```

## 2. The steps in words

1. **Name.** The user gives a model name. Hugging Face style, `owner/model`. Optionally a quantization.
2. **Resolve.** The requester maps the name to files and a manifest. See BACKGROUND_AND_DESIGN.md section 3.
3. **Check.** Unsupported architecture or format: stop and say why.
4. **Discover.** The tracker returns online workers and their profile. Until the tracker exists, a static `peers.toml` (ADR-008).
5. **Plan.** The planner counts memory per machine: weights, context (input plus output), KV cache, runtime buffers, headroom. It picks the smallest legal setup. One machine, replicas, PP, TP, or PP plus TP. Rule learned so far: keep the output layer on the requester.
6. **Load.** Each worker gets its shard. From its local cache if it has it. Otherwise from the requester.
7. **Run.** Prefill the prompt. Decode one token at a time. Each token crosses each machine boundary once.
8. **Survive.** Heartbeats. A lost worker triggers re-plan and replay. Nothing is emitted twice.
9. **Finish.** Return the answer. Save the command, placement and timing in the run folder.

## 3. What exists today, per step

| Step | State on 2026-10-17 |
|---|---|
| 1, 2 | Manual. A GGUF file path in `.env`. No name resolution yet |
| 3 | Manual. llama.cpp accepts or rejects the file |
| 4 | Manual. IP addresses typed on the command line |
| 5 | Manual `-ts`. Memory policy P1 or P2 in `scripts/10_capacity.sh`. No planner code |
| 6 | Works. llama.cpp RPC ships weights. Worker cache `-c` works (130 cache hits seen) |
| 7 | Works for PP on two Macs. TP on 32B failed once, unrecorded |
| 8 | Nothing. A lost worker kills the run |
| 9 | Works. Run folders under `experiments/` |

## 4. Open points for the supervisor

- Is this the agreed flow? What is missing?
- Step 5: should the planner prefer the fewest machines, or the most reliable ones?
- Step 8: is restart from the beginning enough for a first version, or must we continue mid-answer?
