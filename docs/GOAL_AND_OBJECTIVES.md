# Project Goal and Objectives

**Nocturne: Distributed LLM Inference over Volunteer Consumer GPUs**
B.Sc. Final Project, Electrical and Computer Engineering, Ben-Gurion University of the Negev

**Revision v4 (proposed, 2026-10-17).** Rewritten after the supervisor meeting of 2026-10-11 (docs/he/MEETING_CHEN_2026-10-11.md). The goal states only the peer-to-peer idea. Implementation choices live in ARCHITECTURE.md and DECISIONS.md. Objectives are listed first and kept short. Team decisions T1 and T2 still apply. Previous revision v3.2 is kept in docs/archive/.

## Goal

Nocturne is a peer-to-peer system. Volunteer home computers run one large language model together, over the internet. The model may be larger than any single computer. The project shows that this works, how far it scales, and how to split the work well.

## Objectives at a glance

| # | Objective | One line |
|---|---|---|
| O1 | Single machine, single device | Run a model on one machine with llama.cpp, on one compute device. This is the baseline. |
| O2 | Split by layers across machines | Pipeline parallelism: each machine holds a range of layers. Two machines first. |
| O3 | Scale out | More than two machines, over the internet, with generic models in a Hugging Face style format. |
| O4 | Performance study | Several setups. Where does each saturate. PP vs TP. Match model to processor. Basic trade-offs. |
| O5 | Peer-to-peer application | Wrap client, server and tracker into an app the public can use. It must survive node loss (liveness). |

Supporting documents: FLOW.md (what happens when a model arrives), PLATFORM.md (hardware, software, GPU vs CPU, Metal vs CUDA review), BACKGROUND_AND_DESIGN.md (formal definitions, PP vs TP, model format).

## Framing

Team decision T1 (2026-10-10): latency is not a target. The network is a given bottleneck. We do not try to fix it.

Team decision T2 (2026-10-11): the claim is the ability to run a huge model on several home computers over the internet and succeed. It runs, it finishes, it gives a good answer. Priorities: capability and quality, then reliability, then an automatic planner. "A task before sleep, an answer in the morning" is one illustration. It is not a deadline.

## Objectives in detail

### O1. Single machine, single device
Run one model on one machine with llama.cpp. Force the run onto one chosen device: the CPU, or the machine's GPU. Record prefill and decode speed, memory and the exact command.
Acceptance: a table per machine with the same model and prompt on CPU and on GPU. Spread between repeats under 5 percent.
Status: done on both Macs (Table A). Not done on the NVIDIA machine.

### O2. Split by layers across machines
Pipeline parallelism. The model is cut into contiguous layer ranges. Each machine holds one range and its context memory. Activations pass from one range to the next. One machine also holds the input and output layers.
Acceptance: the same model runs on two machines at several splits. The run log proves which layers landed where. A model that fails on each machine alone runs on both together.
Status: done on two Macs on the home LAN. Qwen2.5-32B Q4 fails on each Mac alone with full offload and runs on both at about 4 tokens per second. See experiments/RESULTS.md.

### O3. Scale out
Three parts.
1. More than two machines, with different GPUs (Apple Metal, NVIDIA CUDA).
2. Over the internet. Machines in different homes, on a wider network. Not only one Wi-Fi.
3. Generic models. Any dense decoder-only model in a standard format, taken from Hugging Face. One model format and one manifest. See BACKGROUND_AND_DESIGN.md section 3.
Acceptance: a run with three machines, at least one outside the home network, on a model chosen by name from Hugging Face. The output is correct and complete.
Status: not started. The NVIDIA machine has not run yet. Tailscale exists but no cross-home run has been done.

### O4. Performance study
Questions, each answered by measurement on our machines.
1. Capacity: which model size fits on N machines under a declared memory policy. Weights, context, KV cache, buffers and headroom are all counted.
2. Saturation: where does each setup stop scaling. Throughput at 1, 4 and 8 concurrent requests. Split vs independent replicas.
3. Work split: which layer assignment works best. Where to put the output layer. How much each machine computes and waits.
4. PP vs TP: run both on the same model and machines. Record memory per machine, bytes per token, speed and correctness. Open question from the meeting: TP failed on 32B in an unrecorded attempt. The supervisor expects it to fit, since each machine needs about 1/N of the weights plus overhead. Repeat with logs and explain the result.
5. Match model to processor: which model and quantization suit each machine. GPU vs CPU.
6. Trade-offs: memory vs speed, more machines vs more network, PP vs TP vs replicas.
Acceptance: one table per question. Every number links to a run folder. At least 10 repeats where speed is compared. The result feeds the planner rules in O5.
Status: pilots exist for 1, 2, 3 (two Macs, 7B and 32B). Nothing on 4, 5 with CUDA, or 6 beyond the pilots.

### O5. Peer-to-peer application
Wrap the pieces into one application a volunteer can install and run.
1. A worker that joins and offers its GPU.
2. A tracker that answers who is online.
3. A requester that picks machines, splits the model and runs the job.
4. Liveness: a worker that sleeps, drops or leaves must not kill the job. The requester detects it, re-plans and continues. First version: replay the prompt and the tokens so far on a replacement.
5. An automatic planner inside the requester. It chooses a legal, memory-feasible split from reported memory, speed, network and availability.
Acceptance: three workers announce; a killed worker disappears from the list within a bounded time; a job completes with one injected failure during generation; the planner's choice is logged with its reasons.
Status: not started. Scripts exist for manual runs. Static peer list first (ADR-008). Engine choice is open (ADR-007).

## Hypotheses

- H1. Under a slow network, per-token cost of a split is set by what crosses the boundary. Bytes per token matter. The output layer ships about 608 KB of logits per token when remote and about 14 KB of hidden state when local. Keep the output layer on the requester.
- H2. A model that fits on no single machine runs on N machines. Adding machines gives memory, not single-request speed. For a model that fits on each machine, independent replicas beat a split in throughput.
- H3. Worker-side weight caching cuts the second start-up by more than 10 times.
- H4. Placing layers on machines with higher past availability lowers expected time to answer once departures are common. Secondary.
- H5. TP over a slow network is slower per token than PP. Its memory per machine may still be lower. To be measured once per link type.

## Non-goals

Training or fine-tuning. Payment or incentives. Tracker-less discovery (DHT). Cryptographic verification of remote work. Security beyond a private mesh. Custom GPU kernels. Mixture-of-experts and multimodal models.

## Schedule

| Period | Work |
|---|---|
| 17 Oct to 31 Oct 2026 | Git and docs per the meeting. Recollect Table B. TP on 32B with logs. Flow diagram agreed with the supervisor. |
| Nov 2026 | O3 part 1 and 2: NVIDIA machine joins; 32B over Tailscale between homes. Quality task set. ADR-007 decision by 15 Nov. |
| Dec 2026 to Jan 2027 | O5 parts 4 and 5: failure detection, replay, planner v1. O4 tables. |
| Feb to Apr 2027 | O3 part 3 and O5 parts 1 to 3: generic models, tracker, worker, installable app. |
| May to Jun 2027 | Thesis and defense. |
