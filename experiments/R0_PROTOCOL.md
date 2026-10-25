# R0 — Fair baseline protocol

**Status (2026-10-25):** in force as the measurement protocol behind the O1 and O2 tables (GOAL_AND_OBJECTIVES.md v4). "R0" and "Table A/B/C" are internal names used by scripts/08_r0.sh and 09_r0_report.py.

**Purpose.** Replace the single uncontrolled data point (llama-cli 5.9 t/s vs llama-bench 20 t/s, I002/I003) with rows that are comparable: same tool, same tokens, same lengths, verified placement, recorded network path. R0 does not depend on the engine decision, on H4's wording, or on whether TP becomes mandatory (I005).

**Status:** proposed procedure, 2026-10-10. Edit freely; record deviations in the run log.

## 0. Before any measurement — once per machine

1. `scripts/07_preflight.sh <peer-ip>` on **both** machines. Attach both inventories. The pinned CLI/server use `--version`; bench/RPC use `--help`, since they do not implement `--version`. Resolve loader failures first. Passing binary checks alone is insufficient: verify model/hash, devices, flags and connectivity too.
2. Both machines on the **same llama.cpp SHA** (preflight prints it) and the **same GGUF sha256** (preflight prints it). Different SHA or hash → not comparable; stop.
3. Pin and write down `PP=256 TG=64 REPS=5` consistently. Record batch/ubatch, threads, split mode, resolved flash attention and KV types. The pinned `llama-bench` does not expose the CLI option `-c 2048`; record its actual context configuration. GPU rows use `-ngl 99`; CPU controls use `-ngl 0`.
4. Close other GPU users (browsers with GPU acceleration, Ollama, LM Studio). Plug in the laptop. Note thermal state if the machine is hot.

## 1. The instrument

`llama-bench` for every number (local and `--rpc`). It runs the same synthetic prefill/decode with the same token count on every configuration, repeats `REPS` times, and reports mean ± std. `scripts/05_bench.sh` wraps it and appends to `bench.csv` with host/backend/SHA/date columns.

`llama-cli` / `llama-server` are **not** instruments for R0; they are used once per configuration only to capture the placement log (step 3).

## 2. Network path — recorded for every remote row, at run time

Before each `rpc` block, on the requester:

```
scripts/06_net_baseline.sh <worker-ip>        # ping + iperf3 (worker runs: iperf3 -s)
tailscale status | grep <worker>              # if Tailscale is involved: shows direct vs relay
```

Table C gets **one row per configuration actually used**, with separate columns — do not collapse them:

| location | link | overlay | path | requester→worker endpoints | RTT avg/p95 ms | iperf3 Mbit/s |
|---|---|---|---|---|---|---|
| LAN / WAN | Ethernet / Wi-Fi | plain / Tailscale | direct / relay (DERP) | e.g. 192.168.1.10 → 192.168.1.20:50052 | | |

Minimum set for R0: **plain LAN, Ethernet** (cable or switch) and **plain LAN, Wi-Fi**. Tailscale rows are welcome but are a third configuration, not a substitute.

## 3. Placement verification — once per configuration

Capture the measured `llama-bench` invocation's own `--verbose` stderr and its exact command as primary placement evidence. A separate CLI/server run is only supplementary evidence, because options and placement can differ. Example supplementary smoke test:

```
"$BIN/llama-cli" -m "$MODEL" -ngl 99 --rpc <worker>:50052 -ts 1,1 -c 2048 -n 8 -p "ping" --verbose -st 2>&1 | tee experiments/logs/$(date -u +%Y%m%d-%H%M%S)_$(hostname -s)_placement_rpc_1,1.log
```

For the mixed `1,1` row, the measured-run log must show both intended devices receiving tensors, e.g. lines like
`load_tensors: RPC[<ip>:50052] model buffer size = … MiB` **and** `load_tensors: Metal_Mapped/CUDA0 model buffer size = … MiB`, and `offloaded N/M layers to GPU`. Copy those lines into the results next to the row. A `1,1` split whose log shows one device holding everything is a failed configuration, not a measurement.

## 4. The rows

Worker: `scripts/03_start_worker.sh` (binds the LAN or Tailscale IP, never a public one). Requester runs, in this order, with the same `PP/TG/REPS`:

| Row | Requester | Worker | Command (requester) |
|---|---|---|---|
| A1 | Mac | — | `scripts/05_bench.sh` (gpu_all_layers + cpu_only) |
| A2 | CUDA box | — | `scripts/05_bench.sh` |
| B1 | Mac | CUDA box | `LINK=… OVERLAY=… LOCATION=… scripts/08_r0.sh rpc <cuda-ip>` → splits 1,0 / 1,1 / 0,1 |
| B2 | CUDA box | Mac | `LINK=… OVERLAY=… LOCATION=… scripts/08_r0.sh rpc <mac-ip>` → splits 1,0 / 1,1 / 0,1 |
| B1′/B2′ | same | same | repeat over the second link (Wi-Fi if B1/B2 were Ethernet, or vice-versa) |

Notes on the splits — **order is remote first**, verified 2026-10-10 from `llama-cli --verbose` logs (`layer N assigned to device RPC0/MTL0`) and from `src/llama.cpp` ("add RPC servers at the front of the list"): `1,0` = all layers on the remote worker (one-hop pipeline; requester keeps the input embedding on CPU and sends tokens/receives logits); `0,1` = all layers local **through the RPC code path** — isolates RPC overhead from network transfer, should be close to A1/A2; `1,1` = two stages (layers 0–14 remote, 15–28 local for a 28-layer model). `llama-cli` takes `a,b`; `llama-bench` takes `a/b` and treats `,` as a separator between *sweep configurations* — `08_r0.sh` converts, and the report flags any `bench.json` that contains more than one configuration.

## 5. What goes into RESULTS.md

For each row: the markdown table `llama-bench` prints, preceded by `host= backend= commit= date=`, the link/overlay/path of Table C it was measured under, and the placement-log lines from step 3. Raw rows are already in `bench.csv`.

## 6. Acceptance for R0 (from GOAL_AND_OBJECTIVES O1/O3)

- A1 and A2 present (GPU and CPU rows) with run-to-run std < 5 % of mean.
- B1 and B2 present for all three splits, over at least two link types, each with a placement log and a Table C row.
- Every row carries SHA, model hash, PP/TG/REPS, and the network columns.

## 7. What R0 will and will not tell us

*Framing (team decision T1, 2026-10-10): R0's numbers are read as **efficiency of the split**, not as latency to be reduced. The local-vs-remote gap tells us the network share of per-token time under the given link; `0,1` tells us the protocol's own overhead; both feed `E(N)` and the planner. The link is a recorded condition.*

It **will** give pp/tg under known conditions and observed sensitivity to placement and link, including local-only versus local placement with a registered RPC endpoint. It cannot retroactively identify the cause of September's 5.9 t/s without missing metadata. Local-vs-remote differences include device-speed changes and are not by themselves a pure network-hop cost.

It **will not** give: anything about TP, about PyTorch, or about prefill of long prompts (`PP=256` only). Those are R1–R3.

## 8. Recorder checks required before R0 (I007)

The current `05_bench.sh` runs the benchmark twice: displayed Markdown and stored CSV are separate runs. Its metadata label can contain commas (`split_1,1`) but is not CSV-escaped, so RPC records may be malformed. Repair and verify the recorder before final R0 collection. Preserve each invocation's exact command, raw output and verbose stderr. These are code-inspection findings, not new performance measurements.

**Resolution (2026-10-10):** R0 uses a new recorder, `scripts/08_r0.sh`, instead of `05_bench.sh`. It runs `llama-bench` **once** per row with `-o json`, writes no CSV at run time, and keeps per run `cmd.txt`, `bench.json`, `stderr.log`, `placement.log` / `net.txt` and `manifest.json` under `experiments/r0/runs/<run_id>/`. `scripts/09_r0_report.py` builds Tables A/B/C, a properly quoted `bench_r0.csv` and the §6 acceptance checks from those files only. Step-by-step instructions per machine: `docs/he/RUNBOOK_R0.md`. Both scripts were exercised end to end against fake `llama-bench`/`llama-cli` binaries in a Linux sandbox; they have not yet run on the real machines. `05_bench.sh` is left untouched for the historical `bench.csv`.
