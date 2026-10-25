# Setup and Run Guide (no AI assistant required)

Step-by-step instructions to install the environment for objectives O1 (single machine) and O2 (two machines, manual split) on any participating machine. Written for a team member at a fresh computer with a terminal. Expected time: 30 to 60 min per machine, most of it compiling and downloading.

Updated 2026-10-25. In this guide "PoC-0" means O1 and "PoC-1" means O2 (GOAL_AND_OBJECTIVES.md). Pinned llama.cpp revision: `1e7bcf3`. The scripts referenced here are `scripts/00` to `06`; the recording scripts for experiments (`07` to `14`) are described in docs/he/RUNBOOK_R0.md and docs/he/GUIDE_R0_TO_R3.md. Connecting machines across homes: docs/he/CONNECTIVITY.md.

If something fails, go to §7 Troubleshooting before asking.

---

## 0. What you will end up with

- llama.cpp built from source with the right GPU backend (**Metal** on Mac, **CUDA** on NVIDIA, CPU otherwise) and RPC support.
- A model file (GGUF) identical on every machine.
- Proof that inference runs on the device *you* chose (GPU / CPU / a specific GPU).
- A benchmark table for this machine → `experiments/bench.csv` and `experiments/RESULTS.md`.
- (Two machines) A run where one machine drives the other's GPU over the network.

---

## 1. Prerequisites by platform

### 1a. macOS (Apple Silicon)

```bash
xcode-select --install                      # compilers + Metal toolchain (skip if already done)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # if no brew
brew install cmake git iperf3
```
Check: `cmake --version` prints ≥ 3.14, `git --version` works.

### 1b. Ubuntu / Debian Linux with NVIDIA GPU

```bash
sudo apt update && sudo apt install -y build-essential cmake git iperf3
nvidia-smi                                  # must list your GPU; if not, install the NVIDIA driver first
nvcc --version                              # CUDA toolkit; if missing:
```
Install the CUDA toolkit from https://developer.nvidia.com/cuda-downloads (choose your distro, "deb (network)"). After install add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Open a new terminal, confirm `nvcc --version` works.

### 1c. Windows with NVIDIA GPU — use WSL2

Native Windows builds work but our scripts are bash. WSL2 gives you Ubuntu with GPU access:

1. PowerShell as Administrator: `wsl --install -d Ubuntu` → reboot → set a Linux username/password.
2. Install the **NVIDIA driver for Windows** (normal GeForce/Studio driver, ≥ 5xx). Do **not** install a Linux driver inside WSL.
3. Inside Ubuntu (WSL): `nvidia-smi` must show the GPU. Then follow **1b** for the CUDA toolkit, but use the *WSL-Ubuntu* selection on the CUDA download page.
4. Networking note: WSL2 has its own virtual IP. For PoC-1 the worker must be reachable from the other machine — see §7 "WSL2 port not reachable".

### 1d. Linux/Windows without NVIDIA (AMD/Intel GPU or CPU only)

Follow 1b without the CUDA steps. Build with `BACKEND=cpu` (or `BACKEND=vulkan` if you have Vulkan drivers: `sudo apt install libvulkan-dev glslc`). CPU-only machines are still useful as workers for small models and for tracker/network tests.

---

## 2. Get the repository

```bash
cd ~
git clone <REPO_URL> nocturne          # or unzip nocturne.zip
cd nocturne
cp env.example .env
```

**Mac note:** if the repo lives inside iCloud Drive, set `LLAMA_DIR=$HOME/dev/llama.cpp` in `.env` so the multi-GB build does not sync to the cloud.

---

## 3. Build llama.cpp

> **Build outside the repo if the repo lives in iCloud/Dropbox/OneDrive.** A build inside `third_party/` gets synced (GBs) and, worse, its binaries carry an *absolute* rpath to the directory they were built in — moved to another Mac or user they fail with `dyld: Library not loaded: @rpath/...` (see experiments/RESULTS.md, E13). Recommended on every machine:
>
> ```bash
> git clone https://github.com/ggml-org/llama.cpp ~/dev/llama.cpp && cd ~/dev/llama.cpp && git checkout <same SHA as the others>
> cmake -B build -DGGML_RPC=ON -DGGML_METAL=ON -DCMAKE_BUILD_RPATH='@executable_path'     # macOS
> cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON  -DCMAKE_BUILD_RPATH='$ORIGIN'              # Linux + NVIDIA
> cmake --build build --config Release -j
> ```
>
> `scripts/common.sh` picks `~/dev/llama.cpp` automatically when it has a build; any other location goes in `~/.nocturne.env` as `LLAMA_DIR=...` (per machine, outside the repo — never in the shared `.env`). Verify with `scripts/07_preflight.sh`.

```bash
bash scripts/00_build_llamacpp.sh
```
Auto-detects Metal / CUDA / CPU. Force with `BACKEND=cuda bash scripts/00_build_llamacpp.sh`. Takes 5–20 min.

Success looks like:
```
[nocturne] built. binaries:
ggml-rpc-server
llama-bench
llama-cli
llama-server
```
Record the printed line (`host=… backend=… commit=…`) in `experiments/RESULTS.md` → "Build log".

Then:
```bash
bash scripts/01_list_devices.sh
```
Expected for the inspected llama.cpp revision: `MTL0: Apple M…` on a Mac; `CUDA0: NVIDIA GeForce …` on an NVIDIA box (plus `CUDA1` if two GPUs). **Use the exact device identifier printed by your build** in every command below. If no GPU is listed, investigate backend availability → §7.

---

## 4. Get the model

Everyone must use **byte-identical files**. Download from Hugging Face:

| Purpose | File | Size |
|---|---|---|
| Plumbing test (fast) | `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | ~1.1 GB |
| **Standard measurement model** | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | ~4.7 GB |
| Hebrew-quality reference | `gemma-2-9b-it-Q4_K_M.gguf` | ~5.8 GB |

Install the Hugging Face CLI once. On **macOS**, use Homebrew from §1a:

```bash
brew install hf
hf --help
```

On **Linux/WSL**, use Python 3.11+ in a virtual environment (Ubuntu/Debian may need `sudo apt install python3-venv` first):

```bash
python3 -m venv ~/.venvs/nocturne-hf
source ~/.venvs/nocturne-hf/bin/activate
python3 -m pip install --upgrade huggingface_hub
hf --help
```

Activate that environment again in each new terminal where you use its `hf` command.

Then download on either platform:

```bash
mkdir -p "$HOME/models"
hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir "$HOME/models"
```

Copy commands directly from these code blocks: underscores need no backslashes, and `\~` does not expand to your home directory. If `pip` or `hf` is not found, complete the platform-specific installation above. The `hf` command is included in `huggingface_hub`; no `[cli]` extra is needed.

Alternatively, download the Q4_K_M file in a browser from https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/tree/main and save it in `~/models`.
Verify identical copies across machines: `shasum -a 256 ~/models/*.gguf` (Mac) / `sha256sum` (Linux). Put the hash in RESULTS.md once.

Set the path in `.env`:
```
MODEL=/Users/<you>/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf     # Mac
MODEL=/home/<you>/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf      # Linux/WSL
```

Memory check: a Q4_K_M 7B model needs ~5 GB of GPU memory plus context. 8 GB VRAM is enough for 7B; 24 GB unified memory on a Mac handles 9B comfortably. If the model does not fit, use the 1.5B file.

---

## 5. PoC-0 — run on a device *you* choose

**All `scripts/…` commands are run from the repository root** (the folder containing `CLAUDE.md` and `scripts/`). `bash: scripts/02_run_local.sh: No such file or directory` means you are in another directory — `cd` to the repo first. Tip: `alias noct='cd "<path-to-repo>"'` in `~/.zshrc` / `~/.bashrc`.

### 5a. Smoke test (does it generate?)
```bash
bash scripts/02_run_local.sh MTL0       # Mac GPU; use the exact name from 01_list_devices.sh
bash scripts/02_run_local.sh CUDA0      # NVIDIA GPU (CUDA1 for the second card)
bash scripts/02_run_local.sh none       # CPU only — should be visibly slower
```
You should see a three-sentence answer about BitTorrent, then a timing summary. In the log above the answer look for a line like `load_tensors: offloaded 29/29 layers to GPU` (GPU run) or `offloaded 0/29` (CPU run) — that is the proof the device choice took effect.

### 5b. Measurement (the numbers that go in the thesis)
```bash
bash scripts/05_bench.sh                # GPU all-layers, then CPU-only; 3 reps each
```
Prints two markdown tables with `pp256` (prefill tok/s) and `tg64` (generation tok/s) as mean ± std, and appends rows to `experiments/bench.csv`. Paste the tables into `experiments/RESULTS.md` Table A with the host/commit line the script prints.

On a two-GPU machine also run `bash scripts/05_bench.sh CUDA1`.

Optional heavier run for the final tables: `PP=512 TG=128 REPS=5 bash scripts/05_bench.sh`.

---

## 6. PoC-1 — two machines, one drives the other's GPU

Roles: the **requester** holds the prompt and runs `llama-cli`; the **worker** exposes its GPU with `ggml-rpc-server`. Both need the same model file at the path in their own `.env`. Both on the same LAN (or the same Tailscale network).

### 6a. Network baseline (once per pair)
On the worker: `iperf3 -s`. On the requester:
```bash
bash scripts/06_net_baseline.sh <WORKER_IP>
```
Copy the printed row into RESULTS.md Table C. Ethernet ≈ 0.3–1 ms RTT; Wi-Fi 2–10 ms. This number explains everything that follows.

### 6b. Start the worker
```bash
bash scripts/03_start_worker.sh CUDA0        # or MTL0 if listed on the Mac; omit to expose all devices
```
It prints `endpoint : 192.168.x.y:50052` and `on the requester use: --rpc 192.168.x.y:50052`. Leave this terminal open. First load will populate a weight cache under `~/.cache/llama.cpp/rpc`; later loads are fast.

If the machine has a firewall, allow TCP 50052 from the LAN: macOS → System Settings → Network → Firewall → allow `ggml-rpc-server`; Ubuntu → `sudo ufw allow from 192.168.0.0/16 to any port 50052`.

### 6c. Requester: smoke test at three splits
```bash
bash scripts/04_run_two_machines.sh <WORKER_IP> 0,1    # all layers remote
bash scripts/04_run_two_machines.sh <WORKER_IP> 1,1    # half local, half remote
bash scripts/04_run_two_machines.sh <WORKER_IP> 1,0    # all local (control)
```
Each run should produce the same answer text (temperature 0). The `0,1` run proves the remote GPU did the work — the worker terminal will show activity and the requester log will mention `RPC[…]` devices.

### 6d. Requester: measurement
```bash
bash scripts/05_bench.sh rpc <WORKER_IP>        # runs splits 0,1 / 1,1 / 1,0 automatically
```
Paste tables into RESULTS.md Table B. **Then swap roles** (Mac as worker, NVIDIA as requester) and repeat 6b–6d. The asymmetry between directions is itself a result.

---

## 7. Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `invalid device: Metal0` (or another device name) | Run `bash scripts/01_list_devices.sh` on the machine doing the inference, then pass an identifier from its output. The inspected Metal backend names its first device `MTL0`; a backend name is not necessarily a device identifier. |
| `cmake: command not found` | §1 prerequisites not done |
| Build finishes but `--list-devices` shows only CPU on a GPU machine | Backend not compiled in. Mac: check `xcode-select -p` works. Linux: `nvcc --version` must work in the same shell, then `rm -rf third_party/llama.cpp/build && BACKEND=cuda bash scripts/00_build_llamacpp.sh` |
| `CUDA error: no kernel image is available` | Driver/toolkit mismatch or very old GPU. Update the driver; if GPU is pre-Pascal, use `BACKEND=vulkan` or `cpu` |
| `error: unknown argument: -st` (or any flag) | llama.cpp version drift. Run `third_party/llama.cpp/build/bin/llama-cli --help`, find the equivalent flag, put it in `.env` as `CLI_EXTRA=…` and remove the offending one from the script. Note the change in RESULTS.md "Observations" |
| `MODEL is not set` / `MODEL file not found` | Edit `.env`; use an absolute path; no spaces around `=` |
| Model loads but generation is extremely slow on a Mac | Model larger than the GPU working set → layers fell back to CPU. Use a smaller quant or model; check `offloaded N/M layers` |
| Worker prints `endpoint : 127.0.0.1:50052` | It bound to loopback; pass `BIND=<your LAN IP> bash scripts/03_start_worker.sh …` |
| Requester: `failed to connect to …:50052` | Worker not running / firewall / wrong IP. Test with `nc -vz <WORKER_IP> 50052` |
| WSL2 port not reachable from another machine | WSL2 is NAT-ed. On Windows PowerShell (admin): `netsh interface portproxy add v4tov4 listenport=50052 listenaddress=0.0.0.0 connectport=50052 connectaddress=$(wsl hostname -I)` and allow 50052 in Windows Firewall. Use the Windows host IP as `<WORKER_IP>` |
| Requester run works but `0,1` is barely slower than `1,0` | Good — that is the hop latency tax being small on Ethernet. Try Wi-Fi to see it grow |
| Out of memory on the worker | Worker advertises all its memory; cap it: `MEM_MB=6000 bash scripts/03_start_worker.sh` |
| Different answer text between splits | Expected small numeric drift across backends (Metal vs CUDA). Note it; it is a known limitation (ARCHITECTURE.md §3.4) |
| Hebrew characters in a copied command break the terminal | Never paste from RTL documents; all scripts and this guide are English on purpose |

Log files for every smoke run are in `experiments/logs/`. Attach the relevant one when reporting a problem.

### Debugging both ends of an RPC run

In the current Mac + Windows setup, the Mac is the RPC server/worker and Ubuntu in WSL is the requester. Run all commands from `nocturne`. Replace `100.100.10.20` below with the Mac's actual Tailscale IPv4 address. Stop any active request before restarting the worker with debug enabled.

**Mac worker:** stop the existing server with Ctrl-C, then restart it and save both output streams:

```bash
mkdir -p experiments/logs
GGML_RPC_DEBUG=1 BIND=100.100.10.20 \
  bash -x scripts/03_start_worker.sh MTL0 \
  2>&1 | tee "experiments/logs/worker-debug-$(date +%Y%m%d-%H%M%S).log"
```

Use the exact device name from `01_list_devices.sh` if it differs. Confirm that the printed endpoint uses the intended Tailscale address. Leave the worker running.

**Windows requester, inside Ubuntu/WSL:**

```bash
mkdir -p experiments/logs
GGML_RPC_DEBUG=1 CLI_EXTRA="--verbose --log-prefix --log-timestamps" \
  bash -x scripts/04_run_two_machines.sh 100.100.10.20 \
  2>&1 | tee "experiments/logs/requester-debug-$(date +%Y%m%d-%H%M%S).log"
```

If `.env` already sets `CLI_EXTRA`, add the debug flags there while preserving flags you need: the scripts source `.env` after reading their inherited environment. The requester also retains its normal smoke log; the outer `tee` captures the shell trace and wrapper messages as well. These `tee` pipelines are for interactive diagnosis; their shell exit status may reflect `tee` rather than inference, so inspect the script's success/error message.

| Option | What it shows |
|---|---|
| `bash -x` | Expanded shell commands, selected arguments, and configuration assignments |
| `GGML_RPC_DEBUG=1` | Instrumented RPC operations such as connection setup, device memory queries, buffer allocation, tensor transfer/cache checks, and graph execution |
| `--verbose --log-prefix --log-timestamps` | Detailed requester logs with level prefixes and timestamps, including model loading and device placement |

In the worker log, look for functions such as `alloc_buffer`, `set_tensor`, `set_tensor_hash`, `graph_compute`, and `graph_recompute`. A connection message alone does not prove model execution: correlate graph operations on the worker with RPC device allocation and generation on the requester. RPC logs describe operations and tensor metadata; they do not trace every arithmetic operation inside a GPU kernel. Requester timestamps are not a synchronized timestamp stream from both machines.

The options above were checked against the source in the available llama.cpp checkout. Confirm them with your own `llama-cli --help` if another machine uses a different revision; `ggml-rpc-server` does not accept the requester's `--verbose` flags.

For timing experiments, return to the normal commands without `bash -x` or the verbose CLI flags, and remove any exported `GGML_RPC_DEBUG` (`unset GGML_RPC_DEBUG`) or debug settings in `.env`. Upstream enables RPC debug whenever that variable exists, including when its value is `0`; the worker wrapper now explicitly unsets it for empty/zero values. Debug output can materially affect timings.

---

## 8. Hand-back checklist (what to send after setting up a machine)

- [ ] Build log line (host, backend, commit) in `experiments/RESULTS.md`
- [ ] `01_list_devices.sh` output pasted into RESULTS.md "Observations"
- [ ] Table A rows: GPU and CPU (and CUDA1 if present) from `05_bench.sh`
- [ ] Model SHA-256
- [ ] If paired with another machine: Table C row, Table B rows for all three splits, both directions
- [ ] `experiments/bench.csv` committed (`git add experiments && git commit -m "bench: <host>"`)
- [ ] Anything that surprised you, in "Observations", dated

---

## 9. Beyond one home network

`ggml-rpc-server` has no authentication. Security is deferred by team decision (ADR-009) and recorded as a known limitation. Stop the worker (`Ctrl-C`) when not testing. For machines in different homes, follow docs/he/CONNECTIVITY.md: Tailscale for the first experiment (install from https://tailscale.com/download, sign in with the shared project account, use the `100.x.y.z` address as `<WORKER_IP>` and `BIND`), then port forwarding or a relay to show it works without a VPN.
