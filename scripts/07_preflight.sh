#!/usr/bin/env bash
# Preflight and inventory: do the binaries load, which llama.cpp revision, what hardware, which network path.
# Writes experiments/inventory/<host>_<date>.md. Never modifies the build.
# Usage: scripts/07_preflight.sh [peer-ip]

. "$(dirname "$0")/common.sh"

PEER="${1:-}"
HOST="$(host_label)"; STAMP="$(now_iso)"; PLAT="$(detect_platform)"
OUT_DIR="$REPO_ROOT/experiments/inventory"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${HOST}_$(date -u +%Y%m%d-%H%M%S).md"

section() { printf '\n## %s\n\n' "$1" | tee -a "$OUT"; }
kv()      { printf -- '- **%s:** %s\n' "$1" "$2" | tee -a "$OUT"; }
code()    { printf '```\n%s\n```\n' "$1" | tee -a "$OUT"; }
have()    { command -v "$1" >/dev/null 2>&1; }

printf '# Preflight — %s — %s\n' "$HOST" "$STAMP" | tee "$OUT"
kv "repo" "$REPO_ROOT"
kv "LLAMA_DIR" "$LLAMA_DIR"
kv "BIN" "$BIN"
kv "cwd" "$(pwd)"

section "1. Binary load check (E13)"
PREFLIGHT_OK=1
for b in llama-cli llama-bench llama-server; do
  if [ ! -x "$BIN/$b" ]; then
    PREFLIGHT_OK=0
    kv "$b" "MISSING or not executable"
  fi
done
if [ ! -x "$BIN/ggml-rpc-server" ] && [ ! -x "$BIN/rpc-server" ]; then
  PREFLIGHT_OK=0
  kv "RPC server" "MISSING or not executable"
fi
for b in llama-cli llama-bench llama-server ggml-rpc-server rpc-server; do
  [ -x "$BIN/$b" ] || continue
  case "$b" in
    llama-bench|ggml-rpc-server|rpc-server) CHECK_ARG=--help ;;
    *) CHECK_ARG=--version ;;
  esac
  if out="$("$BIN/$b" "$CHECK_ARG" 2>&1)"; then
    kv "$b $CHECK_ARG" "OK — $(printf '%s' "$out" | grep -m1 -E 'version|build' || printf '%s' "$out" | head -1)"
  else
    rc=$?
    PREFLIGHT_OK=0
    kv "$b $CHECK_ARG" "FAILED (rc=$rc)"
    code "$(printf '%s' "$out" | head -8)"
  fi
done

if [ -e "$BIN/llama-cli" ]; then
  if [ "$PLAT" = macos ] && have otool; then
    rpaths="$(otool -l "$BIN/llama-cli" 2>/dev/null | awk '/LC_RPATH/{f=1} f&&/^[[:space:]]*path /{sub(/^[[:space:]]*path /, ""); sub(/ \(offset [0-9]+\)$/, ""); print; f=0}')"
    kv "LC_RPATH (llama-cli)" "$(printf '%s' "$rpaths" | tr '\n' ' ')"
    case "$rpaths" in
      *@executable_path*|*@loader_path*) kv "rpath kind" "relative — survives moving the folder" ;;
      *) kv "rpath kind" "ABSOLUTE — breaks if the build dir moves or is not visible (sandbox, other user, moved repo)";
         while IFS= read -r p; do
           [ -n "$p" ] || continue
           [ -d "$p" ] && kv "rpath exists here" "$p" || kv "rpath MISSING here" "$p"
         done <<EOF
$rpaths
EOF
         ;;
    esac
  elif [ "$PLAT" = linux ] && have readelf; then
    kv "RUNPATH (llama-cli)" "$(readelf -d "$BIN/llama-cli" 2>/dev/null | grep -E 'RUNPATH|RPATH' | sed 's/.*\[\(.*\)\]/\1/' | tr '\n' ' ')"
    have ldd && { missing="$(ldd "$BIN/llama-cli" 2>/dev/null | grep 'not found' || true)"; [ -n "$missing" ] && code "$missing"; }
  fi
fi

if [ "$PREFLIGHT_OK" = 0 ]; then
  section "How to fix a loader failure"
  cat <<'EOF' | tee -a "$OUT"
The libraries are normally right next to the binary (build/bin). If --version aborts with
"Library not loaded: @rpath/..." the binary carries an ABSOLUTE rpath to the original build dir.
Pick ONE:
  a) Rebuild outside iCloud with a relative rpath (recommended; also stops multi-GB sync):
       export LLAMA_DIR=$HOME/dev/llama.cpp      # put LLAMA_DIR=... in .env too
       git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" && cd "$LLAMA_DIR" && git checkout <SHA>
       cmake -B build -DGGML_RPC=ON -DGGML_METAL=ON -DCMAKE_BUILD_RPATH='@executable_path' -DCMAKE_INSTALL_RPATH='@executable_path'
       cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON -DCMAKE_BUILD_RPATH='$ORIGIN'
       cmake --build build --config Release -j
  b) Patch the existing binaries in place (macOS), no rebuild:
       for b in llama-cli llama-bench llama-server ggml-rpc-server; do
         install_name_tool -add_rpath @executable_path "$BIN/$b"; done
  c) Run from the exact path the build was made in (check CMakeCache.txt: CMAKE_HOME_DIRECTORY).
Then re-run this script. Inspect the error: a dyld/ld failure is a deployment problem;
an unsupported command-line flag is not. Neither alone establishes TP/RPC support.
EOF
fi

section "2. llama.cpp revision and build flags"
kv "git SHA" "$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
kv "git describe" "$(git -C "$LLAMA_DIR" describe --tags --always 2>/dev/null || echo unknown)"
if [ "${PREFLIGHT_GIT_STATUS:-0}" = 1 ]; then
  if dirty="$(git -C "$LLAMA_DIR" status --porcelain --untracked-files=no 2>/dev/null)"; then
    kv "git dirty (tracked files only)" "$( [ -n "$dirty" ] && echo yes || echo no )"
  else
    kv "git dirty" "unknown — git status failed"
  fi
else
  kv "git dirty" "unknown — skipped to avoid a full iCloud tree scan; PREFLIGHT_GIT_STATUS=1 checks tracked files"
fi
CACHE="$LLAMA_BUILD/CMakeCache.txt"
if [ -f "$CACHE" ]; then
  kv "build dir recorded in CMakeCache" "$(grep -m1 '^CMAKE_HOME_DIRECTORY' "$CACHE" | cut -d= -f2)"
  kv "build flags" "$(grep -E '^(GGML_(METAL|CUDA|VULKAN|RPC|BLAS|HIP)|CMAKE_BUILD_TYPE|LLAMA_CURL):' "$CACHE" | sed 's/:[A-Z]*=/=/' | tr '\n' ' ')"
else
  kv "CMakeCache" "not found at $CACHE"
fi
grep -q '"tensor"' "$LLAMA_DIR/common/arg.cpp" 2>/dev/null && kv "--split-mode tensor in source" "yes (presence only; RPC support unverified)" || kv "--split-mode tensor in source" "no"
if [ "$PREFLIGHT_OK" = 1 ] && [ -e "$BIN/llama-cli" ]; then
  kv "--list-devices" ""
  code "$("$BIN/llama-cli" --list-devices 2>&1 | sed -n '1,20p')"
fi

section "3. Machine inventory"
kv "platform" "$PLAT — $(uname -srm)"
case "$PLAT" in
  macos)
    kv "macOS" "$(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
    kv "chip" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    kv "cores (perf/eff)" "$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)/$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)"
    kv "unified memory GB" "$(( $(sysctl -n hw.memsize) / 1073741824 ))"
    kv "GPU cores" "$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
    kv "Metal" "$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Metal/{print $2; exit}')"
    kv "Xcode CLT" "$(xcode-select -p 2>/dev/null)" ;;
  linux)
    kv "distro" "$( . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" )"
    kv "WSL" "$( grep -qi microsoft /proc/version 2>/dev/null && echo yes || echo no )"
    kv "CPU" "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    kv "RAM GB" "$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
    if have nvidia-smi; then
      kv "nvidia-smi" ""
      code "$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1)"
      kv "CUDA toolkit" "$(nvcc --version 2>/dev/null | grep -m1 release || echo 'nvcc not in PATH')"
    else
      kv "NVIDIA" "nvidia-smi not found"
    fi ;;
esac
kv "disk free (repo)" "$(df -h "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
if [ -n "$MODEL" ]; then
  kv "MODEL" "$MODEL"
  if [ -f "$MODEL" ]; then
    kv "model size" "$(du -h "$MODEL" | cut -f1)"
    if have shasum; then kv "model sha256" "$(shasum -a 256 "$MODEL" | cut -d' ' -f1)"; elif have sha256sum; then kv "model sha256" "$(sha256sum "$MODEL" | cut -d' ' -f1)"; fi
  else
    kv "model" "FILE NOT FOUND"
  fi
else
  kv "MODEL" "not set in .env"
fi

section "3b. Python / PyTorch (for R1–R2)"
PY="$(command -v python3 || true)"
if [ -n "$PY" ]; then
  kv "python3" "$("$PY" --version 2>&1) — $PY"
  "$PY" - <<'EOF' 2>/dev/null | tee -a "$OUT" || kv "torch" "not importable"
try:
    import torch
    print(f"- **torch:** {torch.__version__}")
    print(f"- **cuda available:** {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    mps = getattr(torch.backends, "mps", None)
    print(f"- **mps available:** {bool(mps and mps.is_available())}")
    import torch.distributed as dist
    print(f"- **dist.is_available:** {dist.is_available()}  gloo={dist.is_gloo_available() if dist.is_available() else 'n/a'}  nccl={dist.is_nccl_available() if dist.is_available() else 'n/a'}")
except Exception as e:
    print(f"- **torch:** not importable ({e.__class__.__name__})")
try:
    import transformers; print(f"- **transformers:** {transformers.__version__}")
except Exception:
    print("- **transformers:** not installed")
EOF
else
  kv "python3" "not found"
fi

section "4. Network path"
have tailscale && { kv "tailscale" "$(tailscale version 2>/dev/null | head -1)"; kv "tailscale ip" "$(tailscale ip -4 2>/dev/null | head -1)"; } || kv "tailscale" "not installed"
if [ -n "$PEER" ]; then
  kv "peer" "$PEER"
  case "$PLAT" in
    macos) kv "route" "$(route -n get "$PEER" 2>/dev/null | awk '/interface|gateway/{printf "%s ", $0}')" ;;
    linux) kv "route" "$(ip route get "$PEER" 2>/dev/null | head -1)" ;;
  esac
  case "$PEER" in
    100.6[4-9].*|100.[7-9][0-9].*|100.1[01][0-9].*|100.12[0-7].*) kv "address range" "100.64.0.0/10 (CGNAT — consistent with Tailscale, not proof of overlay or LAN/WAN location)" ;;
    10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*) kv "address range" "RFC1918 private address — does not establish LAN/WAN location" ;;
    *) kv "address range" "public / other" ;;
  esac
  have tailscale && kv "tailscale status (peer line)" "$(tailscale status 2>/dev/null | grep -F "$PEER" || echo 'peer not in tailnet / not matched')"
  kv "ping (10 probes)" "$(ping -c 10 -i 0.2 "$PEER" 2>/dev/null | tail -1)"
  if have iperf3; then kv "iperf3" "run: iperf3 -c $PEER -t 5 (peer must run iperf3 -s)"; else kv "iperf3" "not installed"; fi
else
  kv "peer" "none given — re-run with the peer IP to record the path"
fi

section "5. Verdict"
if [ "$PREFLIGHT_OK" = 1 ]; then
  kv "binaries" "load checks passed — still verify model/hash, devices, flags and peer before R0 / Track A"
else
  kv "binaries" "NOT ready — fix the loader issue above first (this is deployment, not a result about TP/RPC)"
fi
kv "inventory written to" "$OUT"
