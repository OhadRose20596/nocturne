#!/usr/bin/env bash
# Build llama.cpp with Metal, CUDA or CPU and RPC support.
# Usage: [BACKEND=metal|cuda|cpu] scripts/00_build_llamacpp.sh   (Windows: run from WSL2)

. "$(dirname "$0")/common.sh"

BACKEND="$(detect_backend)"
log "platform=$(detect_platform) backend=$BACKEND"
log "llama.cpp dir: $LLAMA_DIR"

command -v cmake >/dev/null 2>&1 || die "cmake not found. macOS: brew install cmake · Ubuntu: sudo apt install cmake build-essential"
command -v git   >/dev/null 2>&1 || die "git not found"

mkdir -p "$(dirname "$LLAMA_DIR")"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  log "cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  log "llama.cpp already cloned; pulling"
  git -C "$LLAMA_DIR" pull --ff-only || warn "pull failed, building existing checkout"
fi

CMAKE_FLAGS="-DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release"
case "$BACKEND" in
  metal) CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=ON" ;;
  cuda)
    command -v nvcc >/dev/null 2>&1 || warn "nvcc not on PATH; cmake may still find CUDA via CUDACXX or /usr/local/cuda"
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON" ;;
  vulkan) CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_VULKAN=ON" ;;
  cpu)   CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=OFF -DGGML_CUDA=OFF" ;;
  *) die "unknown BACKEND=$BACKEND (metal|cuda|vulkan|cpu)" ;;
esac

log "cmake $CMAKE_FLAGS"
# shellcheck disable=SC2086
cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD" $CMAKE_FLAGS
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build "$LLAMA_BUILD" --config Release -j "$JOBS" \
  --target llama-cli llama-server llama-bench ggml-rpc-server 2>/dev/null \
  || cmake --build "$LLAMA_BUILD" --config Release -j "$JOBS"   # older trees: build everything

log "built. binaries:"
ls -1 "$BIN" | grep -E '^(llama-cli|llama-server|llama-bench|ggml-rpc-server|rpc-server)$' || true

log "record the build in experiments/RESULTS.md:"
echo "  host=$(host_label) backend=$BACKEND commit=$(git -C "$LLAMA_DIR" rev-parse --short HEAD) date=$(now_iso)"
