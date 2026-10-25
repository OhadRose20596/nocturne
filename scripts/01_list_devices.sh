#!/usr/bin/env bash
# List the compute devices llama.cpp sees. The names (MTL0, CUDA0, ...) are what --device takes.
# Usage: scripts/01_list_devices.sh

. "$(dirname "$0")/common.sh"
require_bin llama-cli

log "host=$(host_label) backend=$(detect_backend)"
"$BIN/llama-cli" --list-devices

case "$(detect_backend)" in
  cuda)
    echo
    log "nvidia-smi view (driver-level; CUDA_VISIBLE_DEVICES filters this list):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true ;;
  metal)
    echo
    log "Apple Silicon shares memory between CPU and GPU. Use the exact GPU device name listed above."
    log "Force CPU-only with:  --device none   (or -ngl 0)"
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "  unified memory: %.0f GB\n", $1/1024/1024/1024}' ;;
esac
