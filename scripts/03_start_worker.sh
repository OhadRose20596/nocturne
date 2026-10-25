#!/usr/bin/env bash
# Start this machine as a worker: ggml-rpc-server on one device, port 50052.
# Usage: [BIND=<lan|tailscale|relay ip>] [MEM_MB=12000] [GGML_RPC_DEBUG=1] scripts/03_start_worker.sh [DEVICE]
# Bind only to an address you mean to expose. See docs/he/CONNECTIVITY.md and ADR-009.

. "$(dirname "$0")/common.sh"
RPC="$(rpc_server_bin)"

DEVICE="${1:-}"
BIND="${BIND:-}"
CACHE_DIR="${LLAMA_CACHE:-$HOME/.cache/llama.cpp/rpc}"

if [ -z "$BIND" ]; then
  case "$(detect_platform)" in
    macos) BIND="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)" ;;
    linux) BIND="$(hostname -I 2>/dev/null | awk '{print $1}')"; BIND="${BIND:-127.0.0.1}" ;;
    *)     BIND="127.0.0.1" ;;
  esac
fi

ARGS=(-H "$BIND" -p "$RPC_PORT" -c)          # -c = local tensor cache -> this worker becomes a "seeder"
[ -n "$DEVICE" ]         && ARGS+=(--device "$DEVICE")
[ -n "${MEM_MB:-}" ]     && ARGS+=(--mem "$MEM_MB")
[ -n "${THREADS:-}" ]    && ARGS+=(-t "$THREADS")

mkdir -p "$CACHE_DIR"
log "worker host=$(host_label) device=${DEVICE:-all} endpoint=$BIND:$RPC_PORT cache=$CACHE_DIR"
log "on the requester use:  --rpc $BIND:$RPC_PORT"
log "Ctrl-C to stop"
if [ "${GGML_RPC_DEBUG:-0}" = "0" ]; then
  unset GGML_RPC_DEBUG
else
  export GGML_RPC_DEBUG
fi
exec env LLAMA_CACHE="$CACHE_DIR" "$RPC" "${ARGS[@]}"
