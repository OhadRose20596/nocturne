#!/usr/bin/env bash
# llama-server for the load experiment, identical flags in every variant. Logs: experiments/r3/load/server_logs/.
# Usage: [MODEL=... CTX=4096 NP=8 PORT=8080] scripts/13_serve.sh solo | split <worker-ip[:port]> [ts]

. "$(dirname "$0")/common.sh"
require_bin llama-server; require_model

MODE="${1:-}"; [ -n "$MODE" ] || die "usage: $0 solo | split <worker-ip[:port]> [ts]"
CTX="${CTX:-4096}"; NP="${NP:-8}"; PORT="${PORT:-8080}"
HOST="$(host_label)"; LOGD="$REPO_ROOT/experiments/r3/load/server_logs"; mkdir -p "$LOGD"

ARGS=("$BIN/llama-server" -m "$MODEL" -ngl 99 -c "$CTX" -np "$NP" --host 0.0.0.0 --port "$PORT" --metrics)
case "$MODE" in
  solo) VARIANT="solo" ;;
  split)
    W="${2:-}"; [ -n "$W" ] || die "split needs <worker-ip[:port]>"; case "$W" in *:*) EP="$W";; *) EP="$W:$RPC_PORT";; esac
    TS="${3:-1,1}"; ARGS+=(--rpc "$EP" -ts "$TS"); VARIANT="split_${TS//,/_}_$(printf '%s' "$EP" | tr '.:' '__')" ;;
  *) die "mode must be solo or split" ;;
esac

LOG="$LOGD/$(date -u +%Y%m%d-%H%M%S)_${HOST}_${VARIANT}.log"
{ echo "# host=$HOST variant=$VARIANT sha=$(git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null) model=$(basename "$MODEL") ctx=$CTX np=$NP port=$PORT started=$(now_iso)"
  printf '# cmd: '; printf '%q ' "${ARGS[@]}"; echo; } > "$LOG"
log "variant=$VARIANT model=$(basename "$MODEL") ctx=$CTX np=$NP -> http://$(case "$(detect_platform)" in macos) ipconfig getifaddr en0 2>/dev/null || echo 0.0.0.0;; *) hostname -I 2>/dev/null | awk '{print $1}';; esac):$PORT"
log "log -> $LOG   (Ctrl-C to stop)"
if command -v script >/dev/null 2>&1; then
  case "$(detect_platform)" in
    macos) exec script -q -a -F "$LOG" "${ARGS[@]}" ;;
    *)     exec script -q -a -f -c "$(printf '%q ' "${ARGS[@]}")" "$LOG" ;;
  esac
else
  exec "${ARGS[@]}" 2>&1 | tee -a "$LOG"
fi
