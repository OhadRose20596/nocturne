#!/usr/bin/env bash
# Measurements with llama-bench. Appends to experiments/bench.csv and prints a markdown row.
# Usage: [PP=256 TG=64 REPS=5] scripts/05_bench.sh [DEVICE | rpc <worker-ip> [split]]

. "$(dirname "$0")/common.sh"
require_bin llama-bench
require_model

PP="${PP:-256}"; TG="${TG:-64}"; REPS="${REPS:-3}"
MODE="${1:-local}"
CSV="$REPO_ROOT/experiments/bench.csv"
COMMIT="$(git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
STAMP="$(now_iso)"; HOST="$(host_label)"; BE="$(detect_backend)"
mkdir -p "$(dirname "$CSV")"

run_bench() {   # label, then llama-bench args...
  local label="$1"; shift
  log "--- $label ---"
  "$BIN/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -o md "$@"
  local tmp; tmp="$(mktemp)"
  "$BIN/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -o csv "$@" > "$tmp" 2>/dev/null || true
  if [ ! -s "$CSV" ]; then
    printf 'timestamp,host,backend,label,llama_commit,%s\n' "$(head -1 "$tmp")" > "$CSV"
  fi
  tail -n +2 "$tmp" | while IFS= read -r line; do
    printf '%s,%s,%s,%s,%s,%s\n' "$STAMP" "$HOST" "$BE" "$label" "$COMMIT" "$line" >> "$CSV"
  done
  rm -f "$tmp"
}

log "host=$HOST backend=$BE model=$(basename "$MODEL") pp=$PP tg=$TG reps=$REPS commit=$COMMIT"
case "$MODE" in
  local)
    run_bench "gpu_all_layers" -ngl 99
    run_bench "cpu_only"       -ngl 0 ;;
  rpc)
    W="${2:-}"; [ -n "$W" ] || die "usage: $0 rpc <worker_ip[:port]>[,...] [tensor-split]"
    RPC_LIST=""; OLDIFS="$IFS"; IFS=','
    for w in $W; do case "$w" in *:*) ep="$w" ;; *) ep="$w:$RPC_PORT" ;; esac; RPC_LIST="${RPC_LIST:+$RPC_LIST,}$ep"; done
    IFS="$OLDIFS"
    if [ -n "${3:-}" ]; then
      run_bench "rpc_${RPC_LIST}_split_$3" -ngl 99 --rpc "$RPC_LIST" -ts "$3"
    else
      run_bench "rpc_${RPC_LIST}_split_0,1" -ngl 99 --rpc "$RPC_LIST" -ts 0,1
      run_bench "rpc_${RPC_LIST}_split_1,1" -ngl 99 --rpc "$RPC_LIST" -ts 1,1
      run_bench "rpc_${RPC_LIST}_split_1,0" -ngl 99 --rpc "$RPC_LIST" -ts 1,0
    fi ;;
  *)
    run_bench "device_${MODE}" -ngl 99 --device "$MODE" ;;
esac

echo
log "appended to experiments/bench.csv. Paste the markdown table(s) above into experiments/RESULTS.md"
log "with a line: host=$HOST backend=$BE commit=$COMMIT date=$STAMP"
