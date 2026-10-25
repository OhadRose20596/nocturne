#!/usr/bin/env bash
# Capacity test under a declared memory policy. Artefacts: experiments/r3/capacity/<run_id>/.
# Usage: scripts/10_capacity.sh local <model.gguf> [P1|P2]
#        [SPLIT_MODE=layer|tensor|row] scripts/10_capacity.sh split <model.gguf> <worker-ip[:port]> [ts] [P1|P2]
#        P1 = -ngl 99 -fit off (everything on accelerators). P2 = -fit on (CPU allowed). N_PREDICT= CTX= TIMEOUT_S=

. "$(dirname "$0")/common.sh"
require_bin llama-cli

MODE="${1:-}"; MDL="${2:-}"
[ -n "$MODE" ] && [ -n "$MDL" ] || die "usage: $0 local <model.gguf> [P1|P2] | split <model.gguf> <worker-ip> [ts] [P1|P2]"
[ -f "$MDL" ] || die "model not found: $MDL"
N_PREDICT="${N_PREDICT:-32}"; CTX="${CTX:-2048}"; TIMEOUT_S="${TIMEOUT_S:-900}"
HOST="$(host_label)"; PLAT="$(detect_platform)"; SHA="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
OUT="$REPO_ROOT/experiments/r3/capacity"; mkdir -p "$OUT"

case "$MODE" in
  local) POLICY="${3:-P1}"; PEER=""; TS="" ;;
  split) PEER="${3:-}"; [ -n "$PEER" ] || die "split needs <worker-ip[:port]>"; case "$PEER" in *:*) ;; *) PEER="$PEER:$RPC_PORT";; esac
         TS="${4:-}"; POLICY="${5:-P1}" ;;
  *) die "mode must be local or split" ;;
esac
case "$POLICY" in P1) PFLAGS=(-ngl 99 -fit off) ;; P2) PFLAGS=(-fit on) ;; *) die "policy must be P1 or P2" ;; esac
SPLIT_MODE="${SPLIT_MODE:-layer}"; SM_TAG=""
case "$SPLIT_MODE" in
  layer) ;;
  tensor) PFLAGS+=(--split-mode tensor); SM_TAG="_tp" ;;
  row)    PFLAGS+=(--split-mode row);    SM_TAG="_row" ;;
  *) die "SPLIT_MODE must be layer, tensor or row" ;;
esac

RUN_ID="$(date -u +%Y%m%d-%H%M%S)_${HOST}_cap_${MODE}_$(basename "$MDL" .gguf | tr ' ' '_')_${POLICY}${SM_TAG}${TS:+_ts_$(printf '%s' "$TS" | tr ',' '_')}"
D="$OUT/$RUN_ID"; mkdir -p "$D"
log "run $RUN_ID"

mem_snapshot() {   # label — every probe is best-effort; never abort the run because a tool is missing
  set +e
  { echo "== $1 $(now_iso)"
    case "$PLAT" in
      macos) sysctl vm.swapusage 2>/dev/null; vm_stat 2>/dev/null | grep -E "free|wired|compressor|Pageins|Pageouts" ;;
      linux) free -m 2>/dev/null; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null; fi ;;
    esac; } >> "$D/mem.txt" 2>/dev/null
  set -e
}
mem_snapshot before

ARGS=("$BIN/llama-cli" -m "$MDL" -c "$CTX" -n "$N_PREDICT" -p "Explain in two sentences what a transformer is." --temp 0 -st --simple-io --verbose "${PFLAGS[@]}" ${CLI_EXTRA:-})
[ -n "$PEER" ] && ARGS+=(--rpc "$PEER")
[ -n "$TS" ]   && ARGS+=(-ts "$TS")
printf '%q ' "${ARGS[@]}" > "$D/cmd.txt"; echo >> "$D/cmd.txt"

TIMEFMT=""; TIMEBIN=""
if [ -x /usr/bin/time ]; then case "$PLAT" in macos) TIMEBIN=(/usr/bin/time -l) ;; linux) TIMEBIN=(/usr/bin/time -v) ;; esac; fi
STARTED="$(now_iso)"; T0=$(date +%s)
set +e
"${TIMEBIN[@]}" "${ARGS[@]}" > "$D/stdout.txt" 2> "$D/stderr.log" & PID=$!
(
  sleep "$TIMEOUT_S"
  pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null; sleep 10
  pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null
  pkill -KILL -f "llama-cli .*$(basename "$MDL")" 2>/dev/null     # belt and braces: any orphaned inference on this model
) & WD=$!
wait "$PID"; RC=$?
kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null
pgrep -f "llama-cli .*$(basename "$MDL")" >/dev/null 2>&1 && { warn "inference still alive after timeout — killing"; pkill -KILL -f "llama-cli .*$(basename "$MDL")"; sleep 2; }
set -e
WALL=$(( $(date +%s) - T0 ))
mem_snapshot after

grep -E 'load_tensors|offloaded|RPC[0-9]*\[|model buffer size|assigned to device|failed to allocate|out of memory|OOM|ggml_metal.*alloc|llama_model_load: error|Meta device|tensor parallelism|SPLIT_MODE_TENSOR|not implemented|unsupported|GGML_ASSERT|flash_attn' "$D/stderr.log" > "$D/placement.log" 2>/dev/null || true
OFF="$(grep -oE 'offloaded [0-9]+/[0-9]+ layers' "$D/stderr.log" 2>/dev/null | tail -1 || true)"
ALLOC_FAIL="$(grep -ciE 'failed to allocate|out of memory|insufficient memory' "$D/stderr.log" || true)"
TOK_LINE="$(grep -hE 'eval time|Generation:' "$D/stderr.log" "$D/stdout.txt" 2>/dev/null | tail -1 || true)"
PEAK="$(grep -E 'maximum resident set size|Maximum resident set size' "$D/stderr.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | tail -1 || true)"   # bytes on macOS, KB on Linux
[ "$PLAT" = linux ] && [ -n "$PEAK" ] && PEAK=$(( PEAK * 1024 ))
GEN_OK=0; [ "$RC" = 0 ] && [ -n "$TOK_LINE" ] && GEN_OK=1
FITS="no"
if [ "$GEN_OK" = 1 ] && [ "$ALLOC_FAIL" = 0 ]; then
  case "$POLICY" in
    P1) case "$OFF" in offloaded\ *) n="${OFF#offloaded }"; a="${n%%/*}"; b="${n#*/}"; b="${b%% *}"; [ "$a" = "$b" ] && FITS="yes" || FITS="no (only $OFF)";; *) FITS="unknown (no offload line)";; esac ;;
    P2) FITS="yes ($OFF)" ;;
  esac
fi
if [ "$RC" = 124 ] || [ "$RC" = 143 ] || [ "$RC" = 137 ]; then FITS="timeout (${TIMEOUT_S}s) — not evidence of memory failure"; fi

cat > "$D/manifest.json" <<EOF
{ "run_id": "$RUN_ID", "kind": "capacity", "mode": "$MODE", "policy": "$POLICY",
  "host": "$HOST", "platform": "$PLAT", "peer": "$PEER", "tensor_split": "$TS", "split_mode": "$SPLIT_MODE",
  "model": "$(basename "$MDL")", "model_bytes": $(stat -f%z "$MDL" 2>/dev/null || stat -c%s "$MDL"),
  "llama_sha": "$SHA", "ctx": $CTX, "n_predict": $N_PREDICT,
  "started_utc": "$STARTED", "wall_s": $WALL, "exit_code": $RC,
  "offloaded": "$OFF", "alloc_failures": ${ALLOC_FAIL:-0}, "generated": $GEN_OK,
  "peak_rss_bytes": ${PEAK:-null}, "fits_under_policy": "$FITS",
  "timing_line": "$(printf '%s' "$TOK_LINE" | sed 's/"/\\"/g' | tr -d '\n')" }
EOF
log "verdict: fits_under_policy=$FITS  rc=$RC  wall=${WALL}s  $OFF  peak_rss=$(( ${PEAK:-0} / 1048576 )) MiB"
sed 's/^/    /' "$D/placement.log" | grep -E 'buffer size|offloaded|assigned' | tail -8
log "saved: $D"
