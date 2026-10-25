#!/usr/bin/env bash
# Smoke test: one prompt on one chosen device. Numbers come from 05_bench.sh, not from here.
# Usage: [PROMPT=... N_PREDICT=128] scripts/02_run_local.sh [MTL0|CUDA0|none]

. "$(dirname "$0")/common.sh"
require_bin llama-cli
require_model

DEVICE="${1:-auto}"
PROMPT="${PROMPT:-Explain in three sentences how BitTorrent distributes a file.}"
N_PREDICT="${N_PREDICT:-64}"
CTX="${CTX:-2048}"

ARGS=(-m "$MODEL" -p "$PROMPT" -n "$N_PREDICT" -c "$CTX" --temp 0 -st --simple-io ${CLI_EXTRA:-})
case "$DEVICE" in
  auto)     ARGS+=(-ngl 99) ;;
  none|cpu) ARGS+=(--device none -ngl 0); DEVICE="CPU" ;;
  *)        ARGS+=(--device "$DEVICE" -ngl 99) ;;
esac

log "host=$(host_label) device=$DEVICE model=$(basename "$MODEL") n_predict=$N_PREDICT"
LOGF="$REPO_ROOT/experiments/logs/$(date +%Y%m%d-%H%M%S)_$(host_label)_smoke_${DEVICE}.log"
mkdir -p "$(dirname "$LOGF")"

set +e
"$BIN/llama-cli" "${ARGS[@]}" 2>&1 | tee "$LOGF"
RC=${PIPESTATUS[0]}
set -e

echo
if [ "$RC" -eq 0 ]; then
  log "OK on $DEVICE. Full log: experiments/logs/$(basename "$LOGF")"
  log "Confirm the device was actually used: look for 'load_tensors' / 'offloaded N/N layers' lines above."
  log "Now run:  bash scripts/05_bench.sh   for measured numbers."
else
  warn "llama-cli exited with code $RC. Log: $LOGF"
  warn "For 'invalid device', run 'bash scripts/01_list_devices.sh' and copy the exact device name (e.g. MTL0)."
  warn "If it complains about an unknown flag, run '$BIN/llama-cli --help' and set CLI_EXTRA in .env (see docs/SETUP_GUIDE.md §7)."
  exit "$RC"
fi
