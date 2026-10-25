#!/usr/bin/env bash
# Smoke test as requester with a manual layer split over one or more workers.
# Usage: scripts/04_run_two_machines.sh <worker-ip[:port]>[,<ip2>...] [tensor-split]
# Split order is remote first: 1,0 = all layers on the worker, 0,1 = all local, 1,1 = half and half.

. "$(dirname "$0")/common.sh"
require_bin llama-cli
require_model

WORKERS="${1:-}"; [ -n "$WORKERS" ] || die "usage: $0 <worker_ip[:port]>[,...] [tensor-split]"
SPLIT="${2:-}"
PROMPT="${PROMPT:-Explain in three sentences how BitTorrent distributes a file.}"
N_PREDICT="${N_PREDICT:-64}"
CTX="${CTX:-2048}"

RPC_LIST=""
OLDIFS="$IFS"; IFS=','
for w in $WORKERS; do
  case "$w" in *:*) ep="$w" ;; *) ep="$w:$RPC_PORT" ;; esac
  RPC_LIST="${RPC_LIST:+$RPC_LIST,}$ep"
done
IFS="$OLDIFS"

ARGS=(-m "$MODEL" -p "$PROMPT" -n "$N_PREDICT" -c "$CTX" --temp 0 -st --simple-io -ngl 99 --rpc "$RPC_LIST" ${CLI_EXTRA:-})
[ -n "$SPLIT" ] && ARGS+=(--tensor-split "$SPLIT")

log "requester host=$(host_label) workers=$RPC_LIST split=${SPLIT:-auto(memory-proportional)}"
LOGF="$REPO_ROOT/experiments/logs/$(date +%Y%m%d-%H%M%S)_$(host_label)_rpc_${SPLIT:-auto}.log"
mkdir -p "$(dirname "$LOGF")"

set +e
T0="$(date +%s)"
"$BIN/llama-cli" "${ARGS[@]}" 2>&1 | tee "$LOGF"
RC=${PIPESTATUS[0]}
T1="$(date +%s)"
set -e

echo
if [ "$RC" -eq 0 ]; then
  log "OK. wall=$((T1-T0))s  log: experiments/logs/$(basename "$LOGF")"
  log "Check above that layers were offloaded to the RPC device(s) (lines mentioning RPC[...])."
  log "Now run:  bash scripts/05_bench.sh rpc $RPC_LIST   for measured numbers."
else
  warn "llama-cli exited with code $RC. Log: $LOGF"
  warn "Common causes: worker not running, firewall on port $RPC_PORT, wrong IP, model file differs between machines."
  exit "$RC"
fi
