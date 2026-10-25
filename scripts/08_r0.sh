#!/usr/bin/env bash
# R0 recorder. Every run saves raw artefacts under experiments/r0/runs/<run_id>/ (cmd, bench.json, stderr, placement, net, manifest).
# Tables are built later by scripts/09_r0_report.py. Nothing is parsed here.
# Usage: scripts/08_r0.sh worker [DEVICE] | local | placement <worker-ip> <split>
#        LINK=wifi|ethernet|thunderbolt|internet OVERLAY=plain|portforward|tailscale|relay|tunnel LOCATION=lan|wan \
#        [PP=256 TG=64 REPS=5] [SPLITS="1,0 1,1 0,1"] scripts/08_r0.sh rpc <worker-ip[:port]>
# Split order is remote first (1,0 = all on the worker). llama-bench separates devices with '/', not ','.

. "$(dirname "$0")/common.sh"

MODE="${1:-}"; [ -n "$MODE" ] || die "usage: $0 worker|local|rpc|placement ..."
PP="${PP:-256}"; TG="${TG:-64}"; REPS="${REPS:-5}"; CTX="${CTX:-2048}"
R0="$REPO_ROOT/experiments/r0"; RUNS="$R0/runs"; mkdir -p "$RUNS" "$R0/worker_logs"
HOST="$(host_label)"; PLAT="$(detect_platform)"; BE="$(detect_backend)"
SHA="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

model_hash() {
  [ -f "$MODEL" ] || { echo "missing"; return; }
  local cache="$R0/.model_hash_$(basename "$MODEL").txt"
  if [ -f "$cache" ] && [ "$cache" -nt "$MODEL" ]; then cat "$cache"; return; fi
  local h
  if command -v shasum >/dev/null 2>&1; then h="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)"
  elif command -v sha256sum >/dev/null 2>&1; then h="$(sha256sum "$MODEL" | cut -d' ' -f1)"
  else h="unavailable"; fi
  printf '%s' "$h" > "$cache"; echo "$h"
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n'; }

new_run() {   # kind label -> sets RUN_DIR, RUN_ID
  RUN_ID="$(date -u +%Y%m%d-%H%M%S)_${HOST}_$1_$(printf '%s' "$2" | tr ',/: ' '____')"
  RUN_DIR="$RUNS/$RUN_ID"; mkdir -p "$RUN_DIR"
  log "run $RUN_ID"
}

write_manifest() {   # kind label [extra json fields]
  cat > "$RUN_DIR/manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "kind": "$1",
  "label": "$(json_escape "$2")",
  "host": "$HOST",
  "platform": "$PLAT",
  "backend": "$BE",
  "role": "$(json_escape "${ROLE:-}")",
  "peer": "$(json_escape "${PEER:-}")",
  "link": "$(json_escape "${LINK:-}")",
  "overlay": "$(json_escape "${OVERLAY:-}")",
  "location": "$(json_escape "${LOCATION:-}")",
  "path": "$(json_escape "${NETPATH:-}")",
  "split": "$(json_escape "${SPLIT:-}")",
  "device": "$(json_escape "${DEVICE:-}")",
  "llama_sha": "$SHA",
  "model": "$(json_escape "$(basename "${MODEL:-}")")",
  "model_sha256": "$(model_hash)",
  "pp": $PP, "tg": $TG, "reps": $REPS, "ctx": $CTX,
  "started_utc": "$STARTED",
  "finished_utc": "$(now_iso)",
  "exit_code": ${EXIT_CODE:-0}
  ${3:-}
}
EOF
}

run_logged() {   # file-stem, then command...   stdout -> <stem>.out, stderr -> stderr.log, exit code -> EXIT_CODE
  local stem="$1"; shift
  printf '%q ' "$@" > "$RUN_DIR/cmd.txt"; echo >> "$RUN_DIR/cmd.txt"
  set +e
  "$@" > "$RUN_DIR/$stem" 2> "$RUN_DIR/stderr.log"
  EXIT_CODE=$?
  set -e
  [ "$EXIT_CODE" = 0 ] || warn "exit code $EXIT_CODE — see $RUN_DIR/stderr.log"
}

bench_row() {   # label, then llama-bench args...
  local label="$1"; shift
  new_run bench "$label"; STARTED="$(now_iso)"
  run_logged bench.json "$BIN/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -o json -v "$@"
  grep -E 'load_tensors|offloaded|RPC[0-9]*\[|model buffer size|assigned to device' "$RUN_DIR/stderr.log" > "$RUN_DIR/placement.log" 2>/dev/null || true
  write_manifest bench "$label"
  if command -v python3 >/dev/null 2>&1 && [ -s "$RUN_DIR/bench.json" ]; then
    python3 - "$RUN_DIR/bench.json" <<'EOF' || true
import json,sys
try:
    rows=json.load(open(sys.argv[1]))
    for r in rows:
        name = r.get("test") or (f"pp{r.get('n_prompt')}" if int(r.get("n_prompt") or 0) > 0 else f"tg{r.get('n_gen')}")
        print(f"    {name:>8}: {float(r.get('avg_ts',0)):8.2f} ± {float(r.get('stddev_ts',0)):.2f} t/s")
except Exception as e:
    print("    (could not parse bench.json:", e, ")")
EOF
  fi
}

placement_row() {   # worker-endpoint split
  local ep="$1"; SPLIT="$2"
  new_run placement "rpc_${SPLIT}"; STARTED="$(now_iso)"
  run_logged placement.out "$BIN/llama-cli" -m "$MODEL" -ngl 99 --rpc "$ep" -ts "$SPLIT" -c "$CTX" -n 8 -p "ping" --temp 0 -st --simple-io --verbose ${CLI_EXTRA:-}
  grep -E 'load_tensors|offloaded|RPC[0-9]*\[|model buffer size|llama_model_load|print_info: n_layer|assigned to device' "$RUN_DIR/stderr.log" > "$RUN_DIR/placement.log" || true
  write_manifest placement "rpc_${SPLIT}"
  local rw="${SPLIT%%,*}" lw="${SPLIT#*,}" has_rpc=0 has_loc=0
  grep -qE 'RPC[0-9]*\[[^]]+\] model buffer size = *[1-9]' "$RUN_DIR/placement.log" && has_rpc=1   # a real (>0 MiB) remote allocation, not the dry-run 0.00 line
  grep -qE '(MTL|Metal|CUDA|Vulkan)[A-Za-z0-9_]* model buffer size = *[1-9]' "$RUN_DIR/placement.log" && has_loc=1
  if [ "$rw" = 0 ] && [ "$has_rpc" = 1 ]; then
    warn "placement: split $SPLIT asked for nothing remote, but an RPC buffer was allocated — check $RUN_DIR/placement.log"
  elif { [ "$rw" = 0 ] || [ "$has_rpc" = 1 ]; } && { [ "$lw" = 0 ] || [ "$has_loc" = 1 ]; } && grep -q offloaded "$RUN_DIR/placement.log"; then
    log "placement: buffers match split $SPLIT (local=$has_loc rpc=$has_rpc)"
  else
    warn "placement: buffers do NOT match split $SPLIT (local=$has_loc rpc=$has_rpc) — check $RUN_DIR/placement.log"
  fi
  sed 's/^/    /' "$RUN_DIR/placement.log" | head -12
}

net_row() {   # peer-ip
  local ip="$1"
  new_run net "$ip"; STARTED="$(now_iso)"
  printf 'ping -c 20 -i 0.2 %q ; iperf3 -c %q -t 10 ; iperf3 -c %q -t 10 -R ; tailscale status/ping\n' "$ip" "$ip" "$ip" > "$RUN_DIR/cmd.txt"
  set +e   # every probe may legitimately fail (no iperf3 server, no tailscale, unreachable peer); record, don't abort
  {
    echo "== date: $(now_iso)  from: $HOST  to: $ip"
    echo "== route"
    case "$PLAT" in macos) route -n get "$ip" 2>&1 | grep -E 'interface|gateway' ;; linux) ip route get "$ip" 2>&1 | head -1 ;; esac
    if command -v tailscale >/dev/null 2>&1; then
      echo "== tailscale status (peer line)"; tailscale status 2>&1 | grep -F "$ip" || echo "(peer not in tailscale status)"
      echo "== tailscale ping"; tailscale ping -c 3 "$ip" 2>&1 || true
    fi
    echo "== ping -c 20 -i 0.2"; ping -c 20 -i 0.2 "$ip" 2>&1
    if command -v iperf3 >/dev/null 2>&1; then
      echo "== iperf3 -c $ip -t 10"; iperf3 -c "$ip" -t 10 2>&1 || echo "(iperf3 failed — is 'iperf3 -s' running on $ip?)"
      echo "== iperf3 -c $ip -t 10 -R (reverse)"; iperf3 -c "$ip" -t 10 -R 2>&1 || true
    else
      echo "== iperf3 not installed"
    fi
  } > "$RUN_DIR/net.txt" 2>&1
  set -e
  EXIT_CODE=0
  grep -qE 'packets transmitted, [1-9]' "$RUN_DIR/net.txt" || warn "ping got no replies from $ip — check the address/link before benchmarking"
  if grep -qE 'via DERP|relay' "$RUN_DIR/net.txt"; then NETPATH="relay"; elif grep -qE 'pong from|direct' "$RUN_DIR/net.txt"; then NETPATH="direct"; else NETPATH="${NETPATH:-unknown}"; fi
  write_manifest net "$ip"
  grep -E 'rtt|round-trip|receiver' "$RUN_DIR/net.txt" | sed 's/^/    /' || true
  printf '%s\n' "$RUN_DIR/net.txt" | sed 's/^/    saved: /'
}

require_net_labels() {
  LINK="${LINK:-}"; OVERLAY="${OVERLAY:-}"; LOCATION="${LOCATION:-}"
  case "$LINK" in ethernet|wifi|thunderbolt|internet) ;; *) die "set LINK=ethernet|wifi|thunderbolt|internet (the physical link of THIS run)";; esac
  case "$OVERLAY" in plain|portforward|tailscale|relay|tunnel) ;; *) die "set OVERLAY=plain|portforward|tailscale|relay|tunnel";; esac
  case "$LOCATION" in lan|wan) ;; *) die "set LOCATION=lan or LOCATION=wan";; esac
}

case "$MODE" in
  worker)
    DEVICE="${2:-}"
    WLOG="$R0/worker_logs/$(date -u +%Y%m%d-%H%M%S)_${HOST}_worker_${DEVICE:-all}.log"
    log "worker log -> $WLOG"
    { echo "# host=$HOST sha=$SHA device=${DEVICE:-all} bind=${BIND:-auto} port=$RPC_PORT started=$(now_iso)"; } > "$WLOG"
    if command -v script >/dev/null 2>&1; then
      case "$PLAT" in
        macos) exec script -q -a -F "$WLOG" bash "$REPO_ROOT/scripts/03_start_worker.sh" ${DEVICE:+"$DEVICE"} ;;
        *)     exec script -q -a -f -c "bash '$REPO_ROOT/scripts/03_start_worker.sh' ${DEVICE:+'$DEVICE'}" "$WLOG" ;;
      esac
    else
      exec bash "$REPO_ROOT/scripts/03_start_worker.sh" ${DEVICE:+"$DEVICE"} 2>&1 | tee -a "$WLOG"
    fi ;;

  local)
    require_bin llama-bench; require_model; ROLE="local"
    log "host=$HOST backend=$BE sha=${SHA:0:7} model=$(basename "$MODEL") pp=$PP tg=$TG reps=$REPS"
    bench_row "gpu_all_layers" -ngl 99
    bench_row "cpu_only"       -ngl 0
    log "done. Table A rows saved under $RUNS (kind=bench, role=local)" ;;

  placement)
    require_bin llama-cli; require_model; ROLE="requester"
    W="${2:-}"; S="${3:-1,1}"; [ -n "$W" ] || die "usage: $0 placement <worker-ip[:port]> [split]"
    case "$W" in *:*) EP="$W";; *) EP="$W:$RPC_PORT";; esac; PEER="$EP"
    placement_row "$EP" "$S" ;;

  rpc)
    require_bin llama-bench; require_bin llama-cli; require_model; require_net_labels; ROLE="requester"
    W="${2:-}"; [ -n "$W" ] || die "usage: LINK=.. OVERLAY=.. LOCATION=.. $0 rpc <worker-ip[:port]>"
    case "$W" in *:*) EP="$W";; *) EP="$W:$RPC_PORT";; esac; PEER="$EP"; IP="${EP%%:*}"
    SPLITS="${SPLITS:-1,0 1,1 0,1}"
    log "requester=$HOST worker=$EP link=$LINK overlay=$OVERLAY location=$LOCATION splits=[$SPLITS] pp=$PP tg=$TG reps=$REPS"
    log "--- Table C: network path"
    net_row "$IP"
    if command -v nc >/dev/null 2>&1; then
      if nc -z -w 3 "$IP" "${EP##*:}" 2>/dev/null; then
        log "worker port ${EP##*:} on $IP is open"
      else
        die "cannot connect to $EP. On the worker run:  BIND=$IP scripts/08_r0.sh worker <DEVICE>   (and check firewall). Net probe kept in $RUN_DIR/net.txt"
      fi
    fi
    for s in $SPLITS; do
      log "--- placement check, split $s"
      placement_row "$EP" "$s"
    done
    for s in $SPLITS; do
      SPLIT="$s"
      log "--- Table B: bench, split $s"
      bench_row "rpc_split_${s}" -ngl 99 --rpc "$EP" -ts "${s//,//}"
    done
    log "done. Now run:  python3 scripts/09_r0_report.py" ;;

  *) die "unknown mode: $MODE (worker|local|rpc|placement)" ;;
esac
