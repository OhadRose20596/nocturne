#!/usr/bin/env bash
# RTT and throughput to a peer. The peer runs: iperf3 -s
# Usage: scripts/06_net_baseline.sh <peer-ip>

. "$(dirname "$0")/common.sh"
PEER="${1:-}"; [ -n "$PEER" ] || die "usage: $0 <peer_ip>"
COUNT="${COUNT:-20}"

log "ping $PEER x$COUNT"
if [ "$(detect_platform)" = "macos" ]; then PING_OUT="$(ping -c "$COUNT" -i 0.2 "$PEER" 2>&1)"; else PING_OUT="$(ping -c "$COUNT" -i 0.2 "$PEER" 2>&1)"; fi
echo "$PING_OUT" | tail -2
RTT_LINE="$(echo "$PING_OUT" | grep -E 'round-trip|rtt' | head -1)"
RTT_AVG="$(echo "$RTT_LINE" | sed -E 's#.*= *[0-9.]+/([0-9.]+)/.*#\1#')"

BW="n/a"
if command -v iperf3 >/dev/null 2>&1; then
  log "iperf3 -c $PEER (10 s)"
  IPERF_OUT="$(iperf3 -c "$PEER" -t 10 2>&1)" || warn "iperf3 failed — is 'iperf3 -s' running on $PEER?"
  echo "$IPERF_OUT" | grep -E 'receiver|sender' | tail -2
  BW="$(echo "$IPERF_OUT" | grep receiver | awk '{print $(NF-2), $(NF-1)}' | head -1)"
else
  warn "iperf3 not installed; RTT only"
fi

echo
log "RESULTS.md row (table C):"
echo "| $(now_iso) | $(host_label) | $PEER | ${RTT_AVG:-?} | ${BW:-?} | <eth/wifi/tailscale> |  |"
