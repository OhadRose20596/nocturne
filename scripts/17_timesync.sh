#!/usr/bin/env bash
# Clock sync of this host before a run: print the measured offset to one NTP server; with --set, step the clock first (sudo).
# Usage: bash scripts/17_timesync.sh [--set] [--server time.apple.com] [--json]   (docs/MEASUREMENT.md section 2; Windows: 17_timesync.ps1)

set -u
SET=0; JSON=""; SERVER="${NOCTURNE_NTP_SERVER:-time.apple.com}"
while [ $# -gt 0 ]; do
  case "$1" in
    --set) SET=1 ;;
    --json) JSON="--json" ;;
    --server) SERVER="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${NOCTURNE_PYTHON:-python3}"
WSL=0; grep -qi microsoft /proc/version 2>/dev/null && WSL=1

if [ "$SET" = 1 ]; then
  OK=0; HOW=""
  case "$(uname -s)" in
    Darwin) HOW="sntp -sS $SERVER"; sudo sntp -sS "$SERVER" >&2 && OK=1 ;;
    Linux)
      if [ "$WSL" = 1 ]; then
        echo "WSL2 follows the Windows clock: run 17_timesync.ps1 -Set on Windows first, then this reads the hardware clock" >&2
        HOW="hwclock -s (the Windows clock)"; sudo hwclock -s >&2 && OK=1
      elif command -v chronyc >/dev/null 2>&1; then
        HOW="chronyc makestep, server $SERVER added"
        sudo chronyc add server "$SERVER" iburst >&2 || true
        sudo chronyc -a "burst 4/4" >&2; sleep 4; sudo chronyc -a makestep >&2 && OK=1
      elif command -v timedatectl >/dev/null 2>&1; then
        HOW="systemd-timesyncd restart (its own servers)"
        sudo timedatectl set-ntp true >&2 && sudo systemctl restart systemd-timesyncd >&2 && OK=1
        sleep 3
      else
        echo "no chrony and no systemd-timesyncd on this host; install chrony" >&2
      fi ;;
    *) echo "unsupported OS: $(uname -s)" >&2; exit 2 ;;
  esac
  if [ "$OK" = 1 ]; then
    mkdir -p "$HOME/.nocturne" && printf '{"at": "%s", "server": "%s", "how": "%s"}\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$SERVER" "$HOW" > "$HOME/.nocturne/timesync_set.json"
  else
    echo "the clock was not stepped ($HOW failed or no tool)" >&2
  fi
fi
exec "$PY" "$HERE/../nocturne/bench/probes/timesync.py" --server "$SERVER" $JSON
