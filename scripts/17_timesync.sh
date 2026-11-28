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
  case "$(uname -s)" in
    Darwin) sudo sntp -sS "$SERVER" >&2 || echo "sntp -sS failed" >&2 ;;
    Linux)
      if [ "$WSL" = 1 ]; then
        echo "WSL2 follows the Windows clock: run 17_timesync.ps1 on Windows first, then this reads the hardware clock" >&2
        sudo hwclock -s >&2 || echo "hwclock -s failed" >&2
      elif command -v chronyc >/dev/null 2>&1; then
        sudo chronyc -a "burst 4/4" >&2; sleep 4; sudo chronyc -a makestep >&2 || echo "chronyc makestep failed" >&2
      elif command -v timedatectl >/dev/null 2>&1; then
        sudo timedatectl set-ntp true >&2 && sudo systemctl restart systemd-timesyncd >&2 || echo "systemd-timesyncd restart failed" >&2
        sleep 3
      else
        echo "no chrony and no systemd-timesyncd on this host; install chrony" >&2
      fi ;;
    *) echo "unsupported OS: $(uname -s)" >&2; exit 2 ;;
  esac
fi
exec "$PY" "$HERE/../nocturne/bench/probes/timesync.py" --server "$SERVER" $JSON
