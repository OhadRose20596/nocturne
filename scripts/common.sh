#!/usr/bin/env bash
# Shared helpers for all scripts. Source only. Runs on macOS bash 3.2 and Linux bash 4+.
# Config priority: .env in the repo (shared) < ~/.nocturne.env (per machine) < shell environment.
# LLAMA_DIR defaults to ~/dev/llama.cpp when a build exists there, else third_party/llama.cpp.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_pre_LLAMA_DIR="${LLAMA_DIR:-}"; _pre_MODEL="${MODEL:-}"; _pre_BIND="${BIND:-}"; _pre_BACKEND="${BACKEND:-}"
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
fi
if [ -f "$HOME/.nocturne.env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.nocturne.env"
fi
[ -n "$_pre_LLAMA_DIR" ] && LLAMA_DIR="$_pre_LLAMA_DIR"
[ -n "$_pre_MODEL" ]     && MODEL="$_pre_MODEL"
[ -n "$_pre_BIND" ]      && BIND="$_pre_BIND"
[ -n "$_pre_BACKEND" ]   && BACKEND="$_pre_BACKEND"

if [ -z "${LLAMA_DIR:-}" ]; then
  if [ -x "$HOME/dev/llama.cpp/build/bin/llama-cli" ]; then
    LLAMA_DIR="$HOME/dev/llama.cpp"
  else
    LLAMA_DIR="$REPO_ROOT/third_party/llama.cpp"
  fi
fi
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_DIR/build}"
BIN="$LLAMA_BUILD/bin"
MODEL="${MODEL:-}"
RESULTS="${RESULTS:-$REPO_ROOT/experiments/RESULTS.md}"
RPC_PORT="${RPC_PORT:-50052}"

log()  { printf '\033[1;34m[nocturne]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nocturne]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[nocturne]\033[0m %s\n' "$*" >&2; exit 1; }

detect_platform() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux)  echo "linux" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *) echo "unknown" ;;
  esac
}

detect_backend() {
  if [ -n "${BACKEND:-}" ]; then echo "$BACKEND"; return; fi
  case "$(detect_platform)" in
    macos) echo "metal" ;;
    linux)
      if command -v nvidia-smi >/dev/null 2>&1; then echo "cuda"; else echo "cpu"; fi ;;
    *) echo "cpu" ;;
  esac
}

rpc_server_bin() {
  if   [ -x "$BIN/ggml-rpc-server" ]; then echo "$BIN/ggml-rpc-server"
  elif [ -x "$BIN/rpc-server" ];      then echo "$BIN/rpc-server"
  else die "RPC server binary not found in $BIN. Run scripts/00_build_llamacpp.sh"; fi
}

require_bin() {
  [ -x "$BIN/$1" ] || die "$1 not found in $BIN. Run scripts/00_build_llamacpp.sh"
}

require_model() {
  [ -n "$MODEL" ] || die "MODEL is not set. Put MODEL=/path/to/model.gguf in .env"
  [ -f "$MODEL" ]  || die "MODEL file not found: $MODEL"
}

host_label() { hostname -s 2>/dev/null || hostname; }
now_iso()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
