#!/usr/bin/env bash
# Join this machine (macOS, Linux, WSL2) to a nocturne run: check, fix what is safe, write one report file.
# Usage: bash 15_join.sh [--check] [--install] [--self] [--key "<pubkey>" | --key-file F] [--coordinator <ip>] [--stop]   (docs/JOIN.md)

set -u
MODE=fix; INSTALL=0; SELF=0; KEY=""; COORD=""; EXTRA=""; STOP=0
TORCH_VERSION="${NOCTURNE_TORCH_VERSION:-}"; TORCH_INDEX="${NOCTURNE_TORCH_INDEX:-}"
REPO_URL="${NOCTURNE_REPO_URL:-https://github.com/OhadRose20596/nocturne.git}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check ;;
    --install) INSTALL=1 ;;
    --key) KEY="$2"; shift ;;
    --key-file) KEY="$(tr -d '\r' < "$2" | head -1)"; shift ;;
    --coordinator) COORD="$2"; shift ;;
    --extra) EXTRA="$2"; shift ;;
    --stop) STOP=1 ;;
    --self) SELF=1 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$(uname -s)" in Darwin) OS=macos ;; Linux) OS=linux ;; *) echo "unsupported OS: $(uname -s)" >&2; exit 2 ;; esac
WSL=0; [ "$OS" = linux ] && grep -qi microsoft /proc/version 2>/dev/null && WSL=1
[ "$WSL" = 1 ] && PATH="/usr/lib/wsl/lib:$PATH"
HOST="$(hostname -s 2>/dev/null || hostname)"
STATE="$HOME/.nocturne"; mkdir -p "$STATE"
REPORT="$HOME/nocturne_join_${HOST}.txt"
if [ "$INSTALL" = 1 ]; then export TMPDIR="$HOME/.cache/nocturne-join-tmp"; mkdir -p "$TMPDIR"; fi
TMP="$(mktemp -d)"; FACTS="$TMP/facts"; CHECKS="$TMP/checks"; LOGS="$TMP/logs"; : >"$FACTS"; : >"$CHECKS"; : >"$LOGS"
KEEP=""
trap 'rm -rf "$TMP"; [ -n "$KEEP" ] && kill "$KEEP" 2>/dev/null' EXIT
VENV="$HOME/dev/nocturne-torch"; PY="$VENV/bin/python"

have()  { command -v "$1" >/dev/null 2>&1; }
fact()  { printf 'fact %s=%s\n' "$1" "$2" >>"$FACTS"; }
check() { printf 'check %s %s %s\n' "$1" "$2" "$3" >>"$CHECKS"; printf '  [%-5s] %-12s %s\n' "$2" "$1" "$3"; }
log()   { { printf -- '--- %s\n$ %s\n' "$1" "$2"; bash -c "$2" 2>&1 | head -80; echo; } >>"$LOGS"; }
SUDO_OK=0
need_sudo() {
  [ "$MODE" = check ] && return 1
  [ "$SUDO_OK" = 1 ] && return 0
  [ "$(id -u)" = 0 ] && { SUDO_OK=1; return 0; }
  echo "  sudo is needed for: $1"
  if sudo -v </dev/tty 2>/dev/null || sudo -v; then
    SUDO_OK=1
    ( while sudo -n -v 2>/dev/null; do sleep 50; done ) & KEEP=$!
    return 0
  fi
  return 1
}
apt_install() {
  have apt-get || return 1
  need_sudo "installing $*" || return 1
  { sudo_ apt-get update -qq; sudo_ env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"; } >>"$LOGS" 2>&1
}
sudo_() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

if [ "$STOP" = 1 ]; then
  if pgrep -f 'ifconfig awdl0 down' >/dev/null 2>&1; then
    sudo pkill -f 'ifconfig awdl0 down' && echo "stopped the AWDL hold-down loop"
  else echo "no AWDL loop is running"; fi
  PF="$STATE/join_caffeinate.pid"
  if [ -f "$PF" ] && ps -p "$(cat "$PF")" -o comm= 2>/dev/null | grep -q caffeinate; then
    kill "$(cat "$PF")" && echo "stopped caffeinate"
  fi
  rm -f "$PF"; exit 0
fi

echo "nocturne join: $HOST ($OS$([ "$WSL" = 1 ] && echo ', WSL2')), mode $MODE$([ "$INSTALL" = 1 ] && echo ' +install')"
fact report_version 1
fact date "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "$(id -u)" = 0 ] && [ "$WSL" = 0 ] && echo "  warning: running as root. The key, venv and repo will go under $HOME. Run as your normal user instead."
fact os "$OS"; fact wsl "$WSL"; fact host "$HOST"; fact user "$(id -un)"; fact home "$HOME"; fact mode "$MODE"
log uname 'uname -a'

# ---- network
LANIF=""; IP=""; VPN=off
if [ "$OS" = macos ]; then
  DEFIF="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
  LANIF="$(netstat -rn -f inet | awk '$1=="default" && $NF !~ /^(utun|ppp|ipsec)/ {print $NF; exit}')"
  [ -n "$LANIF" ] && IP="$(ipconfig getifaddr "$LANIF" 2>/dev/null)"
  case "$DEFIF" in utun*|ppp*|ipsec*) VPN=on ;; esac
  log routes 'netstat -rn -f inet | head -12; scutil --nc list'
  log link 'networksetup -listallhardwareports | grep -B1 -A1 "Device: '"$LANIF"'"'
else
  DEFIF="$(ip -4 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
  LANIF="$(ip -4 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev" && $(i+1) !~ /^(tun|tap|ppp|wg)/){print $(i+1); exit}}')"
  [ -n "$LANIF" ] && IP="$(ip -4 -o addr show dev "$LANIF" 2>/dev/null | awk '{sub(/\/.*/,"",$4); print $4; exit}')"
  [ -z "$IP" ] && IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  case "$DEFIF" in tun*|tap*|ppp*|wg*) VPN=on ;; esac
  log routes 'ip -4 route show; hostname -I'
fi
fact lan_if "$LANIF"; fact ip "$IP"; fact vpn "$VPN"
# link type of the LAN interface: wifi, wired or unknown (WSL2: the Windows adapter, passed in the --extra facts)
LINK=unknown
if [ "$WSL" = 1 ]; then
  WL=""; [ -n "$EXTRA" ] && [ -f "$EXTRA" ] && WL="$(tr -d '\r' <"$EXTRA" | sed -n 's/^fact win_link_type=//p' | head -1)"
  [ -n "$WL" ] && LINK="$WL"
elif [ "$OS" = macos ]; then
  HWPORT="$(networksetup -listallhardwareports 2>/dev/null | awk -v d="$LANIF" '/^Hardware Port:/{p=substr($0,16)} $1=="Device:" && $2==d {print p; exit}')"
  case "$HWPORT" in *Wi-Fi*|*AirPort*|*WLAN*) LINK=wifi ;; "") ;; *) LINK=wired ;; esac
elif [ -n "$LANIF" ] && [ -e "/sys/class/net/$LANIF" ]; then
  if [ -d "/sys/class/net/$LANIF/wireless" ] || [ -d "/sys/class/net/$LANIF/phy80211" ]; then LINK=wifi; else LINK=wired; fi
fi
fact link_type "$LINK"
if [ "$LINK" = unknown ]; then check link SKIP "could not tell whether $LANIF is Wi-Fi or wired"; else check link PASS "$LINK on $LANIF"; fi
if [ -n "$IP" ]; then check network PASS "address $IP on $LANIF"; else check network FAIL "no LAN address found"; fi
if [ "$VPN" = on ]; then check vpn FAIL "default route goes through $DEFIF. Disconnect the VPN by hand, then run again"
else check vpn PASS "no VPN on the default route"; fi
if [ "$WSL" = 1 ] && [ -n "${NOCTURNE_WIN_IP:-}" ]; then
  if hostname -I | tr ' ' '\n' | grep -qx "$NOCTURNE_WIN_IP"; then check wsl_mirror PASS "WSL shares the Windows address $NOCTURNE_WIN_IP"
  else check wsl_mirror FAIL "WSL has $(hostname -I), Windows has $NOCTURNE_WIN_IP. networkingMode=mirrored is not active"; fi
fi

# ---- macOS: AWDL and sleep
if [ "$OS" = macos ]; then
  if pgrep -f 'ifconfig awdl0 down' >/dev/null 2>&1; then check awdl PASS "hold-down loop already running"
  elif ! ifconfig awdl0 2>/dev/null | head -1 | grep -q '<UP'; then check awdl PASS "awdl0 is down (no loop, macOS may raise it again)"
  elif need_sudo "holding AWDL down while runs are active (undo: 15_join.sh --stop)"; then
    sudo sh -c 'nohup sh -c "while true; do ifconfig awdl0 down; sleep 5; done" >/dev/null 2>&1 &'
    sleep 1; check awdl FIXED "started the hold-down loop"
  else check awdl FAIL "awdl0 is up and no hold-down loop is running"; fi
  if pgrep -x caffeinate >/dev/null 2>&1; then check sleep PASS "caffeinate is running"
  elif [ "$MODE" = check ]; then check sleep FAIL "caffeinate is not running"
  else nohup caffeinate -dimsu >/dev/null 2>&1 & echo $! >"$STATE/join_caffeinate.pid"; check sleep FIXED "started caffeinate -dimsu"; fi
  log power 'pmset -g | grep -E " sleep|displaysleep|powernap"; pmset -g batt | head -2'
fi

# ---- ssh server and the coordinator key
SSH_PORT=22; [ "$WSL" = 1 ] && SSH_PORT=2222
listening() {
  if [ "$OS" = macos ]; then nc -z 127.0.0.1 "$1" >/dev/null 2>&1
  else (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$1 "; fi
}
if [ "$SELF" = 1 ]; then check ssh_server SKIP "this machine is the coordinator, nobody logs in to it"
elif listening "$SSH_PORT"; then check ssh_server PASS "listening on port $SSH_PORT"
elif [ "$OS" = macos ]; then
  if need_sudo "turning on Remote Login (ssh server)"; then
    sudo launchctl load -w /System/Library/LaunchDaemons/ssh.plist 2>>"$LOGS"; sleep 1
    if listening 22; then check ssh_server FIXED "Remote Login turned on"
    else check ssh_server FAIL "turn on System Settings > General > Sharing > Remote Login"; fi
  else check ssh_server FAIL "Remote Login is off"; fi
else
  if need_sudo "installing and starting the ssh server on port $SSH_PORT"; then
    have sshd || [ -x /usr/sbin/sshd ] || apt_install openssh-server
    [ "$WSL" = 1 ] && echo "Port 2222" | sudo_ tee /etc/ssh/sshd_config.d/nocturne.conf >/dev/null
    { sudo_ ssh-keygen -A; sudo_ systemctl enable ssh || sudo_ systemctl enable sshd; sudo_ systemctl daemon-reload; sudo_ systemctl restart ssh.socket ssh || sudo_ systemctl restart ssh || sudo_ systemctl restart sshd || sudo_ service ssh restart; } >>"$LOGS" 2>&1
    sleep 1
    if listening "$SSH_PORT"; then check ssh_server FIXED "ssh server started on port $SSH_PORT"
    else check ssh_server FAIL "ssh server is not listening on $SSH_PORT (see log)"; fi
  else check ssh_server FAIL "nothing listens on port $SSH_PORT"; fi
fi
fact ssh_port "$SSH_PORT"
if [ -n "$KEY" ]; then
  if grep -qF "$KEY" "$HOME/.ssh/authorized_keys" 2>/dev/null; then check ssh_key PASS "coordinator key already authorized"
  elif [ "$MODE" = check ]; then check ssh_key FAIL "coordinator key is not in authorized_keys"
  else mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; AK="$HOME/.ssh/authorized_keys"
       [ -s "$AK" ] && [ -n "$(tail -c1 "$AK")" ] && echo >>"$AK"
       printf '%s\n' "$KEY" >>"$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"
       check ssh_key FIXED "coordinator key added to authorized_keys"; fi
else check ssh_key SKIP "no --key given"; fi

# ---- device
if [ "$OS" = macos ]; then
  MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  DEVICE=mps; fact device mps; fact mem_gb "$MEM_GB"; fact chip "$(sysctl -n machdep.cpu.brand_string)"
  check device PASS "$(sysctl -n machdep.cpu.brand_string), ${MEM_GB} GB unified memory"
elif have nvidia-smi && GPU="$(nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>/dev/null | head -1)" && [ -n "$GPU" ]; then
  DEVICE=cuda; fact device cuda; fact gpu "$GPU"
  if [ -z "$TORCH_INDEX" ]; then
    case "$GPU" in *"GTX 9"*|*"GTX 10"*|*"TITAN X"*|*"TITAN V"*|*"Quadro P"*|*"Quadro M"*|*"Tesla P"*|*"Tesla M"*|*"Tesla V"*) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
      *) TORCH_INDEX="https://download.pytorch.org/whl/cu130" ;; esac
  fi
  fact torch_index "$TORCH_INDEX"
  if [ "$WSL" = 1 ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && ! grep -q '/usr/lib/wsl/lib' "$HOME/.bashrc" 2>/dev/null && [ "$MODE" != check ]; then
    printf '%s\n' 'export PATH=/usr/lib/wsl/lib:$PATH' | cat - "$HOME/.bashrc" 2>/dev/null >"$TMP/bashrc" && cat "$TMP/bashrc" >"$HOME/.bashrc"
    check nvidia_path FIXED "nvidia-smi put on the PATH of ssh commands (first line of ~/.bashrc)"
  fi
  fact mem_gb "$(echo "$GPU" | awk -F', ' '{printf "%.1f", $2/1024}')"
  check device PASS "$GPU"
else
  DEVICE=cpu; fact device cpu; fact mem_gb "$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
  check device PASS "no NVIDIA GPU visible, CPU only"
  log nvidia 'nvidia-smi'
fi

# ---- python venv and torch
torch_ok() { [ -x "$PY" ] && "$PY" -c 'import torch' 2>/dev/null; }
TORCH_PIP=ok
if [ "$INSTALL" = 1 ] && ! torch_ok; then
  [ "$OS" = linux ] && { { have git && have curl; } || apt_install git curl; }
  SYS_PY=""
  for c in python3.12 python3.13 python3.11 python3; do
    if have "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then SYS_PY="$c"; break; fi
  done
  if [ -z "$SYS_PY" ]; then check python FAIL "no Python 3.11+ found. Install one (python.org, brew, or apt), then run again"
  else
    echo "  creating $VENV and installing torch (large download)"
    mkdir -p "$HOME/dev"
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
      "$SYS_PY" -m venv --clear "$VENV" >>"$LOGS" 2>&1 || { apt_install python3-venv && "$SYS_PY" -m venv --clear "$VENV" >>"$LOGS" 2>&1; }
    fi
    "$PY" -m pip install -q -U pip >>"$LOGS" 2>&1
    SPEC="torch"; [ -n "$TORCH_VERSION" ] && SPEC="torch==$TORCH_VERSION"
    if [ "$OS" = linux ] && have nvidia-smi; then "$PY" -m pip install -q "$SPEC" --index-url "$TORCH_INDEX" >>"$LOGS" 2>&1 || TORCH_PIP=failed
    else "$PY" -m pip install -q "$SPEC" >>"$LOGS" 2>&1 || TORCH_PIP=failed; fi
  fi
fi
if torch_ok; then
  T="$("$PY" - 2>&1 <<'EOF'
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
print(torch.__version__, dev, "bf16_ok" if (a @ a).float().abs().mean().item() > 0 else "bf16_bad")
EOF
)"
  T="$(echo "$T" | tail -1)"; TDEV="$(echo "$T" | awk '{print $2}')"
  fact python "$PY"; fact torch "$T"
  case "$T" in
    *bf16_ok*)
      if [ "$TDEV" = "$DEVICE" ]; then check torch PASS "$T"
      else check torch FAIL "$T, but this machine's device is $DEVICE. torch does not see it (driver too old for the wheel? set NOCTURNE_TORCH_INDEX)"; fi ;;
    *) check torch FAIL "$T" ;;
  esac
elif [ "$TORCH_PIP" = failed ]; then
  fact python "$PY"; check torch FAIL "pip install torch failed. See the log at the end of the report, then run again with --install"
elif [ -x "$PY" ]; then
  fact python "$PY"; check torch FAIL "venv exists at $VENV but torch does not import. Run again with --install"
else
  fact python ""; check torch FAIL "no venv at $VENV. Run again with --install"
fi

# ---- repo
REPO=""; NOC_OK=0
[ -x "$PY" ] && REPO="$("$PY" -c 'import nocturne,pathlib;print(pathlib.Path(nocturne.__file__).parents[1])' 2>/dev/null)"
[ -n "$REPO" ] && NOC_OK=1
[ -z "$REPO" ] && [ -d "$HOME/dev/nocturne/.git" ] && REPO="$HOME/dev/nocturne"
if [ "$INSTALL" = 1 ] && [ -x "$PY" ]; then
  if [ -z "$REPO" ]; then
    echo "  cloning the repo to $HOME/dev/nocturne"
    GIT_TERMINAL_PROMPT=0 git clone -q "$REPO_URL" "$HOME/dev/nocturne" >>"$LOGS" 2>&1 && REPO="$HOME/dev/nocturne"
  fi
  if [ -n "$REPO" ] && [ "$NOC_OK" = 0 ] && torch_ok; then
    "$PY" -m pip install -q -e "$REPO[bench-torch,dev]" >>"$LOGS" 2>&1 && NOC_OK=1
  fi
fi
if [ -n "$REPO" ] && [ -d "$REPO/.git" ]; then
  GIT_TERMINAL_PROMPT=0 git -C "$REPO" fetch -q 2>>"$LOGS"
  ST="$(git -C "$REPO" status -sb 2>/dev/null | head -1)"
  if [ "$MODE" != check ] && echo "$ST" | grep -q 'behind' && ! echo "$ST" | grep -q 'ahead'; then
    git -C "$REPO" pull -q --ff-only >>"$LOGS" 2>&1 && PULLED=1
    ST="$(git -C "$REPO" status -sb | head -1)"
  fi
  SHA="$(git -C "$REPO" rev-parse --short HEAD)"
  fact repo "$REPO"; fact sha "$SHA"; fact git_status "$ST"
  if echo "$ST" | grep -q 'behind'; then check repo FAIL "$SHA, $ST"
  elif [ "$NOC_OK" = 0 ]; then check repo FAIL "$SHA at $REPO, but the package is not installed in the venv. Run again with --install"
  else check repo "$([ "${PULLED:-0}" = 1 ] && echo FIXED || echo PASS)" "$SHA at $REPO ($ST)"; fi
  log repo "git -C '$REPO' log -1 --oneline; git -C '$REPO' status -sb | head -15"
else
  fact repo ""; check repo FAIL "repo not found. Run again with --install (needs git)"
fi

# ---- llama.cpp: report only, nothing is built here
LLAMA_PIN="${NOCTURNE_LLAMA_PIN:-1e7bcf3}"
LDIR="${LLAMA_DIR:-$HOME/dev/llama.cpp}"; LBIN=""
for d in "$LDIR/build/bin" "$LDIR/build/Release/bin"; do [ -x "$d/llama-server" ] && { LBIN="$d"; break; }; done
[ -z "$LBIN" ] && have llama-server && LBIN="$(dirname "$(command -v llama-server)")"
if [ -n "$LBIN" ]; then
  RPCBIN=""; [ -x "$LBIN/ggml-rpc-server" ] && RPCBIN="$LBIN/ggml-rpc-server"
  LCOMMIT="$("$LBIN/llama-server" --version 2>&1 | sed -n 's/.*commit \([0-9a-f]*\)).*/\1/p' | head -1)"
  BK=cpu
  ls "$LBIN" 2>/dev/null | grep -qi 'ggml-metal' && BK=metal
  ls "$LBIN" 2>/dev/null | grep -qi 'ggml-cuda' && BK=cuda
  if [ "$BK" = cpu ]; then
    DEVS="$("$LBIN/llama-server" --list-devices 2>&1 | head -20)"
    echo "$DEVS" | grep -qi 'metal' && BK=metal
    echo "$DEVS" | grep -qi 'cuda' && BK=cuda
  fi
  fact llama_dir "$LBIN"; fact llama_commit "$LCOMMIT"; fact llama_backend "$BK"; fact ggml_rpc_server "$RPCBIN"
  if [ -z "$RPCBIN" ]; then check llama_cpp FAIL "llama-server found in $LBIN, but ggml-rpc-server is missing (build with -DGGML_RPC=ON)"
  elif [ -z "$LCOMMIT" ]; then check llama_cpp FAIL "llama-server --version printed no commit"
  else
    case "$LCOMMIT" in "$LLAMA_PIN"*) PINNOTE="same as the pinned $LLAMA_PIN" ;; *) PINNOTE="differs from the pinned $LLAMA_PIN" ;; esac
    check llama_cpp PASS "commit $LCOMMIT ($PINNOTE), backend $BK, both binaries in $LBIN"
  fi
else
  fact llama_dir ""; fact llama_commit ""; fact llama_backend ""; fact ggml_rpc_server ""
  check llama_cpp FAIL "llama-server not found under $LDIR/build/bin or on the PATH. Build it with scripts/00_build_llamacpp.sh (not done by this script)"
fi

# ---- models: Hugging Face directories and GGUF files
MODELS="$(ls "$HOME/models/hf" 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
GGUFS=""
for f in "$HOME"/models/*.gguf "$HOME"/models/gguf/*.gguf; do
  [ -f "$f" ] || continue
  GGUFS="$GGUFS${GGUFS:+,}${f#$HOME/models/}:$(wc -c <"$f" | tr -d ' ')"
done
fact models "$MODELS"; fact gguf_models "$GGUFS"
if [ -n "$MODELS$GGUFS" ]; then check models PASS "hf: ${MODELS:-none}; gguf: $(printf '%s' "$GGUFS" | tr ',' '\n' | grep -c .) file(s)"
else check models FAIL "none under ~/models/hf or ~/models/*.gguf. The coordinator copies them (16_join_apply.py prints the commands)"; fi
log disk 'df -h "$HOME" | tail -1'

# ---- firewall
if [ "$OS" = macos ]; then
  FW=/usr/libexec/ApplicationFirewall/socketfilterfw
  if "$FW" --getglobalstate 2>/dev/null | grep -qi 'enabled'; then
    REAL_PY=""; [ -x "$PY" ] && REAL_PY="$("$PY" -c 'import os,sys;print(os.path.realpath(sys.executable))')"
    if [ -n "$REAL_PY" ] && "$FW" --listapps 2>/dev/null | grep -A1 -F "$REAL_PY" | grep -q 'Allow incoming'; then check firewall PASS "firewall on, python is permitted"
    elif [ -n "$REAL_PY" ] && need_sudo "allowing incoming connections to $REAL_PY"; then
      { sudo "$FW" --add "$REAL_PY"; sudo "$FW" --unblockapp "$REAL_PY"; } >>"$LOGS" 2>&1
      check firewall FIXED "firewall on, python allowed"
    else check firewall FAIL "firewall on, python not allowed yet (approve the macOS prompt on the first run)"; fi
  else check firewall PASS "macOS firewall is off"; fi
elif grep -q '^ENABLED=yes' /etc/ufw/ufw.conf 2>/dev/null; then
  if need_sudo "opening ports $SSH_PORT and 50070-50079 in ufw"; then
    { sudo_ ufw allow "$SSH_PORT/tcp"; sudo_ ufw allow 50070:50079/tcp; } >>"$LOGS" 2>&1; check firewall FIXED "ufw rules added"
  else check firewall FAIL "ufw is enabled and the ports are not opened"; fi
elif have firewall-cmd && firewall-cmd --state 2>/dev/null | grep -q running; then
  if need_sudo "opening ports $SSH_PORT and 50070-50079 in firewalld"; then
    { sudo_ firewall-cmd --permanent --add-port="$SSH_PORT/tcp" --add-port=50070-50079/tcp; sudo_ firewall-cmd --reload; } >>"$LOGS" 2>&1
    check firewall FIXED "firewalld ports added"
  else check firewall FAIL "firewalld is running and the ports are not opened"; fi
else check firewall PASS "no enabled Linux firewall seen (ufw, firewalld)"; fi

# ---- reach the coordinator (TCP to the receiver; ping is only logged, some hosts drop ICMP)
if [ -n "$COORD" ]; then
  log ping_coordinator "ping -c 10 $COORD | tail -3"
  if have curl; then C="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$COORD:50079/" 2>/dev/null)"
  else C="$(python3 -c 'import sys,socket;socket.create_connection((sys.argv[1],50079),5);print("tcp")' "$COORD" 2>/dev/null)"; fi
  case "$C" in ""|000) check coordinator FAIL "no answer from $COORD:50079. Is '16_join_apply.py listen' running there, and its firewall open?" ;;
    *) check coordinator PASS "receiver at $COORD:50079 answers" ;; esac
fi

# ---- report
[ -n "$EXTRA" ] && [ -f "$EXTRA" ] && tr -d '\r' <"$EXTRA" | grep -E '^fact ' >>"$FACTS"
[ -n "$EXTRA" ] && [ -f "$EXTRA" ] && tr -d '\r' <"$EXTRA" | grep -E '^check ' >>"$CHECKS"
NFAIL="$(grep -c ' FAIL ' "$CHECKS")"
fact failures "$NFAIL"
{ echo "# nocturne join report"; cat "$FACTS"; echo; cat "$CHECKS"; echo; cat "$LOGS"; } >"$REPORT"
echo
echo "Report: $REPORT  ($NFAIL failed)"
SENT=0
if [ -n "$COORD" ]; then
  URL="http://$COORD:50079/join"
  if have curl; then R="$(curl -s --max-time 20 --retry 1 -H 'Expect:' -X POST --data-binary @"$REPORT" "$URL" 2>/dev/null)"
  else R="$(python3 -c 'import sys,urllib.request;print(urllib.request.urlopen(sys.argv[1],open(sys.argv[2],"rb").read(),timeout=10).read().decode())' "$URL" "$REPORT" 2>/dev/null)"; fi
  case "$R" in saved*) SENT=1; echo "Sent to the coordinator at $COORD. Nothing more to do here." ;; esac
fi
[ "$SENT" = 0 ] && echo "Not sent automatically. Mail this file to the coordinator machine."
[ "$NFAIL" = 0 ]
