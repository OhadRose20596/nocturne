# Session 8 connectivity record

Append only. One section per check. Started 2026-11-20 on the M5.

## Summary (written at the end of Phase A)

The CUDA peer of this session is `gtx`, not `pc`. Lior's Lenovo P16v was not in the house on 2026-11-20.
`gtx` is a Windows 11 Home laptop (host `OHAD`, WSL2 Ubuntu 26.04 host `Ohad`, GTX 1070 8 GB). Ohad approved the swap in the session.
All three machines were prepared with the new join tool (`scripts/15_join.sh`, `15_join.ps1`, `16_join_apply.py`, see `docs/JOIN.md`).

| check | result | value |
|---|---|---|
| A1 M5 local state | pass after fix | 192.168.1.242 on en0 (Wi-Fi). GlobalProtect was up at first, disconnected by Ohad. AWDL held down by the join tool. caffeinate on |
| A2 reachability | pass | Air: 0% loss. gtx: ICMP dropped by Windows, TCP 60 of 60 |
| A3 ssh | pass | Air `ohadrosentswaig@192.168.1.219` torch 2.15.0. gtx alias `gtx` port 2222 torch 2.14.1+cu126, Python 3.14.4 |
| A4 configuration | done | `~/.ssh/config` has `Host gtx`. `~/.nocturne/peers.yaml`: m5 bind 192.168.1.242, new peer `gtx`. `pc` left as it was (machine away) |
| A5 parity | pass | all three at 6196396, same engine hash, same 0.5B and 1.5B model hashes |
| A6 six-way TCP | pass | 6 of 6 return 200. gtx hosted with a raw socket listener, see A6 |
| A7 stability | pass with a note | gtx TCP median 7.6 ms, p95 18.4. Air ICMP median 62.8 ms, p95 123.7 at 1 s spacing. Air TCP median 8.4 ms at 0.2 s spacing. Cause: Wi-Fi power save on an idle link, not AWDL |

## A1. Local state on the M5

Commands: `ipconfig getifaddr en0`, `networksetup -getairportnetwork en0`, `scutil --nc list`, `ifconfig awdl0 | head -3`, `pmset -g assertions | head -20`, `route -n get default`, `netstat -rn -f inet`, `ps aux | grep -i vpn`.

```
ipconfig getifaddr en0            -> 192.168.1.242
networksetup -getairportnetwork en0 -> You are not associated with an AirPort network.
scutil --nc list                  -> (no services listed)
ifconfig awdl0                    -> flags=8863<UP,BROADCAST,SMART,RUNNING,...>  (UP)
pmset: PreventUserIdleSystemSleep 1 (powerd, sharingd, Claude Electron). No caffeinate yet.
route -n get default              -> gateway 10.221.205.100, interface utun8
netstat -rn: default 10.221.205.100 utun8 (UGScg)
             default 192.168.1.1 en0 (UGScIg, ifscope)
             192.168.1 link#15 en0
ifconfig utun8                    -> inet 10.221.205.100 --> 10.221.205.100, mtu 1280
ps: PanGPS, GlobalProtect (running since Tue 13:00)
```

Verdict:
- Address is 192.168.1.242. Pass.
- `networksetup -getairportnetwork` reports "not associated" although en0 holds a home address and ARP shows the router on en0. The command is unreliable on this macOS, so the link type is taken from `networksetup -listallhardwareports` (en0 is the Wi-Fi port).
- VPN is NOT off. GlobalProtect is running and owns the primary default route (utun8). `scutil --nc list` does not show it, so that command cannot be used as the VPN check. The home /24 still routes via en0, so LAN pings work, but the guide requires VPN disconnected. Human checkpoint raised.
- awdl0 is UP on the M5. Needs the sudo loop (human checkpoint).

## A2. Reachability from the M5 (first pass)

Air, 20 packets at 0.3 s to 192.168.1.219:

```
20 transmitted, 20 received, 0.0% loss
min/avg/max/stddev = 6.848/56.552/311.271/73.690 ms
sorted RTT: 6.8 6.9 7.4 7.5 8.6 8.7 9.2 9.7 11.1 16.9 19.6 34.8 38.3 46.1 47.1 47.5 99.0 110.9 125.4 165.2 (+311.3 on a timeout-retry slot)
```

Verdict: no loss, but a heavy tail (several 100+ ms). Consistent with AWDL up on the M5 and the Air (Air `ifconfig awdl0` is UP too). Must be re-measured after AWDL is held down.

PC, 5 packets to 172.20.10.2 (the stale hotspot address from `~/.ssh/config`): 100.0% loss. Expected, the PC is no longer on the hotspot. The new PC address is unknown. Human checkpoint raised.

## A3. ssh (first pass)

Air, `ssh -o BatchMode=yes -o ConnectTimeout=8 ohadrosentswaig@192.168.1.219`:

```
hostname -s              -> Mac
ipconfig getifaddr en0   -> 192.168.1.219
whoami                   -> ohadrosentswaig
torch                    -> 2.15.0
awdl0                    -> UP
pmset sleep              -> 1 (sleep prevented by powerd, caffeinate)
```

Note: `hostname -s` on the Air prints `Mac`, but `peers.yaml` says `host: ohads-MacBook-Air-2`. The tool compares `hostname -s` to the peers `host` field only for the coordinator (see the "coordinator host is m5" symptom), so this is recorded and watched, not changed yet.

PC, `ssh p16v`: `ssh: connect to host 172.20.10.2 port 2222: Operation timed out`. Stale address. Checkpoint raised.

## Change of CUDA peer (2026-11-20)

Ohad wrote: "Lior machine isn't here, i'm using different machine now and Lior will join back in the future".
The machine on the home Wi-Fi is host `OHAD` (Windows 11 Home, build 26300), GTX 1070 8192 MiB, driver 582.78, Wi-Fi MAC 30-24-32-4A-72-84, SSID `Ohad_5`.
WSL had no distro. `wsl --install -d Ubuntu` installed Ubuntu 26.04.1 (user `ohadr`, host `Ohad`, Python 3.14.4).
The peer key on the M5 is `gtx`. The `pc` entry and the `p16v` ssh alias still hold the hotspot address 172.20.10.2 and are not used in this session.

Join tool findings on this machine, all recorded in `docs/JOIN.md`:
- The Windows part ran on the first try: profile set to Private, `networkingMode=mirrored`, both firewall rules, power settings, keep-alive.
- `python3-venv` was missing and needs sudo. Ohad ran `sudo apt-get update && sudo apt-get install -y python3-venv` by hand.
- pip failed with "No space left on device": `/tmp` is a 3.9 GB tmpfs. The tool now sets `TMPDIR` under `~/.cache` for installs.
- `pip install -e` then pulled torch 2.15.0+cu132 from PyPI. That build fails on the GTX 1070: "CUDA error: no kernel image is available for execution on the device".
- Installed by hand over ssh: `pip install --force-reinstall torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`. Result: `2.14.1+cu126 12.6 True NVIDIA GeForce GTX 1070`, bf16 matmul ok.
- `pytest tests/bench/test_hybrid.py -q` on gtx: 32 passed.

## A1 (second pass, after the fixes)

```
route -n get default   -> gateway 192.168.1.1, interface en0   (GlobalProtect disconnected by Ohad)
ifconfig awdl0         -> flags=8822<BROADCAST,SMART,SIMPLEX,MULTICAST>   (down)
pgrep                  -> sh -c while true; do ifconfig awdl0 down; sleep 5; done ; caffeinate -dimsu
bash scripts/15_join.sh --self -> 0 failed
```

Verdict: pass. The VPN check that works on this Mac is the default route, not `scutil --nc list`.

## A2 and A7. Reachability and stability

The M5 does not answer ICMP (ping to its own address loses 100%, and both peers see 100% loss to it). Windows on gtx drops ICMP too.
So ICMP is only usable from the M5 to the Air. For gtx the measure is the TCP connect time to port 2222.

```
ping -c 60 -i 1 192.168.1.219 (M5 to Air)
60 transmitted, 60 received, 0.0% loss, min/avg/max/stddev = 7.865/67.263/289.986/50.433 ms
sorted: median 62.8, p95 123.7, max 290.0 ms

TCP connect M5 to gtx:2222, 60 samples, 1 s spacing
fail 0, min 5.0, median 7.6, avg 8.9, p95 18.4, max 26.6 ms

TCP connect M5 to Air:22, 40 samples, 0.2 s spacing:  min 7.0, median 8.4, p95 10.5, max 19.3 ms
TCP connect M5 to Air:22, 30 samples, 1 s spacing:    min 7.9, median 20.7, p95 117.0, max 125.2 ms
ping -c 20 -i 0.2 192.168.1.1 from the Air: min/avg/max 2.973/4.468/5.274 ms
ping -c 20 -i 0.2 192.168.1.1 from the M5:  min/avg/max 3.237/5.821/35.342 ms
```

Wi-Fi on both Macs: 802.11ax, channel 64 (5 GHz, 80 MHz), signal -56 dBm, noise -94 dBm, transmit rate 907 to 960. AWDL down on both. Air on AC power, low power mode 0.

Verdict: pass with a note. The Air p95 is above 40 ms only when the probes are 1 s apart. With 0.2 s spacing the same link gives a median of 8.4 ms.
AWDL is down on both Macs, so it is not the cause. The pattern fits Wi-Fi power save on an idle link. Not proven: no packet capture was taken.
A run keeps the link busy, so the idle figure is not the run figure. Time is background (T1).

## A3 (second pass). ssh

```
ssh ohadrosentswaig@192.168.1.219 -> host Mac, torch 2.15.0, repo ~/Documents/<project folder>/nocturne
ssh gtx                            -> host Ohad, user ohadr, Ubuntu 26.04.1 LTS, Python 3.14.4, GPU 0: NVIDIA GeForce GTX 1070
gtx torch                          -> 2.14.1+cu126, CUDA 12.6, cuda available True
gtx hostname -I                    -> 192.168.1.168 (equal to the Windows address, mirrored mode active)
```

`hostname -s` on the Air is `Mac`. `peers.yaml` says `host: ohads-MacBook-Air-2`. Left as it is and watched in the first run that uses the Air.
The Air repo lives in an iCloud folder shared with the M5, so its `.git` follows the M5 by itself. `git pull` there printed "cannot lock ref" and HEAD was already 6196396.

## A4. Configuration changes

Backups: `~/.nocturne/join/backups/.ssh_config.bak_20261120-150742`, `.nocturne_peers.yaml.bak_20261120-145814` (and two later ones). Made by `16_join_apply.py apply`.

`~/.ssh/config`:

```
21a22,28
>
> # nocturne peer, written by scripts/16_join_apply.py
> Host gtx
>   HostName 192.168.1.168
>   Port 2222
>   User ohadr
>   ServerAliveInterval 30
```

`~/.nocturne/peers.yaml`:

```
43c43
<     bind: 172.20.10.3
---
>     bind: 192.168.1.242
68a69,77
>   gtx:
>     host: Ohad
>     ssh: gtx
>     os: linux
>     backend: cuda
>     mem_gb: 8.0
>     python: "/home/ohadr/dev/nocturne-torch/bin/python"
>     bind: 192.168.1.168
>     networks: {wifi: 192.168.1.168}
```

`nocturne/bench/peers.yaml`: unchanged. Its m5 and air addresses were already the home ones, and it has no CUDA peer.
Not done: the `p16v` alias and the `pc` peer still carry 172.20.10.2. They are unused. Lior's machine gets the join tool when it returns.

## A5. Engine and model parity

```
host  HEAD     git status            engine *.py sha256   0.5B files     1.5B files
m5    6196396  main...origin/main    4b86096a798291b8     c76bd69f584e3a3a  db125acf7c6a78ad
air   6196396  main...origin/main    4b86096a798291b8     c76bd69f584e3a3a  db125acf7c6a78ad
gtx   6196396  main...origin/main    4b86096a798291b8     c76bd69f584e3a3a  db125acf7c6a78ad
```

The model columns are the first 16 hex digits of the sha256 over the sorted per-file sha256 list (`*.json`, `*.safetensors`).
The 1.5B `model.safetensors` sha256 is dd924a11...c6d3ee on gtx and on the M5. Models were copied to gtx with `scp` from the M5. 7B was not copied.
Verdict: pass.

## A6. Six-way TCP test on port 50079

| host \ client | m5 | air | gtx |
|---|---|---|---|
| m5 (192.168.1.242) | - | 200 | 200 |
| air (192.168.1.219) | 200 | - | 200 |
| gtx (192.168.1.168) | 200 | 200 | - |

m5 and air hosted `python3 -m http.server 50079 --bind <ip>`. On gtx that command never starts listening in time:
`socket.getfqdn("192.168.1.168")` takes 10.0 s there, and `http.server` calls it before `listen`. `ss -ltn` showed no socket and the clients got "connection refused".
gtx was then hosted with a 12 line raw socket listener bound to 192.168.1.168:50079 that answers `HTTP/1.0 200 OK`. Both clients got 200.
The engine ranks use raw sockets, so the slow reverse lookup does not touch them. It would touch anything on gtx that uses `http.server`.
No test process was left on any host (`pgrep` empty on all three).
Verdict: pass.

PHASE_A_VERDICT: pass
