# Joining a machine to a run

One script runs on the new machine. It checks the machine, fixes what is safe to fix, and writes one report file.
The coordinator (the M5) reads that report and updates its own configuration. Nobody copies commands from a guide.

Status on 2026-11-20: `15_join.sh` ran on the M5 in check mode. `16_join_apply.py` ran in dry-run mode and on synthetic files.
`15_join.sh` ran in fix mode on the Air and delivered its report. The Windows half of `15_join.ps1` ran on one Windows 11 machine with no WSL distro and stopped there, as designed. The fix paths that need `sudo`, and `--install`, have not run yet.
A code review found 27 issues on 2026-11-20. The confirmed ones are fixed, by reading and local tests only.

Windows is supported through WSL2 only. A Windows machine without WSL cannot be a peer today.
The engine imports a Unix-only module (`resource` in `hybrid.py`), so a native Windows peer needs an engine change first.

## The three files

| file | runs on | what it does |
|---|---|---|
| `scripts/15_join.sh` | macOS, Linux, WSL2 Ubuntu | checks and fixes, writes `~/nocturne_join_<host>.txt` |
| `scripts/15_join.ps1` | Windows | Windows checks and fixes, then runs `15_join.sh` inside WSL, copies the report to the Desktop |
| `scripts/16_join_apply.py` | the coordinator | prints the invite commands, receives reports, applies a report |

## Flow

1. On the coordinator: `python3 scripts/16_join_apply.py invite`. It prints one command per operating system, with the coordinator address and public key filled in.
2. On the coordinator: `python3 scripts/16_join_apply.py listen`. It waits for reports on port 50079 of the LAN address.
3. On the new machine: paste the invite command. It asks for the password when a fix needs `sudo`. Windows asks twice: the administrator prompt, then the `sudo` password inside WSL.
4. The report reaches the coordinator by itself when the two machines see each other. If not, send the file by mail. It is plain text.
5. On the coordinator: `python3 scripts/16_join_apply.py apply <report> --peer <name>`. Add `--dry-run` to see the diff first.

`apply` edits `~/.ssh/config` and `~/.nocturne/peers.yaml`. In `nocturne/bench/peers.yaml` it changes only the addresses of a peer that already exists there.
Backups go to `~/.nocturne/join/backups/`. It rewrites whole lines, so a trailing comment on a changed line is lost.
`apply` refuses a report whose network, mirrored networking or ssh server check failed, unless `--force` is given.
Then it pings the peer, opens an ssh session without a password, and lists the models the peer still lacks with the `scp` commands.

## What the join script checks

| check | fix it applies | cannot fix |
|---|---|---|
| network | none, reports the LAN address | no address |
| vpn | none | the VPN must be disconnected by hand |
| awdl (macOS) | starts a root loop that holds `awdl0` down | |
| sleep (macOS) | starts `caffeinate -dimsu` | |
| ssh_server | macOS: turns on Remote Login. Linux: starts sshd (port 2222 in WSL), installs it on apt systems | other package managers |
| ssh_key | adds the coordinator key to `authorized_keys` | |
| device | none, reports memory or the NVIDIA card | |
| torch | with `--install`: creates `~/dev/nocturne-torch` and installs torch. Fails if torch does not see the GPU | no Python 3.11+ |
| repo | fast-forward pull. With `--install`: clone and `pip install -e` | diverged history |
| models | none. The coordinator copies models | |
| firewall | macOS: allows the venv python. Linux: opens the ports in ufw or firewalld | other firewalls |
| coordinator | none, opens a TCP connection to the receiver on port 50079 | receiver not running |

Windows adds: network profile set to Private, `networkingMode=mirrored` in `.wslconfig`, the two firewall rules, no sleep on AC power, the NVIDIA driver, and a hidden `sleep infinity` that keeps WSL up without an Ubuntu window.

Flags: `--self` is for the coordinator itself: it skips the ssh server, since nobody logs in to it. `--check` applies no fix. It still runs `git fetch` and writes the report. `--install` also builds the venv and clones the repo. `--stop` ends the AWDL loop and caffeinate.
On Linux with an NVIDIA card the wheel index is `cu130`, or `cu126` for cards older than the RTX 20 series (GTX 10, Pascal and earlier), which CUDA 13 no longer supports. This rule has not been confirmed on a real card yet.
The torch version is not pinned by default. Set `NOCTURNE_TORCH_VERSION` and `NOCTURNE_TORCH_INDEX` to pin it.
A venv where torch already imports is never changed. `--install` repairs a venv where it does not.

## Report format

Plain text. `fact key=value` lines, then `check <name> <PASS|FIXED|FAIL|SKIP> <message>` lines, then raw command logs.
A report holds the user name, addresses and hostnames of the machine. Received reports are kept in `~/.nocturne/join/`, outside the repo.

## Security

Deferred by ADR-009, not forgotten. The receiver binds the LAN address, never `0.0.0.0`, has no authentication, and only stores text. `apply` validates the address, user and host of a report before it uses them.
The join script opens ssh and ports 50070 to 50079 to the local network and authorizes the coordinator key. Run it only on a network you trust.
