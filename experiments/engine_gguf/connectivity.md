# Engine GGUF part 2: connectivity record (2026-11-22)

Coordinator: M5, `192.168.1.242` (Wi-Fi, en0). Receiver `16_join_apply.py listen`, reports kept in `~/.nocturne/join/` (outside the repo, they hold host and user names).

| host | report | repo SHA | torch | device | llama.cpp | link type | notes |
|---|---|---|---|---|---|---|---|
| m5 | `nocturne_join_MMF4FWRG4R.txt`, fix mode, run on the coordinator | bbbfd94 | 2.14.1 | MPS 24 GB | 1e7bcf3da Metal | wifi | AWDL hold-down started (sudo, by Ohad). Remote Login was turned on by the script because it was run without `--self` |
| air | `Mac_20261122-161042.txt` (check mode) | b7cd353 then bbbfd94 (the repo is the iCloud copy of the M5 folder) | 2.15.0 | MPS 16 GB | 1e7bcf3 Metal | wifi | `gguf 0.19.0` installed with uv, torch untouched. caffeinate started over ssh, AWDL hold-down by Ohad (second report `Mac_20261122-162133.txt`: awdl FIXED, 0 failures) |
| gtx | `Ohad_20261122-161830.txt` (WSL side only, check mode) | b7cd353 then bbbfd94 (own clone `~/dev/nocturne`) | 2.14.1+cu126 | GTX 1070 8 GB, CUDA | not installed (not needed by the torch engine) | wifi (second report, `Ohad_20261122-162343.txt`, from the Windows adapter media type; the first one said unknown) | `gguf` installed by Ohad. Windows ran `15_join.ps1` in fix mode afterwards (1 failed check: llama.cpp not installed, not needed here): no sleep on AC power, WSL keep-alive (`sleep infinity` seen running). ICMP is dropped by Windows, ssh on 2222 works |

`16_join_apply.py apply` for both reports: no change to `peers.yaml` or the ssh config (addresses unchanged). For gtx it printed "does not say whether Wi-Fi or wired" and left `networks` as they are, as designed. Diffs: none.
Both Macs and gtx kept awake during the runs (caffeinate and the AWDL hold-down on the Macs, `sleep infinity` on gtx). Checked with `pgrep` and `ifconfig awdl0` (no UP flag) before the first run.

Models copied with `scp`, sha256 equal on both ends: Qwen2.5-7B Q8_0 `862574978752413e9d54e4d104c455797f9c7e477ee256a46e2d8a31f9242cd9` (air, gtx), Qwen2.5-7B Q4_K_M `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` (gtx; the Air already had it with the same hash). The 32B Q8_0 file is not copied yet: it is 34.8 GB, over the 10 GB rule, and the Air has 39 GB free before the 7B copy.

A3 on gtx (CUDA): `NOCTURNE_TEST_DEVICE=cuda pytest tests/bench/test_engine_gguf.py -k "random or chunked or one_two or name_map or unsupported or unknown"`: 34 passed. The same dequant tests also pass on MPS (33).

Link records: `runs/<run id>.link.json`, TCP connect to the Air on port 22 and to gtx on port 2222, 20 samples each, taken right before every run.
