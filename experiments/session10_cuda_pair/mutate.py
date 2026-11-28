"""Mutation check (session 10, V2): break the new code on purpose, one change at a time, and record which test fails first.
Usage: python experiments/session10_cuda_pair/mutate.py <out.json> [name substring ...]
Each mutation edits one file, runs a pytest selection, records the first failing test, and writes the file back."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(Path.home() / "dev/nocturne-torch/bin/python")
E = "nocturne/engines/torch_tp/"
TL = "tests/bench/test_session10_timeline.py"
CO = "tests/bench/test_session10_coordinator.py"
SH = "tests/bench/test_session10_shipping.py"
NOGGUF = ["-k", "not gguf"]

# name, file, old, new, pytest args
MUTATIONS = [
    ("clock: offset sign (local minus remote)", E + "clock.py",
     '"offset_mono_ns": rm - (t0 + rtt // 2), "offset_wall_ns": rw - (w0 + rtt // 2)',
     '"offset_mono_ns": (t0 + rtt // 2) - rm, "offset_wall_ns": (w0 + rtt // 2) - rw', [TL]),
    ("clock: event times shifted the wrong way", E + "clock.py",
     "return t_remote_ns - offset_at(t_remote_ns, start, end)", "return t_remote_ns + offset_at(t_remote_ns, start, end)", [TL]),
    ("clock: keep the first round instead of the smallest round trip", E + "clock.py",
     "t0, t1, w0, rw, rm = min(samples, key=lambda s: s[1] - s[0])", "t0, t1, w0, rw, rm = samples[0]", [TL]),
    ("clock: OS comparison with the hosts swapped", E + "clock.py",
     "pred = round((coord_sntp_offset_s - rank_sntp_offset_s) * 1e9)", "pred = round((rank_sntp_offset_s - coord_sntp_offset_s) * 1e9)", [TL, CO]),
    ("timesync: SNTP offset sign", "nocturne/bench/probes/timesync.py",
     "return ((t2 - t1) + (t3 - t4)) / 2, (t4 - t1) - (t3 - t2)", "return ((t1 - t2) + (t4 - t3)) / 2, (t4 - t1) - (t3 - t2)", [TL]),
    ("recorder: the cursor does not advance", E + "recorder.py",
     "        self.ev.append((kind, self.t, t, self.step, self.req, peer, nbytes, self.tokens))\n        self.t = t",
     "        self.ev.append((kind, self.t, t, self.step, self.req, peer, nbytes, self.tokens))", [TL]),
    ("timeline: sent and received bytes swapped", E + "timeline.py",
     'side = "sent" if e["kind"] == "send" else "recv"\n                s[side][e["peer"]]', 'side = "recv" if e["kind"] == "send" else "sent"\n                s[side][e["peer"]]', [TL]),
    ("shipping: sha256 of a received slice not checked", E + "shipping.py",
     'if not err and cur["sha"].hexdigest() != sl["sha256"]:', 'if False and cur["sha"].hexdigest() != sl["sha256"]:', [SH] + NOGGUF),
    ("shipping: CRC32 of a chunk not checked", E + "shipping.py",
     "if zlib.crc32(payload) != aux1:", "if False:", [SH] + NOGGUF),
    ("shipping: a cached file is accepted whatever its size", E + "shipping.py",
     'if not (p.is_file() and p.stat().st_size == sl["nbytes"]):', "if not p.is_file():", [SH] + NOGGUF),
    ("shipping: manifest slice bounds one short", E + "shipping.py",
     "return None if s is None else [int(s.start or 0), int(s.stop)]", "return None if s is None else [int(s.start or 0), int(s.stop) - 1]", [SH] + NOGGUF),
    ("shipping: row range used as a column range", E + "shipping.py",
     "return (s, None) if dim == 0 else (None, s)\n\n\ndef tensor_bytes", "return (s, None) if dim == 1 else (None, s)\n\n\ndef tensor_bytes", [SH] + NOGGUF),
    ("shipping: hashes of the manifest not checked before they name a file", E + "shipping.py",
     'if not HEX64.fullmatch(str(sl.get("sha256"))) or', 'if False or', [SH] + NOGGUF),
    ("shipping: a transfer that ends early is accepted (review F5)", E + "shipping.py",
     "if cur is not None or missing_now:", "if False:", [SH] + NOGGUF),
    ("shipping: the temporary file of a broken transfer is left (review F6)", E + "shipping.py",
     "                pass\n            cur[\"tmp\"].unlink(missing_ok=True)", "                pass", [SH] + NOGGUF),
    ("shipping: hash memo kept across a change of the slicing code (review F4)", E + "shipping.py",
     'return (d.get("slices") or {} if d.get("code") == code_stamp() else {}), p', 'return (d.get("slices") or {}), p', [SH] + NOGGUF),
    ("operator: pkill without the option terminator (review F1)", E + "operate.py",
     '+ ["-f", "--", f"--tag {re.escape(tag)}( |$)"]', '+ ["-f", f"--tag {re.escape(tag)}( |$)"]', [CO]),
    ("operator: pkill pattern not anchored at the end of the tag (review F1)", E + "operate.py",
     '+ ["-f", "--", f"--tag {re.escape(tag)}( |$)"]', '+ ["-f", "--", f"--tag {re.escape(tag)}"]', [CO]),
    ("Codex F1: the operator never gives up on a silent coordinator", E + "operate.py",
     "            if stall_s and time.time() - last[0] > stall_s:", "            if False:", [CO, "-k", "sign_of_life or heartbeat or stalled"]),
    ("Codex F1: heartbeat lines do not count as a sign of life", E + "operate.py",
     "                last[0] = time.time()\n                if not line.startswith(ALIVE_MARK):",
     "                if not line.startswith(ALIVE_MARK):\n                    last[0] = time.time()", [CO, "-k", "sign_of_life or heartbeat or stalled"]),
    ("Codex F2: the run id is not checked before the run folder is made", E + "hybrid_run.py",
     "    if not valid_tag(a.run_id):", "    if False:", [CO, "-k", "run_id"]),
    ("Codex F2: dot and dot dot accepted as a run tag", E + "operate.py",
     'TAG_OK.fullmatch(tag) is not None and tag not in (".", "..")', "TAG_OK.fullmatch(tag) is not None", [CO, "-k", "run_id"]),
    ("Codex F3: a slice that is not contiguous is copied whole", E + "shipping.py",
     "    if t.is_contiguous():\n        mv = tensor_bytes(t)", "    if True:\n        mv = tensor_bytes(t)", [SH] + NOGGUF),
    ("Codex F4: device sync without the device", E + "hybrid.py",
     "        torch.cuda.synchronize(device)", "        torch.cuda.synchronize()", [TL]),
    ("Codex F5: the end clock exchange also goes down the pipeline links", E + "hybrid.py",
     "        for gid, link in self.ctl.items():\n            link.send(transport.KIND_CLOCK, aux1=clock.BEGIN)",
     "        self.rank.forward_control(transport.KIND_CLOCK, aux1=clock.BEGIN)\n        for gid, link in self.ctl.items():\n            link.send(transport.KIND_CLOCK, aux1=clock.BEGIN)",
     [TL, "-k", "clock_frames"]),
    ("Codex: the request number on the wire with the recorder off", E + "hybrid.py",
     "((self.req_seq << REQ_SHIFT) if rec else 0)", "(self.req_seq << REQ_SHIFT)", [TL, "-k", "clock_frames"]),
    ("Codex: --verify-cache does not hash the cached file", E + "shipping.py",
     'return not verify or file_hash(p) == sl["sha256"]', "return True", [SH] + NOGGUF),
    ("rank: run tag from the wire used as a path (review F7)", E + "hybrid.py",
     'return os.path.join(os.path.expanduser("~/.nocturne/runs"), tag if ok else "untagged")',
     'return os.path.join(os.path.expanduser("~/.nocturne/runs"), tag or "untagged")', [TL]),
    ("recorder: no compute span at the end of a step of a plain group rank (review F17)", E + "hybrid.py",
     "        if rec and not self.down_order:", "        if False and not self.down_order:", [TL]),
    ("recorder: time stamps from the slewed monotonic clock", E + "clock.py",
     "    return time.clock_gettime_ns(_RAW) if _RAW is not None else time.monotonic_ns()", "    return time.monotonic_ns()", [TL]),
    ("transport: a send does not wait out ENOBUFS", E + "transport.py",
     "            if e.errno != errno.ENOBUFS:\n                raise", "            if True:\n                raise", [TL]),
    ("planner: coordinator host taken from the last stage", E + "plan.py",
     "    host = p.stages[0].ranks[0].host\n    if p.coordinator_host and p.coordinator_host != host:",
     "    host = p.stages[-1].ranks[0].host\n    if p.coordinator_host and p.coordinator_host != host:", [CO, "tests/bench/test_hybrid.py"]),
    ("operator: coordinator process started on the host of rank 1", E + "operate.py",
     "p = self.runner.spawn(self.peer(self.coord), self.coordinate_argv(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)",
     "p = self.runner.spawn(self.peer(self.slots[1].host), self.coordinate_argv(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)", [CO]),
    ("coordinator: the host check accepts any machine", E + "coordinate.py",
     "if want and want.split(\".\")[0].lower() != me.lower():", "if False:", [CO]),
    ("operator: model path of the operator instead of the coordinator host", E + "operate.py",
     "self.model_dir = model_dir_of(self.placement.model, self.peers, self.coord)", "self.model_dir = model_dir_of(self.placement.model, self.peers)", [CO]),
]


def first_failure(out: str) -> str | None:
    for line in out.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            return line.split(" - ")[0].split(" ", 1)[1]
    return None


def main() -> int:
    out_path, only = sys.argv[1], sys.argv[2:]
    res = []
    for name, file, old, new, args in MUTATIONS:
        if only and not any(o in name for o in only):
            continue
        p = REPO / file
        src = p.read_text()
        if src.count(old) != 1:
            res.append({"mutation": name, "file": file, "result": f"NOT APPLIED: the text occurs {src.count(old)} times"})
            print(res[-1], flush=True)
            continue
        p.write_text(src.replace(old, new))
        try:
            r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-o", "faulthandler_timeout=300", *args],
                               cwd=REPO, capture_output=True, text=True, timeout=900)
            fail = first_failure(r.stdout)
            res.append({"mutation": name, "file": file, "caught": r.returncode != 0, "first_failing_test": fail,
                        "tail": None if fail else r.stdout[-400:]})
        except subprocess.TimeoutExpired:
            res.append({"mutation": name, "file": file, "caught": True, "first_failing_test": "timeout after 900 s (a test hung)"})
        finally:
            p.write_text(src)
        print(res[-1], flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))
    missed = [r["mutation"] for r in res if not r.get("caught")]
    print(f"{len(res)} mutations, {len(res) - len(missed)} caught, missed: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
