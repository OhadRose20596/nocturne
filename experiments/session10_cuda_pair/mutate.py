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
OV = "tests/bench/test_session10_overlap.py"
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
     "        self.ev.append((kind, self.t, t, self.step, self.req, peer, nbytes, self.tokens, self.slot, ch))\n        self.t = t",
     "        self.ev.append((kind, self.t, t, self.step, self.req, peer, nbytes, self.tokens, self.slot, ch))", [TL]),
    ("timeline: sent and received bytes swapped", E + "timeline.py",
     'side = "sent" if e["kind"] in SENDS else "recv"\n                s[side][e["peer"]]', 'side = "recv" if e["kind"] in SENDS else "sent"\n                s[side][e["peer"]]', [TL]),
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
     "(((self.req_seq & REQ_MASK) << REQ_SHIFT) if rec else 0)", "((self.req_seq & REQ_MASK) << REQ_SHIFT)", [TL, "-k", "clock_frames"]),
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
    # ---- part 3 (overlap). The four breaks named in 03_BUILD_OVERLAP.md V2 come first.
    ("overlap V2a: the coordinator does not put the slot on the wire", E + "hybrid.py",
     "    return flags | (req & REQ_MASK) << REQ_SHIFT | slot << SLOT_SHIFT", "    return flags | (req & REQ_MASK) << REQ_SHIFT | 0 << SLOT_SHIFT", [OV]),
    ("overlap V2a: a rank ignores the slot of a frame", E + "hybrid.py",
     "        slot = flags >> SLOT_SHIFT\n        if ids is not None:", "        slot = 0\n        if ids is not None:", [OV]),
    ("overlap V2a: a head result travels without its slot", E + "hybrid.py",
     "        aux0 = idx | (flags >> SLOT_SHIFT) << SLOT_SHIFT", "        aux0 = idx", [OV]),
    ("overlap V2a: a head result of the sender thread travels without its slot", E + "hybrid.py",
     "                    aux0 = idx | slot << SLOT_SHIFT", "                    aux0 = idx", [OV]),
    ("overlap V2b: under interleave the second request is summed in another order", E + "hybrid.py",
     "            acc = parts[0].float()\n            for r in range(1, self.n):\n                acc = acc + parts[r].float()",
     "            order = list(range(self.n))[::-1] if tag else list(range(self.n))\n            acc = parts[order[0]].float()\n"
     "            for r in order[1:]:\n                acc = acc + parts[r].float()", [OV]),
    ("overlap V2b: a pair reads the second sum before the first", E + "model.py",
     "        ha = ha + comm.recv_sum(ta)\n        hb = hb + comm.recv_sum(tb)\n        ta = comm.send_partial(self.mlp_partial(ha), 0)",
     "        hb = hb + comm.recv_sum(tb)\n        ha = ha + comm.recv_sum(ta)\n        ta = comm.send_partial(self.mlp_partial(ha), 0)", [OV]),
    ("overlap V2c: every prefill chunk starts at position 0", E + "hybrid.py",
     '                st["units"].append((ids[a:b], a, flags))', '                st["units"].append((ids[a:b], 0, flags))', [OV]),
    ("overlap V2d: the planner takes the largest rank of a host, not the sum", E + "plan.py",
     '        h["planned"] += s.planned.get("footprint", s.planned["total"])',
     '        h["planned"] = max(h["planned"], s.planned.get("footprint", s.planned["total"]))', [OV]),
    ("overlap V2d: the planner leaves out the cost of the processes", E + "plan.py",
     '        h["total"] = h["planned"] + h["process_overhead"]', '        h["total"] = h["planned"]', [OV]),
    ("overlap V2d: a host over its budget is accepted", E + "plan.py",
     '        if h["budget"] is not None and h["total"] > h["budget"]:', '        if False:', [OV]),
    ("overlap: the KV cache of slot 0 is written whatever the slot", E + "model.py",
     "        self.k_cache[slot, :, pos0:pos0 + T] = k", "        self.k_cache[0, :, pos0:pos0 + T] = k", [OV]),
    ("overlap: the planner counts one KV cache whatever the slots", E + "plan.py",
     "            kv = kv_slots * L * (kv1 - kv0)", "            kv = L * (kv1 - kv0)", [OV]),
    ("overlap: a prefill chunk that is not the last still gets a head result", E + "hybrid.py",
     "        self.steps += 1\n        if self.spec.head_rows is not None and not flags & FLAG_NOHEAD:\n            res = self.model.head_argmax(h)",
     "        self.steps += 1\n        if self.spec.head_rows is not None:\n            res = self.model.head_argmax(h)", [OV]),
    ("overlap: prefill frame formula counts head frames for every chunk", E + "plan.py",
     '    return n_chunks * (f["total"] - f["head"]) + f["head"]', '    return n_chunks * f["total"]', [OV]),
    ("overlap: a last chunk of one token is left alone", E + "plan.py",
     "    if out[-1][1] - out[-1][0] == 1:", "    if False:", [OV]),
    ("overlap: the pair flag is not sent", E + "hybrid.py",
     "aux = pack_aux(flags | (FLAG_PAIR if pair_first else 0), st[\"req\"] if rec else 0, s)", "aux = pack_aux(flags, st[\"req\"] if rec else 0, s)", [OV]),
    ("overlap: the second large frame of a pair does not wait for the first on its link", E + "hybrid.py",
     "            if prev is not None and not prev.is_alive():\n                prev = None", "            if True:\n                prev = None", [OV]),
    ("overlap: the head on the device takes the last maximum on a tie", E + "model.py",
     "        i = torch.where(lg == m, torch.arange(n, device=lg.device), n).min()", "        i = torch.where(lg == m, torch.arange(n, device=lg.device), -1).max()", [OV]),
    ("overlap: the compute thread does not wait for its partial to reach the host", E + "hybrid.py",
     "        while not h.ready.wait(0.2):\n            self.check()\n        return h.host", "        return h.host", [OV, "-k", "sender_keeps"]),
    ("overlap: a failure of the sender thread is not reported", E + "hybrid.py",
     "        if self.error is not None:\n            raise ConnectionError(", "        if False:\n            raise ConnectionError(", [OV, "-k", "sender_keeps"]),
    ("overlap: a frame is in flight only until the receiver starts to wait", E + "timeline.py",
     '        frames = [[a["t0"], max(a["t0"], b["t1"]), a["bytes"]', '        frames = [[a["t0"], max(a["t0"], b["t0"]), a["bytes"]', [OV]),
    ("overlap: the overlap share counts all compute time", E + "timeline.py",
     '"overlap_share": length(intersect(comp, both)) / total,', '"overlap_share": length(comp) / total,', [OV]),
    ("overlap: the busy share ignores the request window", E + "timeline.py",
     "        comp = intersect(union(comp), win)", "        comp = union(comp)", [OV]),
    ("overlap: sender thread frames are left out of the bytes per link", E + "timeline.py",
     'SENDS = ("send", "tx_send")', 'SENDS = ("send",)', [OV]),
    # ---- after the first review of part 3 (REVIEW_overlap.md)
    ("review F1: no compute span before a hand-over to the sender thread", E + "hybrid.py",
     "            if rec:\n                rec.lap(\"compute\")          # no device sync here", "            if False:\n                rec.lap(\"compute\")          # no device sync here", [OV]),
    ("review F2: frames in flight matched per pair of ranks, not per socket", E + "timeline.py",
     'sends.setdefault((me, e["peer"], e.get("ch")), []).append(e)', 'sends.setdefault((me, e["peer"], None), []).append(e)', [OV]),
    ("review F2: a head frame recorded on the group channel", E + "hybrid.py",
     '            self.rec.lap("send", self.specs[0].name, self.ctl.last_sent, CH_CTL)', '            self.rec.lap("send", self.specs[0].name, self.ctl.last_sent, CH_GROUP)', [OV, TL]),
    ("review F3: a failing sender thread leaves the rank's links open", E + "hybrid.py",
     "            if self.on_fail is not None:    # a compute thread blocked", "            if False:    # a compute thread blocked", [OV, "-k", "failure_of_the_sender"]),
    ("review F4: steps in pairs are not counted", E + "hybrid.py",
     '                    self.schedule["pair_steps"] += 2\n', "", [OV]),
    ("review F7: the scheduler waits for every head result before the next step (nothing in flight at once)", E + "hybrid.py",
     "                    after(s, u, self.rank.step(ids=u[0], pos0=u[2], flags=u[3]))",
     "                    after(s, u, self.rank.step(ids=u[0], pos0=u[2], flags=u[3]))\n                    while table[s] is not None and table[s].get(\"flying\"):\n                        drain(block=True)", [OV]),
    ("review F7: partials are sent on the compute thread although the sender thread is on", E + "hybrid.py",
     "        if self.sender is not None:         # --async-send: the copy and the sends are handed over, this thread goes on",
     "        if False:         # --async-send: the copy and the sends are handed over, this thread goes on", [OV]),
    ("review F6: a frame goes inline whatever the kernel granted", E + "transport.py",
     "        return nbytes <= SMALL_FRAME and outstanding + nbytes <= self.sndbuf // 2", "        return nbytes <= SMALL_FRAME", [OV]),
    ("review F6: inline frames of a pair partner are not counted", E + "hybrid.py",
     "            if prev is None and self.links[r].fits_inline(len(wire), out):", "            if prev is None and self.links[r].fits_inline(len(wire), 0):", [OV]),
    ("review F8: a bit set in a header field of the default path", E + "hybrid.py",
     "        flags = (FLAG_LOGITS if want_logits else 0) | (((self.req_seq & REQ_MASK) << REQ_SHIFT) if rec else 0)",
     "        flags = (FLAG_LOGITS if want_logits else 0) | (((self.req_seq & REQ_MASK) << REQ_SHIFT) if rec else 0) | 1 << SLOT_SHIFT", [OV, "-k", "frames_of_the_engine_before"]),
    ("review F9: the old path writes the request number without a mask", E + "hybrid.py",
     "(((self.req_seq & REQ_MASK) << REQ_SHIFT) if rec else 0)", "((self.req_seq << REQ_SHIFT) if rec else 0)", [OV]),
    ("review F10: a chunk of one token is accepted", E + "plan.py",
     "    if prefill_chunk < 0 or prefill_chunk == 1:", "    if prefill_chunk < 0:", [OV]),
    ("review F5: a host with several processes and no budget is not marked", E + "plan.py",
     '        h["unchecked"] = h["processes"] > 1 and h["budget"] is None', '        h["unchecked"] = False', [OV]),
    ("review F15: a pair plans the work space of one request", E + "plan.py",
     "            if pair and n > 1:\n                ws *= 2\n", "", [OV]),
    # ---- after the second review of part 3 (REVIEW_overlap_pass2.md)
    ("pass 2: frames in flight matched per pair of ranks on both sides (the defect F2 described)", E + "timeline.py",
     'sends.setdefault((me, e["peer"], e.get("ch")), []).append(e)\n            elif e["kind"] in WAITS:\n                recvs.setdefault((e["peer"], me, e.get("ch")), []).append(e)',
     'sends.setdefault((me, e["peer"], None), []).append(e)\n            elif e["kind"] in WAITS:\n                recvs.setdefault((e["peer"], me, None), []).append(e)', [OV]),
    ("pass 2 P1: a frame is not counted against its link", E + "hybrid.py",
     "            self._out[r] = out + len(wire)\n", "", [OV]),
    ("pass 2 P1: only an inline frame is counted (a frame on a helper thread is not)", E + "hybrid.py",
     "            self._out[r] = out + len(wire)\n            if prev is None and self.links[r].fits_inline(len(wire), out):\n"
     "                self.links[r].send(transport.KIND_ALLREDUCE, wire, dt, tuple(xc.shape), decode=dec)\n",
     "            if prev is None and self.links[r].fits_inline(len(wire), out):\n"
     "                self.links[r].send(transport.KIND_ALLREDUCE, wire, dt, tuple(xc.shape), decode=dec)\n                self._out[r] = out + len(wire)\n", [OV]),
    ("pass 2 P2: the count is not lowered when the all-reduce is answered", E + "hybrid.py",
     "                self._out[r] = self._out.get(r, 0) - sent\n", "                pass\n", [OV]),
    ("pass 2 P3: no long prompt compared counts as identical", E + "hybrid_run.py",
     "for p in longs) if longs else None,", "for p in longs) if longs else True,", [OV]),
    ("pass 2 P3: the criterion is met although nothing was compared", E + "hybrid_run.py",
     '"criterion_met": bool(ev) and not any(i in ev_ids for i in missing) and sum(same)', '"criterion_met": sum(same)', [OV]),
    ("pass 2 P3: a reference that cannot be read is found only after the run", E + "hybrid_run.py",
     "    for r in a.reference or []:             # a reference that cannot be read costs a moment here, not a run",
     "    for r in []:             # a reference that cannot be read costs a moment here, not a run", [OV]),
    ("pass 2 P4: the plan command prints the host table without the budgets", E + "__main__.py",
     "        print(format_plan(specs, cfg, p.host_budget_gb))", "        print(format_plan(specs, cfg))", [OV]),
    ("pass 2 P6: the old path records the head wait on the group channel", E + "hybrid.py",
     '                rec.lap("recv_wait", s.name, self.ctl[s.gid].last_recv, CH_CTL)', '                rec.lap("recv_wait", s.name, self.ctl[s.gid].last_recv, CH_GROUP)', [OV]),
    ("pass 2 P6: the old path records the token send on the group channel", E + "hybrid.py",
     '                rec.lap("send", s.name, self.ctl[s.gid].last_sent, CH_CTL)', '                rec.lap("send", s.name, self.ctl[s.gid].last_sent, CH_GROUP)', [OV]),
    ("pass 2 P6: a head frame for a slot outside the table raises an index error", E + "hybrid.py",
     "decode=lambda k, a0, a1: bool(a0 >> SLOT_SHIFT < len(table) and table[a0 >> SLOT_SHIFT] and table[a0 >> SLOT_SHIFT][\"decode\"]))",
     "decode=lambda k, a0, a1: bool(table[a0 >> SLOT_SHIFT] and table[a0 >> SLOT_SHIFT][\"decode\"]))", [OV]),
    ("pass 2 P6: the scheduler counts do not reach the record", E + "coordinate.py",
     '            rec["overlap"] = dict(ov, **co.schedule)', '            rec["overlap"] = dict(ov)', [CO]),
    ("pass 2 P6: run_job does not tell the planner about the pair", E + "coordinate.py",
     '        specs = plan(dims, p, check_budget=not job.get("force"), kv_slots=ov["slots"], pair=ov["tp_interleave"] == 2)\n        rec["planned"]',
     '        specs = plan(dims, p, check_budget=not job.get("force"), kv_slots=ov["slots"])\n        rec["planned"]', [CO]),
    ("pass 2 P6: plan_job does not tell the planner about the pair", E + "coordinate.py",
     '        specs = plan(dims, p, check_budget=not job.get("force"), kv_slots=ov["slots"], pair=ov["tp_interleave"] == 2)\n    except PlanError as e:',
     '        specs = plan(dims, p, check_budget=not job.get("force"), kv_slots=ov["slots"])\n    except PlanError as e:', [CO]),
    ("pass 2 P7: a failing sender sends no error text to the coordinator", E + "hybrid.py",
     "            if self.ctl is not None:        # the coordinator turns an ERROR frame into the text of its record",
     "            if False:        # the coordinator turns an ERROR frame into the text of its record", [OV]),
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
