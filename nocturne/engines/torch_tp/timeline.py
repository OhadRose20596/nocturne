"""Timeline of one recorded run (session 10, A3): reads timeline_<rank>.jsonl.gz and samples_rank_<rank>.jsonl.gz of a run
folder, puts every time on the coordinator's monotonic clock (clock.py), and writes trace.json.gz (Trace Event Format,
opens in ui.perfetto.dev as it is), breakdown.csv (per step and rank: ms per kind, bytes per peer), samples_<host>.jsonl (aligned), and
timeline.html (per rank one compute row and one communication row, colors by peer; bytes per link).
Overlap (session 10 part 3): a frame is in flight from the start of its send to the end of the receiver's wait for it;
the k-th send on a directed link belongs to the k-th receive. Per rank and per device: the busy share (computing) and
the overlap share (computing while a frame it sent, or one addressed to it, is in flight). docs/MEASUREMENT.md.
Usage: python -m nocturne.engines.torch_tp.timeline <run_dir> [--req N] [--from-step K] [--steps 3]
The clock offsets and the request list come from record.json (keys `clock`, `requests`) in the run folder."""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

from . import clock
from .recorder import ASYNC_KINDS, COLS, KINDS

WAITS = ("recv_wait", "allreduce_wait")
SENDS = ("send", "tx_send")            # a frame leaves: on the compute thread, or on the sender thread (--async-send)
ALL_KINDS = KINDS + ASYNC_KINDS
TEMPLATE = Path(__file__).with_name("timeline_page.html")


# ---- loading and alignment -----------------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> tuple[dict, list]:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        head = json.loads(f.readline())
        return head, [json.loads(l) for l in f if l.strip()]


def load(run_dir: str | Path, clocks: dict | None = None) -> dict:
    """{ranks: [{meta, events}], samples: [{meta, rows}]} with times converted to the coordinator clock (float ns)."""
    d = Path(run_dir)
    clocks = clocks or {}
    ranks, samples = [], []
    for p in sorted(d.glob("timeline_*.jsonl*")):
        meta, ev = read_jsonl(p)
        c = clocks.get(meta["rank"]) or {}
        st, en = c.get("start") or clock.ZERO, c.get("end")
        events = [dict(zip(COLS, e)) for e in ev]
        for e in events:
            e["t0"] = clock.to_coordinator(e.pop("t0_ns"), st, en)
            e["t1"] = clock.to_coordinator(e.pop("t1_ns"), st, en)
        ranks.append({"meta": meta, "events": events, "aligned": bool(c.get("start")) or meta["gid"] == 0})
    ranks.sort(key=lambda r: r["meta"]["gid"])
    for p in sorted(d.glob("samples_rank_*.jsonl*")):
        meta, rows = read_jsonl(p)
        c = clocks.get(meta["rank"]) or {}
        st, en = c.get("start") or clock.ZERO, c.get("end")
        for r in rows:
            r["t"] = clock.to_coordinator(r["t"], st, en)
        samples.append({"meta": meta, "rows": rows})
    return {"ranks": ranks, "samples": samples}


# ---- per step breakdown ----------------------------------------------------------------------------------------------------
def breakdown(run: dict, requests: list[dict] | None = None) -> list[dict]:
    rid = {r["req"]: r["rid"] for r in requests or []}
    rows = []
    for rk in run["ranks"]:
        m = rk["meta"]
        steps: dict[int, dict] = {}
        for e in rk["events"]:
            if e["step"] < 0:
                continue
            s = steps.setdefault(e["step"], {"req": e["req"], "rid": rid.get(e["req"], ""), "step": e["step"], "tokens": e["tokens"],
                                             "phase": "decode" if e["tokens"] == 1 else "prefill", "rank": m["rank"], "host": m["host"], "slot": e.get("slot", 0),
                                             "device": m["device"], "t0": e["t0"], "t1": e["t1"], "sent": {}, "recv": {},
                                             **{k: 0.0 for k in ALL_KINDS}})
            s[e["kind"]] += e["t1"] - e["t0"]
            s["t0"], s["t1"] = min(s["t0"], e["t0"]), max(s["t1"], e["t1"])
            if e["bytes"]:
                side = "sent" if e["kind"] in SENDS else "recv"
                s[side][e["peer"]] = s[side].get(e["peer"], 0) + e["bytes"]
        rows.extend(steps[k] for k in sorted(steps))
    return rows


def write_breakdown(rows: list[dict], path: Path) -> None:
    cols = ["req", "rid", "step", "tokens", "phase", "rank", "host", "device", "total_ms"] + [f"{k}_ms" for k in KINDS] + \
           ["sent_bytes", "recv_bytes", "sent_by_peer", "recv_by_peer", "slot"] + [f"{k}_ms" for k in ASYNC_KINDS]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            fmt = lambda d: ";".join(f"{k}={v}" for k, v in sorted(d.items()))
            w.writerow([r["req"], r["rid"], r["step"], r["tokens"], r["phase"], r["rank"], r["host"], r["device"],
                        f"{(r['t1'] - r['t0']) / 1e6:.4f}"] + [f"{r[k] / 1e6:.4f}" for k in KINDS] +
                       [sum(r["sent"].values()), sum(r["recv"].values()), fmt(r["sent"]), fmt(r["recv"]), r["slot"]] +
                       [f"{r[k] / 1e6:.4f}" for k in ASYNC_KINDS])


def summary(rows: list[dict]) -> dict:
    """Decode steps only. Per rank: steps and mean ms per step by kind. Per directed link: bytes per step and total."""
    dec = [r for r in rows if r["phase"] == "decode"]
    per_rank, links = {}, {}
    for r in dec:
        a = per_rank.setdefault(r["rank"], {"host": r["host"], "device": r["device"], "steps": 0, "total_ms": 0.0, **{k: 0.0 for k in KINDS}})
        a["steps"] += 1
        a["total_ms"] += (r["t1"] - r["t0"]) / 1e6
        for k in KINDS:
            a[k] += r[k] / 1e6
        for peer, b in r["sent"].items():
            l = links.setdefault(f"{r['rank']} -> {peer}", {"from": r["rank"], "to": peer, "bytes": 0})
            l["bytes"] += b
    steps = max((a["steps"] for a in per_rank.values()), default=0)
    for a in per_rank.values():
        n = a.pop("steps")
        a["decode_steps"] = n
        for k in ("total_ms",) + KINDS:
            a[k] = a[k] / n if n else None
    for l in links.values():
        l["bytes_per_step"] = l["bytes"] / steps if steps else None
    total = sum(l["bytes"] for l in links.values())
    return {"decode_steps": steps, "per_rank_ms_per_step": per_rank, "links": links, "decode_bytes_total": total,
            "decode_bytes_per_step": total / steps if steps else None}


def all_bytes(run: dict) -> dict:
    """Bytes sent per directed link over the whole run (prefill and decode), from the send events."""
    out: dict[str, int] = {}
    for rk in run["ranks"]:
        for e in rk["events"]:
            if e["kind"] in SENDS and e["bytes"]:
                k = f"{rk['meta']['rank']} -> {e['peer']}"
                out[k] = out.get(k, 0) + e["bytes"]
    return out


# ---- overlap: compute while data is in flight ------------------------------------------------------------------------------
def union(iv: list) -> list:
    """Sorted, disjoint intervals covering the same points as `iv` (a list of (a, b))."""
    out = []
    for a, b in sorted(x for x in iv if x[1] > x[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intersect(x: list, y: list) -> list:
    """Intersection of two sorted disjoint interval lists."""
    out, i, j = [], 0, 0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            out.append([a, b])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def length(iv: list) -> float:
    return sum(b - a for a, b in iv)


def in_flight(run: dict) -> dict:
    """{"A -> B": {"frames": [[t0, t1, bytes, req, slot], ...], "sent": n, "received": n, "bytes_match": bool}}. A
    frame is in flight from the start of its send at A until B has read it (the end of B's wait for it). The k-th send
    on a socket is the k-th receive on it: a socket keeps its order and each side records its frames in that order. Two
    ranks can share two sockets (a first stage rank with head rows has a group link and a control link to the
    coordinator), so the match is per socket: the channel column of the span (`ch`). Files from before the channel
    column match per pair of ranks."""
    sends, recvs = {}, {}
    for rk in run["ranks"]:
        me = rk["meta"]["rank"]
        for e in rk["events"]:
            if not e["bytes"] or not e["peer"]:
                continue
            if e["kind"] in SENDS:
                sends.setdefault((me, e["peer"], e.get("ch")), []).append(e)
            elif e["kind"] in WAITS:
                recvs.setdefault((e["peer"], me, e.get("ch")), []).append(e)
    out = {}
    for key in sorted(set(sends) | set(recvs), key=lambda k: (k[0], k[1], k[2] or "")):
        s = sorted(sends.get(key, []), key=lambda e: e["t0"])
        r = sorted(recvs.get(key, []), key=lambda e: e["t1"])
        frames = [[a["t0"], max(a["t0"], b["t1"]), a["bytes"], a["req"], a.get("slot", 0)] for a, b in zip(s, r)]
        o = out.setdefault(f"{key[0]} -> {key[1]}", {"from": key[0], "to": key[1], "frames": [], "sent": 0, "received": 0, "bytes_match": True})
        o["frames"] += frames
        o["sent"] += len(s)
        o["received"] += len(r)
        o["bytes_match"] = o["bytes_match"] and len(s) == len(r) and all(a["bytes"] == b["bytes"] for a, b in zip(s, r))
    for o in out.values():
        o["frames"].sort()
    return out


def overlap(run: dict, requests: list[dict] | None = None, flights: dict | None = None) -> dict | None:
    """Busy share and overlap share per rank and per device, over the time in which at least one request is in work
    (the union of the requests' intervals on the coordinator clock; without a request list, the coordinator's steps).
    busy: computing. overlap: computing while a frame the rank sent (out), or one addressed to it (in), is in flight."""
    flights = in_flight(run) if flights is None else flights
    win = union([(r["t0_mono_ns"], r["t1_mono_ns"]) for r in requests or [] if r.get("t1_mono_ns")])
    if not win and run["ranks"]:
        ev = [e for e in run["ranks"][0]["events"] if e["step"] >= 0]
        win = [[min(e["t0"] for e in ev), max(e["t1"] for e in ev)]] if ev else []
    total = length(win)
    if not total:
        return None

    def shares(comp: list, out_iv: list, in_iv: list) -> dict:
        comp = intersect(union(comp), win)
        o, i = intersect(union(out_iv), win), intersect(union(in_iv), win)
        both = union(o + i)
        c = length(comp)
        return {"busy_share": c / total, "overlap_share": length(intersect(comp, both)) / total,
                "overlap_out_share": length(intersect(comp, o)) / total, "overlap_in_share": length(intersect(comp, i)) / total,
                "in_flight_share": length(both) / total, "compute_ms": c / 1e6}

    per_rank, dev = {}, {}
    for rk in run["ranks"]:
        m = rk["meta"]
        comp = [(e["t0"], e["t1"]) for e in rk["events"] if e["kind"] == "compute"]
        o = [(f[0], f[1]) for l in flights.values() if l["from"] == m["rank"] for f in l["frames"]]
        i = [(f[0], f[1]) for l in flights.values() if l["to"] == m["rank"] for f in l["frames"]]
        per_rank[m["rank"]] = dict(shares(comp, o, i), host=m["host"], device=m["device"])
        d = dev.setdefault(f"{m['host']} {m['device']}", {"ranks": [], "comp": [], "out": [], "in": []})
        d["ranks"].append(m["rank"])
        d["comp"] += comp
        d["out"] += o
        d["in"] += i
    per_device = {k: dict(shares(d["comp"], d["out"], d["in"]), ranks=d["ranks"]) for k, d in dev.items()}
    links = {k: {"frames": len(l["frames"]), "sent": l["sent"], "received": l["received"], "bytes_match": l["bytes_match"],
                 "in_flight_ms_mean": length([(f[0], f[1]) for f in l["frames"]]) / len(l["frames"]) / 1e6 if l["frames"] else None}
             for k, l in flights.items()}
    return {"window_ms": total / 1e6, "per_rank": per_rank, "per_device": per_device, "links": links,
            "unmatched_links": sorted(k for k, l in flights.items() if l["sent"] != l["received"] or not l["bytes_match"]),
            "async_send": any(e["kind"] in ASYNC_KINDS for rk in run["ranks"] for e in rk["events"])}


# ---- samples ---------------------------------------------------------------------------------------------------------------
def write_host_samples(run: dict, d: Path) -> dict:
    """samples_<host>.jsonl: every sampler row of the host's ranks on the coordinator clock. Returns per host means."""
    by_host: dict[str, list] = {}
    for s in run["samples"]:
        for r in s["rows"]:
            by_host.setdefault(s["meta"]["host"], []).append(dict(r, rank=s["meta"]["rank"]))
    out = {}
    for host, rows in by_host.items():
        rows.sort(key=lambda r: r["t"])
        with (d / f"samples_{host}.jsonl").open("w") as f:
            f.write(json.dumps({"host": host, "clock": "coordinator monotonic ns", "rows": len(rows)}) + "\n")
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    for s in run["samples"]:
        m, rows = s["meta"], [r for r in s["rows"] if "pc" in r]
        h = out.setdefault(m["host"], {"ranks": {}, "notes": []})
        h["notes"] += m.get("notes", [])
        if len(rows) >= 2:
            dt = (rows[-1]["t"] - rows[0]["t"]) / 1e9
            h["ranks"][m["rank"]] = {"process_cpu_cores": (rows[-1]["pc"] - rows[0]["pc"]) / dt if dt else None, "samples": len(rows),
                                     "interval_ms_mean": dt / (len(rows) - 1) * 1000}
            if m.get("host_level") and "hb" in rows[0]:
                h["host_cpu_busy_fraction"] = (rows[-1]["hb"] - rows[0]["hb"]) / dt / (m.get("ncpu") or 1) if dt else None
                net = {}
                for n, (rx, tx) in (rows[-1].get("net") or {}).items():
                    rx0, tx0 = rows[0]["net"][n]
                    net[n] = {"rx_bytes": rx - rx0, "tx_bytes": tx - tx0}
                h["net"] = net
        gpu = [r for r in s["rows"] if "gu" in r]
        if gpu:
            per = {}
            for r in gpu:
                per.setdefault(str(r.get("gi", 0)), []).append(r)
            h["gpu"] = {i: {"util_mean_percent": sum(r["gu"] for r in rs) / len(rs), "mem_used_max_mib": max(r["gm"] for r in rs),
                            "samples": len(rs)} for i, rs in sorted(per.items())}
            h["gpu_util_mean_percent"] = sum(v["util_mean_percent"] for v in h["gpu"].values()) / len(h["gpu"])
            h["gpu_mem_used_max_mib"] = max(v["mem_used_max_mib"] for v in h["gpu"].values())
            h["gpu_samples"] = len(gpu)
    return out


# ---- trace.json ------------------------------------------------------------------------------------------------------------
def trace(run: dict) -> dict:
    hosts = []
    for rk in run["ranks"]:
        if rk["meta"]["host"] not in hosts:
            hosts.append(rk["meta"]["host"])
    ev = [{"ph": "M", "name": "process_name", "pid": i, "args": {"name": h}} for i, h in enumerate(hosts)]
    for rk in run["ranks"]:
        m = rk["meta"]
        pid = hosts.index(m["host"])
        for row, tid in (("compute", 2 * m["gid"]), ("communication", 2 * m["gid"] + 1)):
            ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": f"{m['rank']} {row}"}})
        for e in rk["events"]:
            comm = e["kind"] in SENDS + WAITS + ("host_copy", "tx_wait")
            name = e["kind"] + (f" {e['peer']}" if e["peer"] else "")
            ev.append({"ph": "X", "name": name, "cat": e["kind"], "pid": pid, "tid": 2 * m["gid"] + (1 if comm else 0),
                       "ts": e["t0"] / 1000, "dur": max(0.0, (e["t1"] - e["t0"]) / 1000),
                       "args": {"step": e["step"], "req": e["req"], "peer": e["peer"], "bytes": e["bytes"], "tokens": e["tokens"],
                                "slot": e.get("slot", 0)}})
    for s in run["samples"]:
        m = s["meta"]
        if m["host"] not in hosts:
            continue
        pid, prev = hosts.index(m["host"]), None
        for r in s["rows"]:
            if "gu" in r:
                ev.append({"ph": "C", "name": f"gpu {r.get('gi', 0)} util % ({m['host']})", "pid": pid, "ts": r["t"] / 1000, "args": {"util": r["gu"]}})
            elif prev is not None and r["t"] > prev["t"]:
                dt = (r["t"] - prev["t"]) / 1e9
                ev.append({"ph": "C", "name": f"cpu cores {m['rank']}", "pid": pid, "ts": r["t"] / 1000, "args": {"process": (r["pc"] - prev["pc"]) / dt}})
            if "pc" in r:
                prev = r
    return {"traceEvents": ev, "displayTimeUnit": "ms"}


# ---- timeline.html -----------------------------------------------------------------------------------------------------------
def pick_window(rows: list[dict], req: int | None, from_step: int | None, steps: int) -> tuple[int, int] | None:
    """(first step, last step) of `steps` consecutive decode steps: by default from the request with the most decode
    steps, starting 8 steps in (the first steps of a request are not typical)."""
    co = [r for r in rows if r["rank"] == rows[0]["rank"] and r["phase"] == "decode"] if rows else []
    if not co:
        return None
    by_req: dict[int, list[int]] = {}
    for r in co:
        by_req.setdefault(r["req"], []).append(r["step"])
    if req is None:
        req = max(by_req, key=lambda k: len(by_req[k]))
    lst = sorted(by_req.get(req, []))
    if not lst:
        return None
    a = from_step if from_step is not None else lst[min(8, max(0, len(lst) - steps))]
    win = [s for s in lst if s >= a][:steps]
    return (win[0], win[-1]) if win else None


def page_data(run: dict, rows: list[dict], summ: dict, clocks: dict, hosts: dict, info: dict, window: tuple[int, int] | None,
              flights: dict | None = None, over: dict | None = None) -> dict:
    names = [rk["meta"]["rank"] for rk in run["ranks"]]
    ranks = [{"name": rk["meta"]["rank"], "host": rk["meta"]["host"], "device": rk["meta"]["device"], "aligned": rk["aligned"]} for rk in run["ranks"]]
    spans, fl, w = [], [], None
    if window:
        co = run["ranks"][0]
        in_win = [e for e in co["events"] if window[0] <= e["step"] <= window[1]]
        t0, t1 = min(e["t0"] for e in in_win), max(e["t1"] for e in in_win)
        w = {"first_step": window[0], "last_step": window[1], "ms": (t1 - t0) / 1e6, "req": in_win[0]["req"]}
        marks = sorted({min(e["t0"] for e in in_win if e["step"] == s) for s in range(window[0], window[1] + 1)})
        w["step_starts_ms"] = [(m - t0) / 1e6 for m in marks]
        for i, rk in enumerate(run["ranks"]):
            for e in rk["events"]:
                if e["t1"] <= t0 or e["t0"] >= t1 or e["kind"] == "other":
                    continue
                a, b = max(e["t0"], t0), min(e["t1"], t1)
                spans.append([i, ALL_KINDS.index(e["kind"]), names.index(e["peer"]) if e["peer"] in names else -1,
                              round((a - t0) / 1e6, 4), round((b - t0) / 1e6, 4), e["bytes"], e["step"], e["req"], e.get("slot", 0)])
        for l in (flights or {}).values():
            if l["from"] not in names or l["to"] not in names:
                continue
            for a, b, nb, rq, sl in l["frames"]:
                if b > t0 and a < t1:
                    fl.append([names.index(l["from"]), names.index(l["to"]), round((max(a, t0) - t0) / 1e6, 4),
                               round((min(b, t1) - t0) / 1e6, 4), nb, rq, sl])
    per_step = []
    dec = [r for r in rows if r["phase"] == "decode"]
    for i, n in enumerate(names):
        mine = [r for r in dec if r["rank"] == n]
        k = len(mine) or 1
        seg = {"compute": sum(r["compute"] for r in mine) / k / 1e6, "host_copy": sum(r["host_copy"] for r in mine) / k / 1e6,
               "other": sum(r["other"] for r in mine) / k / 1e6, "send": {}, "wait": {}}
        for rk in run["ranks"][i:i + 1]:
            for e in rk["events"]:
                if e["tokens"] == 1 and e["kind"] in ("send",) + WAITS and e["peer"] in names:      # the sender thread's spans overlap these: not in the bar
                    side = "send" if e["kind"] == "send" else "wait"
                    seg[side][e["peer"]] = seg[side].get(e["peer"], 0.0) + (e["t1"] - e["t0"]) / k / 1e6
        per_step.append(seg)
    links = [{"from": l["from"], "to": l["to"], "bytes_per_step": l["bytes_per_step"], "bytes": l["bytes"]} for l in summ["links"].values()]
    return {"info": info, "ranks": ranks, "kinds": list(ALL_KINDS), "window": w, "spans": spans, "flights": fl, "overlap": over,
            "per_step": per_step, "links": links,
            "decode_steps": summ["decode_steps"], "decode_bytes_per_step": summ["decode_bytes_per_step"],
            "decode_bytes_total": summ["decode_bytes_total"], "all_bytes": all_bytes(run), "clock": clocks, "hosts": hosts}


def write_html(data: dict, path: Path) -> None:
    page = TEMPLATE.read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(page.replace("/*DATA*/null", blob))


# ---- entry -------------------------------------------------------------------------------------------------------------------
def build(run_dir: str | Path, clocks: dict | None = None, requests: list[dict] | None = None, info: dict | None = None,
          req: int | None = None, from_step: int | None = None, steps: int = 3) -> dict | None:
    """Write the four outputs into the run folder; returns the summary that goes into record.json (None without events)."""
    d = Path(run_dir)
    run = load(d, clocks)
    if not run["ranks"]:
        return None
    rows = breakdown(run, requests)
    write_breakdown(rows, d / "breakdown.csv")
    summ = summary(rows)
    hosts = write_host_samples(run, d)
    with gzip.open(d / "trace.json.gz", "wt") as f:
        json.dump(trace(run), f, separators=(",", ":"))
    window = pick_window(rows, req, from_step, steps)
    flights = in_flight(run)
    over = overlap(run, requests, flights)
    write_html(page_data(run, rows, summ, clocks or {}, hosts, info or {}, window, flights, over), d / "timeline.html")
    summ["overlap"] = over
    summ.update({"hosts": hosts, "events": {rk["meta"]["rank"]: len(rk["events"]) for rk in run["ranks"]}, "all_bytes": all_bytes(run),
                 "files": ["trace.json.gz", "breakdown.csv", "timeline.html"] + sorted(p.name for p in d.glob("timeline_*.jsonl*")) +
                          sorted(p.name for p in d.glob("samples_*.jsonl*")),
                 "not_aligned": [rk["meta"]["rank"] for rk in run["ranks"] if not rk["aligned"]]})
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--req", type=int)
    ap.add_argument("--from-step", type=int)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args(argv)
    rec_path = Path(a.run_dir) / "record.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    info = {"run_id": rec.get("run_id"), "model": (rec.get("model") or {}).get("key"), "placement": (rec.get("placement") or {}).get("file"),
            "date": rec.get("date")}
    s = build(a.run_dir, rec.get("clock"), rec.get("requests"), info, a.req, a.from_step, a.steps)
    if s is None:
        print(f"no timeline_*.jsonl.gz in {a.run_dir}")
        return 1
    print(f"decode steps {s['decode_steps']}, bytes per decode step {s['decode_bytes_per_step']}, wrote {', '.join(s['files'][:3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
