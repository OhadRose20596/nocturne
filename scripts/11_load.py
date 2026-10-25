#!/usr/bin/env python3
"""Throughput under load. Drives llama-server instances with C concurrent clients, round-robin across URLs.
Reports Q (output tokens of completed requests per second), req/s, completion rate, latency p50/p95. Stdlib only.
Usage: python3 scripts/11_load.py --url http://A:8080 [--url http://B:8080] --concurrency 4 --requests 32 --prompt-tokens 128 --max-tokens 64 --label NAME
Writes experiments/r3/load/<run_id>/{requests.jsonl,summary.json,summary.md}.
"""
import argparse, json, os, sys, threading, time, urllib.request, urllib.error, statistics, datetime, socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "r3" / "load"

WORDS = ("the quick brown fox jumps over the lazy dog while seven silent owls watch from a tall oak "
         "near the river where small boats drift slowly under a pale evening sky ").split()

def make_prompt(n_tokens):
    return " ".join(WORDS[i % len(WORDS)] for i in range(n_tokens))

def one_request(url, prompt, max_tokens, timeout):
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        t1 = time.perf_counter()
        tim = data.get("timings", {}) or {}
        return {"ok": True, "url": url, "t_start": t0, "t_end": t1, "latency_s": t1 - t0,
                "tokens_predicted": int(data.get("tokens_predicted", tim.get("predicted_n", 0)) or 0),
                "prompt_n": int(tim.get("prompt_n", 0) or 0),
                "prompt_ms": float(tim.get("prompt_ms", 0) or 0), "predicted_ms": float(tim.get("predicted_ms", 0) or 0),
                "stop": data.get("stop_type") or data.get("stopped_limit")}
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, ValueError) as e:
        t1 = time.perf_counter()
        return {"ok": False, "url": url, "t_start": t0, "t_end": t1, "latency_s": t1 - t0, "tokens_predicted": 0, "error": str(e)[:200]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True, help="llama-server base URL; repeat for independent replicas")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=0, help="total requests (0 = use --duration)")
    ap.add_argument("--duration", type=float, default=0, help="seconds to keep issuing requests")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--warmup", type=int, default=2, help="requests per URL to discard before the window")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    if not a.requests and not a.duration:
        a.requests = 8 * a.concurrency

    prompt = make_prompt(a.prompt_tokens)
    host = socket.gethostname().split(".")[0]
    run_id = f"{datetime.datetime.utcnow():%Y%m%d-%H%M%S}_{host}_load_{a.label}_c{a.concurrency}"
    d = OUT / run_id; d.mkdir(parents=True, exist_ok=True)

    for u in a.url:
        for _ in range(a.warmup):
            one_request(u, prompt, a.max_tokens, a.timeout)

    results, lock = [], threading.Lock()
    counter = {"issued": 0}
    t_begin = time.perf_counter()

    def worker(idx):
        while True:
            with lock:
                if a.requests and counter["issued"] >= a.requests:
                    return
                if a.duration and time.perf_counter() - t_begin >= a.duration:
                    return
                i = counter["issued"]; counter["issued"] += 1
            url = a.url[i % len(a.url)]
            r = one_request(url, prompt, a.max_tokens, a.timeout)
            r["idx"] = i; r["worker"] = idx
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    t_finish = time.perf_counter()

    ok = [r for r in results if r["ok"]]
    window = (max(r["t_end"] for r in results) - min(r["t_start"] for r in results)) if results else 0.0
    toks = sum(r["tokens_predicted"] for r in ok)
    lat = sorted(r["latency_s"] for r in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))] if lat else None
    per_url = {u: sum(1 for r in ok if r["url"] == u) for u in a.url}
    summary = {
        "run_id": run_id, "label": a.label, "urls": a.url, "concurrency": a.concurrency,
        "prompt_tokens_requested": a.prompt_tokens, "max_tokens": a.max_tokens,
        "requests_issued": len(results), "requests_completed": len(ok), "completion_rate": (len(ok) / len(results)) if results else 0,
        "window_s": window, "output_tokens": toks, "Q_tok_per_s": (toks / window) if window else 0,
        "req_per_s": (len(ok) / window) if window else 0,
        "latency_p50_s": pct(0.5), "latency_p95_s": pct(0.95), "latency_max_s": lat[-1] if lat else None,
        "mean_prompt_n": statistics.mean([r["prompt_n"] for r in ok]) if ok else None,
        "mean_tokens_predicted": statistics.mean([r["tokens_predicted"] for r in ok]) if ok else None,
        "completed_per_url": per_url, "errors": [r.get("error") for r in results if not r["ok"]][:5],
        "started_utc": datetime.datetime.utcfromtimestamp(time.time() - (t_finish - t_begin)).isoformat() + "Z",
    }
    with (d / "requests.jsonl").open("w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    md = (f"| {a.label} | {a.concurrency} | {len(a.url)} | {len(ok)}/{len(results)} | {summary['Q_tok_per_s']:.2f} | "
          f"{summary['req_per_s']:.3f} | {summary['latency_p50_s'] or 0:.2f} | {summary['latency_p95_s'] or 0:.2f} | {window:.1f} | `{run_id}` |")
    (d / "summary.md").write_text("| label | C | servers | completed | Q tok/s | req/s | p50 s | p95 s | window s | run_id |\n|---|---|---|---|---|---|---|---|---|---|\n" + md + "\n")
    print(json.dumps({k: summary[k] for k in ("label", "concurrency", "requests_completed", "requests_issued", "Q_tok_per_s", "req_per_s", "latency_p50_s", "latency_p95_s", "window_s")}, indent=2))
    print(md)
    print(f"saved: {d}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
