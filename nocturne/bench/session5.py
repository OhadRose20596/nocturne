"""Session 5 orchestrator: every remaining memory edge cell in priority order; resumable, Air gate before each run.
Usage: python -m nocturne.bench.session5 [--dry-run] [--groups G0,G2a,...] [--no-commit]"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "nocturne" / "bench" / "results"
OUT = ROOT / "experiments" / "session5_memory_edge"
PROGRESS = OUT / "PROGRESS.md"
PEERS = Path("~/.nocturne/peers.yaml").expanduser()
PY = sys.executable
SIZE = ["qwen2.5-7b-q4km", "qwen2.5-14b-q4km", "qwen2.5-32b-q4km", "qwen2.5-32b-q5km", "qwen2.5-32b-q6k", "qwen2.5-32b-q8"]
NO_WORKER_COPY = {"qwen2.5-32b-q8"}          # deleted on the Air after the hash check (disk); workers never read it
LABEL = {("A1", ""): "PP default", ("A1", "prop"): "PP 3:2", ("A1", "bal"): "PP balanced", ("B1", ""): "TP",
         ("B1", "prop"): "TP 3:2", ("BASELINE", ""): "local", ("B1t", ""): "TP torch"}
# Device order differs by mode on 1e7bcf3: layer mode lists RPC devices first (-ts Air,M5), tensor mode puts the
# requester first (-ts M5,Air; checked against the worker's set_tensor bytes, 2026-11-07). Both give M5 3, Air 2.
TS = {("A1", "bal"): "1,1", ("A1", "prop"): "2,3", ("B1", "prop"): "3,2"}


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Step:
    def __init__(self, group, cell, model, ctx, var="", pset=None, fixed=None, host="m5", reps=1, subset=8):
        self.group, self.cell, self.model, self.ctx, self.var = group, cell, model, ctx, var
        self.pset, self.fixed, self.host, self.reps, self.subset = pset, fixed, host, reps, subset
        self.status, self.ids, self.t0, self.t1 = "pending", [], "", ""

    @property
    def tag(self) -> str:
        return f"c{self.ctx}{self.var}" + {"eval_long": "long", "eval_kv4k": "kv4k"}.get(self.pset or "", "")

    @property
    def label(self) -> str:
        base = LABEL.get((self.cell, self.var), self.cell)
        if self.cell == "BASELINE":
            base = f"{self.host.upper() if self.host == 'm5' else 'Air'} alone"
        p = f", {self.pset}" if self.pset else ""
        return f"{self.model} ctx {self.ctx} {base}{p}"

    def prefix(self) -> str:
        net = "local" if self.cell == "BASELINE" else "wifi"
        return f"{self.cell}_{net}_{self.model}_{self.host}_{self.tag}_r"

    def records(self) -> list[dict]:
        out = []
        for f in sorted(glob.glob(str(RES / (self.prefix() + "*.json")))):
            if not re.search(r"_r\d+\.json$", f):
                continue
            r = json.loads(Path(f).read_text())
            if any("failed (rc 255" in n or "llama-server not found in" in n for n in r["notes"]):
                continue
            if r["id"] == "B1_wifi_qwen2.5-32b-q6k_m5_c4096prop_r1":
                continue
            out.append(r)
        return out

    def argv(self) -> list[str]:
        a = [PY, "-m", "nocturne.bench.run", "--cell", self.cell, "--model", self.model, "--peers", str(PEERS),
             "--ctx", str(self.ctx), "--runs", "1", "--no-results-md", "--allow-unmeasured-memory", "--tag", self.tag,
             "--load-timeout", "2400", "--request-timeout", "3600"]
        a += ["--host", self.host] if self.cell == "BASELINE" else ["--network", "wifi"]
        if self.subset:
            a += ["--subset", str(self.subset)]
        if self.pset:
            a += ["--prompt-set", self.pset]
        if self.fixed:
            a += ["--fixed-tokens", str(self.fixed)]
        if (self.cell, self.var) in TS:
            a.append(f"--override-flag=-ts={TS[(self.cell, self.var)]}")
        return a


class Orchestrator:
    def __init__(self, a):
        self.a = a
        self.steps: list[Step] = []
        self.cfg = config.load_peers(PEERS)
        self.air = self.cfg["peers"]["air"]["ssh"]
        self.preflighted: set = set()
        self.awake = []
        self.log = []

    # ---- environment ---------------------------------------------------
    def keep_awake(self):
        if self.a.dry_run:
            return
        self.awake.append(subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())]))
        # -s needs the Air on power; -d -i -m hold it otherwise. Ends after 48 h or when this process stops it.
        self.awake.append(subprocess.Popen(["ssh", "-o", "BatchMode=yes", self.air, "caffeinate -dimsu -t 172800"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    def release(self):
        for p in self.awake:
            p.terminate()
        if not self.a.dry_run:
            subprocess.run(["ssh", "-o", "BatchMode=yes", self.air, "pkill -f 'caffeinate -dimsu -t 172800'"],
                           capture_output=True, timeout=30)

    def air_up(self) -> bool:
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.air, "hostname -s"],
                               capture_output=True, text=True, timeout=30)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def gate(self):
        """Wait up to 30 min for the Air; never spend a run on an unreachable worker."""
        if self.a.dry_run:
            return
        t0 = time.time()
        while not self.air_up():
            if time.time() - t0 > 1800:
                self.say("BLOCKED: the Air has not answered ssh for 30 min. Wake it, then rerun this command; finished steps are skipped.")
                self.write_progress()
                self.release()
                sys.exit(2)
            self.say("Air not reachable over ssh; retrying in 60 s")
            time.sleep(60)

    def preflight(self, model: str):
        if model in self.preflighted or self.a.dry_run:
            return
        argv = [PY, "-m", "nocturne.bench.preflight", "--model", model, "--min-free-gb", "25"]
        if model in NO_WORKER_COPY:
            argv.append("--no-worker-model")
        self.say(f"preflight {model}")
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        for line in (r.stdout + r.stderr).strip().splitlines():
            self.say("  " + line)
        if r.returncode != 0:
            self.say(f"STOP: preflight failed for {model}")
            self.write_progress()
            self.release()
            sys.exit(3)
        self.preflighted.add(model)

    # ---- bookkeeping ---------------------------------------------------
    def say(self, msg: str):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.log.append(line)

    def write_progress(self):
        done = sum(s.status not in ("pending", "running") for s in self.steps)
        lines = ["# Session 5 progress", "", f"Updated {now()}. Steps known so far: {len(self.steps)}, finished: {done}. "
                 "Later groups add steps from earlier results (the largest model that fits, the edge per partition).", "",
                 "| group | step | status | records | start | end |", "|---|---|---|---|---|---|"]
        for s in self.steps:
            lines.append(f"| {s.group} | {s.label} | {s.status} | {' '.join(s.ids)} | {s.t0} | {s.t1} |")
        lines += ["", "## Log (last 40 lines)", "", "```"] + self.log[-40:] + ["```", ""]
        PROGRESS.write_text("\n".join(lines))

    def derive(self):
        if self.a.dry_run:
            return
        subprocess.run([PY, "-m", "nocturne.bench.derive_edge"], cwd=ROOT, capture_output=True)
        subprocess.run([PY, "-m", "nocturne.bench.plot_edge", "--ctx", "4096"], cwd=ROOT, capture_output=True)

    def rows(self) -> list[dict]:
        p = OUT / "EDGE_TABLE.csv"
        if not p.exists():
            return []
        import csv
        with p.open() as f:
            return list(csv.DictReader(f))

    def fits(self, model, ctx, part, pset="eval_set") -> bool | None:
        for r in self.rows():
            if r["model"] == model and int(r["ctx"]) == ctx and r["partition"] == part and r["prompts"] == pset:
                return r["fits"] == "yes"
        return None

    def commit(self, group: str, title: str):
        if self.a.dry_run or self.a.no_commit:
            return
        paths = ["experiments/session5_memory_edge"] + [str(Path(f).relative_to(ROOT)) for s in self.steps
                                                         if s.group == group for f in glob.glob(str(RES / (s.prefix() + "*.json")))]
        subprocess.run(["git", "add", "--"] + paths, cwd=ROOT, capture_output=True)
        msg = f"Session 5 {group}: {title}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        r = subprocess.run(["git", "commit", "-q", "-m", msg, "--"] + paths, cwd=ROOT, capture_output=True, text=True)
        self.say(f"commit {group}: {'ok' if r.returncode == 0 else (r.stderr or r.stdout).strip()[:160]}")

    # ---- running -------------------------------------------------------
    def run(self, s: Step):
        have = s.records()
        s.ids = [r["id"] for r in have]
        if len(have) >= s.reps:
            s.status = "done earlier: " + ", ".join("ok" if r["success"] else str(r["failure_class"]) for r in have)
            return
        self.preflight(s.model)
        while len(s.records()) < s.reps:
            self.gate()
            s.status, s.t0 = "running", now()
            self.write_progress()
            self.say(f"run {s.label} ({s.group}): " + " ".join(s.argv()[2:]))
            if self.a.dry_run:
                s.status = "dry run"
                return
            r = subprocess.run(s.argv(), cwd=ROOT, capture_output=True, text=True)
            res = [l for l in r.stdout.splitlines() if ": OK" in l or ": FAIL" in l]
            self.say("  " + (res[-1].split(" -> ")[0] if res else f"no record line, rc {r.returncode}: {r.stderr.strip()[-200:]}"))
            got = s.records()
            if len(got) == len(have):          # nothing valid written (unreachable worker or harness error): do not loop
                if not self.air_up():
                    continue
                s.status = "no valid record"
                break
            have = got
        s.ids = [r["id"] for r in s.records()]
        last = s.records()[-1] if s.records() else None
        s.status = ("ok" if last and last["success"] else f"fail ({last['failure_class']})" if last else s.status)
        s.t1 = now()
        self.derive()
        self.write_progress()

    def group(self, name: str, title: str, steps: list[Step]):
        if self.a.groups and name not in self.a.groups:
            return
        self.say(f"== {name}: {title} ({len(steps)} steps)")
        self.steps += steps
        self.write_progress()
        for s in steps:
            self.run(s)
        self.commit(name, title)

    def main(self):
        self.keep_awake()
        try:
            self.plan()
        finally:
            self.release()
        self.say("all groups finished")
        self.write_progress()

    def plan(self):
        q = ["qwen2.5-32b-q5km", "qwen2.5-32b-q6k", "qwen2.5-32b-q8"]
        self.group("G0", "layer mode -ts order check (32B Q4_K_M, ctx 32768, PP 3:2)",
                   [Step("G0", "A1", "qwen2.5-32b-q4km", 32768, "prop")])
        self.group("G2a", "32B Q5_K_M, Q6_K, Q8_0 at ctx 4096, all partitions",
                   [Step("G2a", c, m, 4096, v) for m in q for c, v in (("A1", ""), ("A1", "prop"), ("A1", "bal"), ("B1", ""), ("B1", "prop"))])
        edge = [m for m in q if not any(self.fits(m, 4096, p) for p in ("PP default", "PP 3:2", "PP balanced", "TP", "TP 3:2"))]
        self.group("G2edge", "models that fit nowhere at 4096, checked at ctx 512",
                   [Step("G2edge", c, m, 512, v) for m in edge for c, v in (("A1", ""), ("A1", "prop"), ("B1", "prop"))])
        pp = [m for m in q if self.fits(m, 4096, "PP default") or self.fits(m, 4096, "PP 3:2")]
        tp = [m for m in q if self.fits(m, 4096, "TP") or self.fits(m, 4096, "TP 3:2")]
        sweep = sorted({x for x in (pp[-1:] + tp[-1:])}, key=SIZE.index)
        self.say(f"largest model fitting at 4096: PP {pp[-1:] or 'none'}, TP {tp[-1:] or 'none'}; context sweep on {sweep}")
        if not self.a.groups or "G2b" in self.a.groups:
            self.say("== G2b: context sweep of the largest fitting models")
            for m in sweep:
                for ctx in (512, 16384, 32768):
                    batch = []
                    for c, v in (("A1", ""), ("A1", "prop"), ("A1", "bal"), ("B1", "prop")):
                        part = LABEL[(c, v)]
                        smaller = [x for x in (4096, 16384) if x < ctx]
                        if ctx > 4096 and any(self.fits(m, x, part) is False for x in smaller):
                            self.say(f"skip {m} ctx {ctx} {part}: already fails at a smaller context")
                            continue
                        batch.append(Step("G2b", c, m, ctx, v))
                    self.steps += batch
                    for s in batch:
                        self.run(s)
            self.commit("G2b", "context sweep of the largest fitting 32B quantizations")
        reps = [Step("G1r", c, "qwen2.5-32b-q4km", ctx, v, reps=3) for ctx in (512, 4096, 16384, 32768) for c, v in (("A1", ""), ("A1", "bal"))]
        reps += [Step("G1r", "B1", "qwen2.5-32b-q4km", 4096, "", reps=2), Step("G1r", "B1", "qwen2.5-32b-q4km", 32768, "prop", reps=2)]
        self.group("G1r", "repeats of the 32B Q4_K_M cells", reps)
        mid = []
        for m in ("qwen2.5-14b-q4km", "qwen2.5-7b-q4km"):
            for ctx in (4096, 32768):
                mid += [Step("G4", "BASELINE", m, ctx, host="m5"), Step("G4", "BASELINE", m, ctx, host="air"),
                        Step("G4", "A1", m, ctx), Step("G4", "B1", m, ctx)]
        self.group("G4", "14B and 7B Q4_K_M, every partition, ctx 4096 and 32768", mid)
        g3 = []
        for c, v in (("A1", ""), ("A1", "prop"), ("B1", "prop")):
            part = LABEL[(c, v)]
            fit = [r for r in self.rows() if r["partition"] == part and r["fits"] == "yes" and r["prompts"] == "eval_set"
                   and r["model"] in SIZE]
            if not fit:
                continue
            best = max(fit, key=lambda r: (SIZE.index(r["model"]), int(r["ctx"])))
            g3 += [Step("G3", c, best["model"], int(best["ctx"]), v, pset="eval_long", fixed=256, subset=3),
                   Step("G3", c, best["model"], int(best["ctx"]), v, pset="eval_kv4k", fixed=64, subset=None)]
        self.group("G3", "real long prompts at the edge of each partition", g3)
        self.group("G5", "torch_tp engine, bf16, ctx 4096",
                   [Step("G5", "B1t", "qwen2.5-1.5b-bf16", 4096), Step("G5", "B1t", "qwen2.5-7b-bf16", 4096)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--groups", type=lambda s: s.split(","), default=None)
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    Orchestrator(a).main()


if __name__ == "__main__":
    main()
