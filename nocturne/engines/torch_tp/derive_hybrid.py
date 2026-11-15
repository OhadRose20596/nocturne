"""Session 6 tables and figures from the run records. No torch: runs in the repo .venv.
Usage: .venv/bin/python -m nocturne.engines.torch_tp.derive_hybrid
Writes experiments/session6_hybrid/HYBRID_TABLE.md and .csv, and fig/: one placement diagram per placement file,
planned against measured bytes per rank, and frames per decode step (formula against measured)."""
import csv
import json
from pathlib import Path

import yaml

from .plan import Dims, frames_per_step, placement_from_dict, plan

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "session6_hybrid"
RUNS, PLACEMENTS, FIG = OUT / "runs", OUT / "placements", OUT / "fig"
HOST = {"m5": "#2a78d6", "air": "#eb6834"}
HOST_NAME = {"m5": "M5", "air": "Air"}
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3df", "#fcfcfb"
MEASURE = [("planned", "planned (weights + KV + workspace)", "#2a78d6", "o"),
           ("current", "MPS tensors allocated", "#eb6834", "s"),
           ("footprint", "process footprint", "#1baf7a", "D")]


def dims_of(model_key: str) -> Dims:
    peers = yaml.safe_load(Path("~/.nocturne/peers.yaml").expanduser().read_text())
    d = (peers["models"].get(model_key) or {}).get("torch_model", model_key)
    c = json.loads((Path(d).expanduser() / "config.json").read_text())
    return Dims(c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["num_attention_heads"],
                c["num_key_value_heads"], c["vocab_size"], c.get("tie_word_embeddings", False))


def records() -> list[dict]:
    out = []
    for f in sorted(RUNS.glob("*/record.json")):
        r = json.loads(f.read_text())
        note = f.parent / "ABORTED.txt"
        if note.exists():            # stopped by hand; the note says why and nothing from it is used
            r["outcome"], r["error"] = "aborted", note.read_text().strip()
        if r.get("outcome"):
            out.append(r)
    return out


def best_corr(r: dict) -> dict | None:
    c = r.get("correctness") or {}
    return c[sorted(c)[0]] if c else None


def table(recs: list[dict]) -> None:
    cols = ["run_id", "placement", "model", "reduce", "outcome", "ranks", "hosts", "reference", "identical", "accuracy",
            "ref_accuracy", "criterion", "logit_maxdiff", "long_first_div", "frames_predicted", "frames_measured", "formula",
            "bytes_per_step", "max_rank_planned_MB", "max_rank_footprint_MB", "ms_per_step_background", "error"]
    rows = []
    for r in recs:
        c = best_corr(r)
        fr = r.get("frames") or {}
        ranks = r.get("ranks") or []
        ms = [p["ms_per_step"] for p in r.get("prompts") or [] if p.get("ms_per_step")]
        rows.append({
            "run_id": r["run_id"], "placement": Path(r["placement"]["file"]).stem, "model": (r.get("model") or {}).get("key", ""),
            "reduce": r.get("reduce_dtype", ""), "outcome": r["outcome"], "ranks": len(r.get("planned") or []),
            "hosts": "+".join(sorted({p["host"] for p in r.get("planned") or []})),
            "reference": c["reference"] if c else "", "identical": f"{c['identical']}/{c['of']}" if c else "",
            "accuracy": c["accuracy"] if c else "", "ref_accuracy": c["ref_accuracy"] if c else "",
            "criterion": ("met" if c["criterion_met"] else "not met") if c else "",
            "logit_maxdiff": f"{c['logit_maxdiff']:.4g}" if c and c["logit_maxdiff"] is not None else "",
            "long_first_div": ",".join(str(v) for v in c["long_first_div"].values()) if c else "",
            "frames_predicted": (fr.get("predicted") or {}).get("total", ""),
            "frames_measured": fr.get("measured_per_step") if fr.get("measured_per_step") is not None else "",
            "formula": {True: "holds", False: "FAILS"}.get(fr.get("formula_holds"), ""),
            "bytes_per_step": f"{fr['bytes_per_step']:.0f}" if fr.get("bytes_per_step") else "",
            "max_rank_planned_MB": f"{max(p['planned']['total'] for p in r['planned']) / 1e6:.0f}" if r.get("planned") else "",
            "max_rank_footprint_MB": f"{max(k.get('phys_footprint_bytes') or 0 for k in ranks) / 1e6:.0f}" if ranks else "",
            "ms_per_step_background": f"{sorted(ms)[len(ms) // 2]:.0f}" if ms else "",
            "error": (r.get("error") or "")[:160]})
    with (OUT / "HYBRID_TABLE.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["# Session 6 hybrid table", "",
          "Built by `python -m nocturne.engines.torch_tp.derive_hybrid` from `runs/*/record.json`. One row per run.",
          "Correctness is against the first R0 reference listed in the record. ms per step is background only (T1).", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(x[c]) for c in cols) + " |" for x in rows]
    (OUT / "HYBRID_TABLE.md").write_text("\n".join(md) + "\n")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def placement_diagram(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    y = yaml.safe_load(path.read_text())
    p = placement_from_dict(y)
    d = dims_of(p.model)
    specs = plan(d, p, check_budget=False)
    stages = sorted({s.stage for s in specs})
    top = max(max(s.planned["total"], s.budget or 0) for s in specs)
    unit = 1.0 / top
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(stages) + 1.5), 4.2), facecolor=SURFACE)
    _style(ax)
    for si in stages:
        ranks = [s for s in specs if s.stage == si]
        y0 = 0.0
        for s in ranks:
            h = max(0.035, s.planned["total"] * unit)
            ax.add_patch(Rectangle((si + 0.1, y0), 0.8, h, facecolor=HOST[s.host], edgecolor=SURFACE, linewidth=2))
            label = f"{s.planned['total'] / 1e9:.2f} GB" if s.planned["total"] >= 1e9 else f"{s.planned['total'] / 1e6:.0f} MB"
            ax.text(si + 0.5, y0 + h / 2, label, ha="center", va="center", fontsize=7, color="white" if h > 0.06 else INK)
            if s.budget:
                ax.plot([si + 0.1, si + 0.9], [y0 + s.budget * unit] * 2, color=INK, linestyle=(0, (3, 2)), linewidth=1)
            y0 += max(h, s.budget * unit if s.budget else 0) + 0.02
        st = p.stages[si]
        what = f"layers {st.layers[0]}-{st.layers[1] - 1}" if st.layers else "head only"
        extra = (" + embedding" if specs[[s.stage for s in specs].index(si)].embed_rows else "") + \
                (" + head" if st.layers and specs[[s.stage for s in specs].index(si)].head_rows else "")
        ax.text(si + 0.5, -0.06, f"{st.name}\n{what}{extra}\n{len(ranks)} rank{'s' if len(ranks) > 1 else ''}",
                ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xlim(0, len(stages))
    ax.set_ylim(-0.25, None)
    ax.autoscale(axis="y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    hs = sorted({s.host for s in specs})
    handles = [Rectangle((0, 0), 1, 1, color=HOST[h]) for h in hs]
    labels = [HOST_NAME.get(h, h) for h in hs]
    if any(s.budget for s in specs):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color=INK, linestyle=(0, (3, 2))))
        labels.append("budget")
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    f = frames_per_step(specs)["total"]
    ax.set_title(f"{y.get('model')}: {len(specs)} ranks in {len(stages)} stages, {f} frames per decode step\n"
                 f"box height = planned bytes per rank (weights, KV at {p.max_seq_len}, workspace)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"placement_{path.stem}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def representative(recs: list[dict]) -> list[dict]:
    """One completed run per cell: the two Mac variant when there is one, else loopback; run 1; no controls."""
    by_cell: dict[str, list[dict]] = {}
    for r in recs:
        stem = Path(r["placement"]["file"]).stem
        if r["outcome"] != "completed" or stem.startswith("R0_") or "bf16ctl" in r["run_id"] or "mpscap" in r["run_id"]:
            continue
        by_cell.setdefault(stem.split("_")[0], []).append(r)
    out = []
    for cell in sorted(by_cell):
        rs = sorted(by_cell[cell], key=lambda r: (0 if any(t in r["run_id"] for t in ("_2mac_", "_wifi_")) else 1, r["run_id"]))
        out.append(rs[0])
    return out


def bytes_figure(recs: list[dict]) -> None:
    import matplotlib.pyplot as plt
    done = representative(recs)
    if not done:
        return
    rows = []
    for r in done:
        cell = Path(r["placement"]["file"]).stem.replace("_2mac", ", two Macs").replace("_wifi", "-wifi, two Macs").replace("_loopback", ", one Mac")
        for k in r["ranks"]:
            rows.append((f"{cell}   {k['name']} ({HOST_NAME.get(k['host'], k['host'])}, {k['device']})", k))
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(rows) + 1.4), facecolor=SURFACE)
    _style(ax)
    for i, (label, k) in enumerate(rows):
        vals = {"planned": k["planned"]["total"], "current": k.get("current_allocated_bytes"),
                "footprint": k.get("phys_footprint_bytes")}
        for key, _, color, marker in MEASURE:
            if vals[key]:
                ax.scatter(vals[key] / 1e9, i, color=color, marker=marker, s=22, zorder=3, edgecolors=SURFACE, linewidths=0.8)
        if k.get("budget"):
            ax.plot([k["budget"] / 1e9] * 2, [i - 0.4, i + 0.4], color=INK, linewidth=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6, color=INK2)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("GB per rank (log scale)", fontsize=8, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker=m, color="none", markerfacecolor=c, markersize=6) for _, _, c, m in MEASURE]
    h.append(Line2D([0], [0], color=INK, linewidth=1.5))
    ax.legend(h, [l for _, l, _, _ in MEASURE] + ["budget"], loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Planned against measured memory per rank", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "bytes_planned_vs_measured.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def frames_figure(recs: list[dict]) -> None:
    import matplotlib.pyplot as plt
    done = [r for r in recs if r["outcome"] == "completed" and (r.get("frames") or {}).get("predicted", {}).get("total")]
    if not done:
        return
    fig, ax = plt.subplots(figsize=(7, 0.25 * len(done) + 1.4), facecolor=SURFACE)
    _style(ax)
    for i, r in enumerate(done):
        p, m = r["frames"]["predicted"]["total"], r["frames"]["measured_per_step"]
        ax.scatter(p, i, s=60, facecolors="none", edgecolors="#2a78d6", linewidths=1.5, zorder=3)
        ax.scatter(m, i, s=14, color="#eb6834", marker="s", zorder=4)
    ax.set_yticks(range(len(done)))
    ax.set_yticklabels([r["run_id"] for r in done], fontsize=6, color=INK2)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("frames per decode step, all processes (log scale)", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker="o", color="none", markeredgecolor="#2a78d6", markerfacecolor="none", markersize=7),
               Line2D([0], [0], marker="s", color="none", markerfacecolor="#eb6834", markersize=5)],
              ["formula (written before the runs)", "measured"], loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Frames per decode step: formula against measured", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "frames_formula_vs_measured.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    recs = records()
    table(recs)
    for f in sorted(PLACEMENTS.glob("*.yaml")):
        if not f.stem.startswith("R0_"):
            placement_diagram(f)
    bytes_figure(recs)
    frames_figure(recs)
    print(f"{len(recs)} records; wrote {OUT / 'HYBRID_TABLE.md'} and {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
