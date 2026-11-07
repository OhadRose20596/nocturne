"""Session 5 figures from EDGE_TABLE.csv: per node memory bars, fit map, KV growth.
Usage: python -m nocturne.bench.plot_edge [--table experiments/session5_memory_edge/EDGE_TABLE.csv] [--ctx 4096]"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "session5_memory_edge"
STEPS = [("qwen2.5-7b-q4km", "7B Q4_K_M"), ("qwen2.5-14b-q4km", "14B Q4_K_M"), ("qwen2.5-32b-q4km", "32B Q4_K_M"),
         ("qwen2.5-32b-q5km", "32B Q5_K_M"), ("qwen2.5-32b-q6k", "32B Q6_K"), ("qwen2.5-32b-q8", "32B Q8_0")]
PARTS = ["local M5", "local Air", "PP default", "PP balanced", "TP", "TP 3:2"]
LABEL = {"local M5": "M5 alone", "local Air": "Air alone", "PP default": "Layer split (default)",
         "PP balanced": "Layer split (1:1)", "TP": "In-layer split (1:1)", "TP 3:2": "In-layer split (3:2)"}
KIND = [("model", "Weights", "#2a78d6"), ("kv", "KV cache", "#eb6834"), ("compute", "Compute buffers", "#1baf7a")]
GOOD, BAD, NA = "#0ca30c", "#d03b3b", "#e4e3df"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
LINE = {"m5": 18186 / 1024, "air": 12124 / 1024}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6})


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in (None, "") else None


def fig_bars(rows, ctx: int, out: Path):
    steps = [(m, l) for m, l in STEPS if any(r["model"] == m and int(r["ctx"]) == ctx for r in rows)]
    parts = ["PP default", "PP balanced", "TP"]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps) + 1, 4.2), sharey=True, squeeze=False)
    for ax, (m, lab) in zip(axes[0], steps):
        xs, ticks = [], []
        for i, p in enumerate(parts):
            r = next((r for r in rows if r["model"] == m and int(r["ctx"]) == ctx and r["partition"] == p), None)
            for j, node in enumerate(("m5", "air")):
                x = i * 2.6 + j
                xs.append(x); ticks.append(("M5" if node == "m5" else "Air"))
                if r is None or num(r[f"{node}_total_gib"]) is None:
                    ax.text(x, 0.4, "not run", rotation=90, ha="center", va="bottom", fontsize=8, color=INK2)
                    continue
                bottom = 0.0
                for k, _, c in KIND:
                    v = num(r[f"{node}_{k}_gib"]) or 0.0
                    ax.bar(x, v, bottom=bottom, width=0.8, color=c, edgecolor=SURF, linewidth=1)
                    bottom += v
                ax.plot([x - 0.45, x + 0.45], [LINE[node]] * 2, color=INK, linewidth=1.5, linestyle="--")
                mark = "fits" if r["fits"] == "yes" else (r["failure_class"] or "fail").split(";")[-1].replace("_", " ")
                ax.text(x, bottom + 0.3, mark, rotation=90, ha="center", va="bottom", fontsize=7, color=INK)
            ax.text(i * 2.6 + 0.5, -0.1, LABEL[p].replace(" (", "\n("), ha="center", va="top", fontsize=8, color=INK,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(xs, ticks, fontsize=8)
        ax.set_title(lab, fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("GiB on the node's GPU")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in KIND] + [plt.Line2D([], [], color=INK, linestyle="--")]
    fig.legend(handles, [l for _, l, _ in KIND] + ["Metal working set line"], loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.suptitle(f"Memory per node at context {ctx}", y=1.0, color=INK)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fitmap(rows, out: Path):
    steps = [(m, l) for m, l in STEPS if any(r["model"] == m for r in rows)]
    ctxs = sorted({int(r["ctx"]) for r in rows})
    parts = [p for p in PARTS if any(r["partition"] == p for r in rows)]
    fig, axes = plt.subplots(1, len(parts), figsize=(2.6 * len(parts) + 0.6, 0.55 * len(ctxs) + 1.8), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], parts):
        for xi, (m, _) in enumerate(steps):
            for yi, c in enumerate(ctxs):
                r = next((r for r in rows if r["model"] == m and int(r["ctx"]) == c and r["partition"] == p), None)
                color, txt = (NA, "") if r is None else ((GOOD, "fits") if r["fits"] == "yes" else (BAD, "fails"))
                ax.add_patch(plt.Rectangle((xi + 0.04, yi + 0.04), 0.92, 0.92, color=color))
                if txt:
                    ax.text(xi + 0.5, yi + 0.5, txt, ha="center", va="center", fontsize=7, color="white")
        ax.set_xlim(0, len(steps)); ax.set_ylim(0, len(ctxs))
        ax.set_xticks([i + 0.5 for i in range(len(steps))], [l.replace(" ", "\n") for _, l in steps], fontsize=7)
        ax.set_yticks([i + 0.5 for i in range(len(ctxs))], [str(c) for c in ctxs], fontsize=8)
        ax.set_title(LABEL[p], fontsize=9, color=INK)
        ax.grid(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    axes[0][0].set_ylabel("context (tokens)")
    fig.suptitle("Where each partition fits (grey: not run)", y=1.02, color=INK)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_kv(rows, out: Path, model="qwen2.5-32b-q4km"):
    parts = [("PP default", "#2a78d6", "o"), ("PP balanced", "#eb6834", "s"), ("TP", "#1baf7a", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, node in zip(axes, ("m5", "air")):
        for p, c, mk in parts:
            pts = sorted((int(r["ctx"]), num(r[f"{node}_kv_gib"])) for r in rows
                         if r["model"] == model and r["partition"] == p and num(r[f"{node}_kv_gib"]) is not None)
            if not pts:
                continue
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=c, marker=mk, markersize=6, linewidth=2, label=LABEL[p])
            ax.annotate(LABEL[p], pts[-1], textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=7, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_title("M5 (requester)" if node == "m5" else "Air (worker)", fontsize=10, color=INK)
        ax.set_xlabel("context (tokens)")
    axes[0].set_ylabel("KV cache on the node, GiB")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("KV cache growth with context, 32B Q4_K_M", y=1.02, color=INK)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, default=OUT / "EDGE_TABLE.csv")
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args(argv)
    rows = load(a.table)
    fig_dir = OUT / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_bars(rows, a.ctx, fig_dir / "a_memory_per_node.png")
    fig_fitmap(rows, fig_dir / "b_fit_map.png")
    fig_kv(rows, fig_dir / "c_kv_growth.png")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
