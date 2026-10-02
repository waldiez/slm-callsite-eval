"""Figures for the repository README, generated from the scored records.

    python3 harness/make_figures.py results/final_scored.jsonl --out figures

Needs matplotlib. Reuses the loaders and metrics of make_paper_tables.py, so
every value plotted is the value printed in the paper's tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_paper_tables import (  # noqa: E402
    CATS,
    DISPLAY,
    ORDER,
    Runs,
    load_expectations,
    safety_rates,
)

SITE_LABELS = {
    "intent": "Intent",
    "ha": "HA action",
    "actuator": "Actuator",
    "planner": "Planner",
    "dynamic": "Codegen",
}


def models_in_order(runs: Runs) -> list[str]:
    present = set(runs.models)
    ordered = [m for m in ORDER if m in present]
    return ordered + sorted(present - set(ordered))


def heatmap(runs: Runs, out: Path) -> None:
    models = models_in_order(runs)
    grid = [[100 * runs.acc(m, c)[0] for c in CATS] for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(models) + 1.4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(CATS)), [SITE_LABELS[c] for c in CATS])
    ax.set_yticks(range(len(models)), [DISPLAY.get(m, m) for m in models])
    ax.xaxis.tick_top()
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9,
                    color="black" if 25 < val < 85 else "white")
    hosted_rows = [i for i, m in enumerate(models) if m.startswith("anthropic:")]
    if hosted_rows:
        ax.axhline(min(hosted_rows) - 0.5, color="black", linewidth=1.5)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.04, pad=0.03)
    ax.set_title("Accuracy per call site (local models above the line, hosted below)",
                 fontsize=10, pad=28)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# Hand-placed labels for the crowded top-left corner (data coordinates);
# every other model is labelled next to its marker.
LABEL_AT = {
    "anthropic:claude-sonnet-5": (30, 104),
    "openai:gemma4:12b": (30, 97.5),
    "openai:gemma4:26b": (30, 91),
    "anthropic:claude-haiku-4-5": (8, 70),
}


def safety_scatter(runs: Runs, must_act: set[str], must_refuse: set[str], out: Path) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.add_patch(plt.Rectangle((-5, 90), 15, 18, color="tab:green", alpha=0.12, zorder=0))
    for m in models_in_order(runs):
        s = safety_rates(runs, m, must_act, must_refuse)
        if not s:
            continue
        x, y = 100 * s["over"], 100 * s["exec"]
        hosted = m.startswith("anthropic:")
        ax.scatter(x, y, s=60, marker="s" if hosted else "o",
                   color="tab:blue" if hosted else "tab:orange", zorder=3)
        name = DISPLAY.get(m, m)
        if m in LABEL_AT:
            ax.annotate(name, (x, y), xytext=LABEL_AT[m], fontsize=8, va="center",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
        else:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 108)
    ax.set_xlabel("Over-actuation on impossible requests (%), lower is better")
    ax.set_ylabel("Execution accuracy on actionable requests (%)")
    ax.set_title("Actuator site: acting vs refusing", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="tab:orange", label="Local"),
        Line2D([], [], marker="s", ls="", color="tab:blue", label="Hosted"),
        Patch(color="tab:green", alpha=0.25, label="Usable to drive hardware"),
    ], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="results.jsonl from harness/evalharness.py")
    ap.add_argument("--bench", default="benchmark", help="directory holding the benchmark JSONL files")
    ap.add_argument("--out", default="figures", help="where to write the PNG files")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = Runs(records)
    must_act, must_refuse = load_expectations(Path(args.bench))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    heatmap(runs, out / "per_site_accuracy.png")
    safety_scatter(runs, must_act, must_refuse, out / "actuation_safety.png")
    print(f"wrote {out}/per_site_accuracy.png, {out}/actuation_safety.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
