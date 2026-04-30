"""Trade-off plot: Bradley-Terry rating (judge view) vs overflow_rate_choice (system view).

Главный график для аргумента: subjective quality и system metrics расходятся.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "results" / "results_learned_full.json", encoding="utf-8") as f:
        param_data = json.load(f)
    with open(ROOT / "results" / "llm_judge_full.json", encoding="utf-8") as f:
        judge_data = json.load(f)

    # Aggregate overflow per policy
    overflow = defaultdict(list)
    for r in param_data["runs"]:
        overflow[r["policy"]].append(r["metrics"]["overflow_rate_choice"])
    overflow = {p: mean(vs) for p, vs in overflow.items()}

    bt = judge_data["bradley_terry"]

    # Combine
    common = set(overflow) & set(bt)
    POLICY_COLORS = {
        "Random": "#888888",
        "Cosine": "#1f77b4",
        "MMR": "#9467bd",
        "Capacity-aware": "#2ca02c",
        "Capacity-aware MMR": "#d62728",
        "Constrained-PPO": "#17becf",
        "LLM-ranker": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for p in sorted(common):
        x = overflow[p]
        y = bt[p]
        ax.scatter(x, y, s=240, color=POLICY_COLORS.get(p, "#999999"),
                   edgecolor="black", linewidth=1.2, label=p, zorder=3)
        ax.annotate(p, xy=(x, y), xytext=(8, 4), textcoords="offset points", fontsize=9.5)

    ax.set_xlabel("Overflow rate (system metric, ниже = лучше)")
    ax.set_ylabel("Bradley-Terry rating (LLM-judge subjective view, выше = лучше)")
    ax.set_title("Trade-off: системные метрики vs субъективное качество\n"
                 "(Capacity-aware: левее лучше — нет переполнений; LLM-ranker: выше лучше — нравится судье)")
    ax.grid(True, alpha=0.3)

    # Pareto frontier hint
    ax.text(0.02, 0.98, "Идеал — верхний левый угол:\nнет переполнений + любим судьёй",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="gray", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))

    fig.tight_layout()
    out = ROOT / "results" / "plots" / "31_judge_vs_overflow_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")
    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
