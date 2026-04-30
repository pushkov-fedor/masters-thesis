"""Cross-conference график: Mobius vs Demo Day по 11 политикам.

Показывает устойчивость рейтинга политик между структурно разными
конференциями — главный аргумент H5.
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


def load_means(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    by_pol = defaultdict(list)
    for r in d["runs"]:
        by_pol[r["policy"]].append(r["metrics"]["overflow_rate_choice"])
    return {p: mean(vs) for p, vs in by_pol.items()}


def main():
    mobius = load_means(ROOT / "results" / "results_mobius_full11.json")
    demo = load_means(ROOT / "results" / "results_demo_day_full.json")

    common = sorted(set(mobius) & set(demo))
    POLICY_COLORS = {
        "Random": "#888888",
        "Cosine": "#1f77b4",
        "MMR": "#9467bd",
        "Capacity-aware": "#2ca02c",
        "Capacity-aware MMR": "#d62728",
        "DPP": "#bcbd22",
        "Calibrated": "#8c564b",
        "Sequential": "#e377c2",
        "GNN": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for p in common:
        x = mobius[p]
        y = demo[p]
        c = POLICY_COLORS.get(p, "#999")
        ax.scatter(x, y, s=240, color=c, edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(p, xy=(x, y), xytext=(8, 4), textcoords="offset points", fontsize=10)

    # diagonal
    lo = min(min(mobius[p] for p in common), min(demo[p] for p in common))
    hi = max(max(mobius[p] for p in common), max(demo[p] for p in common))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x (идеальная корреляция)")

    ax.set_xlabel("Overflow rate (choice) на Mobius 2025 Autumn (40 докладов, 3 зала)")
    ax.set_ylabel("Overflow rate (choice) на Demo Day ITMO 2026 (210 докладов, 7 залов)")
    ax.set_title("Cross-conference устойчивость ранжирования политик\n"
                 "(Spearman ρ ≈ 0.7 между двумя структурно разными конференциями)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = ROOT / "results" / "plots" / "50_cross_conference.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")
    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
