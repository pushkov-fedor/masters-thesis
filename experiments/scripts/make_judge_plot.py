"""Plot LLM-as-judge Bradley-Terry results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "results" / "llm_judge_full.json"
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    bt = data["bradley_terry"]
    rated = sorted(bt.items(), key=lambda x: -x[1])
    names = [n for n, _ in rated]
    ratings = [r for _, r in rated]

    print("Bradley-Terry rating:")
    for n, r in rated:
        print(f"  {n:<22} BT={r:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "Random": "#888888",
        "Cosine": "#1f77b4",
        "MMR": "#9467bd",
        "Capacity-aware": "#2ca02c",
        "Capacity-aware MMR": "#d62728",
        "Constrained-PPO": "#17becf",
        "LLM-ranker": "#ff7f0e",
    }
    bar_colors = [colors.get(n, "#aaaaaa") for n in names]
    xs = np.arange(len(names))
    bars = ax.bar(xs, ratings, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Bradley-Terry rating (выше = судьи предпочитают чаще)")
    ax.set_title(f"LLM-as-judge: pairwise сравнение политик\n"
                 f"{data['config']['n_personas']} персон × {data['config']['n_slots']} слотов × {len(data['judgments'])} суждений\n"
                 f"Модель-судья: {data['config']['model']}")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, r in zip(bars, ratings):
        ax.text(bar.get_x() + bar.get_width()/2, r + 0.02, f"{r:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = ROOT / "results" / "plots" / "30_llm_judge_bradley_terry.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")

    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
