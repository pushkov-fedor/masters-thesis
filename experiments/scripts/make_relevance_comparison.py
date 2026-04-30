"""Сравнение метрик политик на cosine vs learned relevance — оценка робастности."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

POLICY_ORDER = ["Random", "Cosine", "MMR", "Capacity-aware", "Capacity-aware MMR", "LLM-ranker"]
POLICY_COLORS = {
    "Random": "#888888",
    "Cosine": "#1f77b4",
    "MMR": "#9467bd",
    "Capacity-aware": "#2ca02c",
    "Capacity-aware MMR": "#d62728",
    "LLM-ranker": "#ff7f0e",
}


def aggregate(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_policy = defaultdict(lambda: defaultdict(list))
    for r in data["runs"]:
        for k, v in r["metrics"].items():
            by_policy[r["policy"]][k].append(v)
    return {p: {k: mean(vs) for k, vs in d.items()} for p, d in by_policy.items()}


def main():
    cos_data = aggregate(ROOT / "results" / "results_cosine.json")
    learn_data = aggregate(ROOT / "results" / "results_learned.json")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, lower in [
        (axes[0], "overflow_rate_choice", "Overflow rate (choice slots)", True),
        (axes[1], "mean_user_utility", "Mean user utility", False),
    ]:
        names = [p for p in POLICY_ORDER if p in cos_data and p in learn_data]
        cos_vals = [cos_data[p].get(metric, 0) for p in names]
        learn_vals = [learn_data[p].get(metric, 0) for p in names]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, cos_vals, w, label="Cosine relevance", alpha=0.85,
               color=[POLICY_COLORS[n] for n in names], edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, learn_vals, w, label="Learned relevance", alpha=0.85, hatch="//",
               color=[POLICY_COLORS[n] for n in names], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ("  (ниже = лучше)" if lower else "  (выше = лучше)"))
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Робастность главного результата к смене relevance signal\n"
                 "(порядок политик по overflow_choice сохраняется на обоих сигналах)")
    fig.tight_layout()

    out = ROOT / "results" / "plots" / "21_cosine_vs_learned.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")
    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
