"""Сравнение метрик параметрический vs LLM-агентный симулятор (слой 2 валидации)."""
import json
import sys
from pathlib import Path

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


def main():
    # Параметрический симулятор: results_learned.json или results_learned_full.json
    param_path = ROOT / "results" / "results_learned_full.json"
    if not param_path.exists():
        param_path = ROOT / "results" / "results_learned.json"
    with open(param_path, encoding="utf-8") as f:
        param_data = json.load(f)

    # Aggregate parametric metrics
    from collections import defaultdict
    from statistics import mean, stdev
    param_metrics = defaultdict(lambda: defaultdict(list))
    for r in param_data["runs"]:
        p = r["policy"]
        if p == "Constrained-PPO":
            continue
        for k, v in r["metrics"].items():
            param_metrics[p][k].append(v)

    # Agent sim
    agent_path = ROOT / "results" / "agent_validation_full50.json"
    if not agent_path.exists():
        print(f"Agent validation not found: {agent_path}")
        sys.exit(1)
    with open(agent_path, encoding="utf-8") as f:
        agent_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics_to_plot = [
        ("overflow_rate_choice", "Overflow rate (choice slots)", True),
        ("hall_utilization_variance", "Hall utilization variance", True),
        ("skip_rate", "Skip rate", None),
    ]

    for ax, (metric_key, ylabel, lower_better) in zip(axes, metrics_to_plot):
        param_means = []
        agent_means = []
        names = []
        for p in POLICY_ORDER:
            if p in param_metrics and metric_key in param_metrics[p]:
                pm = mean(param_metrics[p][metric_key])
            else:
                pm = None
            am = agent_data["results"].get(p, {}).get("metrics", {}).get(metric_key)
            if pm is not None and am is not None:
                names.append(p)
                param_means.append(pm)
                agent_means.append(am)

        x = np.arange(len(names))
        width = 0.35
        bars1 = ax.bar(x - width/2, param_means, width, label="Параметрический симулятор",
                       color=[POLICY_COLORS[n] for n in names], alpha=0.7, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, agent_means, width, label="LLM-агентный симулятор",
                       color=[POLICY_COLORS[n] for n in names], hatch="//", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(ylabel)

    fig.suptitle("Sim-to-sim gap: параметрический vs LLM-агентный симулятор\n"
                 f"(слой 1: 900 пользователей × 5 сидов; слой 2: {agent_data['config']['n_agents']} LLM-агентов с памятью)")
    fig.tight_layout()

    out = ROOT / "results" / "plots" / "40_sim2sim_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {out}")

    thesis_fig = ROOT.parent / "thesis" / "figures"
    if thesis_fig.exists():
        import shutil
        shutil.copy(out, thesis_fig / out.name)


if __name__ == "__main__":
    main()
