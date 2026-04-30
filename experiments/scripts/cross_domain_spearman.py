"""Spearman ρ между ranking политик в Mobius и MovieLens (cross-domain validation).

Главный вопрос: устойчиво ли ранжирование политик по разным доменам?
Если ρ > 0.5 — выводы переносимы; если < 0.5 — domain-specific.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def aggregate(results_path: Path):
    """Возвращает {policy: {metric: mean}} аггрегированные по сидам."""
    with open(results_path) as f:
        data = json.load(f)
    by_policy = {}
    for r in data["runs"]:
        d = by_policy.setdefault(r["policy"], {})
        for k, v in r["metrics"].items():
            d.setdefault(k, []).append(v)
    out = {}
    for p, dct in by_policy.items():
        out[p] = {k: float(mean(vs)) for k, vs in dct.items()}
    return out


def spearman_for(metric, mobius, movielens):
    """Берём общие политики, считаем Spearman ρ."""
    common = [p for p in mobius if p in movielens]
    common.sort()
    m_vals = [mobius[p][metric] for p in common]
    ml_vals = [movielens[p][metric] for p in common]
    rho, p = stats.spearmanr(m_vals, ml_vals)
    return rho, p, common, m_vals, ml_vals


def main():
    mobius = aggregate(ROOT / "results" / "results_1200_5seeds.json")
    ml = aggregate(ROOT / "results" / "results_movielens.json")

    metrics_to_compare = [
        "mean_overload_excess",
        "hall_utilization_variance",
        "hall_load_gini",
        "overflow_rate_choice",
        "mean_user_utility",
    ]

    summary = {"comparisons": {}}
    print(f"Mobius policies: {sorted(mobius.keys())}")
    print(f"MovieLens policies: {sorted(ml.keys())}\n")
    for m_name in metrics_to_compare:
        rho, p, common, mv, mlv = spearman_for(m_name, mobius, ml)
        summary["comparisons"][m_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "policies": common,
            "mobius_values": mv,
            "movielens_values": mlv,
            "consistent": bool(rho > 0.3),
        }
        print(f"  {m_name:<28} ρ={rho:+.3f}, p={p:.3f}, n={len(common)}")
        print(f"    {'pol':<20} {'Mobius':<10} {'MovieLens':<10}")
        for p_name, mvv, mlvv in zip(common, mv, mlv):
            print(f"    {p_name:<20} {mvv:<10.3f} {mlvv:<10.3f}")
        print()

    out_path = ROOT / "results" / "cross_domain_spearman.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"WROTE: {out_path}")

    # Plot scatter for mean_overload_excess
    plot_path = ROOT / "results" / "plots" / "80_movielens_cross_domain.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric in zip(axes, ["mean_overload_excess", "hall_utilization_variance"]):
        comp = summary["comparisons"][metric]
        mv = comp["mobius_values"]
        mlv = comp["movielens_values"]
        ax.scatter(mv, mlv, s=60)
        for p_name, x, y in zip(comp["policies"], mv, mlv):
            ax.annotate(p_name, (x, y), fontsize=8, xytext=(5, 5),
                       textcoords="offset points")
        rho = comp["spearman_rho"]
        p_v = comp["p_value"]
        ax.set_xlabel(f"Mobius — {metric}")
        ax.set_ylabel(f"MovieLens — {metric}")
        ax.set_title(f"{metric}: ρ={rho:+.2f} (p={p_v:.3f})")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-domain ranking robustness: Mobius vs MovieLens 1M")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"WROTE: {plot_path}")


if __name__ == "__main__":
    main()
