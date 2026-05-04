"""Расширенный Spearman ρ между ranking политик в Mobius и пяти cross-domain датасетах:
MovieLens, Meetup RSVP, ITC-2019 (mary-fal18, bet-spr18), ITC-2007 (T1, T2).

Сводная таблица + матрица Spearman.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def aggregate(results_path: Path):
    if not results_path.exists():
        return None
    with open(results_path) as f:
        data = json.load(f)
    by_policy = {}
    for r in data["runs"]:
        d = by_policy.setdefault(r["policy"], {})
        for k, v in r["metrics"].items():
            d.setdefault(k, []).append(v)
    return {p: {k: float(mean(vs)) for k, vs in dct.items()} for p, dct in by_policy.items()}


def spearman(metric, ref, other):
    common = sorted(p for p in ref if p in other)
    rv = [ref[p][metric] for p in common]
    ov = [other[p][metric] for p in common]
    if len(common) < 3:
        return float("nan"), float("nan"), common, rv, ov
    rho, pv = stats.spearmanr(rv, ov)
    return float(rho), float(pv), common, rv, ov


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="legacy", choices=["legacy", "calibrated"])
    args, _ = ap.parse_known_args()
    suffix = "_calibrated" if args.mode == "calibrated" else ""
    if args.mode == "calibrated":
        datasets = {
            "Mobius (calibrated)": ROOT / "results" / "results_mobius_calibrated.json",
            "Meetup RSVP": ROOT / "results" / "results_meetup_calibrated.json",
            "ITC-2019 mary-fal18": ROOT / "results" / "results_itc2019_mary_calibrated.json",
            "ITC-2019 bet-spr18": ROOT / "results" / "results_itc2019_bet_calibrated.json",
            "ITC-2007 T1 set1": ROOT / "results" / "results_itc2007_t1_calibrated.json",
            "ITC-2007 T2 c1": ROOT / "results" / "results_itc2007_t2_calibrated.json",
            "UMass CICS Fall 2024": ROOT / "results" / "results_umass_cics_calibrated.json",
        }
    else:
        datasets = {
            "MovieLens 1M": ROOT / "results" / "results_movielens.json",
            "Meetup RSVP": ROOT / "results" / "results_meetup_rsvp.json",
            "ITC-2019 mary-fal18": ROOT / "results" / "results_itc2019_mary.json",
            "ITC-2019 bet-spr18": ROOT / "results" / "results_itc2019_bet.json",
            "ITC-2007 T1 set1": ROOT / "results" / "results_itc2007_t1.json",
            "ITC-2007 T2 c1": ROOT / "results" / "results_itc2007_t2.json",
        }
    metrics = [
        "mean_overload_excess",
        "hall_utilization_variance",
        "hall_load_gini",
        "overflow_rate_choice",
        "mean_user_utility",
    ]

    if args.mode == "calibrated":
        mobius = aggregate(ROOT / "results" / "results_mobius_calibrated.json")
    else:
        mobius = aggregate(ROOT / "results" / "results_1200_5seeds.json")
    if mobius is None:
        raise SystemExit("Need Mobius baseline results file")

    table = {}
    for name, path in datasets.items():
        agg = aggregate(path)
        if agg is None:
            print(f"SKIP {name} (no file)")
            continue
        table[name] = {}
        for m in metrics:
            rho, pv, common, _, _ = spearman(m, mobius, agg)
            table[name][m] = {"rho": rho, "p": pv, "n": len(common)}

    # Печать таблицы
    print("\n=== Cross-domain Spearman ρ vs Mobius (mean across 5 seeds) ===\n")
    header = f"{'Dataset':<24} | " + " | ".join(f"{m[:18]:<18}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, mts in table.items():
        cells = []
        for m in metrics:
            v = mts[m]
            star = " *" if v["p"] < 0.05 and not np.isnan(v["rho"]) else "  "
            cells.append(f"ρ={v['rho']:+.2f} p={v['p']:.2f}{star}")
        print(f"{name:<24} | " + " | ".join(f"{c:<18}" for c in cells))

    # Сохранение
    out = {
        "datasets": list(table.keys()),
        "metrics": metrics,
        "table": table,
    }
    out_path = ROOT / "results" / f"cross_domain_spearman_all{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")

    # Compact heatmap
    n = len(table)
    m_count = len(metrics)
    arr = np.zeros((n, m_count))
    sig = np.zeros((n, m_count), dtype=bool)
    for i, name in enumerate(table):
        for j, m in enumerate(metrics):
            arr[i, j] = table[name][m]["rho"]
            sig[i, j] = table[name][m]["p"] < 0.05

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(m_count))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(table.keys()))
    for i in range(n):
        for j in range(m_count):
            txt = f"{arr[i, j]:+.2f}"
            if sig[i, j]:
                txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black" if abs(arr[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Cross-domain robustness: Spearman ρ rankings vs Mobius (* = p < 0.05)")
    fig.tight_layout()
    plot_path = ROOT / "results" / "plots" / f"85_cross_domain_spearman_all{suffix}.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"WROTE: {plot_path}")


if __name__ == "__main__":
    main()
