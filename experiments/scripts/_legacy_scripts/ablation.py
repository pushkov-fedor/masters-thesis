"""Ablation studies: sweep по τ, λ, K — отвечает на вопрос «устойчив ли результат».

Прогон каждой из 6 политик по сетке параметров. Только 3 сида ради скорости.
Графики: для каждого параметра — 6 кривых (по политикам) для overflow_rate_choice
и mean_user_utility.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.mmr_policy import MMRPolicy  # noqa: E402
from src.policies.capacity_aware_mmr_policy import CapacityAwareMMRPolicy  # noqa: E402
from src.policies.llm_ranker_policy import LLMRankerPolicy  # noqa: E402

PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIG = ROOT.parent / "thesis" / "figures"

POLICY_ORDER = ["Random", "Cosine", "MMR", "Capacity-aware", "Capacity-aware MMR", "LLM-ranker"]
POLICY_COLORS = {
    "Random": "#888888",
    "Cosine": "#1f77b4",
    "MMR": "#9467bd",
    "Capacity-aware": "#2ca02c",
    "Capacity-aware MMR": "#d62728",
    "LLM-ranker": "#ff7f0e",
}


def text_of(p):
    return p.get("background") or p.get("profile") or ""


def load_users(name="personas_x3"):
    with open(ROOT / "data" / "personas" / f"{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / f"{name}_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    return [UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]]) for p in meta]


def make_policies(seed, llm_ranker=None):
    p = {
        "Random": RandomPolicy(seed=seed),
        "Cosine": CosinePolicy(),
        "MMR": MMRPolicy(beta=0.7),
        "Capacity-aware": CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95),
        "Capacity-aware MMR": CapacityAwareMMRPolicy(beta=0.6, alpha=0.4, hard_threshold=0.95),
    }
    if llm_ranker is not None:
        p["LLM-ranker"] = llm_ranker
    return p


def run_grid(conf, users, llm_ranker, sweep_param: str, values: list, fixed: dict, seeds: list):
    """Возвращает {(policy, param_value, seed): metrics}."""
    rows = []
    for v in values:
        for seed in seeds:
            cfg_kwargs = {**fixed, sweep_param: v, "seed": seed}
            cfg = SimConfig(**cfg_kwargs)
            policies = make_policies(seed, llm_ranker)
            for pname, pol in policies.items():
                t0 = time.time()
                sim = simulate(conf, users, pol, cfg)
                m = compute_all(conf, sim)
                rows.append({
                    "policy": pname, sweep_param: v, "seed": seed,
                    "elapsed_s": time.time() - t0, "metrics": m,
                })
    return rows


def aggregate(rows, sweep_param, metric):
    """{policy: ([param_values_sorted], [mean], [std])}"""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r[sweep_param]].append(r["metrics"][metric])
    result = {}
    for p, m in by.items():
        xs = sorted(m.keys())
        means = [mean(m[x]) for x in xs]
        stds = [stdev(m[x]) if len(m[x]) > 1 else 0.0 for x in xs]
        result[p] = (xs, means, stds)
    return result


def plot_sweep(rows, sweep_param, xlabel, fname, log_x=False):
    """Двухпанельный график: overflow_rate_choice + mean_user_utility."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(
        axes,
        ["overflow_rate_choice", "mean_user_utility"],
        ["Overflow rate (choice slots)", "Mean user utility"],
    ):
        agg = aggregate(rows, sweep_param, metric)
        for p in POLICY_ORDER:
            if p not in agg:
                continue
            xs, means, stds = agg[p]
            ax.errorbar(xs, means, yerr=stds, label=p,
                        color=POLICY_COLORS[p], marker="o", markersize=6,
                        capsize=3, linewidth=1.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Ablation: {xlabel} (5 сидов, 900 пользователей)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150)
    if THESIS_FIG.exists():
        import shutil
        shutil.copy(PLOT_DIR / fname, THESIS_FIG / fname)
    plt.close(fig)
    print(f"WROTE: {PLOT_DIR / fname}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--with-llm", action="store_true")
    args = p.parse_args()

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )
    users = load_users("personas_x3")
    llm_ranker = LLMRankerPolicy(model="openai/gpt-4o-mini", budget_usd=0.5) if args.with_llm else None
    print(f"Users: {len(users)}, seeds: {args.seeds}, with_llm: {args.with_llm}")

    sweeps = [
        # (param, values, xlabel, fname)
        ("tau", [0.1, 0.2, 0.3, 0.5, 1.0, 2.0], "τ (softmax temperature)", "10_ablation_tau.png"),
        ("lambda_overflow", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0], "λ (overflow penalty)", "11_ablation_lambda.png"),
        ("K", [1, 2, 3], "K (recommendation list size)", "12_ablation_K.png"),
    ]

    fixed = {"K": 2, "tau": 0.3, "lambda_overflow": 2.0, "p_skip_base": 0.05}

    all_rows = {}
    t0 = time.time()
    for param, values, xlabel, fname in sweeps:
        f = {k: v for k, v in fixed.items() if k != param}
        print(f"\n=== Sweep {param} ∈ {values} ===")
        rows = run_grid(conf, users, llm_ranker, param, values, f, args.seeds)
        all_rows[param] = rows
        plot_sweep(rows, param, xlabel, fname)
        # сохраняем сырые числа
        with open(ROOT / "results" / f"ablation_{param}.json", "w", encoding="utf-8") as out:
            json.dump(rows, out, ensure_ascii=False, indent=2)
        print(f"  done in {time.time()-t0:.0f}s")

    # Объединённая таблица
    summary_lines = [f"# Ablation studies\n",
                     f"Параметры по умолчанию: {fixed}\n",
                     f"Сидов: {len(args.seeds)}, пользователей: {len(users)}\n"]
    for param, values, _, _ in sweeps:
        summary_lines.append(f"\n## Sweep {param}")
        rows = all_rows[param]
        agg_overflow = aggregate(rows, param, "overflow_rate_choice")
        agg_util = aggregate(rows, param, "mean_user_utility")
        for pname in POLICY_ORDER:
            if pname not in agg_overflow:
                continue
            xs, means_o, _ = agg_overflow[pname]
            _, means_u, _ = agg_util[pname]
            summary_lines.append(f"\n### {pname}")
            summary_lines.append(f"| {param} | overflow_choice | utility |")
            summary_lines.append("|---|---|---|")
            for x, mo, mu in zip(xs, means_o, means_u):
                summary_lines.append(f"| {x} | {mo:.3f} | {mu:.3f} |")
    with open(ROOT / "results" / "ablation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nWROTE: {ROOT / 'results' / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
