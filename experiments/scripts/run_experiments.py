"""Запуск основных экспериментов: 5 политик × 5 сидов × все персоны.

Выход:
- results/results.json — детальные числа по каждому (policy, seed, metric)
- results/summary.md — markdown-таблица mean ± std по политикам
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

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


def load_users() -> List[UserProfile]:
    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    ids = list(npz["ids"])
    emb = npz["embeddings"]
    by_id = {pid: emb[i] for i, pid in enumerate(ids)}
    return [
        UserProfile(id=p["id"], text=p.get("profile", ""), embedding=by_id[p["id"]])
        for p in meta
    ]


def make_policies(seed: int):
    return {
        "Random": RandomPolicy(seed=seed),
        "Cosine": CosinePolicy(),
        "MMR": MMRPolicy(beta=0.7),
        "Capacity-aware": CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95),
        "Capacity-aware MMR": CapacityAwareMMRPolicy(beta=0.6, alpha=0.4, hard_threshold=0.95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--lambda-overflow", type=float, default=2.0)
    ap.add_argument("--p-skip", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="1 сид, 80 пользователей (для smoke)")
    args = ap.parse_args()

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )
    users = load_users()
    if args.quick:
        users = users[:80]
        args.seeds = [1]
    print(f"Conference: {conf.name}, talks={len(conf.talks)}, halls={len(conf.halls)}, slots={len(conf.slots)}")
    print(f"Users: {len(users)}, seeds: {args.seeds}, K={args.K}, tau={args.tau}")

    results: List[dict] = []
    t0_all = time.time()
    for seed in args.seeds:
        policies = make_policies(seed)
        for pname, pol in policies.items():
            cfg = SimConfig(
                K=args.K, tau=args.tau, lambda_overflow=args.lambda_overflow,
                p_skip_base=args.p_skip, seed=seed,
            )
            t0 = time.time()
            sim = simulate(conf, users, pol, cfg)
            elapsed = time.time() - t0
            m = compute_all(conf, sim)
            results.append({
                "policy": pname,
                "seed": seed,
                "elapsed_s": elapsed,
                "metrics": m,
            })
            print(f"seed={seed} {pname:<22} elapsed={elapsed:5.1f}s "
                  f"overflow={m['overflow_rate']:.3f} var={m['hall_utilization_variance']:.3f} "
                  f"util={m['mean_user_utility']:.3f} gini={m['hall_load_gini']:.3f}")

    out = {
        "config": {
            "K": args.K, "tau": args.tau,
            "lambda_overflow": args.lambda_overflow,
            "p_skip_base": args.p_skip,
            "seeds": args.seeds,
            "n_users": len(users),
            "conference": conf.name,
        },
        "runs": results,
    }
    out_path = ROOT / "results" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")

    # Aggregated summary
    by_policy: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        d = by_policy.setdefault(r["policy"], {})
        for k, v in r["metrics"].items():
            d.setdefault(k, []).append(v)

    metrics_order = ["overflow_rate", "hall_utilization_variance", "mean_user_utility",
                     "hall_load_gini", "skip_rate", "mean_overload_excess"]

    lines = ["# Сводная таблица результатов\n"]
    lines.append(f"Конференция: **{conf.name}** | Пользователей: {len(users)} | Сидов: {len(args.seeds)}")
    lines.append(f"Параметры: K={args.K}, τ={args.tau}, λ={args.lambda_overflow}, p_skip={args.p_skip}\n")
    header = "| Политика | " + " | ".join(metrics_order) + " |"
    sep = "|" + "---|" * (len(metrics_order) + 1)
    lines.append(header)
    lines.append(sep)
    for pname, d in by_policy.items():
        cells = [pname]
        for m in metrics_order:
            vals = d.get(m, [])
            if not vals:
                cells.append("-")
                continue
            mu = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mu:.3f} ± {sd:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    summary = "\n".join(lines) + "\n"
    summary_path = ROOT / "results" / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"WROTE: {summary_path}")
    print(f"\nTOTAL ELAPSED: {time.time() - t0_all:.1f}s")
    print("\n" + summary)


if __name__ == "__main__":
    main()
