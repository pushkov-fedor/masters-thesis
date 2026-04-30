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
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate, LearnedPreferenceFn  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.mmr_policy import MMRPolicy  # noqa: E402
from src.policies.capacity_aware_mmr_policy import CapacityAwareMMRPolicy  # noqa: E402
from src.policies.llm_ranker_policy import LLMRankerPolicy  # noqa: E402
from src.policies.llm_ranker_state_aware_policy import LLMRankerStateAwarePolicy  # noqa: E402
from src.policies.ppo_policy import PPOPolicy  # noqa: E402
from src.policies.dpp_policy import DPPPolicy  # noqa: E402
from src.policies.calibrated_policy import CalibratedPolicy  # noqa: E402
from src.policies.sequential_policy import SequentialPolicy  # noqa: E402
from src.policies.gnn_policy import GNNPolicy  # noqa: E402


def load_users(personas_name: str = "personas") -> List[UserProfile]:
    with open(ROOT / "data" / "personas" / f"{personas_name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / f"{personas_name}_embeddings.npz", allow_pickle=False)
    ids = list(npz["ids"])
    emb = npz["embeddings"]
    by_id = {pid: emb[i] for i, pid in enumerate(ids)}

    def text_of(p):
        # LLM-персоны: background; fallback-персоны: profile
        return p.get("background") or p.get("profile") or ""

    return [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in meta
    ]


def make_policies(seed: int, llm_ranker=None, llm_ranker_sa=None, ppo=None,
                  with_modern=False):
    policies = {
        "Random": RandomPolicy(seed=seed),
        "Cosine": CosinePolicy(),
        "MMR": MMRPolicy(beta=0.7),
        "Capacity-aware": CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95),
        "Capacity-aware MMR": CapacityAwareMMRPolicy(beta=0.6, alpha=0.4, hard_threshold=0.95),
    }
    if with_modern:
        policies["DPP"] = DPPPolicy(alpha=0.5)
        policies["Calibrated"] = CalibratedPolicy(lambda_kl=0.5)
        policies["Sequential"] = SequentialPolicy(history_weight=0.6, history_window=5)
        policies["GNN"] = GNNPolicy(edge_threshold=0.5, n_layers=2, self_weight=0.5)
    if ppo is not None:
        policies["Constrained-PPO"] = ppo
    if llm_ranker is not None:
        policies["LLM-ranker"] = llm_ranker
    if llm_ranker_sa is not None:
        policies["LLM-ranker (state-aware)"] = llm_ranker_sa
    return policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--lambda-overflow", type=float, default=2.0)
    ap.add_argument("--p-skip", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="1 сид, 80 пользователей (для smoke)")
    ap.add_argument("--personas", default="personas", help="имя файла personas в data/personas/ (без .json)")
    ap.add_argument("--with-ppo", action="store_true", help="включить Constrained PPO")
    ap.add_argument("--with-modern", action="store_true", help="включить DPP, Calibrated, Sequential, GNN")
    ap.add_argument("--with-llm", action="store_true", help="включить LLM-ranker как 6-ю политику")
    ap.add_argument("--with-llm-sa", action="store_true", help="включить state-aware LLM-ranker (7-я)")
    ap.add_argument("--llm-budget", type=float, default=2.0, help="USD potolok на LLM-ranker")
    ap.add_argument("--llm-budget-sa", type=float, default=2.0, help="USD potolok на state-aware")
    ap.add_argument("--llm-model", default="openai/gpt-4o-mini")
    ap.add_argument("--relevance", choices=["cosine", "learned"], default="cosine",
                    help="Способ вычисления релевантности: cosine (по умолчанию) или learned")
    ap.add_argument("--results-suffix", default="",
                    help="Суффикс к имени результатов (например _learned)")
    ap.add_argument("--conference", default="mobius_2025_autumn",
                    help="ID конференции для прогона (mobius_2025_autumn или demo_day_2026)")
    args = ap.parse_args()

    conf = Conference.load(
        ROOT / "data" / "conferences" / f"{args.conference}.json",
        ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz",
    )
    users = load_users(args.personas)
    if args.quick:
        users = users[:80]
        args.seeds = [1]
    print(f"Conference: {conf.name}, talks={len(conf.talks)}, halls={len(conf.halls)}, slots={len(conf.slots)}")
    print(f"Users: {len(users)}, seeds: {args.seeds}, K={args.K}, tau={args.tau}")

    relevance_fn = None
    if args.relevance == "learned":
        model_path = ROOT / "data" / "models" / "preference_model.pkl"
        if not model_path.exists():
            raise SystemExit(f"Learned model not found at {model_path}. "
                             "Run scripts/train_preference_model.py first.")
        relevance_fn = LearnedPreferenceFn(model_path)
        print(f"Using LEARNED preference model from {model_path}")
        # Precompute all (persona, talk) preferences to avoid per-call overhead
        print("Precomputing learned preferences for all (persona, talk) pairs...")
        t_pre = time.time()
        persona_dict = {u.id: u.embedding for u in users}
        talk_dict = {tid: t.embedding for tid, t in conf.talks.items()}
        relevance_fn.precompute_all(persona_dict, talk_dict)
        print(f"  done in {time.time()-t_pre:.1f}s, cache size: {len(relevance_fn._cache)}")

    ppo = None
    if args.with_ppo:
        ppo_path = ROOT / "data" / "models" / "ppo_policy.zip"
        if ppo_path.exists():
            ppo = PPOPolicy(ppo_path)
            print(f"PPO loaded from {ppo_path}")

    llm_ranker = None
    llm_ranker_sa = None
    if args.with_llm:
        llm_ranker = LLMRankerPolicy(model=args.llm_model, budget_usd=args.llm_budget)
        print(f"LLM-ranker enabled: model={args.llm_model}, budget=${args.llm_budget}")
        print(f"  cache size at start: {len(llm_ranker.cache)} entries")
    if args.with_llm_sa:
        llm_ranker_sa = LLMRankerStateAwarePolicy(model=args.llm_model, budget_usd=args.llm_budget_sa)
        print(f"State-aware LLM-ranker enabled: budget=${args.llm_budget_sa}")
        print(f"  cache size at start: {len(llm_ranker_sa.cache)} entries")

    results: List[dict] = []
    t0_all = time.time()
    for seed in args.seeds:
        policies = make_policies(seed, llm_ranker=llm_ranker, llm_ranker_sa=llm_ranker_sa, ppo=ppo,
                                 with_modern=args.with_modern)
        for pname, pol in policies.items():
            cfg = SimConfig(
                K=args.K, tau=args.tau, lambda_overflow=args.lambda_overflow,
                p_skip_base=args.p_skip, seed=seed,
            )
            t0 = time.time()
            sim = simulate(conf, users, pol, cfg, relevance_fn=relevance_fn)
            elapsed = time.time() - t0
            m = compute_all(conf, sim)
            results.append({
                "policy": pname,
                "seed": seed,
                "elapsed_s": elapsed,
                "metrics": m,
            })
            print(f"seed={seed} {pname:<22} elapsed={elapsed:5.1f}s "
                  f"OF_all={m['overflow_rate_all']:.3f} OF_choice={m['overflow_rate_choice']:.3f} "
                  f"var={m['hall_utilization_variance']:.3f} "
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
    out_path = ROOT / "results" / f"results{args.results_suffix}.json"
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

    metrics_order = ["overflow_rate_all", "overflow_rate_choice",
                     "hall_utilization_variance", "mean_user_utility",
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
    summary_path = ROOT / "results" / f"summary{args.results_suffix}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"WROTE: {summary_path}")
    print(f"\nTOTAL ELAPSED: {time.time() - t0_all:.1f}s")
    if llm_ranker is not None:
        s = llm_ranker.stats()
        llm_ranker._flush()
        print(f"LLM-ranker: cumulative cost ${s['cumulative_cost_usd']:.4f}, cache size {s['cache_size']}")
    if llm_ranker_sa is not None:
        s = llm_ranker_sa.stats()
        llm_ranker_sa._flush()
        print(f"State-aware LLM-ranker: cumulative cost ${s['cumulative_cost_usd']:.4f}, cache size {s['cache_size']}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
