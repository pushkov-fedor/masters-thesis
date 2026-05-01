"""Async-прогон LLM-ranker (state-aware) на конференции.

Слоты обрабатываются параллельно (asyncio); внутри слота — последовательно.
Это сохраняет state-awareness, но даёт ~10× ускорение за счёт параллельных
API-вызовов между независимыми слотами.

Запуск:
    .venv/bin/python scripts/run_llm_ranker_sa_async.py \
        --conference demo_day_2026 \
        --personas personas_100 \
        --seeds 1 \
        --concurrency 10 \
        --budget 8.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate_async_slots, cosine_relevance  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.policies.llm_ranker_state_aware_policy import LLMRankerStateAwarePolicy  # noqa: E402


def load_users(personas_name: str):
    with open(ROOT / "data" / "personas" / f"{personas_name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / f"{personas_name}_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}

    def text_of(p):
        return p.get("background") or p.get("profile") or ""

    return [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in meta
    ]


async def run_seed(conf, users, policy, cfg, concurrency):
    return await simulate_async_slots(
        conf=conf, users=users, policy=policy, cfg=cfg,
        relevance_fn=cosine_relevance,
        concurrency=concurrency,
    )


async def main_async(args):
    conf = Conference.load(
        ROOT / "data" / "conferences" / f"{args.conference}.json",
        ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz",
    )
    users = load_users(args.personas)
    print(f"Conference: {conf.name}, talks={len(conf.talks)}, halls={len(conf.halls)}, slots={len(conf.slots)}")
    print(f"Users: {len(users)}, seeds: {args.seeds}, K={args.K}, concurrency={args.concurrency}")

    policy = LLMRankerStateAwarePolicy(
        model=args.llm_model,
        budget_usd=args.budget,
        async_concurrency=args.api_concurrency,
    )
    print(f"State-aware LLM-ranker, model={args.llm_model}, "
          f"budget=${args.budget}, api_concurrency={args.api_concurrency}")
    print(f"  cache size at start: {len(policy.cache)} entries")

    n_active_slots = sum(1 for s in conf.slots if s.talk_ids)
    policy.set_progress_total(len(args.seeds) * len(users) * n_active_slots,
                              desc="LLM-ranker-SA-async")

    runs = []
    t0_all = time.time()
    for seed in args.seeds:
        cfg = SimConfig(
            K=args.K, tau=args.tau, lambda_overflow=args.lambda_overflow,
            p_skip_base=args.p_skip, seed=seed, w_fame=args.w_fame,
            user_compliance=args.user_compliance,
            use_calibrated_compliance=args.calibrated_compliance,
            alpha_compliant=args.alpha_compliant,
            alpha_starchaser=args.alpha_starchaser,
            alpha_curious=args.alpha_curious,
        )
        t0 = time.time()
        sim = await run_seed(conf, users, policy, cfg, args.concurrency)
        elapsed = time.time() - t0
        m = compute_all(conf, sim)
        runs.append({"policy": "LLM-ranker (state-aware)", "seed": seed,
                     "elapsed_s": elapsed, "metrics": m})
        print(f"\nseed={seed} elapsed={elapsed:.1f}s OF_choice={m['overflow_rate_choice']:.3f} "
              f"var={m['hall_utilization_variance']:.3f} util={m['mean_user_utility']:.3f} "
              f"excess={m['mean_overload_excess']:.3f}")

    policy.close_progress()
    policy._flush()

    out = {
        "config": {
            "K": args.K, "tau": args.tau, "lambda_overflow": args.lambda_overflow,
            "p_skip_base": args.p_skip, "seeds": args.seeds, "n_users": len(users),
            "conference": conf.name, "concurrency": args.concurrency,
            "api_concurrency": args.api_concurrency, "model": args.llm_model,
        },
        "runs": runs,
    }
    out_path = ROOT / "results" / f"results_llm_sa_async{args.results_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")

    # Summary
    by_metric = {}
    for r in runs:
        for k, v in r["metrics"].items():
            by_metric.setdefault(k, []).append(v)
    lines = ["# LLM-ranker (state-aware) — async\n"]
    lines.append(f"Конференция: {conf.name}, N={len(users)}, сидов: {len(args.seeds)}")
    lines.append(f"concurrency={args.concurrency}, api_concurrency={args.api_concurrency}\n")
    metrics_order = ["overflow_rate_all", "overflow_rate_choice",
                     "hall_utilization_variance", "mean_user_utility",
                     "hall_load_gini", "skip_rate", "mean_overload_excess"]
    for m in metrics_order:
        vals = by_metric.get(m, [])
        if not vals: continue
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        lines.append(f"- {m}: {mu:.3f} ± {sd:.3f}")
    summary_path = ROOT / "results" / f"summary_llm_sa_async{args.results_suffix}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"WROTE: {summary_path}")
    print(f"\nTOTAL ELAPSED: {time.time() - t0_all:.1f}s")
    s = policy.stats()
    print(f"LLM-ranker SA: cumulative cost ${s['cumulative_cost_usd']:.4f}, "
          f"cache size {s['cache_size']}, api_calls={policy.n_api_calls}, hits={policy.n_cache_hits}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="demo_day_2026")
    ap.add_argument("--personas", default="personas_100")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--lambda-overflow", type=float, default=2.0)
    ap.add_argument("--p-skip", type=float, default=0.05)
    ap.add_argument("--w-fame", type=float, default=0.3)
    ap.add_argument("--user-compliance", type=float, default=1.0)
    ap.add_argument("--calibrated-compliance", action="store_true",
                    help="Использовать трёх-типную модель compliance "
                         "(compliant 71.7% / star-chaser 21.3% / curious 7.0%, "
                         "по калибровке на Meetup RSVPs)")
    ap.add_argument("--alpha-compliant", type=float, default=0.717)
    ap.add_argument("--alpha-starchaser", type=float, default=0.213)
    ap.add_argument("--alpha-curious", type=float, default=0.070)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--llm-model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="число одновременно обрабатываемых слотов")
    ap.add_argument("--api-concurrency", type=int, default=20,
                    help="макс. параллельных API-вызовов в семафоре политики")
    ap.add_argument("--results-suffix", default="")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
