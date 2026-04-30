"""Стадия 5: прогон LLM-агентов с памятью на конференции для каждой политики.

Использование:
  python scripts/run_agent_validation.py --n-agents 50 --policies Cosine,Capacity-aware,LLM-ranker

Каждая политика прогоняется на одной и той же популяции агентов; результаты
сравниваются с параметрическим симулятором (по тем же метрикам).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, UserProfile, LearnedPreferenceFn  # noqa: E402
from src.agents.generative_agent import GenerativeAgent  # noqa: E402
from src.agents.agent_simulator import simulate_agents, compute_agent_metrics  # noqa: E402
from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.mmr_policy import MMRPolicy  # noqa: E402
from src.policies.capacity_aware_mmr_policy import CapacityAwareMMRPolicy  # noqa: E402
from src.policies.llm_ranker_policy import LLMRankerPolicy  # noqa: E402

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY not found")


def build_policies(seed, llm_ranker=None):
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


def text_of(p):
    return p.get("background") or p.get("profile") or ""


async def main_async(args):
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Load conf and personas
    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas_full = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}

    # Subsample to N agents (deterministic by seed)
    rng = np.random.default_rng(args.subsample_seed)
    n_total = len(personas_full)
    if args.n_agents >= n_total:
        selected = list(range(n_total))
    else:
        selected = rng.choice(n_total, size=args.n_agents, replace=False).tolist()
        selected = [int(i) for i in selected]

    chosen = [personas_full[i] for i in selected]
    user_profiles = [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in chosen
    ]
    print(f"Conference: {conf.name}, {len(conf.talks)} talks, {len(conf.slots)} slots")
    print(f"Subsampled {len(chosen)} agents from {n_total}")

    # Set up policies
    policies_to_run = args.policies.split(",")
    print(f"Policies: {policies_to_run}")

    # LLM ranker (read-only cache)
    llm_ranker = None
    if "LLM-ranker" in policies_to_run:
        llm_ranker = LLMRankerPolicy(
            model="openai/gpt-4o-mini",
            budget_usd=2.0,
        )

    relevance_fn = None
    if args.relevance == "learned":
        model_path = ROOT / "data" / "models" / "preference_model.pkl"
        relevance_fn = LearnedPreferenceFn(model_path)
        print(f"Using LEARNED relevance for non-LLM policies")

    all_policies = build_policies(seed=args.seed, llm_ranker=llm_ranker)
    selected_policies = {k: v for k, v in all_policies.items() if k in policies_to_run}

    # Run each policy on the SAME agents (fresh memory per policy run)
    all_results = {}
    t0_global = time.time()
    cumulative_cost = 0.0

    for policy_name, policy_fn in selected_policies.items():
        print(f"\n=== Running policy: {policy_name} ===")
        # Fresh agents (clean memory)
        agents = [
            GenerativeAgent(
                agent_id=p["id"],
                persona=text_of(p),
                client=client,
                model=args.model,
            )
            for p in chosen
        ]

        # Inject relevance_fn into user_profiles via state — we already do that in simulator,
        # but agent_simulator builds its own state without it. So let's pre-wrap policies.
        if relevance_fn is not None:
            class WrappedPolicy:
                def __init__(self, base, rfn):
                    self.base = base
                    self.rfn = rfn
                def __call__(self, *, user, slot, conf, state):
                    state = dict(state)
                    state["relevance_fn"] = self.rfn
                    return self.base(user=user, slot=slot, conf=conf, state=state)
            wrapped = WrappedPolicy(policy_fn, relevance_fn)
        else:
            wrapped = policy_fn

        t0 = time.time()
        sim = await simulate_agents(
            conf=conf,
            agents=agents,
            policy=wrapped,
            user_profiles=user_profiles,
            K=args.K,
            concurrency=args.concurrency,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        m = compute_agent_metrics(conf, sim, K=args.K)
        cumulative_cost += sim.total_cost

        print(f"  elapsed: {elapsed:.0f}s, cost: ${sim.total_cost:.3f}, "
              f"errors: {sim.total_errors}, decisions: {len(sim.decisions)}")
        print(f"  metrics: OF_all={m['overflow_rate_all']:.3f}, "
              f"OF_choice={m['overflow_rate_choice']:.3f}, "
              f"var={m['hall_utilization_variance']:.3f}, "
              f"skip_rate={m['skip_rate']:.3f}")

        all_results[policy_name] = {
            "metrics": m,
            "decisions": sim.decisions,
            "elapsed_s": elapsed,
        }
        # Save snapshot after each policy
        out_path = ROOT / "results" / f"agent_validation_{args.suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "n_agents": len(chosen),
                    "policies": list(selected_policies.keys()),
                    "K": args.K,
                    "model": args.model,
                    "relevance": args.relevance,
                    "seed": args.seed,
                },
                "results": all_results,
                "cumulative_cost_usd": cumulative_cost,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nTOTAL: {time.time()-t0_global:.0f}s, ${cumulative_cost:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=50)
    p.add_argument("--policies", default="Random,Cosine,MMR,Capacity-aware,Capacity-aware MMR,LLM-ranker")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-seed", type=int, default=7)
    p.add_argument("--model", default="anthropic/claude-haiku-4.5")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--relevance", choices=["cosine", "learned"], default="cosine")
    p.add_argument("--suffix", default="default")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
