"""Прогон OASIS-style агентов (v2) на программе конференции для каждой политики.

Использует полноценную модель агента: память с retrieval, Big Five personality,
fatigue, социальный граф, рефлексия.

Артефакты для тестирования research-гипотез H1-H5:
- agent_validation_v2_<conf>.json — все decisions + personality + fatigue trajectories
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
from src.agents.generative_agent_v2 import GenerativeAgentV2  # noqa: E402
from src.agents.agent_simulator_v2 import simulate_agents_v2  # noqa: E402
from src.agents.social_graph import SocialGraph  # noqa: E402
from src.agents.personality import BigFive  # noqa: E402
from src.agents.inter_slot_chat import InterSlotChatPool  # noqa: E402

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

    conf = Conference.load(
        ROOT / "data" / "conferences" / f"{args.conference}.json",
        ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas_full = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}

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

    relevance_fn = None
    if args.relevance == "learned":
        model_path = ROOT / "data" / "models" / "preference_model.pkl"
        relevance_fn = LearnedPreferenceFn(model_path)
        persona_dict = {p["id"]: by_id[p["id"]] for p in chosen}
        talk_dict = {tid: t.embedding for tid, t in conf.talks.items()}
        relevance_fn.precompute_all(persona_dict, talk_dict)
        print(f"  precomputed {len(relevance_fn._cache)} learned-relevance values")

    policies_to_run = args.policies.split(",")
    print(f"Policies: {policies_to_run}")

    llm_ranker = None
    if "LLM-ranker" in policies_to_run:
        llm_ranker = LLMRankerPolicy(model="openai/gpt-4o-mini", budget_usd=2.0)

    all_policies = build_policies(seed=args.seed, llm_ranker=llm_ranker)
    selected_policies = {k: v for k, v in all_policies.items() if k in policies_to_run}

    all_results = {}
    t0_global = time.time()
    cumulative_cost = 0.0

    # Социальный граф один раз — общий для всех политик (одни и те же агенты)
    social_graph_template_seed = args.seed

    for policy_name, policy_fn in selected_policies.items():
        print(f"\n=== Running policy: {policy_name} ===")
        # Свежие агенты + свежий граф для каждой политики (чтобы не было утечки)
        agents = []
        for i, p in enumerate(chosen):
            persona = text_of(p)
            personality = BigFive.from_persona_text(persona, seed=hash(p["id"]) % 10000)
            agent = GenerativeAgentV2(
                agent_id=p["id"],
                agent_idx=i,
                persona_text=persona,
                personality=personality,
                client=client,
                model=args.model,
                reflection_threshold=2.5,
            )
            agents.append(agent)

        social_graph = SocialGraph(
            n_agents=len(chosen),
            k=6,
            p_rewire=0.1,
            seed=social_graph_template_seed,
        )

        chat_pool = None
        if args.with_chat:
            chat_pool = InterSlotChatPool(client=client, model=args.model)

        t0 = time.time()
        sim = await simulate_agents_v2(
            conf=conf,
            agents=agents,
            policy=policy_fn,
            user_profiles=user_profiles,
            social_graph=social_graph,
            K=args.K,
            concurrency=args.concurrency,
            seed=args.seed,
            relevance_fn=relevance_fn,
            chat_pool=chat_pool,
            chat_sample_fraction=args.chat_sample,
        )
        chat_cost = chat_pool.cumulative_cost if chat_pool else 0.0
        chat_posts = len(chat_pool.posts) if chat_pool else 0
        elapsed = time.time() - t0
        cumulative_cost += sim.total_cost

        # Метрики
        n_dec = len(sim.decisions)
        n_skip = sum(1 for d in sim.decisions if d["decision"] == "skip")
        skip_rate = n_skip / max(1, n_dec)

        overflow_choice = 0
        total_choice = 0
        for slot in conf.slots:
            if not slot.talk_ids or len(slot.talk_ids) <= 1:
                continue
            halls = {conf.talks[tid].hall for tid in slot.talk_ids}
            for hid in halls:
                cap = conf.halls[hid].capacity
                occ = sim.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
                total_choice += 1
                if occ > cap:
                    overflow_choice += 1
        of_choice = overflow_choice / max(1, total_choice)

        print(f"  elapsed: {elapsed:.0f}s, cost: ${sim.total_cost:.3f}, "
              f"errors: {sim.total_errors}, decisions: {n_dec}")
        print(f"  metrics: OF_choice={of_choice:.3f}, skip_rate={skip_rate:.3f}")
        if chat_pool:
            print(f"  chat: {chat_posts} posts, ${chat_cost:.3f}")
        cumulative_cost += chat_cost

        all_results[policy_name] = {
            "metrics": {
                "overflow_rate_choice": of_choice,
                "skip_rate": skip_rate,
                "total_decisions": n_dec,
                "total_skips": n_skip,
                "total_cost_usd": sim.total_cost,
                "total_errors": sim.total_errors,
                "chat_posts": chat_posts,
                "chat_cost_usd": chat_cost,
            },
            "decisions": sim.decisions,  # для тестов H1, H3
            "skip_rate_per_slot": sim.skip_rate_per_slot,  # для H2 (fatigue)
            "fatigue_per_agent": sim.fatigue_per_agent_per_slot,
            "personality_per_agent": sim.personality_per_agent,
            "social_graph_adjacency": {str(k): list(v) for k, v in social_graph.adjacency.items()},
            "elapsed_s": elapsed,
        }
        # Save snapshot after each policy
        out_path = ROOT / "results" / f"agent_validation_v2_{args.conference}_{args.suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "n_agents": len(chosen),
                    "policies": list(selected_policies.keys()),
                    "K": args.K,
                    "model": args.model,
                    "relevance": args.relevance,
                    "seed": args.seed,
                    "conference": args.conference,
                },
                "results": all_results,
                "cumulative_cost_usd": cumulative_cost,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nTOTAL: {time.time()-t0_global:.0f}s, ${cumulative_cost:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conference", default="mobius_2025_autumn")
    p.add_argument("--n-agents", type=int, default=100)
    p.add_argument("--policies", default="Random,Cosine,MMR,Capacity-aware,Capacity-aware MMR,LLM-ranker")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-seed", type=int, default=7)
    p.add_argument("--model", default="anthropic/claude-haiku-4.5")
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--relevance", choices=["cosine", "learned"], default="learned")
    p.add_argument("--suffix", default="default")
    p.add_argument("--with-chat", action="store_true",
                   help="Включить inter-slot chat pool (MiroFish-вдохновение)")
    p.add_argument("--chat-sample", type=float, default=0.3,
                   help="Доля посетивших, которые пишут пост")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
