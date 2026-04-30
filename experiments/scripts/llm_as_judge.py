"""Стадия 6: LLM-as-judge для финальной оценки качества рекомендаций каждой политики.

Pairwise сравнение: для случайной выборки (персона, слот) пар, для каждой пары политик
запрашиваем у LLM-судьи, какая из двух выдач лучше.

Метрика: Bradley-Terry rating политик на основе wins/losses.

Стоимость: 30 персон × 15 пар политик = 450 вызовов × ~2500 prompt + 100 completion
         = 1.13M prompt + 0.045M completion токенов
         × Sonnet 4.6: $3/M prompt, $15/M completion ≈ $4.05
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, UserProfile, LearnedPreferenceFn  # noqa: E402
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

PRICING = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
    "anthropic/claude-sonnet-4.6": {"prompt": 3.0, "completion": 15.0},
    "openai/gpt-4o": {"prompt": 2.5, "completion": 10.0},
}


SYSTEM_PROMPT = """Ты — независимый эксперт-судья, оценивающий качество рекомендаций программы IT-конференции.

Тебе даётся:
- Профиль участника конференции
- Текущий тайм-слот и параллельные доклады в разных залах с информацией о загрузке
- Две конкурирующие выдачи из top-K (политика A vs политика B)

Оцени, какая из двух выдач лучше для этого участника, по совокупности критериев:
1. **Релевантность** — насколько доклады соответствуют интересам участника
2. **Разнообразие** — есть ли тематическое разнообразие, или только одно
3. **Ёмкость** — учтена ли загрузка залов (плохо направлять в переполненные)
4. **Справедливость** — сбалансировано ли представлены доклады разных категорий/спикеров

Возвращай строго JSON:
{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0-1.0,
  "reasoning": "одно-два предложения почему"
}"""


USER_TEMPLATE = """Профиль участника:
{persona}

Тайм-слот: {slot_time}. Параллельные доклады:
{candidates}

ВЫДАЧА A:
{recs_a}

ВЫДАЧА B:
{recs_b}

Какая выдача лучше — A или B?"""


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


def parse_judgement(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        d = json.loads(text[i : j + 1])
        winner = str(d.get("winner", "tie"))
        confidence = float(d.get("confidence", 0.5))
        reasoning = str(d.get("reasoning", ""))
        if winner not in ("A", "B", "tie"):
            return None
        return {"winner": winner, "confidence": confidence, "reasoning": reasoning}
    except Exception:
        return None


def estimate_cost(model, p, c):
    pr = PRICING.get(model, PRICING["openai/gpt-4o-mini"])
    return p / 1e6 * pr["prompt"] + c / 1e6 * pr["completion"]


async def judge_one(client, model, persona, slot, candidates, recs_a, recs_b, sem, conf, stats):
    cand_lines = [
        f"  - id={t.id[:8]}: {t.title} (зал {t.hall})\n    {t.abstract[:150]}"
        for t in candidates
    ]
    candidates_text = "\n".join(cand_lines)

    def render_recs(rec_ids, prefix):
        lines = []
        for i, tid in enumerate(rec_ids):
            t = conf.talks.get(tid)
            if t:
                lines.append(f"  {prefix}.{i+1}. id={tid[:8]} — «{t.title}» (зал {t.hall})")
        return "\n".join(lines) if lines else "(пусто)"

    user_msg = USER_TEMPLATE.format(
        persona=persona[:1000],
        slot_time=slot.datetime,
        candidates=candidates_text,
        recs_a=render_recs(recs_a, "A"),
        recs_b=render_recs(recs_b, "B"),
    )

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as e:
            stats["errors"] += 1
            return None
        usage = resp.usage
        cost = estimate_cost(
            model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        stats["cost"] += cost
        msg = resp.choices[0].message.content or ""
        parsed = parse_judgement(msg)
        if parsed:
            parsed["cost"] = cost
        else:
            stats["parse_fails"] += 1
        return parsed


def bradley_terry(wins_matrix, n_iter=200):
    """Calculate Bradley-Terry ranking from wins matrix W[i,j]=wins of i over j."""
    n = wins_matrix.shape[0]
    ratings = np.ones(n)
    for _ in range(n_iter):
        new_ratings = np.zeros(n)
        for i in range(n):
            num = np.sum(wins_matrix[i])
            denom = 0
            for j in range(n):
                if j != i:
                    total_ij = wins_matrix[i, j] + wins_matrix[j, i]
                    if total_ij > 0:
                        denom += total_ij / (ratings[i] + ratings[j])
            if denom > 0:
                new_ratings[i] = num / denom
            else:
                new_ratings[i] = ratings[i]
        # Normalize
        if new_ratings.sum() > 0:
            new_ratings = new_ratings / new_ratings.sum() * n
        if np.allclose(ratings, new_ratings, rtol=1e-4):
            break
        ratings = new_ratings
    return ratings


async def main_async(args):
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas_full = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}

    # Subsample personas
    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(personas_full), size=args.n_personas, replace=False)
    chosen = [personas_full[int(i)] for i in chosen_idx]

    # Subsample slots (multi-talk only)
    multi_slots = [s for s in conf.slots if len(s.talk_ids) >= 2]
    if args.n_slots < len(multi_slots):
        slot_idx = rng.choice(len(multi_slots), size=args.n_slots, replace=False)
        slots_to_judge = [multi_slots[int(i)] for i in slot_idx]
    else:
        slots_to_judge = multi_slots

    print(f"Judges: {args.n_personas} personas × {len(slots_to_judge)} slots")

    # Set up policies
    relevance_fn = None
    if args.relevance == "learned":
        model_path = ROOT / "data" / "models" / "preference_model.pkl"
        relevance_fn = LearnedPreferenceFn(model_path)
        # precompute
        persona_dict = {p["id"]: by_id[p["id"]] for p in chosen}
        talk_dict = {tid: t.embedding for tid, t in conf.talks.items()}
        relevance_fn.precompute_all(persona_dict, talk_dict)

    llm_ranker = None
    if "LLM-ranker" in args.policies.split(","):
        llm_ranker = LLMRankerPolicy(model="openai/gpt-4o-mini", budget_usd=2.0)

    all_policies = build_policies(seed=42, llm_ranker=llm_ranker)
    policy_names = args.policies.split(",")
    policies = {k: all_policies[k] for k in policy_names if k in all_policies}
    print(f"Policies: {list(policies.keys())}")

    # Generate REALISTIC hall loads via running a quick Cosine simulation
    # (creates overflow scenarios where capacity-aware matters)
    print("Pre-running Cosine sim to capture realistic hall states...")
    from src.simulator import simulate, SimConfig
    full_users = [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in personas_full[:300]  # use full population for realistic congestion
    ]
    cfg = SimConfig(K=args.K, tau=0.3, lambda_overflow=2.0, p_skip_base=0.05, seed=42)
    cosine_sim = simulate(conf, full_users, CosinePolicy(), cfg, relevance_fn=relevance_fn)
    # Build hall_load at "midpoint" of each slot — half of users have arrived
    realistic_state_per_slot = {}
    for slot in slots_to_judge:
        mid_load = {}
        for h in conf.halls.values():
            final = cosine_sim.hall_load_per_slot.get(slot.id, {}).get(h.id, 0)
            # Half-state — give the policy something to react to
            mid_load[(slot.id, h.id)] = final // 2
        realistic_state_per_slot[slot.id] = mid_load
    print(f"  realistic states captured for {len(realistic_state_per_slot)} slots")

    # Generate recommendations from each policy for each (persona, slot)
    print("Generating recommendations from each policy...")
    recommendations = {}  # (policy_name, persona_id, slot_id) -> list of talk_ids
    for p_name, p_fn in policies.items():
        for persona in chosen:
            user = UserProfile(id=persona["id"], text=text_of(persona), embedding=by_id[persona["id"]])
            for slot in slots_to_judge:
                state = {
                    "hall_load": realistic_state_per_slot[slot.id],
                    "slot_id": slot.id,
                    "K": args.K,
                    "relevance_fn": relevance_fn,
                }
                recs = p_fn(user=user, slot=slot, conf=conf, state=state)
                recommendations[(p_name, persona["id"], slot.id)] = recs
    print(f"Generated {len(recommendations)} recommendations")

    # All pairs of policies
    pairs = list(combinations(policy_names, 2))
    print(f"Pairs: {len(pairs)}")

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"cost": 0.0, "errors": 0, "parse_fails": 0}

    # Pairwise judgments
    judgments = []  # list of {a, b, persona, slot, winner, confidence, reasoning, cost}
    estimate = len(pairs) * args.n_personas * len(slots_to_judge)
    print(f"Estimated calls: {estimate}, cost: ~${estimate * 0.012:.2f} (Sonnet)")

    async def judge_pair(policy_a, policy_b, persona, slot):
        recs_a = recommendations[(policy_a, persona["id"], slot.id)]
        recs_b = recommendations[(policy_b, persona["id"], slot.id)]
        candidates = [conf.talks[tid] for tid in slot.talk_ids]
        result = await judge_one(
            client, args.model,
            persona=text_of(persona),
            slot=slot,
            candidates=candidates,
            recs_a=recs_a, recs_b=recs_b,
            sem=sem, conf=conf, stats=stats,
        )
        if result:
            result.update({
                "policy_a": policy_a,
                "policy_b": policy_b,
                "persona_id": persona["id"],
                "slot_id": slot.id,
            })
            judgments.append(result)

    coros = []
    for a, b in pairs:
        for persona in chosen:
            for slot in slots_to_judge:
                coros.append(judge_pair(a, b, persona, slot))

    BATCH = 100
    t0 = time.time()
    for i in range(0, len(coros), BATCH):
        batch = coros[i : i + BATCH]
        await asyncio.gather(*batch)
        # Save snapshot
        out_path = ROOT / "results" / f"llm_judge_{args.suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "n_personas": args.n_personas,
                    "n_slots": len(slots_to_judge),
                    "policies": policy_names,
                    "K": args.K,
                    "relevance": args.relevance,
                },
                "judgments": judgments,
                "cost_so_far": stats["cost"],
            }, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t0
        print(f"  done {min(i+BATCH, len(coros))}/{len(coros)} | "
              f"cost ${stats['cost']:.3f} | "
              f"errors {stats['errors']} | parse_fails {stats['parse_fails']} | {elapsed:.0f}s")

    # Bradley-Terry analysis
    n_p = len(policy_names)
    name_to_idx = {n: i for i, n in enumerate(policy_names)}
    wins = np.zeros((n_p, n_p))
    for j in judgments:
        ai = name_to_idx[j["policy_a"]]
        bi = name_to_idx[j["policy_b"]]
        w = j["winner"]
        c = j.get("confidence", 0.5)
        if w == "A":
            wins[ai, bi] += c
        elif w == "B":
            wins[bi, ai] += c
        else:
            wins[ai, bi] += 0.5 * c
            wins[bi, ai] += 0.5 * c

    ratings = bradley_terry(wins)
    print("\n=== Bradley-Terry rating ===")
    rated = sorted(zip(policy_names, ratings), key=lambda x: -x[1])
    for name, r in rated:
        print(f"  {name:<22} BT={r:.3f}")

    out_path = ROOT / "results" / f"llm_judge_{args.suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": args.model,
                "n_personas": args.n_personas,
                "n_slots": len(slots_to_judge),
                "policies": policy_names,
                "K": args.K,
                "relevance": args.relevance,
            },
            "judgments": judgments,
            "wins_matrix": wins.tolist(),
            "bradley_terry": dict(zip(policy_names, ratings.tolist())),
            "cost_total": stats["cost"],
            "errors": stats["errors"],
            "parse_fails": stats["parse_fails"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-personas", type=int, default=20)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--policies", default="Random,Cosine,MMR,Capacity-aware,Capacity-aware MMR,LLM-ranker")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--relevance", choices=["cosine", "learned"], default="learned")
    p.add_argument("--suffix", default="default")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
