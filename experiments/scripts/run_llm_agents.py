"""Прогон LLM-агентского симулятора на Mobius.

Задача: показать, что без политики залы переполняются, с простой политикой —
выравниваются. Главный график защиты «было/стало».

Архитектура:
- Один LLM-агент на участника, простая память + текущая загрузка в промпте.
- Агенты в слоте обрабатываются ПОСЛЕДОВАТЕЛЬНО (видят накопленную загрузку).
- Capacity = ceil(N_agents / halls_in_slot) — гарантирует, что переполнения
  возможны при концентрации спроса.
- Политики: NoPolicy (рекомендация=None), Cosine (top-K по cos), Capacity-aware MMR.

Запуск:
    .venv/bin/python scripts/run_llm_agents.py --n 50 --policies no_policy,cosine,cap_aware_mmr \\
        --model openai/gpt-5.4-mini --K 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_agent import LLMAgent

ENV_CANDIDATES = [ROOT.parent / ".env", ROOT.parent.parent / "party-of-one" / ".env"]


def load_api_key():
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            k = cfg.get("OPENROUTER_API_KEY")
            if k:
                return k
    raise SystemExit("OPENROUTER_API_KEY not found")


PRICING = {
    "openai/gpt-5.4-mini": (0.10, 0.40),  # $/Mtok in/out approx
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.41),
    "openai/gpt-4o-mini": (0.15, 0.60),
}


# === Политики ===

def policy_no_policy(user_emb, talk_embs, talk_ids, hall_loads, K):
    """Без рекомендации — агент видит весь слот, политики нет."""
    return None  # no recommendation, agent sees all options


def policy_cosine(user_emb, talk_embs, talk_ids, hall_loads, K):
    """Top-K по cosine."""
    sims = talk_embs @ user_emb
    top = np.argsort(sims)[::-1][:K]
    return [talk_ids[i] for i in top]


def policy_cosine_capacity_filter(user_emb, talk_embs, talk_ids, hall_loads, K,
                                   hall_of_talk=None):
    """Top-K по cosine с маской переполненных залов.

    Контрольная политика: добавляет к cosine только capacity-логику (без
    diversity и без load_penalty). Позволяет отделить вклад capacity-маскирования
    от MMR-разнообразия в эффекте cap_aware_mmr.
    """
    sims = talk_embs @ user_emb
    valid = []
    for i, tid in enumerate(talk_ids):
        hall = hall_of_talk[tid] if hall_of_talk else 0
        load = hall_loads.get(hall, 0.0)
        if load < 1.0:
            valid.append((i, sims[i]))
    valid.sort(key=lambda x: x[1], reverse=True)
    return [talk_ids[i] for i, _ in valid[:K]]


def policy_cap_aware_mmr(user_emb, talk_embs, talk_ids, hall_loads, K,
                         lam=0.5, tau=0.85, hall_of_talk=None):
    """Capacity-aware MMR: cos - load_penalty + diversity, исключаем переполненные."""
    sims = talk_embs @ user_emb
    n = len(talk_ids)
    selected = []
    selected_idx = []
    for _ in range(K):
        best_score = -np.inf
        best_idx = -1
        for i in range(n):
            if i in selected_idx:
                continue
            tid = talk_ids[i]
            hall = hall_of_talk[tid] if hall_of_talk else 0
            load = hall_loads.get(hall, 0.0)
            if load >= 1.0:
                continue  # маскируем переполненные
            penalty = max(0.0, load - tau)
            if selected_idx:
                div = np.max(talk_embs[selected_idx] @ talk_embs[i])
            else:
                div = 0.0
            score = sims[i] - lam * penalty - 0.3 * div
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            break
        selected_idx.append(best_idx)
        selected.append(talk_ids[best_idx])
    return selected


POLICIES = {
    "no_policy": policy_no_policy,
    "cosine": policy_cosine,
    "cosine_capacity_filter": policy_cosine_capacity_filter,
    "cap_aware_mmr": policy_cap_aware_mmr,
}


# === Симулятор ===


def of_count_running(slot_loads, cap_per_hall, slot_id):
    """Сколько залов уже сейчас перегружены в этом слоте."""
    return sum(1 for h, c in slot_loads[slot_id].items()
               if c > cap_per_hall.get(h, 1000))


async def run_one_policy(policy_name, agents, conf, talk_emb_map, hall_caps,
                         K, llm_call, slot_concurrency=10, tqdm_position=0):
    """Прогон политики на копии агентов. Возвращает list[decision_record], stats."""
    # Глубокая копия истории — каждая политика на своих агентах с пустой историей.
    fresh_agents = [LLMAgent(agent_id=a.agent_id, profile=a.profile, history=[])
                    for a in agents]
    agent_by_id = {a.agent_id: a for a in fresh_agents}

    policy_fn = POLICIES[policy_name]
    talks_meta_by_id = {t["id"]: t for t in conf["talks"]}
    hall_of_talk = {t["id"]: t["hall"] for t in conf["talks"]}

    decisions = []
    # Per-(slot, hall) -> count
    slot_loads = defaultdict(lambda: defaultdict(int))
    skips = 0
    total_decisions = 0
    rng = np.random.default_rng(42)
    cost = 0.0

    pbar = tqdm(conf["slots"], desc=f"[{policy_name:<22}]",
                ncols=130, leave=True, dynamic_ncols=False,
                position=tqdm_position)
    for slot in pbar:
        slot_id = slot["id"]
        slot_talk_ids = [t["id"] for t in conf["talks"] if t["slot_id"] == slot_id]
        if not slot_talk_ids:
            continue
        slot_talks = [talks_meta_by_id[tid] for tid in slot_talk_ids]
        slot_halls = sorted({t["hall"] for t in slot_talks})
        cap = slot.get("hall_capacities", {})
        # capacity per hall in this slot
        cap_per_hall = {h: int(cap.get(str(h), cap.get(h, 1000))) for h in slot_halls}

        # Агенты идут БАТЧАМИ (размер = slot_concurrency). Внутри батча — параллельно
        # (политика видит общую загрузку до старта батча). Между батчами — обновляем
        # slot_loads. Это ускоряет прогон в ~slot_concurrency раз и реалистичнее
        # (несколько участников «приходят одновременно»), сохраняя capacity-логику
        # политики на крупном масштабе.
        order = list(range(len(fresh_agents)))
        rng.shuffle(order)

        slot_talk_embs = np.stack([talk_emb_map[tid] for tid in slot_talk_ids])

        batch_size = slot_concurrency
        for b_start in range(0, len(order), batch_size):
            batch = order[b_start:b_start + batch_size]
            cur_loads_pct = {
                h: slot_loads[slot_id][h] / cap_per_hall[h] for h in slot_halls
            }

            async def _one(ai):
                agent = fresh_agents[ai]
                user_emb = agent_emb_map[agent.agent_id]
                if policy_name in ("cap_aware_mmr", "cosine_capacity_filter"):
                    rec = policy_fn(user_emb, slot_talk_embs, slot_talk_ids,
                                    cur_loads_pct, K, hall_of_talk=hall_of_talk)
                else:
                    rec = policy_fn(user_emb, slot_talk_embs, slot_talk_ids,
                                    cur_loads_pct, K)
                return await agent.decide(
                    slot_id=slot_id,
                    talks=slot_talks,
                    hall_loads_pct=cur_loads_pct,
                    recommendation=rec,
                    llm_call=llm_call,
                )

            batch_decisions = await asyncio.gather(*(_one(ai) for ai in batch))
            for ai, decision in zip(batch, batch_decisions):
                decisions.append(decision)
                cost += decision.cost_usd
                if decision.chosen is None:
                    skips += 1
                else:
                    hall = hall_of_talk[decision.chosen]
                    slot_loads[slot_id][hall] += 1
                    fresh_agents[ai].commit(slot_id, talks_meta_by_id[decision.chosen])
                total_decisions += 1
        of_running = of_count_running(slot_loads, cap_per_hall, slot_id)
        pbar.set_postfix({
            "cost": f"${cost:.2f}",
            "skip": f"{skips}/{total_decisions}",
            "load_max": f"{max(slot_loads[slot_id].values(), default=0)}/{max(cap_per_hall.values(), default=0)}",
        })

    # Метрики:
    # OF_choice = доля выборов, когда зал был переполнен в момент входа агента.
    of_count = 0
    of_total = 0
    for d in decisions:
        if d.chosen is None:
            continue
        of_total += 1
        # был ли зал переполнен? Проверяем по slot_loads[slot]: финальный count
        # больше cap? Это не «в момент входа», а «по итогам слота» — для OF_all.
        hall = hall_of_talk[d.chosen]
        cap = next(s.get("hall_capacities", {}).get(str(hall),
                  s.get("hall_capacities", {}).get(hall, 1000))
                  for s in conf["slots"] if s["id"] == d.slot_id)
        if slot_loads[d.slot_id][hall] > cap:
            of_count += 1

    # hall_var: variance of utilization across halls per slot, averaged
    hall_vars = []
    excesses = []
    for slot in conf["slots"]:
        sid = slot["id"]
        if not slot_loads[sid]:
            continue
        cap = slot.get("hall_capacities", {})
        halls = sorted({t["hall"] for t in conf["talks"] if t["slot_id"] == sid})
        if len(halls) < 2:
            continue
        utils = [slot_loads[sid][h] / int(cap.get(str(h), cap.get(h, 1000))) for h in halls]
        hall_vars.append(float(np.var(utils)))
        excesses.append(float(np.mean([max(0, u - 1.0) for u in utils])))

    return {
        "decisions": [d.__dict__ for d in decisions],
        "slot_loads": {k: dict(v) for k, v in slot_loads.items()},
        "metrics": {
            "n_decisions": total_decisions,
            "skip_rate": skips / max(1, total_decisions),
            "OF_choice": of_count / max(1, of_total),
            "hall_var_mean": float(np.mean(hall_vars)) if hall_vars else 0.0,
            "mean_overload_excess": float(np.mean(excesses)) if excesses else 0.0,
            "cost_usd": cost,
        },
    }


async def main_async(args):
    print("Loading conference...", flush=True)
    conf_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    emb_path = ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz"
    conf = json.load(open(conf_path))
    emb = np.load(emb_path)
    talk_emb_map = {tid: emb["embeddings"][i] for i, tid in enumerate(emb["ids"].tolist())}

    # Capacity = ceil(N / halls_in_slot) — переписываем динамически.
    print(f"Setting capacity = ceil({args.n} / halls_in_slot) for each slot...", flush=True)
    talks_by_slot = defaultdict(list)
    for t in conf["talks"]:
        talks_by_slot[t["slot_id"]].append(t)
    for s in conf["slots"]:
        halls = sorted({t["hall"] for t in talks_by_slot[s["id"]]})
        cap = math.ceil(args.n / max(1, len(halls)))
        s["hall_capacities"] = {str(h): cap for h in halls}
    print(f"  e.g. slot[0] {conf['slots'][0]['id']} caps: {conf['slots'][0]['hall_capacities']}",
          flush=True)

    # Personas
    pers_path = ROOT / "data" / "personas" / f"{args.personas}.json"
    pers_emb_path = ROOT / "data" / "personas" / f"{args.personas}_embeddings.npz"
    pers = json.load(open(pers_path))
    p_npz = np.load(pers_emb_path)
    p_emb_map = {pid: p_npz["embeddings"][i] for i, pid in enumerate(p_npz["ids"].tolist())}
    pers = pers[:args.n]
    print(f"Loaded {len(pers)} personas", flush=True)

    global agent_emb_map
    agent_emb_map = {p["id"]: p_emb_map[p["id"]] for p in pers}

    agents = [LLMAgent(agent_id=p["id"], profile=p.get("background", ""))
              for p in pers]

    # LLM client
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    p_in, p_out = PRICING.get(args.model, (1.0, 5.0))

    sem = asyncio.Semaphore(args.concurrency)

    async def llm_call(system, user, max_tokens=200):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.3, max_tokens=max_tokens, timeout=60,
                )
            except Exception as e:
                return f'{{"choice":"skip","reason":"api-error: {e}"}}', 0.0
        msg = resp.choices[0].message.content or ""
        u = resp.usage
        cost = (u.prompt_tokens / 1e6) * p_in + (u.completion_tokens / 1e6) * p_out
        return msg, cost

    # Политики идут ПАРАЛЛЕЛЬНО (каждая на своих копиях агентов).
    # Внутри политики агенты SEQUENTIAL в слоте — capacity-логика политики
    # видит реальную накопительную загрузку.
    policies = [p for p in args.policies.split(",") if p in POLICIES]
    print(f"\n=== Running {len(policies)} policies in parallel: {policies} ===", flush=True)
    t0_global = time.time()

    async def run_with_label(pol, pos):
        t0 = time.time()
        res = await run_one_policy(pol, agents, conf, talk_emb_map, None,
                                   args.K, llm_call,
                                   slot_concurrency=args.batch_size,
                                   tqdm_position=pos)
        return pol, res, time.time() - t0

    pairs = await asyncio.gather(*(run_with_label(p, i) for i, p in enumerate(policies)))
    all_results = {}
    for pol, res, elapsed in pairs:
        all_results[pol] = res
        print(f"\n[{pol}] done in {elapsed:.0f}s. metrics: {res['metrics']}", flush=True)

    out = {
        "config": {
            "conference": args.conference, "personas": args.personas,
            "n_agents": args.n, "K": args.K, "model": args.model,
            "policies": policies, "seed": 42,
        },
        "elapsed_total_s": time.time() - t0_global,
        "results": all_results,
    }
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = ROOT / "results" / f"llm_agents_{args.conference}_n{args.n}{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {out_path}")

    # Сводка
    print("\n=== Сводка ===")
    print(f"{'policy':<20} {'OF_choice':<12} {'hall_var':<12} {'excess':<12} {'skip':<8} {'cost':<8}")
    for pol, r in all_results.items():
        m = r["metrics"]
        print(f"{pol:<20} {m['OF_choice']:<12.3f} {m['hall_var_mean']:<12.4f} "
              f"{m['mean_overload_excess']:<12.4f} {m['skip_rate']:<8.2f} ${m['cost_usd']:<6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn")
    ap.add_argument("--personas", default="personas_100")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--policies", default="no_policy,cosine,cap_aware_mmr")
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="общий семафор LLM API (in-flight calls limit)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="агентов в слоте параллельно. 1 = строгий sequential (корректная capacity-логика). "
                         "Больше = быстрее, но политика не видит обновления загрузки внутри батча.")
    ap.add_argument("--suffix", default="", help="суффикс к имени выходного файла")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
