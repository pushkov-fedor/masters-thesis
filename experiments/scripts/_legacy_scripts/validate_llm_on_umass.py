"""B2'-валидация LLM-агента на UMass CICS Course Allocation Fall 2024.

В отличие от Meetup (где B2=0.434≈random из-за membership-leakage в B1 и тонких
профилей), UMass даёт **прямые preference rankings** (rank 1-7) без утечки.

Логика:
- Из UMass-данных строим тест-пары (student, slot, top-ranked-course):
  - slot имеет ≥2 параллельных курса РАЗНЫХ catalog-номеров
  - студент проставил rank > 0 хотя бы на 2 этих курса
  - ground truth = курс с максимальным rank в слоте (если 2+ tied → берём первый)
- Для каждой пары: LLM получает профиль студента + историю его предпочтений на
  ДРУГИХ курсах (исключая курсы из текущего слота — leakage prevention) +
  список параллельных опций в слоте → предсказывает выбор.
- Метрика: accuracy@1 (LLM-выбор == курс с max rank).

Это ЧЕСТНАЯ B2-валидация LLM-симулятора:
- Real preferences (rank 1-7) — нет leakage через group memberships
- Hold-out current slot — LLM должна обобщать с других курсов
- Reasonable random baseline (1/n_options)

Запуск:
    .venv/bin/python scripts/validate_llm_on_umass.py --max-pairs 500 --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
    "openai/gpt-5.4-mini": (0.10, 0.40),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.41),
    "openai/gpt-4o-mini": (0.15, 0.60),
}


def build_test_pairs():
    """Возвращает (pairs, talk_meta, user_ranks).

    pairs: list of {user_id, slot_id, options: list[talk_id], ranks: list[float],
                    chosen_real: talk_id (max rank)}
    talk_meta: {talk_id: {title, category, course_key, ...}}
    user_ranks: {user_id: {course_key: rank_normalized}}
    """
    conf = json.load(open(ROOT / "data" / "conferences" / "umass_cics.json"))
    talks = {t["id"]: t for t in conf["talks"]}
    # course_key = catalog номер (без секции)
    talk_to_catalog = {}
    for t in conf["talks"]:
        # talk_id = "302-01", catalog = "302"
        cat = t["id"].split("-")[0]
        talk_to_catalog[t["id"]] = cat
        t["course_key"] = cat

    # Загружаем embeddings (нормированные rank-векторы) и users meta
    emb_data = np.load(ROOT / "data" / "personas" / "umass_cics_users_embeddings.npz")
    users_meta = json.load(open(ROOT / "data" / "personas" / "umass_cics_users.json"))
    user_ids = emb_data["ids"].tolist()
    user_embs = emb_data["embeddings"]  # (n_users, 65)

    # Восстановим course_idx из отсортированного списка catalog'ов в talks
    course_keys_sorted = sorted({talk_to_catalog[t] for t in talks.keys()})
    course_idx = {c: i for i, c in enumerate(course_keys_sorted)}
    # ВНИМАНИЕ: эмбеддинги были построены в load_umass_cics на отсортированном
    # списке common_courses из survey ∩ courses; здесь мы получаем тот же список.
    n_courses = len(course_keys_sorted)
    if user_embs.shape[1] != n_courses:
        print(f"WARNING: emb_dim={user_embs.shape[1]} vs n_courses={n_courses}")

    # user_id -> dict[course_key -> rank_score (0..1)]
    user_ranks = {}
    for uid, emb in zip(user_ids, user_embs):
        # эмбеддинг был L2-нормирован при загрузке, но пропорция между rank'ами
        # сохраняется; для теста важна не абсолютная величина, а порядок.
        ranks_on_courses = {}
        for ck, ci in course_idx.items():
            if ci < len(emb) and emb[ci] > 0:
                ranks_on_courses[ck] = float(emb[ci])
        if ranks_on_courses:
            user_ranks[uid] = ranks_on_courses

    print(f"Users with rankings: {len(user_ranks)}")
    print(f"Total courses: {n_courses}")

    # Сборка тест-пар
    pairs = []
    slots_dict = defaultdict(list)
    for tid, t in talks.items():
        slots_dict[t["slot_id"]].append(tid)

    for sid, slot_talks in slots_dict.items():
        # уникальные catalog'и в слоте
        cats_in_slot = list({talk_to_catalog[tid] for tid in slot_talks})
        if len(cats_in_slot) < 2:
            continue  # параллельных курсов разных catalog нет
        # для каждого студента: какие из этих catalog'ов ранжированы?
        for uid, ranks in user_ranks.items():
            ranked_in_slot = {ck: ranks[ck] for ck in cats_in_slot if ck in ranks}
            if len(ranked_in_slot) < 2:
                continue
            # ground truth catalog с макс рангом
            top_cat = max(ranked_in_slot.items(), key=lambda x: x[1])[0]
            # представляем talk-options как «catalog+section»; берём по одной
            # секции каждого catalog'а (первую) — упрощаем выбор
            cat_to_first_tid = {}
            for tid in slot_talks:
                ck = talk_to_catalog[tid]
                if ck not in cat_to_first_tid:
                    cat_to_first_tid[ck] = tid
            options_tids = [cat_to_first_tid[ck] for ck in cats_in_slot
                             if ck in cat_to_first_tid]
            if len(options_tids) < 2:
                continue
            chosen_real_tid = cat_to_first_tid[top_cat]
            pairs.append({
                "user_id": uid,
                "slot_id": sid,
                "options": options_tids,
                "chosen_real": chosen_real_tid,
                "n_options": len(options_tids),
                "user_ranks_in_slot": {ck: ranked_in_slot[ck] for ck in cats_in_slot if ck in ranked_in_slot},
            })

    return pairs, talks, user_ranks, talk_to_catalog


def random_baseline(pairs):
    return sum(1.0 / p["n_options"] for p in pairs) / max(1, len(pairs))


async def main_async(args):
    pairs, talks, user_ranks, talk_to_catalog = build_test_pairs()
    print(f"Total test pairs: {len(pairs)}")
    rb = random_baseline(pairs)
    print(f"Random baseline: {rb:.4f}")

    if args.max_pairs and args.max_pairs < len(pairs):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
        print(f"Subsampled to {len(pairs)} pairs")
        rb = random_baseline(pairs)
        print(f"Random baseline (subsample): {rb:.4f}")

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

    state = {"correct": 0, "answered": 0, "skipped": 0, "cost": 0.0}
    pbar = tqdm(total=len(pairs), desc="B2'-UMass", ncols=120)

    async def process_pair(pair):
        uid = pair["user_id"]
        sid = pair["slot_id"]
        options_tids = pair["options"]
        chosen_real = pair["chosen_real"]

        # История = top предпочтения юзера на ДРУГИХ courses (не текущий слот)
        cats_in_slot = {talk_to_catalog[tid] for tid in options_tids}
        all_user_ranks = user_ranks.get(uid, {})
        # отсортируем catalog'и по убыванию rank, исключив те, что в текущем слоте
        history_cats = sorted(
            ((ck, r) for ck, r in all_user_ranks.items() if ck not in cats_in_slot),
            key=lambda x: x[1], reverse=True
        )[:args.max_history]
        history = [
            {
                "slot_id": "past",
                "talk_id": ck,
                "title": f"Course {ck}",
                "category": ck,
            } for ck, _ in history_cats
        ]

        profile_text = (f"Student at UMass CICS, ranked {len(all_user_ranks)} courses; "
                        f"top preferences indicated by ranks 1-7 (interest level).")
        agent = LLMAgent(
            agent_id=str(uid),
            profile=profile_text,
            history=history,
        )

        # Опции для решения
        slot_talks = []
        for tid in options_tids:
            t = talks[tid]
            slot_talks.append({
                "id": tid,
                "title": t["title"],
                "hall": t.get("hall", 0),
                "abstract": t.get("abstract", ""),
                "category": t.get("category", ""),
            })

        decision = await agent.decide(
            slot_id=sid, talks=slot_talks, hall_loads_pct={},
            recommendation=None, llm_call=llm_call,
        )

        return {
            "user_id": uid,
            "slot_id": sid,
            "chosen_real": chosen_real,
            "chosen_pred": decision.chosen,
            "reason": decision.reason,
            "n_options": pair["n_options"],
            "cost": decision.cost_usd,
        }

    async def _run(p):
        r = await process_pair(p)
        pbar.update(1)
        if r is None:
            return r
        state["cost"] += r["cost"]
        if r["chosen_pred"] is None:
            state["skipped"] += 1
        else:
            state["answered"] += 1
            if r["chosen_pred"] == r["chosen_real"]:
                state["correct"] += 1
        if state["answered"]:
            pbar.set_postfix({
                "acc": f"{state['correct']/state['answered']:.3f}",
                "ans": state["answered"],
                "skip": state["skipped"],
                "cost": f"${state['cost']:.2f}",
            })
        return r

    t0 = time.time()
    results = [r for r in await asyncio.gather(*(_run(p) for p in pairs)) if r is not None]
    pbar.close()

    elapsed = time.time() - t0
    n_correct = state["correct"]
    n_answered = state["answered"]
    n_skipped = state["skipped"]
    accuracy = n_correct / max(1, n_answered)
    accuracy_inc_skip = n_correct / max(1, len(results))

    summary = {
        "config": {
            "model": args.model,
            "n_pairs": len(pairs),
            "n_results": len(results),
            "concurrency": args.concurrency,
            "max_history": args.max_history,
        },
        "metrics": {
            "B2_prime_accuracy_at_1": accuracy,
            "B2_prime_accuracy_inc_skip": accuracy_inc_skip,
            "random_baseline": rb,
            "n_correct": n_correct,
            "n_answered": n_answered,
            "n_skipped": n_skipped,
            "total_cost_usd": state["cost"],
            "elapsed_s": elapsed,
        },
        "details": results[:200],
    }
    out_path = ROOT / "results" / f"llm_umass_b2_{args.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n=== B2'-UMass done in {elapsed:.0f}s ===")
    print(f"Pairs: {len(results)} (of {len(pairs)} requested)")
    print(f"Answered: {n_answered}, skipped: {n_skipped}")
    print(f"Random baseline: {rb:.4f}")
    print(f"B2' accuracy@1 (excl skip): {accuracy:.4f}")
    print(f"B2' accuracy@1 (incl skip as wrong): {accuracy_inc_skip:.4f}")
    print(f"Lift over random: {accuracy - rb:+.4f}")
    print(f"Total cost: ${state['cost']:.3f}")
    print(f"WROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 = all available; >0 = subsample for cost control")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--max-history", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
