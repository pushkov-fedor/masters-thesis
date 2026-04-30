"""Параллельно прогоняет LLM-ranker для всех (user, slot) пар и сохраняет в кэш.

После этого основной эксперимент с --with-llm читает только кэш — без API.
Это даёт детерминированность (один запрос на пару) и в 30+ раз быстрее
последовательного вызова.

Использование:
  python scripts/precompute_llm_rankings.py --concurrency 30 --budget 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference  # noqa: E402

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]
CACHE_PATH = ROOT / "logs" / "llm_ranker_cache.json"
USAGE_LOG = ROOT / "logs" / "openrouter_usage.jsonl"

SYSTEM_PROMPT = """Ты ранжируешь доклады IT-конференции под конкретного пользователя.
Тебе даётся профиль участника и список из 2-4 кандидатов (доклады в одном тайм-слоте).
Верни строго JSON-массив id докладов в порядке убывания приоритета — самый релевантный первым.

Без объяснений, без markdown, без комментариев. Только массив строк."""

USER_TEMPLATE = """Профиль:
{profile}

Кандидаты:
{candidates}

Верни JSON-массив всех id в порядке убывания релевантности для этого участника. Например: ["id_a", "id_c", "id_b"]"""


PRICING = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
}


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY не найден")


def cache_key(user_id: str, slot_id: str, candidate_ids: list) -> str:
    h = hashlib.sha1((user_id + "|" + slot_id + "|" + "|".join(sorted(candidate_ids))).encode("utf-8")).hexdigest()[:20]
    return h


def parse_array(text: str) -> list:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("[")
    j = text.rfind("]")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(text[i : j + 1])
    except Exception:
        return None


async def rank_one(client, model, user, slot, conf, sem, stats):
    cand_ids = list(slot.talk_ids)
    if len(cand_ids) <= 1:
        return slot.id, user["id"], cand_ids, 0, 0
    async with sem:
        candidates_text = "\n".join(
            f"- id={tid}\n  title: {conf.talks[tid].title}\n  category: {conf.talks[tid].category}\n  abstract: {conf.talks[tid].abstract[:400]}"
            for tid in cand_ids
        )
        user_msg = USER_TEMPLATE.format(profile=user["text"], candidates=candidates_text)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=120,
            )
        except Exception as e:
            stats["errors"] += 1
            return slot.id, user["id"], None, 0, 0

        usage = resp.usage
        msg = resp.choices[0].message.content or ""
        arr = parse_array(msg)
        if arr is None:
            stats["parse_fails"] += 1
            return slot.id, user["id"], None, usage.prompt_tokens or 0, usage.completion_tokens or 0
        valid = [tid for tid in arr if tid in cand_ids]
        for tid in cand_ids:
            if tid not in valid:
                valid.append(tid)
        return slot.id, user["id"], valid, usage.prompt_tokens or 0, usage.completion_tokens or 0


async def main_async(args):
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )

    with open(ROOT / "data" / "personas" / "personas.json", encoding="utf-8") as f:
        personas = json.load(f)

    def text_of(p):
        return p.get("background") or p.get("profile") or ""

    users = [{"id": p["id"], "text": text_of(p)} for p in personas]

    # Загрузить существующий кэш
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    print(f"Cache start size: {len(cache)}")

    # Сформировать список задач (только тех, что не в кэше)
    tasks_to_run = []
    for user in users:
        for slot in conf.slots:
            cand_ids = list(slot.talk_ids)
            if len(cand_ids) <= 1:
                continue
            key = cache_key(user["id"], slot.id, cand_ids)
            if key in cache:
                continue
            tasks_to_run.append((user, slot, key))
    print(f"Pending tasks: {len(tasks_to_run)} (skipped {300 * 16 - len(tasks_to_run)} via cache)")

    # Грубая оценка стоимости
    pricing = PRICING.get(args.model, {"prompt": 1.0, "completion": 5.0})
    est = len(tasks_to_run) * (1000 / 1e6 * pricing["prompt"] + 60 / 1e6 * pricing["completion"])
    print(f"Estimated cost: ${est:.4f} (model={args.model})")

    if est > args.budget:
        print(f"WARN: estimated cost ${est:.4f} > budget ${args.budget}")
        if not args.yes:
            print("Use --yes to proceed anyway")
            return

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"errors": 0, "parse_fails": 0}

    cumulative_cost = 0.0
    completed = 0
    last_save = time.time()

    async def run_with_log(user, slot, key):
        nonlocal cumulative_cost, completed
        slot_id, user_id, valid, p, c = await rank_one(client, args.model, user, slot, conf, sem, stats)
        if valid is not None:
            cache[key] = valid
            cost = p / 1e6 * pricing["prompt"] + c / 1e6 * pricing["completion"]
            cumulative_cost += cost
        completed += 1
        if completed % 100 == 0 or completed == len(tasks_to_run):
            print(f"  done {completed}/{len(tasks_to_run)}  cost=${cumulative_cost:.4f}  errors={stats['errors']}  parse_fails={stats['parse_fails']}", flush=True)

    t0 = time.time()
    coros = [run_with_log(u, s, k) for (u, s, k) in tasks_to_run]
    # запускаем партиями чтобы периодически сохраняться
    BATCH = 200
    for i in range(0, len(coros), BATCH):
        batch = coros[i : i + BATCH]
        await asyncio.gather(*batch)
        # сохраняем кэш
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  saved cache, size={len(cache)}, elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nDONE: {completed}/{len(tasks_to_run)} tasks, "
          f"cost=${cumulative_cost:.4f}, errors={stats['errors']}, parse_fails={stats['parse_fails']}, "
          f"elapsed={time.time()-t0:.0f}s")
    print(f"Cache size: {len(cache)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--budget", type=float, default=3.0)
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
