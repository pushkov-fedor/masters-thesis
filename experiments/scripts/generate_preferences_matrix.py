"""Стадия 2: асинхронная LLM-генерация preferences matrix 300×40.

Для каждой пары (persona, talk) запрашиваем gpt-4o-mini оценить
preference ∈ [0, 1] — насколько доклад интересен этой персоне.

Используется как ground truth для обучения параметрической модели выбора
(заменяет cosine как relevance signal в основном симуляторе).

Стоимость: 12000 пар × ~600 prompt + ~30 completion токенов
         = 7.2M prompt + 0.36M completion
         × $0.15/M + $0.6/M ≈ $1.30
Время: ~10-15 мин с concurrency=30.

Кэширование: ответы сохраняются после каждого батча в JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]

OUT_PATH = ROOT / "data" / "preferences_matrix.json"
USAGE_LOG = ROOT / "logs" / "openrouter_usage.jsonl"
PROG_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn.json"
PERSONAS_PATH = ROOT / "data" / "personas" / "personas.json"

SYSTEM_PROMPT = """Ты оцениваешь, насколько доклад на IT-конференции Mobius интересен конкретному участнику.

На вход — профиль участника и описание доклада (название, категория, аннотация).
Возвращай число от 0 до 1:
- 0.0 — точно неинтересно (не его стек, не его уровень, не его темы)
- 0.5 — нейтрально (мог бы зайти из любопытства, но без активного интереса)
- 1.0 — очень интересно (точно в его сфере, явно полезно)

Ответ строго в формате JSON: {"score": 0.XX, "reason": "одно короткое предложение почему"}"""

USER_TEMPLATE = """Профиль участника:
{persona}

Доклад:
- Название: {title}
- Категория: {category}
- Аннотация: {abstract}

Оцени интересность от 0 до 1."""

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
                print(f"Using key from {env}")
                return key
    raise SystemExit("OPENROUTER_API_KEY not found")


def parse_score(text: str) -> tuple[float, str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        d = json.loads(text[i : j + 1])
        score = float(d.get("score", -1))
        reason = str(d.get("reason", ""))
        if 0.0 <= score <= 1.0:
            return score, reason
    except Exception:
        pass
    return None


def estimate_cost(model: str, p: int, c: int) -> float:
    pr = PRICING.get(model, {"prompt": 1.0, "completion": 5.0})
    return p / 1e6 * pr["prompt"] + c / 1e6 * pr["completion"]


def log_usage(rec: dict):
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def rate_pair(client, model, persona, talk, sem, results, stats):
    key = f"{persona['id']}|{talk['id']}"
    if key in results:
        return
    async with sem:
        user_msg = USER_TEMPLATE.format(
            persona=persona.get("background", persona.get("profile", "")),
            title=talk["title"],
            category=talk.get("category", "Other"),
            abstract=talk.get("abstract", "")[:600],
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=80,
            )
        except Exception as e:
            stats["errors"] += 1
            return

        usage = resp.usage
        msg = resp.choices[0].message.content or ""
        parsed = parse_score(msg)
        cost = estimate_cost(
            model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        stats["cumulative_cost"] += cost

        if parsed is None:
            stats["parse_fails"] += 1
            return
        score, reason = parsed
        results[key] = {
            "persona_id": persona["id"],
            "talk_id": talk["id"],
            "score": score,
            "reason": reason,
        }
        stats["completed"] += 1


async def main_async(args):
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    with open(PROG_PATH, encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]

    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas = json.load(f)

    print(f"Personas: {len(personas)}, talks: {len(talks)}")
    print(f"Total pairs to rate: {len(personas) * len(talks)}")

    # Load existing results if present
    existing = {}
    if OUT_PATH.exists() and not args.fresh:
        with open(OUT_PATH, encoding="utf-8") as f:
            existing_list = json.load(f)
        for r in existing_list:
            k = f"{r['persona_id']}|{r['talk_id']}"
            existing[k] = r
    print(f"Existing entries (resume): {len(existing)}")

    # Build task list
    pairs_to_rate = []
    for persona in personas:
        for talk in talks:
            key = f"{persona['id']}|{talk['id']}"
            if key not in existing:
                pairs_to_rate.append((persona, talk))

    print(f"Pending: {len(pairs_to_rate)}")
    if not pairs_to_rate:
        print("All done.")
        return

    # Estimate cost
    pricing = PRICING[args.model]
    est_cost = len(pairs_to_rate) * (600 / 1e6 * pricing["prompt"] + 60 / 1e6 * pricing["completion"])
    print(f"Estimated cost: ${est_cost:.4f}")

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"errors": 0, "parse_fails": 0, "completed": 0, "cumulative_cost": 0.0}
    results = dict(existing)

    BATCH_SIZE = 200
    t0 = time.time()
    coros = [rate_pair(client, args.model, p, t, sem, results, stats) for (p, t) in pairs_to_rate]

    for i in range(0, len(coros), BATCH_SIZE):
        batch = coros[i : i + BATCH_SIZE]
        await asyncio.gather(*batch)
        # save snapshot
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t0
        rate_per_min = stats["completed"] / max(0.001, elapsed) * 60
        print(f"  done {min(i + BATCH_SIZE, len(coros))}/{len(coros)} | "
              f"cost ${stats['cumulative_cost']:.4f} | errors {stats['errors']} | "
              f"parse_fails {stats['parse_fails']} | {rate_per_min:.0f}/min | {elapsed:.0f}s")

    log_usage({
        "ts": time.time(),
        "kind": "preferences_matrix",
        "model": args.model,
        "completed": stats["completed"],
        "cost": stats["cumulative_cost"],
        "errors": stats["errors"],
        "parse_fails": stats["parse_fails"],
    })

    print(f"\nDONE: {stats['completed']} pairs, total cost ${stats['cumulative_cost']:.4f}, "
          f"elapsed {time.time()-t0:.0f}s")
    print(f"WROTE: {OUT_PATH} ({len(results)} entries)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--fresh", action="store_true", help="ignore existing cache")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
