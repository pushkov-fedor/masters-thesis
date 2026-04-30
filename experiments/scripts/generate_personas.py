"""Генерирует синтетических персон через OpenRouter API.

Требование: OPENROUTER_API_KEY в .env проекта party-of-one (читаем из ../party-of-one/.env).

Структура персоны:
{
  "id": "u_001",
  "role": "iOS-разработчик",
  "experience": "senior",
  "company_size": "large",
  "interests": ["SwiftUI", "performance"],
  "preferred_topics": ["Architecture", "Under the Hood"],
  "background": "Свободный текст профиля для эмбеддинга..."
}

Лог токенов и стоимости: logs/openrouter_usage.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
from dotenv import dotenv_values
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
# Сначала пробуем .env проекта masters-degree (свежий ключ),
# затем fallback в ../party-of-one/.env
ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]
LOG_PATH = ROOT / "logs" / "openrouter_usage.jsonl"
OUT_PATH = ROOT / "data" / "personas" / "personas.json"

MODEL = "anthropic/claude-haiku-4.5"  # дешёвая и качественная модель для генерации
BATCH_SIZE = 20

SYSTEM_PROMPT = """Ты создаёшь реалистичные профили участников IT-конференции Mobius (мобильная разработка).
Это многодневная конференция в Санкт-Петербурге со сложной программой по iOS, Android, KMP, кросс-платформе, бэкенду для мобилок, UX/UI и трендам индустрии.

Аудитория Mobius: разработчики разного уровня, тимлиды, архитекторы, отдельные QA, продакты, иногда CTO/руководители.

Твоя задача: сгенерировать N разных правдоподобных профилей для симуляции рекомендаций программы.

Важно: разнообразие. Не все одинаковые senior iOS. Должны встречаться:
- разные стеки: iOS (Swift/SwiftUI/UIKit), Android (Kotlin/Compose), KMP, Flutter, React Native
- разные роли: разработчик, тимлид, архитектор, QA-engineer, devrel, продакт
- разные уровни: junior, middle, senior, lead/principal
- разные размеры компаний: стартап, средний бизнес, корпорация
- разные интересы: производительность, архитектура, UX, безопасность, новые языки/runtime, биохакинг, тренды

Возвращай строго JSON массив объектов, без комментариев и без markdown."""

USER_PROMPT_TEMPLATE = """Сгенерируй {n} разнообразных профилей участников Mobius. Возвращай строго JSON-массив:

[
  {{
    "id": "u_001",
    "role": "iOS senior разработчик",
    "experience": "senior",
    "company_size": "large",
    "interests": ["SwiftUI", "производительность"],
    "preferred_topics": ["Architecture", "Under the Hood"],
    "background": "Подробный профиль 2-4 предложения, описывающий стек, опыт, текущие задачи и интересы. На русском."
  }},
  ...
]

Категории докладов на конференции (используй для preferred_topics): Architecture, Under the Hood, Trends, UX/UI in Mobile Development, Product Quality, Infrastructure, Biohacking, General.

Уровни experience: junior, middle, senior, lead/principal.
Размеры company_size: startup, midsize, large, enterprise.

Дай N={n} разных персон, последовательно нумеруй id от u_{start:03d} до u_{end:03d}."""


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                print(f"Using key from {env}")
                return key
    raise SystemExit(f"OPENROUTER_API_KEY не найден ни в одном из: {ENV_CANDIDATES}")


def log_usage(record: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Стоимость для anthropic/claude-haiku-4.5 на OpenRouter (в $ за 1M токенов)
PRICING = {
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"prompt": 1.0, "completion": 5.0})
    return prompt_tokens / 1e6 * p["prompt"] + completion_tokens / 1e6 * p["completion"]


def parse_json_array(text: str) -> list:
    """Парсит JSON-массив из ответа модели; терпим к окружающему тексту."""
    text = text.strip()
    # удалить markdown-обёртки
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    # найти первую '[' и последнюю ']'
    i = text.find("[")
    j = text.rfind("]")
    if i == -1 or j == -1:
        raise ValueError(f"Не нашёл JSON-массив в ответе: {text[:200]}")
    return json.loads(text[i : j + 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=300)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--budget", type=float, default=2.0, help="Soft cap, USD")
    args = p.parse_args()

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_personas: List[dict] = []
    cumulative_cost = 0.0

    n_batches = (args.total + args.batch - 1) // args.batch
    for b in range(n_batches):
        start = b * args.batch + 1
        end = min((b + 1) * args.batch, args.total)
        n = end - start + 1
        prompt = USER_PROMPT_TEMPLATE.format(n=n, start=start, end=end)

        if cumulative_cost >= args.budget:
            print(f"BUDGET STOP: cost ${cumulative_cost:.4f} >= ${args.budget}")
            break

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=8000,
            )
        except Exception as e:
            print(f"API ERROR batch {b}: {e}")
            log_usage({
                "ts": time.time(), "batch": b, "error": str(e),
            })
            continue

        elapsed = time.time() - t0
        msg = resp.choices[0].message.content or ""
        usage = resp.usage
        cost = estimate_cost(
            args.model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        cumulative_cost += cost
        log_usage({
            "ts": time.time(), "batch": b, "model": args.model,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "cost_usd": cost, "cumulative_cost_usd": cumulative_cost,
            "elapsed_s": elapsed,
        })
        try:
            arr = parse_json_array(msg)
        except Exception as e:
            print(f"PARSE ERROR batch {b}: {e}")
            print("RAW:", msg[:500])
            continue

        # перенумеруем id если модель не следует инструкции
        for i, persona in enumerate(arr):
            persona["id"] = f"u_{start + i:03d}"
        all_personas.extend(arr)

        print(f"Batch {b+1}/{n_batches}: +{len(arr)} personas | "
              f"cost=${cost:.4f} | total=${cumulative_cost:.4f} | "
              f"prompt_tok={usage.prompt_tokens if usage else '?'} "
              f"completion_tok={usage.completion_tokens if usage else '?'} | "
              f"{elapsed:.1f}s")

        # Сохраняем после каждого батча — устойчивость к сбоям
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_personas, f, ensure_ascii=False, indent=2)

    print(f"\nDONE: {len(all_personas)} personas, total cost ${cumulative_cost:.4f}")
    # Эмбеддинги профилей
    print("Computing embeddings...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = [p["background"] for p in all_personas]
    emb = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    np.savez(
        ROOT / "data" / "personas" / "personas_embeddings.npz",
        ids=np.array([p["id"] for p in all_personas]),
        embeddings=emb.astype(np.float32),
    )
    print(f"Embeddings: {emb.shape} -> personas_embeddings.npz")


if __name__ == "__main__":
    main()
