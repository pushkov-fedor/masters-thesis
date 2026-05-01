"""Генерация Demo Day-персон через LLM.

Demo Day ITMO 2026 — студенческая конференция по AI/ML/NLP/EdTech.
Целевая аудитория: ML/NLP-инженеры, AI-исследователи, EdTech-разработчики,
студенты-аспиранты в data science, computer vision-инженеры, разработчики
рекомендательных систем.

Запуск:
    .venv/bin/python scripts/generate_demoday_personas.py --n 300 --batch 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]
OUT_JSON = ROOT / "data" / "personas" / "personas_demoday.json"
OUT_NPZ = ROOT / "data" / "personas" / "personas_demoday_embeddings.npz"


SYSTEM_PROMPT = """Ты создаёшь реалистичные профили участников Demo Day ITMO — студенческой конференции по AI/ML.

Программа конференции включает доклады по следующим направлениям:
NLP (28% всех докладов), Autonomous agents (LLM-агенты), EdTech (образовательные ИИ-сервисы), Computer Vision, Recsys, Industrial ML, LLM, Fintech, A/B testing, MLOps, генеративные модели.

Аудитория Demo Day: студенты магистратуры/аспирантуры ITMO в data science / AI, ML-инженеры из индустрии (Yandex, Sber, Tinkoff, VK, X5, mid/large компании), NLP-инженеры, computer vision-разработчики, devrel и AI-исследователи.

Задача: сгенерировать N разнообразных правдоподобных профилей.

Разнообразие:
- разные направления: NLP/LLM, CV, recsys, time series, RL, MLOps, EdTech-AI, fintech-ML
- разные роли: ML-инженер, NLP-инженер, data scientist, исследователь, тимлид, AI-product manager, аспирант, ML-консультант
- разные уровни: junior, middle, senior, lead/principal, аспирант, профессор
- разные стеки: PyTorch, JAX, transformers, LangChain, classical ML, RL frameworks
- разные интересы: efficient inference, RAG, agent frameworks, multimodal, adversarial robustness, ML-фундаментальные исследования, applied ML

Возвращай строго JSON массив объектов, без markdown, без комментариев."""


USER_PROMPT_TEMPLATE = """Сгенерируй {n} разнообразных профилей участников Demo Day ITMO.

Формат каждого профиля:
{{
  "id": "u_001",
  "role": "NLP middle-инженер",
  "experience": "middle",
  "company_size": "midsize",
  "interests": ["RAG", "LLM-агенты", "production deployment"],
  "preferred_topics": ["NLP", "LLM", "Autonomous agents"],
  "background": "Подробный профиль 2-4 предложения: стек, опыт, текущие задачи, интересы. На русском."
}}

Возможные категории докладов (используй в preferred_topics): NLP, LLM, Autonomous agents, EdTech, CV, Computer Vision, Recsys, Industrial ML, Fintech, MLOps, Time Series, Generative AI, Multimodal, RL.

Уровни experience: junior, middle, senior, lead/principal, postgrad, professor.
Размеры company_size: startup, midsize, large, enterprise, university, research-lab.

Начинай нумерацию id с u_{start_id:03d}.
Возвращай только JSON-массив, без обёрток."""


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY not found")


def parse_json_array(text: str):
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


async def gen_batch(client, model, n, start_id, sem):
    user = USER_PROMPT_TEMPLATE.format(n=n, start_id=start_id)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.85, max_tokens=4000, timeout=120,
            )
        except Exception as e:
            print(f"  error: {e}")
            return []
    msg = resp.choices[0].message.content or ""
    arr = parse_json_array(msg)
    if not arr:
        return []
    return arr


async def main_async(args):
    client = AsyncOpenAI(api_key=load_api_key(), base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.concurrency)

    n_batches = (args.n + args.batch - 1) // args.batch
    print(f"Generating {args.n} personas in {n_batches} batches of {args.batch} via {args.model}")

    tasks = []
    for b in range(n_batches):
        n_this = min(args.batch, args.n - b * args.batch)
        start_id = b * args.batch + 1
        tasks.append(gen_batch(client, args.model, n_this, start_id, sem))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    print(f"Generated {sum(len(r) for r in results)} personas in {time.time()-t0:.1f}s")

    all_personas = []
    next_id = 1
    seen_ids = set()
    for batch in results:
        for p in batch:
            # Re-id, чтобы избежать коллизий между батчами
            new_id = f"u_{next_id:03d}"
            while new_id in seen_ids:
                next_id += 1
                new_id = f"u_{next_id:03d}"
            p["id"] = new_id
            seen_ids.add(new_id)
            next_id += 1
            all_personas.append(p)

    # Trim to exact count
    all_personas = all_personas[: args.n]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
    print(f"WROTE: {OUT_JSON} ({len(all_personas)} personas)")

    # Embeddings
    from src.embedder import embed_texts
    texts = [p.get("background", "") for p in all_personas]
    print(f"Embedding {len(texts)} personas as 'query'...")
    emb = embed_texts(texts, kind="query")
    np.savez(OUT_NPZ, ids=np.array([p["id"] for p in all_personas]), embeddings=emb)
    print(f"WROTE: {OUT_NPZ} (shape={emb.shape})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
