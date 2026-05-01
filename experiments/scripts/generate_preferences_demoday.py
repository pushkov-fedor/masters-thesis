"""Генерация LLM-оценок интереса для пар (Demo Day-персона, Demo Day-доклад).

Сэмплирует N случайных пар, оценивает через gpt-4o-mini, сохраняет в
data/preferences_matrix_demoday.json. По дефолту 10 000 пар (~$3-5).

Запуск:
    .venv/bin/python scripts/generate_preferences_demoday.py --n-pairs 10000
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]

PROG_PATH = ROOT / "data" / "conferences" / "demo_day_2026.json"
PERSONAS_PATH = ROOT / "data" / "personas" / "personas_demoday.json"
OUT_PATH = ROOT / "data" / "preferences_matrix_demoday.json"

SYSTEM_PROMPT = """Ты оцениваешь, насколько доклад на студенческой AI/ML-конференции Demo Day ITMO интересен конкретному участнику.

На вход — профиль участника и описание доклада (название, категория, аннотация).
Возвращай число от 0 до 1:
- 0.0 — точно неинтересно (не его область, не его уровень, не его темы)
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


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY not found")


def parse_score(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        d = json.loads(text[i : j + 1])
        s = float(d.get("score", -1))
        reason = str(d.get("reason", ""))
        if 0.0 <= s <= 1.0:
            return s, reason
    except Exception:
        pass
    return None


async def rate_pair(client, model, persona, talk, sem, pbar):
    user_msg = USER_TEMPLATE.format(
        persona=persona.get("background", "")[:600],
        title=talk["title"],
        category=talk.get("category", "Other"),
        abstract=talk.get("abstract", "")[:500],
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0, max_tokens=80, timeout=60,
            )
        except Exception:
            pbar.update(1)
            return None
    pbar.update(1)
    msg = resp.choices[0].message.content or ""
    parsed = parse_score(msg)
    if parsed is None:
        return None
    score, reason = parsed
    return {
        "persona_id": persona["id"],
        "talk_id": talk["id"],
        "score": score,
        "reason": reason,
    }


async def main_async(args):
    rng = np.random.default_rng(args.seed)

    with open(PROG_PATH, encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]
    talks_by_id = {t["id"]: t for t in talks}

    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas = json.load(f)
    personas_by_id = {p["id"]: p for p in personas}

    persona_ids = list(personas_by_id.keys())
    talk_ids = list(talks_by_id.keys())
    print(f"Personas: {len(personas)}, talks: {len(talks)}, "
          f"max pairs: {len(personas)*len(talks)}")

    # Resume from existing
    existing = {}
    if OUT_PATH.exists() and not args.fresh:
        with open(OUT_PATH, encoding="utf-8") as f:
            for r in json.load(f):
                existing[f"{r['persona_id']}|{r['talk_id']}"] = r
    print(f"Existing entries: {len(existing)}")

    # Sample pairs uniformly. Ensure each persona gets ≥ N/n_personas pairs
    # for group-split balance later.
    target_per_persona = max(1, args.n_pairs // len(personas))
    pairs = []
    for pid in persona_ids:
        sampled_talks = rng.choice(len(talk_ids), size=min(target_per_persona, len(talk_ids)),
                                    replace=False)
        for ti in sampled_talks:
            tid = talk_ids[ti]
            key = f"{pid}|{tid}"
            if key not in existing:
                pairs.append((personas_by_id[pid], talks_by_id[tid]))
    # Top up to n_pairs with extra random pairs if needed
    while len(pairs) + len(existing) < args.n_pairs:
        pid = persona_ids[rng.integers(0, len(persona_ids))]
        tid = talk_ids[rng.integers(0, len(talk_ids))]
        key = f"{pid}|{tid}"
        if key not in existing and not any(p[0]["id"] == pid and p[1]["id"] == tid for p in pairs):
            pairs.append((personas_by_id[pid], talks_by_id[tid]))

    print(f"To rate: {len(pairs)} new pairs (target total {args.n_pairs})")

    pricing = {"openai/gpt-4o-mini": (0.15, 0.6)}
    p_cost, c_cost = pricing.get(args.model, (1.0, 5.0))
    est = len(pairs) * (600 / 1e6 * p_cost + 60 / 1e6 * c_cost)
    print(f"Estimated cost: ${est:.3f}")

    client = AsyncOpenAI(api_key=load_api_key(), base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(pairs), desc="LLM rate", file=sys.stdout, mininterval=1.0)

    t0 = time.time()
    tasks = [rate_pair(client, args.model, p, t, sem, pbar) for p, t in pairs]
    results_new = await asyncio.gather(*tasks)
    pbar.close()
    valid = [r for r in results_new if r is not None]
    print(f"\nDone in {time.time()-t0:.1f}s. Valid: {len(valid)}/{len(pairs)}")

    # Merge with existing and save
    all_results = list(existing.values()) + valid
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"WROTE: {OUT_PATH} ({len(all_results)} total pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=10000)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="ignore existing file")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
