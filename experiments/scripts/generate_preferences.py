"""Унифицированная генерация LLM-оценок интереса для пар (персона, доклад).

Поддерживает любую конференцию: --conference mobius_2025_autumn|demo_day_2026.
Соответствующий набор персон выбирается автоматически: mobius → personas,
demoday → personas_demoday. Сохраняет в data/preferences_matrix_<suffix>.json.

Запуск:
    .venv/bin/python scripts/generate_preferences.py --conference demo_day_2026 --n-pairs 10000
    .venv/bin/python scripts/generate_preferences.py --conference mobius_2025_autumn --n-pairs 12000
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

CONFERENCE_PRESETS = {
    "mobius_2025_autumn": {
        "personas": "personas",
        "out_suffix": "",  # → preferences_matrix.json (Mobius исторически)
        "domain_phrase": "IT-конференции Mobius",
    },
    "demo_day_2026": {
        "personas": "personas_demoday",
        "out_suffix": "_demoday",  # → preferences_matrix_demoday.json
        "domain_phrase": "студенческой AI/ML-конференции Demo Day ITMO",
    },
}


SYSTEM_PROMPT_TEMPLATE = """Ты оцениваешь, насколько доклад на {domain} интересен конкретному участнику.

На вход — профиль участника и описание доклада (название, категория, аннотация).
Возвращай число от 0 до 1:
- 0.0 — точно неинтересно (не его область, не его уровень, не его темы)
- 0.5 — нейтрально (мог бы зайти из любопытства, но без активного интереса)
- 1.0 — очень интересно (точно в его сфере, явно полезно)

Ответ строго в формате JSON: {{"score": 0.XX, "reason": "одно короткое предложение почему"}}"""

USER_TEMPLATE = """Профиль участника:
{persona}

Доклад:
- Название: {title}
- Категория: {category}
- Аннотация: {abstract}

Оцени интересность от 0 до 1."""

# Поведенческий промпт: оценка вероятности, что участник реально пойдёт на
# конкретный доклад при свободном выборе (не строгая тематическая близость).
# Учитывает: основной стек, любопытство к соседним темам, статус спикера,
# уровень аудитории. Шкала 0-100, CoT перед ответом, нормализация в [0, 1].
SYSTEM_PROMPT_TEMPLATE_CONTINUOUS = """Ты прогнозируешь поведение участника на {domain}: насколько вероятно, что конкретный участник реально пойдёт на конкретный доклад при свободном выборе.

Реальный участник конференции выбирает доклад не только по строгому совпадению технологий, но и по соседним темам, любопытству, известности спикера, общей пользе для уровня. iOS-разработчик пойдёт на доклад про Kotlin Multiplatform; senior-бэкендер — на доклад про архитектуру с фронтенд-уклоном; QA — на доклад про DevEx у разработчиков.

Оцени поведенческую релевантность целым числом 0–100:
- 0–10   — нет смысла идти: радикально другая область, формат или уровень.
- 11–25  — пойдёт случайно, если делать нечего; ни тема, ни спикер не цепляют.
- 26–45  — может зайти из любопытства; тема косвенно касается его работы.
- 46–60  — реально рассматривает: тема в смежной области, или уровень подходит, или спикер интересен.
- 61–75  — высокая вероятность: основная область или сильно смежная, явно полезный для роли формат.
- 76–90  — почти наверняка пойдёт: точное попадание по теме И уровню; если конкурирующих сильнее не будет, выберет этот.
- 91–100 — пропустить нельзя: ровно его профиль, ровно его уровень, и (или) звёздный спикер.

Учитывай:
1. Основной стек участника — главный, но не единственный фактор.
2. Любопытство к соседним темам реально (15-30% выборов в среднем по аудитории).
3. Уровень доклада и роль участника (junior не пойдёт на сложный архитектурный доклад; lead — пойдёт скорее).
4. Категория/формат: keynote и доклады лидеров индустрии собирают аудиторию шире своей темы.
5. НЕ занижай: реальное распределение посещаемости конференций имеет mean ≈ 0.5, не ≈ 0.3.

Используй ВЕСЬ диапазон 0-100. Не округляй к якорям 0/25/50/75/100 — различай близкие случаи (разница 67 и 73 содержательна).

Сначала рассуждай 1-2 фразы (тема vs стек, уровень vs роль, любопытство), затем итог.

Ответ строго в формате JSON: {{"reasoning": "1-2 фразы", "score": 0-100}}"""

USER_TEMPLATE_CONTINUOUS = """Профиль участника:
{persona}

Доклад:
- Название: {title}
- Категория: {category}
- Аннотация: {abstract}

С какой вероятностью участник реально выберет этот доклад при свободном выборе? 0-100."""


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise SystemExit("OPENROUTER_API_KEY not found")


def parse_score(text: str, scale: str = "unit"):
    """scale='unit' — score в [0,1]; scale='100' — score 0-100, нормализуется в [0,1]."""
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
        reason = str(d.get("reasoning", "") or d.get("reason", ""))
        if scale == "100":
            if 0 <= s <= 100:
                return s / 100.0, reason
        else:
            if 0.0 <= s <= 1.0:
                return s, reason
    except Exception:
        pass
    return None


async def rate_pair(client, model, persona, talk, system_prompt, user_template,
                    scale, max_tokens, sem, pbar):
    user_msg = user_template.format(
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0, max_tokens=max_tokens, timeout=60,
            )
        except Exception:
            pbar.update(1)
            return None
    pbar.update(1)
    msg = resp.choices[0].message.content or ""
    parsed = parse_score(msg, scale=scale)
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

    if args.conference not in CONFERENCE_PRESETS:
        raise SystemExit(f"Unknown conference: {args.conference}")
    preset = CONFERENCE_PRESETS[args.conference]
    prog_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    personas_path = ROOT / "data" / "personas" / f"{preset['personas']}.json"
    suffix = preset["out_suffix"]
    if args.prompt_style == "continuous":
        suffix = f"{suffix}_continuous"
        system_prompt = SYSTEM_PROMPT_TEMPLATE_CONTINUOUS.format(domain=preset["domain_phrase"])
        user_template = USER_TEMPLATE_CONTINUOUS
        scale = "100"
        max_tokens = 200
    else:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(domain=preset["domain_phrase"])
        user_template = USER_TEMPLATE
        scale = "unit"
        max_tokens = 80
    out_path = ROOT / "data" / f"preferences_matrix{suffix}.json"
    print(f"Conference: {args.conference}")
    print(f"  personas: {personas_path}")
    print(f"  output:   {out_path}")

    with open(prog_path, encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]
    talks_by_id = {t["id"]: t for t in talks}

    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)
    personas_by_id = {p["id"]: p for p in personas}

    persona_ids = list(personas_by_id.keys())
    talk_ids = list(talks_by_id.keys())
    print(f"Personas: {len(personas)}, talks: {len(talks)}, "
          f"max pairs: {len(personas)*len(talks)}")

    # Resume from existing
    existing = {}
    if out_path.exists() and not args.fresh:
        with open(out_path, encoding="utf-8") as f:
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
    tasks = [rate_pair(client, args.model, p, t, system_prompt, user_template,
                       scale, max_tokens, sem, pbar)
             for p, t in pairs]
    results_new = await asyncio.gather(*tasks)
    pbar.close()
    valid = [r for r in results_new if r is not None]
    print(f"\nDone in {time.time()-t0:.1f}s. Valid: {len(valid)}/{len(pairs)}")

    # Merge with existing and save
    all_results = list(existing.values()) + valid
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"WROTE: {out_path} ({len(all_results)} total pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="demo_day_2026",
                    choices=list(CONFERENCE_PRESETS.keys()))
    ap.add_argument("--n-pairs", type=int, default=10000)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="ignore existing file")
    ap.add_argument("--prompt-style", choices=["trichotomy", "continuous"],
                    default="trichotomy",
                    help="trichotomy — старый промпт с якорями 0/0.5/1; "
                         "continuous — шкала 0-100 с CoT-рассуждением")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
