"""Перевод программы Demo Day 2026 с RU на EN через OpenRouter (claude-haiku-4.5).

Сохраняет структуру JSON исходника побитно: переводятся только поля
``title``, ``abstract``, ``category``. Остальные поля (``id``, ``hall``,
``slot_id``, ``date``, ``start_time``, ``end_time``, ``speakers``)
сохраняются без изменений.

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/translate_demoday.py --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [ROOT.parent / ".env", ROOT.parent.parent / "party-of-one" / ".env"]
SRC_PATH = ROOT / "data" / "conferences" / "demo_day_2026.json"
OUT_PATH = ROOT / "data" / "conferences" / "demo_day_2026_en.json"
PARTIAL_PATH = ROOT / "data" / "conferences" / "demo_day_2026_en.partial.jsonl"

# Pricing per Mtok: (prompt, completion). Источник — run_llm_lhs_subset.py.
PRICING = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "openai/gpt-4.1-mini": (0.40, 1.60),
}

SYSTEM_PROMPT = (
    "You are a professional translator specialising in IT and AI/ML technical "
    "content. Translate Russian conference talk metadata to natural, fluent "
    "English. Preserve technical terms, library names, framework names, "
    "company names, and product names exactly as in the source (English "
    "loanwords stay English; do not transliterate). Preserve numbers, "
    "version markers, and acronyms (LLM, RAG, NLP, ML, CV, ASR, OCR, "
    "MLOps, DevOps) verbatim.\n\n"
    "INPUT JSON: {\"title\": \"<ru title>\", \"abstract\": \"<ru abstract>\", "
    "\"category\": \"<ru/en category>\"}.\n\n"
    "OUTPUT JSON ONLY (no commentary, no markdown, no code fences): "
    "{\"title\": \"<en title>\", \"abstract\": \"<en abstract>\", "
    "\"category\": \"<en category>\"}.\n\n"
    "Rules:\n"
    "1. Translate literally and faithfully. Do not summarise, do not embellish, "
    "do not add details that are not in the source.\n"
    "2. If the field is already in English (some categories are mixed RU/EN), "
    "keep it as is.\n"
    "3. For category, normalise to a comma-separated tag list using "
    "established English IT vocabulary (NLP, ML, LLM, RAG, Recsys, Agents, "
    "Autonomous agents, EdTech, Fintech, Industrial ML, MLOps, ASR, OCR, "
    "Computer Vision, CV, Search, RecSys, Personalization, Analytics, "
    "Security, etc.).\n"
    "4. Keep proper nouns and acronyms in their original form.\n"
    "5. The output must be valid JSON parseable by json.loads."
)


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            k = cfg.get("OPENROUTER_API_KEY")
            if k:
                return k
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = [ln for ln in lines if not ln.startswith("```")]
        s = "\n".join(lines).strip()
    return s


async def translate_one(client, model, talk: dict, sem, pricing) -> tuple[dict, float]:
    payload = {
        "title": talk["title"],
        "abstract": talk["abstract"],
        "category": talk.get("category", ""),
    }
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        temperature=0.2,
                        max_tokens=1500,
                        timeout=60,
                    ),
                    timeout=90.0,
                )
                msg = resp.choices[0].message.content or ""
                u = resp.usage
                cost = 0.0
                if u is not None:
                    p_in, p_out = pricing
                    cost = (u.prompt_tokens / 1e6) * p_in + (u.completion_tokens / 1e6) * p_out
                txt = _strip_fences(msg)
                # try to recover JSON if the model added prefix
                lo, hi = txt.find("{"), txt.rfind("}")
                if lo == -1 or hi == -1:
                    raise ValueError(f"no JSON in response: {msg[:200]}")
                obj = json.loads(txt[lo:hi + 1])
                # Sanity: required keys
                for k in ("title", "abstract", "category"):
                    if k not in obj:
                        raise ValueError(f"missing key {k}: {msg[:200]}")
                return obj, cost
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED talk {talk['id']} after 3 attempts: {e}")
                    return {"title": "[TRANSLATE-FAILED] " + talk["title"],
                            "abstract": talk["abstract"],
                            "category": talk.get("category", "")}, 0.0
                await asyncio.sleep(1.5 * (attempt + 1))


async def main_async(args):
    with open(SRC_PATH, encoding="utf-8") as f:
        src = json.load(f)
    talks = src["talks"]
    print(f"Loaded {SRC_PATH.name}: {len(talks)} talks")

    # Resume support: read partial JSONL
    done = {}
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec["translated"]
                except Exception:
                    continue
        print(f"  resumed: {len(done)}/{len(talks)} already translated")

    api_key = load_api_key()
    pricing = PRICING.get(args.model, (1.0, 5.0))
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)

    todo_talks = [t for t in talks if t["id"] not in done]
    print(f"  to translate: {len(todo_talks)}")
    total_cost = 0.0
    t0 = time.perf_counter()

    pbar = tqdm(total=len(todo_talks), desc="translate", smoothing=0.1)
    fpartial = open(PARTIAL_PATH, "a", encoding="utf-8")

    async def worker(talk):
        nonlocal total_cost
        obj, cost = await translate_one(client, args.model, talk, sem, pricing)
        total_cost += cost
        done[talk["id"]] = obj
        fpartial.write(json.dumps({"id": talk["id"], "translated": obj}, ensure_ascii=False) + "\n")
        fpartial.flush()
        pbar.update(1)
        pbar.set_postfix({"cost": f"${total_cost:.4f}"})

    await asyncio.gather(*[worker(t) for t in todo_talks])
    pbar.close()
    fpartial.close()
    print(f"  cost: ${total_cost:.4f}, wall: {time.perf_counter() - t0:.1f}s")

    # Build output preserving original structure
    out = {k: v for k, v in src.items() if k != "talks"}
    out["conf_id"] = "demo_day_2026_en"
    out["name"] = "Demo Day ITMO 2026 (English)"
    out_talks = []
    for t in talks:
        tr = done[t["id"]]
        out_talks.append({
            "id": t["id"],
            "title": tr["title"],
            "speakers": t.get("speakers", ""),  # preserve speakers as is
            "hall": t["hall"],
            "date": t["date"],
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            "category": tr.get("category", t.get("category", "")),
            "abstract": tr["abstract"],
            "slot_id": t["slot_id"],
        })
    out["talks"] = out_talks
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"WROTE {OUT_PATH}  ({len(out_talks)} talks)")
    return total_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
