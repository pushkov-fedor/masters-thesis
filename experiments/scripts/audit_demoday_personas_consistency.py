"""LLM-judge audit на internal consistency 150 EN-персон Demo Day.

Каждую персону прогоняем через `claude-haiku-4.5` с задачей проверить
непротиворечивость полей:
- соответствует ли заявленный уровень опыта (junior/middle/senior/lead)
  background-описанию (количество лет, ответственность);
- согласуется ли роль с уровнем (например, "Junior" не может быть
  одновременно "Tech Lead");
- соответствуют ли interests / preferred_topics роли и фоновому описанию.

Ответ judge — JSON: {"verdict": "consistent" | "inconsistent",
"issue": "<краткая причина или null>"}.

Целевой результат: ≥ 95% consistent.

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/audit_demoday_personas_consistency.py --concurrency 8
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
PERS_PATH = ROOT / "data" / "personas" / "personas_demoday_en.json"
OUT_PATH = ROOT / "data" / "personas" / "test_diversity" / "internal_consistency_demoday.json"
PARTIAL_PATH = ROOT / "data" / "personas" / "test_diversity" / "internal_consistency_demoday.partial.jsonl"

PRICING = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-5.4-nano": (0.20, 1.25),
}

SYSTEM_PROMPT = (
    "You are a strict reviewer auditing a synthetic conference attendee "
    "persona for internal consistency. Check whether the following hold:\n"
    "1. The stated experience level matches the background description "
    "(years on the job and responsibility scope).\n"
    "2. The role does not contradict the experience (e.g. a 'junior' "
    "should not also be described as a 'Tech Lead' or 'Principal').\n"
    "3. The interests and preferred_topics are coherent with the role "
    "and the background narrative — they should not introduce unrelated "
    "domains.\n"
    "4. The company_size is plausible given the background "
    "(e.g. 'large bank' fits large/enterprise but not smallish-startup).\n\n"
    "OUTPUT JSON ONLY (no commentary, no code fences): "
    "{\"verdict\": \"consistent\" | \"inconsistent\", "
    "\"issue\": \"<one short sentence or null>\"}.\n\n"
    "Be moderate. Only flag genuine contradictions. Minor stylistic "
    "imperfections do not count as inconsistencies."
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


async def audit_one(client, model, persona, sem, pricing):
    user_msg = json.dumps({
        "id": persona["id"],
        "role": persona["role"],
        "experience": persona["experience"],
        "company_size": persona["company_size"],
        "interests": persona["interests"],
        "preferred_topics": persona["preferred_topics"],
        "background": persona["background"],
    }, ensure_ascii=False)
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.1,
                        max_tokens=300,
                        timeout=45,
                    ),
                    timeout=60.0,
                )
                msg = resp.choices[0].message.content or ""
                u = resp.usage
                cost = 0.0
                if u is not None:
                    p_in, p_out = pricing
                    cost = (u.prompt_tokens / 1e6) * p_in + (u.completion_tokens / 1e6) * p_out
                txt = _strip_fences(msg)
                lo, hi = txt.find("{"), txt.rfind("}")
                if lo == -1:
                    raise ValueError(f"no JSON: {msg[:200]}")
                obj = json.loads(txt[lo:hi + 1])
                v = obj.get("verdict")
                if v not in ("consistent", "inconsistent"):
                    raise ValueError(f"bad verdict: {obj}")
                issue = obj.get("issue")
                if issue is not None and not isinstance(issue, str):
                    issue = str(issue)
                if isinstance(issue, str) and issue.strip().lower() in ("null", "none", ""):
                    issue = None
                return {
                    "persona_id": persona["id"],
                    "verdict": v,
                    "issue": issue,
                }, cost
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {persona['id']}: {e}")
                    return {
                        "persona_id": persona["id"],
                        "verdict": "audit-error",
                        "issue": str(e),
                    }, 0.0
                await asyncio.sleep(1.0 * (attempt + 1))


async def main_async(args):
    personas = json.loads(PERS_PATH.read_text())
    print(f"Loaded {PERS_PATH.name}: {len(personas)} personas")

    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["persona_id"]] = rec
                except Exception:
                    continue
        print(f"  resumed: {len(done)}/{len(personas)}")

    todo = [p for p in personas if p["id"] not in done]

    api_key = load_api_key()
    pricing = PRICING.get(args.model, (1.0, 5.0))
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    total_cost = 0.0
    t0 = time.perf_counter()

    pbar = tqdm(total=len(todo), desc="audit", smoothing=0.1)
    fpartial = open(PARTIAL_PATH, "a", encoding="utf-8")

    async def worker(persona):
        nonlocal total_cost
        rec, cost = await audit_one(client, args.model, persona, sem, pricing)
        total_cost += cost
        done[persona["id"]] = rec
        fpartial.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fpartial.flush()
        pbar.update(1)
        pbar.set_postfix({"cost": f"${total_cost:.4f}"})

    await asyncio.gather(*[worker(p) for p in todo])
    pbar.close()
    fpartial.close()
    print(f"  cost: ${total_cost:.4f}, wall: {time.perf_counter() - t0:.1f}s")

    # Compose ordered report
    records = [done[p["id"]] for p in personas]
    n_consistent = sum(1 for r in records if r["verdict"] == "consistent")
    n_inconsistent = sum(1 for r in records if r["verdict"] == "inconsistent")
    n_error = sum(1 for r in records if r["verdict"] == "audit-error")
    print(f"  consistent: {n_consistent}/{len(records)} ({n_consistent / len(records):.1%})")
    print(f"  inconsistent: {n_inconsistent}")
    print(f"  audit-error: {n_error}")

    inconsistencies = [r for r in records if r["verdict"] == "inconsistent"]
    report = {
        "n_total": len(records),
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "n_audit_error": n_error,
        "pct_consistent": n_consistent / len(records),
        "inconsistencies": inconsistencies,
        "records": records,
        "model": args.model,
        "cost_usd": total_cost,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"WROTE {OUT_PATH}")
    return total_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
