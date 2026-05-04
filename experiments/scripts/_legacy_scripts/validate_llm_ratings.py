"""Проверка качества LLM-разметки релевантности.

Считает две корреляции на одном наборе пар (профиль × доклад):
1. Test-retest: gpt-5.4-mini, два независимых прогона. Стабильна ли модель сама с собой?
2. Inter-model: gpt-5.4-mini vs claude-haiku-4.5. Согласны ли две независимые модели?

Запуск:
    .venv/bin/python scripts/validate_llm_ratings.py --conference mobius_2025_autumn --n-pairs 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_preferences import (
    CONFERENCE_PRESETS,
    SYSTEM_PROMPT_TEMPLATE_CONTINUOUS,
    USER_TEMPLATE_CONTINUOUS,
    parse_score,
    load_api_key,
)


async def rate_one(client, model, persona, talk, system_prompt, sem, pbar):
    user_msg = USER_TEMPLATE_CONTINUOUS.format(
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
                temperature=0.0, max_tokens=600, timeout=90,
            )
        except Exception as e:
            pbar.update(1)
            return None
    pbar.update(1)
    msg = resp.choices[0].message.content or ""
    parsed = parse_score(msg, scale="100")
    if parsed is None:
        return None
    return parsed[0]


async def run_pass(label, model, pairs, system_prompt, concurrency):
    client = AsyncOpenAI(api_key=load_api_key(), base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(pairs), desc=label, file=sys.stdout, mininterval=1.0)
    t0 = time.time()
    tasks = [rate_one(client, model, p, t, system_prompt, sem, pbar) for p, t in pairs]
    scores = await asyncio.gather(*tasks)
    pbar.close()
    elapsed = time.time() - t0
    valid = sum(1 for s in scores if s is not None)
    print(f"  {label}: {valid}/{len(pairs)} valid in {elapsed:.1f}s")
    return scores


def report_correlation(label, a, b):
    paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not paired:
        print(f"  {label}: нет валидных пар")
        return None
    xs = np.array([p[0] for p in paired])
    ys = np.array([p[1] for p in paired])
    rho_s, p_s = spearmanr(xs, ys)
    rho_p, p_p = pearsonr(xs, ys)
    mae = float(np.mean(np.abs(xs - ys)))
    print(f"  {label}:  n={len(paired)}")
    print(f"    Spearman ρ = {rho_s:.3f} (p={p_s:.2e})")
    print(f"    Pearson  r = {rho_p:.3f} (p={p_p:.2e})")
    print(f"    MAE        = {mae:.3f}  (диапазон [0, 1])")
    return {"n": len(paired), "spearman": rho_s, "pearson": rho_p, "mae": mae}


async def main_async(args):
    rng = np.random.default_rng(args.seed)
    preset = CONFERENCE_PRESETS[args.conference]
    prog_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    personas_path = ROOT / "data" / "personas" / f"{preset['personas']}.json"
    system_prompt = SYSTEM_PROMPT_TEMPLATE_CONTINUOUS.format(domain=preset["domain_phrase"])

    with open(prog_path, encoding="utf-8") as f:
        prog = json.load(f)
    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f)
    talks = prog["talks"]

    # Sample pairs uniformly.
    pids = [p["id"] for p in personas]
    tids = [t["id"] for t in talks]
    p_by_id = {p["id"]: p for p in personas}
    t_by_id = {t["id"]: t for t in talks}
    sampled = set()
    pairs = []
    while len(pairs) < args.n_pairs:
        pid = pids[rng.integers(0, len(pids))]
        tid = tids[rng.integers(0, len(tids))]
        key = (pid, tid)
        if key in sampled:
            continue
        sampled.add(key)
        pairs.append((p_by_id[pid], t_by_id[tid]))

    print(f"Conference: {args.conference}, sampled pairs: {len(pairs)}")
    print()

    print("=== Pass A: gpt-5.4-mini, run #1 ===")
    a = await run_pass("gpt-5.4-mini #1", "openai/gpt-5.4-mini",
                       pairs, system_prompt, args.concurrency)
    print()
    print("=== Pass B: gpt-5.4-mini, run #2 ===")
    b = await run_pass("gpt-5.4-mini #2", "openai/gpt-5.4-mini",
                       pairs, system_prompt, args.concurrency)
    print()
    print("=== Pass C: claude-haiku-4.5 ===")
    c = await run_pass("claude-haiku-4.5", "anthropic/claude-haiku-4.5",
                       pairs, system_prompt, args.concurrency)
    print()

    print("=== Результаты ===")
    print()
    res = {
        "conference": args.conference,
        "n_pairs": len(pairs),
    }
    print("Test-retest (gpt-5.4-mini #1 vs #2):")
    res["test_retest"] = report_correlation("test-retest", a, b)
    print()
    print("Inter-model (gpt-5.4-mini #1 vs claude-haiku-4.5):")
    res["inter_model_a_vs_c"] = report_correlation("inter-model A↔C", a, c)
    print()
    print("Inter-model (gpt-5.4-mini #2 vs claude-haiku-4.5):")
    res["inter_model_b_vs_c"] = report_correlation("inter-model B↔C", b, c)

    a_arr = [s for s in a if s is not None]
    b_arr = [s for s in b if s is not None]
    c_arr = [s for s in c if s is not None]
    print()
    print("Распределения:")
    for label, arr in [("gpt-5.4-mini #1", a_arr), ("gpt-5.4-mini #2", b_arr),
                       ("claude-haiku-4.5", c_arr)]:
        if arr:
            print(f"  {label}: n={len(arr)} mean={np.mean(arr):.3f} "
                  f"std={np.std(arr):.3f} distinct={len(set(arr))}")

    out_path = ROOT / "results" / f"validate_llm_ratings_{args.conference}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res["raw"] = {
        "pairs": [(p["id"], t["id"]) for p, t in pairs],
        "scores_a": a, "scores_b": b, "scores_c": c,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn",
                    choices=list(CONFERENCE_PRESETS.keys()))
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
