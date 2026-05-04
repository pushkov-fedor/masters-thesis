"""B2-валидация LLM-агента на реальных Meetup-RSVPs.

Аналог B1=0.778 (параметрический симулятор), но для LLM-агента.

Логика:
- Из meetup_rsvp_raw_choices.json строим пары (user, slot, chosen_talk_id), где
  у пользователя был ровно один RSVP=yes в этом слоте, и в слоте ≥2 talks.
- Для каждого user история = его RSVPs на ДРУГИХ слотах (исключаем текущий —
  чтобы не было утечки).
- LLM получает: профиль (город+история) + альтернативы в слоте → choice.
- Считаем accuracy@1 (LLM-выбор == реальный RSVP).

Запуск:
    .venv/bin/python scripts/validate_llm_on_meetup.py --max-pairs 3647 --model openai/gpt-5.4-mini --concurrency 20
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


def build_pairs(restrict_to_known_users=True):
    """Из raw_choices строим пары (user_id, slot_id, chosen_talk_id, alt_talk_ids).

    Если restrict_to_known_users=True, оставляем только тех, у кого есть профиль в
    meetup_users.json (467 пользователей) — для apples-to-apples сравнения с B1=0.778.
    """
    raw = json.load(open(ROOT / "data" / "conferences" / "meetup_rsvp_raw_choices.json"))
    talks_meta = json.load(open(ROOT / "data" / "conferences" / "meetup_rsvp.json"))
    talk_by_id = {t["id"]: t for t in talks_meta["talks"]}

    known_users = None
    if restrict_to_known_users:
        users = json.load(open(ROOT / "data" / "personas" / "meetup_users.json"))
        # ID в meetup_users.json типа "mu_30197"; в raw_choices yes_user_ids — int.
        known_users = {int(u["id"].replace("mu_", "")) for u in users}

    talks_in_slot = defaultdict(list)
    for r in raw:
        talks_in_slot[r["slot_id"]].append(r["talk_id"])

    user_slot_yes = defaultdict(lambda: defaultdict(set))
    for r in raw:
        sid = r["slot_id"]
        tid = r["talk_id"]
        for uid in r["yes_user_ids"]:
            if known_users is not None and uid not in known_users:
                continue
            user_slot_yes[uid][sid].add(tid)

    pairs = []
    for uid, by_slot in user_slot_yes.items():
        for sid, yes_set in by_slot.items():
            if len(yes_set) != 1:
                continue
            slot_talks = talks_in_slot[sid]
            if len(slot_talks) < 2:
                continue
            chosen = next(iter(yes_set))
            pairs.append({
                "user_id": uid,
                "slot_id": sid,
                "chosen_talk_id": chosen,
                "talk_ids": slot_talks,
            })
    return pairs, talk_by_id, user_slot_yes


def random_baseline(pairs):
    """Доля «правильно угадаем случайно» = mean(1 / n_options) по всем парам."""
    return sum(1.0 / len(p["talk_ids"]) for p in pairs) / max(1, len(pairs))


def build_user_profile(uid, user_slot_yes, talk_by_id, exclude_slot, max_history=10):
    """Профиль = краткая история RSVP'ов на ДРУГИХ слотах."""
    history = []
    for sid, yes_set in user_slot_yes.get(uid, {}).items():
        if sid == exclude_slot:
            continue  # чтобы не было leak
        for tid in yes_set:
            t = talk_by_id.get(tid)
            if not t:
                continue
            history.append({
                "slot_id": sid,
                "talk_id": tid,
                "title": t.get("title", ""),
                "category": t.get("category", ""),
            })
    # отсортировать по slot_id (хронологически), взять последние N
    history.sort(key=lambda h: h["slot_id"])
    history = history[-max_history:]
    return history


async def main_async(args):
    pairs, talk_by_id, user_slot_yes = build_pairs(
        restrict_to_known_users=not args.all_users)
    rb = random_baseline(pairs)
    print(f"Total candidate pairs: {len(pairs)} (random_baseline={rb:.4f})", flush=True)

    if args.max_pairs and args.max_pairs < len(pairs):
        # детерминированно: сортируем по user/slot и берём первые
        pairs = sorted(pairs, key=lambda p: (p["user_id"], p["slot_id"]))[:args.max_pairs]
        print(f"Subsampled to {len(pairs)} pairs", flush=True)

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

    total_cost = 0.0
    correct = 0
    skipped = 0
    answered = 0
    results = []

    async def process_pair(pair):
        uid = pair["user_id"]
        sid = pair["slot_id"]
        chosen_real = pair["chosen_talk_id"]

        # профиль: city из meetup_users если есть, плюс история
        profile_text = f"Meetup user {uid}"
        history = build_user_profile(uid, user_slot_yes, talk_by_id, exclude_slot=sid,
                                     max_history=args.max_history)

        agent = LLMAgent(
            agent_id=str(uid),
            profile=profile_text,
            history=history,
        )

        # talks в слоте
        slot_talks = []
        for tid in pair["talk_ids"]:
            t = talk_by_id.get(tid)
            if not t:
                continue
            slot_talks.append({
                "id": tid,
                "title": t.get("title", ""),
                "hall": t.get("hall", 0),
                "abstract": t.get("abstract", ""),
                "category": t.get("category", ""),
            })

        if len(slot_talks) < 2:
            return None  # ничего не делать

        decision = await agent.decide(
            slot_id=sid,
            talks=slot_talks,
            hall_loads_pct={},  # не передаётся в промпт
            recommendation=None,
            llm_call=llm_call,
        )
        return {
            "user_id": uid,
            "slot_id": sid,
            "chosen_real": chosen_real,
            "chosen_pred": decision.chosen,
            "reason": decision.reason,
            "n_options": len(slot_talks),
            "cost": decision.cost_usd,
        }

    t0 = time.time()
    pbar = tqdm(total=len(pairs), desc="B2 validation", ncols=120)

    async def _run(p):
        r = await process_pair(p)
        pbar.update(1)
        if r is None:
            return
        results.append(r)
        nonlocal_state["cost"] += r["cost"]
        if r["chosen_pred"] is None:
            nonlocal_state["skipped"] += 1
        else:
            nonlocal_state["answered"] += 1
            if r["chosen_pred"] == r["chosen_real"]:
                nonlocal_state["correct"] += 1
        # update postfix
        ans = nonlocal_state["answered"]
        cor = nonlocal_state["correct"]
        sk = nonlocal_state["skipped"]
        c = nonlocal_state["cost"]
        if ans:
            pbar.set_postfix({
                "acc@1": f"{cor / ans:.3f}",
                "answered": ans,
                "skipped": sk,
                "cost": f"${c:.2f}",
            })

    nonlocal_state = {"cost": 0.0, "correct": 0, "skipped": 0, "answered": 0}

    await asyncio.gather(*(_run(p) for p in pairs))
    pbar.close()

    elapsed = time.time() - t0
    n_answered = nonlocal_state["answered"]
    n_correct = nonlocal_state["correct"]
    n_skipped = nonlocal_state["skipped"]
    total_cost = nonlocal_state["cost"]

    accuracy = n_correct / max(1, n_answered)
    accuracy_inc_skip = n_correct / max(1, len(results))

    # === Distribution match (Agent4Rec-style) ===
    # Per-talk: real_count = сколько раз talk был chosen_real в наших pairs
    # Per-talk: llm_count = сколько раз LLM выбрал этот talk
    real_counts = {}
    llm_counts = {}
    for r in results:
        ct_real = r["chosen_real"]
        real_counts[ct_real] = real_counts.get(ct_real, 0) + 1
        ct_pred = r["chosen_pred"]
        if ct_pred is None:
            continue
        llm_counts[ct_pred] = llm_counts.get(ct_pred, 0) + 1

    all_talks = sorted(set(real_counts) | set(llm_counts))
    real_vec = np.array([real_counts.get(t, 0) for t in all_talks], dtype=np.float64)
    llm_vec = np.array([llm_counts.get(t, 0) for t in all_talks], dtype=np.float64)

    from scipy.stats import spearmanr, pearsonr
    rho_spear, p_spear = spearmanr(real_vec, llm_vec)
    rho_pear, p_pear = pearsonr(real_vec, llm_vec)

    # JS-divergence on normalized distributions
    eps = 1e-12
    rp = real_vec / (real_vec.sum() + eps)
    lp = llm_vec / (llm_vec.sum() + eps)
    rp = rp + eps
    lp = lp + eps
    rp = rp / rp.sum()
    lp = lp / lp.sum()
    m = 0.5 * (rp + lp)
    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    js = 0.5 * kl(rp, m) + 0.5 * kl(lp, m)

    print(f"\n=== Distribution match (Agent4Rec-style) ===")
    print(f"Talks evaluated: {len(all_talks)}")
    print(f"Spearman ρ(real_counts, llm_counts) = {rho_spear:.4f} (p={p_spear:.4g})")
    print(f"Pearson ρ(real_counts, llm_counts)  = {rho_pear:.4f} (p={p_pear:.4g})")
    print(f"JS-divergence on attendance dist     = {js:.4f}")

    summary = {
        "config": {
            "model": args.model,
            "n_pairs_total": len(pairs),
            "n_results": len(results),
            "concurrency": args.concurrency,
            "max_history": args.max_history,
            "all_users": args.all_users,
        },
        "metrics": {
            "B2_accuracy_at_1": accuracy,
            "B2_accuracy_inc_skip": accuracy_inc_skip,
            "random_baseline": rb,
            "n_correct": n_correct,
            "n_answered": n_answered,
            "n_skipped": n_skipped,
            "total_cost_usd": total_cost,
            "elapsed_s": elapsed,
            # distribution match
            "spearman_rho_attendance": float(rho_spear),
            "spearman_p_attendance": float(p_spear),
            "pearson_rho_attendance": float(rho_pear),
            "pearson_p_attendance": float(p_pear),
            "js_divergence_attendance": float(js),
            "n_talks_evaluated": len(all_talks),
        },
        "talk_counts": {t: {"real": real_counts.get(t, 0),
                            "llm": llm_counts.get(t, 0)} for t in all_talks},
        "details_full": results,  # сохраняем все результаты для пересчёта
    }
    out_path = ROOT / "results" / f"llm_meetup_b2_{args.model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n=== B2 validation done in {elapsed:.0f}s ===")
    print(f"Pairs evaluated: {len(results)} (of {len(pairs)} candidates)")
    print(f"Answered: {n_answered}, skipped: {n_skipped}")
    print(f"Random baseline: {rb:.4f}")
    print(f"B2 accuracy@1 (excl skip): {accuracy:.4f}  → vs B1 параметрический=0.7782")
    print(f"B2 accuracy@1 (incl skip as wrong): {accuracy_inc_skip:.4f}")
    print(f"Total cost: ${total_cost:.3f}")
    print(f"WROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 = все доступные (≈3600); >0 = только первые N для теста")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--max-history", type=int, default=8,
                    help="Сколько прошлых RSVP'ов положить в профиль")
    ap.add_argument("--all-users", action="store_true",
                    help="не фильтровать к 467 known users (для full sample)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
