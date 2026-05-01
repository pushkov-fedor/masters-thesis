"""Тест переносимости Mobius-обученной модели интереса на Demo Day.

1. Сэмплирует N пар (персона Demo Day, доклад Demo Day).
2. Просит LLM оценить интерес для каждой пары — это ground truth для Demo Day.
3. Применяет Mobius-обученную модель к этим парам — получает предсказания.
4. Считает Pearson и Spearman между LLM-оценками и предсказаниями модели.
5. Также считает Pearson голого cosine как baseline.

Запуск:
    .venv/bin/python scripts/test_preference_model_transfer.py --n-pairs 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]


SYSTEM_PROMPT = """Ты оцениваешь, насколько доклад на IT-конференции интересен конкретному участнику.

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
        if 0.0 <= s <= 1.0:
            return s
    except Exception:
        pass
    return None


async def rate_pair(client, model, persona, talk, sem, pbar):
    user_msg = USER_TEMPLATE.format(
        persona=persona.get("background", persona.get("profile", ""))[:600],
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
                temperature=0.2, max_tokens=80, timeout=60,
            )
        except Exception as e:
            pbar.update(1)
            return None
    pbar.update(1)
    msg = resp.choices[0].message.content or ""
    return parse_score(msg)


async def main_async(args):
    rng = np.random.default_rng(args.seed)

    with open(ROOT / "data" / "conferences" / "demo_day_2026.json", encoding="utf-8") as f:
        prog = json.load(f)
    talks_meta = {t["id"]: t for t in prog["talks"]}

    with open(ROOT / "data" / "personas" / f"{args.personas}.json", encoding="utf-8") as f:
        personas_list = json.load(f)
    personas_meta = {p["id"]: p for p in personas_list}

    npz_p = np.load(ROOT / "data" / "personas" / f"{args.personas}_embeddings.npz", allow_pickle=False)
    p_emb = {pid: npz_p["embeddings"][i] for i, pid in enumerate(npz_p["ids"])}
    npz_t = np.load(ROOT / "data" / "conferences" / "demo_day_2026_embeddings.npz", allow_pickle=False)
    t_emb = {tid: npz_t["embeddings"][i] for i, tid in enumerate(npz_t["ids"])}

    persona_ids = list(personas_meta.keys())
    talk_ids = list(talks_meta.keys())

    sample_pairs = []
    for _ in range(args.n_pairs):
        pid = persona_ids[rng.integers(0, len(persona_ids))]
        tid = talk_ids[rng.integers(0, len(talk_ids))]
        sample_pairs.append((pid, tid))
    print(f"Sampled {len(sample_pairs)} (persona, talk) pairs from Demo Day")

    client = AsyncOpenAI(api_key=load_api_key(), base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.concurrency)

    from tqdm import tqdm
    pbar = tqdm(total=len(sample_pairs), desc="LLM ratings", file=sys.stdout, mininterval=1.0)

    t0 = time.time()
    tasks = [
        rate_pair(client, args.model, personas_meta[pid], talks_meta[tid], sem, pbar)
        for pid, tid in sample_pairs
    ]
    llm_scores = await asyncio.gather(*tasks)
    pbar.close()
    print(f"\nLLM ratings done in {time.time()-t0:.1f}s")

    valid = [(p, s) for p, s in zip(sample_pairs, llm_scores) if s is not None]
    print(f"Valid ratings: {len(valid)} / {len(sample_pairs)}")
    if not valid:
        raise SystemExit("No valid LLM ratings — abort")
    pairs_v = [p for p, _ in valid]
    llm_v = np.array([s for _, s in valid], dtype=np.float32)

    print("Loading Mobius-trained preference model...")
    with open(ROOT / "data" / "models" / "preference_model.pkl", "rb") as f:
        model = pickle.load(f)

    feats = []
    cosines = []
    for pid, tid in pairs_v:
        pe = p_emb[pid]
        te = t_emb[tid]
        cos = float(np.dot(pe, te))
        feats.append(np.concatenate([pe, te, [cos]]))
        cosines.append(cos)
    feats = np.array(feats, dtype=np.float32)
    cosines = np.array(cosines, dtype=np.float32)

    preds = np.clip(model.predict(feats), 0.0, 1.0)

    print(f"\nLLM scores: mean={llm_v.mean():.3f}, std={llm_v.std():.3f}")
    print(f"Model preds: mean={preds.mean():.3f}, std={preds.std():.3f}")
    print(f"Cosines:     mean={cosines.mean():.3f}, std={cosines.std():.3f}")

    p_model, _ = pearsonr(preds, llm_v)
    s_model, _ = spearmanr(preds, llm_v)
    p_cos, _ = pearsonr(cosines, llm_v)
    s_cos, _ = spearmanr(cosines, llm_v)

    print()
    print(f"Mobius-trained model on Demo Day pairs:")
    print(f"  Pearson  = {p_model:.4f}")
    print(f"  Spearman = {s_model:.4f}")
    print()
    print(f"Cosine baseline on Demo Day pairs:")
    print(f"  Pearson  = {p_cos:.4f}")
    print(f"  Spearman = {s_cos:.4f}")
    print()
    print(f"Mobius-test reference (group split): Pearson 0.74")
    print(f"Cosine-test reference: Pearson 0.34")

    out = {
        "n_pairs": len(valid),
        "model": args.model,
        "personas": args.personas,
        "mobius_trained_model_on_demoday": {
            "pearson": float(p_model), "spearman": float(s_model),
        },
        "cosine_on_demoday": {
            "pearson": float(p_cos), "spearman": float(s_cos),
        },
        "llm_score_stats": {"mean": float(llm_v.mean()), "std": float(llm_v.std())},
        "model_pred_stats": {"mean": float(preds.mean()), "std": float(preds.std())},
    }
    out_path = ROOT / "results" / "preference_model_transfer_demoday.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"WROTE: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--personas", default="personas")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
