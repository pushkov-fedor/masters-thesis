"""Fallback-генератор персон без LLM API.

Стратегия: 10 эталонных профилей × N копий с управляемой вариативностью.
Для каждой копии:
  - текстовый профиль остаётся прежним (для отчётности),
  - эмбеддинг = base_embedding + Gaussian noise (σ управляет разбросом),
  - после нормализации возвращаем на единичную сферу.

Это даёт ~300 «вариантов» вокруг 10 центров. Не настолько разнообразно,
как LLM-генерация (нет новых ролей/стеков), но контролируемо
и позволяет провести полноценный эксперимент.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
USERS_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn_users.json"
OUT_JSON = ROOT / "data" / "personas" / "personas.json"
OUT_NPZ = ROOT / "data" / "personas" / "personas_embeddings.npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--copies", type=int, default=80, help="копий каждого из 10 базовых профилей")
    p.add_argument("--sigma", type=float, default=0.05, help="σ Гауссовского шума в эмбеддинге")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(USERS_PATH, encoding="utf-8") as f:
        ref = json.load(f)

    print(f"Loading model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    base_texts = [u["profile"] for u in ref["users"]]
    base_emb = model.encode(base_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
    print(f"Base profiles encoded: {base_emb.shape}")

    personas = []
    embs = []
    pid = 0
    for ci in range(args.copies):
        for ui, u in enumerate(ref["users"]):
            base = base_emb[ui]
            noise = rng.normal(0, args.sigma, size=base.shape).astype(np.float32)
            v = base + noise
            v = v / max(1e-9, np.linalg.norm(v))
            pid += 1
            personas.append({
                "id": f"u_{pid:04d}",
                "base_id": u["id"],
                "profile": u["profile"],  # текст не меняем
                "variation_idx": ci,
            })
            embs.append(v)

    embs = np.stack(embs).astype(np.float32)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    np.savez(OUT_NPZ, ids=np.array([p["id"] for p in personas]), embeddings=embs)

    print(f"DONE: {len(personas)} personas (10 base × {args.copies} copies)")
    print(f"  σ = {args.sigma}, embeddings: {embs.shape}")
    print(f"WROTE: {OUT_JSON}")
    print(f"WROTE: {OUT_NPZ}")


if __name__ == "__main__":
    main()
