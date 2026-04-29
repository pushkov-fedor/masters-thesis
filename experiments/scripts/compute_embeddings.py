"""Считает эмбеддинги докладов локально через sentence-transformers.

Модель: paraphrase-multilingual-MiniLM-L12-v2 (быстрая, ~120MB, 384-dim).
Текст для эмбеддинга: title + abstract + category.
"""
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PROG_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn.json"
OUT_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def make_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def main():
    with open(PROG_PATH, encoding="utf-8") as f:
        prog = json.load(f)

    talks = prog["talks"]
    texts = [make_text(t) for t in talks]
    ids = [t["id"] for t in talks]

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(texts)} texts...")
    emb = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {emb.shape}")

    np.savez(OUT_PATH, ids=np.array(ids), embeddings=emb.astype(np.float32))
    print(f"WROTE: {OUT_PATH}")


if __name__ == "__main__":
    main()
