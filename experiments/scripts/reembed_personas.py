"""Пересчитывает эмбеддинги существующих персон без изменения JSON.

Используется при смене эмбеддера. Берёт `<name>.json`, эмбеддит поле
`background`/`profile` через единый embedder (e5-small, kind=query),
сохраняет в `<name>_embeddings.npz`.

Пример:
    .venv/bin/python scripts/reembed_personas.py --name personas
    .venv/bin/python scripts/reembed_personas.py --name personas_100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedder import embed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="имя файла personas без .json")
    args = ap.parse_args()

    in_path = ROOT / "data" / "personas" / f"{args.name}.json"
    out_path = ROOT / "data" / "personas" / f"{args.name}_embeddings.npz"

    with open(in_path, encoding="utf-8") as f:
        personas = json.load(f)
    texts = [p.get("background") or p.get("profile") or "" for p in personas]
    ids = [p["id"] for p in personas]

    print(f"Re-embedding {len(texts)} personas as 'query'...")
    emb = embed_texts(texts, kind="query")
    print(f"Embeddings shape: {emb.shape}")
    np.savez(out_path, ids=np.array(ids), embeddings=emb)
    print(f"WROTE: {out_path}")


if __name__ == "__main__":
    main()
