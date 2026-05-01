"""Считает эмбеддинги докладов через intfloat/multilingual-e5-small.

Принимает имя конференции в --conference (например, mobius_2025_autumn,
demo_day_2026). Тексты эмбеддятся как "passage" (e5-конвенция).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedder import embed_texts  # noqa: E402


def make_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn")
    args = ap.parse_args()

    prog_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    out_path = ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz"

    with open(prog_path, encoding="utf-8") as f:
        prog = json.load(f)

    talks = prog["talks"]
    texts = [make_text(t) for t in talks]
    ids = [t["id"] for t in talks]

    print(f"Encoding {len(texts)} talk texts as 'passage'...")
    emb = embed_texts(texts, kind="passage")
    print(f"Embeddings shape: {emb.shape}")

    np.savez(out_path, ids=np.array(ids), embeddings=emb)
    print(f"WROTE: {out_path}")


if __name__ == "__main__":
    main()
