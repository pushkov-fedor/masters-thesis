"""Унифицированный пересчёт эмбеддингов для конференций или персон.

Использует общий модуль `src/embedder.py` (e5-small с query/passage префиксами).

Примеры:
    # эмбеддинги докладов конференции
    .venv/bin/python scripts/embed.py talks --conference mobius_2025_autumn
    .venv/bin/python scripts/embed.py talks --conference demo_day_2026

    # эмбеддинги персон (тот же файл, поле background/profile)
    .venv/bin/python scripts/embed.py personas --name personas
    .venv/bin/python scripts/embed.py personas --name personas_demoday_100
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


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def cmd_talks(args):
    prog_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    out_path = ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz"
    with open(prog_path, encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]
    texts = [make_talk_text(t) for t in talks]
    ids = [t["id"] for t in talks]
    print(f"Encoding {len(texts)} talks as 'passage'...")
    emb = embed_texts(texts, kind="passage")
    np.savez(out_path, ids=np.array(ids), embeddings=emb)
    print(f"WROTE: {out_path} (shape={emb.shape})")


def cmd_personas(args):
    in_path = ROOT / "data" / "personas" / f"{args.name}.json"
    out_path = ROOT / "data" / "personas" / f"{args.name}_embeddings.npz"
    with open(in_path, encoding="utf-8") as f:
        personas = json.load(f)
    texts = [p.get("background") or p.get("profile") or "" for p in personas]
    ids = [p["id"] for p in personas]
    print(f"Encoding {len(texts)} personas as 'query'...")
    emb = embed_texts(texts, kind="query")
    np.savez(out_path, ids=np.array(ids), embeddings=emb)
    print(f"WROTE: {out_path} (shape={emb.shape})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_t = sub.add_parser("talks", help="эмбеддинги докладов конференции")
    p_t.add_argument("--conference", required=True,
                     help="имя файла в data/conferences/ без .json")
    p_t.set_defaults(func=cmd_talks)
    p_p = sub.add_parser("personas", help="эмбеддинги персон")
    p_p.add_argument("--name", required=True,
                     help="имя файла в data/personas/ без .json")
    p_p.set_defaults(func=cmd_personas)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
