"""Реплицирует LLM-персон с шумом эмбеддинга для масштабирования нагрузки.

Идея: 300 LLM-персон → 900 пользователей конференции с разнообразием
вокруг 300 центров.

Каждая копия имеет:
- id = base_id + '_r{i}'
- text/background = тот же
- embedding = base_embedding + N(0, σ), нормализован

Эмбеддинг отличается → разные cosine-предпочтения.
LLM-ranker должен использовать base_id для кэша (сохранять одинаковое ранжирование).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--copies", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input", default="personas")
    p.add_argument("--output", default="personas_x3")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    in_json = ROOT / "data" / "personas" / f"{args.input}.json"
    in_npz = ROOT / "data" / "personas" / f"{args.input}_embeddings.npz"
    out_json = ROOT / "data" / "personas" / f"{args.output}.json"
    out_npz = ROOT / "data" / "personas" / f"{args.output}_embeddings.npz"

    with open(in_json, encoding="utf-8") as f:
        base = json.load(f)
    npz = np.load(in_npz, allow_pickle=False)
    ids = list(npz["ids"])
    emb = npz["embeddings"]
    by_id = {pid: emb[i] for i, pid in enumerate(ids)}

    out_personas = []
    out_embs = []
    for p_meta in base:
        base_id = p_meta["id"]
        base_emb = by_id[base_id]
        for ci in range(args.copies):
            new_id = f"{base_id}_r{ci}"
            v = base_emb + rng.normal(0, args.sigma, size=base_emb.shape).astype(np.float32)
            v = v / max(1e-9, np.linalg.norm(v))
            new = dict(p_meta)
            new["id"] = new_id
            new["base_id"] = base_id
            new["replica_idx"] = ci
            out_personas.append(new)
            out_embs.append(v)

    out_embs = np.stack(out_embs).astype(np.float32)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_personas, f, ensure_ascii=False, indent=2)
    np.savez(out_npz, ids=np.array([p["id"] for p in out_personas]), embeddings=out_embs)
    print(f"DONE: {len(out_personas)} personas (base={len(base)}, copies={args.copies}, σ={args.sigma})")
    print(f"WROTE: {out_json}")
    print(f"WROTE: {out_npz}")


if __name__ == "__main__":
    main()
