"""Уменьшенный набор для state-aware LLM-ranker: 200 пользователей."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(42)
    with open(ROOT / "data" / "personas" / "personas_x3.json", encoding="utf-8") as f:
        full = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_x3_embeddings.npz", allow_pickle=False)
    ids = list(npz["ids"])
    emb_map = {pid: npz["embeddings"][i] for i, pid in enumerate(ids)}

    # Берём первых 200 (ровно 67 базовых × 3 копии = 201, округлим до 200)
    subset = full[:200]
    sub_emb = np.stack([emb_map[p["id"]] for p in subset]).astype(np.float32)

    out_json = ROOT / "data" / "personas" / "personas_x3_200.json"
    out_npz = ROOT / "data" / "personas" / "personas_x3_200_embeddings.npz"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    np.savez(out_npz, ids=np.array([p["id"] for p in subset]), embeddings=sub_emb)
    print(f"DONE: {len(subset)} personas")


if __name__ == "__main__":
    main()
