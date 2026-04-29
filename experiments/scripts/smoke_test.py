"""Smoke test: 10 эталонных пользователей, 4 эвристических политики, 1 сид."""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.mmr_policy import MMRPolicy  # noqa: E402


def main():
    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )
    print(f"Conference: {conf.name}, talks={len(conf.talks)}, halls={len(conf.halls)}, slots={len(conf.slots)}")

    with open(ROOT / "data" / "conferences" / "mobius_2025_autumn_users.json", encoding="utf-8") as f:
        ref = json.load(f)

    # Эмбеддинги для 10 эталонных профилей
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = [u["profile"] for u in ref["users"]]
    user_emb = model.encode(texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
    users = [
        UserProfile(id=u["id"], text=u["profile"], embedding=user_emb[i])
        for i, u in enumerate(ref["users"])
    ]
    print(f"Reference users loaded: {len(users)}")

    # Размножим 10 пользователей до 800 копий — реалистичная аудитория Mobius
    # (1200 общая вместимость, 800 даст лёгкое локальное переполнение при дисбалансе)
    REPLICAS = 80
    users_replicated = []
    for r in range(REPLICAS):
        for u in users:
            users_replicated.append(UserProfile(
                id=f"{u.id}__{r}", text=u.text, embedding=u.embedding,
            ))
    print(f"Replicated to {len(users_replicated)} virtual users (load test)")

    cfg = SimConfig(K=2, tau=0.3, lambda_overflow=2.0, p_skip_base=0.05, seed=42)

    policies = {
        "Random": RandomPolicy(seed=42),
        "Cosine": CosinePolicy(),
        "Capacity-aware": CapacityAwarePolicy(alpha=0.5),
        "MMR": MMRPolicy(beta=0.7),
    }

    for name, pol in policies.items():
        result = simulate(conf, users_replicated, pol, cfg)
        m = compute_all(conf, result)
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
