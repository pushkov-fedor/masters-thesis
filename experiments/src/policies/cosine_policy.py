"""Cosine policy: top-K по косинусной близости профиля и эмбеддинга доклада, без учёта вместимости."""
from __future__ import annotations

import numpy as np


class CosinePolicy:
    name = "Cosine"

    def __init__(self, **_):
        pass

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        scored = []
        for tid in slot.talk_ids:
            t = conf.talks[tid]
            sim = float(np.dot(user.embedding, t.embedding))
            scored.append((sim, tid))
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:K]]
