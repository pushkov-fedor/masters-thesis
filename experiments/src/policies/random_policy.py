"""Random policy: равномерная случайная выборка K докладов из доступных в слоте."""
from __future__ import annotations

import numpy as np

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        ids = list(slot.talk_ids)
        if len(ids) <= K:
            return ids
        idx = self.rng.choice(len(ids), size=K, replace=False)
        return [ids[i] for i in idx]
