"""Random + capacity filter.

Контрольная политика: случайный выбор K из доступных, но залы с заполненностью
≥ hard_threshold исключаются. Если все исключены — fallback на полный набор.

Используется как контр-аргумент: если её результаты близки к Capacity-aware,
значит рекомендательная часть (релевантность, разнообразие) добавляет мало
поверх capacity-фильтра.
"""
from __future__ import annotations

from .base import BasePolicy

import numpy as np


class RandomCapacityAwarePolicy(BasePolicy):
    name = "Random + capacity"

    def __init__(self, hard_threshold: float = 0.95, seed: int = 0):
        self.hard_threshold = hard_threshold
        self.rng = np.random.default_rng(seed)

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        hall_load = state["hall_load"]
        cand_ids = list(slot.talk_ids)

        eligible = []
        for tid in cand_ids:
            t = conf.talks[tid]
            cap = conf.capacity_at(slot.id, t.hall)
            occ = hall_load.get((slot.id, t.hall), 0)
            load_frac = occ / max(1.0, cap)
            if load_frac < self.hard_threshold:
                eligible.append(tid)

        if not eligible:
            eligible = cand_ids

        if len(eligible) <= K:
            return eligible
        idx = self.rng.choice(len(eligible), size=K, replace=False)
        return [eligible[i] for i in idx]
