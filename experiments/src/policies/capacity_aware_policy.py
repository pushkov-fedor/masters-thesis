"""Capacity-aware rule-based policy: top-K по cosine, но с предпочтением менее загруженных залов.

Утилита: relevance - α * load_fraction. Простая жадная эвристика.
Если все залы переполнены — берём top-K по cosine.
"""
from __future__ import annotations

import numpy as np


class CapacityAwarePolicy:
    name = "Capacity-aware"

    def __init__(self, alpha: float = 0.5, hard_threshold: float = 0.95):
        self.alpha = alpha
        self.hard_threshold = hard_threshold

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        hall_load = state["hall_load"]
        scored = []
        backup = []
        for tid in slot.talk_ids:
            t = conf.talks[tid]
            sim = float(np.dot(user.embedding, t.embedding))
            cap = conf.halls[t.hall].capacity
            occ = hall_load.get((slot.id, t.hall), 0)
            load_frac = occ / max(1.0, cap)
            penalty = self.alpha * load_frac
            score = sim - penalty
            scored.append((score, tid, load_frac))
            backup.append((sim, tid))
        # Жёсткая фильтрация: исключаем залы с заполненностью >= hard_threshold,
        # если остаётся хоть какой-то выбор
        soft = [s for s in scored if s[2] < self.hard_threshold]
        if len(soft) >= K or len(soft) > 0:
            soft.sort(reverse=True)
            return [tid for _, tid, _ in soft[:K]]
        # fallback — все залы переполнены, возвращаем top-K по релевантности
        backup.sort(reverse=True)
        return [tid for _, tid in backup[:K]]
