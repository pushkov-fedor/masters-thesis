"""Constrained PPO policy через stable-baselines3 + sb3_contrib MaskablePPO.

Среда: гимнастический wrapper над нашим симулятором.
- state: persona_emb (384) ⊕ hall_load_fractions (3) ⊕ slot_index_one_hot (16) = 403 dim
- action: дискретный выбор top-1 из 3 параллельных кандидатов в слоте
- reward: relevance - β * overflow_excess - γ * variance_increment
- termination: после прохождения всех слотов конференции

Action masking: исключение переполненных залов (cap < 95%).
Lagrangian dual: множитель β обновляется периодически по обратной связи
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class PPOPolicy:
    """Wrapper над обученной MaskablePPO моделью для использования в run_experiments."""
    name = "Constrained-PPO"

    def __init__(self, model_path):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        # cache: (slot_id, persona_id) -> talk_id (для воспроизводимости)
        self._cache = {}

    @staticmethod
    def make_obs(persona_emb, hall_load_fractions, slot_idx, n_slots):
        """Build observation vector."""
        slot_one_hot = np.zeros(n_slots, dtype=np.float32)
        slot_one_hot[slot_idx] = 1.0
        return np.concatenate([
            persona_emb.astype(np.float32),
            np.array(hall_load_fractions, dtype=np.float32),
            slot_one_hot,
        ])

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        # build observation
        n_slots = len(conf.slots)
        slot_idx = next(i for i, s in enumerate(conf.slots) if s.id == slot.id)
        halls_sorted = sorted(conf.halls.keys())
        hall_loads = []
        for hid in halls_sorted:
            cap = conf.halls[hid].capacity
            occ = state["hall_load"].get((slot.id, hid), 0)
            hall_loads.append(occ / max(1.0, cap))

        obs = self.make_obs(user.embedding, hall_loads, slot_idx, n_slots)

        # Action mask: индекс зала, который не переполнен
        # Действие — выбор зала (3 действия). Каждое действие соответствует hall id из halls_sorted.
        action_mask = np.array([1 if hl < 0.95 else 0 for hl in hall_loads], dtype=bool)
        if not action_mask.any():
            action_mask = np.ones_like(action_mask)

        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask)
        chosen_hall = halls_sorted[int(action)]

        # Найти доклад в выбранном зале (среди кандидатов)
        chosen_talk = None
        for tid in cand_ids:
            if conf.talks[tid].hall == chosen_hall:
                chosen_talk = tid
                break

        if chosen_talk is None:
            # fallback: top-K по релевантности
            relevance_fn = state.get("relevance_fn")
            if relevance_fn:
                scored = [(float(relevance_fn(user.embedding, conf.talks[tid].embedding)), tid) for tid in cand_ids]
            else:
                scored = [(float(np.dot(user.embedding, conf.talks[tid].embedding)), tid) for tid in cand_ids]
            scored.sort(reverse=True)
            return [tid for _, tid in scored[:K]]

        # Дополним выдачу до K талков — добавим следующих по релевантности
        relevance_fn = state.get("relevance_fn")
        rest = [tid for tid in cand_ids if tid != chosen_talk]
        if relevance_fn:
            rest_scored = sorted(rest, key=lambda tid: -float(relevance_fn(user.embedding, conf.talks[tid].embedding)))
        else:
            rest_scored = sorted(rest, key=lambda tid: -float(np.dot(user.embedding, conf.talks[tid].embedding)))
        recs = [chosen_talk] + rest_scored
        return recs[:K]
