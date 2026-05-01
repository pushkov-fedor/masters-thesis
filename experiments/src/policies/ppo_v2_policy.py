"""PPO v2 policy wrapper — multi-agent batch episode trained model.

Отличие от ppo_policy.py: использует расширенное observation
(persona_emb + hall_loads + slot_one_hot + users_remain + fame_in_slot).
Обучен в правильно поставленной congestion-game среде.
"""
from __future__ import annotations

from .base import BasePolicy

from pathlib import Path

import numpy as np


class PPOv2Policy(BasePolicy):
    name = "Constrained-PPO-v2"

    def __init__(self, model_path):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self.halls_sorted = None  # будет инициализирован при первом вызове

    def _ensure_halls(self, conf):
        if self.halls_sorted is None:
            self.halls_sorted = sorted(conf.halls.keys())

    @staticmethod
    def make_obs(persona_emb, hall_load_fractions, slot_idx, n_slots,
                 users_remain_frac, fame_in_slot):
        slot_one_hot = np.zeros(n_slots, dtype=np.float32)
        slot_one_hot[slot_idx] = 1.0
        return np.concatenate([
            persona_emb.astype(np.float32),
            np.array(hall_load_fractions, dtype=np.float32),
            slot_one_hot,
            np.array([users_remain_frac], dtype=np.float32),
            np.array(fame_in_slot, dtype=np.float32),
        ])

    def __call__(self, *, user, slot, conf, state):
        self._ensure_halls(conf)
        K = state["K"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        # Build observation
        n_slots = len(conf.slots)
        slot_idx = next(i for i, s in enumerate(conf.slots) if s.id == slot.id)

        hall_loads = []
        for hid in self.halls_sorted:
            cap = conf.capacity_at(slot.id, hid)
            occ = state["hall_load"].get((slot.id, hid), 0)
            hall_loads.append(occ / max(1.0, cap))

        # users_remain ≈ 0.5 (mid-batch — по-нормальному)
        users_remain_frac = 0.5

        # fame in slot per hall
        fame_in_slot = [0.0] * len(self.halls_sorted)
        for tid in cand_ids:
            t = conf.talks[tid]
            if t.hall in self.halls_sorted:
                hi = self.halls_sorted.index(t.hall)
                fame_in_slot[hi] = max(fame_in_slot[hi], getattr(t, "fame", 0.0))

        obs = self.make_obs(user.embedding, hall_loads, slot_idx, n_slots,
                            users_remain_frac, fame_in_slot)

        # Action mask
        halls_in_slot = {conf.talks[tid].hall for tid in cand_ids}
        action_mask = np.array([
            (hid in halls_in_slot and hall_loads[i] < 0.95)
            for i, hid in enumerate(self.halls_sorted)
        ], dtype=bool)
        if not action_mask.any():
            action_mask = np.array([hid in halls_in_slot for hid in self.halls_sorted], dtype=bool)
        if not action_mask.any():
            action_mask = np.ones(len(self.halls_sorted), dtype=bool)

        action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask)
        chosen_hall = self.halls_sorted[int(action)]

        chosen_talk = None
        for tid in cand_ids:
            if conf.talks[tid].hall == chosen_hall:
                chosen_talk = tid
                break

        if chosen_talk is None:
            relevance_fn = state.get("relevance_fn")
            if relevance_fn:
                scored = [(float(relevance_fn(user.embedding, conf.talks[tid].embedding)), tid)
                          for tid in cand_ids]
            else:
                scored = [(float(np.dot(user.embedding, conf.talks[tid].embedding)), tid)
                          for tid in cand_ids]
            scored.sort(reverse=True)
            return [tid for _, tid in scored[:K]]

        # Дополним до K по релевантности
        relevance_fn = state.get("relevance_fn")
        rest = [tid for tid in cand_ids if tid != chosen_talk]
        if relevance_fn:
            rest_scored = sorted(rest, key=lambda tid: -float(relevance_fn(user.embedding, conf.talks[tid].embedding)))
        else:
            rest_scored = sorted(rest, key=lambda tid: -float(np.dot(user.embedding, conf.talks[tid].embedding)))
        recs = [chosen_talk] + rest_scored
        return recs[:K]
