"""Capacity-aware MMR: гибрид MMR (разнообразие) + штраф за загрузку зала.

Жадный отбор top-K с тройным критерием:
  score(j | S) = β · rel(p_i, j) - (1-β) · max_{l ∈ S} cos(e_j, e_l) - α · load(j, s_t)

β = 0.6 (немного больше веса разнообразию, чем в обычном MMR)
α = 0.4 (умеренный штраф за заполненность)

Жёсткая фильтрация залов с заполненностью ≥ 0.95 (если есть альтернатива).
"""
from __future__ import annotations

import numpy as np


class CapacityAwareMMRPolicy:
    name = "Capacity-aware MMR"

    def __init__(self, beta: float = 0.6, alpha: float = 0.4, hard_threshold: float = 0.95):
        self.beta = beta
        self.alpha = alpha
        self.hard_threshold = hard_threshold

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        hall_load = state["hall_load"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        cand_emb = np.stack([conf.talks[tid].embedding for tid in cand_ids])
        rel = cand_emb @ user.embedding
        sim_mat = cand_emb @ cand_emb.T
        np.fill_diagonal(sim_mat, -np.inf)

        load_frac = np.array([
            hall_load.get((slot.id, conf.talks[tid].hall), 0)
            / max(1.0, conf.halls[conf.talks[tid].hall].capacity)
            for tid in cand_ids
        ])
        # жёсткий фильтр
        eligible = [i for i, lf in enumerate(load_frac) if lf < self.hard_threshold]
        if len(eligible) < K:
            # частично снимаем фильтр чтобы было что показать
            eligible = list(range(len(cand_ids)))

        # стартуем с argmax (rel - α·load) среди eligible
        start_score = self.beta * rel - self.alpha * load_frac
        start_score_filtered = np.full(len(cand_ids), -np.inf)
        for i in eligible:
            start_score_filtered[i] = start_score[i]
        first = int(np.argmax(start_score_filtered))
        selected = [first]
        remaining = [i for i in eligible if i != first]

        while len(selected) < K and remaining:
            best_score = -np.inf
            best_idx = remaining[0]
            for i in remaining:
                max_sim = max(sim_mat[i, j] for j in selected)
                score = (
                    self.beta * rel[i]
                    - (1 - self.beta) * max_sim
                    - self.alpha * load_frac[i]
                )
                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [cand_ids[i] for i in selected]
