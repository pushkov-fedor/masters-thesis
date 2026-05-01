"""MMR (Maximal Marginal Relevance): top-K с балансом релевантности и тематического разнообразия.

MMR(j | S) = β · rel(p_i, j) - (1-β) · max_{l ∈ S} cos(e_j, e_l)

β = 0.7 по умолчанию.
"""
from __future__ import annotations

from .base import BasePolicy

import numpy as np


class MMRPolicy(BasePolicy):
    name = "MMR"

    def __init__(self, beta: float = 0.7):
        self.beta = beta

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        relevance_fn = state.get("relevance_fn", None)
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids
        cand_emb = np.stack([conf.talks[tid].embedding for tid in cand_ids])
        if relevance_fn is not None:
            rel = np.array([float(relevance_fn(user.embedding, e)) for e in cand_emb])
        else:
            rel = cand_emb @ user.embedding  # косинус (нормализованы)
        # внутрикандидатская матрица сходства
        sim_mat = cand_emb @ cand_emb.T
        np.fill_diagonal(sim_mat, -np.inf)

        selected = []
        remaining = list(range(len(cand_ids)))
        # стартуем с самого релевантного
        first = int(np.argmax(rel))
        selected.append(first)
        remaining.remove(first)
        while len(selected) < K and remaining:
            best_score = -np.inf
            best_idx = remaining[0]
            for i in remaining:
                max_sim = max(sim_mat[i, j] for j in selected)
                score = self.beta * rel[i] - (1 - self.beta) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [cand_ids[i] for i in selected]
