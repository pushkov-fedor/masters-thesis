"""DPP (Determinantal Point Process) policy — математически принципиальный diversity ranker.

Заменяет эвристический MMR. Greedy MAP inference (Kulesza & Taskar 2012):
выбирает top-K максимизируя det(L[selected]), где L = quality * similarity.

Формула DPP вероятность набора S: P(S) ∝ det(L_S),
где L = диагональ_quality * sim * диагональ_quality.
"""
from __future__ import annotations

from .base import BasePolicy

import numpy as np


class DPPPolicy(BasePolicy):
    name = "DPP"

    def __init__(self, alpha: float = 0.5):
        """alpha балансирует quality vs diversity.

        Quality ~ relevance, Similarity ~ cosine между талками.
        L_ii = q_i^2; L_ij = q_i * q_j * S_ij.
        """
        self.alpha = alpha

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        relevance_fn = state.get("relevance_fn", None)
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        cand_emb = np.stack([conf.talks[tid].embedding for tid in cand_ids])
        # Quality: relevance к пользователю
        if relevance_fn is not None:
            q = np.array([float(relevance_fn(user.embedding, e)) for e in cand_emb])
        else:
            q = cand_emb @ user.embedding
        # exp scaling — quality должна быть положительной для DPP
        q = np.exp(self.alpha * q)
        # Similarity matrix: cosine (эмбеддинги нормализованы)
        S = cand_emb @ cand_emb.T

        # L matrix
        L = np.outer(q, q) * S

        # Greedy MAP: на каждом шаге выбираем доклад,
        # максимизирующий det(L[selected ∪ {i}])
        # Формула обновления: gain_i = L_ii - L_iS @ inv(L_SS) @ L_Si
        n = len(cand_ids)
        selected = []
        cis = np.zeros((K, n))  # vectors c_i для каждого ещё не выбранного
        d2 = np.diag(L).copy()  # квадратные расстояния
        log_det = 0.0

        for k in range(K):
            # выбираем индекс с максимальным d2
            i = int(np.argmax(d2))
            if d2[i] <= 0:
                break
            log_det += np.log(d2[i])
            selected.append(i)
            # обновляем c_j и d2_j для оставшихся
            sqrt_di = np.sqrt(d2[i])
            for j in range(n):
                if j == i or j in selected:
                    d2[j] = -np.inf
                    continue
                # eq (15) Chen2018 fast greedy MAP:
                # c_j = (L_ji - <c_j_prev, c_i_prev>) / sqrt(d_i)
                e_ji = (L[j, i] - cis[:k, j].dot(cis[:k, i])) / sqrt_di
                cis[k, j] = e_ji
                d2[j] = d2[j] - e_ji * e_ji
            d2[i] = -np.inf

        return [cand_ids[i] for i in selected]
