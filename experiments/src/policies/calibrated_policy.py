"""Calibrated recommendations (Steck 2018, ACM RecSys).

Re-ranking поверх Cosine с минимизацией KL-дивергенции от целевого распределения категорий.
Цель: рекомендации должны сохранять пропорции категорий, характерные для пользователя.
Снижает overspecialization (узкоспециализированный recsys).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class CalibratedPolicy:
    name = "Calibrated"

    def __init__(self, lambda_kl: float = 0.5):
        """λ балансирует accuracy (cosine) vs calibration (KL).

        score = (1-λ) * accuracy_term + λ * (-KL(target || actual))
        """
        self.lambda_kl = lambda_kl

    def _category_distribution(self, talks_subset, conf):
        """Распределение категорий по подмножеству докладов."""
        cats = [conf.talks[tid].category for tid in talks_subset if tid in conf.talks]
        if not cats:
            return {}
        c = Counter(cats)
        total = sum(c.values())
        return {k: v / total for k, v in c.items()}

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        relevance_fn = state.get("relevance_fn", None)
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        # Target distribution: пропорции категорий по top-релевантным во всём каталоге
        all_talks = list(conf.talks.values())
        all_relevance = []
        for t in all_talks:
            if relevance_fn is not None:
                r = float(relevance_fn(user.embedding, t.embedding))
            else:
                r = float(np.dot(user.embedding, t.embedding))
            all_relevance.append((r, t.id))
        all_relevance.sort(reverse=True)
        # Топ-20 как индикатор интересов пользователя
        top_subset = [tid for _, tid in all_relevance[:20]]
        target_dist = self._category_distribution(top_subset, conf)

        # Жадно строим выдачу с учётом calibration
        cand_relevance = {}
        for tid in cand_ids:
            t = conf.talks[tid]
            if relevance_fn is not None:
                cand_relevance[tid] = float(relevance_fn(user.embedding, t.embedding))
            else:
                cand_relevance[tid] = float(np.dot(user.embedding, t.embedding))

        selected = []
        for _ in range(K):
            best_score = -np.inf
            best_tid = None
            for tid in cand_ids:
                if tid in selected:
                    continue
                # acc term
                acc = cand_relevance[tid]
                # calibration term: новое распределение
                trial = selected + [tid]
                actual = self._category_distribution(trial, conf)
                # smoothed KL (target || actual)
                kl = 0.0
                for cat, p_t in target_dist.items():
                    p_a = actual.get(cat, 1e-3)
                    kl += p_t * np.log(p_t / max(1e-9, p_a))
                score = (1 - self.lambda_kl) * acc - self.lambda_kl * kl
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is None:
                break
            selected.append(best_tid)

        return selected
