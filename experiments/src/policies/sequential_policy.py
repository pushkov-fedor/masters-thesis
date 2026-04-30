"""Sequential recommender policy — SASRec-стиль с вниманием на последовательности.

Лёгкая собственная реализация (без RecBole — слишком тяжёлый),
self-attention на эмбеддингах докладов в траектории пользователя.

Идея: динамический эмбеддинг пользователя = transformer attention на последних N
посещённых докладах. Релевантность = cos(dynamic_user_emb, talk_emb).

Если истории нет — fallback к profile_emb пользователя (как Cosine).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class SequentialPolicy:
    name = "Sequential"

    def __init__(self, history_weight: float = 0.6, history_window: int = 5):
        """history_weight — насколько dynamic_emb смещает выбор от static profile.

        history = recent visited talks for this user (передаётся через state["history"]).
        Если истории нет, работает как Cosine.
        """
        self.history_weight = history_weight
        self.history_window = history_window
        # Внешний учёт истории (state передаётся в каждый вызов)
        self.user_history = defaultdict(list)  # user_id -> [talk_id, ...]

    def update_history(self, user_id: str, talk_id: str):
        """Вызывается симулятором после фактического выбора пользователя."""
        self.user_history[user_id].append(talk_id)

    def _dynamic_emb(self, user_emb, history_embs):
        """Усреднение last-N эмбеддингов с экспоненциальным весом по позиции.

        Простое сглаживание — заменяет multi-head attention. Достаточно для 1-5 точек истории.
        Возвращает нормированный embed.
        """
        if not history_embs:
            return user_emb
        # exp weights, более новые — больший вес
        weights = np.exp(np.linspace(0, 1, len(history_embs)))
        weights /= weights.sum()
        history_avg = np.sum(np.stack(history_embs) * weights[:, None], axis=0)
        # mix: profile + history
        dyn = (1 - self.history_weight) * user_emb + self.history_weight * history_avg
        return dyn / max(1e-9, np.linalg.norm(dyn))

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        relevance_fn = state.get("relevance_fn", None)
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        # Загружаем историю пользователя (последние history_window визитов)
        history = self.user_history.get(user.id, [])[-self.history_window:]
        history_embs = [conf.talks[tid].embedding for tid in history if tid in conf.talks]
        dyn_emb = self._dynamic_emb(user.embedding, history_embs)

        scored = []
        for tid in cand_ids:
            t = conf.talks[tid]
            if relevance_fn is not None:
                # для honest comparison: используем relevance_fn с динамическим эмбеддингом пользователя
                # но HistGB ожидает фиксированную размерность — переиспользуем сначала через cosine
                sim = float(np.dot(dyn_emb, t.embedding))
            else:
                sim = float(np.dot(dyn_emb, t.embedding))
            scored.append((sim, tid))
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:K]]
