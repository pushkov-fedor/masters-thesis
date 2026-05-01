"""GNN policy — GraphSAGE-стиль message passing на графе докладов.

Граф: узлы — доклады; рёбра — между докладами с cosine ≥ threshold ИЛИ
с одинаковым category. После 2 layers message passing эмбеддинги докладов
обогащаются информацией о соседях (близких темах).

Реализация без torch_geometric — простая численная (numpy + matmul).
Достаточно для 40-210 узлов. Для больших графов нужно PyG.
"""
from __future__ import annotations

from .base import BasePolicy

import numpy as np


class GNNPolicy(BasePolicy):
    name = "GNN"

    def __init__(self, edge_threshold: float = 0.5, n_layers: int = 2,
                 self_weight: float = 0.5):
        """edge_threshold: добавляем ребро если cosine ≥ threshold (или same category).
        n_layers: глубина message passing.
        self_weight: вес собственного эмбеддинга в обновлении (1 - вес соседей).
        """
        self.edge_threshold = edge_threshold
        self.n_layers = n_layers
        self.self_weight = self_weight
        self._gnn_emb_cache = None  # {talk_id: enriched_emb}

    def _build_graph(self, conf):
        """Строим adjacency matrix W (нормализованную) и возвращаем."""
        talks = list(conf.talks.values())
        n = len(talks)
        embs = np.stack([t.embedding for t in talks])  # (n, d)
        # Cosine similarity
        sim = embs @ embs.T
        # Adjacency: cosine ≥ threshold OR same category
        cat_match = np.zeros((n, n), dtype=np.float32)
        cats = [t.category for t in talks]
        for i in range(n):
            for j in range(n):
                if i != j and cats[i] == cats[j]:
                    cat_match[i, j] = 1.0
        edges = ((sim >= self.edge_threshold).astype(np.float32) + cat_match)
        np.fill_diagonal(edges, 0)
        edges = (edges > 0).astype(np.float32)
        # Row-normalize
        row_sum = edges.sum(axis=1, keepdims=True) + 1e-9
        W = edges / row_sum
        return talks, embs, W

    def _enrich(self, conf):
        """Запускает message passing и кэширует результат."""
        if self._gnn_emb_cache is not None:
            return self._gnn_emb_cache
        talks, embs, W = self._build_graph(conf)
        # Update: h^{l+1} = self_weight * h^l + (1-self_weight) * mean(neighbors)
        h = embs.copy()
        for _ in range(self.n_layers):
            agg = W @ h  # weighted mean of neighbors
            h = self.self_weight * h + (1 - self.self_weight) * agg
            # L2 normalize
            h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        self._gnn_emb_cache = {t.id: h[i] for i, t in enumerate(talks)}
        return self._gnn_emb_cache

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids
        enriched = self._enrich(conf)
        scored = []
        for tid in cand_ids:
            enriched_emb = enriched[tid]
            sim = float(np.dot(user.embedding, enriched_emb))
            scored.append((sim, tid))
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:K]]
