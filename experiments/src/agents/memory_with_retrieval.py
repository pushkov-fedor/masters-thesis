"""Park2023-style память с retrieval по relevance × recency × importance.

Заменяет тривиальный entries[-5:] на содержательное хранилище воспоминаний
с поиском по контекстной релевантности.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class MemoryEntry:
    """Одна запись в памяти агента."""
    content: str                  # текст наблюдения
    content_emb: np.ndarray       # эмбеддинг текста (для retrieval)
    timestamp: int                # номер слота когда записано
    importance: float             # 0-1, насколько важна
    last_access: int              # для recency-decay
    kind: str = "observation"     # "observation", "reflection", "decision"


@dataclass
class MemoryWithRetrieval:
    """Park2023 §3.2: retrieval по composite score.

    score = relevance * recency * importance
    relevance = cosine(query_emb, content_emb)
    recency = exp(-α * (now - last_access))
    importance — назначается при добавлении (0..1)
    """
    entries: List[MemoryEntry] = field(default_factory=list)
    decay: float = 0.05  # α в формуле recency

    def add(self, content: str, content_emb: np.ndarray, timestamp: int,
            importance: float = 0.5, kind: str = "observation"):
        self.entries.append(MemoryEntry(
            content=content,
            content_emb=content_emb.astype(np.float32),
            timestamp=timestamp,
            importance=float(importance),
            last_access=timestamp,
            kind=kind,
        ))

    def fetch_relevant(self, query_emb: np.ndarray, now: int, top_k: int = 3) -> List[MemoryEntry]:
        """Park2023 retrieval: ranked top-k entries by composite score."""
        if not self.entries:
            return []
        scores = []
        for e in self.entries:
            relevance = float(np.dot(query_emb, e.content_emb))
            age = max(0, now - e.last_access)
            recency = math.exp(-self.decay * age)
            score = relevance * recency * e.importance
            scores.append(score)
        # топ-k
        idx = np.argsort(scores)[-top_k:][::-1]
        result = []
        for i in idx:
            self.entries[i].last_access = now  # access updates recency
            result.append(self.entries[i])
        return result

    def fetch_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Альтернатива: просто последние n записей (без retrieval)."""
        return self.entries[-n:]

    def aggregate_importance(self, since_timestamp: int = 0) -> float:
        """Сумма importance со времени since_timestamp — для триггера рефлексии."""
        return sum(e.importance for e in self.entries
                   if e.timestamp >= since_timestamp)

    def render_for_prompt(self, query_emb: Optional[np.ndarray] = None,
                          now: int = 0, top_k: int = 5) -> str:
        """Формирует текстовый блок для промпта.

        Если query_emb указан — retrieval по релевантности.
        Иначе — последние top_k записей.
        """
        if not self.entries:
            return "(пока пусто — это первый слот)"
        if query_emb is not None:
            relevant = self.fetch_relevant(query_emb, now, top_k=top_k)
        else:
            relevant = self.fetch_recent(top_k)
        lines = []
        for e in relevant:
            tag = "💭" if e.kind == "reflection" else "•"
            lines.append(f"{tag} (слот {e.timestamp}, важность {e.importance:.1f}) {e.content}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.entries)
