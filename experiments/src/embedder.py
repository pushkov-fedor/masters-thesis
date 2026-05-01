"""Единая обёртка над sentence-transformers для эмбеддингов текстов.

Используем `intfloat/multilingual-e5-small` — современный multilingual encoder
(2023), 384-мерный выход, ~118M параметров, лучше чем устаревший
paraphrase-MiniLM-L12-v2 на retrieval-задачах.

E5 ожидает префикс:
- "query: " для текстов, которые "ищут" — у нас это профили участников.
- "passage: " для текстов, которые "находят" — у нас это описания докладов.

Эмбеддинги нормализуются (для cosine similarity).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


_DEFAULT_MODEL = "intfloat/multilingual-e5-small"
_cache = {}


def get_model(name: str = _DEFAULT_MODEL):
    if name not in _cache:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {name}")
        _cache[name] = SentenceTransformer(name)
    return _cache[name]


def embed_texts(
    texts: List[str],
    kind: str = "passage",  # "passage" | "query"
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Возвращает (n, 384) нормализованные эмбеддинги."""
    if kind not in ("passage", "query"):
        raise ValueError(f"kind must be 'passage' or 'query', got {kind!r}")
    prefixed = [f"{kind}: {t}" for t in texts]
    model = get_model(model_name)
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)
