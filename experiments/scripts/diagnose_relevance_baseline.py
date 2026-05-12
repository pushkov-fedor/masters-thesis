"""Baseline качества функции релевантности на тестовой выборке.

Тестовая выборка — 20 персон и 20 докладов в 5 категориях по 4 в каждой
(iOS/Android/Backend/ML/Frontend). Релевантной парой считается «своя категория
↔ своя категория», нерелевантной — все остальные.

Считает матрицу косинусной близости через тот же e5-small-pipeline, что и
основная система. Сравнивает распределения cos на релевантных и нерелевантных
парах.

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_relevance_baseline.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedder import embed_texts


TEST_DIR = ROOT / "data" / "personas" / "test_diversity"


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def load_data():
    with open(TEST_DIR / "personas.json", encoding="utf-8") as f:
        personas = json.load(f)
    with open(TEST_DIR / "talks.json", encoding="utf-8") as f:
        talks = json.load(f)
    return personas, talks


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Простая токенизация: нижний регистр, альфа-числовые последовательности."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 без лемматизации.

    Стандартные параметры k1=1.5, b=0.75 — индустриальный default
    (Elasticsearch, Lucene).
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_lens = np.array([len(d) for d in docs])
        self.avgdl = self.doc_lens.mean() if len(docs) > 0 else 0.0
        self.doc_freqs = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        n = len(docs)
        # IDF по варианту с +0.5 (BM25+, Lucene)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: list[str], doc_idx: int) -> float:
        score = 0.0
        freqs = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        for term in query:
            if term not in self.idf:
                continue
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = self.idf[term]
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * num / denom
        return score

    def score_matrix(self, queries: list[list[str]]) -> np.ndarray:
        n_q, n_d = len(queries), len(self.docs)
        out = np.zeros((n_q, n_d))
        for i, q in enumerate(queries):
            for j in range(n_d):
                out[i, j] = self.score(q, j)
        return out


def all_but_the_top(vecs: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Postprocessing по Mu, Bhat, Viswanath (2018):
    1) центрирование (вычитание среднего),
    2) проекция ортогонально к топ-n_components PCA-направлениям.

    Идея: топ-PCA направления у контекстуальных эмбеддингов кодируют
    «доменный фон» (то общее, что у всех векторов), а не семантику.
    Их удаление расширяет узкий конус.
    """
    mu = vecs.mean(axis=0, keepdims=True)
    centered = vecs - mu
    # SVD на центрированной матрице
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Проекция ортогонально к первым n_components компонентам
    top_components = Vt[:n_components]  # (n_components, dim)
    projection = centered @ top_components.T @ top_components  # вклад топ-направлений
    out = centered - projection
    return out


def report(personas, talks, sim, label):
    same_pairs = []
    cross_pairs = []
    for i, p in enumerate(personas):
        for j, t in enumerate(talks):
            entry = {
                "p_id": p["id"], "p_cat": p["category"], "p_role": p["role"],
                "t_id": t["id"], "t_cat": t["category"], "t_title": t["title"],
                "cos": float(sim[i, j]),
            }
            if p["category"] == t["category"]:
                same_pairs.append(entry)
            else:
                cross_pairs.append(entry)

    same_cos = np.array([e["cos"] for e in same_pairs])
    cross_cos = np.array([e["cos"] for e in cross_pairs])

    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"  {'':<24} {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'std':>8}")
    print(f"  {'релевантные':<24} {same_cos.mean():>8.3f} {np.median(same_cos):>8.3f}"
          f" {same_cos.min():>8.3f} {same_cos.max():>8.3f} {same_cos.std():>8.3f}")
    print(f"  {'нерелевантные':<24} {cross_cos.mean():>8.3f} {np.median(cross_cos):>8.3f}"
          f" {cross_cos.min():>8.3f} {cross_cos.max():>8.3f} {cross_cos.std():>8.3f}")
    print()
    print(f"  Разница средних:               {same_cos.mean() - cross_cos.mean():+.3f}")
    print(f"  min(same)={same_cos.min():.3f}, max(cross)={cross_cos.max():.3f}"
          f"  → {'ЕСТЬ' if same_cos.min() < cross_cos.max() else 'НЕТ'} перекрытия")

    # NDCG-style: для каждой персоны, доля «своих» докладов в её топ-4
    n_per_cat = 4
    correct_in_top4 = 0
    total = 0
    for i, p in enumerate(personas):
        order = np.argsort(-sim[i])  # убывание
        top4 = order[:n_per_cat]
        for j in top4:
            if talks[j]["category"] == p["category"]:
                correct_in_top4 += 1
        total += n_per_cat
    print(f"  Precision@4: {correct_in_top4}/{total} = {correct_in_top4/total:.2%}"
          f"  (идеально 100%, случайно 20%)")
    print()

    same_sorted = sorted(same_pairs, key=lambda e: e["cos"], reverse=True)
    cross_sorted = sorted(cross_pairs, key=lambda e: e["cos"], reverse=True)
    print("  Худшая релевантная пара:")
    e = same_sorted[-1]
    print(f"    cos={e['cos']:.3f}  [{e['p_cat']}]  {e['p_role']}  →  «{e['t_title']}»")
    print("  Лучшая нерелевантная пара (ловушка):")
    e = cross_sorted[0]
    print(f"    cos={e['cos']:.3f}  [{e['p_cat']}→{e['t_cat']}]  {e['p_role']}  →  «{e['t_title']}»")
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", action="store_true", help="Use extended dataset (50+50)")
    args = ap.parse_args()

    suffix = "_ext" if args.ext else ""
    with open(TEST_DIR / f"personas{suffix}.json", encoding="utf-8") as f:
        personas = json.load(f)
    with open(TEST_DIR / f"talks{suffix}.json", encoding="utf-8") as f:
        talks = json.load(f)
    print(f"Dataset: {'EXTENDED 50+50' if args.ext else 'BASE 20+20'}")
    print(f"Загружено: {len(personas)} персон, {len(talks)} докладов")
    print(f"Категории персон: {sorted({p['category'] for p in personas})}")
    print()

    p_texts = [p["background"] for p in personas]
    t_texts = [make_talk_text(t) for t in talks]

    print("Эмбеддинг персон (kind='query')...")
    p_emb = embed_texts(p_texts, kind="query")
    print(f"  shape={p_emb.shape}")
    print("Эмбеддинг докладов (kind='passage')...")
    t_emb = embed_texts(t_texts, kind="passage")
    print(f"  shape={t_emb.shape}")
    print()

    # ============== ЭТАП 2: BASELINE ==============
    pe = normalize(p_emb)
    te = normalize(t_emb)
    sim_baseline = pe @ te.T
    report(personas, talks, sim_baseline, "ЭТАП 2 — BASELINE: чистая косинусная близость e5-small")

    # ============== ЭТАП 3: ALL-BUT-THE-TOP ==============
    # Объединяем все эмбеддинги (персоны + доклады) в один пул и считаем PCA по нему,
    # чтобы найти и удалить общее «доменное направление».
    all_emb = np.vstack([p_emb, t_emb])
    for n_comp in (1, 2, 3):
        all_processed = all_but_the_top(all_emb, n_components=n_comp)
        p_proc = all_processed[: len(personas)]
        t_proc = all_processed[len(personas):]
        pe_proc = normalize(p_proc)
        te_proc = normalize(t_proc)
        sim_abtt = pe_proc @ te_proc.T
        report(personas, talks, sim_abtt,
               f"ЭТАП 3 — ALL-BUT-THE-TOP (удалено топ-{n_comp} PCA-направлений)")

    # ============== ЭТАП 4: BM25 ==============
    # Тексты для BM25 — те же, что для эмбеддингов (background + title+abstract),
    # но без префиксов query:/passage: и без e5-обработки.
    p_tokens = [tokenize(p["background"]) for p in personas]
    t_tokens = [tokenize(make_talk_text(t).replace("[", " ").replace("]", " "))
                for t in talks]
    bm25 = BM25(t_tokens)
    bm25_raw = bm25.score_matrix(p_tokens)

    # Min-max нормализация для удобства сравнения
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)
    report(personas, talks, bm25_norm, "ЭТАП 4 — BM25 (Okapi, без лемматизации, нормализован в [0,1])")

    # ============== ЭТАП 5: ГИБРИД cos+ABTT-1 + BM25 ==============
    # Нормализуем cos после ABTT-1 в [0,1] глобально, BM25 уже в [0,1].
    all_processed = all_but_the_top(all_emb, n_components=1)
    p_proc = normalize(all_processed[: len(personas)])
    t_proc = normalize(all_processed[len(personas):])
    sim_abtt1 = p_proc @ t_proc.T
    sim_abtt1_norm = (sim_abtt1 - sim_abtt1.min()) / (sim_abtt1.max() - sim_abtt1.min() + 1e-12)

    for alpha in (0.3, 0.5, 0.7, 0.9):
        hybrid = alpha * sim_abtt1_norm + (1 - alpha) * bm25_norm
        report(personas, talks, hybrid,
               f"ЭТАП 5 — ГИБРИД α·ABTT-1 + (1−α)·BM25 (α={alpha})")

    best_alpha = 0.7
    hybrid_best = best_alpha * sim_abtt1_norm + (1 - best_alpha) * bm25_norm

    # ============== ЭТАП 6: CROSS-ENCODER RERANKER ==============
    print()
    print("=" * 78)
    print("Загрузка cross-encoder reranker (BGE-reranker-v2-m3, multilingual)...")
    print("=" * 78)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=False)

    pairs = []
    for p in personas:
        for t in talks:
            pairs.append((p["background"], make_talk_text(t)))
    print(f"Cкоринг {len(pairs)} пар (CPU, может занять минуту)...")
    rerank_scores = reranker.predict(pairs, show_progress_bar=True)
    rerank_matrix = np.array(rerank_scores).reshape(len(personas), len(talks))
    rerank_norm = (rerank_matrix - rerank_matrix.min()) / (rerank_matrix.max() - rerank_matrix.min() + 1e-12)

    report(personas, talks, rerank_norm,
           "ЭТАП 6 — CROSS-ENCODER RERANKER (BGE-reranker-v2-m3)")

    # ============== ПОДРОБНЫЙ ПРИМЕР: всё пять метрик ==============
    print("=" * 78)
    print("Подробный пример: iOS Senior Developer vs все 20 докладов")
    print("=" * 78)
    ios_idx = next(i for i, p in enumerate(personas) if p["category"] == "iOS")
    p = personas[ios_idx]
    print(f"Персона: {p['role']} ({p['id']})")
    print()
    print(f"  {'':<2} {'cos':>8} {'ABTT-1':>8} {'BM25':>8} {'HYBRID':>8} {'RERANK':>8}  category   title")
    rows = []
    for j, t in enumerate(talks):
        rows.append((
            "★" if t["category"] == "iOS" else " ",
            sim_baseline[ios_idx, j],
            sim_abtt1[ios_idx, j],
            bm25_norm[ios_idx, j],
            hybrid_best[ios_idx, j],
            rerank_norm[ios_idx, j],
            t["category"],
            t["title"],
        ))
    rows.sort(key=lambda r: -r[5])  # сортировка по reranker
    for marker, b, a, bm, h, r, cat, title in rows:
        print(f"  {marker} {b:>8.3f} {a:>8.3f} {bm:>8.3f} {h:>8.3f} {r:>8.3f}  [{cat:<8}]  «{title}»")


if __name__ == "__main__":
    main()
