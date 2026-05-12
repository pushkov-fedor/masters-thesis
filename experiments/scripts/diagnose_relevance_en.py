"""Диагностика функции релевантности на английской версии тестовой выборки.

Гипотеза: на переведённых на английский текстах с мощным английским эмбеддером
(BGE-large-en-v1.5, 1024-мерный) разделимость функции релевантности должна быть
лучше, чем на русских текстах с e5-small (384-мерный multilingual).

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_relevance_en.py
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

TEST_DIR = ROOT / "data" / "personas" / "test_diversity"

# BGE-large-en-v1.5 рекомендует prompt для query, passage — без prompt
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.doc_lens = np.array([len(d) for d in docs])
        self.avgdl = self.doc_lens.mean() if len(docs) > 0 else 0.0
        self.doc_freqs = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score_matrix(self, queries):
        n_q, n_d = len(queries), len(self.docs)
        out = np.zeros((n_q, n_d))
        for i, q in enumerate(queries):
            for j in range(n_d):
                freqs = self.doc_freqs[j]
                dl = self.doc_lens[j]
                s = 0.0
                for term in q:
                    if term not in self.idf:
                        continue
                    f = freqs.get(term, 0)
                    if f == 0:
                        continue
                    s += self.idf[term] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
                out[i, j] = s
        return out


def all_but_the_top(vecs, n_components=1):
    mu = vecs.mean(axis=0, keepdims=True)
    c = vecs - mu
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    top = Vt[:n_components]
    return c - c @ top.T @ top


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def report(personas, talks, sim, label):
    same, cross = [], []
    for i, p in enumerate(personas):
        for j, t in enumerate(talks):
            (same if p["category"] == t["category"] else cross).append(float(sim[i, j]))
    same, cross = np.array(same), np.array(cross)

    correct = sum(
        1 for i, p in enumerate(personas)
        for j in np.argsort(-sim[i])[:4]
        if talks[j]["category"] == p["category"]
    )
    total = len(personas) * 4
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"  {'':<24} {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'std':>8}")
    print(f"  {'релевантные':<24} {same.mean():>8.3f} {np.median(same):>8.3f}"
          f" {same.min():>8.3f} {same.max():>8.3f} {same.std():>8.3f}")
    print(f"  {'нерелевантные':<24} {cross.mean():>8.3f} {np.median(cross):>8.3f}"
          f" {cross.min():>8.3f} {cross.max():>8.3f} {cross.std():>8.3f}")
    print(f"  Разница средних: {same.mean() - cross.mean():+.3f}")
    print(f"  min(same)={same.min():.3f}, max(cross)={cross.max():.3f}"
          f"  → {'ЕСТЬ' if same.min() < cross.max() else 'НЕТ'} перекрытия")
    print(f"  Precision@4: {correct}/{total} = {correct/total:.2%}")
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", action="store_true", help="Use extended dataset (50+50)")
    args = ap.parse_args()

    suffix = "_ext" if args.ext else ""
    with open(TEST_DIR / f"personas{suffix}_en.json") as f:
        personas = json.load(f)
    with open(TEST_DIR / f"talks{suffix}_en.json") as f:
        talks = json.load(f)
    print(f"Dataset: {'EXTENDED 50+50' if args.ext else 'BASE 20+20'} (EN)")
    print(f"Загружено: {len(personas)} персон (EN), {len(talks)} докладов (EN)")
    print()

    p_texts_raw = [p["background"] for p in personas]
    t_texts = [make_talk_text(t) for t in talks]
    p_texts_for_emb = [QUERY_PROMPT + t for t in p_texts_raw]

    print("Загрузка эмбеддера BGE-large-en-v1.5 (~1.3GB, при первом запуске скачается)...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    print("Эмбеддинг персон...")
    p_emb = embedder.encode(p_texts_for_emb, normalize_embeddings=True, show_progress_bar=True)
    print("Эмбеддинг докладов...")
    t_emb = embedder.encode(t_texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Размерность: persona {p_emb.shape}, talks {t_emb.shape}")
    print()

    # ============== ЭТАП 2: BASELINE COSINE ==============
    pe, te = normalize(p_emb), normalize(t_emb)
    sim_baseline = pe @ te.T
    report(personas, talks, sim_baseline, "ЭТАП 2 — BASELINE: cos через BGE-large-en-v1.5")

    # ============== ЭТАП 3: ABTT-1 ==============
    all_emb = np.vstack([p_emb, t_emb])
    proc = all_but_the_top(all_emb, n_components=1)
    pe_a, te_a = normalize(proc[:len(personas)]), normalize(proc[len(personas):])
    sim_abtt1 = pe_a @ te_a.T
    report(personas, talks, sim_abtt1, "ЭТАП 3 — ABTT-1 на BGE-large-en")

    # ============== ЭТАП 4: BM25 ==============
    p_tokens = [tokenize(p["background"]) for p in personas]
    t_tokens = [tokenize(make_talk_text(t).replace("[", " ").replace("]", " ")) for t in talks]
    bm25 = BM25(t_tokens)
    bm25_raw = bm25.score_matrix(p_tokens)
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)
    report(personas, talks, bm25_norm, "ЭТАП 4 — BM25 (английский, без лемматизации)")

    # ============== ЭТАП 5: ГИБРИД α=0.7 ==============
    sim_abtt1_norm = (sim_abtt1 - sim_abtt1.min()) / (sim_abtt1.max() - sim_abtt1.min() + 1e-12)
    for alpha in (0.5, 0.7, 0.9):
        hybrid = alpha * sim_abtt1_norm + (1 - alpha) * bm25_norm
        report(personas, talks, hybrid, f"ЭТАП 5 — ГИБРИД α·ABTT + (1−α)·BM25 (α={alpha})")

    # ============== ЭТАП 6: RERANKER ==============
    print("Загрузка cross-encoder reranker (BGE-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(p_texts_raw[i], t_texts[j]) for i in range(len(personas)) for j in range(len(talks))]
    print(f"Скоринг {len(pairs)} пар...")
    scores = reranker.predict(pairs, show_progress_bar=True)
    rerank_matrix = np.array(scores).reshape(len(personas), len(talks))
    rerank_norm = (rerank_matrix - rerank_matrix.min()) / (rerank_matrix.max() - rerank_matrix.min() + 1e-12)
    report(personas, talks, rerank_norm, "ЭТАП 6 — CROSS-ENCODER RERANKER")

    # ============== Подробный пример: iOS Senior vs все 20 ==============
    print("=" * 78)
    print("Подробный пример: iOS Senior Developer vs все 20 докладов (EN)")
    print("=" * 78)
    ios_idx = next(i for i, p in enumerate(personas) if p["category"] == "iOS")
    p = personas[ios_idx]
    hybrid_07 = 0.7 * sim_abtt1_norm + 0.3 * bm25_norm
    print(f"Персона: {p['role']} ({p['id']})")
    print()
    print(f"  {'':<2} {'cos':>8} {'ABTT-1':>8} {'BM25':>8} {'HYBRID':>8} {'RERANK':>8}  category   title")
    rows = []
    for j, t in enumerate(talks):
        rows.append((
            "★" if t["category"] == "iOS" else " ",
            sim_baseline[ios_idx, j], sim_abtt1[ios_idx, j],
            bm25_norm[ios_idx, j], hybrid_07[ios_idx, j], rerank_norm[ios_idx, j],
            t["category"], t["title"],
        ))
    rows.sort(key=lambda r: -r[4])
    for marker, b, a, bm, h, r, cat, title in rows:
        print(f"  {marker} {b:>8.3f} {a:>8.3f} {bm:>8.3f} {h:>8.3f} {r:>8.3f}  [{cat:<8}]  «{title}»")


if __name__ == "__main__":
    main()
