"""Качество пула синтетических персон Mobius (50 шт., новый пул).

Считает три метрики качества аудитории:
1. Vendi Score — разнообразие пула в эмбеддинговом пространстве.
2. Coverage программы Mobius — сколько персон потенциально заинтересованы
   в каждом докладе. Считается двумя метриками близости: чистый cos и
   гибрид cos+BM25 (α=0.7).
3. Распределения по структурным полям (experience, company_size, preferred_topics).

Internal consistency (4-я метрика) считается отдельным subagent-LLM-judge.

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_mobius_personas_quality.py
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

TD = ROOT / "data" / "personas" / "test_diversity"

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text):
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
                    if term in self.idf and freqs.get(term, 0) > 0:
                        f = freqs[term]
                        s += self.idf[term] * f * (self.k1 + 1) / (
                            f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                        )
                out[i, j] = s
        return out


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def vendi_score(vecs):
    n = vecs.shape[0]
    e = normalize(vecs)
    K = (e @ e.T) / n
    eigvals = np.linalg.eigvalsh(K).clip(0.0, None)
    eigvals = eigvals[eigvals > 1e-10]
    eigvals = eigvals / eigvals.sum()
    H = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(H))


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def main():
    print("Загрузка данных...")
    with open(TD / "personas_mobius.json") as f:
        personas = json.load(f)
    with open(TD / "talks_mobius.json") as f:
        talks = json.load(f)
    print(f"  Персоны: {len(personas)}")
    print(f"  Доклады: {len(talks)}")
    print()

    print("Эмбеддинги через e5-small...")
    p_texts = [p["background"] for p in personas]
    t_texts = [make_talk_text(t) for t in talks]
    p_emb = embed_texts(p_texts, kind="query")
    t_emb = embed_texts(t_texts, kind="passage")
    print(f"  shape persona {p_emb.shape}, talks {t_emb.shape}")
    print()

    print("=" * 78)
    print("МЕТРИКА 1 — РАЗНООБРАЗИЕ ПУЛА (Vendi Score)")
    print("=" * 78)
    vs = vendi_score(p_emb)
    print(f"  Vendi Score:        {vs:.2f} из {len(personas)} максимума")
    print(f"  Доля:               {vs/len(personas):.1%}")
    print(f"  Интерпретация:      эффективно {vs:.1f} «существенно разных» персон")
    print()
    pe = normalize(p_emb)
    sim_pp = pe @ pe.T
    iu = np.triu_indices(len(personas), k=1)
    pairs = sim_pp[iu]
    print(f"  Парных косинусов:   {len(pairs)}")
    print(f"  Среднее:            {pairs.mean():.3f}")
    print(f"  Медиана:            {np.median(pairs):.3f}")
    print(f"  Min / Max:          {pairs.min():.3f} / {pairs.max():.3f}")
    print(f"  Квартили p25/p75:   {np.quantile(pairs, 0.25):.3f} / {np.quantile(pairs, 0.75):.3f}")
    print(f"  Пары > 0.95:        {(pairs > 0.95).sum()} (потенциальные дубликаты)")
    print()

    # Score-матрицы
    pe = normalize(p_emb)
    te = normalize(t_emb)
    sim_cos = pe @ te.T  # cos в [0, 1] примерно
    sim_cos_norm = (sim_cos - sim_cos.min()) / (sim_cos.max() - sim_cos.min() + 1e-12)

    p_tok = [tokenize(p["background"]) for p in personas]
    t_tok = [tokenize(make_talk_text(t).replace("[", " ").replace("]", " ")) for t in talks]
    bm25 = BM25(t_tok)
    bm25_raw = bm25.score_matrix(p_tok)
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)

    alpha = 0.7
    sim_hybrid = alpha * sim_cos_norm + (1 - alpha) * bm25_norm

    print("=" * 78)
    print(f"МЕТРИКА 2 — ПОКРЫТИЕ ПРОГРАММЫ MOBIUS")
    print("=" * 78)
    print(f"Для каждого доклада: сколько персон потенциально заинтересованы")
    print(f"(score выше порога τ). Доклад с покрытием 0 — «дыра» в аудитории.")
    print()

    for label, sim_mat in [("cos", sim_cos), ("гибрид cos+BM25 α=0.7", sim_hybrid)]:
        print(f"--- {label} ---")
        if label == "cos":
            taus = (0.70, 0.75, 0.80, 0.85)
        else:
            taus = (0.10, 0.15, 0.20, 0.25)
        for tau in taus:
            per_talk = (sim_mat >= tau).sum(axis=0)
            print(f"  τ ≥ {tau:.2f}: dead docs = {(per_talk==0).sum()}/{len(talks)}, "
                  f"<5 = {(per_talk<5).sum()}, "
                  f"min/median/max = {per_talk.min()}/{int(np.median(per_talk))}/{per_talk.max()}, "
                  f"mean = {per_talk.mean():.1f}")
        print()

    # Топ-5 «самых одиноких» докладов по гибриду при τ=0.20
    print("--- Доклады с наименьшей потенциальной аудиторией (по гибриду, τ=0.20) ---")
    per_talk_h = (sim_hybrid >= 0.20).sum(axis=0)
    rows = []
    for j, t in enumerate(talks):
        rows.append({
            "n_interested": int(per_talk_h[j]),
            "max_score": float(sim_hybrid[:, j].max()),
            "category": t.get("category", "?"),
            "title": t["title"],
        })
    rows.sort(key=lambda r: r["n_interested"])
    for r in rows[:5]:
        print(f"  [{r['n_interested']:>3} заинтер.]  [{r['category']:<20s}]  «{r['title']}»  (max={r['max_score']:.3f})")
    print()
    print("--- Доклады с наибольшей аудиторией (для сравнения) ---")
    for r in rows[-3:]:
        print(f"  [{r['n_interested']:>3} заинтер.]  [{r['category']:<20s}]  «{r['title']}»  (max={r['max_score']:.3f})")
    print()

    print("=" * 78)
    print("МЕТРИКА 3 — РАСПРЕДЕЛЕНИЯ ПО СТРУКТУРНЫМ ПОЛЯМ")
    print("=" * 78)
    for field in ["experience", "company_size", "role", "preferred_topics"]:
        print(f"  {field}:")
        cnt = Counter()
        for p in personas:
            v = p.get(field)
            if isinstance(v, list):
                for x in v:
                    cnt[x] += 1
            elif v is not None:
                cnt[str(v)] += 1
        total = sum(cnt.values())
        for k, v in cnt.most_common(15):
            print(f"    {k:<35s} {v:>3}  ({v/total:.0%})")
        if len(cnt) > 15:
            print(f"    … ещё {len(cnt) - 15} значений")
        print()


if __name__ == "__main__":
    main()
