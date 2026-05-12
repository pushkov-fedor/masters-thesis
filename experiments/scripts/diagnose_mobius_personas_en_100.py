"""Расширенный аудит пула 100 синтетических EN-персон Mobius на BGE-large-en + ABTT.

Аналог `diagnose_mobius_personas_quality.py`, но:
- работает на пуле 100 EN-персон (вторая половина — догенерация под предзащиту);
- использует BGE-large-en (1024-dim) + ABTT-1 как в основном эксперименте;
- считает Vendi Score под 4 kernels (raw cos, ABTT cos, BM25, hybrid α=0.7)
  для сопоставления с цифрами spike_relevance_function_audit.md;
- проверяет coverage программы Mobius 2025 Autumn (40 EN-докладов);
- печатает распределения по структурным полям.

Internal consistency для новой половины (51..100) считается отдельным subagent-LLM-judge.

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_mobius_personas_en_100.py
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


def vendi_score_from_K(K: np.ndarray) -> float:
    """Vendi Score по матрице сходств K (симметричная, ≈ Gram).

    K делится на n; eigenvalues нормализуются в распределение; Vendi = exp(H).
    Принимает K — матрицу попарных сходств (через cos, BM25 или иное).
    """
    n = K.shape[0]
    K = K / n
    eigvals = np.linalg.eigvalsh(K).clip(0.0, None)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
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
    print("Загрузка персон и программы...")
    with open(ROOT / "data/personas/personas_mobius_en.json", encoding="utf-8") as f:
        personas = json.load(f)
    with open(ROOT / "data/conferences/mobius_2025_autumn_en.json", encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]
    print(f"  Персоны: {len(personas)}")
    print(f"  Доклады: {len(talks)}")
    print()

    print("Загрузка ABTT и raw эмбеддингов из .npz (BGE-large-en + ABTT-1)...")
    pers_npz = np.load(ROOT / "data/personas/personas_mobius_en_embeddings.npz",
                       allow_pickle=False)
    talk_npz = np.load(ROOT / "data/conferences/mobius_2025_autumn_en_embeddings.npz",
                       allow_pickle=False)
    pers_ids_npz = list(pers_npz["ids"])
    talk_ids_npz = list(talk_npz["ids"])

    # Match by id (orders могут отличаться)
    pers_id_idx = {pid: i for i, pid in enumerate(pers_ids_npz)}
    talk_id_idx = {tid: i for i, tid in enumerate(talk_ids_npz)}
    p_idx = np.array([pers_id_idx[p["id"]] for p in personas])
    t_idx = np.array([talk_id_idx[t["id"]] for t in talks])

    p_abtt = pers_npz["embeddings"][p_idx]
    t_abtt = talk_npz["embeddings"][t_idx]
    p_raw = pers_npz["embeddings_raw"][p_idx]
    t_raw = talk_npz["embeddings_raw"][t_idx]
    print(f"  p_abtt {p_abtt.shape}, t_abtt {t_abtt.shape}")
    print(f"  p_raw  {p_raw.shape},  t_raw  {t_raw.shape}")
    print()

    # ============== Vendi Score под разными kernels ==============
    print("=" * 78)
    print("МЕТРИКА 1 — РАЗНООБРАЗИЕ ПУЛА (Vendi Score под разными kernels)")
    print("=" * 78)
    n = len(personas)

    # Kernel 1: raw cos через BGE-large-en (до ABTT)
    pe_raw = normalize(p_raw)
    K_raw = pe_raw @ pe_raw.T
    v_raw = vendi_score_from_K(K_raw)
    print(f"  cos raw (BGE без ABTT):        Vendi = {v_raw:6.2f}  ({v_raw/n:.1%})")

    # Kernel 2: cos после ABTT-1
    pe_abtt = normalize(p_abtt)
    K_abtt = pe_abtt @ pe_abtt.T
    v_abtt = vendi_score_from_K(K_abtt)
    print(f"  cos + ABTT-1 (основной):       Vendi = {v_abtt:6.2f}  ({v_abtt/n:.1%})")

    # Kernel 3: BM25 (только лексика)
    p_tok = [tokenize(p["background"]) for p in personas]
    bm25 = BM25(p_tok)
    bm25_raw = bm25.score_matrix(p_tok)
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)
    v_bm25 = vendi_score_from_K(bm25_norm)
    print(f"  BM25 (только лексика):         Vendi = {v_bm25:6.2f}  ({v_bm25/n:.1%})")

    # Kernel 4: hybrid α=0.7 (cos+ABTT) + (1-α) BM25
    K_abtt_norm = (K_abtt - K_abtt.min()) / (K_abtt.max() - K_abtt.min() + 1e-12)
    hybrid = 0.7 * K_abtt_norm + 0.3 * bm25_norm
    v_hyb = vendi_score_from_K(hybrid)
    print(f"  hybrid α=0.7 (ABTT+BM25):      Vendi = {v_hyb:6.2f}  ({v_hyb/n:.1%})")
    print()

    # Пары
    iu = np.triu_indices(n, k=1)
    pairs_raw = K_raw[iu]
    pairs_abtt = K_abtt[iu]
    print("  Парные косинусы (раздача персон в пространстве BGE):")
    print(f"    raw   mean={pairs_raw.mean():.3f}  median={np.median(pairs_raw):.3f}  "
          f"min={pairs_raw.min():.3f}  max={pairs_raw.max():.3f}  "
          f"std={pairs_raw.std():.3f}")
    print(f"    ABTT  mean={pairs_abtt.mean():.3f}  median={np.median(pairs_abtt):.3f}  "
          f"min={pairs_abtt.min():.3f}  max={pairs_abtt.max():.3f}  "
          f"std={pairs_abtt.std():.3f}")
    print(f"    Пары > 0.95 (потенциальные дубликаты):  "
          f"raw {(pairs_raw > 0.95).sum()},  ABTT {(pairs_abtt > 0.95).sum()}")
    print()

    # ============== Coverage программы ==============
    print("=" * 78)
    print(f"МЕТРИКА 2 — ПОКРЫТИЕ ПРОГРАММЫ MOBIUS ({len(talks)} EN-докладов)")
    print("=" * 78)

    te_abtt = normalize(t_abtt)
    sim_pt = pe_abtt @ te_abtt.T  # ABTT cos
    sim_pt_norm = (sim_pt - sim_pt.min()) / (sim_pt.max() - sim_pt.min() + 1e-12)
    print(f"  cos(ABTT) range: [{sim_pt.min():.3f}, {sim_pt.max():.3f}], "
          f"mean={sim_pt.mean():.3f}")
    print()
    print("--- ABTT cos (нормировано [0, 1] по всему пулу пар) ---")
    for tau in (0.4, 0.5, 0.6, 0.7):
        per_talk = (sim_pt_norm >= tau).sum(axis=0)
        print(f"  τ ≥ {tau:.2f}: dead docs = {(per_talk==0).sum()}/{len(talks)}, "
              f"<5 = {(per_talk<5).sum()}, "
              f"min/median/max = {per_talk.min()}/{int(np.median(per_talk))}/{per_talk.max()}, "
              f"mean = {per_talk.mean():.1f}")
    print()

    # Топ-5 самых одиноких докладов и топ-3 самых популярных
    per_talk = (sim_pt_norm >= 0.6).sum(axis=0)
    rows = []
    for j, t in enumerate(talks):
        rows.append({
            "n_interested": int(per_talk[j]),
            "max_score": float(sim_pt_norm[:, j].max()),
            "category": t.get("category", "?"),
            "title": t["title"],
        })
    rows.sort(key=lambda r: r["n_interested"])
    print("--- Доклады с наименьшей потенциальной аудиторией (τ=0.60) ---")
    for r in rows[:5]:
        print(f"  [{r['n_interested']:>3} заинтер.] [{r['category']:<20s}] «{r['title'][:80]}» (max={r['max_score']:.3f})")
    print()
    print("--- Доклады с наибольшей аудиторией ---")
    for r in rows[-3:]:
        print(f"  [{r['n_interested']:>3} заинтер.] [{r['category']:<20s}] «{r['title'][:80]}» (max={r['max_score']:.3f})")
    print()

    # ============== Distributions ==============
    print("=" * 78)
    print("МЕТРИКА 3 — РАСПРЕДЕЛЕНИЯ ПО СТРУКТУРНЫМ ПОЛЯМ (n=100)")
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
        for k, v in cnt.most_common(20):
            print(f"    {k:<40s} {v:>3}  ({v/total:.0%})")
        if len(cnt) > 20:
            print(f"    … ещё {len(cnt) - 20} значений")
        print()

    # Compare first vs second half
    print("--- Сопоставление пулов 1–50 vs 51–100 (sanity check distribution-shift) ---")
    for field in ["experience", "company_size"]:
        first50 = Counter()
        second50 = Counter()
        for i, p in enumerate(personas):
            v = p.get(field)
            target = first50 if i < 50 else second50
            if isinstance(v, list):
                for x in v:
                    target[x] += 1
            elif v is not None:
                target[str(v)] += 1
        all_keys = sorted(set(first50) | set(second50))
        print(f"  {field}:")
        for k in all_keys:
            a, b = first50.get(k, 0), second50.get(k, 0)
            print(f"    {k:<25s}  first50={a:>2}  second50={b:>2}")
        print()


if __name__ == "__main__":
    main()
