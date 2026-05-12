"""Аудит пула 150 EN-персон Demo Day 2026 на BGE-large-en + ABTT-1.

Аналог `diagnose_mobius_personas_en_100.py`, но для второго инстанса
(Demo Day 2026, 210 talks, broad-IT).

Метрики:
1. Vendi Score под 4 kernels (raw cos, ABTT cos, BM25, hybrid α=0.7).
2. Coverage программы: dead docs / <5 / mean — под τ от 0.4 до 0.7
   на нормированных по пулу пар ABTT cos.
3. Распределения по полям (experience, company_size, role,
   preferred_topics) — фактические vs целевые.

Internal consistency считается отдельным скриптом
`audit_demoday_personas_consistency.py`.

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_demoday_personas_en_150.py
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
    """Vendi Score по матрице сходств K (симметричная, ≈ Gram)."""
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
    with open(ROOT / "data/personas/personas_demoday_en.json", encoding="utf-8") as f:
        personas = json.load(f)
    with open(ROOT / "data/conferences/demo_day_2026_en.json", encoding="utf-8") as f:
        prog = json.load(f)
    talks = prog["talks"]
    print(f"  Персоны: {len(personas)}")
    print(f"  Доклады: {len(talks)}")
    print()

    print("Загрузка ABTT и raw эмбеддингов из .npz (BGE-large-en + ABTT-1)...")
    pers_npz = np.load(ROOT / "data/personas/personas_demoday_en_embeddings.npz",
                       allow_pickle=False)
    talk_npz = np.load(ROOT / "data/conferences/demo_day_2026_en_embeddings.npz",
                       allow_pickle=False)
    pers_ids_npz = list(pers_npz["ids"])
    talk_ids_npz = list(talk_npz["ids"])

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

    pe_raw = normalize(p_raw)
    K_raw = pe_raw @ pe_raw.T
    v_raw = vendi_score_from_K(K_raw)
    print(f"  cos raw (BGE без ABTT):        Vendi = {v_raw:6.2f}  ({v_raw/n:.1%})")

    pe_abtt = normalize(p_abtt)
    K_abtt = pe_abtt @ pe_abtt.T
    v_abtt = vendi_score_from_K(K_abtt)
    print(f"  cos + ABTT-1 (основной):       Vendi = {v_abtt:6.2f}  ({v_abtt/n:.1%})")

    p_tok = [tokenize(p["background"]) for p in personas]
    bm25 = BM25(p_tok)
    bm25_raw = bm25.score_matrix(p_tok)
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)
    v_bm25 = vendi_score_from_K(bm25_norm)
    print(f"  BM25 (только лексика):         Vendi = {v_bm25:6.2f}  ({v_bm25/n:.1%})")

    K_abtt_norm = (K_abtt - K_abtt.min()) / (K_abtt.max() - K_abtt.min() + 1e-12)
    hybrid = 0.7 * K_abtt_norm + 0.3 * bm25_norm
    v_hyb = vendi_score_from_K(hybrid)
    print(f"  hybrid α=0.7 (ABTT+BM25):      Vendi = {v_hyb:6.2f}  ({v_hyb/n:.1%})")
    print()

    iu = np.triu_indices(n, k=1)
    pairs_raw = K_raw[iu]
    pairs_abtt = K_abtt[iu]
    print("  Парные косинусы:")
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
    print(f"МЕТРИКА 2 — ПОКРЫТИЕ ПРОГРАММЫ DEMO DAY ({len(talks)} EN-докладов)")
    print("=" * 78)

    te_abtt = normalize(t_abtt)
    sim_pt = pe_abtt @ te_abtt.T
    sim_pt_norm = (sim_pt - sim_pt.min()) / (sim_pt.max() - sim_pt.min() + 1e-12)
    print(f"  cos(ABTT) range: [{sim_pt.min():.3f}, {sim_pt.max():.3f}], "
          f"mean={sim_pt.mean():.3f}")
    print()
    print("--- ABTT cos (нормировано [0, 1] по всему пулу пар) ---")
    coverage_summary = {}
    for tau in (0.4, 0.5, 0.6, 0.7):
        per_talk = (sim_pt_norm >= tau).sum(axis=0)
        dead = int((per_talk == 0).sum())
        under5 = int((per_talk < 5).sum())
        coverage_summary[f"tau_{tau}"] = {
            "dead": dead, "under_5": under5,
            "min": int(per_talk.min()), "median": int(np.median(per_talk)),
            "max": int(per_talk.max()), "mean": float(per_talk.mean()),
        }
        print(f"  τ ≥ {tau:.2f}: dead docs = {dead}/{len(talks)}, "
              f"<5 = {under5}, "
              f"min/median/max = {per_talk.min()}/{int(np.median(per_talk))}/{per_talk.max()}, "
              f"mean = {per_talk.mean():.1f}")
    print()

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
        print(f"  [{r['n_interested']:>3} заинтер.] [{r['category'][:30]:<30s}] «{r['title'][:80]}» (max={r['max_score']:.3f})")
    print()
    print("--- Доклады с наибольшей аудиторией ---")
    for r in rows[-3:]:
        print(f"  [{r['n_interested']:>3} заинтер.] [{r['category'][:30]:<30s}] «{r['title'][:80]}» (max={r['max_score']:.3f})")
    print()

    # ============== Distributions ==============
    print("=" * 78)
    print(f"МЕТРИКА 3 — РАСПРЕДЕЛЕНИЯ ПО СТРУКТУРНЫМ ПОЛЯМ (n={len(personas)})")
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

    # Aggregate diagnostic into JSON for the report
    out = {
        "n_personas": len(personas),
        "n_talks": len(talks),
        "vendi": {
            "raw_cos": v_raw,
            "abtt_cos": v_abtt,
            "bm25": v_bm25,
            "hybrid_0_7": v_hyb,
            "raw_pct": v_raw / n,
            "abtt_pct": v_abtt / n,
            "bm25_pct": v_bm25 / n,
            "hybrid_pct": v_hyb / n,
        },
        "pairs": {
            "raw_mean": float(pairs_raw.mean()),
            "raw_max": float(pairs_raw.max()),
            "abtt_mean": float(pairs_abtt.mean()),
            "abtt_max": float(pairs_abtt.max()),
            "abtt_dup_95": int((pairs_abtt > 0.95).sum()),
        },
        "coverage": coverage_summary,
        "distributions": {
            field: dict(Counter(
                (p[field] if not isinstance(p[field], list) else tuple(p[field]))
                for p in personas
            ).most_common()) if field == "experience" or field == "company_size"
            else None
            for field in ["experience", "company_size"]
        },
    }
    # Filter Nones
    out["distributions"] = {k: v for k, v in out["distributions"].items() if v is not None}
    out_path = ROOT / "data/personas/test_diversity/diagnose_demoday_en.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
