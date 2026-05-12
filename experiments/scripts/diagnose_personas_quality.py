"""Диагностика качества пула синтетических персон.

Считает три метрики:
1. Разнообразие (Vendi Score) — эффективное число различных персон в пуле.
2. Покрытие программы — сколько персон потенциально заинтересованы в каждом докладе.
3. Распределения по структурным полям (experience, company_size, preferred_topics).

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/diagnose_personas_quality.py \
        --personas data/personas/personas_100.json \
        --personas-emb data/personas/personas_100_embeddings.npz \
        --talks data/conferences/mobius_2025_autumn.json \
        --talks-emb data/conferences/mobius_2025_autumn_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_data(personas_json, personas_emb, talks_json, talks_emb):
    with open(personas_json) as f:
        personas = json.load(f)
    p_npz = np.load(personas_emb)
    p_ids = list(p_npz["ids"])
    p_vec = p_npz["embeddings"].astype(np.float32)

    with open(talks_json) as f:
        conf = json.load(f)
    talks = conf["talks"]
    t_npz = np.load(talks_emb)
    t_ids = list(t_npz["ids"])
    t_vec = t_npz["embeddings"].astype(np.float32)

    p_id_to_idx = {pid: i for i, pid in enumerate(p_ids)}
    personas_aligned = [None] * len(p_ids)
    for p in personas:
        if p["id"] in p_id_to_idx:
            personas_aligned[p_id_to_idx[p["id"]]] = p
    personas_aligned = [p for p in personas_aligned if p is not None]

    t_id_to_idx = {tid: i for i, tid in enumerate(t_ids)}
    talks_aligned = [None] * len(t_ids)
    for t in talks:
        if t["id"] in t_id_to_idx:
            talks_aligned[t_id_to_idx[t["id"]]] = t
    talks_aligned = [t for t in talks_aligned if t is not None]

    return personas_aligned, p_vec, talks_aligned, t_vec


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def vendi_score(vecs: np.ndarray) -> tuple[float, dict]:
    """Vendi Score через собственные значения нормированной gram-матрицы.

    K = (1/n) * E @ E.T (для нормированных векторов это нормированный косинус).
    Vendi = exp(-Σ λ_i log λ_i), где λ_i — собств. значения K, Σ λ_i = 1.
    """
    n = vecs.shape[0]
    e = normalize(vecs)
    K = (e @ e.T) / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals.real, 0.0, None)
    eigvals = eigvals[eigvals > 1e-10]
    eigvals = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals * np.log(eigvals))
    vs = float(np.exp(entropy))
    return vs, {
        "n": n,
        "max_possible": n,
        "ratio": vs / n,
    }


def pairwise_cos_stats(vecs: np.ndarray) -> dict:
    e = normalize(vecs)
    sim = e @ e.T
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sims = sim[iu]
    return {
        "n_pairs": int(pair_sims.size),
        "cos_mean": float(pair_sims.mean()),
        "cos_median": float(np.median(pair_sims)),
        "cos_min": float(pair_sims.min()),
        "cos_max": float(pair_sims.max()),
        "cos_p25": float(np.quantile(pair_sims, 0.25)),
        "cos_p75": float(np.quantile(pair_sims, 0.75)),
        "near_duplicates_gt_0_95": int((pair_sims > 0.95).sum()),
        "near_duplicates_gt_0_99": int((pair_sims > 0.99).sum()),
    }


def coverage(p_vec: np.ndarray, t_vec: np.ndarray, thresholds=(0.70, 0.75, 0.80, 0.85)) -> dict:
    pe = normalize(p_vec)
    te = normalize(t_vec)
    sim = pe @ te.T  # (n_personas, n_talks)
    out = {"per_threshold": {}}
    for tau in thresholds:
        per_talk = (sim >= tau).sum(axis=0)
        out["per_threshold"][f"tau_{tau:.2f}"] = {
            "talks_with_zero_interested": int((per_talk == 0).sum()),
            "talks_with_lt_5_interested": int((per_talk < 5).sum()),
            "talks_with_lt_10_interested": int((per_talk < 10).sum()),
            "min_per_talk": int(per_talk.min()),
            "median_per_talk": float(np.median(per_talk)),
            "max_per_talk": int(per_talk.max()),
            "mean_per_talk": float(per_talk.mean()),
        }
    return out


def coverage_per_talk_table(p_vec, t_vec, talks, tau=0.75) -> list[dict]:
    pe = normalize(p_vec)
    te = normalize(t_vec)
    sim = pe @ te.T
    rows = []
    for j, t in enumerate(talks):
        rows.append({
            "talk_idx": j,
            "title": t.get("title", "")[:80],
            "category": t.get("category", "?"),
            "interested_count": int((sim[:, j] >= tau).sum()),
            "max_sim": float(sim[:, j].max()),
            "mean_sim": float(sim[:, j].mean()),
        })
    rows.sort(key=lambda r: r["interested_count"])
    return rows


def field_distribution(personas: list[dict], field: str) -> dict:
    cnt = Counter()
    for p in personas:
        v = p.get(field)
        if isinstance(v, list):
            for x in v:
                cnt[x] += 1
        elif v is not None:
            cnt[str(v)] += 1
    return dict(cnt.most_common())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", required=True)
    parser.add_argument("--personas-emb", required=True)
    parser.add_argument("--talks", required=True)
    parser.add_argument("--talks-emb", required=True)
    parser.add_argument("--coverage-tau", type=float, default=0.75)
    parser.add_argument("--show-bottom-talks", type=int, default=10)
    args = parser.parse_args()

    print(f"Загрузка данных…")
    personas, p_vec, talks, t_vec = load_data(
        args.personas, args.personas_emb, args.talks, args.talks_emb
    )
    print(f"  Персоны: {len(personas)}, эмбеддинг {p_vec.shape}")
    print(f"  Доклады: {len(talks)}, эмбеддинг {t_vec.shape}")
    print()

    print("=" * 78)
    print("1. РАЗНООБРАЗИЕ (Vendi Score + парные косинусы)")
    print("=" * 78)
    vs, vs_info = vendi_score(p_vec)
    print(f"  Vendi Score:        {vs:.2f} из {vs_info['max_possible']} максимума")
    print(f"  Доля от максимума:  {vs_info['ratio']:.2%}")
    print(f"  Интерпретация:      эффективно {vs:.0f} «существенно разных» персон")
    print()
    pst = pairwise_cos_stats(p_vec)
    print(f"  Парных косинусов:   {pst['n_pairs']}")
    print(f"  Среднее:            {pst['cos_mean']:.3f}")
    print(f"  Медиана:            {pst['cos_median']:.3f}")
    print(f"  Минимум:            {pst['cos_min']:.3f}")
    print(f"  Максимум:           {pst['cos_max']:.3f}")
    print(f"  Квартили p25/p75:   {pst['cos_p25']:.3f} / {pst['cos_p75']:.3f}")
    print(f"  Пары с cos > 0.95:  {pst['near_duplicates_gt_0_95']} (потенциальные дубликаты)")
    print(f"  Пары с cos > 0.99:  {pst['near_duplicates_gt_0_99']} (почти дубликаты)")
    print()

    print("=" * 78)
    print("2. ПОКРЫТИЕ ПРОГРАММЫ MOBIUS")
    print("=" * 78)
    cov = coverage(p_vec, t_vec)
    for tau_key, st in cov["per_threshold"].items():
        tau = float(tau_key.split("_")[1])
        print(f"  Порог cos >= {tau:.2f}:")
        print(f"    Докладов с 0 заинтересованных:     {st['talks_with_zero_interested']} из {len(talks)}")
        print(f"    Докладов с < 5 заинтересованных:   {st['talks_with_lt_5_interested']} из {len(talks)}")
        print(f"    Докладов с < 10 заинтересованных:  {st['talks_with_lt_10_interested']} из {len(talks)}")
        print(f"    Min / median / max / mean:         {st['min_per_talk']} / {st['median_per_talk']:.1f} / {st['max_per_talk']} / {st['mean_per_talk']:.1f}")
        print()

    print(f"  --- Топ {args.show_bottom_talks} «самых одиноких» докладов при пороге {args.coverage_tau:.2f} ---")
    rows = coverage_per_talk_table(p_vec, t_vec, talks, tau=args.coverage_tau)
    for r in rows[:args.show_bottom_talks]:
        print(f"    [{r['interested_count']:3d}] [{r['category']:<25s}] {r['title']}")
        print(f"          max_sim={r['max_sim']:.3f}, mean_sim={r['mean_sim']:.3f}")
    print()

    print("=" * 78)
    print("3. РАСПРЕДЕЛЕНИЯ ПО СТРУКТУРНЫМ ПОЛЯМ")
    print("=" * 78)
    for field in ["experience", "company_size", "preferred_topics"]:
        print(f"  {field}:")
        dist = field_distribution(personas, field)
        total = sum(dist.values())
        for k, v in dist.items():
            print(f"    {k:<35s} {v:4d}  ({v/total:.1%})")
        print()


if __name__ == "__main__":
    main()
