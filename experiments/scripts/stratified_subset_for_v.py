"""Стратифицированный отбор 12 LHS-точек для LLM-симулятора V (план #35).

Берёт результат параметрического прогона, разбивает 50 LHS-точек на 4
корзины по `mean_overload_excess` (среднее по политикам), и из каждой
корзины отбирает по 3 точки с максимальной попарной дистанцией в
параметрическом пространстве (capacity_multiplier, w_rec, w_gossip).

Запуск:
    .venv/bin/python scripts/stratified_subset_for_v.py \\
        --input results/lhs_parametric_simplified_2026-05-12_mobius_2025_autumn_en.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Границы корзин по mean_overload_excess (среднее по политикам, по точке)
# Подобраны для случая, когда большинство точек risk-positive.
BUCKETS = [
    ("safe", 0.0, 0.001),
    ("light", 0.001, 0.05),
    ("moderate", 0.05, 0.2),
    ("severe", 0.2, float("inf")),
]
PER_BUCKET = 3


def collect_overload_per_point(rows: List[dict]) -> Dict[int, float]:
    """Для каждой LHS-точки возвращает среднее mean_overload_excess по политикам и репликам."""
    per_pt: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        per_pt[r["lhs_row_id"]].append(r["metric_mean_overload_excess"])
    return {pid: float(np.mean(vals)) for pid, vals in per_pt.items()}


def parametric_distance(u1: List[float], u2: List[float]) -> float:
    """Евклидова дистанция в unit-cube координатах (по трём варьируемым осям).

    Используем u_raw[0] = capacity, u_raw[2] = w_rec, u_raw[3] = w_gossip.
    """
    a = np.array([u1[0], u1[2], u1[3]], dtype=float)
    b = np.array([u2[0], u2[2], u2[3]], dtype=float)
    return float(np.linalg.norm(a - b))


def greedy_maximin(rows: List[dict], k: int) -> List[int]:
    """Жадный maximin: возвращает индексы k точек с максимальной попарной дистанцией."""
    if k <= 0 or len(rows) == 0:
        return []
    if len(rows) <= k:
        return list(range(len(rows)))
    # Начинаем с первой точки (детерминированно), дальше каждый раз добавляем
    # точку, у которой минимальная дистанция до уже выбранных максимальна.
    chosen = [0]
    while len(chosen) < k:
        best_i, best_dist = -1, -1.0
        for i in range(len(rows)):
            if i in chosen:
                continue
            min_d = min(
                parametric_distance(rows[i]["u_raw"], rows[c]["u_raw"])
                for c in chosen
            )
            if min_d > best_dist:
                best_dist = min_d
                best_i = i
        chosen.append(best_i)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_stratified_subset.json"
    )

    data = json.load(open(in_path))
    rows = data["results"]
    lhs_rows = data["lhs_rows"]
    lhs_by_id = {r["lhs_row_id"]: r for r in lhs_rows}

    overload_per_pt = collect_overload_per_point(rows)

    # Сгруппировать точки по корзинам
    buckets: Dict[str, List[Tuple[int, float]]] = {b[0]: [] for b in BUCKETS}
    for pid, ov in overload_per_pt.items():
        for name, lo, hi in BUCKETS:
            if lo <= ov < hi:
                buckets[name].append((pid, ov))
                break

    print("Распределение точек по корзинам:")
    for name, lo, hi in BUCKETS:
        pts = buckets[name]
        print(f"  {name:10s} [{lo:.3f}, {hi:.3f}): {len(pts)} точек")
    print()

    # Из каждой корзины выбираем PER_BUCKET точек жадным maximin'ом
    selected: List[Dict] = []
    for name, lo, hi in BUCKETS:
        pts_in_bucket = buckets[name]
        if not pts_in_bucket:
            print(f"WARN: bucket '{name}' пустой, пропускаем")
            continue
        bucket_rows = [lhs_by_id[pid] for pid, _ in pts_in_bucket]
        n_target = min(PER_BUCKET, len(bucket_rows))
        idx = greedy_maximin(bucket_rows, n_target)
        for j in idx:
            pid = bucket_rows[j]["lhs_row_id"]
            selected.append({
                "lhs_row_id": pid,
                "bucket": name,
                "mean_overload": overload_per_pt[pid],
                "capacity_multiplier": bucket_rows[j]["capacity_multiplier"],
                "w_rec": bucket_rows[j]["w_rec"],
                "w_gossip": bucket_rows[j]["w_gossip"],
            })

    selected.sort(key=lambda s: (s["bucket"], s["lhs_row_id"]))
    selected_ids = [s["lhs_row_id"] for s in selected]
    print(f"Выбрано {len(selected)} точек:")
    print(f"  IDs: {selected_ids}")
    print()
    print("Детальный список:")
    print(f"  {'pid':>4} {'bucket':10s} {'overload':>8} {'cap':>6} {'w_rec':>6} {'w_gossip':>8}")
    for s in selected:
        print(f"  {s['lhs_row_id']:>4} {s['bucket']:10s} "
              f"{s['mean_overload']:>8.4f} {s['capacity_multiplier']:>6.3f} "
              f"{s['w_rec']:>6.3f} {s['w_gossip']:>8.3f}")

    out = {
        "source": str(in_path),
        "method": "stratified_maximin_by_mean_overload_excess",
        "buckets": [{"name": n, "lo": lo, "hi": hi, "n_points": len(buckets[n])}
                    for n, lo, hi in BUCKETS],
        "per_bucket": PER_BUCKET,
        "selected": selected,
        "selected_lhs_row_ids": selected_ids,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
