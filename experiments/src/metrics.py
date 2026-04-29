"""Метрики качества рекомендательной политики из §2.5 Главы 2.

1. overflow_rate — доля (slot, hall) пар с переполнением (occupied > capacity).
2. hall_utilization_variance — средняя по слотам дисперсия загрузки залов в слоте.
3. mean_user_utility — средняя релевантность фактически выбранных докладов
   (учёт отказов через 0).
4. gini_coefficient — коэффициент Джини по распределению посещений между залами.

Дополнительно:
- skip_rate — доля шагов, на которых пользователь отказался.
- mean_hall_overload_excess — среднее превышение вместимости (по слотам с переполнением).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .simulator import Conference, SimResult


def overflow_rate(conf: Conference, result: SimResult) -> float:
    """Доля (slot, hall) пар, где occupied > capacity."""
    overfull = 0
    total = 0
    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        # Только залы, в которых в данном слоте есть доклад
        halls_in_slot = {conf.talks[tid].hall for tid in slot.talk_ids}
        for hid in halls_in_slot:
            cap = conf.halls[hid].capacity
            occ = result.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
            total += 1
            if occ > cap:
                overfull += 1
    return overfull / max(1, total)


def hall_utilization_variance(conf: Conference, result: SimResult) -> float:
    """Средняя по слотам дисперсия загрузки (occupied/capacity) залов внутри слота."""
    variances = []
    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        halls_in_slot = sorted({conf.talks[tid].hall for tid in slot.talk_ids})
        if len(halls_in_slot) < 2:
            continue
        loads = []
        for hid in halls_in_slot:
            cap = conf.halls[hid].capacity
            occ = result.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
            loads.append(occ / max(1.0, cap))
        variances.append(float(np.var(loads)))
    if not variances:
        return 0.0
    return float(np.mean(variances))


def mean_user_utility(result: SimResult) -> float:
    """Средняя релевантность по всем шагам (отказ считается как 0)."""
    if not result.steps:
        return 0.0
    return float(np.mean([s.chosen_relevance for s in result.steps]))


def skip_rate(result: SimResult) -> float:
    if not result.steps:
        return 0.0
    return float(np.mean([1.0 if s.chosen is None else 0.0 for s in result.steps]))


def gini_coefficient(values: np.ndarray) -> float:
    """Коэффициент Джини для неотрицательного вектора (загрузка залов)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    if np.allclose(v, 0):
        return 0.0
    v = np.sort(v)
    n = v.size
    # формула Джини
    cumv = np.cumsum(v)
    return float((n + 1 - 2 * (cumv.sum() / cumv[-1])) / n)


def hall_load_gini(conf: Conference, result: SimResult) -> float:
    """Джини по агрегированной загрузке залов (суммарной за всю конференцию)."""
    by_hall: Dict[int, int] = {h.id: 0 for h in conf.halls.values()}
    for sid, hloads in result.hall_load_per_slot.items():
        for hid, n in hloads.items():
            by_hall[hid] = by_hall.get(hid, 0) + n
    return gini_coefficient(np.array(list(by_hall.values()), dtype=np.float64))


def mean_hall_overload_excess(conf: Conference, result: SimResult) -> float:
    """Среднее по слотам максимальное превышение вместимости (occupied-capacity)/capacity, ≥0."""
    excesses = []
    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        halls_in_slot = {conf.talks[tid].hall for tid in slot.talk_ids}
        max_excess = 0.0
        for hid in halls_in_slot:
            cap = conf.halls[hid].capacity
            occ = result.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
            if occ > cap:
                max_excess = max(max_excess, (occ - cap) / max(1.0, cap))
        excesses.append(max_excess)
    return float(np.mean(excesses)) if excesses else 0.0


def compute_all(conf: Conference, result: SimResult) -> dict:
    return {
        "overflow_rate": overflow_rate(conf, result),
        "hall_utilization_variance": hall_utilization_variance(conf, result),
        "mean_user_utility": mean_user_utility(result),
        "hall_load_gini": hall_load_gini(conf, result),
        "skip_rate": skip_rate(result),
        "mean_overload_excess": mean_hall_overload_excess(conf, result),
    }
