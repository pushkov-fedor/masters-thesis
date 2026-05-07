"""Extreme-condition tests EC1–EC4 (PROJECT_DESIGN §11, традиция Sargent 2013).

Этап I PIVOT_IMPLEMENTATION_PLAN r5. Цель — автоматизированная верификация
четырёх обязательных свойств модели в граничных условиях:

EC1: capacity_multiplier ≥ 3.0 → mean_overload_excess == 0 для всех 4 политик.
EC2: монотонность mean_overload_excess по capacity_multiplier ∈ {0.5, 0.7,
     1.0, 1.5, 3.0} (усреднение по 5 seed).
EC3: w_rec → 0 → CV(metric, policies) < 5%.
EC4: w_rec → 1 → range(metric, policies) > 10× CV(EC3).

Тесты используют `toy_microconf_2slot` + первые 50 персон из `personas_100`
(скорость прогона). П4 (LLMRankerPolicy) исключена — требует сетевые вызовы;
EC проверяются на П1–П3.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import pytest

from src.metrics import mean_hall_overload_excess
from src.simulator import Conference, SimConfig, simulate


def _scale_capacity(conf: Conference, mult: float) -> Conference:
    cloned = copy.deepcopy(conf)
    for h in cloned.halls.values():
        h.capacity = max(1, int(round(h.capacity * mult)))
    for s in cloned.slots:
        if s.hall_capacities:
            s.hall_capacities = {
                hid: max(1, int(round(c * mult)))
                for hid, c in s.hall_capacities.items()
            }
    return cloned


def _avg_overload(conf, users, pol, base_cfg, seeds, w_rec):
    """Усреднённое mean_overload_excess по seeds."""
    vals: List[float] = []
    cfg = SimConfig(
        tau=base_cfg.tau,
        p_skip_base=base_cfg.p_skip_base,
        K=base_cfg.K,
        seed=0,
        w_rel=1.0 - w_rec,
        w_rec=w_rec,
        w_fame=base_cfg.w_fame,
    )
    for s in seeds:
        cfg.seed = s
        res = simulate(conf, users, pol, cfg)
        vals.append(float(mean_hall_overload_excess(conf, res)))
    return float(np.mean(vals))


def _cv(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


# ---------- EC1: при свободной capacity нет переполнения ----------

def test_ec1_loose_capacity_eliminates_overload(
    toy_2slot_conf, personas_50_users, active_pols_no_llm, base_cfg
):
    """EC1: при capacity_multiplier ≥ 3.0 mean_overload_excess == 0 для П1-П3."""
    loose_conf = _scale_capacity(toy_2slot_conf, 3.0)
    seeds = [1, 2, 3, 4, 5]
    failures = []
    for name, pol in active_pols_no_llm.items():
        ov = _avg_overload(loose_conf, personas_50_users, pol, base_cfg,
                           seeds=seeds, w_rec=0.5)
        if ov > 1e-9:
            failures.append((name, ov))
    assert not failures, (
        f"EC1 fail: при cap×3 переполнение должно быть 0, "
        f"но получили: {failures}"
    )


# ---------- EC2: монотонность по capacity_multiplier ----------

def test_ec2_monotone_overload_in_capacity(
    toy_2slot_conf, personas_50_users, active_pols_no_llm, base_cfg
):
    """EC2: mean_overload_excess невозрастает при росте capacity_multiplier.

    Проверяется на каждой из 3 активных политик отдельно: ужесточение
    вместимости (cap_mult ↓) не может снизить overload.
    """
    cap_grid = [0.5, 0.7, 1.0, 1.5, 3.0]
    seeds = [1, 2, 3, 4, 5]
    fails = []
    for name, pol in active_pols_no_llm.items():
        series: Dict[float, float] = {}
        for m in cap_grid:
            scaled = _scale_capacity(toy_2slot_conf, m)
            series[m] = _avg_overload(scaled, personas_50_users, pol, base_cfg,
                                       seeds=seeds, w_rec=0.5)
        # Невозрастание (с маленьким эпсилоном на численный шум семплинга)
        for i in range(len(cap_grid) - 1):
            a, b = cap_grid[i], cap_grid[i + 1]
            if series[a] + 1e-3 < series[b]:
                fails.append({"policy": name, "from": (a, series[a]),
                              "to": (b, series[b])})
    assert not fails, f"EC2 monotonicity fail: {fails}"


# ---------- EC3: при w_rec → 0 политики неразличимы ----------

def test_ec3_invariance_when_w_rec_zero(
    toy_2slot_conf, personas_50_users, active_pols_no_llm, base_cfg
):
    """EC3: при w_rec=0 mean_overload_excess идентичен между политиками.

    PROJECT_DESIGN §9 + accepted decision этапа C: rec-канал управляется
    исключительно через `w_rec`. При w_rec=0 политики обязаны давать строго
    одинаковую загрузку залов (CRN-инвариантность).
    """
    seeds = [1, 2, 3, 4, 5]
    base_conf = _scale_capacity(toy_2slot_conf, 1.0)  # natural
    vals = []
    for name, pol in active_pols_no_llm.items():
        ov = _avg_overload(base_conf, personas_50_users, pol, base_cfg,
                           seeds=seeds, w_rec=0.0)
        vals.append((name, ov))

    overloads = [v for _, v in vals]
    range_metric = max(overloads) - min(overloads)
    cv = _cv(overloads)
    # Ожидание строгое: range == 0 (CRN-инвариантность).
    assert range_metric < 1e-9, (
        f"EC3 fail: при w_rec=0 политики обязаны давать "
        f"идентичный overload, но range={range_metric:.6e}, vals={vals}"
    )
    # CV формально не определена при mean=0; либо 0 (по нашей реализации), либо < 5%.
    assert cv < 0.05, f"EC3 CV(overload, policies) = {cv:.4f} ≥ 0.05"


# ---------- EC4: при w_rec → 1 политики различимы ----------

def test_ec4_policies_differ_when_w_rec_one(
    toy_2slot_conf, personas_50_users, active_pols_no_llm, base_cfg
):
    """EC4: при w_rec=1 на стрессовой capacity range(metric, policies)
    значительно превосходит CV(EC3).

    Стрессовая capacity нужна, чтобы переполнение было ненулевым у части
    политик и различия между ними были видны в `mean_overload_excess`.
    """
    seeds = [1, 2, 3, 4, 5]
    stress_conf = _scale_capacity(toy_2slot_conf, 0.5)
    vals = []
    for name, pol in active_pols_no_llm.items():
        ov = _avg_overload(stress_conf, personas_50_users, pol, base_cfg,
                           seeds=seeds, w_rec=1.0)
        vals.append((name, ov))

    overloads = [v for _, v in vals]
    range_metric = max(overloads) - min(overloads)

    # CV(EC3) на той же конфигурации = 0 (см. EC3). Поэтому формальное
    # «range > 10× CV(EC3)» вырождается; используем абсолютный нижний порог,
    # эквивалентный «политики реально различимы».
    #
    # На toy_microconf_2slot × 50 user × stress×0.5: range >= 0.05 — устойчиво
    # (наблюдалось 0.18 в smoke F на mobius при w_rec=1; на toy сравнимо).
    assert range_metric > 0.02, (
        f"EC4 fail: при w_rec=1 range(overload, policies)={range_metric:.4f} "
        f"≤ 0.02; политики неразличимы. vals={vals}"
    )
