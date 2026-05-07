"""Юнит-тесты метрик (`experiments/src/metrics.py`).

Этап I PIVOT_IMPLEMENTATION_PLAN r5. Метрики проверяются на ручных
SimResult / Conference, без запуска симулятора, чтобы тестировать чистую
функцию метрик, а не пайплайн.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    compute_all,
    gini_coefficient,
    hall_load_gini,
    hall_utilization_variance,
    mean_hall_overload_excess,
    mean_user_utility,
    overflow_rate,
    skip_rate,
)
from src.simulator import SimResult, StepRecord


def _make_sim_result(steps_def, hall_loads):
    """Builder SimResult: steps_def = list[(slot_id, user_id, chosen, rel)]."""
    steps = [
        StepRecord(
            slot_id=s, user_id=u, recommended=[],
            chosen=c, chosen_relevance=r,
            chosen_hall_load_before=0.0,
        )
        for (s, u, c, r) in steps_def
    ]
    return SimResult(steps=steps, hall_load_per_slot=hall_loads)


# ---------- gini_coefficient ----------

def test_gini_zero_when_all_equal():
    v = np.array([10.0, 10.0, 10.0])
    assert gini_coefficient(v) == pytest.approx(0.0, abs=1e-9)


def test_gini_zero_when_all_zero():
    v = np.zeros(5)
    assert gini_coefficient(v) == 0.0


def test_gini_grows_with_concentration():
    eq = gini_coefficient(np.array([10.0, 10.0, 10.0, 10.0]))
    skew = gini_coefficient(np.array([0.0, 0.0, 0.0, 40.0]))
    assert skew > eq
    assert skew > 0.5


def test_gini_empty_returns_zero():
    assert gini_coefficient(np.array([])) == 0.0


# ---------- mean_user_utility ----------

def test_mean_user_utility_averages_relevance():
    res = _make_sim_result(
        steps_def=[
            ("s0", "u1", "t1", 0.4),
            ("s0", "u2", "t2", 0.6),
            ("s0", "u3", None, 0.0),  # skip
        ],
        hall_loads={},
    )
    assert mean_user_utility(res) == pytest.approx((0.4 + 0.6 + 0.0) / 3, abs=1e-9)


def test_mean_user_utility_empty():
    res = SimResult(steps=[], hall_load_per_slot={})
    assert mean_user_utility(res) == 0.0


# ---------- skip_rate ----------

def test_skip_rate_counts_none_chosen():
    res = _make_sim_result(
        steps_def=[
            ("s0", "u1", "t1", 0.5),
            ("s0", "u2", None, 0.0),
            ("s0", "u3", None, 0.0),
        ],
        hall_loads={},
    )
    assert skip_rate(res) == pytest.approx(2 / 3, abs=1e-9)


def test_skip_rate_empty():
    assert skip_rate(SimResult()) == 0.0


# ---------- overflow_rate ----------

def test_overflow_rate_zero_when_under_cap(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 30, 2: 20}})
    assert overflow_rate(conf, res) == 0.0


def test_overflow_rate_positive_when_overflow(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[10, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 30, 2: 20}})  # h=1: 30>10
    assert overflow_rate(conf, res) == pytest.approx(0.5, abs=1e-9)


# ---------- hall_utilization_variance ----------

def test_hall_var_zero_when_balanced(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 25, 2: 25}})
    assert hall_utilization_variance(conf, res) == pytest.approx(0.0, abs=1e-9)


def test_hall_var_positive_when_imbalanced(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 50, 2: 0}})  # 1.0 vs 0.0
    val = hall_utilization_variance(conf, res)
    assert val > 0.2  # var of [1.0, 0.0] = 0.25


def test_hall_var_skips_single_hall_slots(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=1, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 100}})
    # один зал в слоте → дисперсия исключена; нет других слотов → метрика 0
    assert hall_utilization_variance(conf, res) == 0.0


# ---------- mean_hall_overload_excess ----------

def test_overload_excess_zero_when_under_cap(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 50, 2: 50}})
    assert mean_hall_overload_excess(conf, res) == 0.0


def test_overload_excess_positive_when_overflow(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    # 75/50 = 1.5, excess = 0.5
    res = SimResult(hall_load_per_slot={"slot_00": {1: 75, 2: 30}})
    assert mean_hall_overload_excess(conf, res) == pytest.approx(0.5, abs=1e-9)


# ---------- hall_load_gini ----------

def test_hall_load_gini_zero_when_balanced(make_synth_conf):
    conf, _ = make_synth_conf(slots=2, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={
        "slot_00": {1: 25, 2: 25},
        "slot_01": {1: 25, 2: 25},
    })
    assert hall_load_gini(conf, res) == pytest.approx(0.0, abs=1e-9)


def test_hall_load_gini_positive_when_skewed(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(hall_load_per_slot={"slot_00": {1: 0, 2: 50}})
    assert hall_load_gini(conf, res) > 0.0


# ---------- compute_all ----------

def test_compute_all_returns_canonical_keys(make_synth_conf):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[50, 50])
    res = SimResult(
        steps=[],
        hall_load_per_slot={"slot_00": {1: 30, 2: 20}},
    )
    out = compute_all(conf, res)
    expected = {
        "overflow_rate_all",
        "overflow_rate_choice",
        "hall_utilization_variance",
        "mean_user_utility",
        "hall_load_gini",
        "skip_rate",
        "mean_overload_excess",
    }
    assert set(out.keys()) == expected
    for k, v in out.items():
        assert isinstance(v, float), f"{k} must be float"
