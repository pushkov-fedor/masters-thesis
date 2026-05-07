"""Юнит-тесты симулятора.

Этап I PIVOT_IMPLEMENTATION_PLAN r5.

Главный инвариант: при `w_rec = 0` функция полезности агента **не зависит**
от выдачи политики (rec_indicator * 0 = 0). Это формальная EC3 на уровне
ядра: при одних и тех же seed / users / conference две разные политики
должны давать идентичный SimResult.

Дополнительно — проверки cosine_relevance и одиночного «определённого»
выбора (TC4-related, но изолированно как unit на utility).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.simulator import (
    SimConfig,
    cosine_relevance,
    simulate,
)


def test_cosine_relevance_normalized_vectors():
    """Для нормализованных эмбедингов cos = dot ∈ [-1, 1]."""
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_relevance(a, b) == pytest.approx(1.0, abs=1e-6)

    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_relevance(a, c) == pytest.approx(0.0, abs=1e-6)


def test_utility_invariant_under_policy_when_w_rec_zero(
    make_synth_conf, make_synth_users, active_pols_no_llm
):
    """**Формальная EC3 на уровне ядра.**

    При `w_rec = 0` две разные политики на одних и тех же users / conference /
    seed обязаны давать идентичный SimResult: rec_indicator * 0 = 0, поэтому
    utility не зависит от выдачи политики; CRN-инвариантность гарантируется
    тем, что choice_rng и policy_rng — независимые потоки.
    """
    conf, _ = make_synth_conf(slots=2, talks_per_slot=2,
                              hall_capacities=[50, 50], emb_dim=8, seed=42)
    users = make_synth_users(n=30, emb_dim=8, seed=7)
    cfg = SimConfig(tau=0.7, p_skip_base=0.10, K=2, seed=1,
                    w_rel=1.0, w_rec=0.0, w_fame=0.0)

    no_pol = active_pols_no_llm["no_policy"]
    cos_pol = active_pols_no_llm["cosine"]
    cap_pol = active_pols_no_llm["capacity_aware"]

    res_no = simulate(conf, users, no_pol, cfg)
    res_cos = simulate(conf, users, cos_pol, cfg)
    res_cap = simulate(conf, users, cap_pol, cfg)

    chosen_no = [s.chosen for s in res_no.steps]
    chosen_cos = [s.chosen for s in res_cos.steps]
    chosen_cap = [s.chosen for s in res_cap.steps]

    assert chosen_no == chosen_cos, (
        "Cosine ≠ NoPolicy при w_rec=0: rec-канал должен быть выключен"
    )
    assert chosen_no == chosen_cap, (
        "CapacityAware ≠ NoPolicy при w_rec=0: rec-канал должен быть выключен"
    )
    assert res_no.hall_load_per_slot == res_cos.hall_load_per_slot
    assert res_no.hall_load_per_slot == res_cap.hall_load_per_slot


def test_utility_changes_under_policy_when_w_rec_positive(
    make_synth_conf, make_synth_users, active_pols_no_llm
):
    """При `w_rec > 0` cosine и no_policy дают **различные** результаты.

    Контрапозиция к invariance — без неё EC4 невыполнима в принципе.
    """
    conf, _ = make_synth_conf(slots=2, talks_per_slot=2,
                              hall_capacities=[10, 10], emb_dim=8, seed=42)
    users = make_synth_users(n=50, emb_dim=8, seed=7)
    cfg = SimConfig(tau=0.7, p_skip_base=0.10, K=1, seed=1,
                    w_rel=0.0, w_rec=1.0, w_fame=0.0)  # rec доминирует

    no_pol = active_pols_no_llm["no_policy"]
    cos_pol = active_pols_no_llm["cosine"]

    res_no = simulate(conf, users, no_pol, cfg)
    res_cos = simulate(conf, users, cos_pol, cfg)

    assert res_no.hall_load_per_slot != res_cos.hall_load_per_slot, (
        "При w_rec=1 cosine и no_policy обязаны различаться"
    )


def test_skip_rate_bounded_by_p_skip_base(
    make_synth_conf, make_synth_users, active_pols_no_llm
):
    """Базовая sanity: skip-rate в окрестности p_skip_base при штатных параметрах."""
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2,
                              hall_capacities=[100, 100], emb_dim=8, seed=11)
    users = make_synth_users(n=200, emb_dim=8, seed=12)
    cfg = SimConfig(tau=0.7, p_skip_base=0.10, K=1, seed=1,
                    w_rel=0.7, w_rec=0.3, w_fame=0.0)

    no_pol = active_pols_no_llm["no_policy"]
    res = simulate(conf, users, no_pol, cfg)

    n_skipped = sum(1 for s in res.steps if s.chosen is None)
    skip_share = n_skipped / max(1, len(res.steps))
    # На 200 user'ах при p_skip=0.10 биномиальный шум ≈ sqrt(200 * 0.1 * 0.9) / 200 ≈ 0.021;
    # Берём просторный коридор [0.05, 0.20].
    assert 0.05 <= skip_share <= 0.20, f"skip-rate вне коридора: {skip_share:.3f}"


def test_simulate_is_deterministic_under_fixed_seed(
    make_synth_conf, make_synth_users, active_pols_no_llm
):
    """Симулятор детерминирован при фиксированном seed."""
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, emb_dim=8, seed=3)
    users = make_synth_users(n=20, emb_dim=8, seed=4)
    cfg = SimConfig(seed=42, w_rel=0.7, w_rec=0.3)
    pol = active_pols_no_llm["cosine"]

    res_a = simulate(conf, users, pol, cfg)
    res_b = simulate(conf, users, pol, cfg)
    assert [s.chosen for s in res_a.steps] == [s.chosen for s in res_b.steps]
    assert res_a.hall_load_per_slot == res_b.hall_load_per_slot
