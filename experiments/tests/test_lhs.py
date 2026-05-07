"""Тесты `generate_lhs` и `maximin_subset` (этап P).

Источник истины: `docs/spikes/spike_experiment_protocol.md` §6 (каталог осей),
§14 (минимальная реализация), Accepted decision уточнения 2-4.
"""
from __future__ import annotations

import pytest

from src.lhs import (
    AUDIENCE_SIZES,
    DEFAULT_MIN_PER_LEVEL,
    POPULARITY_SOURCES,
    PROGRAM_VARIANT_LEVELS,
    generate_lhs,
    maximin_subset,
)


# ---------- generate_lhs basic ----------

def test_generate_lhs_returns_n_points_50():
    rows = generate_lhs(n_points=50)
    assert len(rows) == 50


def test_generate_lhs_returns_n_points_30():
    """С n_points<50 порог DEFAULT_MIN_PER_LEVEL может быть недостижим;
    нужно передать relaxed min_per_level."""
    rows = generate_lhs(
        n_points=30,
        min_per_level={"program_variant": 3, "audience_size": 7,
                       "popularity_source": 7},
    )
    assert len(rows) == 30


def test_generate_lhs_simplex_satisfied():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert r["w_rec"] + r["w_gossip"] <= 1.0 + 1e-9
        assert abs(r["w_rel"] - (1.0 - r["w_rec"] - r["w_gossip"])) < 1e-9


def test_generate_lhs_w_rec_in_range():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert 0.0 <= r["w_rec"] <= 0.7 + 1e-9


def test_generate_lhs_w_gossip_in_range():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert 0.0 <= r["w_gossip"] <= 0.7 + 1e-9


def test_generate_lhs_capacity_multiplier_in_range():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert 0.5 <= r["capacity_multiplier"] <= 3.0 + 1e-9


def test_generate_lhs_lhs_row_id_sequential():
    rows = generate_lhs(n_points=50)
    ids = [r["lhs_row_id"] for r in rows]
    assert ids == list(range(50))


def test_generate_lhs_audience_size_in_grid():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert r["audience_size"] in AUDIENCE_SIZES


def test_generate_lhs_popularity_source_in_grid():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert r["popularity_source"] in POPULARITY_SOURCES


def test_generate_lhs_program_variant_in_range():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert r["program_variant"] in PROGRAM_VARIANT_LEVELS


def test_generate_lhs_u_raw_present():
    rows = generate_lhs(n_points=50)
    for r in rows:
        assert "u_raw" in r
        assert len(r["u_raw"]) == 6
        for u_i in r["u_raw"]:
            assert 0.0 <= u_i < 1.0


# ---------- Balance after repair (Accepted decision уточнение 4) ----------

def test_generate_lhs_balance_program_variant():
    rows = generate_lhs(n_points=50)
    counts = {lv: sum(1 for r in rows if r["program_variant"] == lv)
              for lv in PROGRAM_VARIANT_LEVELS}
    threshold = DEFAULT_MIN_PER_LEVEL["program_variant"]
    for lv, c in counts.items():
        assert c >= threshold, (
            f"program_variant={lv}: count={c} < {threshold}; counts={counts}"
        )


def test_generate_lhs_balance_audience_size():
    rows = generate_lhs(n_points=50)
    counts = {lv: sum(1 for r in rows if r["audience_size"] == lv)
              for lv in AUDIENCE_SIZES}
    threshold = DEFAULT_MIN_PER_LEVEL["audience_size"]
    for lv, c in counts.items():
        assert c >= threshold, (
            f"audience_size={lv}: count={c} < {threshold}; counts={counts}"
        )


def test_generate_lhs_balance_popularity_source():
    rows = generate_lhs(n_points=50)
    counts = {lv: sum(1 for r in rows if r["popularity_source"] == lv)
              for lv in POPULARITY_SOURCES}
    threshold = DEFAULT_MIN_PER_LEVEL["popularity_source"]
    for lv, c in counts.items():
        assert c >= threshold, (
            f"popularity_source={lv}: count={c} < {threshold}; counts={counts}"
        )


# ---------- Determinism (Accepted decision уточнение 3) ----------

def test_generate_lhs_deterministic_under_master_seed():
    rows_a = generate_lhs(n_points=50, master_seed=2026)
    rows_b = generate_lhs(n_points=50, master_seed=2026)
    assert len(rows_a) == len(rows_b) == 50
    for ra, rb in zip(rows_a, rows_b):
        assert ra["capacity_multiplier"] == rb["capacity_multiplier"]
        assert ra["popularity_source"] == rb["popularity_source"]
        assert ra["w_rec"] == rb["w_rec"]
        assert ra["w_gossip"] == rb["w_gossip"]
        assert ra["audience_size"] == rb["audience_size"]
        assert ra["program_variant"] == rb["program_variant"]


def test_generate_lhs_different_seeds_yield_different_plans():
    rows_a = generate_lhs(n_points=50, master_seed=2026)
    rows_b = generate_lhs(n_points=50, master_seed=2027)
    differs = any(
        ra["capacity_multiplier"] != rb["capacity_multiplier"]
        for ra, rb in zip(rows_a, rows_b)
    )
    assert differs, "разные master_seed дали идентичный план — неверный детерминизм"


# ---------- Edge cases ----------

def test_generate_lhs_invalid_low_max_blocks():
    """При max_blocks=0 (искусственно нулевой) — ValueError диагностики."""
    # Нужно создать ситуацию, где набрать n_points валидных нельзя.
    # max_blocks=0 эффективно отрезает любые попытки rejection sampling.
    with pytest.raises(ValueError):
        generate_lhs(n_points=50, max_blocks=0)


def test_generate_lhs_relaxed_min_per_level_for_small_n():
    """Smoke сценарий: n=5, relaxed thresholds → успешная генерация."""
    rows = generate_lhs(
        n_points=5,
        min_per_level={"program_variant": 1, "audience_size": 1,
                       "popularity_source": 1},
    )
    assert len(rows) == 5


# ---------- maximin_subset ----------

def test_maximin_subset_returns_unique_indices():
    rows = generate_lhs(n_points=50, master_seed=2026)
    selected = maximin_subset(rows, k=12, force_program_variant_zero=True)
    assert len(selected) == 12
    assert len(set(selected)) == 12, "найдены дубликаты в maximin subset"


def test_maximin_subset_includes_program_variant_zero_when_forced():
    rows = generate_lhs(n_points=50, master_seed=2026)
    selected = maximin_subset(rows, k=12, force_program_variant_zero=True)
    assert any(rows[i]["program_variant"] == 0 for i in selected), (
        "force_program_variant_zero=True, но в subset нет program_variant=0"
    )


def test_maximin_subset_indices_in_range():
    rows = generate_lhs(n_points=50, master_seed=2026)
    selected = maximin_subset(rows, k=12)
    for idx in selected:
        assert 0 <= idx < len(rows)


def test_maximin_subset_deterministic():
    rows = generate_lhs(n_points=50, master_seed=2026)
    sel_a = maximin_subset(rows, k=12, force_program_variant_zero=True)
    sel_b = maximin_subset(rows, k=12, force_program_variant_zero=True)
    assert sel_a == sel_b


def test_maximin_subset_k_invalid():
    rows = generate_lhs(n_points=50, master_seed=2026)
    with pytest.raises(ValueError):
        maximin_subset(rows, k=0)
    with pytest.raises(ValueError):
        maximin_subset(rows, k=-1)
    with pytest.raises(ValueError):
        maximin_subset(rows, k=51)


def test_maximin_subset_force_pv_zero_when_no_zero_present():
    """Если в rows нет program_variant=0 и flag=True — ValueError."""
    # Сгенерим rows и руками удалим program_variant=0
    rows = generate_lhs(
        n_points=10,
        min_per_level={"program_variant": 1, "audience_size": 1,
                       "popularity_source": 1},
    )
    # Подменяем все program_variant=0 на 1 для теста
    rows_no_zero = [dict(r, program_variant=1)
                    if r["program_variant"] == 0 else r
                    for r in rows]
    with pytest.raises(ValueError, match="program_variant=0"):
        maximin_subset(rows_no_zero, k=3, force_program_variant_zero=True)


def test_maximin_subset_smaller_k_works():
    rows = generate_lhs(n_points=50, master_seed=2026)
    selected = maximin_subset(rows, k=3, force_program_variant_zero=True)
    assert len(selected) == 3
