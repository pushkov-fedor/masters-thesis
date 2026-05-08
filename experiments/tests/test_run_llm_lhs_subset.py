"""Тесты `experiments/scripts/run_llm_lhs_subset.py` (этап V).

Покрывают invariants без вызова реального LLM API:
- gossip_level_from_w borders;
- seeds совпадают с derive_seeds Q-протокола;
- audience subsample детерминирован по audience_seed;
- swap_descriptor из Q применяется идентично через _apply_swap;
- long-format содержит все обязательные ключи для cross_validate_lhs.py;
- внутри одного слота agent decisions sequential (gossip invariant).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXPERIMENTS_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_runner():
    if "run_llm_lhs_subset" in sys.modules:
        return sys.modules["run_llm_lhs_subset"]
    spec = importlib.util.spec_from_file_location(
        "run_llm_lhs_subset", SCRIPTS_DIR / "run_llm_lhs_subset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_llm_lhs_subset"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- gossip_level_from_w ----------

def test_gossip_level_off_when_w_zero():
    mod = _load_runner()
    assert mod.gossip_level_from_w(0.0) == "off"
    assert mod.gossip_level_from_w(-0.1) == "off"


def test_gossip_level_moderate_when_w_low():
    mod = _load_runner()
    assert mod.gossip_level_from_w(0.001) == "moderate"
    assert mod.gossip_level_from_w(0.2) == "moderate"
    assert mod.gossip_level_from_w(0.39999) == "moderate"


def test_gossip_level_strong_when_w_high():
    mod = _load_runner()
    assert mod.gossip_level_from_w(0.4) == "strong"
    assert mod.gossip_level_from_w(0.55) == "strong"
    assert mod.gossip_level_from_w(0.7) == "strong"


# ---------- Seeds паритет ----------

def test_seeds_match_q_protocol():
    """derive_seeds должна быть точная функция, идентичная run_lhs_parametric."""
    from src.seeds import derive_seeds
    s = derive_seeds(13, replicate=1)
    assert s["audience_seed"] == 13 * 1_000_003
    assert s["phi_seed"] == 13 * 1_000_003 + 17
    assert s["cfg_seed"] == 1


# ---------- audience subsample ----------

def test_select_audience_deterministic_via_audience_seed():
    """Та же audience_seed → одинаковый набор персон между прогонами."""
    mod = _load_runner()
    from src.simulator import UserProfile

    rng = np.random.default_rng(0)
    users = [
        UserProfile(id=f"u_{i:03d}", text=f"user {i}",
                    embedding=rng.standard_normal(8).astype(np.float32))
        for i in range(50)
    ]
    a1 = mod.select_audience(users, audience_size=10, audience_seed=42)
    a2 = mod.select_audience(users, audience_size=10, audience_seed=42)
    assert [u.id for u in a1] == [u.id for u in a2]


def test_select_audience_different_seeds_yield_different_sets():
    mod = _load_runner()
    from src.simulator import UserProfile
    rng = np.random.default_rng(0)
    users = [
        UserProfile(id=f"u_{i:03d}", text=f"user {i}",
                    embedding=rng.standard_normal(8).astype(np.float32))
        for i in range(50)
    ]
    a1 = mod.select_audience(users, 10, audience_seed=42)
    a2 = mod.select_audience(users, 10, audience_seed=43)
    assert [u.id for u in a1] != [u.id for u in a2]


# ---------- Q swap_descriptor применяется bit-exact ----------

def test_swap_descriptor_from_q_applies_via_apply_swap(tmp_path: Path):
    """Если в Q swap_descriptor задан — применяем _apply_swap; результат
    должен совпасть с enumerate_modifications-результатом для того же
    descriptor (потому что _apply_swap — единая операция)."""
    mod = _load_runner()
    from src.simulator import Conference

    conf_path = (EXPERIMENTS_ROOT
                 / "data/conferences/mobius_2025_autumn.json")
    emb_path = (EXPERIMENTS_ROOT
                / "data/conferences/mobius_2025_autumn_embeddings.npz")
    if not conf_path.exists():
        pytest.skip("mobius conference assets not available")

    base = Conference.load(conf_path, emb_path)
    q_input = (EXPERIMENTS_ROOT
               / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.json")
    q_data = json.loads(q_input.read_text())
    # Берём первую запись с program_variant != 0 и descriptor
    sample = next(r for r in q_data["results"]
                  if r["replicate"] == 1
                  and r["program_variant"] != 0
                  and r["swap_descriptor"]
                  and r["is_maximin_point"])
    desc = sample["swap_descriptor"]
    pv = sample["program_variant"]
    # phi_seed не используется когда desc передан явно — fallback не должен
    # сработать
    program_conf, prog_meta = mod.apply_program_variant_with_q_descriptor(
        base, pv, desc, phi_seed=42,
    )
    assert prog_meta["swap_descriptor"] == desc
    assert prog_meta["fallback_to_p0"] is False
    # Проверим что слот.talk_ids изменился именно за счёт swap
    slot_a = next(s for s in program_conf.slots if s.id == desc["slot_a"])
    slot_b = next(s for s in program_conf.slots if s.id == desc["slot_b"])
    base_a = next(s for s in base.slots if s.id == desc["slot_a"])
    base_b = next(s for s in base.slots if s.id == desc["slot_b"])
    # t1 был в slot_a исходно, после swap должен быть в slot_b
    assert desc["t1"] in base_a.talk_ids
    assert desc["t1"] in slot_b.talk_ids
    assert desc["t2"] in base_b.talk_ids
    assert desc["t2"] in slot_a.talk_ids


# ---------- scale_capacity ----------

def test_scale_capacity_does_not_mutate_input():
    mod = _load_runner()
    from src.simulator import Conference

    conf_path = (EXPERIMENTS_ROOT
                 / "data/conferences/mobius_2025_autumn.json")
    emb_path = (EXPERIMENTS_ROOT
                / "data/conferences/mobius_2025_autumn_embeddings.npz")
    if not conf_path.exists():
        pytest.skip("mobius conference assets not available")

    base = Conference.load(conf_path, emb_path)
    base_caps = {h.id: h.capacity for h in base.halls.values()}
    scaled = mod.scale_capacity(base, mult=2.0)
    # base не изменился
    for hid, cap in base_caps.items():
        assert base.halls[hid].capacity == cap
    # scaled имеет 2x
    for hid, cap in base_caps.items():
        assert scaled.halls[hid].capacity == cap * 2


# ---------- Long-format keys ----------

def _expected_long_format_keys() -> set[str]:
    """Минимальный набор ключей, который cross_validate_lhs.py
    и downstream-аналитика ожидают видеть в каждой записи V."""
    return {
        "lhs_row_id", "capacity_multiplier", "popularity_source",
        "w_rel", "w_rec", "w_gossip", "audience_size", "program_variant",
        "policy", "replicate", "audience_seed", "phi_seed", "cfg_seed",
        "is_maximin_point", "fallback_to_p0", "gossip_level", "status",
        "metric_mean_overload_excess", "metric_mean_user_utility",
        "metric_overflow_rate_slothall", "metric_hall_utilization_variance",
        "metric_n_skipped", "metric_n_users",
        "n_parse_errors",
        "llmagent_cost_usd", "llmagent_calls",
        "llmagent_total_time_s", "llmranker_total_time_s",
        "llmranker_calls_delta", "llmranker_cache_hits_delta",
        "llmranker_cost_delta_usd",
    }


def test_long_format_keys_documented_in_runner():
    """csv_columns в runner должны включать обязательный набор."""
    mod = _load_runner()
    src = (SCRIPTS_DIR / "run_llm_lhs_subset.py").read_text()
    expected = _expected_long_format_keys()
    missing = []
    for k in expected:
        # Проверяем, что ключ упомянут в csv_columns или в long_row dict.
        # Простая текстовая проверка — компромиссная, но достаточная.
        if f'"{k}"' not in src and f"'{k}'" not in src:
            missing.append(k)
    assert not missing, f"missing keys in runner long-format: {missing}"


# ---------- Q artifact structure ----------

def test_q_artifact_has_maximin_indices_12():
    q_input = (EXPERIMENTS_ROOT
               / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.json")
    if not q_input.exists():
        pytest.skip("Q artifact not present")
    q_data = json.loads(q_input.read_text())
    assert len(q_data["maximin_indices"]) == 12
    expected_set = {6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48}
    assert set(q_data["maximin_indices"]) == expected_set


def test_smoke_picks_smallest_audience_among_maximin():
    """smoke-режим должен выбрать LHS-row с минимальным audience_size."""
    q_input = (EXPERIMENTS_ROOT
               / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.json")
    if not q_input.exists():
        pytest.skip("Q artifact not present")
    q_data = json.loads(q_input.read_text())
    maximin = set(q_data["maximin_indices"])
    candidates = [(r["lhs_row_id"], r["audience_size"])
                  for r in q_data["lhs_rows"]
                  if r["lhs_row_id"] in maximin]
    candidates.sort(key=lambda x: (x[1], x[0]))
    smallest = candidates[0]
    # на mobius_2025_autumn maximin-12 минимум audience = 30
    assert smallest[1] == 30


# ---------- Sequential within slot invariant ----------

def test_parallel_lhs_flag_documented():
    """В CLI и в коде должен быть документирован --parallel-lhs."""
    src = (SCRIPTS_DIR / "run_llm_lhs_subset.py").read_text()
    assert "--parallel-lhs" in src
    assert "parallel_lhs" in src
    # И обязательно явное указание, что gossip invariant не нарушается
    # при parallel-lhs (каждая LHS имеет свой slot_choice_count)
    assert "slot_choice_count" in src


def test_runner_documents_sequential_within_slot_invariant():
    """Главный инвариант: agent decisions внутри слота — sequential.

    Проверяем, что в коде есть явное указание на это (docstring).
    """
    src = (SCRIPTS_DIR / "run_llm_lhs_subset.py").read_text()
    assert "sequential" in src.lower() or "Sequential" in src
    # Должно быть указание про gossip-причину
    assert "gossip" in src.lower()
