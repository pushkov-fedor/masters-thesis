"""Юнит-тесты активных политик (П1–П3) и реестра.

Этап I PIVOT_IMPLEMENTATION_PLAN r5. П4 (LLMRankerPolicy) требует
OPENROUTER_API_KEY и сетевые вызовы — здесь не тестируется (отдельный
интеграционный сценарий, вне базового pytest-набора).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.policies.capacity_aware_policy import CapacityAwarePolicy
from src.policies.cosine_policy import CosinePolicy
from src.policies.no_policy import NoPolicy
from src.policies.registry import ACTIVE_POLICY_NAMES, active_policies
from src.simulator import Slot, UserProfile


# ---------- Реестр ----------

def test_active_policy_names_constant():
    assert ACTIVE_POLICY_NAMES == ("no_policy", "cosine", "capacity_aware", "llm_ranker")


def test_active_policies_no_llm_returns_three():
    pols = active_policies(include_llm=False)
    assert set(pols.keys()) == {"no_policy", "cosine", "capacity_aware"}
    assert isinstance(pols["no_policy"], NoPolicy)
    assert isinstance(pols["cosine"], CosinePolicy)
    assert isinstance(pols["capacity_aware"], CapacityAwarePolicy)


# ---------- NoPolicy ----------

def test_no_policy_returns_empty_list(make_synth_conf, make_synth_users):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2)
    user = make_synth_users(n=1, seed=0)[0]
    pol = NoPolicy()
    state = {"K": 2, "hall_load": {}, "relevance_fn": None}
    out = pol(user=user, slot=conf.slots[0], conf=conf, state=state)
    assert out == []


# ---------- CosinePolicy ----------

def test_cosine_returns_top_k_by_descending_similarity(make_synth_conf, make_synth_users):
    # Подготовка: 1 слот × 3 talks с явно различными эмбедингами;
    # пользователь близок к talk h=1 → top-1 должен быть h=1.
    user_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    talk_embs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),  # cos=1.0
        np.array([0.5, 0.5, 0.0], dtype=np.float32),  # cos~0.71
        np.array([0.0, 0.0, 1.0], dtype=np.float32),  # cos=0.0
    ]
    talk_embs = [e / (np.linalg.norm(e) + 1e-9) for e in talk_embs]
    conf, _ = make_synth_conf(
        slots=1, talks_per_slot=3, hall_capacities=[50, 50, 50],
        emb_dim=3, talk_emb_per_slot=[talk_embs],
    )
    user = make_synth_users(n=1, seed=0,
                            embeddings=[user_emb / (np.linalg.norm(user_emb) + 1e-9)])[0]
    pol = CosinePolicy()
    state = {"K": 2, "hall_load": {}, "relevance_fn": None}
    out = pol(user=user, slot=conf.slots[0], conf=conf, state=state)
    assert len(out) == 2
    assert out[0] == "t_s0_h1"  # top-1 — самый близкий
    assert out[1] == "t_s0_h2"  # top-2


def test_cosine_respects_K(make_synth_conf, make_synth_users):
    conf, _ = make_synth_conf(slots=1, talks_per_slot=4,
                              hall_capacities=[10, 10, 10, 10], seed=42)
    user = make_synth_users(n=1, emb_dim=8, seed=1)[0]
    pol = CosinePolicy()
    state = {"K": 2, "hall_load": {}, "relevance_fn": None}
    out = pol(user=user, slot=conf.slots[0], conf=conf, state=state)
    assert len(out) == 2


# ---------- CapacityAwarePolicy ----------

def test_capacity_aware_equals_cosine_when_halls_empty(make_synth_conf, make_synth_users):
    """С нулевой загрузкой по всем залам penalty = 0 → ранжирование совпадает с cosine."""
    conf, _ = make_synth_conf(slots=1, talks_per_slot=3,
                              hall_capacities=[10, 10, 10], seed=7)
    user = make_synth_users(n=1, emb_dim=8, seed=1)[0]
    cap_pol = CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95)
    cos_pol = CosinePolicy()

    slot = conf.slots[0]
    empty_load = {(slot.id, h.id): 0 for h in conf.halls.values()}
    state_cap = {"K": 3, "hall_load": empty_load, "relevance_fn": None}
    state_cos = {"K": 3, "hall_load": {}, "relevance_fn": None}

    out_cap = cap_pol(user=user, slot=slot, conf=conf, state=state_cap)
    out_cos = cos_pol(user=user, slot=slot, conf=conf, state=state_cos)
    assert out_cap == out_cos


def test_capacity_aware_demotes_overloaded_hall(make_synth_conf, make_synth_users):
    """Hall с высокой загрузкой получает penalty и опускается в ранжировании."""
    user_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # talk h=1 — самый близкий пользователю; talk h=2 — менее близкий.
    talk_embs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),    # cos = 1.0, hall=1
        np.array([0.95, 0.05, 0.0], dtype=np.float32),  # cos ≈ 0.99, hall=2
    ]
    talk_embs = [e / (np.linalg.norm(e) + 1e-9) for e in talk_embs]
    conf, _ = make_synth_conf(
        slots=1, talks_per_slot=2, hall_capacities=[10, 10],
        emb_dim=3, talk_emb_per_slot=[talk_embs],
    )
    user = make_synth_users(n=1, emb_dim=3, seed=0,
                            embeddings=[user_emb / (np.linalg.norm(user_emb) + 1e-9)])[0]

    pol = CapacityAwarePolicy(alpha=10.0, hard_threshold=0.95)
    slot = conf.slots[0]
    # hall=1 загружен на 90% (penalty = 10*0.9 = 9.0); hall=2 пуст
    loaded = {(slot.id, 1): 9, (slot.id, 2): 0}
    state = {"K": 1, "hall_load": loaded, "relevance_fn": None}
    out = pol(user=user, slot=slot, conf=conf, state=state)
    # При большом alpha и почти-полном hall=1 — top-1 должен быть hall=2.
    assert out == ["t_s0_h2"]


def test_capacity_aware_excludes_above_hard_threshold(make_synth_conf, make_synth_users):
    """Hall с load >= hard_threshold исключается, если есть альтернативы."""
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[10, 10],
                              emb_dim=8, seed=3)
    user = make_synth_users(n=1, emb_dim=8, seed=4)[0]
    pol = CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95)

    slot = conf.slots[0]
    # hall=1 заполнен почти полностью (load = 0.96 ≥ 0.95)
    loaded = {(slot.id, 1): 10, (slot.id, 2): 0}
    state = {"K": 2, "hall_load": loaded, "relevance_fn": None}
    out = pol(user=user, slot=slot, conf=conf, state=state)
    # hall=1 talk должен быть исключён
    assert "t_s0_h1" not in out
    assert out == ["t_s0_h2"]


def test_capacity_aware_fallback_when_all_halls_full(make_synth_conf, make_synth_users):
    """Если все залы переполнены — возвращаем top-K по cosine как fallback."""
    conf, _ = make_synth_conf(slots=1, talks_per_slot=2, hall_capacities=[10, 10],
                              emb_dim=8, seed=5)
    user = make_synth_users(n=1, emb_dim=8, seed=6)[0]
    pol = CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95)

    slot = conf.slots[0]
    loaded = {(slot.id, 1): 10, (slot.id, 2): 10}  # оба зала полные
    state = {"K": 2, "hall_load": loaded, "relevance_fn": None}
    out = pol(user=user, slot=slot, conf=conf, state=state)
    assert len(out) == 2  # fallback должен вернуть top-K
