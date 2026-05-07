"""Тесты CRN-контракта `derive_seeds` (этап P).

Источник истины: `docs/spikes/spike_experiment_protocol.md` §8.
"""
from __future__ import annotations

import pytest

from src.seeds import derive_seeds


def test_audience_seed_invariant_across_replicates():
    """audience_seed зависит ТОЛЬКО от lhs_row_id; одинаковая аудитория
    между всеми seed-репликами одной LHS-точки."""
    s1 = derive_seeds(lhs_row_id=10, replicate=1)
    s2 = derive_seeds(lhs_row_id=10, replicate=2)
    s3 = derive_seeds(lhs_row_id=10, replicate=3)
    assert s1["audience_seed"] == s2["audience_seed"] == s3["audience_seed"]


def test_phi_seed_invariant_across_replicates():
    """phi_seed зависит ТОЛЬКО от lhs_row_id; одинаковый program_variant
    между всеми seed-репликами одной LHS-точки."""
    s1 = derive_seeds(lhs_row_id=10, replicate=1)
    s2 = derive_seeds(lhs_row_id=10, replicate=5)
    assert s1["phi_seed"] == s2["phi_seed"]


def test_cfg_seed_equals_replicate():
    """cfg_seed = replicate; варьируется только между репликами."""
    for r in (1, 2, 3, 5, 10, 100):
        s = derive_seeds(lhs_row_id=42, replicate=r)
        assert s["cfg_seed"] == r


def test_audience_seed_differs_across_lhs_rows():
    """audience_seed различен между разными LHS-точками."""
    s1 = derive_seeds(lhs_row_id=1, replicate=1)
    s2 = derive_seeds(lhs_row_id=2, replicate=1)
    assert s1["audience_seed"] != s2["audience_seed"]


def test_phi_seed_differs_from_audience_seed():
    """phi_seed ≠ audience_seed внутри одной точки (разные RNG-потоки)."""
    s = derive_seeds(lhs_row_id=42, replicate=1)
    assert s["phi_seed"] != s["audience_seed"]


def test_audience_and_phi_seeds_differ_across_lhs_rows():
    """phi_seed различен между разными LHS-точками."""
    s1 = derive_seeds(lhs_row_id=1, replicate=1)
    s2 = derive_seeds(lhs_row_id=2, replicate=1)
    assert s1["phi_seed"] != s2["phi_seed"]


def test_invalid_lhs_row_id_negative():
    with pytest.raises(ValueError, match="lhs_row_id"):
        derive_seeds(lhs_row_id=-1, replicate=1)


def test_invalid_replicate_zero():
    with pytest.raises(ValueError, match="replicate"):
        derive_seeds(lhs_row_id=0, replicate=0)


def test_invalid_replicate_negative():
    with pytest.raises(ValueError, match="replicate"):
        derive_seeds(lhs_row_id=0, replicate=-2)


def test_returned_dict_keys():
    s = derive_seeds(lhs_row_id=0, replicate=1)
    assert set(s.keys()) == {"audience_seed", "phi_seed", "cfg_seed"}


def test_lhs_row_id_zero_works():
    """lhs_row_id=0 валиден; даёт audience_seed=0 и phi_seed=17."""
    s = derive_seeds(lhs_row_id=0, replicate=1)
    assert s["audience_seed"] == 0
    assert s["phi_seed"] == 17
    assert s["cfg_seed"] == 1
