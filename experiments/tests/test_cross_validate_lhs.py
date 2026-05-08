"""Тесты `experiments/scripts/cross_validate_lhs.py`.

Покрывают core-функции расчёта Spearman ρ, Kendall τ, top-1 Hamming
на синтетических данных и проверяют acceptance gate (median Spearman ≥ 0.5).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXPERIMENTS_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_cv():
    if "cross_validate_lhs" in sys.modules:
        return sys.modules["cross_validate_lhs"]
    spec = importlib.util.spec_from_file_location(
        "cross_validate_lhs", SCRIPTS_DIR / "cross_validate_lhs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cross_validate_lhs"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- llm_ranking_vector ----------

def test_llm_ranking_lower_is_better_for_overload():
    cv = _load_cv()
    rows = [
        {"lhs_row_id": 1, "policy": "no_policy",
         "metric_mean_overload_excess": 0.20},
        {"lhs_row_id": 1, "policy": "cosine",
         "metric_mean_overload_excess": 0.30},
        {"lhs_row_id": 1, "policy": "capacity_aware",
         "metric_mean_overload_excess": 0.05},
        {"lhs_row_id": 1, "policy": "llm_ranker",
         "metric_mean_overload_excess": 0.10},
    ]
    ranks = cv.llm_ranking_vector(rows, 1, "metric_mean_overload_excess")
    # capacity_aware лучший (0.05) → rank 1; llm_ranker (0.10) → 2;
    # no_policy (0.20) → 3; cosine (0.30) → 4
    assert ranks["capacity_aware"] == pytest.approx(1.0)
    assert ranks["llm_ranker"] == pytest.approx(2.0)
    assert ranks["no_policy"] == pytest.approx(3.0)
    assert ranks["cosine"] == pytest.approx(4.0)


def test_llm_ranking_higher_is_better_for_utility():
    cv = _load_cv()
    rows = [
        {"lhs_row_id": 7, "policy": "no_policy",
         "metric_mean_user_utility": 0.5},
        {"lhs_row_id": 7, "policy": "cosine",
         "metric_mean_user_utility": 0.8},
    ]
    ranks = cv.llm_ranking_vector(rows, 7, "metric_mean_user_utility")
    # cosine утилита больше → rank 1
    assert ranks["cosine"] == pytest.approx(1.0)
    assert ranks["no_policy"] == pytest.approx(2.0)


# ---------- compute_correlations ----------

def test_correlations_perfect_match_rho_1():
    cv = _load_cv()
    param = {"6": {"no_policy": 1.0, "cosine": 2.0,
                    "capacity_aware": 3.0, "llm_ranker": 4.0}}
    llm = {"6": {"no_policy": 1.0, "cosine": 2.0,
                  "capacity_aware": 3.0, "llm_ranker": 4.0}}
    out = cv.compute_correlations(param, llm)
    assert out["per_lhs"]["6"]["rho"] == pytest.approx(1.0)
    assert out["per_lhs"]["6"]["tau"] == pytest.approx(1.0)
    assert out["per_lhs"]["6"]["top1_match"] is True


def test_correlations_inverse_rho_neg1():
    cv = _load_cv()
    param = {"7": {"no_policy": 1.0, "cosine": 2.0,
                    "capacity_aware": 3.0, "llm_ranker": 4.0}}
    llm = {"7": {"no_policy": 4.0, "cosine": 3.0,
                  "capacity_aware": 2.0, "llm_ranker": 1.0}}
    out = cv.compute_correlations(param, llm)
    assert out["per_lhs"]["7"]["rho"] == pytest.approx(-1.0)
    assert out["per_lhs"]["7"]["top1_match"] is False


def test_correlations_top1_match_when_best_same():
    cv = _load_cv()
    # параметрический: capacity_aware = 1 (лучший)
    param = {"13": {"no_policy": 4.0, "cosine": 3.0,
                     "capacity_aware": 1.0, "llm_ranker": 2.0}}
    # LLM: capacity_aware = 1 (лучший), но остальные перемешаны
    llm = {"13": {"no_policy": 3.0, "cosine": 4.0,
                   "capacity_aware": 1.0, "llm_ranker": 2.0}}
    out = cv.compute_correlations(param, llm)
    assert out["per_lhs"]["13"]["top1_match"] is True
    assert out["per_lhs"]["13"]["rho"] is not None


def test_correlations_summary_median_aggregation():
    cv = _load_cv()
    # 3 LHS: ρ = 1.0, 0.0, -1.0 → median = 0.0
    param = {
        "1": {"no_policy": 1.0, "cosine": 2.0, "capacity_aware": 3.0,
              "llm_ranker": 4.0},
        "2": {"no_policy": 1.0, "cosine": 2.0, "capacity_aware": 3.0,
              "llm_ranker": 4.0},
        "3": {"no_policy": 1.0, "cosine": 2.0, "capacity_aware": 3.0,
              "llm_ranker": 4.0},
    }
    llm = {
        "1": {"no_policy": 1.0, "cosine": 2.0, "capacity_aware": 3.0,
              "llm_ranker": 4.0},  # ρ=1.0
        "2": {"no_policy": 4.0, "cosine": 3.0, "capacity_aware": 2.0,
              "llm_ranker": 1.0},  # ρ=-1.0
        "3": {"no_policy": 1.0, "cosine": 4.0, "capacity_aware": 2.0,
              "llm_ranker": 3.0},  # ρ = 0.4 (1 - 6·6/(4·15))
    }
    out = cv.compute_correlations(param, llm)
    assert out["rho_summary"]["n"] == 3
    # median {1.0, -1.0, 0.4} = 0.4
    assert out["rho_summary"]["median"] == pytest.approx(0.4)


# ---------- acceptance_gate ----------

def test_acceptance_passes_when_median_geq_threshold():
    cv = _load_cv()
    res = cv.acceptance_gate({"median": 0.5}, threshold=0.5)
    assert res["passed"] is True
    res2 = cv.acceptance_gate({"median": 0.7}, threshold=0.5)
    assert res2["passed"] is True


def test_acceptance_fails_when_median_below_threshold():
    cv = _load_cv()
    res = cv.acceptance_gate({"median": 0.49}, threshold=0.5)
    assert res["passed"] is False
    res2 = cv.acceptance_gate({"median": -0.1}, threshold=0.5)
    assert res2["passed"] is False


def test_acceptance_handles_no_data():
    cv = _load_cv()
    res = cv.acceptance_gate({"median": None}, threshold=0.5)
    assert res["passed"] is False
    assert res["median"] is None


# ---------- E2E на synthetic JSON ----------

def _build_param_diag(tmp_path: Path) -> Path:
    """Синтетический analysis_llm_ranker_diagnostic.json."""
    param = {
        "subset": "maximin_12",
        "lhs_row_ids": [6, 7, 13],
        "policies": ["no_policy", "cosine", "capacity_aware", "llm_ranker"],
        "ranking_vectors": {
            "metric_mean_overload_excess": {
                "6": {"no_policy": 3.0, "cosine": 4.0,
                       "capacity_aware": 1.0, "llm_ranker": 2.0},
                "7": {"no_policy": 4.0, "cosine": 3.0,
                       "capacity_aware": 2.0, "llm_ranker": 1.0},
                "13": {"no_policy": 1.0, "cosine": 2.0,
                        "capacity_aware": 3.0, "llm_ranker": 4.0},
            },
            "metric_overflow_rate_slothall": {
                "6": {"no_policy": 3.0, "cosine": 4.0,
                       "capacity_aware": 1.0, "llm_ranker": 2.0},
                "7": {"no_policy": 4.0, "cosine": 3.0,
                       "capacity_aware": 2.0, "llm_ranker": 1.0},
                "13": {"no_policy": 1.0, "cosine": 2.0,
                        "capacity_aware": 3.0, "llm_ranker": 4.0},
            },
            "metric_hall_utilization_variance": {
                "6": {"no_policy": 3.0, "cosine": 4.0,
                       "capacity_aware": 1.0, "llm_ranker": 2.0},
                "7": {"no_policy": 4.0, "cosine": 3.0,
                       "capacity_aware": 2.0, "llm_ranker": 1.0},
                "13": {"no_policy": 1.0, "cosine": 2.0,
                        "capacity_aware": 3.0, "llm_ranker": 4.0},
            },
            "metric_mean_user_utility": {
                "6": {"no_policy": 4.0, "cosine": 1.0,
                       "capacity_aware": 3.0, "llm_ranker": 2.0},
                "7": {"no_policy": 4.0, "cosine": 1.0,
                       "capacity_aware": 3.0, "llm_ranker": 2.0},
                "13": {"no_policy": 4.0, "cosine": 1.0,
                        "capacity_aware": 3.0, "llm_ranker": 2.0},
            },
        },
    }
    p = tmp_path / "param.json"
    p.write_text(json.dumps(param))
    return p


def _build_llm_long(tmp_path: Path, perfect: bool = True) -> Path:
    """Синтетический V long-format. perfect=True → совпадает с param;
    False → инвертированный (rho=-1)."""
    rows = []
    base_overload = {6: {"no_policy": 0.30, "cosine": 0.40,
                          "capacity_aware": 0.05, "llm_ranker": 0.10},
                     7: {"no_policy": 0.40, "cosine": 0.30,
                          "capacity_aware": 0.20, "llm_ranker": 0.10},
                     13: {"no_policy": 0.10, "cosine": 0.20,
                           "capacity_aware": 0.30, "llm_ranker": 0.40}}
    base_utility = {6: {"no_policy": 0.5, "cosine": 0.9,
                         "capacity_aware": 0.6, "llm_ranker": 0.8},
                    7: {"no_policy": 0.5, "cosine": 0.9,
                         "capacity_aware": 0.6, "llm_ranker": 0.8},
                    13: {"no_policy": 0.5, "cosine": 0.9,
                          "capacity_aware": 0.6, "llm_ranker": 0.8}}
    for lhs in (6, 7, 13):
        for pol in ("no_policy", "cosine", "capacity_aware", "llm_ranker"):
            ovl = base_overload[lhs][pol]
            ut = base_utility[lhs][pol]
            if not perfect:
                # инверсия: для lhs=6 заменим overload на 1 - x
                if lhs == 6:
                    ovl = 1.0 - ovl
            rows.append({
                "lhs_row_id": lhs, "policy": pol,
                "metric_mean_overload_excess": ovl,
                "metric_overflow_rate_slothall": ovl,
                "metric_hall_utilization_variance": ovl * 0.5,
                "metric_mean_user_utility": ut,
            })
    payload = {"results": rows, "n_results": len(rows),
               "params": {}}
    p = tmp_path / "llm.json"
    p.write_text(json.dumps(payload))
    return p


def test_cv_e2e_perfect_match_passes(tmp_path: Path):
    cv = _load_cv()
    param_path = _build_param_diag(tmp_path)
    llm_path = _build_llm_long(tmp_path, perfect=True)

    import argparse
    args = argparse.Namespace(
        parametric=str(param_path),
        llm=str(llm_path),
        output_json=str(tmp_path / "out.json"),
        output_md=str(tmp_path / "out.md"),
        plots_dir=str(tmp_path / "plots"),
    )
    rc = cv.run(args)
    assert rc == 0
    out = json.loads((tmp_path / "out.json").read_text())
    # perfect match → ρ ≈ 1 везде, acceptance PASS
    assert out["acceptance_overall"]["passed"] is True
    assert out["rho_overall_across_metrics"]["median"] == pytest.approx(1.0)


def test_cv_e2e_invariant_no_mutation(tmp_path: Path):
    """cross_validate_lhs.py не должен модифицировать входные файлы."""
    cv = _load_cv()
    param_path = _build_param_diag(tmp_path)
    llm_path = _build_llm_long(tmp_path, perfect=True)
    import hashlib
    param_hash_before = hashlib.sha256(param_path.read_bytes()).hexdigest()
    llm_hash_before = hashlib.sha256(llm_path.read_bytes()).hexdigest()

    import argparse
    args = argparse.Namespace(
        parametric=str(param_path), llm=str(llm_path),
        output_json=str(tmp_path / "out.json"),
        output_md=str(tmp_path / "out.md"),
        plots_dir=str(tmp_path / "plots"),
    )
    cv.run(args)
    param_hash_after = hashlib.sha256(param_path.read_bytes()).hexdigest()
    llm_hash_after = hashlib.sha256(llm_path.read_bytes()).hexdigest()
    assert param_hash_before == param_hash_after
    assert llm_hash_before == llm_hash_after


def test_cv_constants_match_etap_S():
    """Имена метрик и политик cross_validate_lhs.py должны совпадать с теми,
    что использовались в этапе S."""
    cv = _load_cv()
    assert set(cv.POLICIES) == {"no_policy", "cosine",
                                  "capacity_aware", "llm_ranker"}
    assert "metric_mean_overload_excess" in cv.METRICS
    assert "metric_mean_user_utility" in cv.METRICS
    assert cv.LOWER_IS_BETTER["metric_mean_overload_excess"] is True
    assert cv.LOWER_IS_BETTER["metric_mean_user_utility"] is False
    assert cv.ACCEPTANCE_THRESHOLD == 0.5
