"""Тесты постобработки этапа S (`experiments/scripts/analyze_lhs.py`).

Минимальный invariant-набор:
- pairwise win-rate на синтетических данных;
- regret лучшей политики == 0 на каждой LHS-row;
- median aggregation по 3 replicate;
- П4 отсутствует в full-50 таблицах;
- П4 присутствует только в maximin/restricted таблицах;
- win/loss/ties по eps-варианту в сумме = 1;
- sign-test для program_variant помечен diagnostic-only;
- analyze_lhs не мутирует исходный Q-артефакт.
"""
from __future__ import annotations

import hashlib
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


def _load_analyze():
    """Импорт скрипта analyze_lhs.py как модуля без побочных эффектов."""
    if "analyze_lhs" in sys.modules:
        return sys.modules["analyze_lhs"]
    spec = importlib.util.spec_from_file_location(
        "analyze_lhs", SCRIPTS_DIR / "analyze_lhs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_lhs"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- Synthetic helpers ----------

def _make_record(lhs_row_id: int, policy: str, replicate: int,
                 metric_overload: float, metric_utility: float = 0.73,
                 *, capacity_multiplier: float = 1.5,
                 popularity_source: str = "cosine_only",
                 w_rec: float = 0.3, w_gossip: float = 0.2, w_rel: float = 0.5,
                 audience_size: int = 60, program_variant: int = 0,
                 is_maximin: bool = False) -> dict:
    return {
        "lhs_row_id": lhs_row_id,
        "capacity_multiplier": capacity_multiplier,
        "popularity_source": popularity_source,
        "w_rel": w_rel,
        "w_rec": w_rec,
        "w_gossip": w_gossip,
        "audience_size": audience_size,
        "program_variant": program_variant,
        "policy": policy,
        "replicate": replicate,
        "audience_seed": 1000 + lhs_row_id,
        "phi_seed": 2000 + lhs_row_id,
        "cfg_seed": replicate,
        "is_maximin_point": is_maximin,
        "swap_descriptor": None,
        "fallback_to_p0": False,
        "metric_mean_overload_excess": metric_overload,
        "metric_mean_user_utility": metric_utility,
        "metric_overflow_rate_slothall": metric_overload,
        "metric_hall_utilization_variance": metric_overload * 0.5,
        "metric_n_skipped": 0,
        "metric_n_users": 100,
    }


def _make_records_3replicates(lhs_row_id, policy, mean_overload,
                              **kwargs) -> list[dict]:
    return [
        _make_record(lhs_row_id, policy, r, mean_overload + (r - 2) * 0.001,
                     **kwargs)
        for r in (1, 2, 3)
    ]


# ---------- Pairwise win-rate ----------

def test_pairwise_winrate_strict_known():
    az = _load_analyze()
    # 4 LHS-row, A < B на 3, A == B на 1
    a_vals = [0.0, 0.0, 0.0, 0.5]
    b_vals = [0.5, 0.5, 0.5, 0.5]
    stat = az.pairwise_winrate(a_vals, b_vals, eps=0.0,
                               lower_is_better=True)
    assert stat["n_paired"] == 4
    assert stat["win_strict"] == pytest.approx(0.75)
    assert stat["loss_strict"] == pytest.approx(0.0)
    # eps=0 → ties только при diff == 0
    assert stat["ties_eps"] == pytest.approx(0.25)


def test_pairwise_winrate_higher_better_inverts_direction():
    az = _load_analyze()
    # higher better (utility): A > B → A wins
    a_vals = [0.8, 0.7]
    b_vals = [0.7, 0.8]
    stat = az.pairwise_winrate(a_vals, b_vals, eps=0.0,
                               lower_is_better=False)
    assert stat["win_strict"] == pytest.approx(0.5)
    assert stat["loss_strict"] == pytest.approx(0.5)


def test_pairwise_winrate_eps_threshold():
    az = _load_analyze()
    a_vals = [0.001, 0.020]
    b_vals = [0.000, 0.000]
    # eps=0.005: первая пара diff=0.001 < eps → ties; вторая diff=0.020 > eps
    # → loss (для A, lower_is_better)
    stat = az.pairwise_winrate(a_vals, b_vals, eps=0.005,
                               lower_is_better=True)
    assert stat["ties_eps"] == pytest.approx(0.5)
    assert stat["loss_eps"] == pytest.approx(0.5)
    assert stat["win_eps"] == pytest.approx(0.0)


def test_winrate_eps_sums_to_one():
    az = _load_analyze()
    rng = np.random.default_rng(42)
    a_vals = rng.random(50).tolist()
    b_vals = rng.random(50).tolist()
    stat = az.pairwise_winrate(a_vals, b_vals, eps=0.05,
                               lower_is_better=True)
    total = stat["win_eps"] + stat["loss_eps"] + stat["ties_eps"]
    assert abs(total - 1.0) < 1e-9, f"win+loss+ties={total}, expected 1"


def test_winrate_handles_none_pairs():
    az = _load_analyze()
    a_vals = [0.1, None, 0.3]
    b_vals = [0.2, 0.5, None]
    stat = az.pairwise_winrate(a_vals, b_vals, eps=0.0,
                               lower_is_better=True)
    assert stat["n_paired"] == 1


# ---------- Regret ----------

def test_regret_zero_for_best_policy_each_lhs_row():
    az = _load_analyze()
    records = []
    # 5 LHS-row, 3 политики (no_policy, cosine, capacity_aware)
    # capacity_aware всегда лучшая (минимальный overload)
    for i in range(5):
        records += _make_records_3replicates(i, "no_policy",
                                              mean_overload=0.10)
        records += _make_records_3replicates(i, "cosine",
                                              mean_overload=0.08)
        records += _make_records_3replicates(i, "capacity_aware",
                                              mean_overload=0.05)
    aggregated = az.aggregate_replicates(records)
    regret = az.compute_regret(aggregated, list(range(5)),
                                az.P123)
    # capacity_aware лучшая везде → её regret ≡ 0 на всех LHS-row
    cap_per_lhs = regret["metric_mean_overload_excess"]["per_lhs_row"]
    for i in range(5):
        assert cap_per_lhs[i]["regret"]["capacity_aware"] == pytest.approx(0.0)


def test_regret_for_utility_uses_max_minus_value():
    az = _load_analyze()
    records = []
    # one LHS, three policies; utility: 0.9, 0.7, 0.5 → best = 0.9
    records += _make_records_3replicates(0, "no_policy", 0.0,
                                          metric_utility=0.5)
    # помещаю utility вручную через _make_record — _make_records_3replicates
    # использует metric_utility=0.73 default; тут переопределим напрямую
    for r in records:
        r["metric_mean_user_utility"] = 0.5
    cosine_recs = _make_records_3replicates(0, "cosine", 0.0)
    for r in cosine_recs:
        r["metric_mean_user_utility"] = 0.7
    capa_recs = _make_records_3replicates(0, "capacity_aware", 0.0)
    for r in capa_recs:
        r["metric_mean_user_utility"] = 0.9
    records += cosine_recs + capa_recs
    aggregated = az.aggregate_replicates(records)
    regret = az.compute_regret(aggregated, [0], az.P123)
    util_per_lhs = regret["metric_mean_user_utility"]["per_lhs_row"]
    assert util_per_lhs[0]["best"] == pytest.approx(0.9)
    assert util_per_lhs[0]["regret"]["capacity_aware"] == pytest.approx(0.0)
    assert util_per_lhs[0]["regret"]["no_policy"] == pytest.approx(0.4)


# ---------- Median aggregation ----------

def test_median_aggregation_over_3_replicates():
    az = _load_analyze()
    records = []
    # vals = [0.10, 0.20, 0.30] → median == 0.20, mean == 0.20
    for r, val in zip((1, 2, 3), (0.10, 0.20, 0.30)):
        records.append(_make_record(0, "cosine", r, val))
    agg = az.aggregate_replicates(records)
    cell = agg[(0, "cosine")]
    assert cell["metrics"]["metric_mean_overload_excess"]["median"] == pytest.approx(0.20)
    assert cell["metrics"]["metric_mean_overload_excess"]["mean"] == pytest.approx(0.20)
    # vals = [0.10, 0.10, 0.40] → median=0.10, mean≈0.20
    records2 = [_make_record(1, "cosine", r, val)
                for r, val in zip((1, 2, 3), (0.10, 0.10, 0.40))]
    agg2 = az.aggregate_replicates(records2)
    cell2 = agg2[(1, "cosine")]
    assert cell2["metrics"]["metric_mean_overload_excess"]["median"] == pytest.approx(0.10)
    assert cell2["metrics"]["metric_mean_overload_excess"]["mean"] == pytest.approx(0.20, abs=1e-6)


# ---------- P4 separation ----------

def _build_minimal_dataset(tmp_path: Path) -> Path:
    """Микро-Q-JSON: 6 LHS-row × П1–П3 (full) + 2 maximin × П4."""
    records = []
    # 6 LHS-row × 3 policies × 3 replicates = 54
    for i in range(6):
        is_max = i in (2, 4)
        for pol in ("no_policy", "cosine", "capacity_aware"):
            base = 0.05 + (0.01 if pol == "cosine" else 0.0) + i * 0.005
            for r in (1, 2, 3):
                records.append(_make_record(i, pol, r, base + 0.001 * (r - 2),
                                            is_maximin=is_max,
                                            program_variant=i % 6))
    # П4 только на 2 maximin LHS
    for i in (2, 4):
        for r in (1, 2, 3):
            records.append(_make_record(i, "llm_ranker", r,
                                        0.04 + 0.001 * (r - 2),
                                        is_maximin=True,
                                        program_variant=i % 6))
    payload = {
        "etap": "Q",
        "conference": "synthetic",
        "params": {"n_points": 6, "replicates": 3, "master_seed": 1,
                   "maximin_k": 2},
        "lhs_rows": [
            {"u_raw": [0.1] * 6,
             "capacity_multiplier": 1.5,
             "popularity_source": "cosine_only",
             "w_rec": 0.3, "w_gossip": 0.2, "w_rel": 0.5,
             "audience_size": 60, "program_variant": i % 6,
             "lhs_row_id": i}
            for i in range(6)
        ],
        "maximin_indices": [2, 4],
        "results": records,
        "n_results": len(records),
        "n_evals_by_policy": {
            "no_policy": 18, "cosine": 18, "capacity_aware": 18,
            "llm_ranker": 6,
        },
        "n_p4_evals": 6,
        "p4_cost_usd": 0.0,
        "timings": {},
        "elapsed_total_s": 0.0,
        "acceptance": {},
    }
    p = tmp_path / "synthetic_q.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def test_p4_absent_in_full_50_tables(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    rc = az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    assert rc == 0
    pairwise = json.loads((out_dir / "analysis_pairwise.json").read_text())
    full_pairs = pairwise["full_50"]["metric_mean_overload_excess"]
    for pair_label in full_pairs:
        assert "llm_ranker" not in pair_label, (
            f"П4 просочился в full-50 pairwise: {pair_label}"
        )
    # distribution full тоже без П4
    dist_full = pairwise["distribution_full_50"]
    assert "llm_ranker" not in dist_full
    # явный флаг
    assert pairwise["p4_in_full_50"] is False


def test_p4_present_only_in_maximin_tables(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    pairwise = json.loads((out_dir / "analysis_pairwise.json").read_text())
    maximin_pairs = pairwise["maximin_12"]["metric_mean_overload_excess"]
    has_p4_pair = any("llm_ranker" in p for p in maximin_pairs)
    assert has_p4_pair, "П4 должна участвовать в maximin pairwise"
    # llm_ranker_diagnostic существует и содержит П4
    llm_diag = json.loads(
        (out_dir / "analysis_llm_ranker_diagnostic.json").read_text()
    )
    assert "llm_ranker" in llm_diag["policies"]


def test_program_effect_sign_test_marked_diagnostic_only(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    program = json.loads(
        (out_dir / "analysis_program_effect.json").read_text()
    )
    assert program["interpretation"] == "diagnostic only — no conf-matching"
    assert program["policies_excluded"] == ["llm_ranker"]
    # каждая ячейка sign_test_diagnostic_only должна иметь interpretation
    sign = program["sign_test_diagnostic_only"]
    for pi, by_metric in sign.items():
        for m, by_pv in by_metric.items():
            for k, cell in by_pv.items():
                assert cell["interpretation"] == "diagnostic only — no conf-matching"


def test_analyze_does_not_mutate_q_input(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    raw_before = input_path.read_bytes()
    hash_before = hashlib.sha256(raw_before).hexdigest()
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    raw_after = input_path.read_bytes()
    hash_after = hashlib.sha256(raw_after).hexdigest()
    assert hash_before == hash_after, (
        "analyze_lhs мутировал входной Q-файл — это запрещено"
    )


def test_full_run_produces_expected_artifacts(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    expected_json = [
        "analysis_pairwise.json",
        "analysis_sensitivity.json",
        "analysis_program_effect.json",
        "analysis_gossip_effect.json",
        "analysis_risk_utility.json",
        "analysis_llm_ranker_diagnostic.json",
        "analysis_stability.json",
    ]
    for name in expected_json:
        assert (out_dir / name).exists(), f"missing {name}"
    assert (out_dir / "synthetic.md").exists()
    # 3 plots
    plots = list((out_dir / "plots").glob("*.png"))
    assert len(plots) >= 3, f"expected >= 3 plots, got {len(plots)}"


def test_capacity_audit_runs_without_conference_json(tmp_path: Path):
    """Если --conference-json пустой/невалидный — секция строится без
    capacity numbers, но отчёт собирается."""
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    rc = az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
        "--conference-json", "",
    ])
    assert rc == 0
    audit = json.loads((out_dir / "analysis_capacity_audit.json").read_text())
    assert audit["conference"]["loaded"] is False
    # overload_occurrence всё равно посчитан
    assert "n_lhs_with_any_overload_p123" in audit["overload_occurrence"]


def test_capacity_audit_with_real_mobius(tmp_path: Path):
    """С реальной mobius_2025_autumn.json: проверить что numbers разумные."""
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    real_conf = (EXPERIMENTS_ROOT
                 / "data/conferences/mobius_2025_autumn.json")
    if not real_conf.exists():
        pytest.skip("mobius_2025_autumn.json отсутствует")
    rc = az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
        "--conference-json", str(real_conf),
    ])
    assert rc == 0
    audit = json.loads((out_dir / "analysis_capacity_audit.json").read_text())
    conf = audit["conference"]
    assert conf["loaded"] is True
    assert conf["n_slots"] > 0
    assert conf["per_slot_capacity_min"] >= 1
    assert conf["per_slot_capacity_min"] <= conf["per_slot_capacity_max"]


def test_capacity_audit_overload_consistency(tmp_path: Path):
    """fraction_lhs_with_any_overload_p123 = n / total."""
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
        "--conference-json", "",
    ])
    audit = json.loads((out_dir / "analysis_capacity_audit.json").read_text())
    occ = audit["overload_occurrence"]
    expected = occ["n_lhs_with_any_overload_p123"] / occ["n_lhs_total"]
    assert occ["fraction_lhs_with_any_overload_p123"] == pytest.approx(expected)


def test_winrate_eps_sums_in_real_pairwise_block(tmp_path: Path):
    az = _load_analyze()
    input_path = _build_minimal_dataset(tmp_path)
    out_dir = tmp_path / "out"
    az.main([
        "--input", str(input_path),
        "--output-dir", str(out_dir),
        "--report-name", "synthetic.md",
    ])
    pairwise = json.loads((out_dir / "analysis_pairwise.json").read_text())
    full_pairs = pairwise["full_50"]["metric_mean_overload_excess"]
    for pair_label, stat in full_pairs.items():
        if stat["n_paired"] == 0:
            continue
        s = stat["win_eps"] + stat["loss_eps"] + stat["ties_eps"]
        assert abs(s - 1.0) < 1e-9, (
            f"{pair_label}: win+loss+ties={s} ≠ 1"
        )
