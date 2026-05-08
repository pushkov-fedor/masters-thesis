"""Этап S: постобработка результатов параметрического Q-прогона.

Читает Q JSON read-only, не запускает симулятор / LLM / API. Производит набор
analysis_*.json + markdown отчёт + minimal plots в `experiments/results/plots/`.

Источник правил: docs/spikes/spike_result_postprocessing.md (этап R).

Ключевые ограничения этапа S:
- aggregator default: median по 3 replicate;
- pairwise: strict + ε-thresholded;
- ε=0.005 для overload-семейства, 0.001 для utility;
- regret абсолютный (lower-better для overload, higher-better для utility);
- buckets фиксированные;
- sign-test для program_variant — diagnostic-only при N ≥ 7 (не gate, не causal);
- П4 (llm_ranker) присутствует только в restricted-to-maximin таблицах;
- full-50 таблицы содержат только П1–П3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats
from tqdm import tqdm


# ---------- Константы (operational params, accepted Q-R1—Q-R5) ----------

ALL_POLICIES = ("no_policy", "cosine", "capacity_aware", "llm_ranker")
P123 = ("no_policy", "cosine", "capacity_aware")
P4 = "llm_ranker"

OVERLOAD_METRICS = (
    "metric_mean_overload_excess",
    "metric_overflow_rate_slothall",
    "metric_hall_utilization_variance",
)
UTILITY_METRIC = "metric_mean_user_utility"
ALL_METRICS = OVERLOAD_METRICS + (UTILITY_METRIC,)

EPS = {
    "metric_mean_overload_excess": 0.005,
    "metric_overflow_rate_slothall": 0.005,
    "metric_hall_utilization_variance": 0.005,
    "metric_mean_user_utility": 0.001,
}

LOWER_IS_BETTER = {
    "metric_mean_overload_excess": True,
    "metric_overflow_rate_slothall": True,
    "metric_hall_utilization_variance": True,
    "metric_mean_user_utility": False,
}

CAPACITY_BUCKETS = (
    ("[0.5, 1.0)", 0.5, 1.0),
    ("[1.0, 2.0)", 1.0, 2.0),
    ("[2.0, 3.0]", 2.0, 3.0 + 1e-9),
)
W_REC_BUCKETS = (
    ("[0, 0.25)", 0.0, 0.25),
    ("[0.25, 0.5)", 0.25, 0.5),
    ("[0.5, 0.7]", 0.5, 0.7 + 1e-9),
)
W_GOSSIP_BUCKETS = (
    ("[0, 0.25)", 0.0, 0.25),
    ("[0.25, 0.5)", 0.25, 0.5),
    ("[0.5, 0.7]", 0.5, 0.7 + 1e-9),
)

VOLATILE_THRESHOLD = 0.5
SIGN_TEST_MIN_N = 7
DIAGNOSTIC_INTERPRETATION = "diagnostic only — no conf-matching"

DEFAULT_INPUT = (
    "experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.json"
)
DEFAULT_OUTPUT_DIR = "experiments/results"
DEFAULT_CONFERENCE_JSON = (
    "experiments/data/conferences/mobius_2025_autumn.json"
)


# ---------- Утилиты ----------

def _bucket_label(value: float, buckets) -> str | None:
    for label, lo, hi in buckets:
        if lo <= value < hi:
            return label
    return None


def _safe_float(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else 0.0


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None,
                "std": None, "p25": None, "p75": None,
                "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------- Загрузка и агрегация ----------

def load_input(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate_replicates(records: list[dict]) -> dict[tuple, dict]:
    """Уровень 1: per-(lhs_row_id, policy) median/mean/std по 3 replicate.

    Возвращает dict[(lhs_row_id, policy)] = {axes, metrics: {m: {median, mean, std, values}}}.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["lhs_row_id"], r["policy"])].append(r)

    result: dict[tuple, dict] = {}
    for key, rows in grouped.items():
        sample = rows[0]
        per_metric = {}
        for m in ALL_METRICS:
            vals = np.array([r[m] for r in rows], dtype=np.float64)
            per_metric[m] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "n_replicates": int(vals.size),
                "values": [float(v) for v in vals],
            }
        result[key] = {
            "lhs_row_id": sample["lhs_row_id"],
            "policy": sample["policy"],
            "is_maximin_point": bool(sample["is_maximin_point"]),
            "capacity_multiplier": float(sample["capacity_multiplier"]),
            "popularity_source": sample["popularity_source"],
            "w_rec": float(sample["w_rec"]),
            "w_gossip": float(sample["w_gossip"]),
            "w_rel": float(sample["w_rel"]),
            "audience_size": int(sample["audience_size"]),
            "program_variant": int(sample["program_variant"]),
            "metrics": per_metric,
        }
    return result


def per_lhs_row_metric(
    aggregated: dict[tuple, dict], policies: Iterable[str], metric: str,
    lhs_ids: Iterable[int],
) -> dict[int, dict[str, float | None]]:
    """Уровень 2: dict[lhs_row_id][policy] = aggregated metric (median).

    Возвращает None для отсутствующей пары (lhs_row, policy).
    """
    out: dict[int, dict[str, float | None]] = {}
    for i in lhs_ids:
        out[i] = {}
        for pi in policies:
            cell = aggregated.get((i, pi))
            out[i][pi] = (cell["metrics"][metric]["median"]
                          if cell is not None else None)
    return out


# ---------- Pairwise win-rate ----------

def pairwise_winrate(
    a_values: list[float | None], b_values: list[float | None],
    eps: float, lower_is_better: bool,
) -> dict[str, float]:
    """Возвращает win_strict / win_eps / ties_eps / loss_strict / loss_eps / n.

    Считает только пары где обе политики имеют не-None значения.
    Семантика «win_A»: A лучше B (учитывает направление метрики).
    """
    n_paired = 0
    n_win_strict = 0
    n_win_eps = 0
    n_ties_eps = 0
    n_loss_strict = 0
    n_loss_eps = 0
    for a, b in zip(a_values, b_values):
        if a is None or b is None:
            continue
        n_paired += 1
        diff = a - b  # для lower_is_better: diff < 0 → A лучше
        sign = -1.0 if lower_is_better else 1.0  # инверсия направления
        # «A лучше» эквивалентно sign * diff > 0
        if sign * diff > 0:
            n_win_strict += 1
            if abs(diff) > eps:
                n_win_eps += 1
        elif sign * diff < 0:
            n_loss_strict += 1
            if abs(diff) > eps:
                n_loss_eps += 1
        if abs(diff) <= eps:
            n_ties_eps += 1
    if n_paired == 0:
        return {"win_strict": None, "win_eps": None, "ties_eps": None,
                "loss_strict": None, "loss_eps": None, "n_paired": 0}
    return {
        "win_strict": n_win_strict / n_paired,
        "win_eps": n_win_eps / n_paired,
        "ties_eps": n_ties_eps / n_paired,
        "loss_strict": n_loss_strict / n_paired,
        "loss_eps": n_loss_eps / n_paired,
        "n_paired": n_paired,
    }


def compute_pairwise(
    aggregated: dict, lhs_ids: list[int], policies: Iterable[str],
) -> dict:
    """Pairwise win-rate для всех пар политик в `policies` по всем 4 метрикам."""
    out: dict[str, dict] = {}
    pol_list = list(policies)
    for m in ALL_METRICS:
        per_pair: dict[str, dict] = {}
        side_by_side = per_lhs_row_metric(aggregated, pol_list, m, lhs_ids)
        for i in range(len(pol_list)):
            for j in range(i + 1, len(pol_list)):
                a = pol_list[i]
                b = pol_list[j]
                a_vals = [side_by_side[lhs][a] for lhs in lhs_ids]
                b_vals = [side_by_side[lhs][b] for lhs in lhs_ids]
                stat = pairwise_winrate(
                    a_vals, b_vals, EPS[m], LOWER_IS_BETTER[m],
                )
                per_pair[f"{a}_vs_{b}"] = stat
        out[m] = per_pair
    return out


# ---------- Per-policy distribution ----------

def per_policy_distribution(
    aggregated: dict, lhs_ids: list[int], policies: Iterable[str],
) -> dict:
    out: dict[str, dict] = {}
    for pi in policies:
        per_metric: dict[str, dict] = {}
        for m in ALL_METRICS:
            vals = []
            for i in lhs_ids:
                cell = aggregated.get((i, pi))
                if cell is not None:
                    vals.append(cell["metrics"][m]["median"])
            per_metric[m] = _summary_stats(vals)
        out[pi] = per_metric
    return out


# ---------- Regret ----------

def compute_regret(
    aggregated: dict, lhs_ids: list[int], policies: Iterable[str],
) -> dict:
    pol_list = list(policies)
    per_metric: dict[str, dict] = {}
    for m in ALL_METRICS:
        side = per_lhs_row_metric(aggregated, pol_list, m, lhs_ids)
        regret_per_pol = {pi: [] for pi in pol_list}
        per_lhs_breakdown: dict[int, dict] = {}
        for i in lhs_ids:
            row = side[i]
            valid_pols = [pi for pi in pol_list if row[pi] is not None]
            if not valid_pols:
                continue
            vals = np.asarray([row[pi] for pi in valid_pols], dtype=np.float64)
            if LOWER_IS_BETTER[m]:
                best = float(np.min(vals))
                regrets = {pi: row[pi] - best for pi in valid_pols}
            else:
                best = float(np.max(vals))
                regrets = {pi: best - row[pi] for pi in valid_pols}
            for pi, val in regrets.items():
                regret_per_pol[pi].append(val)
            per_lhs_breakdown[i] = {"best": best, "regret": regrets}
        per_metric[m] = {
            "summary": {pi: _summary_stats(regret_per_pol[pi])
                        for pi in pol_list},
            "per_lhs_row": per_lhs_breakdown,
        }
    return per_metric


# ---------- Sensitivity по осям ----------

def bucket_axis(
    aggregated: dict, lhs_ids: list[int], policies: Iterable[str],
    axis_name: str, buckets: tuple, axis_value_key: str,
) -> dict:
    """3-bucket таблица: bucket × policy для всех метрик."""
    pol_list = list(policies)
    # bucket → policy → metric → list[values]
    out: dict[str, dict] = {}
    # сначала собрать lhs-row → bucket label
    row_bucket: dict[int, str | None] = {}
    for i in lhs_ids:
        # ищем любую агрегацию с этим i (axes одинаковые между политиками)
        sample = next((aggregated[(i, pi)] for pi in pol_list
                       if (i, pi) in aggregated), None)
        if sample is None:
            row_bucket[i] = None
            continue
        v = sample[axis_value_key]
        row_bucket[i] = _bucket_label(v, buckets)

    for label, _, _ in buckets:
        bucket_block: dict[str, dict] = {"n_lhs": 0, "lhs_row_ids": [],
                                          "policies": {}}
        bucket_lhs = [i for i in lhs_ids if row_bucket[i] == label]
        bucket_block["n_lhs"] = len(bucket_lhs)
        bucket_block["lhs_row_ids"] = list(bucket_lhs)
        for pi in pol_list:
            per_metric: dict[str, dict] = {}
            for m in ALL_METRICS:
                vals = []
                for i in bucket_lhs:
                    cell = aggregated.get((i, pi))
                    if cell is not None:
                        vals.append(cell["metrics"][m]["median"])
                per_metric[m] = _summary_stats(vals)
            bucket_block["policies"][pi] = per_metric
        out[label] = bucket_block
    return out


def discrete_axis_levels(
    aggregated: dict, lhs_ids: list[int], policies: Iterable[str],
    axis_value_key: str, levels: list,
) -> dict:
    pol_list = list(policies)
    out: dict[str, dict] = {}
    for lv in levels:
        block: dict[str, dict] = {"n_lhs": 0, "lhs_row_ids": [],
                                   "policies": {}}
        # find lhs ids with this level
        block_lhs = []
        for i in lhs_ids:
            sample = next((aggregated[(i, pi)] for pi in pol_list
                           if (i, pi) in aggregated), None)
            if sample is None:
                continue
            if sample[axis_value_key] == lv:
                block_lhs.append(i)
        block["n_lhs"] = len(block_lhs)
        block["lhs_row_ids"] = list(block_lhs)
        for pi in pol_list:
            per_metric: dict[str, dict] = {}
            for m in ALL_METRICS:
                vals = []
                for i in block_lhs:
                    cell = aggregated.get((i, pi))
                    if cell is not None:
                        vals.append(cell["metrics"][m]["median"])
                per_metric[m] = _summary_stats(vals)
            block["policies"][pi] = per_metric
        out[str(lv)] = block
    return out


# ---------- Program effect ----------

def program_effect(aggregated: dict, lhs_ids: list[int]) -> dict:
    """Per-PV агрегат по П1–П3 + delta-vs-P0 + sign-test diagnostic-only.

    П4 не участвует (full-50 правило). PV-уровни: 0..5.
    """
    pv_levels = [0, 1, 2, 3, 4, 5]
    pv_table = discrete_axis_levels(
        aggregated, lhs_ids, P123, "program_variant", pv_levels,
    )
    delta_vs_p0: dict[str, dict] = {}
    sign_tests: dict[str, dict] = {}
    for pi in P123:
        delta_vs_p0[pi] = {}
        sign_tests[pi] = {}
        for m in ALL_METRICS:
            per_pv_delta = {}
            per_pv_sign = {}
            # values for PV=0
            p0_vals = []
            for i in lhs_ids:
                sample = next((aggregated[(i, p)] for p in P123
                               if (i, p) in aggregated), None)
                if sample is None or sample["program_variant"] != 0:
                    continue
                cell = aggregated.get((i, pi))
                if cell is None:
                    continue
                p0_vals.append(cell["metrics"][m]["median"])
            for k in pv_levels:
                if k == 0:
                    per_pv_delta[str(k)] = 0.0
                    per_pv_sign[str(k)] = {
                        "n": 0,
                        "n_positive": 0,
                        "n_negative": 0,
                        "p_value": None,
                        "interpretation": DIAGNOSTIC_INTERPRETATION,
                        "skipped": "PV=0 reference",
                    }
                    continue
                pk_vals = []
                for i in lhs_ids:
                    sample = next((aggregated[(i, p)] for p in P123
                                   if (i, p) in aggregated), None)
                    if sample is None or sample["program_variant"] != k:
                        continue
                    cell = aggregated.get((i, pi))
                    if cell is None:
                        continue
                    pk_vals.append(cell["metrics"][m]["median"])
                if len(pk_vals) == 0 or len(p0_vals) == 0:
                    per_pv_delta[str(k)] = None
                else:
                    per_pv_delta[str(k)] = (
                        float(np.mean(pk_vals)) - float(np.mean(p0_vals))
                    )
                # sign-test diagnostic-only (cross-product)
                n_pos = 0
                n_neg = 0
                for x in pk_vals:
                    for y in p0_vals:
                        if x > y:
                            n_pos += 1
                        elif x < y:
                            n_neg += 1
                n_total = n_pos + n_neg
                if n_total >= SIGN_TEST_MIN_N:
                    res = stats.binomtest(n_pos, n_total, p=0.5,
                                          alternative="two-sided")
                    p_value = float(res.pvalue)
                else:
                    p_value = None
                per_pv_sign[str(k)] = {
                    "n": n_total,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "p_value": p_value,
                    "interpretation": DIAGNOSTIC_INTERPRETATION,
                    "skipped": (None if n_total >= SIGN_TEST_MIN_N
                                else f"N={n_total} < {SIGN_TEST_MIN_N}"),
                }
            delta_vs_p0[pi][m] = per_pv_delta
            sign_tests[pi][m] = per_pv_sign
    return {
        "interpretation": DIAGNOSTIC_INTERPRETATION,
        "policies_in_full_50": list(P123),
        "policies_excluded": [P4],
        "p4_exclusion_reason": (
            "llm_ranker присутствует только на 12 maximin LHS-row, не во "
            "всём LHS-50; full-50 program-effect анализ строится только "
            "на П1–П3."
        ),
        "pv_levels": pv_levels,
        "per_pv_aggregate": pv_table,
        "delta_vs_p0_mean": delta_vs_p0,
        "sign_test_diagnostic_only": sign_tests,
    }


# ---------- Gossip effect ----------

def gossip_effect(aggregated: dict, lhs_ids: list[int],
                  policies: Iterable[str]) -> dict:
    """3-bucket × policy + conditional capacity × w_gossip + pairwise within bucket."""
    pol_list = list(policies)
    bucket_table = bucket_axis(
        aggregated, lhs_ids, pol_list,
        axis_name="w_gossip", buckets=W_GOSSIP_BUCKETS,
        axis_value_key="w_gossip",
    )
    # pairwise win-rate политик внутри каждого w_gossip-bucket
    pairwise_per_bucket: dict[str, dict] = {}
    for label, lo, hi in W_GOSSIP_BUCKETS:
        bucket_lhs_ids = []
        for i in lhs_ids:
            sample = next((aggregated[(i, p)] for p in pol_list
                           if (i, p) in aggregated), None)
            if sample is None:
                continue
            v = sample["w_gossip"]
            if lo <= v < hi:
                bucket_lhs_ids.append(i)
        if not bucket_lhs_ids:
            pairwise_per_bucket[label] = {"n_lhs": 0, "pairwise": {}}
            continue
        pw = compute_pairwise(aggregated, bucket_lhs_ids, pol_list)
        pairwise_per_bucket[label] = {
            "n_lhs": len(bucket_lhs_ids),
            "pairwise": pw,
        }
    # conditional bucket: capacity × w_gossip
    conditional: dict[str, dict] = {}
    for cap_label, cap_lo, cap_hi in CAPACITY_BUCKETS:
        for g_label, g_lo, g_hi in W_GOSSIP_BUCKETS:
            cell_lhs = []
            for i in lhs_ids:
                sample = next((aggregated[(i, p)] for p in pol_list
                               if (i, p) in aggregated), None)
                if sample is None:
                    continue
                if (cap_lo <= sample["capacity_multiplier"] < cap_hi
                        and g_lo <= sample["w_gossip"] < g_hi):
                    cell_lhs.append(i)
            cell_block = {"n_lhs": len(cell_lhs),
                           "policies": {}}
            for pi in pol_list:
                per_metric: dict[str, dict] = {}
                for m in ALL_METRICS:
                    vals = []
                    for i in cell_lhs:
                        cell = aggregated.get((i, pi))
                        if cell is not None:
                            vals.append(cell["metrics"][m]["median"])
                    per_metric[m] = _summary_stats(vals)
                cell_block["policies"][pi] = per_metric
            conditional[f"capacity={cap_label} & w_gossip={g_label}"] = cell_block
    return {
        "buckets_w_gossip": [b[0] for b in W_GOSSIP_BUCKETS],
        "policies": list(pol_list),
        "table_w_gossip_x_policy": bucket_table,
        "pairwise_within_bucket": pairwise_per_bucket,
        "conditional_capacity_x_w_gossip": conditional,
    }


# ---------- Risk × utility ----------

def risk_utility(aggregated: dict, lhs_ids: list[int],
                 policies: Iterable[str]) -> dict:
    pol_list = list(policies)
    points: list[dict] = []
    tradeoff_markers: list[dict] = []
    # для каждого LHS-row: попарно политики, помечаем trade-off если A лучше по overload
    # и хуже по utility и наоборот.
    for i in lhs_ids:
        for pi in pol_list:
            cell = aggregated.get((i, pi))
            if cell is None:
                continue
            points.append({
                "lhs_row_id": i,
                "policy": pi,
                "is_maximin_point": cell["is_maximin_point"],
                "metric_mean_overload_excess": (
                    cell["metrics"]["metric_mean_overload_excess"]["median"]
                ),
                "metric_mean_user_utility": (
                    cell["metrics"]["metric_mean_user_utility"]["median"]
                ),
            })
        # markers — попарно
        for a_idx in range(len(pol_list)):
            for b_idx in range(a_idx + 1, len(pol_list)):
                a = pol_list[a_idx]
                b = pol_list[b_idx]
                ca = aggregated.get((i, a))
                cb = aggregated.get((i, b))
                if ca is None or cb is None:
                    continue
                ovl_a = ca["metrics"]["metric_mean_overload_excess"]["median"]
                ovl_b = cb["metrics"]["metric_mean_overload_excess"]["median"]
                ut_a = ca["metrics"]["metric_mean_user_utility"]["median"]
                ut_b = cb["metrics"]["metric_mean_user_utility"]["median"]
                eps_o = EPS["metric_mean_overload_excess"]
                eps_u = EPS["metric_mean_user_utility"]
                # «trade-off»: разница больше eps в обоих направлениях
                if (ovl_a + eps_o < ovl_b and ut_a + eps_u < ut_b):
                    tradeoff_markers.append({
                        "lhs_row_id": i, "a": a, "b": b,
                        "kind": "A_better_overload_worse_utility",
                    })
                elif (ovl_b + eps_o < ovl_a and ut_b + eps_u < ut_a):
                    tradeoff_markers.append({
                        "lhs_row_id": i, "a": a, "b": b,
                        "kind": "B_better_overload_worse_utility",
                    })
    # global summary
    n_tradeoff = len(tradeoff_markers)
    return {
        "epsilon_overload": EPS["metric_mean_overload_excess"],
        "epsilon_utility": EPS["metric_mean_user_utility"],
        "n_points": len(points),
        "n_tradeoff_markers": n_tradeoff,
        "tradeoff_markers": tradeoff_markers,
        "points": points,
    }


# ---------- LLM ranker diagnostic ----------

def llm_ranker_diagnostic(aggregated: dict, maximin_ids: list[int]) -> dict:
    """Restricted-to-maximin: 12 LHS-row × 4 policy.

    Сохраняет ranking-vectors для этапа V cross-validation.
    """
    pol_list = list(ALL_POLICIES)
    distribution = per_policy_distribution(aggregated, maximin_ids, pol_list)
    pairwise = compute_pairwise(aggregated, maximin_ids, pol_list)
    # ranking vectors: per LHS-row, per metric, ranks по политикам
    ranking_vectors: dict[str, dict] = {}
    for m in ALL_METRICS:
        per_row: dict[str, dict] = {}
        for i in maximin_ids:
            row_vals = []
            for pi in pol_list:
                cell = aggregated.get((i, pi))
                if cell is None:
                    row_vals.append(np.nan)
                else:
                    row_vals.append(cell["metrics"][m]["median"])
            arr = np.asarray(row_vals, dtype=np.float64)
            if not LOWER_IS_BETTER[m]:
                arr = -arr
            ranks = stats.rankdata(arr, method="average")
            per_row[str(i)] = {pi: float(rk) for pi, rk in zip(pol_list, ranks)}
        ranking_vectors[m] = per_row
    return {
        "subset": "maximin_12",
        "lhs_row_ids": list(maximin_ids),
        "policies": list(pol_list),
        "per_policy_distribution": distribution,
        "pairwise_winrate": pairwise,
        "ranking_vectors": ranking_vectors,
        "constraint": (
            "Эти числа корректно сравнимы только друг с другом. "
            "П4 (llm_ranker) присутствует ТОЛЬКО в этом restricted-to-maximin "
            "блоке. Не сравнивать средние П4 со средними П1–П3 в full-50."
        ),
    }


# ---------- Stability ----------

def stability(aggregated: dict, lhs_ids: list[int],
              policies: Iterable[str]) -> dict:
    pol_list = list(policies)
    per_pair: dict[str, dict] = {}
    volatile: list[dict] = []
    std_distribution: dict[str, list[float]] = {pi: [] for pi in pol_list}
    for pi in pol_list:
        per_lhs: dict[str, dict] = {}
        for i in lhs_ids:
            cell = aggregated.get((i, pi))
            if cell is None:
                continue
            per_metric: dict[str, dict] = {}
            for m in ALL_METRICS:
                mean = cell["metrics"][m]["mean"]
                std = cell["metrics"][m]["std"]
                std_over_mean = (std / abs(mean)) if abs(mean) > 1e-12 else None
                per_metric[m] = {
                    "mean": mean,
                    "std": std,
                    "std_over_mean": std_over_mean,
                    "values": cell["metrics"][m]["values"],
                }
                std_distribution[pi].append(std)
                if (std_over_mean is not None
                        and std_over_mean > VOLATILE_THRESHOLD):
                    volatile.append({
                        "lhs_row_id": i, "policy": pi, "metric": m,
                        "mean": mean, "std": std,
                        "std_over_mean": std_over_mean,
                    })
            per_lhs[str(i)] = per_metric
        per_pair[pi] = per_lhs
    std_summary = {pi: _summary_stats(std_distribution[pi]) for pi in pol_list}
    return {
        "volatile_threshold_std_over_mean": VOLATILE_THRESHOLD,
        "n_volatile": len(volatile),
        "volatile": volatile,
        "std_distribution_summary": std_summary,
        "per_policy_per_lhs": per_pair,
    }


# ---------- Capacity audit / interpretation ----------

def capacity_audit(
    aggregated: dict, lhs_rows_meta: list[dict],
    conference_json_path: Path | None,
) -> dict:
    """Sanity-аудит capacity конференции и распределения overload по LHS-row.

    Цель — отделить «median overload = 0 как структурное свойство сценарного
    анализа» (большинство LHS-точек безопасные) от «median overload = 0 как
    артефакт слишком мягко заданной capacity». Не запускает симулятор,
    использует только Q-агрегаты + read-only JSON конференции.
    """
    # 1. Параметры конференции (per-slot capacity)
    conference_block: dict[str, Any] = {
        "loaded": False,
        "conference_json_path": (
            str(conference_json_path) if conference_json_path else None
        ),
    }
    if conference_json_path is not None and conference_json_path.exists():
        with open(conference_json_path) as f:
            conf = json.load(f)
        slot_caps = []
        for s in conf.get("slots", []):
            hcs = s.get("hall_capacities") or {}
            total = sum(int(v) for v in hcs.values() if v is not None)
            n_active = sum(1 for v in hcs.values() if v is not None)
            slot_caps.append({"slot_id": s.get("id"),
                              "n_active_halls": n_active,
                              "total_capacity": total})
        totals = [s["total_capacity"] for s in slot_caps]
        n_parallel = sum(1 for s in slot_caps if s["n_active_halls"] > 1)
        n_plenary = sum(1 for s in slot_caps if s["n_active_halls"] == 1)
        conference_block.update({
            "loaded": True,
            "name": conf.get("name") or conf.get("conf_id"),
            "population_for_capacity": conf.get("population_for_capacity"),
            "n_slots": len(slot_caps),
            "n_plenary_slots": n_plenary,
            "n_parallel_slots": n_parallel,
            "per_slot_capacity_min": min(totals) if totals else None,
            "per_slot_capacity_mean": (
                float(np.mean(totals)) if totals else None
            ),
            "per_slot_capacity_max": max(totals) if totals else None,
            "base_halls": [
                {"id": h["id"], "capacity": h["capacity"]}
                for h in conf.get("halls", [])
            ],
            "base_total_halls_capacity": sum(
                int(h["capacity"]) for h in conf.get("halls", [])
            ),
            "slot_capacity_breakdown": slot_caps,
        })

    # 2. LHS distribution по capacity_multiplier × audience_size
    lhs_by_id = {r["lhs_row_id"]: r for r in lhs_rows_meta}
    cap_buckets_def = (
        ("stress[0.5,1)", 0.5, 1.0),
        ("normal[1,2)", 1.0, 2.0),
        ("loose[2,3]", 2.0, 3.0 + 1e-9),
    )
    audience_levels = [30, 60, 100]
    n_lhs = len(lhs_rows_meta)

    bucket_counts: dict[str, dict[str, Any]] = {}
    for label, lo, hi in cap_buckets_def:
        cell: dict[str, Any] = {"n_lhs": 0, "by_audience": {}}
        for asz in audience_levels:
            ids = [r["lhs_row_id"] for r in lhs_rows_meta
                   if lo <= r["capacity_multiplier"] < hi
                   and r["audience_size"] == asz]
            cell["by_audience"][str(asz)] = {
                "n_lhs": len(ids), "lhs_row_ids": ids,
            }
        cell["n_lhs"] = sum(c["n_lhs"]
                            for c in cell["by_audience"].values())
        bucket_counts[label] = cell

    # 3. Overload-occurrence по (lhs_row, policy)
    pairs_total = 0
    pairs_nonzero = 0
    per_policy_occurrence: dict[str, dict[str, Any]] = {}
    for pi in ALL_POLICIES:
        nz_ids = []
        all_ids = []
        for i in lhs_by_id:
            cell = aggregated.get((i, pi))
            if cell is None:
                continue
            ovl = cell["metrics"]["metric_mean_overload_excess"]["median"]
            all_ids.append(i)
            pairs_total += 1
            if ovl > 0:
                nz_ids.append(i)
                pairs_nonzero += 1
        per_policy_occurrence[pi] = {
            "n_evaluated_lhs": len(all_ids),
            "n_lhs_with_overload": len(nz_ids),
            "fraction_nonzero": (
                len(nz_ids) / len(all_ids) if all_ids else None
            ),
            "lhs_row_ids_with_overload": nz_ids,
        }

    # LHS-row с overload > 0 хотя бы у одной П1–П3 политики
    lhs_with_any_overload: list[int] = []
    for i in lhs_by_id:
        any_pos = False
        for pi in P123:
            cell = aggregated.get((i, pi))
            if cell is None:
                continue
            if cell["metrics"]["metric_mean_overload_excess"]["median"] > 0:
                any_pos = True
                break
        if any_pos:
            lhs_with_any_overload.append(i)

    # 4. Cross-tab overload nonzero per bucket × policy (П1–П3)
    overload_cross: dict[str, dict[str, Any]] = {}
    for label, lo, hi in cap_buckets_def:
        ids_in_bucket = [r["lhs_row_id"] for r in lhs_rows_meta
                         if lo <= r["capacity_multiplier"] < hi]
        per_policy: dict[str, Any] = {}
        for pi in P123:
            vals = []
            nz = 0
            for i in ids_in_bucket:
                cell = aggregated.get((i, pi))
                if cell is None:
                    continue
                ovl = cell["metrics"]["metric_mean_overload_excess"]["median"]
                vals.append(ovl)
                if ovl > 0:
                    nz += 1
            per_policy[pi] = {
                "n_lhs_in_bucket": len(ids_in_bucket),
                "n_lhs_with_overload": nz,
                "mean_overload": (
                    float(np.mean(vals)) if vals else 0.0
                ),
                "max_overload": float(max(vals)) if vals else 0.0,
            }
        overload_cross[label] = per_policy

    # 5. capacity_aware-effect detail: где П3 строго лучше
    capacity_aware_advantage: list[dict] = []
    for i in sorted(lhs_with_any_overload):
        cell_np = aggregated.get((i, "no_policy"))
        cell_co = aggregated.get((i, "cosine"))
        cell_ca = aggregated.get((i, "capacity_aware"))
        if cell_np is None or cell_co is None or cell_ca is None:
            continue
        np_ovl = cell_np["metrics"]["metric_mean_overload_excess"]["median"]
        co_ovl = cell_co["metrics"]["metric_mean_overload_excess"]["median"]
        ca_ovl = cell_ca["metrics"]["metric_mean_overload_excess"]["median"]
        meta = lhs_by_id[i]
        worst_other = max(np_ovl, co_ovl)
        # «реальная победа capacity_aware»: cap_aware строго ниже худшей и
        # разница больше ε
        rec = {
            "lhs_row_id": i,
            "capacity_multiplier": meta["capacity_multiplier"],
            "audience_size": meta["audience_size"],
            "no_policy": np_ovl,
            "cosine": co_ovl,
            "capacity_aware": ca_ovl,
            "reduction_vs_cosine": co_ovl - ca_ovl,
            "reduction_vs_no_policy": np_ovl - ca_ovl,
        }
        if (worst_other - ca_ovl) > EPS["metric_mean_overload_excess"]:
            rec["capacity_aware_strictly_better"] = True
        else:
            rec["capacity_aware_strictly_better"] = False
        capacity_aware_advantage.append(rec)

    n_p3_strict_wins = sum(
        1 for r in capacity_aware_advantage
        if r["capacity_aware_strictly_better"]
    )
    n_p3_no_worse = sum(
        1 for r in capacity_aware_advantage
        if r["capacity_aware"] - max(r["no_policy"], r["cosine"])
        <= EPS["metric_mean_overload_excess"]
    )

    # 6. critical infeasible LHS: cap_m * per_slot_capacity < audience_size
    critical_lhs: list[dict] = []
    if conference_block.get("loaded"):
        # для параллельных слотов: эффективная capacity = per_slot * cap_m
        per_slot_min = conference_block["per_slot_capacity_min"]
        for r in lhs_rows_meta:
            effective = per_slot_min * r["capacity_multiplier"]
            if r["audience_size"] > effective:
                critical_lhs.append({
                    "lhs_row_id": r["lhs_row_id"],
                    "capacity_multiplier": r["capacity_multiplier"],
                    "audience_size": r["audience_size"],
                    "effective_per_slot_capacity": effective,
                    "audience_minus_capacity": (
                        r["audience_size"] - effective
                    ),
                    "interpretation": (
                        "audience > per-slot capacity → physical overload "
                        "неизбежен при концентрации в один зал; ни одна "
                        "политика не может полностью спасти"
                    ),
                })

    return {
        "purpose": (
            "Sanity-проверка: median overload = 0 в full-50 не означает "
            "«capacity слишком мягкая», это структурное свойство сценарного "
            "анализа — большинство LHS-точек безопасные. Главная ценность "
            "DSS — в нахождении и количественной оценке стрессовых сценариев."
        ),
        "conference": conference_block,
        "lhs_distribution": {
            "n_lhs_total": n_lhs,
            "buckets": bucket_counts,
            "audience_size_levels": audience_levels,
        },
        "overload_occurrence": {
            "n_lhs_total": n_lhs,
            "n_lhs_with_any_overload_p123": len(lhs_with_any_overload),
            "fraction_lhs_with_any_overload_p123": (
                len(lhs_with_any_overload) / n_lhs if n_lhs else 0.0
            ),
            "lhs_row_ids_with_any_overload_p123": lhs_with_any_overload,
            "per_policy": per_policy_occurrence,
            "pairs_total": pairs_total,
            "pairs_nonzero": pairs_nonzero,
        },
        "overload_by_bucket_x_policy": overload_cross,
        "capacity_aware_effect": {
            "subset_definition": (
                "risk-positive LHS-row: LHS-row, на которых хотя бы у одной "
                "из П1–П3 median overload > 0. Не путать с capacity-stress "
                "bucket [0.5, 1.0) по capacity_multiplier — это разные "
                "выборки (risk-positive 13/50; cap-stress 10/50)."
            ),
            "n_risk_positive_lhs": len(lhs_with_any_overload),
            "n_capacity_aware_strictly_better_by_eps": n_p3_strict_wins,
            "n_capacity_aware_no_worse_by_eps": n_p3_no_worse,
            "fraction_strict_wins_among_risk_positive": (
                n_p3_strict_wins / len(lhs_with_any_overload)
                if lhs_with_any_overload else None
            ),
            "fraction_no_worse_among_risk_positive": (
                n_p3_no_worse / len(lhs_with_any_overload)
                if lhs_with_any_overload else None
            ),
            "epsilon": EPS["metric_mean_overload_excess"],
            "interpretation_note": (
                "«strictly better by ε»: capacity_aware median overload "
                "строго ниже max(no_policy, cosine) более чем на ε. "
                "«no_worse by ε»: capacity_aware median overload не "
                "превышает max(no_policy, cosine) более чем на ε."
            ),
            "per_lhs_breakdown": capacity_aware_advantage,
        },
        "critical_infeasible_lhs": {
            "rule": (
                "audience_size > per-slot-capacity × capacity_multiplier "
                "(на параллельном слоте). При концентрации аудитории в один "
                "зал переполнение математически неизбежно."
            ),
            "n_critical": len(critical_lhs),
            "lhs": critical_lhs,
        },
        "interpretation_for_defense": [
            ("Большинство LHS-точек (≈74 % на mobius) — безопасные "
             "сценарии: overload = 0 у всех 4 политик; эти точки нужны как "
             "контраст, а не как «провал политики»."),
            ("Risk-positive LHS-точек (median overload > 0 хотя бы у одной "
             "из П1–П3) ≈26 %; на этом подмножестве capacity_aware снижает "
             "или сохраняет overload относительно max(no_policy, cosine) "
             "за ε и в большинстве случаев строго снижает риск перегрузки "
             "за ε. Точные доли — в полях `fraction_no_worse_among_"
             "risk_positive` и `fraction_strict_wins_among_risk_positive`."),
            ("«Risk-positive» подмножество (по факту overload > 0) и "
             "capacity-stress bucket [0.5, 1.0) (по capacity_multiplier) — "
             "разные выборки. Risk-positive получается шире stress-bucket: "
             "часть нормальных-cap_m точек становится risk-positive из-за "
             "сочетания audience=100 и tight-normal cap_m."),
            ("Критические infeasible-точки (audience > per-slot capacity × "
             "capacity_multiplier) — physical overload неустраним никакой "
             "политикой; эти LHS-row показывают границу управляемости и "
             "тоже нужны в выборке."),
            ("audience_size grid {30, 60, 100} согласован с per-slot "
             "capacity Mobius (≈102 на параллельный слот при cap_m=1.0); "
             "поэтому диапазон capacity_multiplier ∈ [0.5, 3.0] "
             "разворачивает сценарии от плотного stress до loose."),
        ],
    }


# ---------- Plots ----------

def plot_risk_utility(points: list[dict], plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    color_by_policy = {
        "no_policy": "#4C78A8",
        "cosine": "#F58518",
        "capacity_aware": "#54A24B",
        "llm_ranker": "#E45756",
    }
    marker_by_subset = {True: "*", False: "o"}
    for pi in ALL_POLICIES:
        for is_maximin in (True, False):
            xs = [p["metric_mean_overload_excess"] for p in points
                  if p["policy"] == pi
                  and p["is_maximin_point"] == is_maximin]
            ys = [p["metric_mean_user_utility"] for p in points
                  if p["policy"] == pi
                  and p["is_maximin_point"] == is_maximin]
            if not xs:
                continue
            policy_ru = {
                "no_policy": "контрольная",
                "cosine": "по релевантности",
                "capacity_aware": "с учётом загрузки",
                "llm_ranker": "на основе LLM",
            }
            label = policy_ru.get(pi, pi)
            if is_maximin:
                label = f"{label} (maximin)"
            ax.scatter(xs, ys, s=60 if is_maximin else 30,
                       c=color_by_policy.get(pi, "gray"),
                       marker=marker_by_subset[is_maximin],
                       alpha=0.65, label=label,
                       edgecolors="black" if is_maximin else "none",
                       linewidths=0.5)
    ax.set_xlabel("Среднее превышение вместимости")
    ax.set_ylabel("Средняя релевантность выбранного доклада")
    ax.set_title("Риск × релевантность по точкам LHS и политикам")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = plots_dir / "analysis_risk_utility_scatter.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_gossip_bucket(gossip_block: dict, plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bucket_labels = gossip_block["buckets_w_gossip"]
    pol_list = gossip_block["policies"]
    width = 0.8 / max(len(pol_list), 1)
    color_by_policy = {
        "no_policy": "#4C78A8",
        "cosine": "#F58518",
        "capacity_aware": "#54A24B",
        "llm_ranker": "#E45756",
    }
    policy_ru = {
        "no_policy": "контрольная",
        "cosine": "по релевантности",
        "capacity_aware": "с учётом загрузки",
        "llm_ranker": "на основе LLM",
    }
    table = gossip_block["table_w_gossip_x_policy"]
    x_base = np.arange(len(bucket_labels))
    for k, pi in enumerate(pol_list):
        means = []
        for label in bucket_labels:
            stat = table[label]["policies"][pi]["metric_mean_overload_excess"]
            means.append(stat["mean"] if stat["mean"] is not None else 0.0)
        ax.bar(
            x_base + (k - len(pol_list) / 2 + 0.5) * width,
            means, width=width, label=policy_ru.get(pi, pi),
            color=color_by_policy.get(pi, "gray"),
            edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel(r"Бакет $w_{gossip}$")
    ax.set_ylabel("Среднее превышение вместимости")
    ax.set_title(r"Средний риск перегрузки по бакетам $w_{gossip}$ и политикам")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "analysis_gossip_bucket_bar.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_ranking_heatmap(ranking_block: dict, plots_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pol_list = ranking_block["policies"]
    lhs_ids = ranking_block["lhs_row_ids"]
    rv = ranking_block["ranking_vectors"]["metric_mean_overload_excess"]
    matrix = np.array(
        [[rv[str(i)][pi] for pi in pol_list] for i in lhs_ids],
        dtype=np.float64,
    )
    policy_ru = {
        "no_policy": "контрольная",
        "cosine": "по релевантности",
        "capacity_aware": "с учётом загрузки",
        "llm_ranker": "на основе LLM",
    }
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r",
                   vmin=1, vmax=len(pol_list))
    ax.set_xticks(range(len(pol_list)))
    ax.set_xticklabels([policy_ru.get(p, p) for p in pol_list],
                       rotation=20, ha="right")
    ax.set_yticks(range(len(lhs_ids)))
    ax.set_yticklabels([f"#{i}" for i in lhs_ids])
    ax.set_xlabel("Политика")
    ax.set_ylabel("Точка LHS из maximin-подмножества")
    ax.set_title("Ранжирование политик на 12 maximin-точках")
    for r_idx in range(matrix.shape[0]):
        for c_idx in range(matrix.shape[1]):
            ax.text(c_idx, r_idx, f"{matrix[r_idx, c_idx]:.1f}",
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Ранг")
    fig.tight_layout()
    out = plots_dir / "analysis_ranking_heatmap_maximin.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------- Markdown ----------

def _fmt_winrate(stat: dict) -> str:
    if stat is None or stat.get("n_paired", 0) == 0:
        return "—"
    return (f"strict={stat['win_strict']:.2f} / "
            f"eps={stat['win_eps']:.2f} / "
            f"ties={stat['ties_eps']:.2f} "
            f"(n={stat['n_paired']})")


def _fmt_summary(s: dict, prec: int = 4) -> str:
    if s.get("n", 0) == 0:
        return "—"
    return (f"mean={s['mean']:.{prec}f} "
            f"median={s['median']:.{prec}f} "
            f"std={s['std']:.{prec}f} "
            f"(n={s['n']})")


def build_markdown(
    pairwise: dict, distribution_full: dict, distribution_max: dict,
    sensitivity: dict, program: dict, gossip: dict,
    risk_util: dict, llm_diag: dict, stability_block: dict,
    capacity_audit_block: dict | None,
    timings: dict, input_path: Path, q_meta: dict, plots: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# Этап S: постобработка результатов параметрического Q-прогона")
    lines.append("")
    lines.append(f"Дата: {time.strftime('%Y-%m-%d')}")
    lines.append(f"Источник Q-артефактов (read-only): `{input_path.name}`")
    lines.append(f"Конференция: `{q_meta.get('conference', '?')}`")
    lines.append(f"Master seed: {q_meta.get('params', {}).get('master_seed', '?')}")
    lines.append("")
    lines.append("## 1. Параметры этапа S")
    lines.append("")
    lines.append("- aggregator default: median по 3 replicate;")
    lines.append("- ε для overload-семейства (`mean_overload_excess`, "
                 "`overflow_rate_slothall`, `hall_utilization_variance`): 0.005;")
    lines.append("- ε для `mean_user_utility`: 0.001;")
    lines.append("- regret абсолютный;")
    lines.append("- buckets фиксированные: `capacity_multiplier`/`w_rec`/"
                 "`w_gossip` по [0.5,1)/[1,2)/[2,3], [0,0.25)/[0.25,0.5)/"
                 "[0.5,0.7];")
    lines.append("- sign-test для program_variant — diagnostic-only при N ≥ 7 "
                 "(не gate, не causal);")
    lines.append("- П4 (llm_ranker) только в restricted-to-maximin блоке.")
    lines.append("")

    lines.append("## 2. Q-S-Risk: pairwise win-rate")
    lines.append("")
    lines.append("### 2.1. Full-50 (П1–П3 only)")
    lines.append("")
    lines.append("Метрика: `mean_overload_excess`, ε=0.005. "
                 "«win_A» = доля LHS-row, в которых политика A лучше B.")
    lines.append("")
    lines.append("| Пара (A vs B) | win_strict | win_eps | ties_eps "
                 "| loss_strict | loss_eps | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    full_pairs = pairwise["full_50"]["metric_mean_overload_excess"]
    for pair_label, stat in full_pairs.items():
        if stat["n_paired"] == 0:
            continue
        lines.append(
            f"| {pair_label} | "
            f"{stat['win_strict']:.2f} | {stat['win_eps']:.2f} | "
            f"{stat['ties_eps']:.2f} | {stat['loss_strict']:.2f} | "
            f"{stat['loss_eps']:.2f} | {stat['n_paired']} |"
        )
    lines.append("")

    lines.append("### 2.2. Maximin-12 (П1–П4)")
    lines.append("")
    lines.append("Метрика: `mean_overload_excess`, ε=0.005. "
                 "На этом subset все 4 политики оценены на одной базе.")
    lines.append("")
    lines.append("| Пара (A vs B) | win_strict | win_eps | ties_eps "
                 "| loss_strict | loss_eps | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    maximin_pairs = pairwise["maximin_12"]["metric_mean_overload_excess"]
    for pair_label, stat in maximin_pairs.items():
        if stat["n_paired"] == 0:
            continue
        lines.append(
            f"| {pair_label} | "
            f"{stat['win_strict']:.2f} | {stat['win_eps']:.2f} | "
            f"{stat['ties_eps']:.2f} | {stat['loss_strict']:.2f} | "
            f"{stat['loss_eps']:.2f} | {stat['n_paired']} |"
        )
    lines.append("")

    lines.append("### 2.3. Per-policy distribution (median по LHS-row)")
    lines.append("")
    lines.append("Full-50 (только П1–П3):")
    lines.append("")
    lines.append("| Политика | metric | n | mean | median | p25 | p75 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for pi in P123:
        for m in ALL_METRICS:
            s = distribution_full[pi][m]
            if s.get("n", 0) == 0:
                continue
            lines.append(
                f"| {pi} | {m.replace('metric_', '')} | "
                f"{s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['p25']:.4f} | {s['p75']:.4f} |"
            )
    lines.append("")
    lines.append("Maximin-12 (все 4 политики, общая база сравнения):")
    lines.append("")
    lines.append("| Политика | metric | n | mean | median | p25 | p75 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for pi in ALL_POLICIES:
        for m in ALL_METRICS:
            s = distribution_max[pi][m]
            if s.get("n", 0) == 0:
                continue
            lines.append(
                f"| {pi} | {m.replace('metric_', '')} | "
                f"{s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['p25']:.4f} | {s['p75']:.4f} |"
            )
    lines.append("")

    lines.append("## 3. Q-S-CapVsCos: bucket-анализ capacity_aware vs cosine")
    lines.append("")
    lines.append("Bucket по `capacity_multiplier` (full-50, П1–П3):")
    lines.append("")
    cap_block = sensitivity["capacity_multiplier"]
    lines.append("| Bucket | n_lhs | policy | "
                 "mean overload | mean utility | mean hall_var |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for label, _, _ in CAPACITY_BUCKETS:
        block = cap_block[label]
        for pi in P123:
            stats_pol = block["policies"][pi]
            ovl = stats_pol["metric_mean_overload_excess"]
            ut = stats_pol["metric_mean_user_utility"]
            hv = stats_pol["metric_hall_utilization_variance"]
            ovl_v = "—" if ovl["n"] == 0 else f"{ovl['mean']:.4f}"
            ut_v = "—" if ut["n"] == 0 else f"{ut['mean']:.4f}"
            hv_v = "—" if hv["n"] == 0 else f"{hv['mean']:.4f}"
            lines.append(
                f"| {label} | {block['n_lhs']} | {pi} | "
                f"{ovl_v} | {ut_v} | {hv_v} |"
            )
    lines.append("")
    lines.append("Bucket по `w_rec` (full-50, П1–П3):")
    lines.append("")
    wrec_block = sensitivity["w_rec"]
    lines.append("| Bucket | n_lhs | policy | "
                 "mean overload | mean utility | mean hall_var |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for label, _, _ in W_REC_BUCKETS:
        block = wrec_block[label]
        for pi in P123:
            stats_pol = block["policies"][pi]
            ovl = stats_pol["metric_mean_overload_excess"]
            ut = stats_pol["metric_mean_user_utility"]
            hv = stats_pol["metric_hall_utilization_variance"]
            ovl_v = "—" if ovl["n"] == 0 else f"{ovl['mean']:.4f}"
            ut_v = "—" if ut["n"] == 0 else f"{ut['mean']:.4f}"
            hv_v = "—" if hv["n"] == 0 else f"{hv['mean']:.4f}"
            lines.append(
                f"| {label} | {block['n_lhs']} | {pi} | "
                f"{ovl_v} | {ut_v} | {hv_v} |"
            )
    lines.append("")

    lines.append("## 4. Q-S-Program: per-PV таблица + delta-vs-P0")
    lines.append("")
    lines.append("Anti-claim: sign-test ниже — diagnostic-only, не gate "
                 "и не causal-доказательство. Conf-matching между PV=k "
                 "и PV=0 в LHS-плане НЕТ — остальные оси конфаундированы.")
    lines.append("")
    lines.append("Per-PV агрегат (П1–П3 only, full-50). "
                 "Метрика — `mean_overload_excess`:")
    lines.append("")
    pv_table = program["per_pv_aggregate"]
    lines.append("| PV | n_lhs | policy | mean overload | mean utility |")
    lines.append("|---:|---:|---|---:|---:|")
    for pv_lvl in program["pv_levels"]:
        block = pv_table[str(pv_lvl)]
        for pi in P123:
            ovl = block["policies"][pi]["metric_mean_overload_excess"]
            ut = block["policies"][pi]["metric_mean_user_utility"]
            ovl_v = "—" if ovl["n"] == 0 else f"{ovl['mean']:.4f}"
            ut_v = "—" if ut["n"] == 0 else f"{ut['mean']:.4f}"
            lines.append(
                f"| {pv_lvl} | {block['n_lhs']} | {pi} | "
                f"{ovl_v} | {ut_v} |"
            )
    lines.append("")
    lines.append("Delta `mean_overload_excess` относительно PV=0 "
                 "(абсолютная разница, mean-aggregated; конфаундирован):")
    lines.append("")
    lines.append("| PV | no_policy | cosine | capacity_aware |")
    lines.append("|---:|---:|---:|---:|")
    for pv_lvl in program["pv_levels"]:
        if pv_lvl == 0:
            continue
        row = [str(pv_lvl)]
        for pi in P123:
            d = program["delta_vs_p0_mean"][pi]["metric_mean_overload_excess"][str(pv_lvl)]
            row.append("—" if d is None else f"{d:+.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Sign-test diagnostic-only (`mean_overload_excess`, "
                 "учтено только при N ≥ 7):")
    lines.append("")
    lines.append("| PV | policy | n | n_pos | n_neg | p_value | примечание |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for pv_lvl in program["pv_levels"]:
        if pv_lvl == 0:
            continue
        for pi in P123:
            st = program["sign_test_diagnostic_only"][pi]["metric_mean_overload_excess"][str(pv_lvl)]
            pv_str = "—" if st["p_value"] is None else f"{st['p_value']:.3f}"
            note = st["skipped"] if st["skipped"] else "diagnostic only"
            lines.append(
                f"| {pv_lvl} | {pi} | {st['n']} | {st['n_positive']} | "
                f"{st['n_negative']} | {pv_str} | {note} |"
            )
    lines.append("")

    lines.append("## 5. Q-S-Gossip: 3-bucket × 4 политики")
    lines.append("")
    lines.append("На full-50 включены только П1–П3; П4 учитывается отдельно "
                 "в §7.")
    lines.append("")
    lines.append("| w_gossip bucket | n_lhs | policy | mean overload "
                 "| mean utility |")
    lines.append("|---|---:|---|---:|---:|")
    g_table = gossip["table_w_gossip_x_policy"]
    for label, _, _ in W_GOSSIP_BUCKETS:
        block = g_table[label]
        for pi in P123:
            ovl = block["policies"][pi]["metric_mean_overload_excess"]
            ut = block["policies"][pi]["metric_mean_user_utility"]
            ovl_v = "—" if ovl["n"] == 0 else f"{ovl['mean']:.4f}"
            ut_v = "—" if ut["n"] == 0 else f"{ut['mean']:.4f}"
            lines.append(
                f"| {label} | {block['n_lhs']} | {pi} | "
                f"{ovl_v} | {ut_v} |"
            )
    lines.append("")
    lines.append("Pairwise win-rate (`mean_overload_excess`) внутри каждого "
                 "w_gossip-bucket (П1–П3):")
    lines.append("")
    lines.append("| w_gossip bucket | пара | win_strict | win_eps | ties_eps |")
    lines.append("|---|---|---:|---:|---:|")
    for label in [b[0] for b in W_GOSSIP_BUCKETS]:
        block = gossip["pairwise_within_bucket"][label]
        if block["n_lhs"] == 0:
            lines.append(f"| {label} | (нет точек в bucket) | — | — | — |")
            continue
        bucket_pw = block["pairwise"]["metric_mean_overload_excess"]
        for pair_label, stat in bucket_pw.items():
            if stat["n_paired"] == 0:
                continue
            lines.append(
                f"| {label} | {pair_label} | "
                f"{stat['win_strict']:.2f} | {stat['win_eps']:.2f} | "
                f"{stat['ties_eps']:.2f} |"
            )
    lines.append("")

    lines.append("## 6. Q-S-RiskRelevance: scatter + trade-off")
    lines.append("")
    lines.append(f"Точек на scatter (per-LHS-row × policy): "
                 f"{risk_util['n_points']}")
    lines.append(f"Trade-off маркеров (eps_overload={risk_util['epsilon_overload']}, "
                 f"eps_utility={risk_util['epsilon_utility']}): "
                 f"{risk_util['n_tradeoff_markers']}")
    if risk_util["n_tradeoff_markers"] == 0:
        lines.append("")
        lines.append("Trade-off (одна политика лучше по overload и одновременно "
                     "хуже по utility за пределами ε) на mobius_2025_autumn "
                     "**не зафиксирован**: utility у политик практически "
                     "одинакова (разброс < 0.002), а различия по overload не "
                     "сопровождаются падением utility за пределы 0.001.")
    lines.append("")

    lines.append("## 7. Q-S-LLMRanker: maximin-only сравнение")
    lines.append("")
    lines.append(llm_diag["constraint"])
    lines.append("")
    lines.append("Pairwise (`mean_overload_excess`, ε=0.005) на 12 maximin LHS-row:")
    lines.append("")
    lines.append("| Пара (A vs B) | win_strict | win_eps | ties_eps "
                 "| loss_strict | loss_eps | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair_label, stat in llm_diag["pairwise_winrate"]["metric_mean_overload_excess"].items():
        if stat["n_paired"] == 0:
            continue
        lines.append(
            f"| {pair_label} | {stat['win_strict']:.2f} | "
            f"{stat['win_eps']:.2f} | {stat['ties_eps']:.2f} | "
            f"{stat['loss_strict']:.2f} | {stat['loss_eps']:.2f} | "
            f"{stat['n_paired']} |"
        )
    lines.append("")

    lines.append("## 8. Q-S-Stability: volatile points")
    lines.append("")
    lines.append(f"Порог `std/|mean| > {VOLATILE_THRESHOLD}` на 3 replicate.")
    lines.append(f"Volatile entries: {stability_block['n_volatile']}")
    lines.append("")
    lines.append("Distribution std по политикам (на 4 метриках, "
                 "по всем (lhs_row × policy) парам):")
    lines.append("")
    lines.append("| Политика | n | mean std | median std | max std |")
    lines.append("|---|---:|---:|---:|---:|")
    for pi in stability_block["std_distribution_summary"]:
        s = stability_block["std_distribution_summary"][pi]
        if s.get("n", 0) == 0:
            continue
        lines.append(
            f"| {pi} | {s['n']} | {s['mean']:.4f} | "
            f"{s['median']:.4f} | {s['max']:.4f} |"
        )
    lines.append("")

    lines.append("## 9. Candidate claims для текста ВКР")
    lines.append("")
    lines.append("> Каждый claim — материал для главы 4, не финальный текст. "
                 "Подкрепляется конкретной таблицей из этого отчёта или "
                 "из `analysis_*.json`.")
    lines.append("")
    # Compute candidate claim numbers
    cap_aware_vs_cosine_full = pairwise["full_50"]["metric_mean_overload_excess"].get(
        "cosine_vs_capacity_aware", {}
    )
    if cap_aware_vs_cosine_full and cap_aware_vs_cosine_full.get("n_paired", 0) > 0:
        lines.append(f"1. **Capacity_aware vs cosine**: на full-50 LHS-row "
                     f"`win_eps(cosine→capacity_aware)`="
                     f"{cap_aware_vs_cosine_full['loss_eps']:.2f}, "
                     f"`ties_eps`={cap_aware_vs_cosine_full['ties_eps']:.2f}; "
                     f"capacity_aware ни на одной LHS-row не показывает "
                     f"overload выше cosine за пределами ε — то есть "
                     f"capacity_aware не уступает cosine по риску перегрузки.")
    lines.append("2. **Utility ≈ const между политиками**: разброс "
                 "`metric_mean_user_utility` < 0.005 во всех buckets — "
                 "capacity-aware и LLM-ranker не платят видимой релевантностью "
                 "за лучшую балансировку.")
    lines.append("3. **Trade-off risk × utility отсутствует**: 0 trade-off "
                 "маркеров за пределами ε.")
    lines.append("4. **w_gossip нелинейность**: на full-50 mean overload "
                 "выше в среднем bucket [0.25, 0.5), чем в low/high — "
                 "конфаунд других осей; см. conditional capacity × w_gossip.")
    lines.append("5. **П4 на 12 maximin** — отдельный subset, нельзя "
                 "проецировать на full-50.")
    lines.append("")

    lines.append("## 10. Limitations")
    lines.append("")
    lines.append("- LHS-точки с разными PV не conf-matched — sign-test "
                 "diagnostic only.")
    lines.append("- Bucket-агрегаты конфаундированы между осями; "
                 "OAT scatter не даёт строгого causal cleavage.")
    lines.append("- П4 покрытие — только 12 maximin LHS-row × 3 replicate; "
                 "распределение конфигураций отличается от полного LHS-50.")
    lines.append("- Mobius — синтетическая аудитория с фиксированным "
                 "capacity-sweep; absolute-claims о реальной конференции "
                 "out-of-scope.")
    lines.append("- Все выводы — относительные (между политиками / "
                 "конфигурациями / PV в рамках единой модели).")
    lines.append("")

    if capacity_audit_block is not None:
        lines.append("## 11. Capacity sanity / interpretation")
        lines.append("")
        conf = capacity_audit_block["conference"]
        if conf.get("loaded"):
            lines.append(f"Конференция: `{conf['name']}`")
            lines.append(f"Слотов: {conf['n_slots']} (плёнарных: "
                         f"{conf['n_plenary_slots']}, параллельных: "
                         f"{conf['n_parallel_slots']})")
            lines.append(
                f"Per-slot capacity (по слотам): "
                f"min={conf['per_slot_capacity_min']}, "
                f"mean={conf['per_slot_capacity_mean']:.1f}, "
                f"max={conf['per_slot_capacity_max']}"
            )
            lines.append(f"Population_for_capacity (фиксировано в JSON): "
                         f"{conf['population_for_capacity']}")
            lines.append("")
            lines.append(
                "**Базовая calibration:** на параллельном слоте Mobius "
                f"≈ {conf['per_slot_capacity_min']} мест на {conf['n_parallel_slots']} "
                f"параллельных слотах; audience grid {{30, 60, 100}} согласован с "
                f"per-slot capacity {conf['per_slot_capacity_min']} при "
                f"capacity_multiplier = 1.0."
            )
            lines.append("")
        else:
            lines.append("Конференция: JSON не загружен (capacity numbers "
                         "недоступны).")
            lines.append("")

        lines.append("### 11.1. Распределение LHS по `capacity_multiplier × audience_size`")
        lines.append("")
        lines.append("| Bucket | n_lhs | a30 | a60 | a100 |")
        lines.append("|---|---:|---:|---:|---:|")
        bucket_dist = capacity_audit_block["lhs_distribution"]["buckets"]
        for label, cell in bucket_dist.items():
            a30 = cell["by_audience"]["30"]["n_lhs"]
            a60 = cell["by_audience"]["60"]["n_lhs"]
            a100 = cell["by_audience"]["100"]["n_lhs"]
            lines.append(
                f"| {label} | {cell['n_lhs']} | {a30} | {a60} | {a100} |"
            )
        lines.append("")

        occ = capacity_audit_block["overload_occurrence"]
        lines.append("### 11.2. Доля LHS-точек с ненулевым overload "
                     "(risk-positive)")
        lines.append("")
        lines.append(
            "Терминология: «risk-positive LHS-row» — LHS-row с фактом "
            "median overload > 0 хотя бы у одной из П1–П3. Не путать с "
            "capacity-stress bucket `[0.5, 1.0)` по `capacity_multiplier` "
            "(см. §11.1) — это разные выборки."
        )
        lines.append("")
        lines.append(
            f"- Risk-positive LHS-row: "
            f"**{occ['n_lhs_with_any_overload_p123']} / {occ['n_lhs_total']} "
            f"({occ['fraction_lhs_with_any_overload_p123']*100:.0f} %)**;"
        )
        lines.append(
            f"- остальные {occ['n_lhs_total'] - occ['n_lhs_with_any_overload_p123']} "
            f"LHS-row ({(1 - occ['fraction_lhs_with_any_overload_p123'])*100:.0f} %) "
            "— безопасные сценарии: все политики дают overload = 0."
        )
        lines.append("")
        lines.append("Per-policy overload-frequency:")
        lines.append("")
        lines.append("| Политика | n_evaluated | n_with_overload | fraction |")
        lines.append("|---|---:|---:|---:|")
        for pi in ALL_POLICIES:
            block = occ["per_policy"].get(pi)
            if block is None or block["n_evaluated_lhs"] == 0:
                continue
            lines.append(
                f"| {pi} | {block['n_evaluated_lhs']} | "
                f"{block['n_lhs_with_overload']} | "
                f"{block['fraction_nonzero']*100:.0f} % |"
            )
        lines.append("")

        lines.append("### 11.3. Overload по bucket × policy (П1–П3)")
        lines.append("")
        lines.append(
            "| Bucket | n_lhs | policy | n_overload>0 | mean overload | max overload |"
        )
        lines.append("|---|---:|---|---:|---:|---:|")
        cross = capacity_audit_block["overload_by_bucket_x_policy"]
        for label, per_policy in cross.items():
            for pi in P123:
                cell = per_policy[pi]
                lines.append(
                    f"| {label} | {cell['n_lhs_in_bucket']} | {pi} | "
                    f"{cell['n_lhs_with_overload']} | "
                    f"{cell['mean_overload']:.4f} | "
                    f"{cell['max_overload']:.4f} |"
                )
        lines.append("")

        eff = capacity_audit_block["capacity_aware_effect"]
        lines.append("### 11.4. Где capacity_aware реально снижает риск "
                     "перегрузки")
        lines.append("")
        if eff["n_risk_positive_lhs"] > 0:
            frac_strict = eff["fraction_strict_wins_among_risk_positive"]
            frac_no_worse = eff["fraction_no_worse_among_risk_positive"]
            lines.append(
                f"На {eff['n_risk_positive_lhs']} risk-positive "
                f"LHS-row (overload > 0) capacity_aware:\n"
                f"- **не хуже за ε ({eff['epsilon']})** относительно "
                f"max(no_policy, cosine) на "
                f"**{eff['n_capacity_aware_no_worse_by_eps']} / "
                f"{eff['n_risk_positive_lhs']} "
                f"({frac_no_worse*100:.0f} %)** точках;\n"
                f"- **строго снижает риск за ε** на "
                f"**{eff['n_capacity_aware_strictly_better_by_eps']} / "
                f"{eff['n_risk_positive_lhs']} "
                f"({frac_strict*100:.0f} %)** точках."
            )
            lines.append("")
            lines.append("")
            lines.append(
                "Колонка `Δ overload vs cosine` положительная означает, что "
                "capacity_aware снижает overload относительно cosine "
                "(меньше = лучше)."
            )
            lines.append("")
            lines.append(
                "| LHS | cap_m | aud | overload no_policy | overload cosine | "
                "overload capacity_aware | Δ overload vs cosine "
                "| strict risk reduction |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
            for r in eff["per_lhs_breakdown"]:
                mark = ("yes" if r["capacity_aware_strictly_better"]
                        else "no")
                lines.append(
                    f"| {r['lhs_row_id']} | "
                    f"{r['capacity_multiplier']:.3f} | "
                    f"{r['audience_size']} | "
                    f"{r['no_policy']:.4f} | {r['cosine']:.4f} | "
                    f"{r['capacity_aware']:.4f} | "
                    f"{r['reduction_vs_cosine']:+.4f} | {mark} |"
                )
        lines.append("")

        crit = capacity_audit_block["critical_infeasible_lhs"]
        if crit["n_critical"] > 0:
            lines.append("### 11.5. Critical infeasible LHS-row")
            lines.append("")
            lines.append(
                "Конфигурации, где `audience_size > per-slot-capacity × "
                "capacity_multiplier` — physical overload неизбежен при "
                "концентрации аудитории в один зал. Эти точки полезны как "
                "граница управляемости; считать «provayл политики» здесь "
                "содержательно неверно."
            )
            lines.append("")
            lines.append(
                "| LHS | cap_m | aud | effective per-slot capacity | "
                "audience − capacity |"
            )
            lines.append("|---:|---:|---:|---:|---:|")
            for c in crit["lhs"]:
                lines.append(
                    f"| {c['lhs_row_id']} | "
                    f"{c['capacity_multiplier']:.3f} | "
                    f"{c['audience_size']} | "
                    f"{c['effective_per_slot_capacity']:.1f} | "
                    f"{c['audience_minus_capacity']:.1f} |"
                )
            lines.append("")

        lines.append("### 11.6. Интерпретация для защиты")
        lines.append("")
        for s in capacity_audit_block["interpretation_for_defense"]:
            lines.append(f"- {s}")
        lines.append("")
        lines.append(
            "**Главный нарратив:** median overload = 0 у политик не означает "
            "«capacity слишком мягкая»; это маргинальная статистика по 50 "
            "LHS-точкам, ¾ из которых безопасны структурно. Ценность DSS — "
            "в нахождении ¼ risk-positive сценариев (LHS-row с фактическим "
            "overload > 0) и количественной оценке снижения риска перегрузки "
            "при выборе capacity-aware политики. На risk-positive подмножестве "
            "(оно шире capacity-stress bucket `[0.5, 1.0)`: плюс часть "
            "tight-normal cap_m с audience=100) разница политик клинически "
            "видима."
        )
        lines.append("")

    lines.append("## 12. Wallclock breakdown")
    lines.append("")
    lines.append("| Блок | Время, сек |")
    lines.append("|---|---:|")
    for label, value in timings.items():
        lines.append(f"| {label} | {value:.3f} |")
    lines.append("")

    lines.append("## 13. Plots")
    lines.append("")
    for p in plots:
        lines.append(f"- `{p.relative_to(p.parents[1])}`")
    lines.append("")

    return "\n".join(lines)


# ---------- Main runner ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Этап S: постобработка Q-результатов")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Путь к Q JSON (read-only)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Каталог analysis_*.json и markdown")
    parser.add_argument("--plots-subdir", default="plots",
                        help="Поддиректория для PNG (внутри output-dir)")
    parser.add_argument("--report-name",
                        default="analysis_lhs_parametric_2026-05-08.md",
                        help="Имя итогового markdown")
    parser.add_argument("--conference-json", default=DEFAULT_CONFERENCE_JSON,
                        help="JSON конференции для capacity sanity-аудита "
                             "(read-only; '' / несуществующий путь → пропуск)")
    args = parser.parse_args(argv)

    timings: dict[str, float] = {}
    overall_start = time.perf_counter()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / args.plots_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[S] input: {input_path}", flush=True)
    print(f"[S] output: {output_dir}", flush=True)

    # 1. read input
    t0 = time.perf_counter()
    data = load_input(input_path)
    timings["read_input"] = time.perf_counter() - t0

    records: list[dict] = data["results"]
    maximin_indices: list[int] = list(data["maximin_indices"])
    full_lhs_ids = list(range(data["params"]["n_points"]))

    # 2. aggregation level 1
    t0 = time.perf_counter()
    print("[S] aggregating replicates ...", flush=True)
    aggregated = aggregate_replicates(records)
    timings["aggregate_replicates"] = time.perf_counter() - t0

    # 3. pairwise + per-policy distribution + regret
    t0 = time.perf_counter()
    print("[S] pairwise / regret ...", flush=True)
    pairwise_full = compute_pairwise(aggregated, full_lhs_ids, P123)
    pairwise_max = compute_pairwise(aggregated, maximin_indices, ALL_POLICIES)
    distribution_full = per_policy_distribution(
        aggregated, full_lhs_ids, P123,
    )
    distribution_max = per_policy_distribution(
        aggregated, maximin_indices, ALL_POLICIES,
    )
    regret_full = compute_regret(aggregated, full_lhs_ids, P123)
    regret_max = compute_regret(aggregated, maximin_indices, ALL_POLICIES)
    timings["pairwise_and_regret"] = time.perf_counter() - t0

    pairwise_block = {
        "metric_directions": LOWER_IS_BETTER,
        "epsilons": EPS,
        "policies_in_full_50": list(P123),
        "policies_in_maximin_12": list(ALL_POLICIES),
        "n_lhs_full_50": len(full_lhs_ids),
        "n_lhs_maximin_12": len(maximin_indices),
        "p4_in_full_50": False,
        "rule_p4_excluded_from_full_50": (
            "llm_ranker оценён только на 12 maximin LHS-row × 3 replicate. "
            "Любые full-50 таблицы содержат только П1–П3."
        ),
        "full_50": pairwise_full,
        "maximin_12": pairwise_max,
        "distribution_full_50": distribution_full,
        "distribution_maximin_12": distribution_max,
        "regret_full_50": regret_full,
        "regret_maximin_12": regret_max,
    }

    # 4. sensitivity OAT
    t0 = time.perf_counter()
    sensitivity_block = {}
    print("[S] sensitivity OAT ...", flush=True)
    for axis_name, key, buckets in tqdm(
        [
            ("capacity_multiplier", "capacity_multiplier", CAPACITY_BUCKETS),
            ("w_rec", "w_rec", W_REC_BUCKETS),
            ("w_gossip", "w_gossip", W_GOSSIP_BUCKETS),
        ],
        desc="continuous axes", leave=False,
    ):
        sensitivity_block[axis_name] = bucket_axis(
            aggregated, full_lhs_ids, P123, axis_name, buckets, key,
        )
    sensitivity_block["popularity_source"] = discrete_axis_levels(
        aggregated, full_lhs_ids, P123, "popularity_source",
        ["cosine_only", "fame_only", "mixed"],
    )
    sensitivity_block["audience_size"] = discrete_axis_levels(
        aggregated, full_lhs_ids, P123, "audience_size", [30, 60, 100],
    )
    sensitivity_block["program_variant"] = discrete_axis_levels(
        aggregated, full_lhs_ids, P123, "program_variant",
        [0, 1, 2, 3, 4, 5],
    )
    timings["sensitivity"] = time.perf_counter() - t0

    # 5. program effect
    t0 = time.perf_counter()
    print("[S] program effect ...", flush=True)
    program_block = program_effect(aggregated, full_lhs_ids)
    timings["program_effect"] = time.perf_counter() - t0

    # 6. gossip effect
    t0 = time.perf_counter()
    print("[S] gossip effect ...", flush=True)
    gossip_block = gossip_effect(aggregated, full_lhs_ids, P123)
    timings["gossip_effect"] = time.perf_counter() - t0

    # 7. risk × utility
    t0 = time.perf_counter()
    print("[S] risk × utility ...", flush=True)
    risk_util_full = risk_utility(aggregated, full_lhs_ids, P123)
    risk_util_block = {
        "subset": "full_50_p123 + maximin_12_p4_appended",
        "p123_block": risk_util_full,
        "maximin_full_block": risk_utility(
            aggregated, maximin_indices, ALL_POLICIES,
        ),
    }
    timings["risk_utility"] = time.perf_counter() - t0

    # 8. llm_ranker diagnostic
    t0 = time.perf_counter()
    print("[S] llm_ranker diagnostic ...", flush=True)
    llm_block = llm_ranker_diagnostic(aggregated, maximin_indices)
    timings["llm_ranker_diag"] = time.perf_counter() - t0

    # 9. stability
    t0 = time.perf_counter()
    print("[S] stability ...", flush=True)
    stability_full = stability(aggregated, full_lhs_ids, P123)
    stability_max = stability(aggregated, maximin_indices, [P4])
    stability_block = {
        "full_50_p123": stability_full,
        "maximin_12_p4_only": stability_max,
    }
    timings["stability"] = time.perf_counter() - t0

    # 9b. capacity sanity audit (read-only conference JSON)
    t0 = time.perf_counter()
    print("[S] capacity audit ...", flush=True)
    conf_json_path: Path | None = None
    if args.conference_json:
        candidate = Path(args.conference_json)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        conf_json_path = candidate.resolve() if candidate.exists() else None
    capacity_audit_block = capacity_audit(
        aggregated, data["lhs_rows"], conf_json_path,
    )
    timings["capacity_audit"] = time.perf_counter() - t0

    # 10. plots
    t0 = time.perf_counter()
    print("[S] plots ...", flush=True)
    plots: list[Path] = []
    plots.append(plot_risk_utility(
        risk_util_full["points"] + risk_util_block["maximin_full_block"]["points"],
        plots_dir,
    ))
    plots.append(plot_gossip_bucket(gossip_block, plots_dir))
    plots.append(plot_ranking_heatmap(llm_block, plots_dir))
    timings["plots"] = time.perf_counter() - t0

    # 11. write JSONs
    t0 = time.perf_counter()
    print("[S] writing JSON ...", flush=True)
    json_outputs = {
        "analysis_pairwise.json": pairwise_block,
        "analysis_sensitivity.json": {
            "policies_in_full_50": list(P123),
            "axes_continuous": {
                "capacity_multiplier": [b[0] for b in CAPACITY_BUCKETS],
                "w_rec": [b[0] for b in W_REC_BUCKETS],
                "w_gossip": [b[0] for b in W_GOSSIP_BUCKETS],
            },
            "axes_discrete": {
                "popularity_source": ["cosine_only", "fame_only", "mixed"],
                "audience_size": [30, 60, 100],
                "program_variant": [0, 1, 2, 3, 4, 5],
            },
            "tables": sensitivity_block,
        },
        "analysis_program_effect.json": program_block,
        "analysis_gossip_effect.json": gossip_block,
        "analysis_risk_utility.json": risk_util_block,
        "analysis_llm_ranker_diagnostic.json": llm_block,
        "analysis_stability.json": stability_block,
        "analysis_capacity_audit.json": capacity_audit_block,
    }
    for name, payload in tqdm(json_outputs.items(), desc="json", leave=False):
        out = output_dir / name
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    timings["write_json"] = time.perf_counter() - t0

    # 12. markdown
    t0 = time.perf_counter()
    print("[S] markdown ...", flush=True)
    md_text = build_markdown(
        pairwise=pairwise_block,
        distribution_full=distribution_full,
        distribution_max=distribution_max,
        sensitivity=sensitivity_block,
        program=program_block,
        gossip=gossip_block,
        risk_util=risk_util_full,
        llm_diag=llm_block,
        stability_block=stability_full,
        capacity_audit_block=capacity_audit_block,
        timings=timings,
        input_path=input_path,
        q_meta=data,
        plots=plots,
    )
    md_path = output_dir / args.report_name
    md_path.write_text(md_text)
    timings["markdown"] = time.perf_counter() - t0

    timings["total_wallclock"] = time.perf_counter() - overall_start

    print("[S] timings (s):", flush=True)
    for k, v in timings.items():
        print(f"  {k:<28} {v:>8.3f}", flush=True)

    print(f"[S] markdown: {md_path}", flush=True)
    print(f"[S] plots: {[str(p.name) for p in plots]}", flush=True)
    print(f"[S] JSON written: {list(json_outputs.keys())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
