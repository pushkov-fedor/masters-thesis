"""Smoke этапа P PIVOT_IMPLEMENTATION_PLAN r5: малый LHS на Mobius.

Цель — проверить инфраструктуру этапа P (lhs.py + seeds.py +
run_lhs_parametric.py), не запуская полный Q-прогон.

Что проверяется (Accepted decision уточнение 5 spike_experiment_protocol):
- long-format структура результата (правильные колонки);
- CRN-инвариант: одна аудитория и тот же program_variant между политиками
  внутри одной (lhs_row_id, replicate) пары;
- связка LHS → audience selection → Φ → policies → simulator → metrics
  работает без падений;
- wallclock ≤ 5 минут (PIVOT этап P).

Что НЕ проверяется:
- EC3 на случайных LHS-точках (на random LHS-5 точках `w_rec=0` почти не
  появится; EC3 уже покрыт forced-row тестом `test_ec3_invariance_when_w_rec_zero`
  в pytest этапа I).

П4 (`llm_ranker`) по умолчанию НЕ активируется в smoke, чтобы не делать
реальные API-вызовы. Можно включить через `--include-llm-ranker`.

Запуск:
    .venv/bin/python scripts/run_smoke_lhs.py --conference mobius_2025_autumn
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import run_lhs  # noqa: E402


REQUIRED_KEYS = {
    "lhs_row_id", "capacity_multiplier", "popularity_source",
    "w_rel", "w_rec", "w_gossip",
    "audience_size", "program_variant",
    "policy", "replicate",
    "audience_seed", "phi_seed", "cfg_seed",
    "is_maximin_point",
}

REQUIRED_METRIC_KEYS = {
    "metric_mean_overload_excess",
    "metric_mean_user_utility",
    "metric_overflow_rate_slothall",
    "metric_hall_utilization_variance",
}


def check_smoke_acceptance(out: dict, wallclock_s: float) -> dict:
    """Возвращает словарь acceptance-чеков. Все True = smoke PASS."""
    rows = out["results"]
    checks: dict = {}

    # 1. Long-format structure
    if not rows:
        checks["long_format_ok"] = False
        checks["long_format_missing"] = list(REQUIRED_KEYS)
        return checks
    actual_keys = set(rows[0].keys())
    missing = REQUIRED_KEYS - actual_keys
    checks["long_format_ok"] = (len(missing) == 0)
    checks["long_format_missing"] = sorted(missing)

    # 2. Метрики присутствуют
    metric_keys_present = {k for k in rows[0] if k.startswith("metric_")}
    metric_missing = REQUIRED_METRIC_KEYS - metric_keys_present
    checks["metrics_present"] = (len(metric_missing) == 0)
    checks["metrics_missing"] = sorted(metric_missing)

    # 3. CRN: audience_seed/phi_seed одинаковы для (lhs_row_id, replicate)
    #    между разными политиками.
    crn_violations = []
    by_key: dict = {}
    for r in rows:
        key = (r["lhs_row_id"], r["replicate"])
        by_key.setdefault(key, []).append(r)
    for key, group in by_key.items():
        seeds_seen = {(r["audience_seed"], r["phi_seed"]) for r in group}
        if len(seeds_seen) > 1:
            crn_violations.append({"key": list(key),
                                   "seeds_seen": [list(s) for s in seeds_seen]})
    checks["crn_audience_phi_invariant"] = (len(crn_violations) == 0)
    checks["crn_violations"] = crn_violations

    # 4. cfg_seed = replicate
    cfg_seed_ok = all(r["cfg_seed"] == r["replicate"] for r in rows)
    checks["cfg_seed_equals_replicate"] = cfg_seed_ok

    # 5. n_results
    checks["n_results"] = len(rows)
    checks["n_results_positive"] = (len(rows) > 0)

    # 6. wallclock
    checks["wallclock_s"] = float(wallclock_s)
    checks["wallclock_within_budget"] = (wallclock_s <= 300.0)

    # 7. Никаких падений (если дошли сюда — нет exceptions)
    checks["no_exceptions"] = True

    return checks


def smoke_passed(checks: dict) -> bool:
    return all([
        checks.get("long_format_ok", False),
        checks.get("metrics_present", False),
        checks.get("crn_audience_phi_invariant", False),
        checks.get("cfg_seed_equals_replicate", False),
        checks.get("n_results_positive", False),
        checks.get("wallclock_within_budget", False),
        checks.get("no_exceptions", False),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn")
    ap.add_argument("--n-points", type=int, default=5)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--include-llm-ranker", action="store_true",
                    help="Активирует П4 в smoke. По умолчанию НЕ активирована, "
                         "чтобы не делать реальные API-вызовы.")
    args = ap.parse_args()

    # Для малого n_points relax MIN_PER_LEVEL (Accepted decision уточнение 4
    # касается основной матрицы; для smoke с 5 точками порог по умолчанию
    # недостижим, нужен override).
    relaxed_min = {"program_variant": 1, "audience_size": 1,
                   "popularity_source": 1}

    t0 = time.time()
    out = run_lhs(
        conference=args.conference,
        n_points=args.n_points,
        replicates=args.replicates,
        master_seed=args.master_seed,
        maximin_k=min(3, args.n_points),
        include_llm_ranker=args.include_llm_ranker,
        min_per_level=relaxed_min,
        force_pv_zero_in_maximin=False,
        verbose=True,
    )
    elapsed = time.time() - t0
    out["elapsed_total_s"] = elapsed

    checks = check_smoke_acceptance(out, wallclock_s=elapsed)
    out["smoke_checks"] = checks

    date = dt.date.today().isoformat()
    out_path = ROOT / "results" / f"lhs_smoke_{args.conference}_{date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                   default=str))

    print(f"\nWROTE: {out_path}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  n_results: {checks.get('n_results', 0)}")
    print("  checks:")
    for k, v in checks.items():
        print(f"    {k}: {v}")

    passed = smoke_passed(checks)
    print(f"\n=== Smoke acceptance: {'PASS' if passed else 'FAIL'} ===")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
