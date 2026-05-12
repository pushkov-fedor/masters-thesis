"""Runner полного параметрического LHS-прогона (этапы P/Q PIVOT_IMPLEMENTATION_PLAN r5).

Реализация по принятому memo O (`docs/spikes/spike_experiment_protocol.md`),
Q-O9 accepted = вариант (в) компромисс:

- П1–П3 (`no_policy`, `cosine`, `capacity_aware`) на всех 50 LHS-точках × 3 seed
  = 450 evals;
- П4 (`llm_ranker`) только на 12 maximin-точках × 3 seed = 36 evals;
  активируется флагом `--include-llm-ranker`;
- ИТОГО полный параметрический Q = **486 evals**.

CRN-контракт: `audience_seed`/`phi_seed` фикс по lhs_row_id (одинаковая
аудитория и program_variant между политиками внутри LHS-точки и между
seed-репликами); `cfg_seed = replicate` варьируется только между репликами.

В этап P этот скрипт **создаётся как инфраструктура**, полный 486-eval
прогон в этап Q запускается отдельным сообщением пользователя.

Запуск (smoke / dry-run без П4):
    .venv/bin/python scripts/run_lhs_parametric.py \\
        --conference mobius_2025_autumn --n-points 5 --replicates 1

Полный Q-прогон (с П4 на maximin subset):
    .venv/bin/python scripts/run_lhs_parametric.py \\
        --conference mobius_2025_autumn --n-points 50 --replicates 3 \\
        --maximin-k 12 --include-llm-ranker
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lhs import (  # noqa: E402
    generate_lhs,
    maximin_subset,
    DEFAULT_MIN_PER_LEVEL,
)
from src.metrics import (  # noqa: E402
    hall_utilization_variance,
    mean_hall_overload_excess,
    mean_user_utility,
    overflow_rate,
)
from src.policies.registry import active_policies  # noqa: E402
from src.program_modification import enumerate_modifications  # noqa: E402
from src.seeds import derive_seeds  # noqa: E402
from src.simulator import Conference, SimConfig, UserProfile, simulate  # noqa: E402

# ---------- Конференции (используем personas_100 для всех; см. memo O Q-O2) ----------

CONFERENCES: Dict[str, Tuple[str, str, str]] = {
    "mobius_2025_autumn": (
        "data/conferences/mobius_2025_autumn.json",
        "data/conferences/mobius_2025_autumn_embeddings.npz",
        "data/personas/personas_100.json",
    ),
    "mobius_2025_autumn_en": (
        "data/conferences/mobius_2025_autumn_en.json",
        "data/conferences/mobius_2025_autumn_en_embeddings.npz",
        "data/personas/personas_mobius_en.json",
    ),
    "demo_day_2026_en": (
        "data/conferences/demo_day_2026_en.json",
        "data/conferences/demo_day_2026_en_embeddings.npz",
        "data/personas/personas_demoday_en.json",
    ),
    "toy_microconf_2slot": (
        "data/conferences/toy_microconf_2slot.json",
        "data/conferences/toy_microconf_2slot_embeddings.npz",
        "data/personas/personas_100.json",
    ),
}

# Маппинг popularity_source → cfg.w_fame (см. memo O §6).
POP_SRC_TO_W_FAME: Dict[str, float] = {
    "cosine_only": 0.0,
    "fame_only":   1.0,
    "mixed":       0.3,
}


# ---------- Загрузка ----------

def load_conference(name: str) -> Tuple[Conference, List[UserProfile]]:
    conf_path, emb_path, pers_path = CONFERENCES[name]
    conf = Conference.load(ROOT / conf_path, ROOT / emb_path)
    pers = json.loads((ROOT / pers_path).read_text())
    pers_emb_path = (ROOT / pers_path).with_name(
        Path(pers_path).stem + "_embeddings.npz"
    )
    npz = np.load(pers_emb_path, allow_pickle=False)
    emb_map = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    users = [
        UserProfile(
            id=p["id"],
            text=p.get("background", p.get("role", p["id"])),
            embedding=emb_map[p["id"]],
        )
        for p in pers
    ]
    return conf, users


# ---------- Capacity scaling (прецедент run_smoke.py) ----------

def scale_capacity(conf: Conference, mult: float) -> Conference:
    cloned = copy.deepcopy(conf)
    for h in cloned.halls.values():
        h.capacity = max(1, int(round(h.capacity * mult)))
    for s in cloned.slots:
        if s.hall_capacities:
            s.hall_capacities = {
                hid: max(1, int(round(c * mult)))
                for hid, c in s.hall_capacities.items()
            }
    return cloned


# ---------- Audience subsample (CRN: фикс по audience_seed) ----------

def select_audience(
    users: List[UserProfile], audience_size: int, audience_seed: int,
) -> List[UserProfile]:
    """Детерминированный subsample audience_size персон.

    Важно: rng, ходящий по `audience_seed`, гарантирует одинаковый набор
    персон между всеми политиками и seed-репликами для одной LHS-точки
    (CRN-инвариант, accepted Q-O9 + §8 memo O).
    """
    rng = np.random.default_rng(audience_seed)
    indices = rng.choice(len(users), size=audience_size, replace=False)
    return [users[int(i)] for i in indices]


# ---------- program_variant → Conference ----------

def build_program_variant(
    base_conf: Conference,
    program_variant: int,
    phi_seed: int,
    k_max: int = 5,
) -> Tuple[Conference, dict]:
    """program_variant=0 → P_0 как есть; иначе → enumerate_modifications[idx-1].

    `phi_seed` детерминирует выдачу Φ; одинаковый seed ⇒ одинаковый
    program_variant-эффект между политиками и репликами.

    Если требуемый индекс модификации недоступен (Φ вернула меньше k_max
    валидных swap'ов из-за speaker-конфликтов на конкретной программе) —
    fallback на P_0 с явной пометкой `fallback_to_p0` в metadata.
    """
    if program_variant == 0:
        return base_conf, {"program_variant": 0, "swap_descriptor": None}
    rng = np.random.default_rng(phi_seed)
    mods = enumerate_modifications(
        base_conf, k_max=k_max, rng=rng, same_day_only=True,
    )
    if program_variant - 1 >= len(mods):
        return base_conf, {
            "program_variant": program_variant,
            "swap_descriptor": None,
            "fallback_to_p0": True,
            "n_available_mods": len(mods),
        }
    modified, desc = mods[program_variant - 1]
    return modified, {
        "program_variant": program_variant,
        "swap_descriptor": {
            "slot_a": desc.slot_a, "slot_b": desc.slot_b,
            "t1": desc.t1, "t2": desc.t2,
        },
    }


# ---------- Метрики ----------

def compute_metrics_dict(conf: Conference, result) -> Dict[str, float]:
    return {
        "mean_overload_excess":      float(mean_hall_overload_excess(conf, result)),
        "mean_user_utility":         float(mean_user_utility(result)),
        "overflow_rate_slothall":    float(overflow_rate(conf, result, choice_only=False)),
        "hall_utilization_variance": float(hall_utilization_variance(conf, result)),
        "n_skipped":                 int(sum(1 for s in result.steps if s.chosen is None)),
        "n_users":                   int(len(result.steps)),
    }


# ---------- Главный прогон ----------

def run_lhs(
    conference: str,
    n_points: int = 50,
    replicates: int = 3,
    master_seed: int = 2026,
    maximin_k: int = 12,
    include_llm_ranker: bool = False,
    K: int = 3,
    min_per_level=None,
    force_pv_zero_in_maximin: bool = True,
    verbose: bool = True,
) -> dict:
    """Полный (или smoke) LHS-прогон.

    П1–П3 на всех `n_points` × `replicates`.
    П4 — только на maximin-точках × `replicates`, если
    `include_llm_ranker=True` (Q-O9 accepted вариант (в)).

    Параметры
    ---------
    conference : str
        Имя из `CONFERENCES`.
    n_points, replicates, master_seed, maximin_k
        Размеры и seeds. Для основной матрицы PROJECT_DESIGN §11:
        `n_points=50, replicates=3, maximin_k=12`.
    include_llm_ranker : bool
        Активирует П4 на maximin subset. Default False для smoke / offline-тестов.
    K : int
        Top-K рекомендаций.
    min_per_level : dict, optional
        Override порога repair дискретных осей; для smoke с малым n.
    force_pv_zero_in_maximin : bool
        Принудительное включение точки с program_variant=0 в maximin subset.
        Default True (Q-O4 accepted). False — для smoke с малым n_points.
    """
    timings: Dict[str, float] = {
        "load_conference_s": 0.0,
        "generate_lhs_s":    0.0,
        "maximin_subset_s":  0.0,
        "prep_total_s":      0.0,   # capacity scale + Φ + audience select
        "p1_p3_total_s":     0.0,
        "p4_total_s":        0.0,
    }

    t0_load = time.time()
    base_conf, all_users = load_conference(conference)
    timings["load_conference_s"] = time.time() - t0_load
    if verbose:
        print(
            f"loaded {conference}: {len(base_conf.talks)} talks, "
            f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots, "
            f"{len(all_users)} personas "
            f"({timings['load_conference_s']:.2f}s)",
            flush=True,
        )

    t0_lhs = time.time()
    rows = generate_lhs(
        n_points=n_points,
        master_seed=master_seed,
        min_per_level=min_per_level,
    )
    timings["generate_lhs_s"] = time.time() - t0_lhs

    t0_maximin = time.time()
    effective_k = min(maximin_k, n_points)
    maximin_idx = maximin_subset(
        rows, k=effective_k,
        force_program_variant_zero=force_pv_zero_in_maximin,
    )
    maximin_set = set(maximin_idx)
    timings["maximin_subset_s"] = time.time() - t0_maximin
    if verbose:
        print(
            f"generated {len(rows)} LHS rows ({timings['generate_lhs_s']:.2f}s); "
            f"maximin subset (k={effective_k}, {timings['maximin_subset_s']:.2f}s): "
            f"{sorted(maximin_idx)}",
            flush=True,
        )

    pols_no_llm = active_policies(include_llm=False)
    if include_llm_ranker:
        # active_policies(include_llm=True) тянет llm_ranker с реальным API;
        # отделяем его и используем только на maximin subset.
        pols_with_llm = active_policies(include_llm=True)
        llm_ranker_pol = pols_with_llm["llm_ranker"]
    else:
        llm_ranker_pol = None

    long_rows: List[dict] = []
    n_evals_by_policy: Dict[str, int] = {}
    n_p4_evals = 0  # отдельный счётчик для удобной диагностики

    # tqdm: общий progress по LHS-точкам. Не меняет порядок вычислений
    # (`for row in rows:` остаётся sequential), не вмешивается в RNG-потоки.
    pbar = tqdm(rows, desc="LHS rows", unit="row", disable=not verbose,
                dynamic_ncols=True)
    for row in pbar:
        # 1. Capacity scaling + Φ + audience subset (prep block)
        t0_prep = time.time()
        cfg_capacity_conf = scale_capacity(base_conf, row["capacity_multiplier"])
        seeds_const = derive_seeds(row["lhs_row_id"], replicate=1)
        program_conf, program_meta = build_program_variant(
            cfg_capacity_conf,
            row["program_variant"],
            phi_seed=seeds_const["phi_seed"],
            k_max=5,
        )
        audience_users = select_audience(
            all_users, row["audience_size"], seeds_const["audience_seed"],
        )
        w_fame = POP_SRC_TO_W_FAME[row["popularity_source"]]
        timings["prep_total_s"] += time.time() - t0_prep

        # 2. Какие политики — П1-П3 всегда; П4 только на maximin subset
        policies_to_run: Dict[str, object] = dict(pols_no_llm)
        is_maximin = row["lhs_row_id"] in maximin_set
        if include_llm_ranker and is_maximin:
            policies_to_run["llm_ranker"] = llm_ranker_pol

        # 3. Реплики × политики
        for replicate in range(1, replicates + 1):
            seeds = derive_seeds(row["lhs_row_id"], replicate=replicate)
            cfg = SimConfig(
                tau=0.7, p_skip_base=0.10, K=K,
                seed=seeds["cfg_seed"],
                w_rel=row["w_rel"], w_rec=row["w_rec"], w_gossip=row["w_gossip"],
                w_fame=w_fame,
            )
            for pol_name, pol in policies_to_run.items():
                t0_eval = time.time()
                res = simulate(program_conf, audience_users, pol, cfg)
                eval_dt = time.time() - t0_eval
                if pol_name == "llm_ranker":
                    timings["p4_total_s"] += eval_dt
                    n_p4_evals += 1
                else:
                    timings["p1_p3_total_s"] += eval_dt

                metrics = compute_metrics_dict(program_conf, res)
                n_evals_by_policy[pol_name] = n_evals_by_policy.get(pol_name, 0) + 1
                long_rows.append({
                    "lhs_row_id":           row["lhs_row_id"],
                    "capacity_multiplier":  row["capacity_multiplier"],
                    "popularity_source":    row["popularity_source"],
                    "w_rel":                row["w_rel"],
                    "w_rec":                row["w_rec"],
                    "w_gossip":             row["w_gossip"],
                    "audience_size":        row["audience_size"],
                    "program_variant":      row["program_variant"],
                    "policy":               pol_name,
                    "replicate":            replicate,
                    "audience_seed":        seeds["audience_seed"],
                    "phi_seed":             seeds["phi_seed"],
                    "cfg_seed":             seeds["cfg_seed"],
                    "is_maximin_point":     is_maximin,
                    "swap_descriptor":      program_meta.get("swap_descriptor"),
                    "fallback_to_p0":       program_meta.get("fallback_to_p0", False),
                    **{f"metric_{k}": v for k, v in metrics.items()},
                })

        # progress постфикс: текущая точка, evals по политикам, П4-метка
        pbar.set_postfix({
            "n_evals": sum(n_evals_by_policy.values()),
            "p4": n_p4_evals,
            "p4_now": "✓" if is_maximin and include_llm_ranker else "·",
        })

    # Стоимость П4 (если LLMRankerPolicy была активна)
    p4_cost_usd = None
    if llm_ranker_pol is not None:
        p4_cost_usd = float(getattr(llm_ranker_pol, "cumulative_cost", 0.0))

    return {
        "etap": "Q",
        "conference": conference,
        "params": {
            "n_points": n_points,
            "replicates": replicates,
            "master_seed": master_seed,
            "maximin_k": effective_k,
            "include_llm_ranker": include_llm_ranker,
            "K": K,
            "force_pv_zero_in_maximin": force_pv_zero_in_maximin,
            "min_per_level": min_per_level or DEFAULT_MIN_PER_LEVEL,
        },
        "lhs_rows": rows,
        "maximin_indices": sorted(maximin_idx),
        "results": long_rows,
        "n_results": len(long_rows),
        "n_evals_by_policy": n_evals_by_policy,
        "n_p4_evals": n_p4_evals,
        "p4_cost_usd": p4_cost_usd,
        "timings": timings,
    }


# ---------- CSV / Markdown / Acceptance ----------

CSV_COLUMNS = (
    "lhs_row_id", "capacity_multiplier", "popularity_source",
    "w_rel", "w_rec", "w_gossip",
    "audience_size", "program_variant",
    "policy", "replicate",
    "audience_seed", "phi_seed", "cfg_seed",
    "is_maximin_point", "fallback_to_p0",
    "swap_slot_a", "swap_slot_b", "swap_t1", "swap_t2",
    "metric_mean_overload_excess", "metric_mean_user_utility",
    "metric_overflow_rate_slothall", "metric_hall_utilization_variance",
    "metric_n_skipped", "metric_n_users",
)


def write_csv(out_path: Path, results: List[dict]) -> None:
    """Long-format CSV (одна строка на eval)."""
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            sd = row.pop("swap_descriptor", None)
            if sd:
                row["swap_slot_a"] = sd.get("slot_a")
                row["swap_slot_b"] = sd.get("slot_b")
                row["swap_t1"] = sd.get("t1")
                row["swap_t2"] = sd.get("t2")
            else:
                row["swap_slot_a"] = ""
                row["swap_slot_b"] = ""
                row["swap_t1"] = ""
                row["swap_t2"] = ""
            writer.writerow(row)


def compute_acceptance_checks(out: dict) -> dict:
    """Acceptance-чеки этапа Q (PIVOT_IMPLEMENTATION_PLAN + Q-O9)."""
    rows = out["results"]
    params = out["params"]
    n_points = params["n_points"]
    replicates = params["replicates"]
    maximin_k = params["maximin_k"]
    include_p4 = params["include_llm_ranker"]
    maximin_set = set(out["maximin_indices"])

    expected_p1_p3 = n_points * replicates * 3      # П1, П2, П3
    expected_p4 = (maximin_k * replicates) if include_p4 else 0
    expected_total = expected_p1_p3 + expected_p4

    n_by_pol = out["n_evals_by_policy"]
    actual_p1_p3 = sum(n_by_pol.get(p, 0) for p in ("no_policy", "cosine", "capacity_aware"))
    actual_p4 = n_by_pol.get("llm_ranker", 0)
    actual_total = sum(n_by_pol.values())

    # П4 только на maximin
    p4_outside_maximin = [
        r for r in rows
        if r["policy"] == "llm_ranker" and r["lhs_row_id"] not in maximin_set
    ]

    # CRN: audience_seed/phi_seed одинаковы для (lhs_row_id, replicate) между политиками
    by_key: Dict[tuple, list] = {}
    for r in rows:
        key = (r["lhs_row_id"], r["replicate"])
        by_key.setdefault(key, []).append(r)
    crn_violations = []
    for key, group in by_key.items():
        seeds_seen = {(r["audience_seed"], r["phi_seed"]) for r in group}
        if len(seeds_seen) > 1:
            crn_violations.append({"key": list(key), "seeds_count": len(seeds_seen)})

    # cfg_seed = replicate
    cfg_seed_violations = [
        r["lhs_row_id"] for r in rows if r["cfg_seed"] != r["replicate"]
    ]

    # Long-format columns: проверка минимального набора
    required = {
        "lhs_row_id", "capacity_multiplier", "popularity_source",
        "w_rel", "w_rec", "w_gossip", "audience_size", "program_variant",
        "policy", "replicate", "audience_seed", "phi_seed", "cfg_seed",
        "is_maximin_point",
        "metric_mean_overload_excess",
    }
    missing_keys = []
    if rows:
        missing_keys = sorted(required - set(rows[0].keys()))

    # Fallback-к-P0 случаи (не silent — фиксируется явным флагом)
    n_fallback = sum(1 for r in rows if r.get("fallback_to_p0"))

    return {
        "expected_p1_p3":  expected_p1_p3,
        "expected_p4":     expected_p4,
        "expected_total":  expected_total,
        "actual_p1_p3":    actual_p1_p3,
        "actual_p4":       actual_p4,
        "actual_total":    actual_total,
        "n_eval_match":    actual_total == expected_total,
        "p1_p3_match":     actual_p1_p3 == expected_p1_p3,
        "p4_match":        actual_p4 == expected_p4,
        "p4_only_on_maximin":          len(p4_outside_maximin) == 0,
        "p4_outside_maximin_count":    len(p4_outside_maximin),
        "crn_audience_phi_invariant":  len(crn_violations) == 0,
        "crn_violations_count":        len(crn_violations),
        "cfg_seed_equals_replicate":   len(cfg_seed_violations) == 0,
        "cfg_seed_violations_count":   len(cfg_seed_violations),
        "long_format_keys_ok":         len(missing_keys) == 0,
        "missing_keys":                missing_keys,
        "n_fallback_to_p0":            n_fallback,
    }


def acceptance_passed(checks: dict) -> bool:
    return all([
        checks["n_eval_match"],
        checks["p1_p3_match"],
        checks["p4_match"],
        checks["p4_only_on_maximin"],
        checks["crn_audience_phi_invariant"],
        checks["cfg_seed_equals_replicate"],
        checks["long_format_keys_ok"],
    ])


def render_markdown(out: dict, checks: dict, paths: dict, wallclock_s: float) -> str:
    """Markdown-отчёт этапа Q."""
    p = out["params"]
    t = out["timings"]
    lines = []
    lines.append(f"# Этап Q: полный параметрический LHS-прогон")
    lines.append("")
    lines.append(f"Дата: {dt.date.today().isoformat()}")
    lines.append(f"Конференция: `{out['conference']}`")
    lines.append(f"Master seed: {p['master_seed']}")
    lines.append("")

    lines.append("## Параметры")
    lines.append("")
    lines.append(f"- n_points: **{p['n_points']}**")
    lines.append(f"- replicates: **{p['replicates']}**")
    lines.append(f"- maximin_k: **{p['maximin_k']}**")
    lines.append(f"- include_llm_ranker: **{p['include_llm_ranker']}**")
    lines.append(f"- K (top-K): {p['K']}")
    lines.append(f"- audience_size grid: {{30, 60, 100}}")
    lines.append(f"- popularity_source grid: {{cosine_only, fame_only, mixed}}")
    lines.append(f"- w_rec ∈ [0, 0.7], w_gossip ∈ [0, 0.7], симплекс w_rel + w_rec + w_gossip = 1")
    lines.append(f"- program_variant ∈ {{0..5}}; P_0 control + до 5 swap-модификаций (Φ)")
    lines.append("")

    lines.append("## Wallclock breakdown")
    lines.append("")
    lines.append(f"| Блок | Время, сек |")
    lines.append(f"|---|---:|")
    lines.append(f"| load_conference | {t['load_conference_s']:.2f} |")
    lines.append(f"| generate_lhs | {t['generate_lhs_s']:.2f} |")
    lines.append(f"| maximin_subset | {t['maximin_subset_s']:.2f} |")
    lines.append(f"| prep (capacity / Φ / audience) | {t['prep_total_s']:.2f} |")
    lines.append(f"| П1–П3 evals | {t['p1_p3_total_s']:.2f} |")
    lines.append(f"| П4 (llm_ranker) evals | {t['p4_total_s']:.2f} |")
    lines.append(f"| **итого (run_lhs внутренний)** | **{sum(t.values()):.2f}** |")
    lines.append(f"| **wallclock полный** | **{wallclock_s:.2f}** |")
    lines.append("")

    lines.append("## Сводка evals по политикам")
    lines.append("")
    lines.append(f"| Политика | Evals |")
    lines.append(f"|---|---:|")
    for pol in ("no_policy", "cosine", "capacity_aware", "llm_ranker"):
        n = out["n_evals_by_policy"].get(pol, 0)
        lines.append(f"| {pol} | {n} |")
    lines.append(f"| **итого** | **{out['n_results']}** |")
    lines.append("")

    if out.get("p4_cost_usd") is not None:
        lines.append(f"П4 LLMRankerPolicy cumulative cost: **${out['p4_cost_usd']:.4f}**")
        lines.append("")

    lines.append("## Maximin subset")
    lines.append("")
    lines.append(f"Indices ({len(out['maximin_indices'])}): "
                 f"{out['maximin_indices']}")
    lines.append("")

    lines.append("## Acceptance")
    lines.append("")
    lines.append(f"| Чек | Значение | Статус |")
    lines.append(f"|---|---|---|")
    lines.append(f"| П1–П3 evals == ожидаемое | {checks['actual_p1_p3']} == "
                 f"{checks['expected_p1_p3']} | "
                 f"{'PASS' if checks['p1_p3_match'] else 'FAIL'} |")
    lines.append(f"| П4 evals == ожидаемое | {checks['actual_p4']} == "
                 f"{checks['expected_p4']} | "
                 f"{'PASS' if checks['p4_match'] else 'FAIL'} |")
    lines.append(f"| total evals == ожидаемое | {checks['actual_total']} == "
                 f"{checks['expected_total']} | "
                 f"{'PASS' if checks['n_eval_match'] else 'FAIL'} |")
    lines.append(f"| П4 только на maximin | "
                 f"violations={checks['p4_outside_maximin_count']} | "
                 f"{'PASS' if checks['p4_only_on_maximin'] else 'FAIL'} |")
    lines.append(f"| CRN audience/phi инвариант | "
                 f"violations={checks['crn_violations_count']} | "
                 f"{'PASS' if checks['crn_audience_phi_invariant'] else 'FAIL'} |")
    lines.append(f"| cfg_seed = replicate | "
                 f"violations={checks['cfg_seed_violations_count']} | "
                 f"{'PASS' if checks['cfg_seed_equals_replicate'] else 'FAIL'} |")
    lines.append(f"| long-format ключи | "
                 f"missing={checks['missing_keys']} | "
                 f"{'PASS' if checks['long_format_keys_ok'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"Дополнительно (диагностика, не блокатор):")
    lines.append(f"- fallback_to_p0 случаев: **{checks['n_fallback_to_p0']}** "
                 f"(когда `enumerate_modifications` вернула меньше swap-вариантов "
                 f"чем требовался индекс program_variant; не silent — флаг записан "
                 f"в каждой соответствующей строке)")
    lines.append("")

    overall = acceptance_passed(checks)
    lines.append(f"### Итог: **{'PASS' if overall else 'FAIL'}**")
    lines.append("")

    lines.append("## Артефакты")
    lines.append("")
    lines.append(f"- JSON: `{paths['json'].name}`")
    lines.append(f"- CSV (long-format): `{paths['csv'].name}`")
    lines.append(f"- этот отчёт: `{paths['md'].name}`")
    lines.append("")

    return "\n".join(lines)


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn",
                    choices=list(CONFERENCES.keys()))
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--maximin-k", type=int, default=12)
    ap.add_argument("--include-llm-ranker", action="store_true",
                    help="Активирует П4 LLMRankerPolicy на maximin subset "
                         "(Q-O9 accepted вариант (в)). Без флага П4 не "
                         "запускается.")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out-prefix", default=None,
                    help="Путь без расширения; будут созданы .json/.csv/.md")
    args = ap.parse_args()

    t0 = time.time()
    out = run_lhs(
        conference=args.conference,
        n_points=args.n_points,
        replicates=args.replicates,
        master_seed=args.master_seed,
        maximin_k=args.maximin_k,
        include_llm_ranker=args.include_llm_ranker,
        K=args.K,
    )
    elapsed = time.time() - t0
    out["elapsed_total_s"] = elapsed

    checks = compute_acceptance_checks(out)
    out["acceptance"] = checks

    date = dt.date.today().isoformat()
    if args.out_prefix:
        prefix = Path(args.out_prefix)
    else:
        prefix = ROOT / "results" / f"lhs_parametric_{args.conference}_{date}"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "json": prefix.with_suffix(".json"),
        "csv":  prefix.with_suffix(".csv"),
        "md":   prefix.with_suffix(".md"),
    }
    paths["json"].write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str)
    )
    write_csv(paths["csv"], out["results"])
    paths["md"].write_text(render_markdown(out, checks, paths, elapsed))

    print(f"\nWROTE: {paths['json']}")
    print(f"WROTE: {paths['csv']}")
    print(f"WROTE: {paths['md']}")
    print(f"\n  n_results: {out['n_results']}")
    print(f"  evals by policy: {out['n_evals_by_policy']}")
    print(f"  П4 cost: ${out.get('p4_cost_usd', 0.0):.4f}"
          if out.get("p4_cost_usd") is not None else "  П4 cost: n/a (не запускалась)")
    print(f"  elapsed total: {elapsed:.1f}s")
    print(f"\n  acceptance: {'PASS' if acceptance_passed(checks) else 'FAIL'}")
    for k, v in checks.items():
        if isinstance(v, bool):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
