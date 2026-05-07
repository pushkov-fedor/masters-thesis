"""Этап E PIVOT_IMPLEMENTATION_PLAN: toy-checks этапа D через ядро simulator.py.

Воспроизводит acceptance-эксперименты этапа D
(``experiments/scripts/run_toy_microconf.py``), но теперь через
обновлённое ядро ``experiments.src.simulator.simulate`` и активный реестр
политик (``experiments.src.policies.registry.active_policies``).

Что проверяется:
    1. Юнит-инвариант EC3 на уровне функции utility: при ``w_rec = 0``
       utility и финальное распределение посещений строго не зависят от
       политики. Реализовано как inline-блок ``check_utility_invariance``.
    2. Acceptance-checks этапа D через ядро:
       - EC3 strict (no_policy / cosine / capacity_aware) при w_rec=0;
       - MC3 на asymmetric capacity 20/80;
       - EC1 на loose capacity (×3.0);
       - EC2 на tight capacity (×0.5);
       - TC-D3: capacity_aware vs cosine на asymmetric.

LLMRankerPolicy в этом скрипте сознательно НЕ инстанцируется: spike по
LLM-симулятору — этап G, реализация — этап H. Состав через ядро = три
детерминированные политики основного реестра (П1, П2, П3).

Скрипт читает ту же toy-конференцию и тех же 100 персон, что и этап D,
но не использует ``run_toy_microconf.simulate_local`` — все вычисления
идут через обновлённое ядро.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.no_policy import NoPolicy  # noqa: E402
from src.simulator import (  # noqa: E402
    Conference,
    SimConfig,
    UserProfile,
    simulate,
)

DATA_DIR = EXPERIMENTS_ROOT / "data"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"

CAPACITY_SCENARIOS: Dict[str, Dict[int, int]] = {
    "base_50_50":         {1: 50,  2: 50},
    "loose_150_150":      {1: 150, 2: 150},   # EC1
    "tight_25_25":        {1: 25,  2: 25},    # EC2
    "asymmetric_20_80":   {1: 20,  2: 80},    # MC3 / TC-D3
}

W_REC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
K = 1
TAU = 0.7
P_SKIP = 0.10


# ============================================================
# Загрузка toy-данных в формат ядра
# ============================================================

def load_toy_conference() -> Conference:
    return Conference.load(
        DATA_DIR / "conferences" / "toy_microconf.json",
        DATA_DIR / "conferences" / "toy_microconf_embeddings.npz",
    )


def load_toy_users() -> List[UserProfile]:
    pj = json.loads(
        (DATA_DIR / "personas" / "toy_personas_100.json").read_text()
    )
    npz = np.load(
        DATA_DIR / "personas" / "toy_personas_100_embeddings.npz",
        allow_pickle=False,
    )
    emb_map = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    return [
        UserProfile(
            id=p["id"],
            text=p.get("background", p.get("role", p["id"])),
            embedding=emb_map[p["id"]],
        )
        for p in pj
    ]


def with_capacity(conf: Conference, overrides: Dict[int, int]) -> Conference:
    """Создать копию Conference с заданной вместимостью залов."""
    cloned = copy.deepcopy(conf)
    for hid, cap in overrides.items():
        if hid in cloned.halls:
            cloned.halls[hid].capacity = int(cap)
    return cloned


def policies_factory():
    """Только детерминированные П1-П3. LLMRankerPolicy — этапы G/H, не toy."""
    return {
        "no_policy":      NoPolicy(),
        "cosine":         CosinePolicy(),
        "capacity_aware": CapacityAwarePolicy(),
    }


# ============================================================
# Метрики
# ============================================================

def compute_metrics(result, conf: Conference) -> dict:
    """Считаем mean_overload_excess и mean_user_utility для одного прогона."""
    # На toy slot один; у simulate_async hall_load_per_slot[slot.id] = {hid: occ}
    overloads: List[float] = []
    for slot_id, loads in result.hall_load_per_slot.items():
        for hid, occ in loads.items():
            cap = conf.capacity_at(slot_id, hid)
            if occ > cap:
                overloads.append((occ - cap) / max(1.0, cap))
    chosen_rels = [s.chosen_relevance for s in result.steps if s.chosen is not None]
    return {
        "hall_loads": {
            slot_id: {str(h): int(o) for h, o in loads.items()}
            for slot_id, loads in result.hall_load_per_slot.items()
        },
        "n_skipped": sum(1 for s in result.steps if s.chosen is None),
        "n_users": len(result.steps),
        "mean_overload_excess": float(np.mean(overloads)) if overloads else 0.0,
        "mean_user_utility": float(np.mean(chosen_rels)) if chosen_rels else 0.0,
    }


def aggregate_seeds(per_seed: List[dict]) -> dict:
    keys = ["mean_overload_excess", "mean_user_utility", "n_skipped"]
    out: Dict[str, object] = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ============================================================
# Юнит-инвариант EC3
# ============================================================

def check_utility_invariance(conf: Conference, users: List[UserProfile]) -> dict:
    """Юнит-инвариант EC3 на уровне ядра.

    При ``w_rec = 0`` финальные результаты должны быть строго инвариантны
    к политике (любой). Это операционализация PROJECT_DESIGN §11 EC3 на
    уровне функции simulate_async — не статистика, а точное равенство:
    утилитный вклад политики обнуляется (w_rec=0), и CRN-разделённый
    choice_rng идёт по той же траектории.
    """
    pols = policies_factory()
    pol_results: Dict[str, dict] = {}
    cfg_w0 = SimConfig(
        tau=TAU, p_skip_base=P_SKIP, K=K, seed=0,
        w_rel=1.0, w_rec=0.0,
    )
    for name, pol in pols.items():
        res = simulate(conf, users, pol, cfg_w0)
        pol_results[name] = compute_metrics(res, conf)

    util_vals = [v["mean_user_utility"] for v in pol_results.values()]
    overl_vals = [v["mean_overload_excess"] for v in pol_results.values()]
    range_util = float(max(util_vals) - min(util_vals))
    range_overl = float(max(overl_vals) - min(overl_vals))

    # Сравнение распределений по залам — тоже должно быть тождественным
    distinct_loads = {
        json.dumps(v["hall_loads"], sort_keys=True) for v in pol_results.values()
    }

    print("\n=== Юнит-инвариант EC3: w_rec = 0 ===")
    for name, m in pol_results.items():
        print(f"    {name:14s}: utility={m['mean_user_utility']:.10f}, "
              f"overload={m['mean_overload_excess']:.10f}, "
              f"hall_loads={m['hall_loads']}")
    print(f"    range(utility)  = {range_util:.2e}")
    print(f"    range(overload) = {range_overl:.2e}")
    print(f"    distinct hall_load distributions: {len(distinct_loads)}  (ожидание: 1)")

    passed = (range_util < 1e-12) and (range_overl < 1e-12) and (len(distinct_loads) == 1)
    print(f"    PASS = {passed}")
    return {
        "range_utility": range_util,
        "range_overload": range_overl,
        "n_distinct_hall_load_distributions": len(distinct_loads),
        "per_policy": pol_results,
        "pass": bool(passed),
    }


# ============================================================
# Acceptance-checks этапа D через ядро
# ============================================================

def run_grid(conf: Conference, users: List[UserProfile]) -> List[dict]:
    rows = []
    for cap_name, cap_overrides in CAPACITY_SCENARIOS.items():
        local_conf = with_capacity(conf, cap_overrides)
        for w_rec in W_REC_GRID:
            cfg = SimConfig(
                tau=TAU, p_skip_base=P_SKIP, K=K, seed=0,
                w_rel=1.0 - w_rec, w_rec=w_rec,
            )
            for pol_name, pol_obj in policies_factory().items():
                per_seed = []
                for s in SEEDS:
                    cfg.seed = s
                    res = simulate(local_conf, users, pol_obj, cfg)
                    per_seed.append(compute_metrics(res, local_conf))
                rows.append({
                    "capacity_scenario": cap_name,
                    "policy":            pol_name,
                    "w_rec":             w_rec,
                    "seeds":             SEEDS,
                    "agg":               aggregate_seeds(per_seed),
                    "per_seed":          per_seed,
                })
    return rows


def _cv(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values) / abs(mean))


def check_acceptance(rows: List[dict]) -> dict:
    print("\n=== Acceptance-checks этапа D через ядро ===")
    checks: Dict[str, object] = {}

    # EC3 strict (П1-П3) at base, w_rec=0
    base_w0 = [r for r in rows
               if r["capacity_scenario"] == "base_50_50" and r["w_rec"] == 0.0]
    util = [r["agg"]["mean_user_utility_mean"] for r in base_w0]
    overl = [r["agg"]["mean_overload_excess_mean"] for r in base_w0]
    range_util = float(max(util) - min(util)) if util else 0.0
    range_overl = float(max(overl) - min(overl)) if overl else 0.0
    print("\n[EC3 strict] @ base, w_rec=0:")
    for r in base_w0:
        a = r["agg"]
        print(f"    {r['policy']:14s}: utility={a['mean_user_utility_mean']:.10f}, "
              f"overload={a['mean_overload_excess_mean']:.10f}")
    print(f"    range(utility)  = {range_util:.2e}")
    print(f"    range(overload) = {range_overl:.2e}")
    checks["ec3_range_utility"] = range_util
    checks["ec3_range_overload"] = range_overl
    checks["ec3_pass"] = bool(range_util < 1e-12 and range_overl < 1e-12)

    # MC3 @ asymmetric: монотонный рост
    print("\n[MC3] @ asymmetric_20_80: range(overload across policies) по w_rec:")
    ranges = []
    for w in W_REC_GRID:
        bs = [r for r in rows
              if r["capacity_scenario"] == "asymmetric_20_80" and r["w_rec"] == w]
        ovs = [r["agg"]["mean_overload_excess_mean"] for r in bs]
        rng_w = float(max(ovs) - min(ovs))
        ranges.append({"w_rec": w, "range": rng_w})
        print(f"    w_rec={w}: range={rng_w:.4f}")
    is_monotone = all(
        ranges[i]["range"] <= ranges[i + 1]["range"] + 1e-9
        for i in range(len(ranges) - 1)
    )
    checks["mc3_ranges"] = ranges
    checks["mc3_monotone_pass"] = bool(is_monotone)
    print(f"    Монотонно неубывает на asymmetric? {is_monotone}")

    # EC1 @ loose
    loose = [r for r in rows if r["capacity_scenario"] == "loose_150_150"]
    max_ov_loose = max(r["agg"]["mean_overload_excess_mean"] for r in loose)
    print(f"\n[EC1] @ loose (cap×3.0): max overload = {max_ov_loose:.4f}  (== 0?)")
    checks["ec1_max_overload"] = max_ov_loose
    checks["ec1_pass"] = bool(max_ov_loose == 0.0)

    # EC2 @ tight
    tight = [r for r in rows if r["capacity_scenario"] == "tight_25_25"]
    max_ov_tight = max(r["agg"]["mean_overload_excess_mean"] for r in tight)
    print(f"\n[EC2] @ tight (cap×0.5): max overload = {max_ov_tight:.4f}  (> 0?)")
    checks["ec2_max_overload"] = max_ov_tight
    checks["ec2_pass"] = bool(max_ov_tight > 0.0)

    # TC-D3 @ asymmetric: cosine vs capacity_aware
    print("\n[TC-D3] @ asymmetric_20_80, по w_rec:")
    asym_rows = []
    for w in W_REC_GRID:
        rows_w = [r for r in rows
                  if r["capacity_scenario"] == "asymmetric_20_80" and r["w_rec"] == w]
        by_pol = {r["policy"]: r["agg"] for r in rows_w}
        if "cosine" in by_pol and "capacity_aware" in by_pol:
            ov_cos = by_pol["cosine"]["mean_overload_excess_mean"]
            ov_cap = by_pol["capacity_aware"]["mean_overload_excess_mean"]
            u_cos = by_pol["cosine"]["mean_user_utility_mean"]
            u_cap = by_pol["capacity_aware"]["mean_user_utility_mean"]
            ratio = (u_cap / u_cos) if u_cos else float("nan")
            print(
                f"    w_rec={w}: cos ov={ov_cos:.4f} u={u_cos:.4f} | "
                f"cap_aware ov={ov_cap:.4f} u={u_cap:.4f} | "
                f"Δov={ov_cos - ov_cap:+.4f}, ratio_u(cap/cos)={ratio:.3f}"
            )
            asym_rows.append({
                "w_rec": w, "ov_cos": ov_cos, "ov_cap": ov_cap,
                "u_cos": u_cos, "u_cap": u_cap,
                "delta_ov": ov_cos - ov_cap, "util_ratio": ratio,
            })
    central = [s for s in asym_rows if s["w_rec"] >= 0.5]
    p3_no_worse = all(s["ov_cap"] <= s["ov_cos"] + 1e-9 for s in central)
    util_ok = all(s["util_ratio"] > 0.6 for s in central if s["u_cos"] > 0)
    print(f"    П3 не хуже П2 по overload @ w_rec≥0.5: {p3_no_worse}")
    print(f"    util_ratio П3/П2 > 0.6 @ w_rec≥0.5: {util_ok}")
    checks["asym_summary"] = asym_rows
    checks["asym_p3_overload_pass"] = bool(p3_no_worse)
    checks["asym_util_ratio_pass"] = bool(util_ok)

    overall = (
        checks["ec3_pass"] and checks["mc3_monotone_pass"]
        and checks["ec1_pass"] and checks["ec2_pass"]
        and checks["asym_p3_overload_pass"] and checks["asym_util_ratio_pass"]
    )
    checks["overall_pass"] = bool(overall)
    print(f"\n=== Итого acceptance: {'OK' if overall else 'NOT OK'} ===")
    return checks


# ============================================================
# Сравнение с локальной симуляцией этапа D
# ============================================================

def compare_with_local(rows_core: List[dict]) -> dict:
    """Сопоставить ядро vs локальная симуляция этапа D, для пересечения политик.

    Локальный toy этапа D использует ту же формулу utility, но:
      - его policy_rng = default_rng(seed*31+1), а ядро policy_rng = default_rng(seed*1_000_003+slot_idx+31).
      - его choice_rng = default_rng(seed), ядро choice_rng = default_rng(seed*1_000_003+slot_idx).
    Поэтому на одном seed численные значения будут отличаться. Что мы сверяем:
      - качественные ожидания этапа D (EC1/EC2/MC3 monotonicity/asym ordering)
        совпадают между двумя реализациями;
      - распределения mean_user_utility и mean_overload_excess лежат в близком
        диапазоне (различия от RNG, не от формулы).
    """
    local_path = RESULTS_DIR / "toy_microconf.json"
    if not local_path.exists():
        print("\n[compare] локальный toy_microconf.json не найден — пропускаю.")
        return {"skipped": True}
    local = json.loads(local_path.read_text())

    # Сборка таблиц по (capacity_scenario, w_rec, policy)
    real_pols = {"no_policy", "cosine", "capacity_aware"}
    def index(rows):
        return {
            (r["capacity_scenario"], r["w_rec"], r["policy"]):
                (r["agg"]["mean_overload_excess_mean"],
                 r["agg"]["mean_user_utility_mean"])
            for r in rows
            if r["policy"] in real_pols
        }

    core_idx = index(rows_core)
    local_idx = index(local["results"])
    common_keys = sorted(set(core_idx.keys()) & set(local_idx.keys()))
    diffs_overload = []
    diffs_utility = []
    for k in common_keys:
        co = core_idx[k]
        lo = local_idx[k]
        diffs_overload.append(abs(co[0] - lo[0]))
        diffs_utility.append(abs(co[1] - lo[1]))
    summary = {
        "n_common_cells": len(common_keys),
        "max_abs_diff_overload": float(max(diffs_overload)) if diffs_overload else 0.0,
        "mean_abs_diff_overload": float(np.mean(diffs_overload)) if diffs_overload else 0.0,
        "max_abs_diff_utility": float(max(diffs_utility)) if diffs_utility else 0.0,
        "mean_abs_diff_utility": float(np.mean(diffs_utility)) if diffs_utility else 0.0,
    }
    print("\n=== Сопоставление ядро vs локальная симуляция этапа D (только П1-П3) ===")
    print(f"    Общих ячеек: {summary['n_common_cells']}")
    print(f"    max |Δoverload| = {summary['max_abs_diff_overload']:.4f}, "
          f"mean = {summary['mean_abs_diff_overload']:.4f}")
    print(f"    max |Δutility|  = {summary['max_abs_diff_utility']:.4f}, "
          f"mean = {summary['mean_abs_diff_utility']:.4f}")
    print("    (несовпадение значений ожидаемо: разные RNG-seedы; формула — одна.)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "toy_microconf_via_core.json"))
    args = parser.parse_args()

    conf = load_toy_conference()
    users = load_toy_users()
    print(f"loaded toy: {len(conf.talks)} talks, {len(conf.halls)} halls, "
          f"{len(conf.slots)} slots, {len(users)} users")

    base_for_invariance = with_capacity(conf, CAPACITY_SCENARIOS["base_50_50"])
    invariance = check_utility_invariance(base_for_invariance, users)

    rows = run_grid(conf, users)
    acceptance = check_acceptance(rows)
    compare = compare_with_local(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "etap": "E",
        "model": ("U = w_rel*effective_rel + w_rec*1{t in recs}; "
                  "consider_ids = full slot; capacity-effect only in policy "
                  "capacity_aware; p_skip = 0.10 outside option."),
        "params": {
            "K": K, "tau": TAU, "p_skip": P_SKIP,
            "n_personas": len(users), "seeds": SEEDS, "w_rec_grid": W_REC_GRID,
            "w_rel_normalisation": "w_rel = 1 - w_rec (toy default)",
        },
        "policies": list(policies_factory().keys()),
        "policies_note": ("LLMRankerPolicy в этом скрипте сознательно не "
                          "инстанцируется: spike — этап G, реализация — этап H. "
                          "Mock_random этапа D в этой проверке тоже не нужен — "
                          "EC3 strict проверяется на детерминированных П1-П3, "
                          "ядро гарантирует CRN на уровне split RNG."),
        "capacity_scenarios": {k: {str(h): c for h, c in v.items()}
                               for k, v in CAPACITY_SCENARIOS.items()},
        "results": rows,
        "checks": {
            "utility_invariance": invariance,
            "acceptance":         acceptance,
            "compare_with_local": compare,
        },
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
