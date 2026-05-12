"""Изолированная проверка эффекта оператора локальных модификаций Φ.

Цель: на параметрическом симуляторе с одной политикой (no_policy) проверить,
влияют ли перестановки в программе на распределение перегрузки залов при
зафиксированных остальных осях LHS. Это closes-the-loop проверка для
заявления PROJECT_OVERVIEW §12.7 о diagnostic-only характере оси
program_variant в основном LHS-плане.

Сетка прогона:
  - program_variant: 0 (P_0) + все валидные swap-пары до cap=200
  - capacity_multiplier: {0.7, 1.0}
  - w_gossip: {0.0, 0.3, 0.6}  (w_rel = 1 − w_gossip; w_rec = 0)
  - audience_size: 100 (фикс), audience_seed=0 (фикс)
  - replicate: {1, 2, 3} (cfg_seed = replicate)
  - policy: no_policy (П1)
  - conference: mobius_2025_autumn_en (BGE+ABTT)

Запуск:
    .venv/bin/python scripts/run_phi_isolated.py

Выход:
  - results/phi_isolated_2026-05-12.csv  — long-format таблица всех evals
  - results/phi_isolated_2026-05-12.md   — сводка с диапазоном overload по PV
                                            в каждой ячейке (capacity × gossip)
  - results/phi_isolated_2026-05-12.json — машиночитаемые агрегаты
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import (  # noqa: E402
    load_conference,
    scale_capacity,
    select_audience,
    compute_metrics_dict,
)
from src.policies.registry import active_policies  # noqa: E402
from src.program_modification import (  # noqa: E402
    SwapDescriptor,
    _enumerate_all_pairs,
    has_speaker_conflict,
)
from src.simulator import Conference, SimConfig, simulate  # noqa: E402

import copy


def has_hall_conflict(conf: Conference) -> bool:
    """True если в каком-то слоте два талка в одном зале."""
    for slot in conf.slots:
        seen = set()
        for tid in slot.talk_ids:
            h = conf.talks[tid].hall
            if h in seen:
                return True
            seen.add(h)
    return False


def apply_full_swap(conf: Conference, desc: SwapDescriptor) -> Conference:
    """FULL swap двух докладов: обмениваются и slot_id, и hall.

    Это исправление дизайна base `_apply_swap`, который сохраняет hall
    и из-за этого может создавать физически невозможную конфигурацию
    (два доклада в одном зале одного слота). При full swap каждый зал
    в каждом слоте остаётся однократно занят, hall-конфликт невозможен
    по построению.
    """
    cloned = copy.deepcopy(conf)
    t1 = cloned.talks[desc.t1]
    t2 = cloned.talks[desc.t2]
    t1.slot_id, t2.slot_id = desc.slot_b, desc.slot_a
    t1.hall, t2.hall = t2.hall, t1.hall
    talk_ids_by_slot: dict = {s.id: [] for s in cloned.slots}
    for tid, t in cloned.talks.items():
        talk_ids_by_slot.setdefault(t.slot_id, []).append(tid)
    for s in cloned.slots:
        s.talk_ids = sorted(talk_ids_by_slot.get(s.id, []))
    return cloned


def enumerate_full_swaps(
    conf: Conference,
    k_max: int,
    rng: np.random.Generator,
    same_day_only: bool = True,
):
    """Аналог enumerate_full_swaps, но с full swap (slot+hall)."""
    if k_max <= 0:
        return []
    candidates = _enumerate_all_pairs(conf, same_day_only=same_day_only)
    valid = []
    for desc in candidates:
        modified = apply_full_swap(conf, desc)
        if not has_speaker_conflict(modified):
            valid.append((modified, desc))
    if not valid:
        return []
    if len(valid) <= k_max:
        return valid
    idx = rng.choice(len(valid), size=k_max, replace=False)
    return [valid[i] for i in sorted(idx)]


CONFERENCE = "mobius_2025_autumn_en"
AUDIENCE_SIZE = 100
AUDIENCE_SEED = 0
PHI_SEED = 17
PHI_KMAX = 500  # больше реального числа валидных пар (344), берём все
CAPACITY_LEVELS = [0.7, 1.0]
GOSSIP_LEVELS = [0.0, 0.3, 0.6]
REPLICATES = [1, 2, 3, 4, 5]
TAU = 0.7
P_SKIP_BASE = 0.10
K = 3

OUT_STEM = "phi_isolated_v2_fullswap_2026-05-12"


def _summarise_cell(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "range": float(arr.max() - arr.min()),
    }


def main():
    t_global = time.time()
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] loading conference={CONFERENCE}...", flush=True)
    base_conf, all_users = load_conference(CONFERENCE)
    print(
        f"  {len(base_conf.talks)} talks, "
        f"{len(base_conf.halls)} halls, "
        f"{len(base_conf.slots)} slots, "
        f"{len(all_users)} personas",
        flush=True,
    )

    audience_users = select_audience(all_users, AUDIENCE_SIZE, AUDIENCE_SEED)
    print(f"  audience: {len(audience_users)} (seed={AUDIENCE_SEED})", flush=True)

    pol = active_policies(include_llm=False)["no_policy"]

    rows: List[dict] = []
    swap_descriptors: Dict[int, dict] = {0: {"swap": None}}
    n_swaps_by_cap: Dict[float, int] = {}

    total_planned = None  # выставим после первого enumerate

    for cap_m in CAPACITY_LEVELS:
        print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] capacity_multiplier={cap_m}", flush=True)
        cap_conf = scale_capacity(base_conf, cap_m)

        rng = np.random.default_rng(PHI_SEED)
        mods = enumerate_full_swaps(
            cap_conf, k_max=PHI_KMAX, rng=rng, same_day_only=True,
        )
        n_swaps = len(mods)
        n_swaps_by_cap[cap_m] = n_swaps
        pv_levels = [(0, cap_conf, None)] + [
            (i + 1, m, d) for i, (m, d) in enumerate(mods)
        ]
        print(f"  enumerate_full_swaps → {n_swaps} valid swaps (cap k_max={PHI_KMAX})", flush=True)
        for pv, _, desc in pv_levels[1:]:
            if pv not in swap_descriptors:
                swap_descriptors[pv] = {
                    "swap": {
                        "slot_a": desc.slot_a, "slot_b": desc.slot_b,
                        "t1": desc.t1, "t2": desc.t2,
                    }
                }

        cell_total = len(pv_levels) * len(GOSSIP_LEVELS) * len(REPLICATES)
        if total_planned is None:
            total_planned = cell_total * len(CAPACITY_LEVELS)
            print(f"  total planned evals across both capacity levels ≈ {total_planned}", flush=True)

        eval_in_cap = 0
        t0_cap = time.time()
        for gossip in GOSSIP_LEVELS:
            w_rel = 1.0 - gossip
            for pv, prog_conf, desc in pv_levels:
                for replicate in REPLICATES:
                    cfg = SimConfig(
                        tau=TAU, p_skip_base=P_SKIP_BASE, K=K,
                        seed=replicate,
                        w_rel=w_rel, w_rec=0.0, w_gossip=gossip,
                        w_fame=0.0,
                    )
                    res = simulate(prog_conf, audience_users, pol, cfg)
                    metrics = compute_metrics_dict(prog_conf, res)
                    rows.append({
                        "capacity_multiplier": cap_m,
                        "w_gossip": gossip,
                        "w_rel": w_rel,
                        "program_variant": pv,
                        "replicate": replicate,
                        **metrics,
                    })
                    eval_in_cap += 1
                    if eval_in_cap % 200 == 0:
                        elapsed = time.time() - t0_cap
                        rate = eval_in_cap / elapsed if elapsed > 0 else 0
                        print(
                            f"    progress: cap={cap_m} evals={eval_in_cap}/{cell_total} "
                            f"({rate:.1f}/s, {elapsed:.1f}s)",
                            flush=True,
                        )
        print(
            f"  done capacity={cap_m}: {eval_in_cap} evals in {time.time()-t0_cap:.1f}s",
            flush=True,
        )

    # --- aggregates per (capacity, gossip) over program_variant × replicate ---
    print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] aggregating...", flush=True)
    per_cell: Dict[str, dict] = {}
    pv_means: Dict[str, Dict[int, float]] = {}
    for cap_m in CAPACITY_LEVELS:
        for gossip in GOSSIP_LEVELS:
            cell_rows = [
                r for r in rows
                if r["capacity_multiplier"] == cap_m and r["w_gossip"] == gossip
            ]
            # per-PV mean over replicates (для агрегата по PV)
            pv_to_vals: Dict[int, List[float]] = {}
            for r in cell_rows:
                pv_to_vals.setdefault(r["program_variant"], []).append(
                    r["mean_overload_excess"]
                )
            pv_mean_map = {pv: float(np.mean(vals)) for pv, vals in pv_to_vals.items()}
            cell_key = f"cap={cap_m}_gossip={gossip}"
            pv_means[cell_key] = pv_mean_map
            per_cell[cell_key] = {
                "capacity_multiplier": cap_m,
                "w_gossip": gossip,
                "n_program_variants": len(pv_to_vals),
                "overload_by_pv": _summarise_cell(list(pv_mean_map.values())),
                "pv0_baseline": pv_mean_map.get(0),
            }

    # --- write outputs ---
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{OUT_STEM}.csv"
    json_path = out_dir / f"{OUT_STEM}.json"
    md_path = out_dir / f"{OUT_STEM}.md"

    fieldnames = [
        "capacity_multiplier", "w_gossip", "w_rel",
        "program_variant", "replicate",
        "mean_overload_excess", "mean_user_utility",
        "overflow_rate_slothall", "hall_utilization_variance",
        "n_skipped", "n_users",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}", flush=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "conference": CONFERENCE,
                    "audience_size": AUDIENCE_SIZE,
                    "audience_seed": AUDIENCE_SEED,
                    "phi_seed": PHI_SEED,
                    "phi_kmax": PHI_KMAX,
                    "capacity_levels": CAPACITY_LEVELS,
                    "gossip_levels": GOSSIP_LEVELS,
                    "replicates": REPLICATES,
                    "tau": TAU, "p_skip_base": P_SKIP_BASE, "K": K,
                    "n_swaps_by_capacity": n_swaps_by_cap,
                    "total_evals": len(rows),
                    "elapsed_s": float(time.time() - t_global),
                },
                "per_cell_aggregates": per_cell,
                "pv_means_by_cell": pv_means,
                "swap_descriptors": swap_descriptors,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"wrote {json_path}", flush=True)

    # --- markdown summary ---
    lines: List[str] = []
    lines.append(f"# Isolated Φ effect — {CONFERENCE} — {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(
        f"Conference: **{CONFERENCE}** ({len(base_conf.talks)} talks, "
        f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots). "
        f"Audience: {AUDIENCE_SIZE} fixed personas (seed={AUDIENCE_SEED}). "
        f"Policy: `no_policy` (П1). `w_rec=0`, `w_fame=0`. "
        f"Replicates: {REPLICATES}. Total evals: **{len(rows)}**, "
        f"wallclock: {time.time()-t_global:.1f}s."
    )
    lines.append("")
    lines.append("## Number of valid swap pairs")
    lines.append("")
    lines.append("| capacity_multiplier | n_valid_swaps |")
    lines.append("|---:|---:|")
    for cap_m, n in n_swaps_by_cap.items():
        lines.append(f"| {cap_m} | {n} |")
    lines.append("")
    lines.append(
        "*Capacity scaling не меняет структуру слотов и спикеров, поэтому "
        "число валидных пар одинаковое в обоих cap-уровнях.*"
    )
    lines.append("")
    lines.append("## Range of mean_overload_excess across program_variants")
    lines.append("")
    lines.append(
        "Каждая ячейка — диапазон средней перегрузки между разными "
        "перестановками программы (mean по 3 replicate per PV), при "
        "зафиксированных остальных осях."
    )
    lines.append("")
    lines.append(
        "| capacity | w_gossip | n_PV | PV=0 (baseline) | min | max | range | mean | std |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cap_m in CAPACITY_LEVELS:
        for gossip in GOSSIP_LEVELS:
            cell = per_cell[f"cap={cap_m}_gossip={gossip}"]
            s = cell["overload_by_pv"]
            base = cell["pv0_baseline"]
            lines.append(
                f"| {cap_m} | {gossip} | {cell['n_program_variants']} | "
                f"{base:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{s['range']:.4f} | {s['mean']:.4f} | {s['std']:.4f} |"
            )
    lines.append("")
    lines.append("## Interpretation cheat-sheet")
    lines.append("")
    lines.append(
        "- **range < 0.005** — оператор Φ практически не меняет перегрузку "
        "в этом режиме; перестановки можно считать неактивной осью."
    )
    lines.append(
        "- **range 0.005–0.05** — слабый, но видимый эффект; имеет смысл "
        "как scenario-axis, но один свап не даёт сильного сдвига."
    )
    lines.append(
        "- **range ≥ 0.05** — оператор содержательно перераспределяет "
        "нагрузку; защита формулировки «Φ — ось сценарного эксперимента» "
        "имеет фактический фундамент."
    )
    lines.append("")
    lines.append("## Top swaps (max overload across cells)")
    lines.append("")
    # find PV with highest overload across all cells (mean over cells)
    pv_overall_max: Dict[int, float] = {}
    for cell_key, pvm in pv_means.items():
        for pv, v in pvm.items():
            pv_overall_max[pv] = max(pv_overall_max.get(pv, 0.0), v)
    top_pvs = sorted(pv_overall_max.items(), key=lambda kv: -kv[1])[:10]
    lines.append("| program_variant | max overload (across cells) | swap_descriptor |")
    lines.append("|---:|---:|---|")
    for pv, ov in top_pvs:
        desc = swap_descriptors.get(pv, {}).get("swap")
        if desc:
            d = f"{desc['t1']} ⇄ {desc['t2']}  ({desc['slot_a']} ⇄ {desc['slot_b']})"
        else:
            d = "P_0 (baseline)"
        lines.append(f"| {pv} | {ov:.4f} | {d} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)

    print(f"\nTotal wallclock: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
