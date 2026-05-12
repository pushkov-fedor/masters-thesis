"""Multi-swap вариант изолированного Φ-эксперимента.

В каждом program_variant — k последовательных полных свапов (k случайно
∈ [K_MIN, K_MAX]). Цель: проверить, остаётся ли эффект перестановок
слабым даже при существенно изменённой программе. Если да — оператор
Φ надёжно слабый, и его можно явно депрекейтить как сильную ось работы.

Особенности дизайна:
  - используется apply_full_swap из run_phi_isolated.py (slot + hall);
  - валидация speaker-conflict отсекает несовместимые комбинации;
  - валидация anti-cancel: после k свапов хотя бы k+1 талк должен
    действительно сменить (slot_id, hall) относительно исходной
    программы — это исключает варианты, где свапы взаимно гасятся;
  - детерминизм: master_seed → те же варианты программы.

Сетка прогона:
  - N_VARIANTS=400 уникальных мульти-свап программ + 1 baseline (P_0)
  - capacity: {0.7, 1.0}
  - w_gossip: {0.0, 0.3, 0.6}
  - replicate: {1, 2, 3, 4, 5}
  - audience_size=100 фикс, audience_seed=0 фикс

Запуск:
    .venv/bin/python scripts/run_phi_isolated_kswap.py
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import (  # noqa: E402
    load_conference,
    scale_capacity,
    select_audience,
    compute_metrics_dict,
)
from scripts.run_phi_isolated import apply_full_swap, has_hall_conflict  # noqa: E402
from src.policies.registry import active_policies  # noqa: E402
from src.program_modification import (  # noqa: E402
    SwapDescriptor,
    _enumerate_all_pairs,
    has_speaker_conflict,
)
from src.simulator import Conference, SimConfig, simulate  # noqa: E402


CONFERENCE = "mobius_2025_autumn_en"
AUDIENCE_SIZE = 100
AUDIENCE_SEED = 0
MASTER_SEED = 20260512
CAPACITY_LEVELS = [0.7, 1.0]
GOSSIP_LEVELS = [0.0, 0.3, 0.6]
REPLICATES = [1, 2, 3, 4, 5]
N_VARIANTS = 400
K_MIN = 3
K_MAX = 8
MAX_ATTEMPTS_PER_VARIANT = 30
TAU = 0.7
P_SKIP_BASE = 0.10
K_TOPK = 3

OUT_STEM = "phi_isolated_kswap_v2_2026-05-12"


def make_kswap_variant(
    base_conf: Conference,
    k: int,
    rng: np.random.Generator,
    all_pairs: List[SwapDescriptor],
    max_attempts: int = MAX_ATTEMPTS_PER_VARIANT,
) -> Tuple[Optional[Conference], Optional[List[SwapDescriptor]], int]:
    """Сэмплирует k полных свапов и проверяет валидность.

    Возвращает (modified_conf, descriptors, n_changed_talks) или
    (None, None, 0) при невозможности после max_attempts попыток.
    """
    for _ in range(max_attempts):
        idx = rng.choice(len(all_pairs), size=k, replace=False)
        descriptors = [all_pairs[int(i)] for i in idx]

        modified = base_conf
        for desc in descriptors:
            modified = apply_full_swap(modified, desc)

        if has_speaker_conflict(modified):
            continue

        # Защита от hall-конфликтов, возникающих при последовательном
        # применении свапов: если несколько свапов «попадают» одной парой
        # (slot, hall), второй переписывает поле и может создать дубликат.
        if has_hall_conflict(modified):
            continue

        n_changed = sum(
            1 for tid, t_orig in base_conf.talks.items()
            if (
                modified.talks[tid].slot_id != t_orig.slot_id
                or modified.talks[tid].hall != t_orig.hall
            )
        )
        # Каждый свап трогает максимум 2 талка. k свапов → ≤ 2k.
        # Требование n_changed >= k+1 гарантирует, что не больше (k-1)/2
        # свапов взаимно отменились (по факту обычно n_changed = 2k, когда
        # все пары не пересекаются по талкам).
        if n_changed < k + 1:
            continue

        return modified, descriptors, n_changed
    return None, None, 0


def _summarise(values: List[float]) -> Dict[str, float]:
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

    # ---- generate variants (same set across capacity levels for fair comparison) ----
    rng_master = np.random.default_rng(MASTER_SEED)
    all_pairs = _enumerate_all_pairs(base_conf, same_day_only=True)
    print(f"  candidate pair pool: {len(all_pairs)} (same-day pairwise, before any filter)", flush=True)

    variants: List[Tuple[Conference, List[SwapDescriptor], int, int]] = []  # (conf, descs, k, n_changed)
    n_failed = 0
    k_distribution: Dict[int, int] = {}
    while len(variants) < N_VARIANTS:
        k = int(rng_master.integers(K_MIN, K_MAX + 1))
        modified, descs, n_changed = make_kswap_variant(
            base_conf, k, rng_master, all_pairs,
        )
        if modified is None:
            n_failed += 1
            if n_failed > N_VARIANTS * 5:
                print(f"WARN: too many failed attempts ({n_failed}); stopping at {len(variants)} variants", flush=True)
                break
            continue
        variants.append((modified, descs, k, n_changed))
        k_distribution[k] = k_distribution.get(k, 0) + 1

    n_kswap = len(variants)
    print(f"  generated {n_kswap} valid k-swap variants ({n_failed} attempts failed)", flush=True)
    print(f"  k distribution: {dict(sorted(k_distribution.items()))}", flush=True)

    # ---- enrich with PV=0 baseline ----
    pv_levels: List[Tuple[int, Conference, Optional[List[SwapDescriptor]], int, int]] = [
        (0, base_conf, None, 0, 0)
    ]
    for i, (mod, descs, k, n_changed) in enumerate(variants, start=1):
        pv_levels.append((i, mod, descs, k, n_changed))

    # ---- run grid ----
    rows: List[dict] = []
    total = len(pv_levels) * len(CAPACITY_LEVELS) * len(GOSSIP_LEVELS) * len(REPLICATES)
    print(f"\n  total planned evals: {total} = {len(pv_levels)} PV × {len(CAPACITY_LEVELS)} cap × {len(GOSSIP_LEVELS)} gossip × {len(REPLICATES)} rep", flush=True)

    for cap_m in CAPACITY_LEVELS:
        print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] capacity_multiplier={cap_m}", flush=True)
        cap_base = scale_capacity(base_conf, cap_m)
        # для каждого PV пересоздать capacity-scaled версию
        cap_pv_confs = [(0, cap_base, None, 0, 0)]
        for pv, mod_full, descs, k, n_ch in pv_levels[1:]:
            cap_pv_confs.append((pv, scale_capacity(mod_full, cap_m), descs, k, n_ch))

        cap_total = len(cap_pv_confs) * len(GOSSIP_LEVELS) * len(REPLICATES)
        eval_in_cap = 0
        t0_cap = time.time()
        for gossip in GOSSIP_LEVELS:
            w_rel = 1.0 - gossip
            for pv, prog_conf, descs, k, n_ch in cap_pv_confs:
                for replicate in REPLICATES:
                    cfg = SimConfig(
                        tau=TAU, p_skip_base=P_SKIP_BASE, K=K_TOPK,
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
                        "k_swaps": k,
                        "n_changed_talks": n_ch,
                        "replicate": replicate,
                        **metrics,
                    })
                    eval_in_cap += 1
                    if eval_in_cap % 500 == 0:
                        elapsed = time.time() - t0_cap
                        rate = eval_in_cap / elapsed if elapsed > 0 else 0
                        print(
                            f"    progress: cap={cap_m} evals={eval_in_cap}/{cap_total} "
                            f"({rate:.1f}/s, {elapsed:.1f}s)",
                            flush=True,
                        )
        print(
            f"  done capacity={cap_m}: {eval_in_cap} evals in {time.time()-t0_cap:.1f}s",
            flush=True,
        )

    # ---- aggregates per (capacity, gossip) ----
    print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] aggregating...", flush=True)
    per_cell: Dict[str, dict] = {}
    pv_means: Dict[str, Dict[int, float]] = {}
    for cap_m in CAPACITY_LEVELS:
        for gossip in GOSSIP_LEVELS:
            cell_rows = [
                r for r in rows
                if r["capacity_multiplier"] == cap_m and r["w_gossip"] == gossip
            ]
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
                "overload_by_pv": _summarise(list(pv_mean_map.values())),
                "pv0_baseline": pv_mean_map.get(0),
            }

    # ---- write outputs ----
    out_dir = ROOT / "results"
    csv_path = out_dir / f"{OUT_STEM}.csv"
    json_path = out_dir / f"{OUT_STEM}.json"
    md_path = out_dir / f"{OUT_STEM}.md"

    fieldnames = [
        "capacity_multiplier", "w_gossip", "w_rel",
        "program_variant", "k_swaps", "n_changed_talks", "replicate",
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
                    "master_seed": MASTER_SEED,
                    "n_variants_target": N_VARIANTS,
                    "n_variants_actual": n_kswap,
                    "k_min": K_MIN, "k_max": K_MAX,
                    "capacity_levels": CAPACITY_LEVELS,
                    "gossip_levels": GOSSIP_LEVELS,
                    "replicates": REPLICATES,
                    "tau": TAU, "p_skip_base": P_SKIP_BASE, "K": K_TOPK,
                    "total_evals": len(rows),
                    "elapsed_s": float(time.time() - t_global),
                    "k_distribution": k_distribution,
                },
                "per_cell_aggregates": per_cell,
                "pv_means_by_cell": pv_means,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"wrote {json_path}", flush=True)

    # ---- markdown summary ----
    lines: List[str] = []
    lines.append(f"# Multi-swap Φ effect — {CONFERENCE} — {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(
        f"В каждом из {n_kswap} вариантов программы применено k полных свапов "
        f"(slot+hall обмен) подряд, k∈[{K_MIN}, {K_MAX}] случайно. Свапы "
        f"проходят валидацию: speaker-conflict отсекается, в итоговой "
        f"программе хотя бы k+1 талк должен реально сменить позицию "
        f"(защита от self-cancel). Конференция: {len(base_conf.talks)} talks, "
        f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots. "
        f"Audience: {AUDIENCE_SIZE} fixed (seed={AUDIENCE_SEED}). Policy: `no_policy`. "
        f"Total evals: **{len(rows)}**, wallclock: {time.time()-t_global:.1f}s."
    )
    lines.append("")
    lines.append(f"k distribution: {dict(sorted(k_distribution.items()))}")
    lines.append("")
    lines.append("## Range of mean_overload_excess across program_variants")
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
    lines.append("## Сравнение с pairwise (1-swap) экспериментом")
    lines.append("")
    lines.append(
        "Если range при k=3..8 свапах сравним с range при k=1 — оператор "
        "действительно слабый рычаг во всём своём пространстве. Если "
        "range при k=3..8 заметно больше — программа сильно меняется "
        "только при больших комбинациях, единичные свапы недостаточны."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)

    print(f"\nTotal wallclock: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
