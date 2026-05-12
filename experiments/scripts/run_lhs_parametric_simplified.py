"""Упрощённый параметрический LHS-прогон по плану #35 PREDZASHCHITA_PLAN.md.

Отличия от `run_lhs_parametric.py`:
- 3 варьируемых оси: `capacity_multiplier ∈ [0.5, 1.5]`, `w_rec`, `w_gossip`;
- `audience_size = 100` фиксировано;
- `popularity_source = cosine_only` (w_fame = 0) фиксировано;
- `program_variant = 0` (исходная программа) фиксировано;
- 3 политики: `no_policy`, `cosine`, `capacity_aware` (без `llm_ranker`).

Реализация — переиспользуется `generate_lhs` из `src.lhs`, но в каждой строке
переопределяются фиксированные поля (audience_size, popularity_source,
program_variant) и пересчитывается capacity_multiplier по узкому диапазону
из той же unit-cube координаты u_raw[0].

Запуск:
    .venv/bin/python scripts/run_lhs_parametric_simplified.py \\
        --conference mobius_2025_autumn_en --n-points 50 --replicates 3
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import (  # noqa: E402
    CONFERENCES,
    POP_SRC_TO_W_FAME,
    load_conference,
    scale_capacity,
    select_audience,
    build_program_variant,
    compute_metrics_dict,
)
from src.lhs import generate_lhs  # noqa: E402
from src.policies.registry import active_policies  # noqa: E402
from src.seeds import derive_seeds  # noqa: E402
from src.simulator import SimConfig, simulate  # noqa: E402


CAPACITY_LOW = 0.5
CAPACITY_HIGH = 1.5
FIXED_AUDIENCE_SIZE = 100
FIXED_POPULARITY_SOURCE = "cosine_only"
FIXED_PROGRAM_VARIANT = 0


def _override_axes(rows: List[Dict]) -> List[Dict]:
    """Перезаписывает фиксированные оси и узкий диапазон capacity."""
    out = []
    for row in rows:
        u_raw = row.get("u_raw", [0.5])
        new_cap = CAPACITY_LOW + float(u_raw[0]) * (CAPACITY_HIGH - CAPACITY_LOW)
        new_row = dict(row)
        new_row["capacity_multiplier"] = new_cap
        new_row["audience_size"] = FIXED_AUDIENCE_SIZE
        new_row["popularity_source"] = FIXED_POPULARITY_SOURCE
        new_row["program_variant"] = FIXED_PROGRAM_VARIANT
        out.append(new_row)
    return out


def run(conference: str, n_points: int, replicates: int, master_seed: int,
        K: int, verbose: bool) -> Dict:
    t_load = time.time()
    base_conf, all_users = load_conference(conference)
    if verbose:
        print(f"loaded {conference}: {len(base_conf.talks)} talks, "
              f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots, "
              f"{len(all_users)} personas ({time.time()-t_load:.2f}s)", flush=True)

    rows_orig = generate_lhs(
        n_points=n_points, master_seed=master_seed,
        min_per_level={"program_variant": 0, "audience_size": 0,
                       "popularity_source": 0},
    )
    rows = _override_axes(rows_orig)
    if verbose:
        caps = [r["capacity_multiplier"] for r in rows]
        print(f"generated {len(rows)} rows; capacity range "
              f"[{min(caps):.3f}, {max(caps):.3f}]", flush=True)

    pols = active_policies(include_llm=False)
    long_rows: List[Dict] = []
    n_evals = 0
    t_run = time.time()

    pbar = tqdm(rows, desc="LHS", unit="row", disable=not verbose)
    for row in pbar:
        cap_conf = scale_capacity(base_conf, row["capacity_multiplier"])
        seeds_const = derive_seeds(row["lhs_row_id"], replicate=1)
        program_conf, _ = build_program_variant(
            cap_conf, row["program_variant"],
            phi_seed=seeds_const["phi_seed"], k_max=5,
        )
        audience_users = select_audience(
            all_users, row["audience_size"], seeds_const["audience_seed"],
        )
        w_fame = POP_SRC_TO_W_FAME[row["popularity_source"]]
        for replicate in range(1, replicates + 1):
            seeds = derive_seeds(row["lhs_row_id"], replicate=replicate)
            cfg = SimConfig(
                tau=0.7, p_skip_base=0.10, K=K,
                seed=seeds["cfg_seed"],
                w_rel=row["w_rel"], w_rec=row["w_rec"],
                w_gossip=row["w_gossip"], w_fame=w_fame,
            )
            for pol_name, pol in pols.items():
                res = simulate(program_conf, audience_users, pol, cfg)
                metrics = compute_metrics_dict(program_conf, res)
                long_rows.append({
                    "lhs_row_id": row["lhs_row_id"],
                    "capacity_multiplier": row["capacity_multiplier"],
                    "popularity_source": row["popularity_source"],
                    "w_rel": row["w_rel"], "w_rec": row["w_rec"],
                    "w_gossip": row["w_gossip"],
                    "audience_size": row["audience_size"],
                    "program_variant": row["program_variant"],
                    "policy": pol_name, "replicate": replicate,
                    **{f"metric_{k}": v for k, v in metrics.items()},
                })
                n_evals += 1

    if verbose:
        print(f"\n{n_evals} evals in {time.time()-t_run:.1f}s", flush=True)

    return {
        "etap": "Q_simplified",
        "conference": conference,
        "params": {
            "n_points": n_points, "replicates": replicates,
            "master_seed": master_seed, "K": K,
            "capacity_range": [CAPACITY_LOW, CAPACITY_HIGH],
            "fixed_audience_size": FIXED_AUDIENCE_SIZE,
            "fixed_popularity_source": FIXED_POPULARITY_SOURCE,
            "fixed_program_variant": FIXED_PROGRAM_VARIANT,
            "policies": list(pols.keys()),
        },
        "lhs_rows": rows,
        "results": long_rows,
        "n_results": len(long_rows),
        "n_evals": n_evals,
    }


CSV_COLUMNS = (
    "lhs_row_id", "capacity_multiplier", "popularity_source",
    "w_rel", "w_rec", "w_gossip", "audience_size", "program_variant",
    "policy", "replicate",
    "metric_mean_overload_excess", "metric_mean_user_utility",
    "metric_overflow_rate_slothall", "metric_hall_utilization_variance",
    "metric_n_skipped", "metric_n_users",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", default="mobius_2025_autumn_en",
                    choices=list(CONFERENCES.keys()))
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out-stem", default="lhs_parametric_simplified_2026-05-12")
    args = ap.parse_args()

    t0 = time.time()
    result = run(
        conference=args.conference,
        n_points=args.n_points,
        replicates=args.replicates,
        master_seed=args.master_seed,
        K=args.K,
        verbose=True,
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{args.out_stem}_{args.conference}"
    json_path = out_dir / f"{out_stem}.json"
    csv_path = out_dir / f"{out_stem}.csv"
    md_path = out_dir / f"{out_stem}.md"

    result["wallclock_total_s"] = float(time.time() - t0)
    result["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"wrote {json_path}", flush=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in result["results"]:
            w.writerow(r)
    print(f"wrote {csv_path}", flush=True)

    # markdown summary
    rows = result["results"]
    by_pol: Dict[str, List[float]] = {}
    for r in rows:
        by_pol.setdefault(r["policy"], []).append(r["metric_mean_overload_excess"])
    L = [
        f"# LHS-parametric (simplified, plan #35) — {args.conference}",
        f"",
        f"Timestamp: {result['timestamp']}; wallclock: {result['wallclock_total_s']:.1f}s",
        f"",
        f"## Configuration",
        f"- n_points: {args.n_points} × replicates: {args.replicates} × policies: {len(result['params']['policies'])}",
        f"- capacity_multiplier ∈ [{CAPACITY_LOW}, {CAPACITY_HIGH}]",
        f"- audience_size = {FIXED_AUDIENCE_SIZE} fixed",
        f"- popularity_source = {FIXED_POPULARITY_SOURCE} fixed",
        f"- program_variant = {FIXED_PROGRAM_VARIANT} fixed",
        f"- policies: {', '.join(result['params']['policies'])}",
        f"",
        f"## Per-policy mean_overload_excess (across all evals)",
        f"",
        f"| policy | n | mean | median | std | p75 |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for pol_name in result["params"]["policies"]:
        v = np.asarray(by_pol.get(pol_name, []), dtype=float)
        if len(v) == 0:
            continue
        L.append(f"| {pol_name} | {len(v)} | {v.mean():.4f} | "
                 f"{np.median(v):.4f} | {v.std(ddof=0):.4f} | "
                 f"{np.percentile(v, 75):.4f} |")
    L.append("")
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
