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
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
    base_conf, all_users = load_conference(conference)
    if verbose:
        print(
            f"loaded {conference}: {len(base_conf.talks)} talks, "
            f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots, "
            f"{len(all_users)} personas",
            flush=True,
        )

    rows = generate_lhs(
        n_points=n_points,
        master_seed=master_seed,
        min_per_level=min_per_level,
    )
    effective_k = min(maximin_k, n_points)
    maximin_idx = maximin_subset(
        rows, k=effective_k,
        force_program_variant_zero=force_pv_zero_in_maximin,
    )
    maximin_set = set(maximin_idx)
    if verbose:
        print(
            f"generated {len(rows)} LHS rows; "
            f"maximin subset (k={effective_k}): {sorted(maximin_idx)}",
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
    n_total_evals = 0
    for row in rows:
        # 1. Capacity scaling
        cfg_capacity_conf = scale_capacity(base_conf, row["capacity_multiplier"])
        # 2. program_variant (фикс по lhs_row_id через phi_seed)
        seeds_const = derive_seeds(row["lhs_row_id"], replicate=1)
        program_conf, program_meta = build_program_variant(
            cfg_capacity_conf,
            row["program_variant"],
            phi_seed=seeds_const["phi_seed"],
            k_max=5,
        )
        # 3. audience subset (фикс по lhs_row_id через audience_seed)
        audience_users = select_audience(
            all_users, row["audience_size"], seeds_const["audience_seed"],
        )
        # 4. cfg parameters
        w_fame = POP_SRC_TO_W_FAME[row["popularity_source"]]
        # 5. Какие политики — П1-П3 всегда; П4 только на maximin subset
        policies_to_run: Dict[str, object] = dict(pols_no_llm)
        if include_llm_ranker and row["lhs_row_id"] in maximin_set:
            policies_to_run["llm_ranker"] = llm_ranker_pol

        for replicate in range(1, replicates + 1):
            seeds = derive_seeds(row["lhs_row_id"], replicate=replicate)
            cfg = SimConfig(
                tau=0.7, p_skip_base=0.10, K=K,
                seed=seeds["cfg_seed"],
                w_rel=row["w_rel"], w_rec=row["w_rec"], w_gossip=row["w_gossip"],
                w_fame=w_fame,
            )
            for pol_name, pol in policies_to_run.items():
                res = simulate(program_conf, audience_users, pol, cfg)
                metrics = compute_metrics_dict(program_conf, res)
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
                    "is_maximin_point":     row["lhs_row_id"] in maximin_set,
                    "swap_descriptor":      program_meta.get("swap_descriptor"),
                    "fallback_to_p0":       program_meta.get("fallback_to_p0", False),
                    **{f"metric_{k}": v for k, v in metrics.items()},
                })
                n_total_evals += 1
        if verbose and ((row["lhs_row_id"] + 1) % 10 == 0
                        or row["lhs_row_id"] + 1 == len(rows)):
            print(
                f"  ... done {row['lhs_row_id'] + 1}/{n_points} rows, "
                f"{n_total_evals} evals",
                flush=True,
            )

    return {
        "etap": "P/Q",
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
    }


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
    ap.add_argument("--out", default=None)
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

    date = dt.date.today().isoformat()
    out_path = Path(args.out) if args.out else (
        ROOT / "results" / f"lhs_parametric_{args.conference}_{date}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                   default=str))
    print(f"\nWROTE: {out_path}")
    print(f"  n_results: {out['n_results']}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
