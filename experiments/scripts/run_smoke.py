"""Этап F PIVOT_IMPLEMENTATION_PLAN: smoke-прогон через ядро.

Запускает обновлённое ядро ``experiments.src.simulator.simulate`` (этап E)
на toy-микроконференции и Mobius с активным реестром П1-П3 (без LLM).
Цель — убедиться, что качественные выводы этапа D воспроизводятся на
полном инстансе Mobius, а не только на изолированной toy.

Состав политик: ``active_policies(include_llm=False)`` →
``no_policy``, ``cosine``, ``capacity_aware``. LLMRankerPolicy на этапе F
сознательно не запускается (spike — этап G, реализация — этап H).

Capacity-сценарии:
    natural        — штатные ёмкости конференции (как в JSON);
    stress_x0_5    — все ёмкости умножены на 0.5 (для EC2 / TC-D3);
    loose_x3_0     — все ёмкости умножены на 3.0 (для EC1).

Mobius (16 слотов / 40 talks / 3 hall × 100 cap, в параллельных слотах
per-slot override 34/34/34) при 100 personas даёт лёгкий стресс уже в
natural-сценарии, но для гарантированной наблюдаемости EC2 и TC-D3
прогон stress_x0_5 обязателен. Это явно зафиксировано в выводе.

Параметры по умолчанию (из PIVOT §9.F):
    --w-rec   "0.0,0.5,1.0"
    --seeds   "1,2,3"
    --K       2

CLI:
    uv run --with numpy python experiments/scripts/run_smoke.py \\
        --conference toy_microconf
    uv run --with numpy python experiments/scripts/run_smoke.py \\
        --conference mobius_2025_autumn

На выходе:
    experiments/results/smoke_<conference>_<date>.json   (полные результаты)
    experiments/results/smoke_<conference>_<date>.md     (краткая сводка)
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.metrics import (  # noqa: E402
    hall_utilization_variance,
    mean_hall_overload_excess,
    mean_user_utility,
    overflow_rate,
)
from src.policies.registry import active_policies  # noqa: E402
from src.simulator import Conference, SimConfig, UserProfile, simulate  # noqa: E402

DATA_DIR = EXPERIMENTS_ROOT / "data"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"

# ---------- Conferences ----------

CONFERENCES: Dict[str, Tuple[str, str, str]] = {
    "toy_microconf": (
        "data/conferences/toy_microconf.json",
        "data/conferences/toy_microconf_embeddings.npz",
        "data/personas/toy_personas_100.json",
    ),
    "mobius_2025_autumn": (
        "data/conferences/mobius_2025_autumn.json",
        "data/conferences/mobius_2025_autumn_embeddings.npz",
        "data/personas/personas_100.json",
    ),
}

CAPACITY_SCENARIOS = {
    "natural":     1.0,
    "stress_x0_5": 0.5,
    "loose_x3_0":  3.0,
}


def load_conference(name: str) -> Tuple[Conference, List[UserProfile]]:
    conf_path, emb_path, pers_path = CONFERENCES[name]
    conf = Conference.load(EXPERIMENTS_ROOT / conf_path,
                           EXPERIMENTS_ROOT / emb_path)
    pj = json.loads((EXPERIMENTS_ROOT / pers_path).read_text())
    pers_emb_path = (EXPERIMENTS_ROOT / pers_path).with_name(
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
        for p in pj
    ]
    return conf, users


def scale_capacity(conf: Conference, mult: float) -> Conference:
    """Размножаем ёмкости как hall.capacity, так и per-slot hall_capacities."""
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


# ---------- Метрики ----------

def compute_metrics(conf: Conference, result) -> dict:
    return {
        "mean_overload_excess":      float(mean_hall_overload_excess(conf, result)),
        "mean_user_utility":         float(mean_user_utility(result)),
        "overflow_rate_slothall":    float(overflow_rate(conf, result, choice_only=False)),
        "hall_utilization_variance": float(hall_utilization_variance(conf, result)),
        "n_skipped":                 int(sum(1 for s in result.steps if s.chosen is None)),
        "n_users":                   int(len(result.steps)),
    }


def aggregate_seeds(per_seed: List[dict]) -> dict:
    keys = ["mean_overload_excess", "mean_user_utility", "overflow_rate_slothall",
            "hall_utilization_variance", "n_skipped"]
    out: Dict[str, object] = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


# ---------- Сетка прогонов ----------

def run_grid(
    conf_name: str,
    w_rec_grid: List[float],
    w_gossip_grid: List[float],
    seeds: List[int],
    K: int,
    cap_scenarios: List[str],
) -> List[dict]:
    base_conf, users = load_conference(conf_name)
    print(f"loaded {conf_name}: {len(base_conf.talks)} talks, "
          f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots, "
          f"{len(users)} users")

    pols = active_policies(include_llm=False)  # П1-П3, без openai/dotenv
    print(f"policies: {sorted(pols.keys())}")

    rows = []
    for cap_name in cap_scenarios:
        scaled_conf = scale_capacity(base_conf, CAPACITY_SCENARIOS[cap_name])
        for w_rec in w_rec_grid:
            for w_gossip in w_gossip_grid:
                # Симплексная нормировка (Q-J4 accepted): w_rel + w_rec + w_gossip = 1.
                # Конфигурации с w_rec + w_gossip > 1 пропускаются явно с предупреждением.
                if w_rec + w_gossip > 1.0 + 1e-9:
                    print(f"  skip: w_rec={w_rec} + w_gossip={w_gossip} > 1.0")
                    continue
                w_rel = max(0.0, 1.0 - w_rec - w_gossip)
                cfg = SimConfig(
                    tau=0.7, p_skip_base=0.10, K=K, seed=0,
                    w_rel=w_rel, w_rec=w_rec, w_gossip=w_gossip,
                )
                for pol_name, pol_obj in pols.items():
                    per_seed = []
                    for s in seeds:
                        cfg.seed = s
                        res = simulate(scaled_conf, users, pol_obj, cfg)
                        per_seed.append(compute_metrics(scaled_conf, res))
                    rows.append({
                        "capacity_scenario": cap_name,
                        "policy":            pol_name,
                        "w_rec":             w_rec,
                        "w_gossip":          w_gossip,
                        "seeds":             seeds,
                        "agg":               aggregate_seeds(per_seed),
                        "per_seed":          per_seed,
                    })
    return rows


# ---------- Acceptance: 5 ожиданий этапа D ----------

def _cv(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values) / abs(mean))


def check_expectations(rows: List[dict], stress_scenario: str) -> dict:
    """Все 5 ожиданий этапа D через ядро + L.6 sensitivity по w_gossip
    (этап K, gossip-инкремент). Старые EC берутся при w_gossip = 0; новый
    L.6-чек проверяет видимость gossip-эффекта на любой паре политик / w_rec
    при изменении w_gossip ∈ {0, 0.3, 0.7} (если в сетке есть).
    """
    checks: Dict[str, object] = {}

    # Базовые EC при w_gossip = 0 (baseline; точка совместимости с этапом E).
    baseline_rows = [r for r in rows if abs(r.get("w_gossip", 0.0)) < 1e-9]

    # 1. EC3 strict при w_rec=0, w_gossip=0 на natural — все политики строго одинаковы
    nat_w0 = [r for r in baseline_rows
              if r["capacity_scenario"] == "natural" and r["w_rec"] == 0.0]
    util = [r["agg"]["mean_user_utility_mean"] for r in nat_w0]
    overl = [r["agg"]["mean_overload_excess_mean"] for r in nat_w0]
    range_util = float(max(util) - min(util)) if util else 0.0
    range_overl = float(max(overl) - min(overl)) if overl else 0.0
    checks["ec3_range_utility"] = range_util
    checks["ec3_range_overload"] = range_overl
    checks["ec3_pass"] = bool(range_util < 1e-9 and range_overl < 1e-9)

    # 2. MC3: монотонный рост различий по w_rec на stress-сценарии (w_gossip=0)
    ranges = []
    w_rec_grid = sorted({r["w_rec"] for r in baseline_rows})
    for w in w_rec_grid:
        bs = [r for r in baseline_rows
              if r["capacity_scenario"] == stress_scenario and r["w_rec"] == w]
        ovs = [r["agg"]["mean_overload_excess_mean"] for r in bs]
        if not ovs:
            continue
        ranges.append({"w_rec": w, "range": float(max(ovs) - min(ovs))})
    is_monotone = all(
        ranges[i]["range"] <= ranges[i + 1]["range"] + 1e-9
        for i in range(len(ranges) - 1)
    )
    checks["mc3_ranges_stress"] = ranges
    checks["mc3_monotone_pass"] = bool(is_monotone)

    # 3. EC1 на loose
    loose = [r for r in baseline_rows if r["capacity_scenario"] == "loose_x3_0"]
    max_ov_loose = max((r["agg"]["mean_overload_excess_mean"] for r in loose),
                       default=0.0)
    checks["ec1_max_overload_loose"] = max_ov_loose
    checks["ec1_pass"] = bool(max_ov_loose == 0.0)

    # 4. EC2 на stress
    stress = [r for r in baseline_rows if r["capacity_scenario"] == stress_scenario]
    max_ov_stress = max((r["agg"]["mean_overload_excess_mean"] for r in stress),
                        default=0.0)
    checks["ec2_max_overload_stress"] = max_ov_stress
    checks["ec2_pass"] = bool(max_ov_stress > 0.0)

    # 5. TC-D3 на stress: capacity_aware vs cosine
    asym_rows = []
    for w in w_rec_grid:
        rows_w = [r for r in baseline_rows
                  if r["capacity_scenario"] == stress_scenario and r["w_rec"] == w]
        by_pol = {r["policy"]: r["agg"] for r in rows_w}
        if "cosine" in by_pol and "capacity_aware" in by_pol:
            ov_cos = by_pol["cosine"]["mean_overload_excess_mean"]
            ov_cap = by_pol["capacity_aware"]["mean_overload_excess_mean"]
            u_cos = by_pol["cosine"]["mean_user_utility_mean"]
            u_cap = by_pol["capacity_aware"]["mean_user_utility_mean"]
            ratio = (u_cap / u_cos) if u_cos else float("nan")
            asym_rows.append({
                "w_rec": w,
                "ov_cos": ov_cos, "ov_cap": ov_cap,
                "u_cos": u_cos, "u_cap": u_cap,
                "delta_ov": ov_cos - ov_cap,
                "util_ratio": ratio,
            })
    central = [s for s in asym_rows if s["w_rec"] >= 0.5]
    p3_no_worse = all(s["ov_cap"] <= s["ov_cos"] + 1e-9 for s in central)
    util_ok = all(s["util_ratio"] > 0.6 for s in central if s["u_cos"] > 0)
    checks["asym_summary"] = asym_rows
    checks["asym_p3_overload_pass"] = bool(p3_no_worse)
    checks["asym_util_ratio_pass"] = bool(util_ok)

    # 6. L.6 sensitivity по w_gossip (новое в этапе K). Считаем range overload
    # в фиксированной точке (capacity=stress, w_rec=0.5) по всем доступным
    # значениям w_gossip; если range >= 0.05 хотя бы для одной политики —
    # gossip-эффект видим.
    w_gossip_grid = sorted({r.get("w_gossip", 0.0) for r in rows})
    gossip_sens = []
    if len(w_gossip_grid) >= 2:
        for pol_name in {r["policy"] for r in rows}:
            pol_rows = [r for r in rows
                        if r["capacity_scenario"] == stress_scenario
                        and abs(r["w_rec"] - 0.5) < 1e-9
                        and r["policy"] == pol_name]
            if len(pol_rows) >= 2:
                ovs = [r["agg"]["mean_overload_excess_mean"] for r in pol_rows]
                gossip_sens.append({
                    "policy": pol_name,
                    "w_gossip_range": [min(r["w_gossip"] for r in pol_rows),
                                       max(r["w_gossip"] for r in pol_rows)],
                    "overload_range": float(max(ovs) - min(ovs)),
                })
    gossip_visible = (
        bool(gossip_sens)
        and any(s["overload_range"] >= 0.05 for s in gossip_sens)
    )
    checks["gossip_sensitivity"] = gossip_sens
    checks["gossip_visible_pass"] = bool(gossip_visible) if len(w_gossip_grid) >= 2 else None

    overall = (
        checks["ec3_pass"] and checks["mc3_monotone_pass"]
        and checks["ec1_pass"] and checks["ec2_pass"]
        and checks["asym_p3_overload_pass"] and checks["asym_util_ratio_pass"]
    )
    # gossip_visible — диагностика, не блокатор overall_pass этапа F (он остаётся
    # в своих рамках); полная acceptance этапа L — отдельно.
    checks["overall_pass"] = bool(overall)
    return checks


# ---------- Markdown summary ----------

def render_markdown(conf_name: str, rows: List[dict], checks: dict,
                    params: dict, stress_scenario: str) -> str:
    lines = []
    lines.append(f"# Smoke-прогон этапа F: `{conf_name}`")
    lines.append("")
    lines.append(f"Дата: {params['date']}")
    lines.append(f"Ядро: `experiments/src/simulator.py` (этап E).")
    lines.append(f"Реестр политик: `active_policies(include_llm=False)` → "
                 f"{params['policies']}.")
    lines.append(f"Параметры: K={params['K']}, τ=0.7, p_skip=0.10, "
                 f"seeds={params['seeds']}, w_rec={params['w_rec_grid']}, "
                 f"w_gossip={params.get('w_gossip_grid', [0.0])}.")
    lines.append(f"Capacity-сценарии: {list(CAPACITY_SCENARIOS.keys())}.")
    lines.append("")

    lines.append("## Ключевые метрики (mean over seeds)")
    lines.append("")
    for cap_name in CAPACITY_SCENARIOS.keys():
        lines.append(f"### Capacity scenario: `{cap_name}` "
                     f"(×{CAPACITY_SCENARIOS[cap_name]})")
        lines.append("")
        lines.append("| w_rec | w_gossip | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |")
        lines.append("|------:|---------:|--------|----------------:|-------------:|--------------:|---------:|-------:|")
        for r in rows:
            if r["capacity_scenario"] != cap_name:
                continue
            a = r["agg"]
            lines.append(
                f"| {r['w_rec']:.2f} | {r.get('w_gossip', 0.0):.2f} | "
                f"{r['policy']:14s} | "
                f"{a['mean_overload_excess_mean']:.4f} | "
                f"{a['mean_user_utility_mean']:.4f} | "
                f"{a['overflow_rate_slothall_mean']:.4f} | "
                f"{a['hall_utilization_variance_mean']:.4f} | "
                f"{a['n_skipped_mean']:.1f} |"
            )
        lines.append("")

    lines.append("## Acceptance: 5 ожиданий этапа D через ядро")
    lines.append("")
    lines.append(f"Stress-сценарий для MC3 / EC2 / TC-D3: `{stress_scenario}`.")
    lines.append("")
    lines.append(f"- **EC3 strict** (w_rec=0, natural): "
                 f"range(utility)={checks['ec3_range_utility']:.2e}, "
                 f"range(overload)={checks['ec3_range_overload']:.2e} → "
                 f"**{'PASS' if checks['ec3_pass'] else 'FAIL'}**")
    mc3_summary = ", ".join(
        f"w={r['w_rec']:.2f}: {r['range']:.4f}"
        for r in checks["mc3_ranges_stress"]
    )
    lines.append(f"- **MC3 monotone** на `{stress_scenario}`: [{mc3_summary}] → "
                 f"**{'PASS' if checks['mc3_monotone_pass'] else 'FAIL'}**")
    lines.append(f"- **EC1** (loose ×3.0): max overload = "
                 f"{checks['ec1_max_overload_loose']:.4f} → "
                 f"**{'PASS' if checks['ec1_pass'] else 'FAIL'}**")
    lines.append(f"- **EC2** (`{stress_scenario}`): max overload = "
                 f"{checks['ec2_max_overload_stress']:.4f} → "
                 f"**{'PASS' if checks['ec2_pass'] else 'FAIL'}**")
    lines.append(f"- **TC-D3** (П3 vs П2 на `{stress_scenario}`):")
    for s in checks["asym_summary"]:
        lines.append(f"    - w_rec={s['w_rec']:.2f}: "
                     f"Δoverload(cos-cap)={s['delta_ov']:+.4f}, "
                     f"util_ratio(cap/cos)={s['util_ratio']:.3f}")
    lines.append(f"  - П3 не хуже П2 по overload @ w_rec≥0.5: "
                 f"**{'PASS' if checks['asym_p3_overload_pass'] else 'FAIL'}**")
    lines.append(f"  - util_ratio(П3/П2) > 0.6 @ w_rec≥0.5: "
                 f"**{'PASS' if checks['asym_util_ratio_pass'] else 'FAIL'}**")
    lines.append("")

    # L.6 sensitivity по w_gossip — диагностика, появляется только если в сетке
    # есть ≥ 2 точек w_gossip.
    if checks.get("gossip_visible_pass") is not None:
        lines.append("## Sensitivity по `w_gossip` (диагностика этапа K, L.6)")
        lines.append("")
        for s in checks.get("gossip_sensitivity", []):
            lines.append(
                f"- policy=`{s['policy']}`: w_gossip ∈ "
                f"{s['w_gossip_range']}, range(mean_overload_excess) = "
                f"{s['overload_range']:.4f}"
            )
        lines.append(
            f"- видимый эффект (range ≥ 0.05 хоть для одной политики): "
            f"**{'PASS' if checks['gossip_visible_pass'] else 'FAIL'}**"
        )
        lines.append("")

    lines.append(f"### Итог: **{'OK — все ожидания выполнены' if checks['overall_pass'] else 'NOT OK'}**")
    lines.append("")
    return "\n".join(lines)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference", required=True,
                        choices=list(CONFERENCES.keys()))
    parser.add_argument("--w-rec", default="0.0,0.5,1.0")
    parser.add_argument("--w-gossip", default="0.0",
                        help="comma-separated w_gossip values; default '0.0' "
                             "сохраняет поведение этапа F. Для проверки "
                             "gossip-инкремента (этап L) используй '0.0,0.3,0.7'. "
                             "Симплексная нормировка: w_rec + w_gossip ≤ 1.")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--capacity-scenarios",
                        default="natural,stress_x0_5,loose_x3_0")
    parser.add_argument("--stress-scenario", default="stress_x0_5",
                        help="Capacity scenario, на котором проверяются "
                             "MC3 / EC2 / TC-D3 (требует stress).")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    w_rec_grid = [float(x) for x in args.w_rec.split(",")]
    w_gossip_grid = [float(x) for x in args.w_gossip.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cap_scenarios = [s.strip() for s in args.capacity_scenarios.split(",")]
    K = args.K

    # Для toy 2 talks ⇒ effective_K = 1 (ядро capped); для Mobius K=2.
    rows = run_grid(args.conference, w_rec_grid, w_gossip_grid, seeds, K,
                    cap_scenarios)
    checks = check_expectations(rows, args.stress_scenario)

    date = dt.date.today().isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"smoke_{args.conference}_{date}.json"
    out_md = out_dir / f"smoke_{args.conference}_{date}.md"

    pols = active_policies(include_llm=False)
    params = {
        "date": date, "K": K, "seeds": seeds,
        "w_rec_grid": w_rec_grid,
        "w_gossip_grid": w_gossip_grid,
        "policies": sorted(pols.keys()),
        "stress_scenario": args.stress_scenario,
    }
    out_json.write_text(json.dumps({
        "etap": "F",
        "conference": args.conference,
        "params": params,
        "capacity_scenarios": CAPACITY_SCENARIOS,
        "results": rows,
        "checks": checks,
    }, indent=2, ensure_ascii=False))
    out_md.write_text(render_markdown(args.conference, rows, checks,
                                      params, args.stress_scenario))
    print(f"\nsaved: {out_json}")
    print(f"saved: {out_md}")
    print(f"\n=== Итог: {'OK' if checks['overall_pass'] else 'NOT OK'} ===")


if __name__ == "__main__":
    main()
