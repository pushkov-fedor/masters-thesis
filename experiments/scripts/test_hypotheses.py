"""Тесты research-гипотез H1-H5 для главы 4.

Каждая гипотеза проверяется статистическим тестом + эффект-размером,
чтобы можно было сослаться на конкретные числа в тексте работы.

H1 — эмерджентный «эффект звёздного спикера»
H2 — fatigue gradient (skip rate растёт по слотам)
H3 — социальное копирование (herd-эффект)
H4 — sequential beats cosine (опционально, требует prepared baseline)
H5 — устойчивость capacity-aware: Spearman ρ ranking между конфигурациями
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def h2_fatigue_gradient(agent_results_path):
    """Skip rate растёт по слоту? Linear regression slope > 0."""
    with open(agent_results_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []  # (policy, slot_num, skip_rate)
    for policy_name, pd in data["results"].items():
        # Strict per-slot skip rate computed from decisions
        by_slot_num = defaultdict(lambda: {"skip": 0, "total": 0})
        for d in pd["decisions"]:
            by_slot_num[d["slot_num"]]["total"] += 1
            if d["decision"] == "skip":
                by_slot_num[d["slot_num"]]["skip"] += 1
        for slot_num, counts in by_slot_num.items():
            sr = counts["skip"] / max(1, counts["total"])
            rows.append((policy_name, slot_num, sr))

    # Линейная регрессия для каждой политики
    results = {}
    for policy in set(r[0] for r in rows):
        sub = [(r[1], r[2]) for r in rows if r[0] == policy]
        if len(sub) < 3:
            continue
        slot_nums = [s[0] for s in sub]
        skip_rates = [s[1] for s in sub]
        try:
            slope, intercept, r, p, _ = linregress(slot_nums, skip_rates)
        except Exception:
            continue
        results[policy] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": float(r),
            "p_value": float(p),
            "n_slots": len(sub),
        }

    return {
        "hypothesis": "H2: skip rate монотонно растёт по слоту (fatigue)",
        "expected": "slope > 0, p < 0.05",
        "by_policy": results,
        "verdict": "supported" if any(d["slope"] > 0 and d["p_value"] < 0.05
                                       for d in results.values()) else "not_supported",
    }


def h3_social_contagion(agent_results_path):
    """Корреляция между долей друзей в зале и собственным выбором."""
    with open(agent_results_path, encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for policy_name, pd in data["results"].items():
        decisions = pd["decisions"]
        adjacency = pd["social_graph_adjacency"]
        # Группируем по slot_id, потом для каждого агента смотрим social signal
        by_slot = defaultdict(list)
        for d in decisions:
            by_slot[d["slot_id"]].append(d)

        # X: доля друзей агента, выбравших ту же hall
        # Y: 1 если агент пошёл, 0 если skipped
        x_values = []
        y_values = []
        for slot_id, slot_decs in by_slot.items():
            # Build map agent_idx -> chosen_hall (если talk_id известен — нужно достать hall_id)
            # decisions содержит talk_id, но для simplicity используем агрегат:
            # social pressure = доля друзей не-skipped (proxy for «коллеги активны»)
            decs_by_idx = {d["agent_idx"]: d for d in slot_decs}
            for agent_idx, d in decs_by_idx.items():
                friends = adjacency.get(str(agent_idx), [])
                if not friends:
                    continue
                friends_active = sum(1 for f in friends
                                     if int(f) in decs_by_idx and decs_by_idx[int(f)]["decision"] != "skip")
                friend_active_share = friends_active / len(friends)
                self_active = 1.0 if d["decision"] != "skip" else 0.0
                x_values.append(friend_active_share)
                y_values.append(self_active)

        if len(x_values) < 30:
            continue
        try:
            r, p = pearsonr(x_values, y_values)
        except Exception:
            continue
        results[policy_name] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "n_observations": len(x_values),
        }

    return {
        "hypothesis": "H3: социальное копирование — коррелирует ли активность друзей с моей",
        "expected": "Pearson r > 0.1, p < 0.05",
        "by_policy": results,
        "verdict": "supported" if any(d["pearson_r"] > 0.1 and d["p_value"] < 0.05
                                       for d in results.values()) else "not_supported",
    }


def h5_robustness(*results_paths_with_labels):
    """Устойчивость ranking политик по overflow_rate_choice между конфигурациями.

    Каждый results_path — путь к results_*.json, label — короткое имя конфигурации.
    """
    all_data = []
    for path, label in results_paths_with_labels:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        # aggregate per policy
        by_policy = defaultdict(list)
        for run in r["runs"]:
            by_policy[run["policy"]].append(run["metrics"]["overflow_rate_choice"])
        means = {p: mean(vs) for p, vs in by_policy.items()}
        all_data.append((label, means))

    # Spearman correlation между парами конфигураций
    common_policies = None
    for _, m in all_data:
        if common_policies is None:
            common_policies = set(m.keys())
        else:
            common_policies = common_policies & set(m.keys())
    common_policies = sorted(common_policies) if common_policies else []

    if len(all_data) < 2 or len(common_policies) < 3:
        return {
            "hypothesis": "H5: ранжирование политик устойчиво между конфигурациями",
            "verdict": "insufficient_data",
        }

    correlations = {}
    for i in range(len(all_data)):
        for j in range(i+1, len(all_data)):
            li, di = all_data[i]
            lj, dj = all_data[j]
            xs = [di[p] for p in common_policies]
            ys = [dj[p] for p in common_policies]
            r, p = spearmanr(xs, ys)
            correlations[f"{li}_vs_{lj}"] = {
                "spearman_rho": float(r),
                "p_value": float(p),
                "n_policies": len(common_policies),
            }

    avg_rho = mean([c["spearman_rho"] for c in correlations.values()])
    return {
        "hypothesis": "H5: ранжирование политик устойчиво между конфигурациями",
        "expected": "Spearman ρ ≥ 0.5 между конфигурациями (top-3 устойчивы)",
        "common_policies": list(common_policies),
        "pairwise_correlations": correlations,
        "avg_spearman_rho": float(avg_rho),
        "verdict": "supported" if avg_rho >= 0.5 else "weak",
    }


def main():
    out = ROOT / "results" / "hypothesis_tests.json"
    findings = {}

    # H2: fatigue (нужен agent v2 result)
    v2_path = ROOT / "results" / "agent_validation_v2_mobius_2025_autumn_v2.json"
    if v2_path.exists():
        findings["H2_fatigue"] = h2_fatigue_gradient(v2_path)
    else:
        findings["H2_fatigue"] = {"verdict": "data_missing", "path": str(v2_path)}

    # H3: social contagion (нужен agent v2 result)
    if v2_path.exists():
        findings["H3_social"] = h3_social_contagion(v2_path)
    else:
        findings["H3_social"] = {"verdict": "data_missing"}

    # H5: robustness — Spearman ρ между разными конфигурациями
    configs = [
        (ROOT / "results" / "results_cosine.json", "mobius_cosine"),
        (ROOT / "results" / "results_learned.json", "mobius_learned"),
        (ROOT / "results" / "results_learned_full.json", "mobius_learned_with_ppo_llm"),
        (ROOT / "results" / "results_mobius_full11.json", "mobius_11policies"),
        (ROOT / "results" / "results_demo_day_full.json", "demo_day_full"),
        (ROOT / "results" / "results_demo_day_learned.json", "demo_day_learned"),
    ]
    findings["H5_robustness"] = h5_robustness(*configs)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RESEARCH HYPOTHESIS TESTS")
    print("=" * 60)
    for h, data in findings.items():
        print(f"\n{h}: verdict={data.get('verdict', 'N/A')}")
        if h == "H5_robustness" and "avg_spearman_rho" in data:
            print(f"  avg Spearman ρ across configs: {data['avg_spearman_rho']:.3f}")
            for pair, c in data["pairwise_correlations"].items():
                print(f"    {pair}: ρ={c['spearman_rho']:+.3f} (p={c['p_value']:.3g})")
        elif h == "H2_fatigue" and "by_policy" in data:
            print(f"  expected: {data['expected']}")
            for policy, d in data["by_policy"].items():
                print(f"    {policy:<22} slope={d['slope']:+.4f} (p={d['p_value']:.3g})")
        elif h == "H3_social" and "by_policy" in data:
            print(f"  expected: {data['expected']}")
            for policy, d in data["by_policy"].items():
                print(f"    {policy:<22} r={d['pearson_r']:+.3f} (p={d['p_value']:.3g}, n={d['n_observations']})")

    print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
