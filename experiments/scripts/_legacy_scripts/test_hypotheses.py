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


def h3_social_contagion(agent_results_path, n_perm: int = 100):
    """Корреляция между долей друзей в зале и собственным выбором.

    Чтобы исключить reflection problem (Manski 1993) — общую траекторию активности
    в популяции (все skip в late slots из-за fatigue), мы делаем permutation test:
    случайно перетасовываем adjacency между агентами и пересчитываем r.
    Истинный social effect = (raw_r) - (mean(perm_r)).
    """
    import random
    with open(agent_results_path, encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for policy_name, pd in data["results"].items():
        decisions = pd["decisions"]
        adjacency = pd["social_graph_adjacency"]
        # Группируем по slot_id
        by_slot = defaultdict(list)
        for d in decisions:
            by_slot[d["slot_id"]].append(d)

        def compute_r(adj_map):
            x_values, y_values = [], []
            for slot_id, slot_decs in by_slot.items():
                decs_by_idx = {d["agent_idx"]: d for d in slot_decs}
                for agent_idx, d in decs_by_idx.items():
                    friends = adj_map.get(str(agent_idx), [])
                    if not friends:
                        continue
                    friends_active = sum(1 for f in friends
                                         if int(f) in decs_by_idx and decs_by_idx[int(f)]["decision"] != "skip")
                    fa = friends_active / len(friends)
                    sa = 1.0 if d["decision"] != "skip" else 0.0
                    x_values.append(fa)
                    y_values.append(sa)
            if len(x_values) < 30:
                return None, None, 0
            try:
                r, p = pearsonr(x_values, y_values)
                return float(r), float(p), len(x_values)
            except Exception:
                return None, None, 0

        # Raw correlation
        raw_r, raw_p, n_obs = compute_r(adjacency)
        if raw_r is None:
            continue

        # Permutation test: перетасовываем adjacency между агентами случайным образом.
        # Если raw_r объясняется общим трендом (reflection problem), то после перетасовки r не упадёт.
        rng = random.Random(42)
        perm_rs = []
        agent_keys = list(adjacency.keys())
        adj_lists = list(adjacency.values())
        for _ in range(n_perm):
            shuffled_lists = adj_lists[:]
            rng.shuffle(shuffled_lists)
            perm_adj = dict(zip(agent_keys, shuffled_lists))
            pr, _, _ = compute_r(perm_adj)
            if pr is not None:
                perm_rs.append(pr)

        perm_mean = float(np.mean(perm_rs)) if perm_rs else 0.0
        perm_std = float(np.std(perm_rs)) if perm_rs else 0.0
        adjusted_r = raw_r - perm_mean

        results[policy_name] = {
            "raw_pearson_r": raw_r,
            "raw_p_value": raw_p,
            "n_observations": n_obs,
            "permutation_mean_r": perm_mean,
            "permutation_std_r": perm_std,
            "adjusted_r_minus_perm_mean": float(adjusted_r),
            "z_score_vs_perm": float((raw_r - perm_mean) / max(1e-9, perm_std)),
            "n_permutations": n_perm,
        }

    return {
        "hypothesis": "H3: социальное копирование (с permutation control для reflection problem)",
        "expected": "raw_r - perm_mean > 0.1 — истинный social effect отдельно от общего тренда",
        "by_policy": results,
        "verdict": "supported" if any(d["adjusted_r_minus_perm_mean"] > 0.1
                                       for d in results.values()) else "not_supported_after_permutation",
    }


def h5_robustness(*results_paths_with_labels):
    """Устойчивость ranking политик по overflow_rate_choice.

    Делим на 2 независимых ветви:
    1. intra-Mobius: устойчивость к relevance signal (cosine vs learned) — sanity check.
    2. cross-conference: Mobius vs Demo Day — настоящий тест устойчивости.
    """
    all_data = []
    for path, label in results_paths_with_labels:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        by_policy = defaultdict(list)
        for run in r["runs"]:
            by_policy[run["policy"]].append(run["metrics"]["overflow_rate_choice"])
        means = {p: mean(vs) for p, vs in by_policy.items()}
        all_data.append((label, means))

    # Делим на mobius и demo_day группы по префиксу label
    mobius_configs = [(l, m) for (l, m) in all_data if l.startswith("mobius")]
    demo_configs = [(l, m) for (l, m) in all_data if l.startswith("demo_day")]

    def pairwise_rho(configs):
        """Возвращает (avg_rho, list_of_pairs_with_rho_p)."""
        common = None
        for _, m in configs:
            if common is None:
                common = set(m.keys())
            else:
                common = common & set(m.keys())
        common = sorted(common) if common else []
        if len(configs) < 2 or len(common) < 3:
            return None, [], common
        pairs = []
        for i in range(len(configs)):
            for j in range(i+1, len(configs)):
                li, di = configs[i]
                lj, dj = configs[j]
                xs = [di[p] for p in common]
                ys = [dj[p] for p in common]
                rho, p = spearmanr(xs, ys)
                pairs.append({"pair": f"{li}_vs_{lj}",
                              "spearman_rho": float(rho),
                              "p_value": float(p),
                              "n_policies": len(common)})
        avg = mean([p["spearman_rho"] for p in pairs])
        return float(avg), pairs, common

    # Intra-Mobius: устойчивость к смене relevance signal
    intra_avg, intra_pairs, intra_common = pairwise_rho(mobius_configs)
    # Intra-DemoDay
    demo_avg, demo_pairs, demo_common = pairwise_rho(demo_configs)
    # Cross-conference: одна пара (последний mobius vs последний demo)
    cross_pairs = []
    cross_common = []
    if mobius_configs and demo_configs:
        # Берём наиболее полную mobius-конфигу и любую demo-конфигу
        mob_label, mob_means = mobius_configs[-1]
        for d_label, d_means in demo_configs:
            common = sorted(set(mob_means.keys()) & set(d_means.keys()))
            if len(common) < 3:
                continue
            xs = [mob_means[p] for p in common]
            ys = [d_means[p] for p in common]
            rho, p = spearmanr(xs, ys)
            cross_pairs.append({"pair": f"{mob_label}_vs_{d_label}",
                                "spearman_rho": float(rho),
                                "p_value": float(p),
                                "n_policies": len(common)})
            cross_common = common

    cross_avg = mean([p["spearman_rho"] for p in cross_pairs]) if cross_pairs else None

    # Vердикт: cross-conference — главный показатель.
    # intra = sanity check, cross = реальный тест.
    if cross_avg is None:
        verdict = "insufficient_data"
    elif cross_avg >= 0.7 and any(p["p_value"] < 0.05 for p in cross_pairs):
        verdict = "supported"
    elif cross_avg >= 0.5:
        verdict = "weak"
    else:
        verdict = "not_supported"

    return {
        "hypothesis": "H5: устойчивость ranking политик. Главный тест — cross-conference Mobius vs Demo Day.",
        "intra_mobius": {
            "avg_spearman_rho": intra_avg,
            "n_pairs": len(intra_pairs),
            "common_policies": intra_common,
            "interpretation": "sanity check: устойчивость к смене relevance signal внутри одной конференции",
            "pairs": intra_pairs,
        },
        "intra_demo_day": {
            "avg_spearman_rho": demo_avg,
            "n_pairs": len(demo_pairs),
            "pairs": demo_pairs,
        },
        "cross_conference": {
            "avg_spearman_rho": cross_avg,
            "n_pairs": len(cross_pairs),
            "common_policies": cross_common,
            "interpretation": "ГЛАВНЫЙ ТЕСТ: устойчивость к смене конференции (Mobius 40 → Demo Day 210)",
            "pairs": cross_pairs,
        },
        "verdict": verdict,
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
        if h == "H5_robustness" and "cross_conference" in data:
            cross = data["cross_conference"]
            intra = data["intra_mobius"]
            print(f"  intra-Mobius (sanity, к relevance signal): avg ρ={intra['avg_spearman_rho']:.3f} (n_pairs={intra['n_pairs']})")
            print(f"  ГЛАВНЫЙ ТЕСТ — cross-conference (Mobius vs Demo Day): avg ρ={cross['avg_spearman_rho']:.3f} (n_pairs={cross['n_pairs']}, common={cross['n_pairs'] and cross['pairs'][0]['n_policies']})")
            for c in cross["pairs"]:
                print(f"    {c['pair']}: ρ={c['spearman_rho']:+.3f} (p={c['p_value']:.3g}, n={c['n_policies']})")
        elif h == "H2_fatigue" and "by_policy" in data:
            print(f"  expected: {data['expected']}")
            for policy, d in data["by_policy"].items():
                print(f"    {policy:<22} slope={d['slope']:+.4f} (p={d['p_value']:.3g})")
        elif h == "H3_social" and "by_policy" in data:
            print(f"  expected: {data['expected']}")
            for policy, d in data["by_policy"].items():
                print(f"    {policy:<22} raw_r={d['raw_pearson_r']:+.3f}, "
                      f"perm_mean={d['permutation_mean_r']:+.3f}, "
                      f"adjusted={d['adjusted_r_minus_perm_mean']:+.3f}, "
                      f"z={d['z_score_vs_perm']:+.2f}")

    print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
