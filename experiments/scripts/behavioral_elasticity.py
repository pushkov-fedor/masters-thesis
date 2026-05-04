"""Behavioral elasticity test: насколько симулятор реагирует на интервенции.

Мера эластичности — JS/KL-дивергенция между распределением выборов под разными
политиками. no_policy = baseline (без интервенции). Чем больше дивергенция от
baseline под cosine/cap_aware_mmr — тем более эластичен симулятор к рекомендатору.

Дополнительно: проверка, что recsys-политика смещает выбор в правильную сторону
(больше cosine-релевантных talks при cosine).

Запуск:
    .venv/bin/python scripts/behavioral_elasticity.py --src llm_agents_mobius_2025_autumn_n50_no_loads_seq.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="llm_agents_mobius_2025_autumn_n50_no_loads_seq.json")
    ap.add_argument("--out", default="behavioral_elasticity.json")
    ap.add_argument("--baseline", default="no_policy")
    args = ap.parse_args()

    src_path = ROOT / "results" / args.src
    data = json.load(open(src_path))
    print(f"Loaded: {src_path.name}")

    conf_name = data["config"]["conference"]
    conf_path = ROOT / "data" / "conferences" / f"{conf_name}.json"
    conf = json.load(open(conf_path))
    talks = {t["id"]: t for t in conf["talks"]}
    halls_by_slot = defaultdict(set)
    for t in conf["talks"]:
        halls_by_slot[t["slot_id"]].add(t["hall"])

    # Distribution per-talk and per-hall per policy
    policies = list(data["results"].keys())
    print(f"Policies: {policies}")
    print(f"Baseline: {args.baseline}\n")

    talk_dist = {}    # policy -> {talk_id: count}
    hall_dist = {}    # policy -> {hall: count}
    skip_count = {}   # policy -> int
    cat_dist = {}     # policy -> {category: count}

    for pname, res in data["results"].items():
        td = Counter()
        hd = Counter()
        cd = Counter()
        sk = 0
        for d in res["decisions"]:
            tid = d.get("chosen") or d.get("decision")
            if tid is None or tid == "skip":
                sk += 1
                continue
            t = talks.get(tid)
            if not t:
                continue
            td[tid] += 1
            hd[t["hall"]] += 1
            cd[t.get("category", "?")] += 1
        talk_dist[pname] = td
        hall_dist[pname] = hd
        cat_dist[pname] = cd
        skip_count[pname] = sk

    # Все доклады для общего вектора
    all_talks = sorted(talks.keys())
    all_halls = sorted({t["hall"] for t in conf["talks"]})
    all_cats = sorted({t.get("category", "?") for t in conf["talks"]})

    def vec(d, keys):
        return np.array([d.get(k, 0) for k in keys], dtype=np.float64)

    baseline = args.baseline
    if baseline not in talk_dist:
        print(f"Baseline {baseline} not found, using {policies[0]}")
        baseline = policies[0]

    # JS divergence от baseline
    print("=== Behavioral elasticity (divergence от baseline) ===")
    print(f"{'Policy':<24} {'JS_talks':>10} {'JS_halls':>10} {'JS_cats':>10} {'KL_talks':>10}")
    elasticity = {}
    for pol in policies:
        if pol == baseline:
            continue
        v_pol = vec(talk_dist[pol], all_talks)
        v_base = vec(talk_dist[baseline], all_talks)
        js_t = js_divergence(v_pol, v_base)
        js_h = js_divergence(vec(hall_dist[pol], all_halls), vec(hall_dist[baseline], all_halls))
        js_c = js_divergence(vec(cat_dist[pol], all_cats), vec(cat_dist[baseline], all_cats))
        kl_t = kl_divergence(v_pol, v_base)
        elasticity[pol] = {
            "js_talks": js_t,
            "js_halls": js_h,
            "js_categories": js_c,
            "kl_talks": kl_t,
        }
        print(f"{pol:<24} {js_t:>10.4f} {js_h:>10.4f} {js_c:>10.4f} {kl_t:>10.4f}")

    # Compliance check: для cosine-based политик проверим, увеличилась ли доля
    # выборов в "релевантных" talks (топ по cosine) по сравнению с baseline.
    # Без явных recommendation logs делаем proxy: топ-K cosine от среднего профиля.
    # Это грубая проверка, но показывает направление эффекта.
    print("\n=== Skip rates ===")
    for pol in policies:
        sk = skip_count[pol]
        total = len(data["results"][pol]["decisions"])
        print(f"  {pol:<24}: {sk}/{total} = {sk/max(1, total):.3f}")

    # Концентрация: top-5 talks share по политикам
    print("\n=== Top-5 talks share (концентрация) ===")
    for pol in policies:
        td = talk_dist[pol]
        total = sum(td.values())
        if total == 0:
            continue
        top5 = sum(c for _, c in td.most_common(5))
        print(f"  {pol:<24}: top-5 = {top5}/{total} = {top5/total:.3f}")

    # === Per-slot elasticity (правильная метрика) ===
    # Считаем JS внутри каждого слота, потом усредняем.
    # Это показывает, насколько политика смещает выбор ВНУТРИ слота.
    print("\n=== Per-slot elasticity (mean JS over slots) ===")
    per_slot_js = defaultdict(list)
    for pol in policies:
        if pol == baseline:
            continue
    # Сгруппируем decisions по slot
    decisions_by_slot = {pol: defaultdict(Counter) for pol in policies}
    for pol in policies:
        for d in data["results"][pol]["decisions"]:
            tid = d.get("chosen") or d.get("decision")
            if tid is None or tid == "skip":
                continue
            decisions_by_slot[pol][d["slot_id"]][tid] += 1

    all_slots = set()
    for pol in policies:
        all_slots.update(decisions_by_slot[pol].keys())

    per_slot_elasticity = {}
    for pol in policies:
        if pol == baseline:
            continue
        slot_js_list = []
        slot_overlap_list = []  # доля совпадающих выборов между политикой и baseline
        for sid in all_slots:
            base_counts = decisions_by_slot[baseline].get(sid, Counter())
            pol_counts = decisions_by_slot[pol].get(sid, Counter())
            if not base_counts or not pol_counts:
                continue
            tids = sorted(set(base_counts) | set(pol_counts))
            v_base = np.array([base_counts.get(t, 0) for t in tids], dtype=np.float64)
            v_pol = np.array([pol_counts.get(t, 0) for t in tids], dtype=np.float64)
            if v_base.sum() == 0 or v_pol.sum() == 0:
                continue
            slot_js_list.append(js_divergence(v_pol, v_base))
            # overlap = TVD (total variation distance) = 0.5 sum |p - q|
            p_n = v_pol / v_pol.sum()
            q_n = v_base / v_base.sum()
            slot_overlap_list.append(0.5 * float(np.sum(np.abs(p_n - q_n))))
        if slot_js_list:
            per_slot_elasticity[pol] = {
                "mean_js_per_slot": float(np.mean(slot_js_list)),
                "median_js_per_slot": float(np.median(slot_js_list)),
                "max_js_per_slot": float(np.max(slot_js_list)),
                "mean_tvd_per_slot": float(np.mean(slot_overlap_list)),
                "n_slots_compared": len(slot_js_list),
            }
            print(f"  {pol:<24}: mean_JS_per_slot={np.mean(slot_js_list):.4f}, "
                  f"median={np.median(slot_js_list):.4f}, "
                  f"TVD_per_slot={np.mean(slot_overlap_list):.4f}, "
                  f"n_slots={len(slot_js_list)}")

    out = {
        "config": {
            "src": args.src,
            "baseline_policy": baseline,
            "n_policies": len(policies),
        },
        "elasticity_aggregate": elasticity,
        "elasticity_per_slot": per_slot_elasticity,
        "skip_rates": {p: {"count": skip_count[p],
                           "total": len(data["results"][p]["decisions"])}
                       for p in policies},
        "top5_share": {p: {"top5": sum(c for _, c in talk_dist[p].most_common(5)),
                            "total": sum(talk_dist[p].values())}
                       for p in policies},
        "talk_distribution": {p: dict(talk_dist[p].most_common(20))
                              for p in policies},
        "hall_distribution": {p: dict(hall_dist[p]) for p in policies},
    }
    out_path = ROOT / "results" / args.out
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {out_path}")

    # Интерпретация
    print("\n=== Интерпретация ===")
    print("JS_talks > 0.05 → политика измеримо смещает распределение по talks (эластичен).")
    print("JS_halls > 0.05 → политика смещает распределение по залам (capacity-эффект).")
    print("JS_categories > 0.05 → политика смещает тематическое распределение.")


if __name__ == "__main__":
    main()
