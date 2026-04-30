"""Capacity sensitivity sweep на Demo Day: ±30% от каждой capacity.

Главный вопрос: устойчив ли вывод «Capacity-aware лидер» к выбору значений
вместимости залов (которые в Demo Day назначены экспертно)?
"""
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate, LearnedPreferenceFn  # noqa: E402
from src.metrics import compute_all  # noqa: E402

from src.policies.random_policy import RandomPolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.capacity_aware_mmr_policy import CapacityAwareMMRPolicy  # noqa: E402
from src.policies.mmr_policy import MMRPolicy  # noqa: E402
from src.policies.dpp_policy import DPPPolicy  # noqa: E402


def run_with_caps(conf, users, scale, relevance_fn):
    """Запускает 6 политик на конференции с capacity * scale."""
    # Создаём копию конференции с пере-масштабированными capacities
    from copy import deepcopy
    conf2 = deepcopy(conf)
    for hid, h in conf2.halls.items():
        h.capacity = max(10, int(h.capacity * scale))

    cfg = SimConfig(K=2, tau=0.3, lambda_overflow=2.0, p_skip_base=0.05, seed=42)
    policies = {
        "Random": RandomPolicy(seed=42),
        "Cosine": CosinePolicy(),
        "MMR": MMRPolicy(beta=0.7),
        "Capacity-aware": CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95),
        "Capacity-aware MMR": CapacityAwareMMRPolicy(beta=0.6, alpha=0.4, hard_threshold=0.95),
        "DPP": DPPPolicy(alpha=0.5),
    }
    rows = {}
    for name, pol in policies.items():
        sim = simulate(conf2, users, pol, cfg, relevance_fn=relevance_fn)
        m = compute_all(conf2, sim)
        rows[name] = {
            "overflow_rate_choice": m["overflow_rate_choice"],
            "mean_user_utility": m["mean_user_utility"],
            "hall_utilization_variance": m["hall_utilization_variance"],
        }
    return rows


def main():
    conf = Conference.load(
        ROOT / "data" / "conferences" / "demo_day_2026.json",
        ROOT / "data" / "conferences" / "demo_day_2026_embeddings.npz",
    )
    print(f"Conference: {conf.name}, base capacities: {[h.capacity for h in conf.halls.values()]}")

    with open(ROOT / "data" / "personas" / "personas_x3.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_x3_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}

    def text_of(p):
        return p.get("background") or p.get("profile") or ""

    users = [UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
             for p in meta]

    relevance_fn = LearnedPreferenceFn(ROOT / "data" / "models" / "preference_model.pkl")
    relevance_fn.precompute_all(
        {u.id: u.embedding for u in users},
        {tid: t.embedding for tid, t in conf.talks.items()},
    )

    scales = [0.7, 0.85, 1.0, 1.15, 1.3]
    print(f"\nScale sweep: {scales}")
    print(f"{'Scale':<8}{'Random':<10}{'Cosine':<10}{'MMR':<10}{'C-aware':<10}{'C-aware MMR':<14}{'DPP':<10}")
    print("-" * 70)

    all_results = {}
    for scale in scales:
        rows = run_with_caps(conf, users, scale, relevance_fn)
        all_results[str(scale)] = rows
        of = {p: rows[p]["overflow_rate_choice"] for p in rows}
        print(f"{scale:<8}{of['Random']:<10.3f}{of['Cosine']:<10.3f}{of['MMR']:<10.3f}"
              f"{of['Capacity-aware']:<10.3f}{of['Capacity-aware MMR']:<14.3f}{of['DPP']:<10.3f}")

    out = ROOT / "results" / "capacity_sensitivity.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "scales": scales,
            "results": all_results,
            "interpretation": "Если Capacity-aware OF_choice минимальное во всех scales — лидерство устойчиво к выбору capacity",
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out}")

    # Проверка: лидерство Capacity-aware устойчиво?
    print("\nЛидерство Capacity-aware по OF_choice:")
    for scale, rows in all_results.items():
        ranking = sorted(rows.items(), key=lambda x: x[1]["overflow_rate_choice"])
        leader = ranking[0][0]
        print(f"  scale={scale}: лидер = {leader} (OF={ranking[0][1]['overflow_rate_choice']:.3f})")


if __name__ == "__main__":
    main()
