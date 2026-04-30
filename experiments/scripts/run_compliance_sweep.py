"""Sensitivity sweep по user_compliance с включённым fame.

Главный вопрос: при каком compliance Capacity-aware теряет лидерство?
Это **прикладная** граница: насколько recsys полезен в зависимости от того,
насколько пользователи следуют рекомендациям.
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


def text_of(p):
    return p.get("background") or p.get("profile") or ""


def main():
    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )
    print(f"Conference: {conf.name}")
    print(f"Talks with fame > 0.5: {sum(1 for t in conf.talks.values() if t.fame > 0.5)}")

    with open(ROOT / "data" / "personas" / "personas_x3.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / "personas_x3_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    users = [UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
             for p in meta]

    relevance_fn = LearnedPreferenceFn(ROOT / "data" / "models" / "preference_model.pkl")
    relevance_fn.precompute_all(
        {u.id: u.embedding for u in users},
        {tid: t.embedding for tid, t in conf.talks.items()},
    )

    policies_def = {
        "Random": lambda s: RandomPolicy(seed=s),
        "Cosine": lambda s: CosinePolicy(),
        "MMR": lambda s: MMRPolicy(beta=0.7),
        "Capacity-aware": lambda s: CapacityAwarePolicy(alpha=0.5, hard_threshold=0.95),
        "Capacity-aware MMR": lambda s: CapacityAwareMMRPolicy(beta=0.6, alpha=0.4, hard_threshold=0.95),
        "DPP": lambda s: DPPPolicy(alpha=0.5),
    }

    compliance_values = [0.3, 0.5, 0.7, 0.9, 1.0]
    w_fame = 0.3  # включён star-effect
    seeds = [1, 2, 3]

    all_results = {}
    for compliance in compliance_values:
        print(f"\n=== compliance={compliance}, w_fame={w_fame} ===")
        per_policy = {}
        for pname, pol_factory in policies_def.items():
            mse_overflow = []
            mse_utility = []
            for seed in seeds:
                cfg = SimConfig(K=2, tau=0.3, lambda_overflow=2.0, p_skip_base=0.05,
                                seed=seed, w_fame=w_fame, user_compliance=compliance)
                pol = pol_factory(seed)
                sim = simulate(conf, users, pol, cfg, relevance_fn=relevance_fn)
                m = compute_all(conf, sim)
                mse_overflow.append(m["overflow_rate_choice"])
                mse_utility.append(m["mean_user_utility"])
            per_policy[pname] = {
                "overflow_choice_mean": mean(mse_overflow),
                "overflow_choice_std": float(np.std(mse_overflow)),
                "utility_mean": mean(mse_utility),
            }
            print(f"  {pname:<22} OF={mean(mse_overflow):.3f} util={mean(mse_utility):.3f}")
        all_results[str(compliance)] = per_policy

    # Какая политика лидер при каждом compliance?
    print("\nЛидеры по overflow_choice по compliance:")
    for compliance, rows in all_results.items():
        ranking = sorted(rows.items(), key=lambda x: x[1]["overflow_choice_mean"])
        leader = ranking[0]
        print(f"  compliance={compliance}: {leader[0]} (OF={leader[1]['overflow_choice_mean']:.3f})")

    out = ROOT / "results" / "compliance_sweep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"w_fame": w_fame, "seeds": seeds, "policies": list(policies_def.keys())},
            "compliance_values": compliance_values,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
