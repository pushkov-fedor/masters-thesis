"""Smoke EC1/EC3/EC4 на Demo Day EN с BGE-large-en + ABTT-1.

Аналог `smoke_ec_mobius_en.py`, второй инстанс конференции для усиления
аргумента «результаты переносятся».

Цель — подтвердить, что архитектурные инварианты сохраняются на
расширенной программе Demo Day (210 talks, 7 halls, 57 slots).

Сценарии:
- EC1: при capacity_multiplier = 3.0 все 3 политики (П1, П2, П3) дают
  ``mean_overload_excess`` = 0.
- EC3: при ``w_rec = 0`` все политики дают идентичный
  ``mean_overload_excess`` (CRN-инвариантность, range == 0).
- EC4 (бонус): при ``w_rec = 1`` и cap×0.5 политики различимы.

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/smoke_ec_demoday_en.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import mean_hall_overload_excess  # noqa: E402
from src.policies.registry import active_policies  # noqa: E402
from src.simulator import (  # noqa: E402
    Conference,
    SimConfig,
    UserProfile,
    simulate,
)

CONF_JSON = ROOT / "data/conferences/demo_day_2026_en.json"
CONF_EMB = ROOT / "data/conferences/demo_day_2026_en_embeddings.npz"
PERS_JSON = ROOT / "data/personas/personas_demoday_en.json"
PERS_EMB = ROOT / "data/personas/personas_demoday_en_embeddings.npz"


def scale_capacity(conf: Conference, mult: float) -> Conference:
    c = copy.deepcopy(conf)
    for h in c.halls.values():
        h.capacity = max(1, int(round(h.capacity * mult)))
    for s in c.slots:
        if s.hall_capacities:
            s.hall_capacities = {
                hid: max(1, int(round(cap * mult)))
                for hid, cap in s.hall_capacities.items()
            }
    return c


def load_users(n: int = 50):
    pers = json.loads(PERS_JSON.read_text())
    npz = np.load(PERS_EMB, allow_pickle=False)
    emb_map = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    users = [
        UserProfile(id=p["id"], text=p.get("background", ""),
                    embedding=emb_map[p["id"]])
        for p in pers[:n]
    ]
    return users


def avg_overload(conf, users, pol, w_rec: float, seeds=(1, 2, 3, 4, 5)):
    vals = []
    for s in seeds:
        cfg = SimConfig(
            tau=0.7, p_skip_base=0.10, K=3, seed=s,
            w_rel=1.0 - w_rec, w_rec=w_rec, w_gossip=0.0, w_fame=0.0,
        )
        res = simulate(conf, users, pol, cfg)
        vals.append(float(mean_hall_overload_excess(conf, res)))
    return float(np.mean(vals)), vals


def main():
    print(f"Loading {CONF_JSON.name} ...")
    conf = Conference.load(CONF_JSON, CONF_EMB)
    print(f"  talks={len(conf.talks)} slots={len(conf.slots)} halls={len(conf.halls)}")
    print(f"  hall caps: {[(h.id, h.capacity) for h in conf.halls.values()]}")

    print(f"Loading {PERS_JSON.name} (first 50)...")
    users = load_users(50)
    print(f"  users={len(users)}")
    print()

    pols = active_policies(include_llm=False)
    print(f"Policies: {list(pols.keys())}")
    print()

    # ---- EC1: cap×3 → overload = 0 ----
    print("=" * 78)
    print("EC1 — при capacity×3.0 mean_overload_excess = 0 для всех политик")
    print("=" * 78)
    loose = scale_capacity(conf, 3.0)
    print(f"  loose hall caps: {[(h.id, h.capacity) for h in loose.halls.values()]}")
    ec1_fails = []
    for name, pol in pols.items():
        ov, vals = avg_overload(loose, users, pol, w_rec=0.5)
        status = "PASS" if ov < 1e-9 else "FAIL"
        print(f"  {name:<20} mean_overload = {ov:.6f}  per-seed={['%.4f' % v for v in vals]}  [{status}]")
        if ov > 1e-9:
            ec1_fails.append((name, ov))
    if ec1_fails:
        print(f"  EC1 OVERALL: FAIL — {ec1_fails}")
    else:
        print(f"  EC1 OVERALL: PASS — все политики дают overload = 0")
    print()

    # ---- EC3: w_rec=0 → range = 0 ----
    print("=" * 78)
    print("EC3 — при w_rec=0 политики идентичны (CRN-инвариантность)")
    print("=" * 78)
    natural = scale_capacity(conf, 1.0)
    ec3_vals = []
    for name, pol in pols.items():
        ov, vals = avg_overload(natural, users, pol, w_rec=0.0)
        ec3_vals.append((name, ov))
        print(f"  {name:<20} mean_overload = {ov:.6f}")
    overloads = [v for _, v in ec3_vals]
    rng = max(overloads) - min(overloads)
    print(f"  range = {rng:.6e}")
    print(f"  EC3 OVERALL: {'PASS' if rng < 1e-9 else 'FAIL'}")
    print()

    # ---- EC4 (бонус, для контекста): w_rec=1, cap×0.5 ----
    print("=" * 78)
    print("EC4 (bonus) — при w_rec=1, cap×0.5 политики различимы")
    print("=" * 78)
    stress = scale_capacity(conf, 0.5)
    ec4_vals = []
    for name, pol in pols.items():
        ov, vals = avg_overload(stress, users, pol, w_rec=1.0)
        ec4_vals.append((name, ov))
        print(f"  {name:<20} mean_overload = {ov:.6f}")
    overloads = [v for _, v in ec4_vals]
    rng = max(overloads) - min(overloads)
    print(f"  range = {rng:.6f}")
    print(f"  EC4 (bonus): {'PASS (различимы)' if rng > 0.02 else 'FAIL (неразличимы)'}")


if __name__ == "__main__":
    main()
