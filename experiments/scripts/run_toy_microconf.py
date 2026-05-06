"""Этап D PIVOT_IMPLEMENTATION_PLAN: toy-микроконференция и sanity-check.

Изолированно проверяем НОВУЮ модель поведения участника, принятую в
``docs/spikes/spike_behavior_model.md`` (блок Accepted decision):

    U(t | i, slot, hat_pi) = w_rel * rel(t, profile_i) + w_rec * rec(t, hat_pi)
    w_rel + w_rec = 1
    rec(t, hat_pi) = 1{t in recs}
    consider_ids = весь slot   (НЕ top-K от политики)
    capacity-effect: только в политике П3 capacity_aware (НЕ в utility)
    p_skip_base = 0.10 — outside option

Скрипт НЕ вызывает ``experiments/src/simulator.py`` — все вычисления
локальные. Это даёт чистую проверку выбора этапа C до правки ядра на
этапе E. После прохождения acceptance checks ядро будет приведено к этой
же формуле минимальной правкой.

Состав политик в toy:
    no_policy        — П1 (контрольная), recs = [].
    cosine           — П2 (по релевантности), top-K по dot-product.
    capacity_aware   — П3 (с учётом загрузки), score = sim - alpha*load_frac.
    mock_random      — sanity-заглушка ВМЕСТО П4 LLM-ranker для toy. НЕ
                       соответствует поведению П4 и не является валидацией
                       LLM-policy. LLMRankerPolicy в toy сознательно не
                       используется: spike по LLM-симулятору — этап G,
                       реализация — этап H. Mock_random нужен только как
                       4-я политика, чтобы EC3-проверка покрыла случай
                       стохастической политики (правильность фиксации
                       common random numbers).

Параметр K:
    K = 1 в toy — локальное упрощение, продиктованное тем, что в slot
    всего 2 talks. При K = 2 рекомендация тривиально равна всему слоту,
    и EC4 структурно не наблюдается. Это НЕ меняет соглашение K = 2 для
    Mobius smoke (этап F PIVOT_IMPLEMENTATION_PLAN).

Ожидания (PIVOT_IMPLEMENTATION_PLAN §9.D):
    1. w_rec=0 → все 4 политики неразличимы (CV метрик < 5% при 3 seed).
    2. Различия политик по mean_overload_excess монотонно растут с w_rec.
    3. Capacity x3.0 → mean_overload_excess ≈ 0 для всех политик.
    4. Capacity x0.5 → mean_overload_excess > 0.
    5. Asymmetric capacity: П3 снижает mean_overload_excess относительно П2,
       при этом mean_user_utility(П3) > 0.6 * mean_user_utility(П2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # .../experiments
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

EMB_DIM = 8
SLOT_ID = "slot_00"


# ============================================================
# Toy data
# ============================================================

def build_toy_conf() -> dict:
    """1 slot, 2 halls, 2 talks. Минимальный toy из плана §9.D."""
    return {
        "name": "toy_microconf",
        "talks": [
            {"id": "t_ml", "title": "Toy ML talk", "hall": 1, "slot_id": SLOT_ID,
             "category": "ml", "abstract": "Synthetic ML talk for toy sanity."},
            {"id": "t_java", "title": "Toy Java talk", "hall": 2, "slot_id": SLOT_ID,
             "category": "java", "abstract": "Synthetic Java talk for toy sanity."},
        ],
        "halls": [
            {"id": 1, "capacity": 50},
            {"id": 2, "capacity": 50},
        ],
        "slots": [
            {"id": SLOT_ID, "datetime": "2026-01-01T10:00:00"},
        ],
    }


def build_toy_talk_embeddings() -> Tuple[List[str], np.ndarray]:
    """8-мерные синтетические эмбеддинги. e0 — ось ML, e2 — ось Java.

    Орты выбраны чтобы dot(p_ml, t_ml) ≈ 1, dot(p_ml, t_java) ≈ 0
    с небольшой шумовой компонентой персон.
    """
    embs = np.zeros((2, EMB_DIM), dtype=np.float32)
    embs[0, 0] = 1.0  # t_ml
    embs[1, 2] = 1.0  # t_java
    return ["t_ml", "t_java"], embs


def build_toy_personas(n: int = 100, seed: int = 42) -> Tuple[List[dict], np.ndarray]:
    """100 персон: 50 ML-leaning + 50 Java-leaning, 8-мерные нормализованные.

    Базовый вектор персоны лежит вдоль оси своего талка; шумовая компонента
    `noise ~ N(0, 0.15)` по всем 8 координатам обеспечивает уникальность.
    """
    rng = np.random.default_rng(seed)
    personas: List[dict] = []
    embs = np.zeros((n, EMB_DIM), dtype=np.float32)
    half = n // 2
    for i in range(n):
        if i < half:
            cat = "ml"
            base = np.zeros(EMB_DIM)
            base[0] = 1.0
        else:
            cat = "java"
            base = np.zeros(EMB_DIM)
            base[2] = 1.0
        noise = rng.normal(0.0, 0.15, EMB_DIM)
        v = base + noise
        v = v / np.linalg.norm(v)
        embs[i] = v.astype(np.float32)
        personas.append({
            "id": f"p_{i:03d}",
            "category": cat,
            "role": f"toy_{cat}_persona",
            "interests": [cat],
            "background": f"Synthetic toy persona #{i} leaning {cat}.",
        })
    return personas, embs


def save_toy_data(conf: dict, talk_ids: List[str], talk_embs: np.ndarray,
                  personas: List[dict], persona_embs: np.ndarray) -> None:
    confs_dir = DATA_DIR / "conferences"
    pers_dir = DATA_DIR / "personas"
    confs_dir.mkdir(parents=True, exist_ok=True)
    pers_dir.mkdir(parents=True, exist_ok=True)

    (confs_dir / "toy_microconf.json").write_text(
        json.dumps(conf, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        confs_dir / "toy_microconf_embeddings.npz",
        ids=np.array(talk_ids), embeddings=talk_embs,
    )
    (pers_dir / "toy_personas_100.json").write_text(
        json.dumps(personas, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        pers_dir / "toy_personas_100_embeddings.npz",
        ids=np.array([p["id"] for p in personas]),
        embeddings=persona_embs,
    )


# ============================================================
# Policies (локальные стабы для toy; ядро не вызывается)
# ============================================================

PolicyFn = Callable[[np.ndarray, list, np.ndarray, dict, np.random.Generator], List[str]]


def policy_no_policy(user_emb, talks, talk_embs, state, rng) -> List[str]:
    return []


def policy_cosine(user_emb, talks, talk_embs, state, rng) -> List[str]:
    K = state["K"]
    sims = talk_embs @ user_emb
    order = np.argsort(-sims)
    return [talks[i]["id"] for i in order[:K]]


def policy_capacity_aware(user_emb, talks, talk_embs, state, rng,
                          alpha: float = 0.5,
                          hard_threshold: float = 0.95) -> List[str]:
    """Локальный аналог CapacityAwarePolicy: score = sim - alpha*load_frac;
    жёсткий фильтр загрузки >= hard_threshold; fallback по чистому cosine.
    """
    K = state["K"]
    hall_load = state["hall_load"]
    capacities = state["capacities"]
    sims = talk_embs @ user_emb
    scored = []
    for i, t in enumerate(talks):
        cap = capacities[t["hall"]]
        occ = hall_load.get(t["hall"], 0)
        load_frac = occ / max(1.0, cap)
        scored.append((sims[i] - alpha * load_frac, i, load_frac))
    soft = [s for s in scored if s[2] < hard_threshold]
    if soft:
        soft.sort(reverse=True)
        order = [i for _, i, _ in soft[:K]]
    else:
        order = sorted(range(len(talks)), key=lambda i: -sims[i])[:K]
    return [talks[i]["id"] for i in order]


def policy_mock_random(user_emb, talks, talk_embs, state, rng) -> List[str]:
    K = state["K"]
    n = len(talks)
    k = min(K, n)
    idx = rng.choice(n, size=k, replace=False)
    return [talks[i]["id"] for i in idx]


POLICIES: Dict[str, PolicyFn] = {
    "no_policy":      policy_no_policy,
    "cosine":         policy_cosine,
    "capacity_aware": policy_capacity_aware,
    "mock_random":    policy_mock_random,
}


# ============================================================
# Local simulation (НЕ ядро simulator.py)
# ============================================================

def simulate_local(
    conf: dict,
    personas: List[dict],
    persona_embs: np.ndarray,
    talk_embs: np.ndarray,
    policy_fn: PolicyFn,
    w_rec: float,
    *,
    K: int = 1,
    tau: float = 0.7,
    p_skip: float = 0.10,
    seed: int = 0,
) -> dict:
    """Один прогон одного слота с НОВОЙ utility-формулой spike (accepted).

    Common random numbers (CRN). Внутри одного seed используются ДВА
    независимых потока RNG:
      - choice_rng — для shuffle user_order и финального softmax-choice;
      - policy_rng — для стохастичности самой политики (mock_random).

    Это критично: при общем RNG стохастическая политика (mock_random)
    «съедает» состояние RNG в каждом шаге, и последующий choice_rng идёт
    по другой траектории. Для детерминированных политик (no_policy,
    cosine, capacity_aware) поток не сдвигается. Эффект: при w_rec = 0,
    когда utility не зависит от recs, разные политики тем не менее дают
    разный финальный выбор — структурно ломая EC3.

    Разделение потоков обеспечивает: при w_rec = 0 финальный выбор
    зависит только от rels и choice_rng → одинаков для всех 4 политик.
    """
    choice_rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed * 31 + 1)
    talks = conf["talks"]
    capacities = {h["id"]: h["capacity"] for h in conf["halls"]}
    hall_loads = {h["id"]: 0 for h in conf["halls"]}

    user_order = list(range(len(personas)))
    choice_rng.shuffle(user_order)

    chosen_relevances: List[float] = []
    chosen_per_hall = {h["id"]: 0 for h in conf["halls"]}
    chosen_per_talk = {t["id"]: 0 for t in talks}
    n_skipped = 0

    w_rel = 1.0 - w_rec

    for ui in user_order:
        user_emb = persona_embs[ui]
        rels = talk_embs @ user_emb  # (n_talks,)

        state = {
            "K": K,
            "hall_load": dict(hall_loads),
            "capacities": capacities,
        }
        recs = policy_fn(user_emb, talks, talk_embs, state, policy_rng)
        rec_indicator = np.array(
            [1.0 if t["id"] in recs else 0.0 for t in talks],
            dtype=np.float64,
        )

        # consider_ids = весь slot (не recs); capacity-эффект НЕ в utility
        utils = w_rel * rels + w_rec * rec_indicator
        scaled = utils / max(tau, 1e-6)
        scaled = scaled - scaled.max()
        exps = np.exp(scaled)
        probs_recs = (1.0 - p_skip) * (exps / exps.sum())
        probs = np.concatenate([probs_recs, [p_skip]])
        probs = probs / probs.sum()

        idx = int(choice_rng.choice(len(probs), p=probs))
        if idx == len(probs) - 1:
            n_skipped += 1
            continue

        t = talks[idx]
        hall_loads[t["hall"]] += 1
        chosen_per_hall[t["hall"]] += 1
        chosen_per_talk[t["id"]] += 1
        chosen_relevances.append(float(rels[idx]))

    overload_per_hall: List[float] = []
    for h in conf["halls"]:
        occ = hall_loads[h["id"]]
        cap = capacities[h["id"]]
        if occ > cap:
            overload_per_hall.append((occ - cap) / max(1.0, cap))

    return {
        "hall_loads":           {str(k): v for k, v in hall_loads.items()},
        "capacities":           {str(k): v for k, v in capacities.items()},
        "chosen_per_talk":      chosen_per_talk,
        "n_skipped":            int(n_skipped),
        "n_users":              len(personas),
        "mean_overload_excess": float(np.mean(overload_per_hall)) if overload_per_hall else 0.0,
        "mean_user_utility":    float(np.mean(chosen_relevances)) if chosen_relevances else 0.0,
    }


def aggregate_seeds(per_seed: List[dict]) -> dict:
    keys = ["mean_overload_excess", "mean_user_utility", "n_skipped"]
    out: Dict[str, object] = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    hall_ids = list(per_seed[0]["hall_loads"].keys())
    out["hall_loads_mean"] = {
        h: float(np.mean([r["hall_loads"][h] for r in per_seed]))
        for h in hall_ids
    }
    return out


# ============================================================
# Experiment grid + acceptance
# ============================================================

CAPACITY_SCENARIOS: Dict[str, Dict[int, int]] = {
    "base_50_50":         {1: 50,  2: 50},   # multiplier 1.0
    "loose_150_150":      {1: 150, 2: 150},  # multiplier 3.0 — EC1
    "tight_25_25":        {1: 25,  2: 25},   # multiplier 0.5 — EC2
    "asymmetric_20_80":   {1: 20,  2: 80},   # TC-D3
}

W_REC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
K = 1  # 2 talks → K=1: рекомендация не равна тривиально всему слоту
TAU = 0.7
P_SKIP = 0.10
N_PERSONAS = 100


def run_experiment(conf, personas, persona_embs, talk_embs):
    rows = []
    for cap_name, cap_overrides in CAPACITY_SCENARIOS.items():
        local_conf = json.loads(json.dumps(conf))
        for h in local_conf["halls"]:
            if h["id"] in cap_overrides:
                h["capacity"] = cap_overrides[h["id"]]
        for w_rec in W_REC_GRID:
            for pol_name, pol_fn in POLICIES.items():
                per_seed = []
                for seed in SEEDS:
                    r = simulate_local(
                        local_conf, personas, persona_embs, talk_embs,
                        pol_fn, w_rec=w_rec, K=K, tau=TAU, p_skip=P_SKIP,
                        seed=seed,
                    )
                    per_seed.append(r)
                rows.append({
                    "capacity_scenario": cap_name,
                    "policy":            pol_name,
                    "w_rec":             w_rec,
                    "seeds":             SEEDS,
                    "agg":               aggregate_seeds(per_seed),
                    "per_seed":          per_seed,
                })
    return rows


def _cv(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) < 1e-9:
        return 0.0
    return float(np.std(values) / abs(mean))


def check_expectations(rows: List[dict]) -> dict:
    """5 ожиданий этапа D (PIVOT_IMPLEMENTATION_PLAN §9.D)."""
    print("\n=== Acceptance checks (этап D) ===")
    checks: Dict[str, object] = {}

    # 1. EC3 (TC-D1): w_rec=0 → политики неразличимы.
    # Strict-блок (только П1–П3, реальные политики основного эксперимента):
    # ожидание — CV строго ноль (или < 1e-12 от float-арифметики), потому что
    # после CRN-фикса финальный choice не зависит от детерминированной политики.
    # Broad-блок включает mock_random как sanity-заглушку: она не равна П4,
    # но при правильном RNG-разделении тоже даёт идентичный финальный выбор.
    base_w0 = [r for r in rows
               if r["capacity_scenario"] == "base_50_50" and r["w_rec"] == 0.0]
    real_pol_names = {"no_policy", "cosine", "capacity_aware"}
    base_w0_real = [r for r in base_w0 if r["policy"] in real_pol_names]
    base_w0_all = base_w0

    print("\n[1a] EC3 strict (только П1–П3) @ base, w_rec=0:")
    for r in base_w0_real:
        a = r["agg"]
        print(
            f"    {r['policy']:15s}: overload={a['mean_overload_excess_mean']:.4f}"
            f"±{a['mean_overload_excess_std']:.4f}, "
            f"utility={a['mean_user_utility_mean']:.4f}"
            f"±{a['mean_user_utility_std']:.4f}"
        )
    overl_real = [r["agg"]["mean_overload_excess_mean"] for r in base_w0_real]
    util_real = [r["agg"]["mean_user_utility_mean"] for r in base_w0_real]
    cv_overl_real = _cv(overl_real)
    cv_util_real = _cv(util_real)
    range_util_real = float(max(util_real) - min(util_real)) if util_real else 0.0
    print(f"    CV(overload)={cv_overl_real:.6e}, CV(utility)={cv_util_real:.6e}  "
          f"(ожидание после CRN: ≈ 0)")
    print(f"    range(utility) across П1–П3 = {range_util_real:.2e}  "
          f"(ожидание: < 1e-9)")
    checks["ec3_strict_cv_overload"] = cv_overl_real
    checks["ec3_strict_cv_utility"] = cv_util_real
    checks["ec3_strict_range_utility"] = range_util_real
    checks["ec3_strict_pass"] = bool(range_util_real < 1e-9)

    print("\n[1b] EC3 broad (П1–П3 + mock_random sanity) @ base, w_rec=0:")
    for r in base_w0_all:
        a = r["agg"]
        marker = "  *" if r["policy"] == "mock_random" else "   "
        print(
            f"  {marker}{r['policy']:15s}: overload={a['mean_overload_excess_mean']:.4f}"
            f"±{a['mean_overload_excess_std']:.4f}, "
            f"utility={a['mean_user_utility_mean']:.4f}"
            f"±{a['mean_user_utility_std']:.4f}"
        )
    overl_all = [r["agg"]["mean_overload_excess_mean"] for r in base_w0_all]
    util_all = [r["agg"]["mean_user_utility_mean"] for r in base_w0_all]
    cv_overl_all = _cv(overl_all)
    cv_util_all = _cv(util_all)
    range_util_all = float(max(util_all) - min(util_all)) if util_all else 0.0
    print(f"    CV(overload)={cv_overl_all:.6e}, CV(utility)={cv_util_all:.6e}")
    print(f"    range(utility) across all 4 = {range_util_all:.2e}  "
          f"(если CRN корректен — тоже ≈ 0)")
    print("    (* mock_random — sanity-заглушка вместо П4, не валидация LLM-ranker)")
    checks["ec3_broad_cv_overload"] = cv_overl_all
    checks["ec3_broad_cv_utility"] = cv_util_all
    checks["ec3_broad_range_utility"] = range_util_all
    checks["ec3_broad_pass"] = bool(range_util_all < 1e-9)
    checks["ec3_pass"] = checks["ec3_strict_pass"]  # фильтр блокирующий — strict

    # 2. MC3: монотонный рост различий по w_rec.
    # На симметричной base (50/50 cap, 50 ML + 50 Java personas) различий
    # политик по overload структурно нет: любая политика даёт идеальный
    # split в свои залы. MC3 информативен только при capacity-стрессе,
    # т.е. в сценарии asymmetric_20_80, где cosine упирается в малый зал,
    # а capacity_aware его разгружает.
    print("\n[2] MC3 @ asymmetric_20_80: range(overload across policies) по w_rec:")
    ranges = []
    for w in W_REC_GRID:
        bs = [r for r in rows
              if r["capacity_scenario"] == "asymmetric_20_80" and r["w_rec"] == w]
        ovs = [r["agg"]["mean_overload_excess_mean"] for r in bs]
        rng_w = float(max(ovs) - min(ovs))
        ranges.append({"w_rec": w, "range": rng_w})
        print(f"    w_rec={w}: range={rng_w:.4f}")
    is_monotone = all(
        ranges[i]["range"] <= ranges[i + 1]["range"] + 1e-6
        for i in range(len(ranges) - 1)
    )
    # Дополнительная диагностика: на симметричном base различий быть не должно
    base_ranges = []
    for w in W_REC_GRID:
        bs = [r for r in rows
              if r["capacity_scenario"] == "base_50_50" and r["w_rec"] == w]
        ovs = [r["agg"]["mean_overload_excess_mean"] for r in bs]
        base_ranges.append({"w_rec": w, "range": float(max(ovs) - min(ovs))})
    print("    diag @ base_50_50 (для контекста, на симметрии различий быть не должно):")
    for d in base_ranges:
        print(f"      w_rec={d['w_rec']}: range={d['range']:.4f}")
    checks["mc3_ranges_asym"] = ranges
    checks["mc3_ranges_base_diag"] = base_ranges
    checks["mc3_monotone_pass"] = bool(is_monotone)
    print(f"    Монотонно неубывает на asymmetric? {is_monotone}")

    # 3. EC1: cap × 3.0 → overload ≈ 0 для всех политик
    loose = [r for r in rows if r["capacity_scenario"] == "loose_150_150"]
    max_ov_loose = max(r["agg"]["mean_overload_excess_mean"] for r in loose)
    print(f"\n[3] EC1 @ loose (cap×3.0): max overload по политикам/w_rec = "
          f"{max_ov_loose:.4f}  (== 0?)")
    checks["ec1_max_overload_loose"] = max_ov_loose
    checks["ec1_pass"] = bool(max_ov_loose == 0.0)

    # 4. EC2: cap × 0.5 → overload > 0
    tight = [r for r in rows if r["capacity_scenario"] == "tight_25_25"]
    max_ov_tight = max(r["agg"]["mean_overload_excess_mean"] for r in tight)
    print(f"\n[4] EC2 @ tight (cap×0.5): max overload по политикам/w_rec = "
          f"{max_ov_tight:.4f}  (> 0?)")
    checks["ec2_max_overload_tight"] = max_ov_tight
    checks["ec2_pass"] = bool(max_ov_tight > 0.0)

    # 5. Asymmetric: П3 лучше П2 по overload, util_ratio П3/П2 > 0.6
    print("\n[5] TC-D3 @ asymmetric_20_80, по w_rec:")
    asym_rows = []
    for w in W_REC_GRID:
        rows_w = [r for r in rows
                  if r["capacity_scenario"] == "asymmetric_20_80" and r["w_rec"] == w]
        by_pol = {r["policy"]: r["agg"] for r in rows_w}
        if "cosine" in by_pol and "capacity_aware" in by_pol:
            ov_cos = by_pol["cosine"]["mean_overload_excess_mean"]
            ov_cap = by_pol["capacity_aware"]["mean_overload_excess_mean"]
            u_cos = by_pol["cosine"]["mean_user_utility_mean"]
            u_cap = by_pol["capacity_aware"]["mean_user_utility_mean"]
            ratio = (u_cap / u_cos) if u_cos else float("nan")
            print(
                f"    w_rec={w}: cosine ov={ov_cos:.4f} u={u_cos:.4f} | "
                f"cap_aware ov={ov_cap:.4f} u={u_cap:.4f} | "
                f"Δov(cos-cap)={ov_cos - ov_cap:+.4f}, ratio_u(cap/cos)={ratio:.3f}"
            )
            asym_rows.append({
                "w_rec": w,
                "ov_cos": ov_cos, "ov_cap": ov_cap,
                "u_cos": u_cos, "u_cap": u_cap,
                "delta_ov": ov_cos - ov_cap,
                "util_ratio": ratio,
            })
    central = [s for s in asym_rows if s["w_rec"] >= 0.5]
    p3_no_worse = all(s["ov_cap"] <= s["ov_cos"] + 1e-6 for s in central)
    util_ok = all(s["util_ratio"] > 0.6 for s in central if s["u_cos"] > 0)
    print(f"    П3 не хуже П2 по overload @ w_rec>=0.5: {p3_no_worse}")
    print(f"    util_ratio П3/П2 > 0.6 @ w_rec>=0.5: {util_ok}")
    checks["asym_summary"] = asym_rows
    checks["asym_p3_overload_pass"] = bool(p3_no_worse)
    checks["asym_util_ratio_pass"] = bool(util_ok)

    all_pass = (
        checks["ec3_pass"] and checks["mc3_monotone_pass"]
        and checks["ec1_pass"] and checks["ec2_pass"]
        and checks["asym_p3_overload_pass"] and checks["asym_util_ratio_pass"]
    )
    checks["overall_pass"] = bool(all_pass)
    print(f"\n=== Итого: {'OK — все ожидания выполнены' if all_pass else 'NOT OK'} ===\n")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "toy_microconf.json"))
    args = parser.parse_args()

    conf = build_toy_conf()
    talk_ids, talk_embs = build_toy_talk_embeddings()
    personas, persona_embs = build_toy_personas(n=N_PERSONAS, seed=42)
    save_toy_data(conf, talk_ids, talk_embs, personas, persona_embs)

    rows = run_experiment(conf, personas, persona_embs, talk_embs)
    checks = check_expectations(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "etap": "D",
        "model": ("U = w_rel*rel + w_rec*rec_indicator; consider_ids = full slot; "
                  "capacity-effect only in policy П3 capacity_aware; "
                  "p_skip = 0.10 outside option."),
        "params": {
            "K": K, "tau": TAU, "p_skip": P_SKIP,
            "n_personas": len(personas), "seeds": SEEDS, "w_rec_grid": W_REC_GRID,
        },
        "capacity_scenarios": {k: {str(h): c for h, c in v.items()}
                               for k, v in CAPACITY_SCENARIOS.items()},
        "policies": list(POLICIES.keys()),
        "policies_note": ("mock_random — заглушка для П4 LLM-ranker; LLMRankerPolicy "
                          "в toy не используется (spike — этап G, реализация — этап H). "
                          "Mock_random нужен как 4-я политика для ужесточения EC3."),
        "results": rows,
        "checks": checks,
    }, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
