"""Stylized facts replication: проверяет, воспроизводит ли симулятор три известных
факта из литературы про конференции и event recommendation.

1. **Pareto-attendance** (закон 80/20): топ 20% докладов собирают ~80% посещений.
2. **Time-of-day effect**: утренние слоты (первые) собирают больше людей, чем поздние.
3. **Track-affinity**: пользователи концентрируются вокруг 1-3 категорий, не равномерно.

Источник логов: results_1200_5seeds.json (symbolic simulator, 1200 агентов × 5 seeds × 10 политик).

Метрики и тесты:
- Для Pareto: Lorenz curve + Gini coefficient + p-value Kolmogorov-Smirnov vs uniform
- Для time-of-day: regression slope per slot_idx, p-value
- Для track-affinity: средний H_user (entropy of categories visited per user) vs uniform-bound
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_conference():
    from src.simulator import Conference
    return Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )


def attendance_per_talk_from_logs(conf, results_path: Path, src_file: str = None):
    """Из results.json достаём decisions и считаем количество посещений на talk.

    Поддерживает два формата:
    - старый (agent_validation_full50.json): decision={'decision': talk_id}
    - новый (llm_agents_*.json): decision={'chosen': talk_id}
    """
    candidates = []
    if src_file:
        candidates = [ROOT / "results" / src_file]
    else:
        for name in ["agent_validation_full50.json",
                     "agent_validation_v2_mobius_2025_autumn__100agents_no_chat.json",
                     "llm_agents_mobius_2025_autumn_n50_no_loads_seq.json",
                     "llm_agents_mobius_2025_autumn_n50.json"]:
            p = ROOT / "results" / name
            if p.exists():
                candidates.append(p)
    if not candidates:
        return None
    cand = candidates[0]
    with open(cand) as f:
        data = json.load(f)
    print(f"Reading attendance from: {cand.name}")
    by_policy = {}
    for pname, res in data["results"].items():
        counts = {}
        for d in res["decisions"]:
            tid = d.get("decision") or d.get("chosen")
            if tid and tid != "skip":
                counts[tid] = counts.get(tid, 0) + 1
        by_policy[pname] = counts
    return by_policy


def decisions_per_user_from_logs(conf, src_file: str):
    """Возвращает dict[policy] -> dict[user_id] -> list[talk_id_in_order]."""
    p = ROOT / "results" / src_file
    if not p.exists():
        return {}
    data = json.load(open(p))
    out = {}
    for pname, res in data["results"].items():
        per_user = defaultdict(list)
        for d in res["decisions"]:
            tid = d.get("decision") or d.get("chosen")
            uid = d.get("agent_id")
            if tid and tid != "skip":
                per_user[uid].append(tid)
        out[pname] = dict(per_user)
    return out


def fact_1_pareto(by_policy_counts, conf):
    """Pareto: какая доля топ-20% talks собирает >= 50% посещений?
    Pareto 80/20 говорит, что 20% продуктов = 80% продаж. У нас expectation: для
    Cosine policy концентрация ВЫСОКАЯ (популярные доклады собирают всех),
    для Random — низкая.
    """
    out = {}
    for pname, counts in by_policy_counts.items():
        if not counts:
            continue
        all_attendance = np.zeros(len(conf.talks))
        for i, tid in enumerate(conf.talks.keys()):
            all_attendance[i] = counts.get(tid, 0)
        sorted_attendance = np.sort(all_attendance)[::-1]
        total = sorted_attendance.sum()
        if total == 0:
            continue
        n_talks = len(sorted_attendance)
        n_top20 = max(1, int(0.2 * n_talks))
        share_top20 = sorted_attendance[:n_top20].sum() / total

        # Gini
        cum = np.cumsum(sorted_attendance)
        # Lorenz curve area
        lorenz_x = np.arange(n_talks + 1) / n_talks
        lorenz_y = np.concatenate([[0], cum / total])
        gini = 1 - 2 * np.trapezoid(lorenz_y[::-1], lorenz_x)

        # KS-test против uniform
        ecdf = cum / total
        uniform_cdf = np.arange(1, n_talks + 1) / n_talks
        ks_stat, ks_p = stats.ks_2samp(ecdf, uniform_cdf)

        out[pname] = {
            "top20_share": float(share_top20),
            "gini": float(gini),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "pareto_holds": bool(share_top20 >= 0.5),
        }
    return out


def fact_2_time_of_day(by_policy_decisions, conf, src_file: str = None):
    """Time-of-day: утренние слоты популярнее. Регрессия attendance vs slot_num.
    Поддерживает оба формата (decision/chosen, slot_num/slot_id)."""
    out = {}
    slot_id_to_num = {s.id: i for i, s in enumerate(conf.slots)}
    n_slots = len(conf.slots)

    candidates = []
    if src_file:
        candidates = [ROOT / "results" / src_file]
    else:
        for name in ["agent_validation_full50.json",
                     "llm_agents_mobius_2025_autumn_n50_no_loads_seq.json",
                     "llm_agents_mobius_2025_autumn_n50.json"]:
            p = ROOT / "results" / name
            if p.exists():
                candidates.append(p)
    if not candidates:
        return out
    with open(candidates[0]) as f:
        data = json.load(f)

    for pname, res in data["results"].items():
        per_slot = np.zeros(n_slots)
        per_slot_skips = np.zeros(n_slots)
        for d in res["decisions"]:
            # старый формат: slot_num (1-indexed)
            if "slot_num" in d:
                sn = d["slot_num"] - 1
            else:
                # новый формат: slot_id → ищем по conf
                sn = slot_id_to_num.get(d.get("slot_id"), -1)
            if sn < 0 or sn >= n_slots:
                continue
            decision = d.get("decision") or d.get("chosen")
            if decision is None or decision == "skip":
                per_slot_skips[sn] += 1
            else:
                per_slot[sn] += 1
        total_per_slot = per_slot + per_slot_skips
        attended_share = per_slot / np.maximum(total_per_slot, 1)

        x = np.arange(n_slots)
        slope, intercept, r, p, se = stats.linregress(x, attended_share)
        out[pname] = {
            "attended_share_per_slot": attended_share.tolist(),
            "slope": float(slope),
            "p_value": float(p),
            "r": float(r),
            "decline_per_10_slots": float(slope * 10),
            "fact_holds": bool(slope < 0 and p < 0.05),
        }
    return out


def fact_3_track_affinity(by_policy_decisions, conf, src_file: str = None):
    """Track-affinity: каждый пользователь концентрируется вокруг 1-3 категорий.
    Метрика: средний нормированный entropy of attended categories per user.
    Низкий entropy = высокая концентрация (фактор подтверждается).
    Поддерживает оба формата (decision/chosen).
    """
    out = {}
    candidates = []
    if src_file:
        candidates = [ROOT / "results" / src_file]
    else:
        for name in ["agent_validation_full50.json",
                     "llm_agents_mobius_2025_autumn_n50_no_loads_seq.json",
                     "llm_agents_mobius_2025_autumn_n50.json"]:
            p = ROOT / "results" / name
            if p.exists():
                candidates.append(p)
    if not candidates:
        return out
    with open(candidates[0]) as f:
        data = json.load(f)

    talk_to_cat = {tid: t.category for tid, t in conf.talks.items()}
    categories = sorted({t.category for t in conf.talks.values()})
    n_cats = len(categories)
    max_entropy = np.log(n_cats)

    for pname, res in data["results"].items():
        per_user = {}
        for d in res["decisions"]:
            decision = d.get("decision") or d.get("chosen")
            if decision is None or decision == "skip":
                continue
            cat = talk_to_cat.get(decision, "Unknown")
            per_user.setdefault(d["agent_id"], []).append(cat)

        entropies = []
        for uid, cats in per_user.items():
            if len(cats) < 3:
                continue
            unique, counts = np.unique(cats, return_counts=True)
            p = counts / counts.sum()
            H = -np.sum(p * np.log(p + 1e-12))
            entropies.append(H)
        if not entropies:
            continue
        mean_H = float(np.mean(entropies))
        # Permutation test: H реальных vs H случайных назначений
        rng = np.random.default_rng(42)
        all_categories_observed = []
        for cats in per_user.values():
            all_categories_observed.extend(cats)
        if len(all_categories_observed) < 10:
            continue
        random_entropies = []
        for _ in range(200):
            shuffled = rng.permutation(all_categories_observed)
            ix = 0
            for uid, cats in per_user.items():
                if len(cats) < 3:
                    continue
                shuf = shuffled[ix:ix + len(cats)]
                ix += len(cats)
                _, c = np.unique(shuf, return_counts=True)
                p_ = c / c.sum()
                random_entropies.append(-np.sum(p_ * np.log(p_ + 1e-12)))
        random_entropies = np.array(random_entropies)
        # one-sided p: real H значимо ниже random?
        p_value = (random_entropies <= mean_H).mean()
        out[pname] = {
            "mean_entropy": mean_H,
            "max_entropy": float(max_entropy),
            "normalized_entropy": mean_H / float(max_entropy),
            "random_baseline_entropy": float(np.mean(random_entropies)),
            "p_value_lower_than_random": float(p_value),
            "fact_holds": bool(p_value < 0.05 and mean_H < float(np.mean(random_entropies))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="имя JSON-файла в results/ (auto-detect если не указано)")
    ap.add_argument("--out-suffix", default="",
                    help="суффикс к stylized_facts{suffix}.json")
    args = ap.parse_args()

    conf = load_conference()
    counts_by_policy = attendance_per_talk_from_logs(conf, ROOT / "results",
                                                     src_file=args.src)
    if not counts_by_policy:
        print("Не найдено логов с decisions.")
        return

    print(f"Loaded decisions for {len(counts_by_policy)} policies\n")

    print("=== Stylized Fact 1: Pareto-attendance (top-20% доклады) ===")
    pareto = fact_1_pareto(counts_by_policy, conf)
    for p, m in pareto.items():
        print(f"  {p:<22} top20_share={m['top20_share']:.3f}, gini={m['gini']:.3f}, "
              f"KS_p={m['ks_p_value']:.4f}, Pareto={'✓' if m['pareto_holds'] else '✗'}")

    print("\n=== Stylized Fact 2: Time-of-day (наклон attendance vs slot) ===")
    tod = fact_2_time_of_day(counts_by_policy, conf, src_file=args.src)
    for p, m in tod.items():
        print(f"  {p:<22} slope={m['slope']:+.4f}, p={m['p_value']:.4f}, "
              f"decline_per_10_slots={m['decline_per_10_slots']:+.3f}, "
              f"holds={'✓' if m['fact_holds'] else '✗'}")

    print("\n=== Stylized Fact 3: Track-affinity (entropy of categories per user) ===")
    aff = fact_3_track_affinity(counts_by_policy, conf, src_file=args.src)
    for p, m in aff.items():
        print(f"  {p:<22} norm_H={m['normalized_entropy']:.3f}, "
              f"random_H={m['random_baseline_entropy']:.3f}, "
              f"p_lower={m['p_value_lower_than_random']:.4f}, "
              f"holds={'✓' if m['fact_holds'] else '✗'}")

    # Save
    out = {
        "fact_1_pareto": pareto,
        "fact_2_time_of_day": tod,
        "fact_3_track_affinity": aff,
    }
    suffix = args.out_suffix or ""
    out_path = ROOT / "results" / f"stylized_facts{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")

    # Plot 81: Lorenz curves (Pareto)
    plot_path = ROOT / "results" / "plots" / "81_pareto_lorenz.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pname, counts in counts_by_policy.items():
        if not counts:
            continue
        attendance = np.array([counts.get(tid, 0) for tid in conf.talks.keys()])
        sorted_a = np.sort(attendance)
        total = sorted_a.sum()
        if total == 0:
            continue
        cum = np.cumsum(sorted_a) / total
        x = np.arange(len(sorted_a) + 1) / len(sorted_a)
        y = np.concatenate([[0], cum])
        ax.plot(x, y, label=pname)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="uniform")
    ax.set_xlabel("Доля talks (по нарастающей популярности)")
    ax.set_ylabel("Кумулятивная доля посещений")
    ax.set_title("Stylized Fact 1: Lorenz curves (Pareto-attendance)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"WROTE: {plot_path}")

    # Plot 82: time-of-day
    plot_path = ROOT / "results" / "plots" / "82_time_of_day.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for pname, m in tod.items():
        ax.plot(m["attended_share_per_slot"], "o-", label=pname)
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Attended share per slot")
    ax.set_title("Stylized Fact 2: time-of-day attendance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"WROTE: {plot_path}")

    # Plot 83: track-affinity
    plot_path = ROOT / "results" / "plots" / "83_track_affinity.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    policies = list(aff.keys())
    real_H = [aff[p]["mean_entropy"] for p in policies]
    rand_H = [aff[p]["random_baseline_entropy"] for p in policies]
    x = np.arange(len(policies))
    ax.bar(x - 0.2, real_H, 0.4, label="real users")
    ax.bar(x + 0.2, rand_H, 0.4, label="random shuffle")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("Mean entropy across users")
    ax.set_title("Stylized Fact 3: track-affinity (real H < random H)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"WROTE: {plot_path}")


if __name__ == "__main__":
    main()
