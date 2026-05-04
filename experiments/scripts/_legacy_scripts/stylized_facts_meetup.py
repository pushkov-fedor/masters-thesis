"""B3: проверка stylized facts из литературы на реальных Meetup RSVPs.

Три факта (те же, что проверены на симуляторе в Главе 4):

1. **Pareto-attendance** (80/20). Тест: Gini коэффициент распределения
   yes-RSVPs по событиям + KS-тест vs равномерное распределение.

2. **Time-of-day**: ранние слоты дня (вечер раньше) собирают больше yes-RSVPs.
   Тест: регрессия count(events) ~ hour, p-value наклона.

3. **Track-affinity**: пользователи возвращаются в 1-3 группы, не размазаны.
   Тест: средняя энтропия distribution-по-группам у users с ≥ 5 RSVPs vs
   равномерное (= log2(K)).

Источник: data/external/.../events.json (с фильтром rsvp_limit + geo).

Выход:
- results/stylized_facts_meetup.json
- results/plots/90_meetup_pareto.png
- results/plots/91_meetup_timeofday.png
- results/plots/92_meetup_trackaffinity.png
"""
from __future__ import annotations

import ast
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "external" / "deep_search_2026_05" / "round2" / "meetup_rsvp" / "RSVP-Prediction-Meetup" / "data"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_topics(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return [t for t in v if isinstance(t, str)] if isinstance(v, list) else []
        except Exception:
            return []
    return []


def gini(x):
    x = np.array(sorted(x), dtype=np.float64)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main():
    print("Loading Meetup raw data...")
    events = json.load(open(SRC / "events.json"))
    venues = {v["venue_id"]: v for v in json.load(open(SRC / "venues.json"))}
    groups = {g["group_id"]: g for g in json.load(open(SRC / "groups.json"))}

    # Фильтр: rsvp_limit ≠ null + венюшка с координатами + та же геозона
    GEO = (52.37, 4.90, 50.0)
    valid = []
    for e in events:
        if not e.get("time"):
            continue
        if e.get("rsvp_limit") is None:
            continue
        v = venues.get(e.get("venue_id"))
        if not v or v.get("lat") is None or v.get("lon") is None:
            continue
        if haversine_km(v["lat"], v["lon"], GEO[0], GEO[1]) > GEO[2]:
            continue
        valid.append(e)
    print(f"  valid events (rsvp_limit + geo): {len(valid)}")

    # =========================================================================
    # F1: Pareto / Gini по yes-RSVPs на событие
    # =========================================================================
    yes_per_event = []
    for e in valid:
        yc = sum(1 for r in (e.get("rsvps") or []) if r.get("response") == "yes")
        yes_per_event.append(yc)
    yes_per_event = [y for y in yes_per_event if y > 0]
    g = gini(yes_per_event)

    # KS test против равномерного: cdf(уровень) сильно отличается?
    # Build empirical cdf vs uniform[0, max]
    arr = np.array(yes_per_event)
    arr_sorted = np.sort(arr)
    cum = np.cumsum(arr_sorted) / arr_sorted.sum()
    # 80/20: какая доля ивентов несёт 80% всех yes?
    target = 0.80
    # Накопление с конца (топовые ивенты)
    arr_desc = np.sort(arr)[::-1]
    csum = np.cumsum(arr_desc) / arr_desc.sum()
    n_top80 = int(np.searchsorted(csum, target) + 1)
    pct_top_for_80 = n_top80 / len(arr) * 100.0

    # Permutation/KS test: distance from uniform [1, max]
    uniform_sample = np.random.default_rng(42).integers(1, arr.max() + 1, size=len(arr))
    ks = stats.ks_2samp(arr, uniform_sample)
    print(f"\n[F1 Pareto]")
    print(f"  N events: {len(arr)}, yes-RSVPs sum: {arr.sum()}")
    print(f"  Gini: {g:.3f}")
    print(f"  Top {pct_top_for_80:.1f}% events accounted for 80% of yes-RSVPs")
    print(f"  KS vs uniform: D={ks.statistic:.3f}, p={ks.pvalue:.2e}")

    # Lorenz plot
    fig, ax = plt.subplots(figsize=(5, 5))
    arr_a = np.sort(arr)
    cum_a = np.cumsum(arr_a) / arr_a.sum()
    x = np.linspace(0, 1, len(cum_a))
    ax.plot(x, cum_a, lw=2, color='darkblue', label=f'Meetup RSVPs (Gini={g:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='равенство')
    ax.set_xlabel('Доля событий (отсортированных по числу yes-RSVPs)')
    ax.set_ylabel('Кумулятивная доля yes-RSVPs')
    ax.set_title('Lorenz curve: распределение посещаемости Meetup-событий')
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = ROOT / "results" / "plots" / "90_meetup_pareto.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)

    # =========================================================================
    # F2: Time-of-day — count yes-RSVPs зависит от часа события
    # =========================================================================
    yes_by_hour = defaultdict(int)
    for e in valid:
        if not e.get("time"):
            continue
        dt = datetime.fromtimestamp(e["time"] / 1000, tz=timezone.utc)
        h = dt.hour
        yc = sum(1 for r in (e.get("rsvps") or []) if r.get("response") == "yes")
        yes_by_hour[h] += yc
    hours = np.array(sorted(yes_by_hour))
    counts = np.array([yes_by_hour[h] for h in hours])
    if len(hours) >= 3:
        lr = stats.linregress(hours, counts)
        slope, intercept = float(lr.slope), float(lr.intercept)
        r2, p = float(lr.rvalue) ** 2, float(lr.pvalue)
    else:
        slope = intercept = r2 = p = float("nan")
    print(f"\n[F2 Time-of-day]")
    print(f"  Hours: {hours.tolist()}")
    print(f"  Yes-RSVPs by hour: {counts.tolist()}")
    print(f"  Linregress slope: {slope:+.2f}, R²={r2:.3f}, p={p:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hours, counts, color='steelblue', alpha=0.8)
    if not math.isnan(slope):
        x_line = np.linspace(hours.min(), hours.max(), 20)
        ax.plot(x_line, intercept + slope * x_line, 'r--',
                label=f'регрессия: slope={slope:+.0f}, p={p:.3f}')
        ax.legend()
    ax.set_xlabel('Час события (UTC)')
    ax.set_ylabel('Сумма yes-RSVPs')
    ax.set_title('Time-of-day: распределение посещаемости по часам')
    ax.grid(alpha=0.3, axis='y')
    plot_path2 = ROOT / "results" / "plots" / "91_meetup_timeofday.png"
    fig.tight_layout()
    fig.savefig(plot_path2, dpi=110)
    plt.close(fig)

    # =========================================================================
    # F3: Track-affinity — пользователи концентрируются в 1-3 группах
    # =========================================================================
    user_groups = defaultdict(Counter)
    for e in valid:
        gid = e.get("group_id")
        for r in (e.get("rsvps") or []):
            if r.get("response") == "yes":
                user_groups[r["user_id"]][gid] += 1

    entropies = []
    n_groups_per_user = []
    for uid, gc in user_groups.items():
        total = sum(gc.values())
        if total < 5:
            continue
        probs = np.array([c / total for c in gc.values()])
        ent = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(ent)
        n_groups_per_user.append(len(gc))
    entropies = np.array(entropies)
    print(f"\n[F3 Track-affinity]")
    print(f"  Active users (≥5 RSVPs): {len(entropies)}")
    print(f"  Mean entropy of group choices: {entropies.mean():.3f} bits")
    print(f"  Median # groups per user: {np.median(n_groups_per_user):.0f}")
    # Сравнение с равномерным выбором по num_groups: H_uniform = log2(num_groups)
    if len(entropies) > 0:
        unif_bound = np.log2(np.mean(n_groups_per_user))
        ratio = entropies.mean() / unif_bound if unif_bound > 0 else float("nan")
        # one-sample test: H_user < H_uniform?
        t_stat, t_p = stats.ttest_1samp(entropies, unif_bound)
        print(f"  Uniform bound (log2(mean #groups)): {unif_bound:.3f}")
        print(f"  H_user / H_uniform: {ratio:.3f} (1.0 = равномерно)")
        print(f"  t-test (H_user < uniform): t={t_stat:.2f}, p={t_p:.2e}")
    else:
        unif_bound = ratio = t_stat = t_p = float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropies, bins=20, color='seagreen', alpha=0.7,
            label=f'эмпирические users (N={len(entropies)})')
    if not math.isnan(unif_bound):
        ax.axvline(unif_bound, color='red', linestyle='--', lw=2,
                   label=f'равномерный выбор (log2 = {unif_bound:.2f})')
    ax.set_xlabel('Энтропия выбора по группам (bits)')
    ax.set_ylabel('Число пользователей')
    ax.set_title('Track-affinity: концентрация выбора пользователей по тематическим группам')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plot_path3 = ROOT / "results" / "plots" / "92_meetup_trackaffinity.png"
    fig.tight_layout()
    fig.savefig(plot_path3, dpi=110)
    plt.close(fig)

    # Финал
    out = {
        "n_valid_events": len(valid),
        "F1_pareto": {
            "n_events_with_yes": len(arr),
            "yes_total": int(arr.sum()),
            "gini": float(g),
            "top_pct_for_80pct": float(pct_top_for_80),
            "ks_D": float(ks.statistic),
            "ks_p": float(ks.pvalue),
        },
        "F2_timeofday": {
            "hours": hours.tolist(),
            "counts": counts.tolist(),
            "slope": float(slope) if not math.isnan(slope) else None,
            "intercept": float(intercept) if not math.isnan(intercept) else None,
            "r_squared": float(r2) if not math.isnan(r2) else None,
            "p_value": float(p) if not math.isnan(p) else None,
        },
        "F3_track_affinity": {
            "n_active_users": int(len(entropies)),
            "mean_entropy_bits": float(entropies.mean()) if len(entropies) else None,
            "median_groups_per_user": float(np.median(n_groups_per_user)) if n_groups_per_user else None,
            "uniform_bound_bits": float(unif_bound) if not math.isnan(unif_bound) else None,
            "ratio_to_uniform": float(ratio) if not math.isnan(ratio) else None,
            "t_stat": float(t_stat) if not math.isnan(t_stat) else None,
            "t_p": float(t_p) if not math.isnan(t_p) else None,
        },
    }
    out_path = ROOT / "results" / "stylized_facts_meetup.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")
    print(f"WROTE: {plot_path}")
    print(f"WROTE: {plot_path2}")
    print(f"WROTE: {plot_path3}")


if __name__ == "__main__":
    main()
