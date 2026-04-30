"""Графики для главы 4."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import Conference, SimConfig, UserProfile, simulate  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.capacity_aware_mmr_policy import CapacityAwareMMRPolicy  # noqa: E402

PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIG_DIR = ROOT.parent / "thesis" / "figures"

POLICY_ORDER = ["Random", "Cosine", "MMR", "Capacity-aware", "Capacity-aware MMR", "LLM-ranker"]
POLICY_COLORS = {
    "Random": "#888888",
    "Cosine": "#1f77b4",
    "MMR": "#9467bd",
    "Capacity-aware": "#2ca02c",
    "Capacity-aware MMR": "#d62728",
    "LLM-ranker": "#ff7f0e",
}


def load_results():
    with open(ROOT / "results" / "results.json", encoding="utf-8") as f:
        return json.load(f)


def aggregate(results, metric):
    by_policy = defaultdict(list)
    for r in results["runs"]:
        by_policy[r["policy"]].append(r["metrics"][metric])
    return {
        p: (mean(vs), stdev(vs) if len(vs) > 1 else 0.0)
        for p, vs in by_policy.items()
    }


def plot_bar(metric_name: str, title: str, ylabel: str, results, fname: str, lower_is_better=True):
    agg = aggregate(results, metric_name)
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(POLICY_ORDER))
    means = [agg[p][0] for p in POLICY_ORDER]
    stds = [agg[p][1] for p in POLICY_ORDER]
    colors = [POLICY_COLORS[p] for p in POLICY_ORDER]
    bars = ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(POLICY_ORDER, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.5,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    if lower_is_better:
        ax.text(0.99, 0.97, "ниже = лучше", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="gray", style="italic")
    else:
        ax.text(0.99, 0.97, "выше = лучше", transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_tradeoff(results, fname: str):
    """Scatter: overflow vs utility — нижний правый угол лучший."""
    overflow_agg = aggregate(results, "overflow_rate")
    util_agg = aggregate(results, "mean_user_utility")
    fig, ax = plt.subplots(figsize=(8, 6))
    for p in POLICY_ORDER:
        ox, ostd = overflow_agg[p]
        uy, ustd = util_agg[p]
        ax.errorbar(ox, uy, xerr=ostd, yerr=ustd, fmt="o",
                    color=POLICY_COLORS[p], markersize=12, label=p,
                    capsize=4, linewidth=1.5)
        ax.annotate(p, xy=(ox, uy), xytext=(8, -8), textcoords="offset points",
                    fontsize=10)
    ax.set_xlabel("Overflow rate (доля переполненных (slot, hall))")
    ax.set_ylabel("Mean user utility (средняя релевантность выбранных докладов)")
    ax.set_title("Trade-off: переполнение vs полезность для пользователя")
    # стрелка к утопии
    ax.annotate("идеальная политика",
                xy=(0.0, max(u for u, _ in util_agg.values()) + 0.005),
                xytext=(0.18, max(u for u, _ in util_agg.values()) + 0.02),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray", style="italic")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_hall_load_heatmap(conf, results_dict, fname):
    """Heatmap: средняя загрузка (occ/cap) залов по слотам для двух политик."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(7 * len(results_dict), 4.5),
                             sharey=True)
    if len(results_dict) == 1:
        axes = [axes]
    halls_sorted = sorted(conf.halls.keys())
    slots_with_data = [s for s in conf.slots if s.talk_ids]
    slot_labels = [s.datetime[-8:-3] for s in slots_with_data]  # HH:MM

    vmax = 1.5
    for ax, (pname, sim) in zip(axes, results_dict.items()):
        mat = np.zeros((len(halls_sorted), len(slots_with_data)))
        for j, slot in enumerate(slots_with_data):
            for i, hid in enumerate(halls_sorted):
                cap = conf.halls[hid].capacity
                occ = sim.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
                # доклад вообще есть в этом зале/слоте?
                slot_halls = {conf.talks[tid].hall for tid in slot.talk_ids}
                if hid not in slot_halls:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = occ / max(1.0, cap)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(slot_labels)))
        ax.set_xticklabels(slot_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(halls_sorted)))
        ax.set_yticklabels([f"Hall {h} (cap={conf.halls[h].capacity})" for h in halls_sorted])
        ax.set_title(pname)
        # рисуем граничную линию вместимости
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Загрузка (occupied / capacity)")
    fig.suptitle("Загрузка залов по слотам: красное — переполнено")
    fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_simulation_for_heatmaps(conf, users):
    """Запускает Cosine и Capacity-aware MMR на одном сиде для сравнения heat maps."""
    cfg = SimConfig(K=2, tau=0.3, lambda_overflow=2.0, p_skip_base=0.05, seed=42)
    cosine = simulate(conf, users, CosinePolicy(), cfg)
    cmmr = simulate(conf, users, CapacityAwareMMRPolicy(beta=0.6, alpha=0.4), cfg)
    return {"Cosine": cosine, "Capacity-aware MMR": cmmr}


def main():
    results = load_results()
    print(f"Loaded {len(results['runs'])} run records")

    plot_bar("overflow_rate", "Доля переполненных (slot, hall) пар",
             "overflow_rate", results, "01_overflow.png", lower_is_better=True)
    plot_bar("hall_utilization_variance", "Дисперсия загрузки залов внутри слота",
             "variance", results, "02_variance.png", lower_is_better=True)
    plot_bar("mean_user_utility", "Средняя релевантность выбранных докладов",
             "mean_user_utility", results, "03_utility.png", lower_is_better=False)
    plot_bar("hall_load_gini", "Коэффициент Джини суммарной загрузки залов",
             "Gini", results, "04_gini.png", lower_is_better=True)

    plot_tradeoff(results, "05_tradeoff.png")

    # heat maps для двух контрастных политик
    print("Loading data for heatmaps...")
    conf = Conference.load(
        ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
        ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz",
    )
    personas_name = "personas_x3"  # тот же датасет, что в run_experiments.py
    with open(ROOT / "data" / "personas" / f"{personas_name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(ROOT / "data" / "personas" / f"{personas_name}_embeddings.npz", allow_pickle=False)
    by_id = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    def text_of(p):
        return p.get("background") or p.get("profile") or ""
    users = [
        UserProfile(id=p["id"], text=text_of(p), embedding=by_id[p["id"]])
        for p in meta
    ]
    sims = make_simulation_for_heatmaps(conf, users)
    plot_hall_load_heatmap(conf, sims, "06_hall_load_heatmap.png")

    # копируем в thesis/figures для будущего использования
    if THESIS_FIG_DIR.exists():
        import shutil
        for f in PLOT_DIR.glob("*.png"):
            shutil.copy(f, THESIS_FIG_DIR / f.name)
        print(f"Copied to {THESIS_FIG_DIR}")

    files = sorted(PLOT_DIR.glob("*.png"))
    print(f"\nPlots ({len(files)}):")
    for f in files:
        print(f"  {f.relative_to(ROOT)} ({f.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
