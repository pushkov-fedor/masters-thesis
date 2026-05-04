"""Картинка для защиты: hall loads по политикам на LLM-агентском симуляторе.

Слева — без политики (свободный выбор), залы переполнены неравномерно.
Справа — с capacity-aware MMR, залы выровнены.

Запуск:
    python scripts/plot_llm_agents_results.py --result llm_agents_mobius_2025_autumn_n50.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="llm_agents_mobius_2025_autumn_n50.json")
    ap.add_argument("--out", default=None,
                    help="имя выходного файла png без пути")
    args = ap.parse_args()

    res_path = ROOT / "results" / args.result
    with open(res_path) as f:
        data = json.load(f)
    cfg = data["config"]
    print(f"Loaded: {res_path.name}, n={cfg['n_agents']}, conf={cfg['conference']}")

    conf_path = ROOT / "data" / "conferences" / f"{cfg['conference']}.json"
    with open(conf_path) as f:
        conf = json.load(f)
    talks_by_slot = defaultdict(list)
    for t in conf["talks"]:
        talks_by_slot[t["slot_id"]].append(t)
    multi_slots = [s for s in conf["slots"]
                   if len({t["hall"] for t in talks_by_slot[s["id"]]}) >= 2]
    print(f"Multi-hall slots: {len(multi_slots)} of {len(conf['slots'])}")

    pol_results = data["results"]
    pol_names = list(pol_results.keys())
    print(f"Policies: {pol_names}")

    # === FIGURE 1: barchart распределения по залам в одном горячем слоте ===
    # Найдём slot с самой большой неравномерностью у no_policy
    no_pol_loads = pol_results.get("no_policy", {}).get("slot_loads", {})
    if no_pol_loads:
        worst_slot = None
        worst_imbalance = 0
        for sid, loads in no_pol_loads.items():
            slot_meta = next((s for s in conf["slots"] if s["id"] == sid), None)
            if not slot_meta or len(loads) < 2:
                continue
            cap = slot_meta.get("hall_capacities", {})
            halls = sorted(loads.keys(), key=int)
            utils = [loads[h] / int(cap.get(str(h), cap.get(h, 1000))) for h in halls]
            imb = max(utils) - min(utils)
            if imb > worst_imbalance:
                worst_imbalance = imb
                worst_slot = sid
        print(f"Worst-imbalance slot: {worst_slot}, gap={worst_imbalance:.2f}")
    else:
        worst_slot = None

    if worst_slot:
        fig, axes = plt.subplots(1, len(pol_names), figsize=(5 * len(pol_names), 4),
                                 sharey=True)
        if len(pol_names) == 1:
            axes = [axes]
        for ax, pol in zip(axes, pol_names):
            loads = pol_results[pol]["slot_loads"].get(worst_slot, {})
            slot_meta = next(s for s in conf["slots"] if s["id"] == worst_slot)
            cap_dict = slot_meta.get("hall_capacities", {})
            halls = sorted(loads.keys(), key=int)
            counts = [loads[h] for h in halls]
            caps = [int(cap_dict.get(str(h), cap_dict.get(h, 1000))) for h in halls]
            x = np.arange(len(halls))
            colors = ["#d62728" if c > cap else "#2ca02c" for c, cap in zip(counts, caps)]
            ax.bar(x, counts, color=colors, edgecolor="black")
            for i, (c, cap) in enumerate(zip(counts, caps)):
                ax.axhline(y=cap, xmin=(i / len(halls)) + 0.05,
                           xmax=((i + 1) / len(halls)) - 0.05,
                           color="black", linestyle="--", linewidth=1.5)
                ax.text(i, c + 0.5, f"{c}", ha="center", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Зал {h}" for h in halls])
            ax.set_title(f"{pol}\n(slot={worst_slot})", fontsize=12)
            ax.set_ylabel("Число агентов в зале") if pol == pol_names[0] else None
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Распределение агентов по залам в слоте {worst_slot} "
                     f"(LLM-симулятор, N={cfg['n_agents']}, {cfg['conference']})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        out1 = args.out or f"llm_agents_{cfg['conference']}_n{cfg['n_agents']}_slot_breakdown.png"
        out_path1 = ROOT / "results" / "plots" / out1
        out_path1.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"WROTE: {out_path1}")

    # === FIGURE 2: hall_load distribution across all multi-slots, per policy ===
    fig, axes = plt.subplots(1, len(pol_names), figsize=(5 * len(pol_names), 4),
                             sharey=True)
    if len(pol_names) == 1:
        axes = [axes]
    for ax, pol in zip(axes, pol_names):
        utilizations = []
        for slot in multi_slots:
            sid = slot["id"]
            loads = pol_results[pol]["slot_loads"].get(sid, {})
            cap_dict = slot.get("hall_capacities", {})
            for h, c in loads.items():
                cap = int(cap_dict.get(str(h), cap_dict.get(h, 1000)))
                utilizations.append(c / cap)
        ax.hist(utilizations, bins=np.arange(0, 2.0, 0.1), color="#1f77b4",
                edgecolor="black", alpha=0.8)
        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="capacity")
        m = np.mean(utilizations) if utilizations else 0
        v = np.var(utilizations) if utilizations else 0
        ax.set_title(f"{pol}\nmean={m:.2f}, var={v:.3f}", fontsize=12)
        ax.set_xlabel("Заполненность зала (load / capacity)")
        ax.set_ylabel("Количество (зал × слот)") if pol == pol_names[0] else None
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Распределение заполненности залов по всем слотам "
                 f"(LLM-симулятор, N={cfg['n_agents']}, {cfg['conference']})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out2 = f"llm_agents_{cfg['conference']}_n{cfg['n_agents']}_hall_dist.png"
    out_path2 = ROOT / "results" / "plots" / out2
    fig.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out_path2}")

    # === FIGURE 3: aggregate metrics bar chart ===
    metrics_keys = ["OF_choice", "hall_var_mean", "mean_overload_excess"]
    metrics_labels = ["OF_choice\n(доля выборов\nв переполн. залы)",
                      "Hall variance\n(дисперсия\nзагрузки)",
                      "Mean overload excess\n(средний избыток\nзагрузки)"]
    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(5 * len(metrics_keys), 4))
    pol_colors = {"no_policy": "#d62728", "cosine": "#ff7f0e", "cap_aware_mmr": "#2ca02c"}
    for ax, mkey, mlabel in zip(axes, metrics_keys, metrics_labels):
        values = [pol_results[p]["metrics"][mkey] for p in pol_names]
        colors = [pol_colors.get(p, "#1f77b4") for p in pol_names]
        bars = ax.bar(pol_names, values, color=colors, edgecolor="black")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                    f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(mlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    fig.suptitle(f"Метрики на LLM-агентском симуляторе "
                 f"(N={cfg['n_agents']}, {cfg['conference']})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out3 = f"llm_agents_{cfg['conference']}_n{cfg['n_agents']}_metrics.png"
    out_path3 = ROOT / "results" / "plots" / out3
    fig.savefig(out_path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out_path3}")


if __name__ == "__main__":
    main()
