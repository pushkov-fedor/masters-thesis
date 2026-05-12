"""Инспекция топ-N худших перестановок из phi_isolated прогона.

Для каждой худшей перестановки:
1. Показывает swap descriptor (title, category, hall обоих свапаемых talks).
2. Печатает состав слотов до и после свапа (какие talks в каких залах).
3. Прогоняет дополнительные replicate (по умолчанию 10) на режиме
   cap=0.7, gossip=0 — чтобы убедиться, что эффект устойчивый.
4. Раскладывает суммарную перегрузку по слотам и залам — видно, где
   именно сконцентрирована проблема.
5. Сравнивает с baseline P_0 на тех же seed'ах.

Запуск:
    .venv/bin/python scripts/inspect_top_phi_swaps.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import (  # noqa: E402
    load_conference, scale_capacity, select_audience,
)
from src.policies.registry import active_policies  # noqa: E402
from src.program_modification import enumerate_modifications  # noqa: E402
from src.simulator import SimConfig, simulate  # noqa: E402


CONFERENCE = "mobius_2025_autumn_en"
AUDIENCE_SIZE = 100
AUDIENCE_SEED = 0
PHI_SEED = 17
PHI_KMAX = 500
CAPACITY = 0.7
GOSSIP = 0.0
N_REPLICATES = 10
TAU = 0.7
P_SKIP_BASE = 0.10
K = 3
TOP_N = 3

ISOLATED_JSON = ROOT / "results" / "phi_isolated_2026-05-12.json"
OUT_MD = ROOT / "results" / "phi_isolated_top_swaps_2026-05-12.md"


def slot_load_breakdown(conf, result) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Возвращает {slot_id: {hall_id: (occupied, capacity)}}."""
    occ: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for step in result.steps:
        if step.chosen is None:
            continue
        talk = conf.talks[step.chosen]
        occ[talk.slot_id][talk.hall] += 1
    out: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for slot in conf.slots:
        per_hall: Dict[str, Tuple[int, int]] = {}
        for tid in slot.talk_ids:
            talk = conf.talks[tid]
            cap = conf.halls[talk.hall].capacity
            if slot.hall_capacities and talk.hall in slot.hall_capacities:
                cap = slot.hall_capacities[talk.hall]
            per_hall[talk.hall] = (occ[slot.id].get(talk.hall, 0), cap)
        out[slot.id] = per_hall
    return out


def overload_per_slot(load: Dict[str, Dict[str, Tuple[int, int]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for slot_id, halls in load.items():
        if len(halls) < 2:
            continue
        worst = 0.0
        for occ_val, cap_val in halls.values():
            ex = max(0.0, (occ_val - cap_val) / cap_val) if cap_val > 0 else 0.0
            worst = max(worst, ex)
        out[slot_id] = worst
    return out


def main():
    print(f"loading {CONFERENCE}...", flush=True)
    base_conf, all_users = load_conference(CONFERENCE)
    audience = select_audience(all_users, AUDIENCE_SIZE, AUDIENCE_SEED)
    cap_conf = scale_capacity(base_conf, CAPACITY)

    rng = np.random.default_rng(PHI_SEED)
    mods = enumerate_modifications(cap_conf, k_max=PHI_KMAX, rng=rng, same_day_only=True)
    print(f"  {len(mods)} valid swaps", flush=True)

    iso = json.loads(ISOLATED_JSON.read_text())
    # ищем худшие PV по среднему overload на ячейке cap=0.7, gossip=0
    target_cell = f"cap={CAPACITY}_gossip={GOSSIP}"
    pv_means: Dict[str, float] = iso["pv_means_by_cell"][target_cell]
    # JSON ключи — строки, приведём к int
    pv_means_int = {int(k): v for k, v in pv_means.items()}
    sorted_pvs = sorted(pv_means_int.items(), key=lambda kv: -kv[1])
    top_pvs = [pv for pv, _ in sorted_pvs[:TOP_N]]

    pol = active_policies(include_llm=False)["no_policy"]

    lines: List[str] = []
    lines.append(f"# Top-{TOP_N} worst Φ swaps — {CONFERENCE}")
    lines.append("")
    lines.append(
        f"Базовый прогон phi_isolated_2026-05-12.json. Для топ-{TOP_N} худших "
        f"перестановок (по среднему overload на cap={CAPACITY}, gossip={GOSSIP}) "
        f"проведена расширенная инспекция: {N_REPLICATES} replicate + разбор "
        f"состава задействованных слотов + per-hall разложение перегрузки."
    )
    lines.append("")
    lines.append(f"Конференция: {len(base_conf.talks)} talks, {len(base_conf.halls)} halls "
                 f"(номинальная capacity {next(iter(base_conf.halls.values())).capacity}), "
                 f"{len(base_conf.slots)} slots. "
                 f"Capacity multiplier = {CAPACITY} → "
                 f"эффективные размеры {[int(h.capacity * CAPACITY) for h in base_conf.halls.values()]}.")
    lines.append("")

    # baseline P_0 — прогоняем с теми же seed'ами для сопоставимости
    print("\n[baseline] running P_0 with N_REPLICATES seeds...", flush=True)
    base_overloads = []
    base_slot_overload_acc: Dict[str, List[float]] = defaultdict(list)
    for rep in range(1, N_REPLICATES + 1):
        cfg = SimConfig(
            tau=TAU, p_skip_base=P_SKIP_BASE, K=K,
            seed=rep, w_rel=1.0 - GOSSIP, w_rec=0.0, w_gossip=GOSSIP, w_fame=0.0,
        )
        res = simulate(cap_conf, audience, pol, cfg)
        from src.metrics import mean_hall_overload_excess
        base_overloads.append(float(mean_hall_overload_excess(cap_conf, res)))
        slot_ov = overload_per_slot(slot_load_breakdown(cap_conf, res))
        for sid, v in slot_ov.items():
            base_slot_overload_acc[sid].append(v)
    base_mean = float(np.mean(base_overloads))
    base_std = float(np.std(base_overloads, ddof=0))
    base_slot_overload = {sid: float(np.mean(vals)) for sid, vals in base_slot_overload_acc.items()}
    print(f"  P_0: mean overload = {base_mean:.4f} ± {base_std:.4f} (n={N_REPLICATES})", flush=True)

    lines.append(f"## Baseline (P_0)")
    lines.append("")
    lines.append(f"`mean_overload_excess` over {N_REPLICATES} seeds: **{base_mean:.4f} ± {base_std:.4f}**")
    lines.append("")
    lines.append("Top-5 наиболее загруженных слотов в исходной программе:")
    lines.append("")
    lines.append("| slot_id | mean per-slot overload |")
    lines.append("|---|---:|")
    for sid, v in sorted(base_slot_overload.items(), key=lambda kv: -kv[1])[:5]:
        lines.append(f"| {sid} | {v:.4f} |")
    lines.append("")

    for rank, pv in enumerate(top_pvs, start=1):
        print(f"\n[swap rank {rank}] PV={pv}, expected mean overload = {pv_means_int[pv]:.4f}", flush=True)
        modified_conf, desc = mods[pv - 1]

        t1 = base_conf.talks[desc.t1]
        t2 = base_conf.talks[desc.t2]
        slot_a_id = desc.slot_a
        slot_b_id = desc.slot_b

        lines.append(f"## Rank {rank}: PV={pv}")
        lines.append("")
        lines.append(f"Mean overload в основном прогоне: **{pv_means_int[pv]:.4f}** (vs P_0 = {base_mean:.4f}, "
                     f"то есть **+{pv_means_int[pv] - base_mean:.4f}** = "
                     f"{((pv_means_int[pv] - base_mean) / base_mean * 100):.0f}% хуже).")
        lines.append("")
        lines.append("### Что меняется")
        lines.append("")
        lines.append(f"**Talk 1** (`{desc.t1[:8]}…`): «{t1.title}»")
        lines.append(f"  — category: {t1.category}; hall: {t1.hall}")
        lines.append(f"  — был в slot **{slot_a_id}**, переезжает в slot **{slot_b_id}**")
        lines.append("")
        lines.append(f"**Talk 2** (`{desc.t2[:8]}…`): «{t2.title}»")
        lines.append(f"  — category: {t2.category}; hall: {t2.hall}")
        lines.append(f"  — был в slot **{slot_b_id}**, переезжает в slot **{slot_a_id}**")
        lines.append("")

        # before/after composition of affected slots
        def slot_composition(conf, slot_id):
            slot = next((s for s in conf.slots if s.id == slot_id), None)
            if slot is None:
                return []
            return [(tid, conf.talks[tid]) for tid in slot.talk_ids]

        lines.append("### Состав задействованных слотов")
        lines.append("")
        for slot_id in [slot_a_id, slot_b_id]:
            lines.append(f"**Slot `{slot_id}` — до свапа:**")
            for tid, talk in slot_composition(cap_conf, slot_id):
                marker = "  ←" if tid in (desc.t1, desc.t2) else "    "
                lines.append(f"- {marker} `{tid[:8]}…` hall={talk.hall}: {talk.title[:90]} ({talk.category})")
            lines.append("")
            lines.append(f"**Slot `{slot_id}` — после свапа:**")
            for tid, talk in slot_composition(modified_conf, slot_id):
                marker = "  ←" if tid in (desc.t1, desc.t2) else "    "
                lines.append(f"- {marker} `{tid[:8]}…` hall={talk.hall}: {talk.title[:90]} ({talk.category})")
            lines.append("")

        # extended replicates
        print(f"  running {N_REPLICATES} replicates on modified program...", flush=True)
        ovs = []
        slot_overload_acc: Dict[str, List[float]] = defaultdict(list)
        load_acc: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        for rep in range(1, N_REPLICATES + 1):
            cfg = SimConfig(
                tau=TAU, p_skip_base=P_SKIP_BASE, K=K,
                seed=rep, w_rel=1.0 - GOSSIP, w_rec=0.0, w_gossip=GOSSIP, w_fame=0.0,
            )
            res = simulate(modified_conf, audience, pol, cfg)
            from src.metrics import mean_hall_overload_excess
            ovs.append(float(mean_hall_overload_excess(modified_conf, res)))
            ld = slot_load_breakdown(modified_conf, res)
            so = overload_per_slot(ld)
            for sid, v in so.items():
                slot_overload_acc[sid].append(v)
            for sid, halls in ld.items():
                for hall_id, (occ_v, cap_v) in halls.items():
                    load_acc[sid][hall_id].append((occ_v, cap_v))

        mean_ov = float(np.mean(ovs))
        std_ov = float(np.std(ovs, ddof=0))
        print(f"  modified: mean overload = {mean_ov:.4f} ± {std_ov:.4f} (n={N_REPLICATES})", flush=True)

        lines.append(f"### Расширенная проверка ({N_REPLICATES} seeds)")
        lines.append("")
        lines.append(f"`mean_overload_excess` (modified): **{mean_ov:.4f} ± {std_ov:.4f}**")
        lines.append(f"`mean_overload_excess` (baseline P_0): {base_mean:.4f} ± {base_std:.4f}")
        lines.append(f"  → разница **+{mean_ov - base_mean:.4f}**, "
                     f"std эффекта оценивается как ~√(σ²_mod + σ²_base) = ~{(std_ov ** 2 + base_std ** 2) ** 0.5:.4f}.")
        lines.append("")

        lines.append("### Изменение перегрузки по слотам (modified − baseline)")
        lines.append("")
        lines.append("| slot_id | baseline overload | modified overload | Δ |")
        lines.append("|---|---:|---:|---:|")
        modified_slot_overload = {sid: float(np.mean(vs)) for sid, vs in slot_overload_acc.items()}
        all_slots = set(modified_slot_overload) | set(base_slot_overload)
        rows = []
        for sid in all_slots:
            b = base_slot_overload.get(sid, 0.0)
            m = modified_slot_overload.get(sid, 0.0)
            rows.append((sid, b, m, m - b))
        rows.sort(key=lambda r: -abs(r[3]))
        for sid, b, m, d in rows[:6]:
            marker = " **(swap)**" if sid in (slot_a_id, slot_b_id) else ""
            lines.append(f"| `{sid}`{marker} | {b:.4f} | {m:.4f} | {d:+.4f} |")
        lines.append("")

        # per-hall occupancy in the two affected slots after the swap
        lines.append("### Заполненность залов в задействованных слотах (после свапа, mean over seeds)")
        lines.append("")
        lines.append("| slot | hall | capacity | occupied | overload |")
        lines.append("|---|---|---:|---:|---:|")
        for sid in [slot_a_id, slot_b_id]:
            for hid, vals in load_acc[sid].items():
                occ_mean = float(np.mean([v[0] for v in vals]))
                cap = vals[0][1]
                ex = max(0.0, (occ_mean - cap) / cap) if cap > 0 else 0.0
                lines.append(f"| `{sid}` | {hid} | {cap} | {occ_mean:.1f} | {ex:+.3f} |")
        lines.append("")

        # find the slot that contributes most to the worsening
        worst_slot = max(rows, key=lambda r: r[3])
        lines.append(f"### Объяснение")
        lines.append("")
        lines.append(
            f"Главный вклад в ухудшение даёт slot `{worst_slot[0]}` — перегрузка "
            f"выросла с {worst_slot[1]:.3f} до {worst_slot[2]:.3f} (+{worst_slot[3]:.3f}). "
        )
        if worst_slot[0] in (slot_a_id, slot_b_id):
            lines.append(
                "Это **один из двух слотов, где произошёл свап** — то есть участники, ранее "
                "распределявшиеся между докладами в этом слоте более ровно, после свапа "
                "стянулись в один зал."
            )
        else:
            lines.append(
                "Этот слот **не задействован свапом напрямую**, но его перегрузка изменилась "
                "из-за того, что часть аудитории перераспределилась по сцепным эффектам "
                "(одни персоны, которые шли на свапнутые доклады, теперь идут в другие слоты)."
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
