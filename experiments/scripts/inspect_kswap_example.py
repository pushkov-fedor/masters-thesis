"""Инспекция одного конкретного варианта k-swap эксперимента.

Находит вариант на cap=1.0, gossip=0, где исходная программа имела
малую перегрузку, а после k свапов перегрузка существенно выросла.
Показывает (на русском, по исходному mobius_2025_autumn.json):

- какие свапы применялись (что куда переехало);
- per-slot, per-hall заполненность до и после;
- какой слот стал самым проблемным и почему.

Запуск:
    .venv/bin/python scripts/inspect_kswap_example.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lhs_parametric import (  # noqa: E402
    load_conference, scale_capacity, select_audience,
)
from scripts.run_phi_isolated import apply_full_swap, has_hall_conflict  # noqa: E402
from scripts.run_phi_isolated_kswap import make_kswap_variant  # noqa: E402
from src.policies.registry import active_policies  # noqa: E402
from src.program_modification import (  # noqa: E402
    _enumerate_all_pairs, has_speaker_conflict,
)
from src.simulator import SimConfig, simulate  # noqa: E402


CONFERENCE = "mobius_2025_autumn_en"
RU_CONF_JSON = ROOT / "data" / "conferences" / "mobius_2025_autumn.json"
AUDIENCE_SIZE = 100
AUDIENCE_SEED = 0
MASTER_SEED = 20260512
N_REPLICATES = 10
CAPACITY = 1.0
GOSSIP = 0.0
TAU = 0.7
P_SKIP_BASE = 0.10
K_TOPK = 3
K_MIN = 3
K_MAX = 8


def slot_load_breakdown(conf, result) -> Dict[str, Dict[int, Tuple[int, int]]]:
    occ: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for step in result.steps:
        if step.chosen is None:
            continue
        talk = conf.talks[step.chosen]
        occ[talk.slot_id][talk.hall] += 1
    out: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for slot in conf.slots:
        per_hall: Dict[int, Tuple[int, int]] = {}
        for tid in slot.talk_ids:
            talk = conf.talks[tid]
            cap = conf.halls[talk.hall].capacity
            if slot.hall_capacities and talk.hall in slot.hall_capacities:
                cap = slot.hall_capacities[talk.hall]
            per_hall[talk.hall] = (occ[slot.id].get(talk.hall, 0), cap)
        out[slot.id] = per_hall
    return out


def main():
    print(f"loading {CONFERENCE}...", flush=True)
    base_conf, all_users = load_conference(CONFERENCE)
    audience = select_audience(all_users, AUDIENCE_SIZE, AUDIENCE_SEED)
    cap_conf = scale_capacity(base_conf, CAPACITY)

    # russian titles
    ru = json.loads(RU_CONF_JSON.read_text(encoding="utf-8"))
    ru_titles = {t["id"]: t for t in ru["talks"]}

    # повторим генерацию вариантов с тем же MASTER_SEED, как в основном прогоне
    rng_master = np.random.default_rng(MASTER_SEED)
    all_pairs = _enumerate_all_pairs(base_conf, same_day_only=True)

    variants = []
    n_failed = 0
    while len(variants) < 400:
        k = int(rng_master.integers(K_MIN, K_MAX + 1))
        modified, descs, n_changed = make_kswap_variant(
            base_conf, k, rng_master, all_pairs,
        )
        if modified is None:
            n_failed += 1
            if n_failed > 2000:
                break
            continue
        variants.append((modified, descs, k, n_changed))

    print(f"  reproduced {len(variants)} variants", flush=True)

    # ---- запускаем baseline + все 400 вариантов на cap=1.0, gossip=0,
    # ищем тот, где было ~ ноль перегрузки в исходной, стало ≥0.07 после свапов
    pol = active_policies(include_llm=False)["no_policy"]

    def run_and_metrics(conf, n_seeds=N_REPLICATES):
        ovs = []
        slot_overload_acc: Dict[str, List[float]] = defaultdict(list)
        load_acc: Dict[str, Dict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        for rep in range(1, n_seeds + 1):
            cfg = SimConfig(
                tau=TAU, p_skip_base=P_SKIP_BASE, K=K_TOPK,
                seed=rep, w_rel=1.0 - GOSSIP, w_rec=0.0, w_gossip=GOSSIP, w_fame=0.0,
            )
            res = simulate(conf, audience, pol, cfg)
            from src.metrics import mean_hall_overload_excess
            ovs.append(float(mean_hall_overload_excess(conf, res)))
            ld = slot_load_breakdown(conf, res)
            for sid, halls in ld.items():
                per_slot_overload = max(
                    (max(0.0, (o - c) / c) for o, c in halls.values() if c > 0),
                    default=0.0,
                )
                if len(halls) >= 2:
                    slot_overload_acc[sid].append(per_slot_overload)
                for hid, (o, c) in halls.items():
                    load_acc[sid][hid].append((o, c))
        return {
            "mean_overload": float(np.mean(ovs)),
            "std_overload": float(np.std(ovs, ddof=0)),
            "slot_overload": {sid: float(np.mean(vs)) for sid, vs in slot_overload_acc.items()},
            "load_per_slot_hall": {
                sid: {hid: (float(np.mean([v[0] for v in vals])), vals[0][1])
                      for hid, vals in halls.items()}
                for sid, halls in load_acc.items()
            },
        }

    print(f"\n[baseline] running P_0 on cap={CAPACITY}, gossip={GOSSIP}...", flush=True)
    base_metrics = run_and_metrics(cap_conf)
    print(f"  P_0: mean overload = {base_metrics['mean_overload']:.4f}", flush=True)
    base_slot = base_metrics["slot_overload"]

    print(f"\n[scan] looking for variants with strong overload increase...", flush=True)
    candidates = []
    for pv_idx, (mod_full, descs, k, n_ch) in enumerate(variants):
        mod_cap = scale_capacity(mod_full, CAPACITY)
        ovs = []
        for rep in range(1, 4):
            cfg = SimConfig(
                tau=TAU, p_skip_base=P_SKIP_BASE, K=K_TOPK,
                seed=rep, w_rel=1.0 - GOSSIP, w_rec=0.0, w_gossip=GOSSIP, w_fame=0.0,
            )
            res = simulate(mod_cap, audience, pol, cfg)
            from src.metrics import mean_hall_overload_excess
            ovs.append(float(mean_hall_overload_excess(mod_cap, res)))
        m = float(np.mean(ovs))
        candidates.append((pv_idx + 1, m, descs, k))

    # сортируем по убыванию overload, ищем top-1 worst
    candidates.sort(key=lambda x: -x[1])
    top_pv, top_overload, top_descs, top_k = candidates[0]
    print(f"  worst PV={top_pv}: 3-seed mean overload = {top_overload:.4f} "
          f"(k={top_k} swaps, baseline = {base_metrics['mean_overload']:.4f})", flush=True)

    # детальная инспекция top-1
    mod_full = variants[top_pv - 1][0]
    mod_cap = scale_capacity(mod_full, CAPACITY)
    print(f"\n[detailed] running {N_REPLICATES} replicates on PV={top_pv}...", flush=True)
    mod_metrics = run_and_metrics(mod_cap)

    # ---- markdown отчёт ----
    out = ROOT / "results" / "phi_isolated_kswap_worst_example_2026-05-12.md"
    L: List[str] = []
    L.append(f"# Конкретный пример сильного эффекта k-swap")
    L.append("")
    L.append(f"Режим: capacity_multiplier={CAPACITY}, w_gossip={GOSSIP}, "
             f"audience={AUDIENCE_SIZE} (seed={AUDIENCE_SEED}), N_replicates={N_REPLICATES}.")
    L.append("")
    L.append(f"**Baseline P_0**: mean overload = **{base_metrics['mean_overload']:.4f} ± "
             f"{base_metrics['std_overload']:.4f}**")
    L.append(f"**Modified PV={top_pv}** (k={top_k} swaps): mean overload = "
             f"**{mod_metrics['mean_overload']:.4f} ± {mod_metrics['std_overload']:.4f}**")
    L.append("")
    L.append(f"Разница: **+{mod_metrics['mean_overload'] - base_metrics['mean_overload']:.4f}** "
             f"= в {mod_metrics['mean_overload'] / max(base_metrics['mean_overload'], 1e-4):.1f} раз "
             f"выше baseline.")
    L.append("")

    L.append("## Какие свапы применялись (по-русски)")
    L.append("")
    L.append(f"Всего k={top_k} свапов:")
    L.append("")
    for i, desc in enumerate(top_descs, 1):
        t1_ru = ru_titles.get(desc.t1, {}).get("title", "?")
        t2_ru = ru_titles.get(desc.t2, {}).get("title", "?")
        L.append(f"{i}. «{t1_ru}» (slot **{desc.slot_a}**) ⇄ «{t2_ru}» (slot **{desc.slot_b}**)")
    L.append("")

    L.append("## Какие слоты пострадали (изменение перегрузки)")
    L.append("")
    diffs = []
    for sid in set(base_slot) | set(mod_metrics["slot_overload"]):
        b = base_slot.get(sid, 0.0)
        m = mod_metrics["slot_overload"].get(sid, 0.0)
        diffs.append((sid, b, m, m - b))
    diffs.sort(key=lambda r: -r[3])
    L.append("| slot_id | overload до | overload после | Δ |")
    L.append("|---|---:|---:|---:|")
    for sid, b, m, d in diffs[:5]:
        L.append(f"| `{sid}` | {b:.4f} | {m:.4f} | **{d:+.4f}** |")
    L.append("")

    # для самого пострадавшего слота — показать состав до и после
    worst_slot_id = diffs[0][0]
    L.append(f"## Что в слоте `{worst_slot_id}` до и после свапов")
    L.append("")

    def slot_composition_ru(conf, slot_id):
        slot = next((s for s in conf.slots if s.id == slot_id), None)
        if slot is None:
            return []
        rows = []
        for tid in slot.talk_ids:
            talk = conf.talks[tid]
            ru = ru_titles.get(tid, {})
            rows.append((talk.hall, tid, ru.get("title", "?"), ru.get("category", "?")))
        return sorted(rows)

    L.append(f"**До свапов (исходная программа):**")
    L.append("")
    L.append("| зал | категория | название |")
    L.append("|---:|---|---|")
    for hall, tid, title, cat in slot_composition_ru(cap_conf, worst_slot_id):
        L.append(f"| {hall} | {cat} | {title} |")
    L.append("")

    L.append(f"**После k={top_k} свапов:**")
    L.append("")
    L.append("| зал | категория | название |")
    L.append("|---:|---|---|")
    for hall, tid, title, cat in slot_composition_ru(mod_cap, worst_slot_id):
        L.append(f"| {hall} | {cat} | {title} |")
    L.append("")

    # per-hall занятость в этом слоте
    L.append(f"## Заполненность залов в слоте `{worst_slot_id}` (среднее по {N_REPLICATES} прогонам)")
    L.append("")
    L.append("**До свапов:**")
    L.append("")
    L.append("| зал | вместимость | пришло | перегрузка |")
    L.append("|---:|---:|---:|---:|")
    for hid, (occ, cap) in sorted(base_metrics["load_per_slot_hall"].get(worst_slot_id, {}).items()):
        ex = max(0.0, (occ - cap) / cap) if cap > 0 else 0.0
        L.append(f"| {hid} | {cap} | {occ:.1f} | {ex:+.3f} |")
    L.append("")
    L.append("**После свапов:**")
    L.append("")
    L.append("| зал | вместимость | пришло | перегрузка |")
    L.append("|---:|---:|---:|---:|")
    for hid, (occ, cap) in sorted(mod_metrics["load_per_slot_hall"].get(worst_slot_id, {}).items()):
        ex = max(0.0, (occ - cap) / cap) if cap > 0 else 0.0
        L.append(f"| {hid} | {cap} | {occ:.1f} | {ex:+.3f} |")
    L.append("")

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
