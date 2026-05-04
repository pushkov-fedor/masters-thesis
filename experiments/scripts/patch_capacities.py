"""Унифицированный патч вместимостей залов для конференций.

Применяет controlled-схему вместимостей: для каждого слота
hall_capacities[h] = ceil(N / halls_in_slot), глобальная Hall.capacity = N.

Поддерживает несколько источников по ключу --conf:
    mobius      → data/conferences/mobius_2025_autumn.json
    demo_day    → data/conferences/demo_day_2026.json (с нормализацией времени)

Запуск:
    python scripts/patch_capacities.py --conf mobius   --population 1200
    python scripts/patch_capacities.py --conf demo_day --population 900
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONF_REGISTRY = {
    "mobius":   {"path": ROOT / "data" / "conferences" / "mobius_2025_autumn.json",
                 "normalize_time": False},
    "demo_day": {"path": ROOT / "data" / "conferences" / "demo_day_2026.json",
                 "normalize_time": True},
}


def normalize_time(s):
    if not s:
        return s
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, choices=sorted(CONF_REGISTRY.keys()))
    ap.add_argument("--population", type=int, required=True,
                    help="N — целевой размер популяции; cap_per_hall_in_slot = ceil(N / halls_in_slot)")
    args = ap.parse_args()
    cfg = CONF_REGISTRY[args.conf]
    json_path = cfg["path"]
    N = args.population

    with open(json_path, encoding="utf-8") as f:
        prog = json.load(f)

    talks = prog["talks"]

    if cfg["normalize_time"]:
        for t in talks:
            t["start_time"] = normalize_time(t.get("start_time"))
            t["end_time"] = normalize_time(t.get("end_time"))

        # Перестраиваем slot_id по (date, start_time), чтобы убрать дубли вида
        # 10:30 / 10:30:00 как два разных слота.
        slot_keys_ordered = []
        seen = set()
        for t in talks:
            key = f"{t['date']} {t['start_time']}"
            if key not in seen:
                slot_keys_ordered.append(key)
                seen.add(key)
        slot_keys_ordered.sort()
        slot_id_by_key = {k: f"slot_{i:02d}" for i, k in enumerate(slot_keys_ordered)}
        for t in talks:
            key = f"{t['date']} {t['start_time']}"
            t["slot_id"] = slot_id_by_key[key]
    else:
        # У конференций с уже корректными slot_id (Mobius) сохраняем порядок и
        # datetime из исходного prog["slots"] (если он был).
        existing_slots = prog.get("slots") or []
        slot_keys_ordered = []
        slot_id_by_key = {}
        if existing_slots:
            for s in existing_slots:
                k = s.get("datetime") or s["id"]
                slot_keys_ordered.append(k)
                slot_id_by_key[k] = s["id"]
        else:
            seen = set()
            for t in talks:
                k = f"{t.get('date','')} {t.get('start_time','')}".strip()
                if k not in seen:
                    slot_keys_ordered.append(k)
                    slot_id_by_key[k] = t["slot_id"]
                    seen.add(k)

    # Группируем доклады по слотам.
    talks_by_slot = defaultdict(list)
    for t in talks:
        talks_by_slot[t["slot_id"]].append(t)

    # Per-slot per-hall capacity.
    slots_meta = []
    for k in slot_keys_ordered:
        sid = slot_id_by_key[k]
        halls_in_slot = sorted({t["hall"] for t in talks_by_slot[sid]})
        cap = math.ceil(N / max(1, len(halls_in_slot)))
        slots_meta.append({
            "id": sid,
            "datetime": k,
            "hall_capacities": {str(h): cap for h in halls_in_slot},
        })

    halls = sorted({t["hall"] for t in talks})
    halls_meta = [{"id": h, "capacity": N} for h in halls]

    new_prog = {
        "conf_id": prog.get("conf_id"),
        "name": prog.get("name"),
        "date": prog.get("date"),
        "population_for_capacity": N,
        "talks": talks,
        "halls": halls_meta,
        "slots": slots_meta,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_prog, f, ensure_ascii=False, indent=2)

    print(f"WROTE: {json_path}")
    print(f"  talks: {len(talks)}, halls: {len(halls)}, slots: {len(slots_meta)}, N={N}")
    print()
    print("Распределение по числу залов в слоте:")
    dist = defaultdict(int)
    for s in slots_meta:
        dist[len(s["hall_capacities"])] += 1
    for n_halls in sorted(dist):
        cap = math.ceil(N / max(1, n_halls))
        print(f"  {n_halls} залов: {dist[n_halls]} слотов, cap/зал = {cap}")


if __name__ == "__main__":
    main()
