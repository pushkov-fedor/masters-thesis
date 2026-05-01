"""Одноразовый патч demo_day_2026.json:

1. Нормализует start_time/end_time докладов (10:30 → 10:30:00), чтобы исчезли
   дубли слотов с одинаковым фактическим временем (баг исходного парсера).
2. Перестраивает slot-список после объединения дубликатов.
3. Добавляет на каждый слот hall_capacities = {hall_id: ceil(N / halls_in_slot)}.
4. Глобальная Hall.capacity заменяется на N (нерестриктивный fallback).

Запуск:
    python scripts/patch_demo_day_capacities.py --population 900

Источник Excel недоступен, поэтому патч идёт поверх готового JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "conferences" / "demo_day_2026.json"


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
    ap.add_argument("--population", type=int, default=900)
    args = ap.parse_args()
    N = args.population

    with open(JSON_PATH, encoding="utf-8") as f:
        prog = json.load(f)

    talks = prog["talks"]
    # Нормализуем времена и собираем (date, start_time) → talk_ids.
    for t in talks:
        t["start_time"] = normalize_time(t.get("start_time"))
        t["end_time"] = normalize_time(t.get("end_time"))

    # Перестраиваем слоты на основе (date, start_time)
    slot_keys_ordered = []
    seen = set()
    for t in talks:
        key = f"{t['date']} {t['start_time']}"
        if key not in seen:
            slot_keys_ordered.append(key)
            seen.add(key)
    slot_keys_ordered.sort()
    slot_id_by_key = {k: f"slot_{i:02d}" for i, k in enumerate(slot_keys_ordered)}

    # Обновляем slot_id у talks
    for t in talks:
        key = f"{t['date']} {t['start_time']}"
        t["slot_id"] = slot_id_by_key[key]

    # Группируем доклады по слотам
    talks_by_slot = defaultdict(list)
    for t in talks:
        talks_by_slot[t["slot_id"]].append(t)

    # Per-slot per-hall capacity
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
        "conf_id": prog.get("conf_id", "demo_day_2026"),
        "name": prog.get("name", "Demo Day ITMO 2026"),
        "date": prog.get("date", "2026-01-22"),
        "population_for_capacity": N,
        "talks": talks,
        "halls": halls_meta,
        "slots": slots_meta,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_prog, f, ensure_ascii=False, indent=2)

    print(f"WROTE: {JSON_PATH}")
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
