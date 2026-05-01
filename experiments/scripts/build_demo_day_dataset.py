"""Парсит Demo Day Topics.xlsx -> нормализованный JSON программы конференции.

Demo Day — студенческая конференция ITMO, январь 2026, 210 докладов, 2 дня.

Постановка эксперимента: вместимости залов задаются на уровне СЛОТА по правилу
controlled experiment — суммарная вместимость каждого тайм-слота равна
численности популяции участников, распределённой равномерно между залами слота.
Это устраняет произвол в выборе вместимостей (нет «оценок по фото») и сводит
переполнение к чистому эффекту поведения политики, а не неоднородности
предложенной вместимости.

Per-slot capacity[hall] = ceil(N_population / halls_in_slot).
- В keynote-слоте (1 зал): cap = N_population → все умещаются по построению.
- В 6-зальном слоте: cap = N/6 на зал → суммарно слот вмещает ровно N.
- Переполнение возникает только если политика концентрирует аудиторию.

Глобальная Hall.capacity сохраняется как нерестриктивный fallback (= N).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "Demo Day Topics.xlsx"
OUT_PROG = ROOT / "data" / "conferences" / "demo_day_2026.json"


def _normalize_time(s):
    """Приводит '10:30' → '10:30:00'. Без этого слоты с одинаковым
    фактическим временем разводятся на разные slot_id (баг исходного
    парсера — slot_00 '10:30' и slot_01 '10:30:00' были разными)."""
    if not s:
        return s
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=900,
                    help="Численность синтетической популяции участников. "
                         "Используется для расчёта вместимостей залов в слоте: "
                         "cap_per_hall_in_slot = ceil(population / halls_in_slot).")
    args = ap.parse_args()
    N = args.population

    topics = pd.read_excel(SRC, sheet_name="dataset_jug")
    topics = topics[topics["conf"] == "Demo Day"].reset_index(drop=True)

    talks = []
    for _, r in topics.iterrows():
        if pd.isna(r["title"]) or pd.isna(r["start_time"]):
            continue
        talks.append({
            "id": str(r["id"]),
            "title": str(r["title"]).strip(),
            "speakers": str(r["speakers"]).strip() if pd.notna(r["speakers"]) else "",
            "hall": int(r["hall"]) if pd.notna(r["hall"]) else 1,
            "date": str(r["date"])[:10] if pd.notna(r["date"]) else None,
            "start_time": _normalize_time(r["start_time"]) if pd.notna(r["start_time"]) else None,
            "end_time": _normalize_time(r["end_time"]) if pd.notna(r["end_time"]) else None,
            "category": str(r["category"]) if pd.notna(r["category"]) else "Other",
            "abstract": str(r["abstract"]).strip() if pd.notna(r["abstract"]) else "",
        })

    slot_index = defaultdict(list)
    for t in talks:
        key = f"{t['date']} {t['start_time']}"
        slot_index[key].append(t["id"])

    slots = sorted(slot_index.keys())
    slot_id_by_key = {k: f"slot_{i:02d}" for i, k in enumerate(slots)}
    for t in talks:
        key = f"{t['date']} {t['start_time']}"
        t["slot_id"] = slot_id_by_key[key]

    halls = sorted({t["hall"] for t in talks if t["hall"] is not None})
    # Глобальная вместимость = N (нерестриктивный fallback на случай отсутствия
    # переопределения; per-slot значения покрывают все используемые слоты).
    halls_meta = [{"id": h, "capacity": N} for h in halls]

    # Per-slot per-hall capacity = ceil(N / halls_in_slot)
    slot_meta = []
    for k in slots:
        sid = slot_id_by_key[k]
        halls_in_slot = sorted({
            next(t["hall"] for t in talks if t["id"] == tid)
            for tid in slot_index[k]
        })
        cap = math.ceil(N / max(1, len(halls_in_slot)))
        slot_meta.append({
            "id": sid,
            "datetime": k,
            "hall_capacities": {str(h): cap for h in halls_in_slot},
        })

    program = {
        "conf_id": "demo_day_2026",
        "name": "Demo Day ITMO 2026",
        "date": "2026-01-22",
        "population_for_capacity": N,
        "talks": talks,
        "halls": halls_meta,
        "slots": slot_meta,
    }

    OUT_PROG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PROG, "w", encoding="utf-8") as f:
        json.dump(program, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(talks)} talks, {len(halls)} halls, {len(slots)} slots, N={N}")
    print(f"WROTE: {OUT_PROG}")

    print("\nРаспределение слотов по числу параллельных залов и вместимости/зал:")
    parallel_dist = defaultdict(int)
    for k in slots:
        n_halls = len({next(t["hall"] for t in talks if t["id"] == tid)
                       for tid in slot_index[k]})
        parallel_dist[n_halls] += 1
    for n_halls, count in sorted(parallel_dist.items()):
        cap = math.ceil(N / max(1, n_halls))
        print(f"  {n_halls} залов в слоте: {count} слотов, cap/зал = {cap}")

    print("\nКатегории (топ-15):")
    cats = defaultdict(int)
    for t in talks:
        cats[t["category"]] += 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
