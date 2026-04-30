"""Парсит Demo Day Topics.xlsx -> нормализованный JSON программы конференции.

Формат на выходе совместим с mobius_2025_autumn.json — это позволяет использовать
все существующие политики, симулятор и evaluation pipeline без модификаций.

Demo Day — студенческая конференция ITMO, 2026-01-22, 210 докладов, 1 день.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "Demo Day Topics.xlsx"
OUT_PROG = ROOT / "data" / "conferences" / "demo_day_2026.json"

# Залы Demo Day проходят в ITMO — типичные аудитории университета меньше
# чем конференц-залы JUG. Экспертная оценка вместимостей:
# ITMO Demo Day: студенческая конференция, занимает несколько аудиторий
# главного корпуса. Реалистичные оценки вместимости (по фото залов ИТМО):
HALL_CAPACITIES = {
    1: 400,  # главная аудитория (поток)
    2: 250,
    3: 150,
    4: 120,
    5: 120,
    6: 100,
    8: 80,
}


def main():
    topics = pd.read_excel(SRC, sheet_name="dataset_jug")
    # Demo Day всё, остальные конференции отбрасываем
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
            "start_time": str(r["start_time"]) if pd.notna(r["start_time"]) else None,
            "end_time": str(r["end_time"]) if pd.notna(r["end_time"]) else None,
            "category": str(r["category"]) if pd.notna(r["category"]) else "Other",
            "abstract": str(r["abstract"]).strip() if pd.notna(r["abstract"]) else "",
        })

    # тайм-слоты: группируем по (date, start_time)
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
    halls_meta = [{"id": h, "capacity": HALL_CAPACITIES.get(h, 80)} for h in halls]

    program = {
        "conf_id": "demo_day_2026",
        "name": "Demo Day ITMO 2026",
        "date": "2026-01-22",
        "talks": talks,
        "halls": halls_meta,
        "slots": [{"id": slot_id_by_key[k], "datetime": k} for k in slots],
    }

    OUT_PROG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PROG, "w", encoding="utf-8") as f:
        json.dump(program, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(talks)} talks, {len(halls)} halls, {len(slots)} slots")
    print(f"WROTE: {OUT_PROG}")
    print("\nКатегории (топ-15):")
    cats = defaultdict(int)
    for t in talks:
        cats[t["category"]] += 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c}: {n}")
    print("\nЗалы:")
    by_hall = defaultdict(int)
    for t in talks:
        by_hall[t["hall"]] += 1
    for h, n in sorted(by_hall.items()):
        print(f"  hall {h}: {n} talks (cap={HALL_CAPACITIES.get(h, 80)})")
    print(f"\nСлотов: {len(slots)}")
    parallel_dist = defaultdict(int)
    for k in slots:
        parallel_dist[len(slot_index[k])] += 1
    for n, count in sorted(parallel_dist.items()):
        print(f"  {n} параллельных докладов: {count} слотов")


if __name__ == "__main__":
    main()
