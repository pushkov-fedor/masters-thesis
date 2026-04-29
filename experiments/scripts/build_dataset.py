"""Парсит Mobius_Topics.xlsx -> нормализованный JSON программы конференции.

Выход:
- data/conferences/mobius_2025_autumn.json — программа: {conf_id, talks, halls, slots}
- data/conferences/mobius_2025_autumn_users.json — 10 эталонных профилей + scores (для калибровки)
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "conferences" / "Mobius" / "Mobius_Topics.xlsx"
OUT_PROG = ROOT / "data" / "conferences" / "mobius_2025_autumn.json"
OUT_USERS = ROOT / "data" / "conferences" / "mobius_2025_autumn_users.json"

HALL_CAPACITIES = {
    1: 600,
    2: 350,
    3: 250,
    4: 180,
}

def main():
    topics = pd.read_excel(SRC, sheet_name="mobius_topics")
    users = pd.read_excel(SRC, sheet_name="mobius_users")
    scores = pd.read_excel(SRC, sheet_name="mobius_topics_users")

    talks = []
    for _, r in topics.iterrows():
        talks.append({
            "id": str(r["id"]),
            "title": str(r["title"]).strip(),
            "speakers": str(r["speakers"]).strip() if pd.notna(r["speakers"]) else "",
            "hall": int(r["hall"]) if pd.notna(r["hall"]) else None,
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
    halls_meta = [{"id": h, "capacity": HALL_CAPACITIES.get(h, 200)} for h in halls]

    program = {
        "conf_id": "mobius_2025_autumn",
        "name": "Mobius 2025 Autumn",
        "date": "2025-11-18",
        "talks": talks,
        "halls": halls_meta,
        "slots": [{"id": slot_id_by_key[k], "datetime": k} for k in slots],
    }

    with open(OUT_PROG, "w", encoding="utf-8") as f:
        json.dump(program, f, ensure_ascii=False, indent=2)

    # эталонный мини-датасет (10 пользователей + scores)
    user_profiles = [
        {"id": str(r["id"]), "profile": str(r["profile"]).strip()}
        for _, r in users.iterrows()
    ]
    score_records = [
        {
            "user_id": str(r["user_id"]),
            "topic_id": str(r["topic_id"]),
            "score": float(r["score"]),
        }
        for _, r in scores.iterrows()
    ]
    with open(OUT_USERS, "w", encoding="utf-8") as f:
        json.dump({"users": user_profiles, "scores": score_records}, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(talks)} talks, {len(halls)} halls, {len(slots)} slots")
    print(f"OK: {len(user_profiles)} reference users, {len(score_records)} scores")
    print(f"WROTE: {OUT_PROG}")
    print(f"WROTE: {OUT_USERS}")
    print("\nКатегории:")
    cats = defaultdict(int)
    for t in talks:
        cats[t["category"]] += 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print("\nЗалы:")
    by_hall = defaultdict(int)
    for t in talks:
        by_hall[t["hall"]] += 1
    for h, n in sorted(by_hall.items()):
        print(f"  hall {h}: {n} talks (cap={HALL_CAPACITIES.get(h, 200)})")
    print("\nСлоты:")
    for k in slots:
        print(f"  {k}: {len(slot_index[k])} parallel talks")


if __name__ == "__main__":
    main()
