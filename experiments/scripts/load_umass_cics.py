"""Cross-domain валидация на UMass CICS Course Allocation (Bissias et al. 2025, arXiv:2502.10592).

Адаптация → наша постановка (talks, halls, slots, users):
- talk = курс-секция (Catalog × Section), 96 секций
- slot = уникальная пара (zc.days, Mtg Time) — параллельные курсы делят один слот
- hall = секция (id внутри слота); capacity = Enrl Capacity
- user = студент (ResponseId), 1063 респондента
- preference: rank 1-7 на каждый из 71 курса (8 = required for major)

Эмбеддинг:
- talk: one-hot вектор по course_idx (длина 71)
- user: вектор предпочтений (длина 71), нормирован по L2:
        user[i] = (rank-1)/6 если 1<=rank<=7; rank=8 → +1.0; пропуск → 0

cosine(user, talk_one_hot) ≈ нормированный rank пользователя на этот курс
→ полностью соответствует семантике real preferences без leakage.

Запуск:
    .venv/bin/python scripts/load_umass_cics.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "umass_cics"


def normalize_time_slot(days: str, mtg_time: str) -> str:
    """Канонизация (days, time) → slot_id."""
    return f"{str(days).strip()}-{str(mtg_time).strip()}"


def parse_course_from_mapping(html_desc: str) -> str | None:
    """Извлечь 'DEPT 302-01' из HTML-описания survey-колонки."""
    if not isinstance(html_desc, str):
        return None
    # <strong>Course:&nbsp;</strong>DEPT 302-01 &nbsp;<strong>Schedule...
    m = re.search(r"Course:[^A-Z]*([A-Z]+\s+\d+[\-\,\d]*)", html_desc)
    if not m:
        return None
    return m.group(1).strip()


def main():
    # --- Загрузка курсов ---
    courses = pd.read_excel(DATA_DIR / "anonymized_courses.xlsx")
    print(f"Loaded courses: {len(courses)} sections")

    # Канонизация slot_id из (zc.days, Mtg Time)
    courses["slot_id"] = courses.apply(
        lambda r: normalize_time_slot(r["zc.days"], r["Mtg Time"]), axis=1)

    # talk_id = "Catalog-Section" (e.g. "302-01")
    courses["talk_id"] = courses.apply(
        lambda r: f"{r['Catalog']}-{str(r['Section']).zfill(2)}", axis=1)

    # course_key = просто Catalog (без секции) — используем для матчинга с survey
    courses["course_key"] = courses["Catalog"].astype(str)

    # --- Survey column mapping → course_key ---
    mapping_df = pd.read_csv(DATA_DIR / "survey_column_mapping.csv", sep="|")
    # колонки 7_X и 7 _X указывают на конкретные секции (DEPT 302-01)
    # извлекаем catalog (302) из строк
    surveycol_to_catalog = {}
    for _, row in mapping_df.iterrows():
        q = row["question"]
        desc = row["description"]
        course_str = parse_course_from_mapping(desc)
        if course_str is None:
            continue
        # course_str ~ "DEPT 302-01" или "DEPT 301-01,02"
        m = re.match(r"[A-Z]+\s+(\d+)", course_str)
        if m:
            cat = m.group(1)
            surveycol_to_catalog[q] = cat
    print(f"Mapped {len(surveycol_to_catalog)} survey-columns to catalog numbers")

    # --- Загрузка survey ---
    survey = pd.read_csv(DATA_DIR / "survey_data.csv")
    print(f"Loaded survey: {len(survey)} responses")

    # фильтр: только finished == 1 и progress == 100
    survey = survey[(survey["Finished"] == 1) & (survey["Progress"] == 100)].copy()
    print(f"After filter (finished+100% progress): {len(survey)}")

    # --- Уникальные курсы для эмбеддинг-индекса ---
    # Используем catalog numbers, встречающиеся И в survey-mapping, И в courses
    courses_in_xlsx = set(courses["course_key"].astype(str).unique())
    courses_in_survey = set(surveycol_to_catalog.values())
    common_courses = sorted(courses_in_xlsx & courses_in_survey)
    print(f"Common courses (in both files): {len(common_courses)}")

    # фильтр секций к только тем, что есть в обоих
    courses_active = courses[courses["course_key"].isin(common_courses)].copy()
    print(f"Active sections: {len(courses_active)}")

    course_idx = {c: i for i, c in enumerate(common_courses)}
    n_courses = len(common_courses)

    # --- Сборка talks/halls/slots ---
    talks = []
    halls_set = {}
    slots_dict = defaultdict(list)
    talk_emb_rows = []
    talk_ids = []

    hall_id_counter = 0
    for _, row in courses_active.iterrows():
        tid = row["talk_id"]
        ck = row["course_key"]
        sid = row["slot_id"]
        cap = int(row["Enrl Capacity"]) if pd.notna(row["Enrl Capacity"]) else 30
        section = row["Section"]
        # hall = одна на talk
        hall = hall_id_counter
        hall_id_counter += 1
        halls_set[hall] = cap

        emb_vec = np.zeros(n_courses, dtype=np.float32)
        emb_vec[course_idx[ck]] = 1.0

        talks.append({
            "id": tid,
            "title": f"{row['Subject']} {ck}-{str(section).zfill(2)}",
            "hall": hall,
            "slot_id": sid,
            "category": str(ck),  # for stylized facts compatibility
            "abstract": f"Subject: {row['Subject']}, Categories: {row['Categories']}, "
                        f"Days: {row['zc.days']}, Time: {row['Mtg Time']}",
            "fame": 0.0,
        })
        talk_ids.append(tid)
        talk_emb_rows.append(emb_vec)
        slots_dict[sid].append(tid)

    halls = [{"id": h, "capacity": c} for h, c in sorted(halls_set.items())]
    slots = [{"id": s, "datetime": s, "talk_ids": sorted(tids)}
             for s, tids in slots_dict.items()]
    slots.sort(key=lambda s: s["id"])
    print(f"Talks: {len(talks)}, Halls: {len(halls)}, Slots: {len(slots)}")

    # отфильтровать слоты с одним talk (нет параллельного выбора → нет смысла)
    multi_slots = [s for s in slots if len(s["talk_ids"]) >= 2]
    print(f"Multi-talk slots: {len(multi_slots)} of {len(slots)} (используем все, recsys всё равно работает)")

    # --- Сборка users ---
    users = []
    user_emb_rows = []
    user_ids = []
    n_with_prefs = 0

    for _, row in survey.iterrows():
        uid = row["ResponseId"]
        emb = np.zeros(n_courses, dtype=np.float32)
        n_prefs = 0
        for q_col, cat in surveycol_to_catalog.items():
            if q_col not in row.index:
                continue
            v = row[q_col]
            if pd.isna(v):
                continue
            try:
                rank = int(v)
            except (ValueError, TypeError):
                continue
            if cat not in course_idx:
                continue
            ci = course_idx[cat]
            # rank 1-7 → нормированный score; 8 (required) → 1.0
            if 1 <= rank <= 7:
                emb[ci] = max(emb[ci], (rank - 1) / 6.0)
                n_prefs += 1
            elif rank == 8:
                emb[ci] = 1.0
                n_prefs += 1

        if n_prefs == 0:
            continue
        n_with_prefs += 1
        # L2 нормировка для совместимости с cosine
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        users.append({
            "id": str(uid),
            "background": f"UMass CS student, status_level={row.get('1', '?')}, "
                          f"n_courses_planned={row.get('2', '?')}, "
                          f"n_preferences_given={n_prefs}",
        })
        user_ids.append(str(uid))
        user_emb_rows.append(emb)

    print(f"Users with preferences: {n_with_prefs}")

    # --- Сохраняем ---
    out_conf = ROOT / "data" / "conferences" / "umass_cics.json"
    out_conf_emb = ROOT / "data" / "conferences" / "umass_cics_embeddings.npz"
    out_pers = ROOT / "data" / "personas" / "umass_cics_users.json"
    out_pers_emb = ROOT / "data" / "personas" / "umass_cics_users_embeddings.npz"

    with open(out_conf, "w", encoding="utf-8") as f:
        json.dump({
            "name": "UMass CICS Course Allocation Fall 2024",
            "talks": talks,
            "halls": halls,
            "slots": slots,
        }, f, ensure_ascii=False, indent=2)
    np.savez(out_conf_emb, ids=np.array(talk_ids),
             embeddings=np.array(talk_emb_rows, dtype=np.float32))

    with open(out_pers, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    np.savez(out_pers_emb, ids=np.array(user_ids),
             embeddings=np.array(user_emb_rows, dtype=np.float32))

    print(f"\nWROTE: {out_conf}")
    print(f"WROTE: {out_conf_emb}")
    print(f"WROTE: {out_pers}")
    print(f"WROTE: {out_pers_emb}")
    print(f"\nDataset summary:")
    print(f"  - {len(talks)} sections (talks)")
    print(f"  - {len(halls)} halls (1-per-section)")
    print(f"  - {len(slots)} time-slots ({len(multi_slots)} multi-talk)")
    print(f"  - {len(users)} students (users)")
    print(f"  - {n_courses} unique courses (embedding dim)")


if __name__ == "__main__":
    main()
