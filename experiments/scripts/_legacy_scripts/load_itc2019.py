"""Cross-domain валидация на ITC-2019 (International Timetabling Competition):
адаптация университетского расписания (courses, classes, rooms, students) →
наша постановка (talks, halls, slots, users).

Маппинг:
- talk = class (одна секция курса; у каждого курса 1+ subpart × 1+ class)
- hall = room (capacity = атрибут <room capacity="..."/>)
- slot = выбранный (days, start) ∈ доступных <time> у класса —
        для cross-domain прогона достаточно отнести каждый class в один
        синтетический tomeslot по его первому допустимому <time> в weeks/days/start.
- user = student (записан на курс через <student><course id="..."/>)
- эмбеддинг: bag-of-courses пользователя для users; bag-of-students-features (one-hot
  по курсу) для talks. Для cross-domain нам нужна функция релевантности cosine.
  Используем topic-vector: эмбеддинг класса = one-hot его course_id (и индексы
  курсов дают estate); эмбеддинг студента = bag-of-courses, нормированный.
  Это даёт cosine = доля курсов студента, на которые попадает класс.

Принимаем XML inst (mary-fal18 по умолчанию), пишем стандартный наш контракт.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ITC_DIR = ROOT / "data" / "external" / "deep_search_2026_05" / "round2" / "itc2019" / "MPPTimetables" / "data" / "input" / "ITC-2019"

MAX_TALKS = 600
MAX_USERS = 1000


def slot_id_of_class(class_elem):
    """Берём первый <time> у класса как назначение слота (грубо, для cross-domain)."""
    for t in class_elem.findall("time"):
        days = t.attrib.get("days", "0000000")
        weeks = t.attrib.get("weeks", "0000")
        start = t.attrib.get("start", "0")
        # Берём первый "1" в days как день недели, week — взять самую раннюю
        try:
            day_idx = days.index("1")
        except ValueError:
            day_idx = 0
        try:
            week_idx = weeks.index("1")
        except ValueError:
            week_idx = 0
        return f"w{week_idx:02d}d{day_idx}s{int(start):04d}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="mary-fal18", help="ITC-2019 instance basename")
    ap.add_argument("--max-talks", type=int, default=MAX_TALKS)
    ap.add_argument("--max-users", type=int, default=MAX_USERS)
    args = ap.parse_args()

    xml_path = ITC_DIR / f"{args.instance}.xml"
    if not xml_path.exists():
        raise SystemExit(f"Not found: {xml_path}")
    print(f"Loading {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Rooms → halls
    rooms = root.find("rooms")
    halls_raw = []
    room_id_map = {}
    for r in rooms:
        rid = int(r.attrib["id"])
        cap = int(r.attrib["capacity"])
        room_id_map[rid] = len(halls_raw) + 1
        halls_raw.append({"id": room_id_map[rid], "capacity": cap})
    print(f"  rooms: {len(halls_raw)}; cap range {min(h['capacity'] for h in halls_raw)}..{max(h['capacity'] for h in halls_raw)}")

    # 2. Courses → классы → talks
    courses = root.find("courses")
    course_ids = []  # list of course_id строк
    class_records = []  # dict per class
    course_id_idx = {}
    for c in courses:
        cid = int(c.attrib["id"])
        if cid not in course_id_idx:
            course_id_idx[cid] = len(course_id_idx)
            course_ids.append(cid)
        for cfg in c.findall("config"):
            for sp in cfg.findall("subpart"):
                for cls in sp.findall("class"):
                    class_id = int(cls.attrib["id"])
                    limit = int(cls.attrib.get("limit", 0))
                    rooms_allowed = []
                    for rr in cls.findall("room"):
                        if int(rr.attrib["id"]) in room_id_map:
                            rooms_allowed.append(room_id_map[int(rr.attrib["id"])])
                    sid = slot_id_of_class(cls)
                    if sid is None or not rooms_allowed:
                        continue
                    # Берём первую допустимую комнату как «зал». Это упрощение:
                    # в реальной timetabling нужен solver, но для cross-domain
                    # достаточно фиксированного назначения.
                    hall = rooms_allowed[0]
                    class_records.append({
                        "class_id": class_id,
                        "course_id": cid,
                        "limit": limit,
                        "hall": hall,
                        "slot_id": sid,
                    })
    print(f"  classes (talks-candidates): {len(class_records)}; courses: {len(course_ids)}")

    # 3. Подсэмпл: оставим только слоты, в которых ≥ 2 классов в разных комнатах
    slot_classes = defaultdict(list)
    for cr in class_records:
        slot_classes[cr["slot_id"]].append(cr)
    parallel = {sid: lst for sid, lst in slot_classes.items() if len(lst) >= 2}
    print(f"  slots with ≥2 parallel classes: {len(parallel)}")

    # Берём top-N слотов по числу классов (чтобы получить плотные слоты)
    sorted_slots = sorted(parallel.items(), key=lambda kv: -len(kv[1]))
    talks = []
    used_halls = set()
    talk_ids_per_slot = defaultdict(list)
    sorted_slots_taken = []
    for sid, classes in sorted_slots:
        if len(talks) + len(classes) > args.max_talks:
            classes = classes[: args.max_talks - len(talks)]
            if len(classes) < 2:
                break
        for cr in classes:
            tid = f"c{cr['course_id']}_cls{cr['class_id']}"
            talks.append({
                "id": tid,
                "title": f"course {cr['course_id']} class {cr['class_id']}",
                "hall": cr["hall"],
                "slot_id": cr["slot_id"],
                "category": f"c{cr['course_id']}",
                "abstract": "",
                "fame": 0.0,
                "_class_id": cr["class_id"],
                "_course_id": cr["course_id"],
                "_limit": cr["limit"],
            })
            used_halls.add(cr["hall"])
            talk_ids_per_slot[cr["slot_id"]].append(tid)
        sorted_slots_taken.append(sid)
        if len(talks) >= args.max_talks:
            break
    print(f"  selected talks: {len(talks)} in {len(sorted_slots_taken)} slots")

    # 4. Halls — оставляем только использованные
    halls = [{"id": h["id"], "capacity": h["capacity"]} for h in halls_raw if h["id"] in used_halls]

    # 5. Slots
    slots = [
        {"id": sid, "datetime": f"2018-09-{(int(sid[1:3])%30)+1:02d}T08:00:00",
         "talk_ids": talk_ids_per_slot[sid]}
        for sid in sorted_slots_taken
    ]

    # 6. Эмбеддинги: classes — one-hot по course_id (нормированный, размерность = N courses)
    n_courses = len(course_ids)
    emb_dim = min(n_courses, 256)  # урезаем для скорости
    # PCA-like: top-emb_dim самых частых курсов; остальные коллабсируем в "other"
    course_count = defaultdict(int)
    for t in talks:
        course_count[t["_course_id"]] += 1
    top_courses = [c for c, _ in sorted(course_count.items(), key=lambda kv: -kv[1])[:emb_dim - 1]]
    course_idx_map = {c: i for i, c in enumerate(top_courses)}
    other_idx = emb_dim - 1

    talk_ids = [t["id"] for t in talks]
    talk_embs = np.zeros((len(talks), emb_dim), dtype=np.float32)
    for i, t in enumerate(talks):
        idx = course_idx_map.get(t["_course_id"], other_idx)
        talk_embs[i, idx] = 1.0
    # Нормализация (one-hot уже единичной длины)

    # 7. Users → берём студентов, у которых хотя бы 1 курс из top
    students = root.find("students")
    student_records = []
    for s in students:
        sid = int(s.attrib["id"])
        s_courses = [int(c.attrib["id"]) for c in s.findall("course")]
        if not s_courses:
            continue
        # Проверяем пересечение с нашими курсами (которые были в выбранных talks)
        v = np.zeros(emb_dim, dtype=np.float32)
        n_hit = 0
        for cid in s_courses:
            idx = course_idx_map.get(cid, None)
            if idx is not None:
                v[idx] += 1.0
                n_hit += 1
        if n_hit == 0:
            continue
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        student_records.append({"sid": sid, "emb": v, "n_courses": len(s_courses)})
    print(f"  students with ≥1 course in selected: {len(student_records)}")

    rng = np.random.default_rng(11)
    if len(student_records) > args.max_users:
        idxs = rng.choice(len(student_records), size=args.max_users, replace=False)
        sampled = [student_records[i] for i in idxs]
    else:
        sampled = student_records

    user_meta = []
    user_embs = []
    for r in sampled:
        user_meta.append({
            "id": f"itc_s{r['sid']}",
            "background": f"ITC-2019 student {r['sid']}, courses={r['n_courses']}",
        })
        user_embs.append(r["emb"])
    print(f"  selected users: {len(user_meta)}")

    # 8. Сохранение
    out_dir_conf = ROOT / "data" / "conferences"
    out_dir_pers = ROOT / "data" / "personas"
    base = f"itc2019_{args.instance.replace('-', '_')}"

    talks_clean = [{k: v for k, v in t.items() if not k.startswith("_")} for t in talks]
    conf_path = out_dir_conf / f"{base}.json"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": f"ITC-2019 {args.instance} (cross-domain)",
            "talks": talks_clean,
            "halls": halls,
            "slots": slots,
        }, f, ensure_ascii=False, indent=2)
    np.savez(out_dir_conf / f"{base}_embeddings.npz",
             ids=np.array(talk_ids),
             embeddings=talk_embs)

    pers_path = out_dir_pers / f"{base}_users.json"
    with open(pers_path, "w", encoding="utf-8") as f:
        json.dump(user_meta, f, ensure_ascii=False, indent=2)
    np.savez(out_dir_pers / f"{base}_users_embeddings.npz",
             ids=np.array([u["id"] for u in user_meta]),
             embeddings=np.array(user_embs, dtype=np.float32))

    print(f"\nWROTE: {conf_path}")
    print(f"WROTE: {pers_path}")
    print(f"Summary: {len(talks)} talks, {len(halls)} halls, {len(slots)} slots, {len(user_meta)} users")


if __name__ == "__main__":
    main()
