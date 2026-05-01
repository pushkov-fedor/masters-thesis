"""Cross-domain валидация на ITC-2007 (Track 1 Examination + Track 2 Course Timetabling):
адаптация → наша постановка (talks, halls, slots, users).

Track 1 (.exam):
- секции [Exams:N], [Periods:M], [Rooms:K], + constraint-секции
- talk = exam (со списком студентов и длительностью)
- hall = room (capacity)
- slot = period (M периодов, экзамены ходят параллельно в один период)
- user = student (id из списков экзаменов)
- эмбеддинг = bag-of-exams для студента, bag-of-students для экзамена

Track 2 (.tim):
- N_events N_rooms N_features N_students в заголовке
- N_rooms строк capacity
- бинарная матрица student×event (хочет/не хочет)
- + матрицы room-features и event-features
- + 45 availability-флагов на event
- talk = event, hall = room, slot из 45 фиксированных
- user = student
- эмбеддинг: bag-of-features

Использование: python load_itc2007.py --track 1 [--instance exam_comp_set1]
                python load_itc2007.py --track 2 [--instance comp-2007-2-1]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T1_DIR = ROOT / "data" / "external" / "deep_search_2026_05" / "itc_track1_solver" / "data" / "exam"
T2_DIR = ROOT / "data" / "external" / "deep_search_2026_05" / "itc_track1_solver" / "data" / "tim"

MAX_TALKS = 600
MAX_USERS = 1000


def load_track1(inst: str):
    path = T1_DIR / f"{inst}.exam"
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # Exams
    m = re.search(r"\[Exams:(\d+)\]\s*\n(.+?)(?=\[)", txt, re.S)
    n_ex = int(m.group(1))
    exams_block = m.group(2).strip().splitlines()
    exams = []
    for i, line in enumerate(exams_block[:n_ex]):
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        duration = int(parts[0])
        students = [int(s) for s in parts[1:] if s.strip()]
        exams.append({"id": i, "duration": duration, "students": students})

    # Periods
    m = re.search(r"\[Periods:(\d+)\]\s*\n(.+?)(?=\[)", txt, re.S)
    n_p = int(m.group(1))
    periods = []
    for i, line in enumerate(m.group(2).strip().splitlines()[:n_p]):
        parts = [p.strip() for p in line.split(",")]
        periods.append({"id": i, "date": parts[0], "time": parts[1],
                        "duration": int(parts[2]), "penalty": int(parts[3])})

    # Rooms
    m = re.search(r"\[Rooms:(\d+)\]\s*\n(.+?)(?=\[)", txt, re.S)
    n_r = int(m.group(1))
    rooms = []
    for i, line in enumerate(m.group(2).strip().splitlines()[:n_r]):
        parts = [p.strip() for p in line.split(",")]
        rooms.append({"id": i, "capacity": int(parts[0]), "penalty": int(parts[1])})

    return exams, periods, rooms


def assign_track1(exams, periods, rooms, max_talks):
    """Heuristic: распределяем экзамены по периодам через round-robin (sorted by len(students))
    и по комнатам через жадную capacity-первичную. Это не оптимальное расписание, но даёт
    структуру параллельности зал × слот, как нам нужно."""
    n_p = len(periods)
    n_r = len(rooms)
    # сортируем по убыванию числа студентов
    sorted_exams = sorted(exams, key=lambda e: -len(e["students"]))
    period_idx = 0
    placement = []
    period_load = defaultdict(lambda: defaultdict(int))  # period -> room -> sum students
    for ex in sorted_exams:
        if len(placement) >= max_talks:
            break
        # ищем period × room с наименьшей текущей нагрузкой и подходящей capacity
        best = None
        for p in range(n_p):
            for r in range(n_r):
                if period_load[p][r] + len(ex["students"]) > rooms[r]["capacity"] * 1.3:
                    continue  # слишком переполнит
                if best is None or period_load[p][r] < period_load[best[0]][best[1]]:
                    best = (p, r)
        if best is None:
            # fallback: round-robin
            best = (period_idx % n_p, period_idx % n_r)
            period_idx += 1
        placement.append({**ex, "period_id": best[0], "room_id": best[1]})
        period_load[best[0]][best[1]] += len(ex["students"])
    return placement


def main_track1(inst: str, max_talks: int, max_users: int):
    print(f"Loading Track 1 instance: {inst}")
    exams, periods, rooms = load_track1(inst)
    print(f"  exams={len(exams)} periods={len(periods)} rooms={len(rooms)}")

    placement = assign_track1(exams, periods, rooms, max_talks)
    print(f"  placement: {len(placement)} talks distributed")

    # Halls
    used_rooms = sorted({p["room_id"] for p in placement})
    halls = [{"id": rid + 1, "capacity": rooms[rid]["capacity"]} for rid in used_rooms]
    rid_to_hall = {rid: rid + 1 for rid in used_rooms}

    # Slots
    used_periods = sorted({p["period_id"] for p in placement})
    pid_to_slot = {pid: f"p{pid:03d}" for pid in used_periods}

    talks = []
    talk_ids = []
    talk_to_students = {}
    for p in placement:
        tid = f"ex{p['id']:04d}"
        talks.append({
            "id": tid,
            "title": f"exam {p['id']} dur={p['duration']}",
            "hall": rid_to_hall[p["room_id"]],
            "slot_id": pid_to_slot[p["period_id"]],
            "category": f"d{p['duration']}",
            "abstract": "",
            "fame": 0.0,
            "_n_students": len(p["students"]),
        })
        talk_ids.append(tid)
        talk_to_students[tid] = p["students"]

    talk_id_set = {t["id"]: t for t in talks}
    slot_to_talks = defaultdict(list)
    for t in talks:
        slot_to_talks[t["slot_id"]].append(t["id"])
    slots = [
        {"id": pid_to_slot[pid], "datetime": periods[pid]["date"] + "T" + periods[pid]["time"],
         "talk_ids": slot_to_talks[pid_to_slot[pid]]}
        for pid in used_periods
    ]

    # Эмбеддинги: one-hot по talk_id (топ-emb_dim самых "крупных" экзаменов)
    n_dim = min(len(talks), 256)
    talk_idx = {tid: i for i, tid in enumerate(talk_ids[:n_dim])}
    other_idx = n_dim - 1 if len(talks) > n_dim else None
    talk_embs = np.zeros((len(talks), n_dim), dtype=np.float32)
    for i, tid in enumerate(talk_ids):
        idx = talk_idx.get(tid, other_idx if other_idx is not None else 0)
        talk_embs[i, idx] = 1.0

    # Users — студенты, у которых ≥ 2 экзаменов в выбранных
    student_exams = defaultdict(list)
    for tid, students in talk_to_students.items():
        for s in students:
            student_exams[s].append(tid)
    qualified = {s: tids for s, tids in student_exams.items() if len(tids) >= 2}
    print(f"  students with ≥2 exams in selected talks: {len(qualified)}")

    rng = np.random.default_rng(13)
    sids = list(qualified.keys())
    if len(sids) > max_users:
        sids = list(rng.choice(sids, size=max_users, replace=False))
    user_meta = []
    user_embs = []
    for sid in sids:
        v = np.zeros(n_dim, dtype=np.float32)
        for tid in qualified[sid]:
            idx = talk_idx.get(tid, other_idx if other_idx is not None else None)
            if idx is not None:
                v[idx] += 1.0
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        user_meta.append({
            "id": f"itc7t1_s{sid}",
            "background": f"ITC-2007 T1 student {sid}",
        })
        user_embs.append(v)
    print(f"  selected users: {len(user_meta)}")

    base = f"itc2007_t1_{inst}"
    save_outputs(base, "ITC-2007 Track 1 " + inst, talks, halls, slots, talk_ids, talk_embs,
                 user_meta, user_embs)


def load_track2(inst: str):
    path = T2_DIR / f"{inst}.tim"
    lines = path.read_text().splitlines()
    n_ev, n_r, n_f, n_s = map(int, lines[0].split())
    caps = [int(lines[1 + i]) for i in range(n_r)]
    cur = 1 + n_r
    # student×event matrix: n_s × n_ev строк по 0/1
    attendance = np.zeros((n_s, n_ev), dtype=np.int8)
    for s in range(n_s):
        for e in range(n_ev):
            attendance[s, e] = int(lines[cur].strip())
            cur += 1
    # room-features
    room_features = np.zeros((n_r, n_f), dtype=np.int8)
    for r in range(n_r):
        for k in range(n_f):
            room_features[r, k] = int(lines[cur].strip())
            cur += 1
    # event-features
    event_features = np.zeros((n_ev, n_f), dtype=np.int8)
    for e in range(n_ev):
        for k in range(n_f):
            event_features[e, k] = int(lines[cur].strip())
            cur += 1
    # event-availability 45 slots
    event_avail = np.zeros((n_ev, 45), dtype=np.int8)
    for e in range(n_ev):
        for sl in range(45):
            event_avail[e, sl] = int(lines[cur].strip())
            cur += 1
    return {
        "n_events": n_ev, "n_rooms": n_r, "n_features": n_f, "n_students": n_s,
        "capacities": caps, "attendance": attendance, "room_features": room_features,
        "event_features": event_features, "event_avail": event_avail,
    }


def assign_track2(d, max_talks):
    n_ev = d["n_events"]
    n_r = d["n_rooms"]
    # Жадно: каждому event назначаем slot из его availability с наименьшей загрузкой,
    # и room с подходящей capacity (max attendees) и feature-совместимости.
    # Для cross-domain достаточно простого назначения.
    event_demand = d["attendance"].sum(axis=0)  # сколько students хочет каждый event
    placement = []
    slot_load = defaultdict(lambda: defaultdict(int))  # slot -> room -> demand
    sorted_events = sorted(range(n_ev), key=lambda e: -event_demand[e])
    for e in sorted_events[:max_talks * 3]:  # с запасом, потом обрежем
        if len(placement) >= max_talks:
            break
        avail_slots = [sl for sl in range(45) if d["event_avail"][e, sl]]
        if not avail_slots:
            continue
        # подходящие комнаты по features
        compat_rooms = []
        for r in range(n_r):
            if all(d["event_features"][e, k] <= d["room_features"][r, k]
                   for k in range(d["n_features"])):
                compat_rooms.append(r)
        if not compat_rooms:
            compat_rooms = list(range(n_r))
        # выбираем (slot, room) с наименьшей загрузкой и cap >= demand*0.5
        best = None
        for sl in avail_slots:
            for r in compat_rooms:
                if d["capacities"][r] < event_demand[e] * 0.3:
                    continue
                key = (sl, r)
                if best is None or slot_load[sl][r] < slot_load[best[0]][best[1]]:
                    best = key
        if best is None:
            best = (avail_slots[0], compat_rooms[0])
        placement.append({"event_id": e, "slot": best[0], "room": best[1]})
        slot_load[best[0]][best[1]] += int(event_demand[e])
    return placement


def main_track2(inst: str, max_talks: int, max_users: int):
    print(f"Loading Track 2 instance: {inst}")
    d = load_track2(inst)
    print(f"  events={d['n_events']} rooms={d['n_rooms']} students={d['n_students']} features={d['n_features']}")

    placement = assign_track2(d, max_talks)
    print(f"  placement: {len(placement)} talks")

    used_rooms = sorted({p["room"] for p in placement})
    halls = [{"id": r + 1, "capacity": d["capacities"][r]} for r in used_rooms]
    rid_to_hall = {r: r + 1 for r in used_rooms}

    used_slots = sorted({p["slot"] for p in placement})
    sid_to_slot_id = {s: f"s{s:02d}" for s in used_slots}

    talks = []
    talk_ids = []
    for p in placement:
        tid = f"ev{p['event_id']:03d}"
        talks.append({
            "id": tid,
            "title": f"event {p['event_id']}",
            "hall": rid_to_hall[p["room"]],
            "slot_id": sid_to_slot_id[p["slot"]],
            "category": "ev",
            "abstract": "",
            "fame": 0.0,
            "_event_id": p["event_id"],
        })
        talk_ids.append(tid)

    slot_to_talks = defaultdict(list)
    for t in talks:
        slot_to_talks[t["slot_id"]].append(t["id"])
    slots = [{"id": sid_to_slot_id[s], "datetime": f"2007-09-{(s%30)+1:02d}T08:00:00",
              "talk_ids": slot_to_talks[sid_to_slot_id[s]]} for s in used_slots]

    # Эмбеддинги: features (n_features-dim)
    n_dim = d["n_features"]
    talk_embs = np.zeros((len(talks), n_dim), dtype=np.float32)
    for i, t in enumerate(talks):
        v = d["event_features"][t["_event_id"]].astype(np.float32)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        talk_embs[i] = v

    # Users
    selected_event_ids = {t["_event_id"] for t in talks}
    student_attendance = []
    for s in range(d["n_students"]):
        eids = [e for e in selected_event_ids if d["attendance"][s, e]]
        if len(eids) >= 2:
            student_attendance.append((s, eids))
    print(f"  students with ≥2 selected events: {len(student_attendance)}")

    rng = np.random.default_rng(17)
    if len(student_attendance) > max_users:
        idxs = rng.choice(len(student_attendance), size=max_users, replace=False)
        student_attendance = [student_attendance[i] for i in idxs]

    user_meta = []
    user_embs = []
    for sid, eids in student_attendance:
        v = np.zeros(n_dim, dtype=np.float32)
        for e in eids:
            v += d["event_features"][e].astype(np.float32)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        user_meta.append({
            "id": f"itc7t2_s{sid}",
            "background": f"ITC-2007 T2 student {sid}, events={len(eids)}",
        })
        user_embs.append(v)
    print(f"  selected users: {len(user_meta)}")

    base = f"itc2007_t2_{inst.replace('-', '_')}"
    save_outputs(base, "ITC-2007 Track 2 " + inst, talks, halls, slots, talk_ids, talk_embs,
                 user_meta, user_embs)


def save_outputs(base, name, talks, halls, slots, talk_ids, talk_embs, user_meta, user_embs):
    out_dir_conf = ROOT / "data" / "conferences"
    out_dir_pers = ROOT / "data" / "personas"
    talks_clean = [{k: v for k, v in t.items() if not k.startswith("_")} for t in talks]
    conf_path = out_dir_conf / f"{base}.json"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "talks": talks_clean, "halls": halls, "slots": slots},
                  f, ensure_ascii=False, indent=2)
    np.savez(out_dir_conf / f"{base}_embeddings.npz",
             ids=np.array(talk_ids), embeddings=talk_embs)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, choices=[1, 2], required=True)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--max-talks", type=int, default=MAX_TALKS)
    ap.add_argument("--max-users", type=int, default=MAX_USERS)
    args = ap.parse_args()
    inst = args.instance or ("exam_comp_set1" if args.track == 1 else "comp-2007-2-1")
    if args.track == 1:
        main_track1(inst, args.max_talks, args.max_users)
    else:
        main_track2(inst, args.max_talks, args.max_users)
