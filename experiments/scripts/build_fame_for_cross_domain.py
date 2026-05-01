"""Сгенерировать fame-файлы для cross-domain датасетов.

Для каждого датасета fame — это нормированная мера «популярности» айтема в его
домене. Используется в симуляторе для star-chaser-механизма.

- Meetup: fame(event) = total yes-RSVPs группы события (нормировано по
  максимуму в датасете).
- ITC-2019: fame(class) = число студентов, записанных на course (нормировано).
- ITC-2007 T1: fame(exam) = число студентов, сдающих этот экзамен.
- ITC-2007 T2: fame(event) = sum по столбцу attendance матрицы.

Каждый fame-файл хранится рядом с программой как <dataset>_fame.json
с тем же форматом, что mobius_2025_autumn_fame.json.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXT = ROOT / "data" / "external" / "deep_search_2026_05"
MEETUP_SRC = EXT / "round2" / "meetup_rsvp" / "RSVP-Prediction-Meetup" / "data"
ITC2019_SRC = EXT / "round2" / "itc2019" / "MPPTimetables" / "data" / "input" / "ITC-2019"
ITC2007_T1_SRC = EXT / "itc_track1_solver" / "data" / "exam"
ITC2007_T2_SRC = EXT / "itc_track1_solver" / "data" / "tim"


def build_meetup():
    """fame(event) = popularity группы / max popularity."""
    print("[meetup_rsvp]")
    conf = json.load(open(ROOT / "data" / "conferences" / "meetup_rsvp.json"))
    events = json.load(open(MEETUP_SRC / "events.json"))
    group_pop = Counter()
    for e in events:
        gid = e.get("group_id")
        if not gid:
            continue
        yc = sum(1 for r in (e.get("rsvps") or []) if r.get("response") == "yes")
        group_pop[gid] += yc
    max_pop = max(group_pop.values()) if group_pop else 1
    cat_to_gid = {}
    for full_gid in group_pop:
        key = full_gid[:40]
        if key not in cat_to_gid or group_pop[full_gid] > group_pop[cat_to_gid[key]]:
            cat_to_gid[key] = full_gid
    fame = {}
    for t in conf["talks"]:
        full_gid = cat_to_gid.get(t.get("category", ""))
        if full_gid:
            fame[t["id"]] = round(group_pop[full_gid] / max_pop, 4)
        else:
            fame[t["id"]] = 0.0
    nonzero = sum(1 for v in fame.values() if v > 0)
    print(f"  {nonzero}/{len(fame)} talks with non-zero fame; max_pop={max_pop}")
    out = ROOT / "data" / "conferences" / "meetup_rsvp_fame.json"
    json.dump({"fame": fame}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  WROTE: {out}")


def build_itc2019(instance: str):
    """fame(class) = #students enrolled on the course / max."""
    print(f"[itc2019_{instance.replace('-', '_')}]")
    base = f"itc2019_{instance.replace('-', '_')}"
    conf = json.load(open(ROOT / "data" / "conferences" / f"{base}.json"))
    tree = ET.parse(ITC2019_SRC / f"{instance}.xml")
    root = tree.getroot()
    course_demand = Counter()
    for s in root.find("students"):
        for c in s.findall("course"):
            course_demand[int(c.attrib["id"])] += 1
    max_d = max(course_demand.values()) if course_demand else 1
    fame = {}
    for t in conf["talks"]:
        # category in itc adapter: "c{course_id}"
        cat = t.get("category", "")
        if cat.startswith("c"):
            try:
                course_id = int(cat[1:])
                fame[t["id"]] = round(course_demand.get(course_id, 0) / max_d, 4)
            except ValueError:
                fame[t["id"]] = 0.0
        else:
            fame[t["id"]] = 0.0
    nonzero = sum(1 for v in fame.values() if v > 0)
    print(f"  {nonzero}/{len(fame)} talks; max demand {max_d}")
    out = ROOT / "data" / "conferences" / f"{base}_fame.json"
    json.dump({"fame": fame}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  WROTE: {out}")


def build_itc2007_t1(instance: str):
    """fame(exam) = #students taking this exam / max."""
    print(f"[itc2007_t1_{instance}]")
    base = f"itc2007_t1_{instance}"
    conf = json.load(open(ROOT / "data" / "conferences" / f"{base}.json"))
    txt = (ITC2007_T1_SRC / f"{instance}.exam").read_text(errors="ignore")
    import re
    m = re.search(r"\[Exams:(\d+)\]\s*\n(.+?)(?=\[)", txt, re.S)
    n_ex = int(m.group(1))
    counts = []
    for i, line in enumerate(m.group(2).strip().splitlines()[:n_ex]):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            counts.append(0)
            continue
        counts.append(max(0, len(parts) - 1))
    max_c = max(counts) if counts else 1
    fame = {}
    for t in conf["talks"]:
        # talk id: "ex{i:04d}"
        try:
            idx = int(t["id"].replace("ex", ""))
            fame[t["id"]] = round(counts[idx] / max_c, 4) if idx < len(counts) else 0.0
        except (ValueError, IndexError):
            fame[t["id"]] = 0.0
    nonzero = sum(1 for v in fame.values() if v > 0)
    print(f"  {nonzero}/{len(fame)} talks; max count {max_c}")
    out = ROOT / "data" / "conferences" / f"{base}_fame.json"
    json.dump({"fame": fame}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  WROTE: {out}")


def build_itc2007_t2(instance: str):
    """fame(event) = sum по столбцу attendance / max sum."""
    print(f"[itc2007_t2_{instance.replace('-', '_')}]")
    base = f"itc2007_t2_{instance.replace('-', '_')}"
    conf = json.load(open(ROOT / "data" / "conferences" / f"{base}.json"))
    lines = (ITC2007_T2_SRC / f"{instance}.tim").read_text().splitlines()
    n_ev, n_r, n_f, n_s = map(int, lines[0].split())
    cur = 1 + n_r
    attendance = np.zeros((n_s, n_ev), dtype=np.int8)
    for s in range(n_s):
        for e in range(n_ev):
            attendance[s, e] = int(lines[cur].strip())
            cur += 1
    demand = attendance.sum(axis=0)
    max_d = int(demand.max()) if demand.size else 1
    fame = {}
    for t in conf["talks"]:
        try:
            idx = int(t["id"].replace("ev", ""))
            fame[t["id"]] = round(int(demand[idx]) / max_d, 4) if idx < len(demand) else 0.0
        except (ValueError, IndexError):
            fame[t["id"]] = 0.0
    nonzero = sum(1 for v in fame.values() if v > 0)
    print(f"  {nonzero}/{len(fame)} talks; max demand {max_d}")
    out = ROOT / "data" / "conferences" / f"{base}_fame.json"
    json.dump({"fame": fame}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  WROTE: {out}")


if __name__ == "__main__":
    build_meetup()
    build_itc2019("mary-fal18")
    build_itc2019("bet-spr18")
    build_itc2007_t1("exam_comp_set1")
    build_itc2007_t2("comp-2007-2-1")
