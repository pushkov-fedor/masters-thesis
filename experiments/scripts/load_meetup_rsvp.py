"""Cross-domain валидация на Meetup RSVP dataset (ozgekoroglu/RSVP-Prediction-Meetup):
адаптация (events, users, venues, groups) → (talks, users, halls, slots).

Маппинг (вариант A+Y, согласован с автором):
- talk = event с rsvp_limit ≠ null
- hall = venue_id; capacity_hall = max rsvp_limit среди событий venue
- slot = (date, hour) дискретизация event.time
- user = пользователь с ≥ 5 yes-RSVPs из выбранных слотов
- эмбеддинги: топики групп, one-hot по top-K тем, нормализованы

Геофильтр: радиус 50 км от центра Амстердама (52.37, 4.90) — большая часть данных.
Слот валиден, если в нём ≥ 2 событий с rsvp_limit.

Артефакты:
- data/conferences/meetup_rsvp.json + meetup_rsvp_embeddings.npz
- data/personas/meetup_users.json + meetup_users_embeddings.npz

Plus saves раздельный raw_choices.json — реальные выборы пользователей в каждом
слоте (для B1 валидации симулятора).
"""
from __future__ import annotations

import ast
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_DIR = ROOT / "data" / "external" / "deep_search_2026_05" / "round2" / "meetup_rsvp" / "RSVP-Prediction-Meetup" / "data"

# Геозона: Амстердам ± 50 км
GEO_CENTER_LAT = 52.37
GEO_CENTER_LON = 4.90
GEO_RADIUS_KM = 50.0

# Слот = час
SLOT_BUCKET_HOURS = 1

# Параметры подсэмпла
N_USERS = 1000
MIN_USER_RSVPS = 5

# Эмбеддинги: top-K топиков групп
TOPIC_DIM = 128


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_topics(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return [t for t in v if isinstance(t, str)]
        except Exception:
            return []
    return []


def slot_id_of(time_ms: int) -> str:
    if not time_ms:
        return ""
    dt = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
    bucket = dt.hour // SLOT_BUCKET_HOURS * SLOT_BUCKET_HOURS
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{bucket:02d}"


def main():
    print("Loading Meetup RSVP dataset...")
    events = json.load(open(SRC_DIR / "events.json"))
    users_raw = json.load(open(SRC_DIR / "users.json"))
    venues = {v["venue_id"]: v for v in json.load(open(SRC_DIR / "venues.json"))}
    groups = {g["group_id"]: g for g in json.load(open(SRC_DIR / "groups.json"))}
    print(f"  events={len(events)} users={len(users_raw)} venues={len(venues)} groups={len(groups)}")

    # 1. Топики групп → top-K по частоте
    topic_count = Counter()
    for g in groups.values():
        for t in parse_topics(g.get("topics")):
            topic_count[t] += 1
    top_topics = [t for t, _ in topic_count.most_common(TOPIC_DIM)]
    topic_idx = {t: i for i, t in enumerate(top_topics)}
    print(f"  unique group topics: {len(topic_count)}; using top-{TOPIC_DIM}")

    # 2. Группа → topic embedding
    group_emb = {}
    for gid, g in groups.items():
        v = np.zeros(TOPIC_DIM, dtype=np.float32)
        for t in parse_topics(g.get("topics")):
            j = topic_idx.get(t)
            if j is not None:
                v[j] = 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        group_emb[gid] = v

    # 3. Фильтрация событий: rsvp_limit ≠ null, venue в геозоне, у группы есть topics
    valid_events = []
    for e in events:
        if e.get("rsvp_limit") is None:
            continue
        if not e.get("time") or not e.get("venue_id"):
            continue
        v = venues.get(e["venue_id"])
        if v is None or v.get("lat") is None or v.get("lon") is None:
            continue
        if haversine_km(v["lat"], v["lon"], GEO_CENTER_LAT, GEO_CENTER_LON) > GEO_RADIUS_KM:
            continue
        gid = e.get("group_id")
        if gid not in group_emb:
            continue
        if np.linalg.norm(group_emb[gid]) == 0:
            continue
        valid_events.append(e)
    print(f"  events after rsvp_limit + geo + topic filter: {len(valid_events)}")

    # 4. Группировка по слотам, оставить слоты с ≥ 2 параллельных
    slot_to_events = defaultdict(list)
    for e in valid_events:
        s = slot_id_of(e["time"])
        if s:
            slot_to_events[s].append(e)
    parallel_slots = {s: evs for s, evs in slot_to_events.items() if len(evs) >= 2}
    print(f"  slots with ≥2 parallel events: {len(parallel_slots)}")
    parallel_event_count = sum(len(evs) for evs in parallel_slots.values())
    print(f"  events in parallel slots: {parallel_event_count}")

    # Сортировка слотов по дате
    sorted_slots = sorted(parallel_slots.keys())

    # 5. Halls: venues с capacity = max(rsvp_limit) среди событий в этом venue
    venue_caps = defaultdict(int)
    for evs in parallel_slots.values():
        for e in evs:
            venue_caps[e["venue_id"]] = max(venue_caps[e["venue_id"]], int(e["rsvp_limit"]))
    venue_id_to_hall = {vid: i + 1 for i, vid in enumerate(sorted(venue_caps))}
    halls = [{"id": venue_id_to_hall[vid], "capacity": cap} for vid, cap in venue_caps.items()]
    print(f"  halls (venues): {len(halls)}; cap range {min(venue_caps.values())}..{max(venue_caps.values())}")

    # 6. Talks: события из параллельных слотов
    talks = []
    talk_ids = []
    talk_embs = []
    talk_in_slot = defaultdict(list)
    for s in sorted_slots:
        for e in parallel_slots[s]:
            tid = f"e{e.get('group_id','x')}_{e['time']}_{e['venue_id']}".replace("/", "_")[:80]
            # Уникализация на коллизии (в рамках слота)
            base = tid
            n = 0
            while tid in talk_in_slot[s] or any(t["id"] == tid for t in talks):
                n += 1
                tid = f"{base}_{n}"
            gid = e["group_id"]
            talks.append({
                "id": tid,
                "title": (e.get("name") or "")[:80],
                "hall": venue_id_to_hall[e["venue_id"]],
                "slot_id": s,
                "category": gid[:40],
                "abstract": "",
                "fame": 0.0,
                "_meetup_event_id_in_data": (e["time"], e["venue_id"], gid),
                "_rsvp_limit": int(e["rsvp_limit"]),
                "_yes_user_ids": [r["user_id"] for r in (e.get("rsvps") or [])
                                   if r.get("response") == "yes"],
            })
            talk_ids.append(tid)
            talk_embs.append(group_emb[gid])
            talk_in_slot[s].append(tid)

    slots = [
        {"id": s, "datetime": s + ":00:00Z",
         "talk_ids": [t["id"] for t in talks if t["slot_id"] == s]}
        for s in sorted_slots
    ]

    # 7. Users: подсэмпл с ≥ MIN_USER_RSVPS yes-RSVPs из выбранных слотов
    user_rsvps = defaultdict(list)  # user_id -> list of (slot_id, talk_id)
    for t in talks:
        for uid in t["_yes_user_ids"]:
            user_rsvps[uid].append((t["slot_id"], t["id"]))
    qualified = [uid for uid, rs in user_rsvps.items() if len(rs) >= MIN_USER_RSVPS]
    print(f"  users with ≥{MIN_USER_RSVPS} yes-RSVPs in selected slots: {len(qualified)}")

    # Эмбеддинги пользователей: усреднение по группам, в которых они состоят (memberships).
    user_meta_by_id = {u["user_id"]: u for u in users_raw}
    rng = np.random.default_rng(7)
    if len(qualified) > N_USERS:
        sampled_uids = list(rng.choice(qualified, size=N_USERS, replace=False))
    else:
        sampled_uids = qualified

    user_meta_out = []
    user_embs = []
    for uid in sampled_uids:
        u = user_meta_by_id.get(uid, {})
        memberships = u.get("memberships") or []
        v = np.zeros(TOPIC_DIM, dtype=np.float32)
        n_grp = 0
        for m in memberships:
            gid = m.get("group_id")
            if gid in group_emb:
                v += group_emb[gid]
                n_grp += 1
        if n_grp == 0:
            # fallback: усреднение по группам, на которые он RSVPил yes
            g_counts = Counter()
            for s, tid in user_rsvps[uid]:
                # Найти talk и взять его group через категорию
                t_obj = next(t for t in talks if t["id"] == tid)
                g_counts[t_obj["category"]] += 1
            for gname, c in g_counts.items():
                # категория talk = gid[:40], могут быть обрезаны — найдём оригинальный gid
                for full_gid in groups:
                    if full_gid[:40] == gname and full_gid in group_emb:
                        v += group_emb[full_gid] * c
                        n_grp += 1
                        break
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        else:
            # совсем нечего — нулевой эмбеддинг (user будет ходить «случайно» по cosine)
            pass
        user_meta_out.append({
            "id": f"mu_{uid}",
            "background": f"Meetup user {uid}, city={u.get('city','?')}, country={u.get('country','?')}",
        })
        user_embs.append(v)
    print(f"  selected users: {len(user_meta_out)}")

    # 8. Сохранение
    out_dir_conf = ROOT / "data" / "conferences"
    out_dir_pers = ROOT / "data" / "personas"

    # Очистим talks от служебных _-полей перед сохранением (но сохраним B1-данные отдельно)
    raw_choices = []
    for t in talks:
        raw_choices.append({
            "talk_id": t["id"],
            "slot_id": t["slot_id"],
            "rsvp_limit": t["_rsvp_limit"],
            "yes_user_ids": t["_yes_user_ids"],
            "hall_id": t["hall"],
            "group_id": t["category"],
        })
    talks_clean = [{k: v for k, v in t.items() if not k.startswith("_")} for t in talks]

    conf_path = out_dir_conf / "meetup_rsvp.json"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": "Meetup RSVP NL (cross-domain)",
            "talks": talks_clean,
            "halls": halls,
            "slots": slots,
        }, f, ensure_ascii=False, indent=2)
    np.savez(out_dir_conf / "meetup_rsvp_embeddings.npz",
             ids=np.array(talk_ids),
             embeddings=np.array(talk_embs, dtype=np.float32))
    raw_path = out_dir_conf / "meetup_rsvp_raw_choices.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_choices, f, ensure_ascii=False)

    pers_path = out_dir_pers / "meetup_users.json"
    with open(pers_path, "w", encoding="utf-8") as f:
        json.dump(user_meta_out, f, ensure_ascii=False, indent=2)
    np.savez(out_dir_pers / "meetup_users_embeddings.npz",
             ids=np.array([u["id"] for u in user_meta_out]),
             embeddings=np.array(user_embs, dtype=np.float32))

    print(f"\nWROTE: {conf_path}")
    print(f"WROTE: {out_dir_conf / 'meetup_rsvp_embeddings.npz'}")
    print(f"WROTE: {raw_path}  (для B1 валидации симулятора)")
    print(f"WROTE: {pers_path}")
    print(f"WROTE: {out_dir_pers / 'meetup_users_embeddings.npz'}")
    print()
    print(f"Summary: {len(talks_clean)} talks, {len(halls)} halls, {len(slots)} slots, {len(user_meta_out)} users")
    print(f"Capacity range: {min(venue_caps.values())}..{max(venue_caps.values())} (per venue)")
    print(f"Talks per slot mean: {len(talks_clean)/max(1,len(slots)):.2f}")


if __name__ == "__main__":
    main()
