"""Калибровка модели compliance на реальных Meetup RSVPs.

Задача: оценить долю трёх типов поведения пользователей при выборе из
параллельных событий:
- A (compliant): идёт по релевантности — argmax cosine(user, event)
- C (star-chaser): идёт на популярное событие — argmax popularity(event)
- B (curious / mix): выбор не совпал ни с топ-релевантным, ни с топ-популярным

Популярность события: суммарное число yes-RSVPs группы (за всю историю датасета).
Это прокси «насколько эта тематика обычно собирает аудиторию».

Источник:
- experiments/data/conferences/meetup_rsvp.json (talks с group_id и slot_id)
- experiments/data/conferences/meetup_rsvp_embeddings.npz
- experiments/data/personas/meetup_users_embeddings.npz
- experiments/data/conferences/meetup_rsvp_raw_choices.json (real yes_user_ids)
- experiments/data/external/.../events.json (для popularity всей группы)

Выход:
- experiments/results/compliance_calibration_meetup.json
- консольная сводка
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "external" / "deep_search_2026_05" / "round2" / "meetup_rsvp" / "RSVP-Prediction-Meetup" / "data"


def main():
    print("Loading Meetup data...")
    conf = json.load(open(ROOT / "data" / "conferences" / "meetup_rsvp.json"))
    emb = np.load(ROOT / "data" / "conferences" / "meetup_rsvp_embeddings.npz")
    talk_emb = {tid: emb["embeddings"][i] for i, tid in enumerate(emb["ids"])}
    raw = json.load(open(ROOT / "data" / "conferences" / "meetup_rsvp_raw_choices.json"))
    pemb = np.load(ROOT / "data" / "personas" / "meetup_users_embeddings.npz")
    user_emb = {pid: pemb["embeddings"][i] for i, pid in enumerate(pemb["ids"])}

    # popularity per group: суммарное число yes-RSVPs во всей истории дата
    print("Computing group popularity from full Meetup events.json...")
    events = json.load(open(SRC / "events.json"))
    group_popularity = Counter()
    for e in events:
        gid = e.get("group_id")
        if not gid:
            continue
        yc = sum(1 for r in (e.get("rsvps") or []) if r.get("response") == "yes")
        group_popularity[gid] += yc

    # talk_meta: какой group_id у talk_id (через "category" — это и есть gid[:40])
    # точнее восстановим: сохранили group_id урезанным до 40 символов в category
    # для стабильного маппинга найдём original gid каждого talk
    cat_to_gid = {}
    for full_gid in group_popularity:
        key = full_gid[:40]
        # возможно коллизии, но это rare; берём максимально популярный
        if key not in cat_to_gid or group_popularity[full_gid] > group_popularity[cat_to_gid[key]]:
            cat_to_gid[key] = full_gid

    # talk → popularity
    talk_meta = {t["id"]: t for t in conf["talks"]}
    talk_pop = {}
    for tid, t in talk_meta.items():
        cat = t.get("category", "")
        full_gid = cat_to_gid.get(cat)
        talk_pop[tid] = group_popularity.get(full_gid, 0) if full_gid else 0

    # slot → talks
    slot_talks = defaultdict(list)
    for t in conf["talks"]:
        slot_talks[t["slot_id"]].append(t["id"])

    # real choices: slot → talk → list of user_ids
    slot_real = defaultdict(lambda: defaultdict(list))
    for r in raw:
        for uid in r["yes_user_ids"]:
            slot_real[r["slot_id"]][r["talk_id"]].append(uid)

    # Классификация
    counts = Counter()
    n_slots_evaluated = 0
    n_slots_with_tie = 0  # relevance_argmax == popularity_argmax
    detail_by_slot = []

    for sid, real_map in slot_real.items():
        slot_tids = slot_talks[sid]
        if len(slot_tids) < 2:
            continue
        # Активные пользователи: те, кого знаем + RSVPили yes на одно из событий
        active = []
        for tid, uids in real_map.items():
            for uid in uids:
                key = f"mu_{uid}"
                if key in user_emb:
                    active.append((uid, key, tid))
        if not active:
            continue
        n_slots_evaluated += 1

        # popularity_argmax — общий для слота
        pops = [(tid, talk_pop[tid]) for tid in slot_tids]
        pop_arg = max(pops, key=lambda x: x[1])[0]

        slot_compliant = slot_starchaser = slot_curious = 0
        slot_relevance_only = 0
        slot_pop_only = 0
        for uid, key, real_tid in active:
            ue = user_emb[key]
            scores = [(tid, float(np.dot(ue, talk_emb[tid]))) for tid in slot_tids]
            rel_arg = max(scores, key=lambda x: x[1])[0]

            is_rel = (real_tid == rel_arg)
            is_pop = (real_tid == pop_arg)
            tie = (rel_arg == pop_arg)

            if tie:
                if is_rel:  # совпали оба, нельзя разделить — учитываем как "rel/pop_tie"
                    counts["tie_match"] += 1
                else:
                    counts["tie_other"] += 1
            else:
                if is_rel and not is_pop:
                    counts["compliant_only"] += 1
                    slot_compliant += 1
                    slot_relevance_only += 1
                elif is_pop and not is_rel:
                    counts["starchaser_only"] += 1
                    slot_starchaser += 1
                    slot_pop_only += 1
                elif is_rel and is_pop:
                    # этого не будет (мы в ветке tie=False)
                    counts["both"] += 1
                else:
                    counts["curious"] += 1
                    slot_curious += 1
        if rel_arg == pop_arg:
            n_slots_with_tie += 1
        detail_by_slot.append({
            "slot_id": sid,
            "n_active": len(active),
            "rel_argmax": rel_arg,
            "pop_argmax": pop_arg,
            "tie": (rel_arg == pop_arg),
            "compliant_only": slot_compliant,
            "starchaser_only": slot_starchaser,
            "curious": slot_curious,
        })

    total_decided = (counts["compliant_only"] + counts["starchaser_only"]
                     + counts["curious"])
    total_tie = counts["tie_match"] + counts["tie_other"]
    total = total_decided + total_tie

    print(f"\n=== Calibration of compliance on Meetup RSVPs ===")
    print(f"  slots evaluated:          {n_slots_evaluated}")
    print(f"  slots with rel == pop tie:{n_slots_with_tie} "
          f"({n_slots_with_tie/max(1,n_slots_evaluated)*100:.1f}%)")
    print(f"\n  total user×slot pairs:    {total}")
    print(f"  pairs in tie-slots:       {total_tie} ({total_tie/max(1,total)*100:.1f}%)")
    print(f"  pairs in non-tie-slots:   {total_decided}")
    print(f"\n  Among non-tie-slots:")
    if total_decided > 0:
        a = counts["compliant_only"] / total_decided
        c = counts["starchaser_only"] / total_decided
        b = counts["curious"] / total_decided
        print(f"    α_A (compliant)    = {a:.3f}  ({counts['compliant_only']} / {total_decided})")
        print(f"    α_C (star-chaser)  = {c:.3f}  ({counts['starchaser_only']} / {total_decided})")
        print(f"    α_B (curious)      = {b:.3f}  ({counts['curious']} / {total_decided})")
    print(f"\n  All pairs (treating ties as 'either compliant or starchaser'):")
    if total > 0:
        # Если tie_match — пользователь выбрал rel-argmax и pop-argmax одновременно,
        # засчитываем половину в A, половину в C (равновесная гипотеза)
        a_all = (counts["compliant_only"] + counts["tie_match"] / 2) / total
        c_all = (counts["starchaser_only"] + counts["tie_match"] / 2) / total
        b_all = (counts["curious"] + counts["tie_other"]) / total
        print(f"    α_A_all = {a_all:.3f}")
        print(f"    α_C_all = {c_all:.3f}")
        print(f"    α_B_all = {b_all:.3f}")

    # Random baseline для контекста: при K=средний размер слота
    avg_slot_size = np.mean([len(slot_talks[s]) for s in slot_real if len(slot_talks[s]) >= 2])
    print(f"\n  avg parallel events per slot: {avg_slot_size:.2f}")
    print(f"  random baseline (any choice == argmax): {1/avg_slot_size:.3f}")

    out = {
        "n_slots_evaluated": n_slots_evaluated,
        "n_slots_with_tie": n_slots_with_tie,
        "n_pairs_total": total,
        "n_pairs_tie": total_tie,
        "n_pairs_decided": total_decided,
        "counts": dict(counts),
        "alpha_decided": {
            "A_compliant": counts["compliant_only"] / max(1, total_decided),
            "C_starchaser": counts["starchaser_only"] / max(1, total_decided),
            "B_curious": counts["curious"] / max(1, total_decided),
        },
        "alpha_all": {
            "A_compliant": (counts["compliant_only"] + counts["tie_match"] / 2) / max(1, total),
            "C_starchaser": (counts["starchaser_only"] + counts["tie_match"] / 2) / max(1, total),
            "B_curious": (counts["curious"] + counts["tie_other"]) / max(1, total),
        },
        "avg_slot_size": float(avg_slot_size),
        "random_baseline": float(1 / avg_slot_size),
    }
    out_path = ROOT / "results" / "compliance_calibration_meetup.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")


if __name__ == "__main__":
    main()
