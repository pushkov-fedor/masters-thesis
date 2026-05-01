"""B1: внешняя валидация параметрического симулятора на реальных Meetup RSVPs.

Постановка:
- На датасете Meetup RSVP знаем для каждого пользователя факт его RSVP "yes" на
  конкретное событие (talk).
- Берём слоты, где ≥ 2 параллельных событий с rsvp_limit и есть ≥ 1 реальный
  пользователь с yes-RSVP на одно из них.
- Для каждого active user в каждом таком слоте симулируем его выбор по
  параметрической модели Главы 3:

      score(user, event) = (1 - w_fame) * cos(user, event) + w_fame * fame
                        - λ * max(0, load(hall(event)) - 0.85)
      P(event | user) ∝ exp(score / τ),  плюс альтернатива skip с p_skip_base

  (load в этом эксперименте = 0 для всех залов — нам нужен «чистый» сигнал
  модели предпочтения; capacity-эффект изучается отдельно через политики)

- Сравниваем предсказанное распределение выбора с реальным.

Метрики:
- Accuracy@1: argmax по симулятору совпадает с реальным выбором (per user, per slot).
- Jensen-Shannon divergence между реальным и предсказанным распределением выбора
  внутри слота (per slot, потом усреднение).
- Spearman между real_count[event] и simulated_expected_count[event] по слоту.

Выход:
- experiments/results/sim_validation_meetup.json — числа per-slot и aggregate.
- experiments/results/summary_sim_validation_meetup.md — markdown.

B5 (калибровка τ, λ) — расширенный режим через --grid: считается JS на сетке
τ ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} × λ ∈ {0, 0.5, 1.0, 2.0, 4.0}, выбирается
аргмин average JS.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_meetup():
    conf_path = ROOT / "data" / "conferences" / "meetup_rsvp.json"
    emb_path = ROOT / "data" / "conferences" / "meetup_rsvp_embeddings.npz"
    raw_path = ROOT / "data" / "conferences" / "meetup_rsvp_raw_choices.json"
    pers_path = ROOT / "data" / "personas" / "meetup_users.json"
    pers_emb = ROOT / "data" / "personas" / "meetup_users_embeddings.npz"

    conf = json.load(open(conf_path))
    emb = np.load(emb_path)
    talk_emb = {tid: emb["embeddings"][i] for i, tid in enumerate(emb["ids"])}

    raw = json.load(open(raw_path))
    talk_meta = {t["id"]: t for t in conf["talks"]}
    halls = {h["id"]: h["capacity"] for h in conf["halls"]}

    pers_meta = json.load(open(pers_path))
    pemb = np.load(pers_emb)
    user_emb = {pid: pemb["embeddings"][i] for i, pid in enumerate(pemb["ids"])}

    # raw_choices: список словарей {talk_id, slot_id, yes_user_ids, hall_id, group_id}
    # построим slot_id -> {talk_id: [user_id, ...]}
    slot_real = defaultdict(lambda: defaultdict(list))
    for r in raw:
        sid = r["slot_id"]
        for uid in r["yes_user_ids"]:
            slot_real[sid][r["talk_id"]].append(uid)

    # slot_id -> talks
    slot_talks = defaultdict(list)
    for t in conf["talks"]:
        slot_talks[t["slot_id"]].append(t["id"])

    return {
        "conf": conf,
        "talk_emb": talk_emb,
        "talk_meta": talk_meta,
        "halls": halls,
        "user_emb": user_emb,
        "slot_real": slot_real,
        "slot_talks": slot_talks,
    }


def predict_probs(user_emb, talk_ids, talk_emb_map, halls, talk_meta,
                  tau, lambda_overflow, p_skip, w_fame=0.0,
                  hall_loads=None):
    """Вернёт вероятности выбора (включая последнюю — skip) для одного пользователя."""
    if hall_loads is None:
        hall_loads = {}
    utils = []
    for tid in talk_ids:
        t_emb = talk_emb_map[tid]
        cos = float(np.dot(user_emb, t_emb))
        t_meta = talk_meta[tid]
        hall_id = t_meta["hall"]
        cap = halls[hall_id]
        load_frac = hall_loads.get(hall_id, 0) / max(1.0, cap)
        eff_rel = (1 - w_fame) * cos + w_fame * t_meta.get("fame", 0.0)
        u = eff_rel - lambda_overflow * max(0.0, load_frac - 0.85)
        utils.append(u)
    utils = np.array(utils, dtype=np.float64)
    scaled = utils / max(tau, 1e-6)
    scaled -= scaled.max()
    exps = np.exp(scaled)
    s = exps.sum()
    rec_probs = (1.0 - p_skip) * exps / s
    full = np.concatenate([rec_probs, [p_skip]])
    return full / full.sum()


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate(data, tau, lambda_overflow, p_skip):
    """Один прогон на фиксированных гиперпараметрах. Возвращает агрегированные метрики."""
    correct = 0
    total = 0
    js_per_slot = []
    spearman_pairs = []  # (real_counts_arr, predicted_counts_arr)

    user_emb = data["user_emb"]
    talk_emb = data["talk_emb"]
    halls = data["halls"]
    talk_meta = data["talk_meta"]

    for sid, real_map in data["slot_real"].items():
        slot_tids = data["slot_talks"][sid]
        if len(slot_tids) < 2:
            continue
        # реальные пользователи: те, кого знаем в нашем embedding
        active_users = []
        real_choice = {}  # uid -> tid
        for tid, uids in real_map.items():
            for uid in uids:
                key = f"mu_{uid}"
                if key in user_emb:
                    active_users.append(key)
                    real_choice[key] = tid
        # уникализация: пользователь может RSVPить на несколько событий — берём все
        if not active_users:
            continue

        # Предсказанные распределения для каждого активного пользователя
        sim_count = np.zeros(len(slot_tids), dtype=np.float64)  # ожидаемое число выборов
        real_count = np.zeros(len(slot_tids), dtype=np.int64)
        tid_index = {tid: i for i, tid in enumerate(slot_tids)}

        for uid in active_users:
            ue = user_emb[uid]
            probs = predict_probs(ue, slot_tids, talk_emb, halls, talk_meta,
                                  tau=tau, lambda_overflow=lambda_overflow, p_skip=p_skip)
            # последний элемент — skip
            probs_no_skip = probs[:-1]
            # accuracy@1: argmax совпадает с реальным выбором
            chosen_idx = int(np.argmax(probs_no_skip))
            if slot_tids[chosen_idx] == real_choice[uid]:
                correct += 1
            total += 1
            sim_count += probs_no_skip
            real_count[tid_index[real_choice[uid]]] += 1

        # JS на распределении по событиям слота
        if real_count.sum() > 0 and sim_count.sum() > 0:
            r_dist = real_count / real_count.sum()
            s_dist = sim_count / sim_count.sum()
            js_per_slot.append(js_divergence(r_dist, s_dist))
            spearman_pairs.append((real_count.copy(), sim_count.copy()))

    # Spearman агрегированный (по всем слотам флэтом)
    from scipy.stats import spearmanr
    if spearman_pairs:
        all_real = np.concatenate([p[0] for p in spearman_pairs])
        all_sim = np.concatenate([p[1] for p in spearman_pairs])
        rho, pv = spearmanr(all_real, all_sim)
    else:
        rho, pv = float("nan"), float("nan")

    return {
        "tau": tau,
        "lambda_overflow": lambda_overflow,
        "p_skip": p_skip,
        "n_user_slot_pairs": total,
        "accuracy_at_1": correct / max(1, total),
        "js_mean": float(np.mean(js_per_slot)) if js_per_slot else float("nan"),
        "js_median": float(np.median(js_per_slot)) if js_per_slot else float("nan"),
        "n_slots_evaluated": len(js_per_slot),
        "spearman_count_rho": float(rho),
        "spearman_count_p": float(pv),
    }


def random_baseline(data):
    """Базовая accuracy@1 для случайной политики: 1 / |slot_tids|."""
    sums = []
    weights = []
    for sid, real_map in data["slot_real"].items():
        slot_tids = data["slot_talks"][sid]
        if len(slot_tids) < 2:
            continue
        total = sum(len(uids) for uids in real_map.values())
        # exclude users not in user_emb
        n = 0
        for tid, uids in real_map.items():
            for uid in uids:
                if f"mu_{uid}" in data["user_emb"]:
                    n += 1
        if n == 0:
            continue
        sums.append(1.0 / len(slot_tids))
        weights.append(n)
    if not weights:
        return float("nan")
    return float(np.average(sums, weights=weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--lambda-overflow", type=float, default=2.0)
    ap.add_argument("--p-skip", type=float, default=0.05)
    ap.add_argument("--grid", action="store_true",
                    help="Запустить сетку τ × λ для калибровки (B5)")
    args = ap.parse_args()

    print("Loading Meetup data...")
    data = load_meetup()
    print(f"  slots with real yes-RSVPs: {len(data['slot_real'])}")
    print(f"  users with embeddings: {len(data['user_emb'])}")

    rb = random_baseline(data)
    print(f"\nBaseline accuracy@1 (uniform random over slot events): {rb:.4f}")

    if args.grid:
        print("\n=== B5: grid τ × λ ===")
        results = []
        taus = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        lams = [0.0, 0.5, 1.0, 2.0, 4.0]
        for tau in taus:
            for lam in lams:
                m = evaluate(data, tau=tau, lambda_overflow=lam, p_skip=args.p_skip)
                m["random_baseline"] = rb
                results.append(m)
                print(f"  τ={tau:>4}  λ={lam:>4}  acc@1={m['accuracy_at_1']:.4f}  "
                      f"JS_mean={m['js_mean']:.4f}  ρ={m['spearman_count_rho']:+.3f}")
        # Лучшая точка по JS
        best = min(results, key=lambda r: r["js_mean"])
        best_acc = max(results, key=lambda r: r["accuracy_at_1"])
        print(f"\nMin JS_mean:    τ={best['tau']}, λ={best['lambda_overflow']}, "
              f"JS={best['js_mean']:.4f}, acc@1={best['accuracy_at_1']:.4f}")
        print(f"Max accuracy@1: τ={best_acc['tau']}, λ={best_acc['lambda_overflow']}, "
              f"acc@1={best_acc['accuracy_at_1']:.4f}, JS={best_acc['js_mean']:.4f}")
        out = {"baseline_random": rb, "grid": results, "best_by_js": best,
               "best_by_acc": best_acc}
    else:
        m = evaluate(data, tau=args.tau, lambda_overflow=args.lambda_overflow,
                     p_skip=args.p_skip)
        m["random_baseline"] = rb
        print(f"\n=== Single point: τ={args.tau}, λ={args.lambda_overflow}, "
              f"p_skip={args.p_skip} ===")
        print(f"  n_user_slot_pairs: {m['n_user_slot_pairs']}")
        print(f"  n_slots_evaluated: {m['n_slots_evaluated']}")
        print(f"  accuracy@1: {m['accuracy_at_1']:.4f} (random baseline {rb:.4f})")
        print(f"  JS_mean:    {m['js_mean']:.4f}")
        print(f"  JS_median:  {m['js_median']:.4f}")
        print(f"  Spearman ρ: {m['spearman_count_rho']:+.3f} (p={m['spearman_count_p']:.4f})")
        out = m

    out_path = ROOT / "results" / ("sim_validation_meetup_grid.json" if args.grid
                                    else "sim_validation_meetup.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWROTE: {out_path}")


if __name__ == "__main__":
    main()
