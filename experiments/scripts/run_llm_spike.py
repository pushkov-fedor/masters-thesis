"""Ранний LLM-spike (этап H PIVOT_IMPLEMENTATION_PLAN r5).

Реализация строго по принятому memo G (`docs/spikes/spike_llm_simulator.md`).

Состав:
- агент V3 status quo: profile + history + recommendation; hall_loads_pct в промпт
  не передаётся (см. accepted decision этапа C, инвариант spike поведения);
- toy-конференция `toy_microconf_2slot` (2 слота × 2 зала × 2 доклада в каждом
  слоте), отдельный файл данных от `toy_microconf` этапов D–F;
- 10 агентов из `personas_100.json`, отбор детерминированный и разнообразный
  через KMeans(k=10) по эмбедингам + ближайший к центру кластера → 10 ID;
- inline политики `no_policy` + `cosine` (K=1); активный реестр П1–П4 в
  LLM-симуляторе выравнивается на этапе V, не здесь;
- hard cap $5: при достижении прогон завершается со статусом `budget_exceeded`,
  оставшиеся решения **не превращаются в skip** (см. memo G §9.2.1);
- результат пишется в smoke-совместимом каноне (`etap / conference / status /
  params / results: [{...}]`) с каноническими именами метрик в `agg`.

Запуск:
    .venv/bin/python scripts/run_llm_spike.py
    .venv/bin/python scripts/run_llm_spike.py --model openai/gpt-5.4-mini --budget-cap 5.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import dotenv_values
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_agent import LLMAgent

ENV_CANDIDATES = [ROOT.parent / ".env", ROOT.parent.parent / "party-of-one" / ".env"]


PRICING = {
    # $ / Mtok (prompt, completion); приблизительно по OpenRouter
    "openai/gpt-5.4-mini":         (0.10, 0.40),
    "anthropic/claude-haiku-4.5":  (1.00, 5.00),
    "deepseek/deepseek-v3.2-exp":  (0.27, 0.41),
    "openai/gpt-4o-mini":          (0.15, 0.60),
}


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            k = cfg.get("OPENROUTER_API_KEY")
            if k:
                return k
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


# === Детерминированный разнообразный отбор персон через k-means ===

def select_personas_kmeans(
    pers_ids: List[str],
    pers_embs: np.ndarray,
    k: int = 10,
    random_state: int = 42,
) -> Tuple[List[int], str]:
    """Возвращает индексы k разнообразных персон через KMeans.

    Из каждого кластера выбирается персона, ближайшая к центру (по cosine,
    эмбединги нормализованы). Порядок результата — по индексу кластера, что
    делает выбор детерминированным.
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(pers_embs)
    selected: List[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        center = km.cluster_centers_[c]
        # cosine эквивалент max(p_emb @ center / |center|), эмбединги нормализованы
        sims = pers_embs[members] @ center
        best = int(members[int(np.argmax(sims))])
        selected.append(best)
    method = f"kmeans_k{k}_seed{random_state}"
    return selected, method


# === Inline-политики (только для этапа H; реестр П1–П4 — этап V) ===

def policy_no_policy(*args, **kwargs) -> Optional[List[str]]:
    """П1: рекомендация не предъявляется."""
    return None


def policy_cosine(
    user_emb: np.ndarray,
    talk_embs: np.ndarray,
    talk_ids: List[str],
    K: int,
) -> List[str]:
    """П2: top-K по cosine релевантности."""
    sims = talk_embs @ user_emb
    order = np.argsort(sims)[::-1][:K]
    return [talk_ids[i] for i in order]


POLICIES = {
    "no_policy": policy_no_policy,
    "cosine":    policy_cosine,
}


# === Метрики (smoke-compatible имена) ===

def compute_metrics(
    decisions: List[Dict[str, Any]],
    slot_loads: Dict[str, Dict[int, int]],
    cap_per_slot_hall: Dict[Tuple[str, int], int],
    talks_by_slot: Dict[str, List[Dict[str, Any]]],
    talk_emb_map: Dict[str, np.ndarray],
    agent_emb_map: Dict[str, np.ndarray],
    cost_usd: float,
    n_parse_errors: int,
    n_decisions_aborted: int,
) -> Dict[str, Any]:
    """Считает агрегатные метрики по smoke-совместимому канону."""
    n_decisions = len(decisions)
    n_skipped = sum(1 for d in decisions if d["chosen"] is None)

    excesses: List[float] = []
    overflow_flags: List[int] = []
    hall_vars: List[float] = []
    for sid, talks in talks_by_slot.items():
        halls = sorted({t["hall"] for t in talks})
        if len(halls) < 1:
            continue
        utils = []
        for h in halls:
            cap = cap_per_slot_hall[(sid, h)]
            count = slot_loads.get(sid, {}).get(h, 0)
            u = count / cap if cap > 0 else 0.0
            utils.append(u)
            excesses.append(max(0.0, u - 1.0))
            overflow_flags.append(1 if u > 1.0 else 0)
        if len(halls) >= 2:
            hall_vars.append(float(np.var(utils)))

    mean_overload_excess = float(np.mean(excesses)) if excesses else 0.0
    overflow_rate_slothall = float(np.mean(overflow_flags)) if overflow_flags else 0.0
    hall_utilization_variance = float(np.mean(hall_vars)) if hall_vars else 0.0

    # mean_user_utility — cosine между chosen talk и agent profile
    utils = []
    for d in decisions:
        if d["chosen"] is None:
            continue
        if d["chosen"] in talk_emb_map and d["agent_id"] in agent_emb_map:
            sim = float(talk_emb_map[d["chosen"]] @ agent_emb_map[d["agent_id"]])
            utils.append(sim)
    mean_user_utility = float(np.mean(utils)) if utils else 0.0

    return {
        "mean_overload_excess": mean_overload_excess,
        "hall_utilization_variance": hall_utilization_variance,
        "overflow_rate_slothall": overflow_rate_slothall,
        "n_decisions": n_decisions,
        "n_skipped": n_skipped,
        "n_parse_errors": n_parse_errors,
        "n_decisions_aborted": n_decisions_aborted,
        "cost_usd": float(cost_usd),
        "mean_user_utility": mean_user_utility,
    }


# === Прогон одной политики ===

def gossip_level_from_w(w_gossip: float) -> str:
    """Маппинг cfg.w_gossip → дискретный уровень L2 (Q-J8 accepted 2026-05-07).

    Граница 0.4: согласовано в spike_gossip_llm_amendment §6 и §8 Q-J8.
    """
    if w_gossip <= 0.0:
        return "off"
    if w_gossip < 0.4:
        return "moderate"
    return "strong"


async def run_one_policy(
    policy_name: str,
    agents: List[LLMAgent],
    conf: Dict[str, Any],
    talk_emb_map: Dict[str, np.ndarray],
    agent_emb_map: Dict[str, np.ndarray],
    cap_per_slot_hall: Dict[Tuple[str, int], int],
    K: int,
    llm_call,
    budget_cap: float,
    cumulative_cost_ref: List[float],
    seed: int,
    w_gossip: float = 0.0,
) -> Dict[str, Any]:
    """Прогон одной политики на свежих копиях агентов.

    cumulative_cost_ref — list-обёртка для shared mutable cost между политиками.
    w_gossip — управляет L2 LLM-gossip через дискретный уровень
    (off/moderate/strong, см. gossip_level_from_w).
    Возвращает dict с per_decision, slot_loads, метриками и status.
    """
    fresh_agents = [
        LLMAgent(agent_id=a.agent_id, profile=a.profile, history=[])
        for a in agents
    ]
    talks_by_id = {t["id"]: t for t in conf["talks"]}
    talks_by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in conf["talks"]:
        talks_by_slot[t["slot_id"]].append(t)
    hall_of_talk = {t["id"]: t["hall"] for t in conf["talks"]}

    decisions: List[Dict[str, Any]] = []
    slot_loads: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # L2 gossip: per-talk счётчик в текущем слоте (параллельный slot_loads
    # per-hall). Семантически gossip-канал — про «выбор по докладу», см.
    # spike_gossip §6 V5 / §8 и amendment §1.3.
    slot_choice_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    gossip_level = gossip_level_from_w(w_gossip)
    gossip_n_total = len(fresh_agents)
    n_parse_errors = 0
    n_decisions_aborted = 0
    status = "ok"
    rng = np.random.default_rng(seed)

    slots_in_order = sorted(conf["slots"], key=lambda s: s["id"])
    pbar = tqdm(
        total=len(fresh_agents) * len(slots_in_order),
        desc=f"[{policy_name:<10}]",
        ncols=120,
        leave=True,
        dynamic_ncols=False,
    )
    aborted = False
    for slot in slots_in_order:
        if aborted:
            break
        sid = slot["id"]
        slot_talks = talks_by_slot.get(sid, [])
        if not slot_talks:
            continue
        slot_talk_ids = [t["id"] for t in slot_talks]
        slot_talk_embs = np.stack([talk_emb_map[tid] for tid in slot_talk_ids])

        # Sequential по агентам в случайном порядке, но детерминированном для seed
        order = list(range(len(fresh_agents)))
        rng.shuffle(order)

        for ai in order:
            # Пред-проверка бюджета: если уже исчерпали — прерываем
            if cumulative_cost_ref[0] >= budget_cap:
                aborted = True
                status = "budget_exceeded"
                # сколько решений останется не сделано
                remaining_in_slot = (len(order) - order.index(ai))
                remaining_slots = (
                    len(slots_in_order) - slots_in_order.index(slot) - 1
                ) * len(fresh_agents)
                n_decisions_aborted = remaining_in_slot + remaining_slots
                break

            agent = fresh_agents[ai]
            user_emb = agent_emb_map[agent.agent_id]
            policy_fn = POLICIES[policy_name]
            if policy_name == "no_policy":
                rec = policy_fn()
            else:
                rec = policy_fn(user_emb, slot_talk_embs, slot_talk_ids, K)

            # gossip_counts формируется ДО вызова decide(): отражает выбор
            # первых N-1 агентов в текущем слоте (sequential causality, как и
            # в параметрическом ядре). При gossip_level='off' блок в промпт
            # не попадает (Q-J9 accepted).
            gossip_counts_now = (
                dict(slot_choice_count[sid]) if gossip_level != "off" else None
            )

            decision = await agent.decide(
                slot_id=sid,
                talks=slot_talks,
                hall_loads_pct={},  # capacity в промпт НЕ передаём (Q-G accepted)
                recommendation=rec,
                llm_call=llm_call,
                gossip_counts=gossip_counts_now,
                gossip_n_total=(gossip_n_total if gossip_level != "off" else None),
                gossip_level=gossip_level,
            )
            cumulative_cost_ref[0] += decision.cost_usd

            rec_record = {
                "agent_id": decision.agent_id,
                "slot_id": decision.slot_id,
                "chosen": decision.chosen,
                "reason": decision.reason,
                "cost_usd": decision.cost_usd,
                "recommended": rec,
            }
            decisions.append(rec_record)

            if decision.chosen is None:
                if "parse-error" in decision.reason or "invalid-choice" in decision.reason:
                    n_parse_errors += 1
            else:
                hall = hall_of_talk[decision.chosen]
                slot_loads[sid][hall] += 1
                slot_choice_count[sid][decision.chosen] += 1
                fresh_agents[ai].commit(sid, talks_by_id[decision.chosen])

            pbar.update(1)
            pbar.set_postfix({
                "cost": f"${cumulative_cost_ref[0]:.4f}",
                "skip": sum(1 for d in decisions if d["chosen"] is None),
                "perr": n_parse_errors,
            })

    pbar.close()

    metrics = compute_metrics(
        decisions=decisions,
        slot_loads={k: dict(v) for k, v in slot_loads.items()},
        cap_per_slot_hall=cap_per_slot_hall,
        talks_by_slot=dict(talks_by_slot),
        talk_emb_map=talk_emb_map,
        agent_emb_map=agent_emb_map,
        cost_usd=sum(d["cost_usd"] for d in decisions),
        n_parse_errors=n_parse_errors,
        n_decisions_aborted=n_decisions_aborted,
    )
    return {
        "status": status,
        "per_decision": decisions,
        "slot_loads": {k: dict(v) for k, v in slot_loads.items()},
        "agg": metrics,
    }


# === Main ===

async def main_async(args: argparse.Namespace) -> int:
    print(f"Loading conference {args.conference}...", flush=True)
    conf_path = ROOT / "data" / "conferences" / f"{args.conference}.json"
    emb_path = ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz"
    conf = json.load(open(conf_path, encoding="utf-8"))
    emb = np.load(emb_path)
    talk_emb_map = {tid: emb["embeddings"][i] for i, tid in enumerate(emb["ids"].tolist())}

    # Capacity = ceil(N / halls_in_slot) — стресс по умолчанию
    talks_by_slot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in conf["talks"]:
        talks_by_slot[t["slot_id"]].append(t)
    cap_per_slot_hall: Dict[Tuple[str, int], int] = {}
    for sid, talks in talks_by_slot.items():
        halls = sorted({t["hall"] for t in talks})
        cap = math.ceil(args.n / max(1, len(halls)))
        for h in halls:
            cap_per_slot_hall[(sid, h)] = cap
    print(
        f"  capacity per slot×hall = ceil({args.n} / halls_in_slot) "
        f"= {next(iter(cap_per_slot_hall.values()))}",
        flush=True,
    )

    # Personas + эмбединги, отбор k-means
    pers_path = ROOT / "data" / "personas" / f"{args.personas}.json"
    pers_emb_path = ROOT / "data" / "personas" / f"{args.personas}_embeddings.npz"
    pers_all = json.load(open(pers_path, encoding="utf-8"))
    pers_emb_npz = np.load(pers_emb_path)
    pers_ids_all = pers_emb_npz["ids"].tolist()
    pers_embs_all = pers_emb_npz["embeddings"]

    selected_idx, selection_method = select_personas_kmeans(
        pers_ids=pers_ids_all,
        pers_embs=pers_embs_all,
        k=args.n,
        random_state=42,
    )
    pers_id_to_record = {p["id"]: p for p in pers_all}
    pers_id_to_emb = {pid: pers_embs_all[i] for i, pid in enumerate(pers_ids_all)}
    selected_ids = [pers_ids_all[i] for i in selected_idx]
    print(f"Selected {len(selected_ids)} personas via {selection_method}:", flush=True)
    for pid in selected_ids:
        rec = pers_id_to_record[pid]
        role = rec.get("role", "?")
        print(f"  - {pid}: {role}", flush=True)

    agents = [
        LLMAgent(agent_id=pid, profile=pers_id_to_record[pid].get("background", ""))
        for pid in selected_ids
    ]
    agent_emb_map = {pid: pers_id_to_emb[pid] for pid in selected_ids}

    # LLM client
    api_key = load_api_key()
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    p_in, p_out = PRICING.get(args.model, (1.0, 5.0))
    sem = asyncio.Semaphore(args.concurrency)

    async def llm_call(system, user, max_tokens=200):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                    timeout=60,
                )
            except Exception as e:
                return f'{{"choice":"skip","reason":"api-error: {e}"}}', 0.0
        msg = resp.choices[0].message.content or ""
        u = resp.usage
        cost = 0.0
        if u is not None:
            cost = (u.prompt_tokens / 1e6) * p_in + (u.completion_tokens / 1e6) * p_out
        return msg, cost

    policies = [p for p in args.policies.split(",") if p in POLICIES]
    w_gossip_grid = [float(x) for x in args.w_gossip.split(",")]
    print(f"\n=== Running {len(policies)} policies × "
          f"{len(w_gossip_grid)} w_gossip points: "
          f"policies={policies}, w_gossip={w_gossip_grid} ===",
          flush=True)
    print(f"  budget_cap = ${args.budget_cap}", flush=True)
    t0_global = time.time()

    results: List[Dict[str, Any]] = []
    cumulative_cost_ref = [0.0]
    overall_status = "ok"
    aborted_outer = False
    for w_g in w_gossip_grid:
        if aborted_outer:
            break
        for pol in policies:
            t0 = time.time()
            per_pol = await run_one_policy(
                policy_name=pol,
                agents=agents,
                conf=conf,
                talk_emb_map=talk_emb_map,
                agent_emb_map=agent_emb_map,
                cap_per_slot_hall=cap_per_slot_hall,
                K=args.K,
                llm_call=llm_call,
                budget_cap=args.budget_cap,
                cumulative_cost_ref=cumulative_cost_ref,
                seed=args.seed,
                w_gossip=w_g,
            )
            elapsed = time.time() - t0
            print(
                f"[{pol} | w_gossip={w_g}] done in {elapsed:.0f}s. "
                f"status={per_pol['status']} agg={per_pol['agg']}",
                flush=True,
            )
            results.append({
                "capacity_scenario": "natural",
                "policy": pol,
                "w_gossip": w_g,
                "w_rec": 1.0,
                "seed": args.seed,
                "per_decision_status": per_pol["status"],
                "agg": per_pol["agg"],
                "slot_loads": per_pol["slot_loads"],
                "per_decision": per_pol["per_decision"],
            })
            if per_pol["status"] != "ok":
                overall_status = per_pol["status"]
                aborted_outer = True
                # дальнейшие комбинации не запускаем — бюджет исчерпан
                break

    elapsed_total_s = time.time() - t0_global
    out = {
        "etap": "H",
        "conference": args.conference,
        "status": overall_status,
        "elapsed_total_s": elapsed_total_s,
        "n_decisions_aborted": sum(
            r["agg"]["n_decisions_aborted"] for r in results
        ),
        "params": {
            "date": time.strftime("%Y-%m-%d"),
            "n_agents": args.n,
            "n_slots": len(conf["slots"]),
            "K": args.K,
            "model": args.model,
            "seeds": [args.seed],
            "policies": policies,
            "w_rec_values": [1.0],
            "w_gossip_values": w_gossip_grid,
            "capacity_scenarios": ["natural"],
            "personas_source": args.personas,
            "personas_selection": selection_method,
            "selected_persona_ids": selected_ids,
            "budget_cap_usd": args.budget_cap,
        },
        "results": results,
    }

    # Запись JSON-результата
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = ROOT / "results" / f"llm_spike_{time.strftime('%Y_%m_%d')}{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {out_path}")

    # Краткая сводка
    print("\n=== Сводка ===")
    print(
        f"{'policy':<12} {'w_g':<5} {'overload':<10} {'hall_var':<10} "
        f"{'overflow':<10} {'skip':<6} {'perr':<5} {'cost':<8}"
    )
    for r in results:
        a = r["agg"]
        print(
            f"{r['policy']:<12} "
            f"{r['w_gossip']:<5.2f} "
            f"{a['mean_overload_excess']:<10.4f} "
            f"{a['hall_utilization_variance']:<10.4f} "
            f"{a['overflow_rate_slothall']:<10.4f} "
            f"{a['n_skipped']:<6d} "
            f"{a['n_parse_errors']:<5d} "
            f"${a['cost_usd']:<6.4f}"
        )
    print(f"\nstatus = {overall_status}; elapsed = {elapsed_total_s:.1f}s; "
          f"cum_cost = ${cumulative_cost_ref[0]:.4f}")

    if overall_status != "ok":
        print(
            "\n[!] прогон неполный. Acceptance этапа H не считается пройденным "
            "(см. memo G §9.2.1, §9.5).",
            flush=True,
        )
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description="Ранний LLM-spike (этап H)")
    ap.add_argument("--conference", default="toy_microconf_2slot")
    ap.add_argument("--personas", default="personas_100")
    ap.add_argument("--n", type=int, default=10, help="число агентов")
    ap.add_argument("--K", type=int, default=1, help="top-K рекомендаций")
    ap.add_argument(
        "--policies",
        default="no_policy,cosine",
        help="comma-separated; допустимы no_policy,cosine",
    )
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="общий семафор LLM API (in-flight calls limit)")
    ap.add_argument("--budget-cap", type=float, default=5.0,
                    help="hard cap на суммарную стоимость, $")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--w-gossip", default="0.0",
                    help="comma-separated w_gossip values; for stage L verification "
                         "use '0.0,0.5'. Discrete LLM-gossip levels: "
                         "off (w=0), moderate (0<w<0.4), strong (w>=0.4).")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
