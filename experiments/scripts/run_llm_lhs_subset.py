"""Этап V: LLM-симулятор на 12 maximin LHS-точках.

Цель: прогнать LLM-симулятор на тех же 12 maximin LHS-точках, что и
параметрический симулятор в Q, с теми же CRN-сидами / program_variant /
audience_size / w_gossip. Внутри каждой LHS-точки — все 4 политики на
одной и той же аудитории (CRN внутри точки), 1 seed на eval (replicate=1).

ИНВАРИАНТЫ:

1. **Sequential within slot** — agent decisions внутри одного слота идут
   СТРОГО последовательно. Это нельзя параллелить, потому что L2 gossip
   формируется из счётчика выборов уже сделанных агентов в этом слоте.
   Concurrency используется только МЕЖДУ независимыми (LHS-row, policy)
   парами через `asyncio.gather` для параллельного прогона 4 политик
   на одной LHS-row.

2. **Q/S read-only** — этот скрипт читает Q JSON и S артефакты, но не
   пишет в них. Sha256 проверяется в smoke-отчёте.

3. **CRN-paritет с Q** — `derive_seeds(lhs_row_id, replicate=1)`
   гарантирует ту же `audience_seed`/`phi_seed`, что и параметрический
   прогон. cfg_seed=1 обеспечивает максимальное переиспользование
   `llm_ranker_cache.json` от Q.

4. **Раздельный учёт стоимости и времени:**
   - LLMAgent — каждый decide() = новый API call;
   - LLMRankerPolicy — sync, на cache имеет отдельный cumulative_cost,
     n_api_calls, n_cache_hits.

Запуск:
    .venv/bin/python scripts/run_llm_lhs_subset.py --smoke
    .venv/bin/python scripts/run_llm_lhs_subset.py --budget-cap 20 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import httpx
from dotenv import dotenv_values
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_agent import LLMAgent  # noqa: E402
from src.policies.capacity_aware_policy import CapacityAwarePolicy  # noqa: E402
from src.policies.cosine_policy import CosinePolicy  # noqa: E402
from src.policies.no_policy import NoPolicy  # noqa: E402
from src.program_modification import (  # noqa: E402
    SwapDescriptor,
    _apply_swap,
    enumerate_modifications,
)
from src.seeds import derive_seeds  # noqa: E402
from src.simulator import Conference, UserProfile  # noqa: E402

# ---------- Константы ----------

DEFAULT_LHS_INPUT = (
    "experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.json"
)
DEFAULT_OUTPUT_PREFIX = "experiments/results/llm_agents_lhs_subset_12pts"
# Default = gpt-5.4-nano: $0.20/$1.25 per Mtok, та же OpenAI-семья и
# токенайзер что у gpt-5.4-mini. Subagent live-тесты 2026-05-08 показали:
# 4/4 pass, чистый JSON, latency 0.92–1.29s стабильно. КРИТИЧНО передавать
# reasoning.enabled=False (см. llm_call ниже): иначе биллятся reasoning
# tokens и ломается latency. До 2026-05-08 было gpt-5.4-mini ($0.75/$4.50)
# — не помещался в cap $20 (см. два прерванных прогона).
DEFAULT_MODEL = "openai/gpt-5.4-nano"
DEFAULT_BUDGET_CAP = 20.0
# concurrency=16 + parallel-lhs=4 (16 streams) — vanilla full режим.
# Зависания 2026-05-08 устранены тремя фиксами одновременно:
# (1) per-call asyncio.wait_for(timeout=90) — hard cancel hung calls;
# (2) AsyncOpenAI(max_retries=0) — отключение SDK auto-retry, который
#     создавал retry-loop для всех 16 streams при transient 5xx;
# (3) убран custom httpx.AsyncClient — SDK сам управляет pool.
DEFAULT_CONCURRENCY = 16
DEFAULT_K = 3
DEFAULT_CFG_SEED = 1
DEFAULT_PERSONAS = "personas_100"
DEFAULT_CONFERENCE = "mobius_2025_autumn"

POLICY_NAMES = ("no_policy", "cosine", "capacity_aware", "llm_ranker")

POP_SRC_TO_W_FAME = {
    "cosine_only": 0.0,
    "fame_only": 1.0,
    "mixed": 0.3,
}

# Pricing per Mtok: (prompt, completion). Источник — OpenRouter actual rates
# по dashboard на 2026-05-08. ВНИМАНИЕ: gpt-5.4-mini ставки 0.75/4.50 (не
# 0.10/0.40 как было в старых скриптах). Перед использованием другой модели
# проверять dashboard.
PRICING = {
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "x-ai/grok-4-fast": (0.20, 0.50),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.41),
}

ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]


# ---------- Утилиты ----------

def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            k = cfg.get("OPENROUTER_API_KEY")
            if k:
                return k
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


def gossip_level_from_w(w_gossip: float) -> str:
    """w_gossip → дискретный уровень L2 (Q-J8 accepted).

    Граница 0.4 согласована с spike_gossip_llm_amendment §6 и §8.
    Идентичная функция в run_llm_spike.py — здесь дублируется ради
    самодостаточности скрипта V.
    """
    if w_gossip <= 0.0:
        return "off"
    if w_gossip < 0.4:
        return "moderate"
    return "strong"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def select_audience(users: list[UserProfile], audience_size: int,
                    audience_seed: int) -> list[UserProfile]:
    """Идентичная функция из run_lhs_parametric.select_audience.

    rng по audience_seed → одинаковый набор персон для всех политик
    в одной LHS-точке. CRN-инвариант.
    """
    rng = np.random.default_rng(audience_seed)
    indices = rng.choice(len(users), size=audience_size, replace=False)
    return [users[int(i)] for i in indices]


def apply_program_variant_with_q_descriptor(
    base_conf: Conference,
    program_variant: int,
    q_swap_descriptor: dict | None,
    phi_seed: int,
    k_max: int = 5,
) -> tuple[Conference, dict]:
    """Применяет program_variant ИДЕНТИЧНО Q.

    Если в Q-артефакте есть `swap_descriptor` для этого LHS-row — используем
    его напрямую (это гарантирует bit-exact paritet с Q). Иначе fallback
    к `enumerate_modifications` по тому же phi_seed (то же поведение, что
    в `run_lhs_parametric.build_program_variant`).
    """
    if program_variant == 0 or q_swap_descriptor is None:
        return base_conf, {"program_variant": program_variant,
                            "swap_descriptor": None,
                            "fallback_to_p0": program_variant != 0}
    desc = SwapDescriptor(
        slot_a=q_swap_descriptor["slot_a"],
        slot_b=q_swap_descriptor["slot_b"],
        t1=q_swap_descriptor["t1"],
        t2=q_swap_descriptor["t2"],
    )
    modified = _apply_swap(base_conf, desc)
    return modified, {
        "program_variant": program_variant,
        "swap_descriptor": q_swap_descriptor,
        "fallback_to_p0": False,
    }


def scale_capacity(conf: Conference, mult: float) -> Conference:
    """Масштабирует capacity_multiplier ИДЕНТИЧНО Q."""
    import copy
    cloned = copy.deepcopy(conf)
    for h in cloned.halls.values():
        h.capacity = max(1, int(round(h.capacity * mult)))
    for s in cloned.slots:
        if s.hall_capacities:
            s.hall_capacities = {
                hid: max(1, int(round(c * mult)))
                for hid, c in s.hall_capacities.items()
            }
    return cloned


def cosine_topk_recommendation(user_emb, slot_talk_ids, talk_emb_map, K):
    embs = np.stack([talk_emb_map[tid] for tid in slot_talk_ids])
    sims = embs @ user_emb
    order = np.argsort(sims)[::-1][:K]
    return [slot_talk_ids[i] for i in order]


# ---------- Метрики (smoke-compatible) ----------

def compute_metrics(decisions: list[dict],
                    slot_loads: dict, conf: Conference,
                    talk_emb_map: dict,
                    audience_emb_map: dict) -> dict:
    """Идентичные имена и формулы, что в Q (см. run_llm_spike.compute_metrics)."""
    n_decisions = len(decisions)
    n_skipped = sum(1 for d in decisions if d["chosen"] is None)

    excesses = []
    overflow_flags = []
    hall_vars = []
    for slot in conf.slots:
        sid = slot.id
        halls = sorted({conf.talks[tid].hall for tid in slot.talk_ids})
        if not halls:
            continue
        utils = []
        for h in halls:
            cap = conf.capacity_at(sid, h)
            count = slot_loads.get(sid, {}).get(h, 0)
            u = count / cap if cap > 0 else 0.0
            utils.append(u)
            excesses.append(max(0.0, u - 1.0))
            overflow_flags.append(1 if u > 1.0 else 0)
        if len(halls) >= 2:
            hall_vars.append(float(np.var(utils)))

    util_sims = []
    for d in decisions:
        if d["chosen"] is None:
            continue
        if d["chosen"] in talk_emb_map and d["agent_id"] in audience_emb_map:
            util_sims.append(
                float(talk_emb_map[d["chosen"]]
                      @ audience_emb_map[d["agent_id"]])
            )

    return {
        "metric_mean_overload_excess":
            float(np.mean(excesses)) if excesses else 0.0,
        "metric_overflow_rate_slothall":
            float(np.mean(overflow_flags)) if overflow_flags else 0.0,
        "metric_hall_utilization_variance":
            float(np.mean(hall_vars)) if hall_vars else 0.0,
        "metric_mean_user_utility":
            float(np.mean(util_sims)) if util_sims else 0.0,
        "metric_n_skipped": int(n_skipped),
        "metric_n_users": int(n_decisions),
    }


# ---------- Прогон одной (LHS-row, policy) пары ----------

async def run_lhs_policy(
    *,
    lhs_row: dict,
    policy_name: str,
    program_conf: Conference,
    audience: list[UserProfile],
    audience_emb_map: dict,
    talk_emb_map: dict,
    K: int,
    llm_call,
    cumulative_cost_ref: list[float],
    budget_cap: float,
    cfg_seed: int,
    capacity_aware_pol: CapacityAwarePolicy,
    llm_ranker_pol,  # Optional[LLMRankerPolicy]
    pbar: tqdm | None = None,
    language: str = "ru",
) -> dict:
    """Прогон одной политики на одной LHS-row (CRN-aware).

    INVARIANT: внутри слота агенты обрабатываются СТРОГО SEQUENTIAL.
    `slot_choice_count` обновляется после каждого decide → следующий
    агент видит обновлённый gossip-сигнал.
    """
    # Свежие копии агентов (история пустая — каждая политика начинает с нуля)
    agents = [LLMAgent(agent_id=u.id, profile=u.text, language=language)
              for u in audience]
    agent_emb_map = {u.id: u.embedding for u in audience}

    decisions: list[dict] = []
    slot_loads: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    slot_choice_count: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    n_parse_errors = 0
    n_decisions_aborted = 0
    status = "ok"

    rng = np.random.default_rng(cfg_seed)
    gossip_level = gossip_level_from_w(lhs_row["w_gossip"])
    gossip_n_total = len(agents)

    # Метрики времени по компонентам
    t_llmagent_total = 0.0
    t_recommendation_total = 0.0
    t_llmranker_total = 0.0  # подмножество t_recommendation_total

    # snapshot ranker counters перед политикой (для дельты)
    if llm_ranker_pol is not None:
        ranker_calls_before = int(llm_ranker_pol.n_api_calls)
        ranker_hits_before = int(llm_ranker_pol.n_cache_hits)
        ranker_cost_before = float(llm_ranker_pol.cumulative_cost)
    else:
        ranker_calls_before = 0
        ranker_hits_before = 0
        ranker_cost_before = 0.0

    aborted = False
    for slot in program_conf.slots:
        if aborted:
            break
        if not slot.talk_ids:
            continue
        slot_talk_ids = list(slot.talk_ids)
        slot_talks_meta = [
            {
                "id": tid,
                "title": program_conf.talks[tid].title,
                "hall": program_conf.talks[tid].hall,
                "category": program_conf.talks[tid].category,
                "abstract": program_conf.talks[tid].abstract,
            }
            for tid in slot_talk_ids
        ]

        # Sequential within slot — random order detrministic by cfg_seed
        order = list(range(len(agents)))
        rng.shuffle(order)

        for ai in order:
            if cumulative_cost_ref[0] >= budget_cap:
                aborted = True
                status = "budget_exceeded"
                break

            agent = agents[ai]
            user_profile = audience[ai]
            user_emb = agent_emb_map[agent.agent_id]

            # === recommendation ===
            t_rec_start = time.perf_counter()
            rec: list[str] | None = None
            if policy_name == "no_policy":
                rec = None
            elif policy_name == "cosine":
                rec = cosine_topk_recommendation(
                    user_emb, slot_talk_ids, talk_emb_map, K,
                )
            elif policy_name == "capacity_aware":
                # Передаём те же state-ключи, что и параметрический симулятор
                hall_load = {
                    (slot.id, h): slot_loads[slot.id].get(h, 0)
                    for h in {program_conf.talks[tid].hall
                              for tid in slot_talk_ids}
                }
                state = {
                    "K": K, "hall_load": hall_load,
                    "relevance_fn": lambda a, b: float(np.dot(a, b)),
                }
                rec = capacity_aware_pol(
                    user=user_profile, slot=slot, conf=program_conf, state=state,
                )
            elif policy_name == "llm_ranker":
                if llm_ranker_pol is None:
                    raise RuntimeError("llm_ranker_pol must be provided "
                                       "for policy=llm_ranker")
                t_ranker_start = time.perf_counter()
                state = {"K": K}
                rec = llm_ranker_pol(
                    user=user_profile, slot=slot, conf=program_conf, state=state,
                )
                t_llmranker_total += time.perf_counter() - t_ranker_start
            else:
                raise ValueError(f"unknown policy: {policy_name}")
            t_recommendation_total += time.perf_counter() - t_rec_start

            # === LLM agent decision ===
            gossip_counts_now = (
                dict(slot_choice_count[slot.id]) if gossip_level != "off" else None
            )

            t_dec_start = time.perf_counter()
            decision = await agent.decide(
                slot_id=slot.id,
                talks=slot_talks_meta,
                hall_loads_pct={},  # capacity не передаётся (Q-G accepted)
                recommendation=rec,
                llm_call=llm_call,
                gossip_counts=gossip_counts_now,
                gossip_n_total=(gossip_n_total
                                if gossip_level != "off" else None),
                gossip_level=gossip_level,
            )
            t_llmagent_total += time.perf_counter() - t_dec_start
            cumulative_cost_ref[0] += decision.cost_usd

            if decision.chosen is None:
                if ("parse-error" in decision.reason
                        or "invalid-choice" in decision.reason):
                    n_parse_errors += 1
            else:
                hall = program_conf.talks[decision.chosen].hall
                slot_loads[slot.id][hall] += 1
                slot_choice_count[slot.id][decision.chosen] += 1
                agents[ai].commit(slot.id, {
                    "id": decision.chosen,
                    "title": program_conf.talks[decision.chosen].title,
                    "category": program_conf.talks[decision.chosen].category,
                })

            decisions.append({
                "agent_id": decision.agent_id,
                "slot_id": decision.slot_id,
                "chosen": decision.chosen,
                "reason": decision.reason,
                "cost_usd": decision.cost_usd,
                "recommended": rec,
            })

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "cost": f"${cumulative_cost_ref[0]:.3f}",
                    "perr": n_parse_errors,
                })

    metrics = compute_metrics(
        decisions=decisions,
        slot_loads={k: dict(v) for k, v in slot_loads.items()},
        conf=program_conf, talk_emb_map=talk_emb_map,
        audience_emb_map=agent_emb_map,
    )

    # ranker delta (для П4 — реальное число вызовов / хитов / cost)
    if llm_ranker_pol is not None:
        ranker_calls_delta = (int(llm_ranker_pol.n_api_calls)
                              - ranker_calls_before)
        ranker_hits_delta = (int(llm_ranker_pol.n_cache_hits)
                             - ranker_hits_before)
        ranker_cost_delta = (float(llm_ranker_pol.cumulative_cost)
                             - ranker_cost_before)
    else:
        ranker_calls_delta = 0
        ranker_hits_delta = 0
        ranker_cost_delta = 0.0

    return {
        "status": status,
        "metrics": metrics,
        "decisions": decisions,
        "slot_loads": {k: dict(v) for k, v in slot_loads.items()},
        "n_parse_errors": n_parse_errors,
        "n_decisions_aborted": n_decisions_aborted,
        "gossip_level": gossip_level,
        "llmagent_cost_usd": float(sum(d["cost_usd"] for d in decisions)),
        "llmagent_calls": len(decisions),
        "llmagent_total_time_s": t_llmagent_total,
        "llmranker_total_time_s": t_llmranker_total,
        "llmranker_calls_delta": ranker_calls_delta,
        "llmranker_cache_hits_delta": ranker_hits_delta,
        "llmranker_cost_delta_usd": ranker_cost_delta,
        "recommendation_total_time_s": t_recommendation_total,
    }


# ---------- Главный runner ----------

async def main_async(args: argparse.Namespace) -> int:
    timings: dict[str, float] = {"start": time.time()}

    # ===== 1. Load Q + S read-only =====
    q_path = Path(args.input).resolve()
    print(f"[V] reading Q artifact (read-only): {q_path}", flush=True)
    q_data = json.loads(q_path.read_text(encoding="utf-8"))
    maximin_indices = list(q_data["maximin_indices"])
    lhs_rows_meta = q_data["lhs_rows"]
    q_results_by_pair: dict[tuple, dict] = {}
    for r in q_data["results"]:
        if r["replicate"] != 1:
            continue
        if not r["is_maximin_point"]:
            continue
        q_results_by_pair[(r["lhs_row_id"], r["policy"])] = r

    # Pre-snapshot of Q + S sha256 for invariant check
    q_artifact_paths = [
        ROOT / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.json",
        ROOT / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.csv",
        ROOT / "results/lhs_parametric_mobius_2025_autumn_2026-05-08.md",
    ]
    s_artifact_paths = [
        ROOT / "results/analysis_pairwise.json",
        ROOT / "results/analysis_sensitivity.json",
        ROOT / "results/analysis_program_effect.json",
        ROOT / "results/analysis_gossip_effect.json",
        ROOT / "results/analysis_risk_utility.json",
        ROOT / "results/analysis_llm_ranker_diagnostic.json",
        ROOT / "results/analysis_stability.json",
        ROOT / "results/analysis_capacity_audit.json",
        ROOT / "results/analysis_lhs_parametric_2026-05-08.md",
    ]
    sha_before = {p.name: sha256_of_file(p)
                  for p in q_artifact_paths + s_artifact_paths if p.exists()}

    # ===== 2. Load conference + personas =====
    conf_path = ROOT / "data/conferences" / f"{args.conference}.json"
    emb_path = ROOT / "data/conferences" / f"{args.conference}_embeddings.npz"
    base_conf = Conference.load(conf_path, emb_path)

    pers_path = ROOT / "data/personas" / f"{args.personas}.json"
    pers_emb_path = ROOT / "data/personas" / f"{args.personas}_embeddings.npz"
    pers_records = json.loads(pers_path.read_text(encoding="utf-8"))
    pers_emb_npz = np.load(pers_emb_path, allow_pickle=False)
    pers_emb_map = {
        pid: pers_emb_npz["embeddings"][i]
        for i, pid in enumerate(pers_emb_npz["ids"].tolist())
    }
    all_users = [
        UserProfile(
            id=p["id"],
            text=p.get("background", p.get("role", p["id"])),
            embedding=pers_emb_map[p["id"]],
        )
        for p in pers_records
    ]
    talk_emb_map = {tid: t.embedding for tid, t in base_conf.talks.items()}

    print(f"[V] loaded {args.conference}: {len(base_conf.talks)} talks, "
          f"{len(base_conf.halls)} halls, {len(base_conf.slots)} slots, "
          f"{len(all_users)} personas", flush=True)

    # ===== 3. Determine which LHS rows / policies =====
    if args.smoke:
        # smoke = 1 LHS row × 2 policies (no_policy, capacity_aware)
        # выбираем первую maximin LHS-row с минимальным audience_size
        candidates = [(r["lhs_row_id"], r["audience_size"])
                      for r in lhs_rows_meta
                      if r["lhs_row_id"] in maximin_indices]
        candidates.sort(key=lambda x: (x[1], x[0]))
        smoke_lhs_id = candidates[0][0]
        target_lhs_ids = [smoke_lhs_id]
        target_policies = ("no_policy", "capacity_aware")
        print(f"[V] SMOKE mode: LHS row #{smoke_lhs_id} "
              f"(audience={candidates[0][1]}) × policies={target_policies}",
              flush=True)
    else:
        target_lhs_ids = list(maximin_indices)
        if args.policies:
            requested = [p.strip() for p in args.policies.split(",") if p.strip()]
            invalid = [p for p in requested if p not in POLICY_NAMES]
            if invalid:
                raise ValueError(f"unknown policies in --policies: {invalid}; "
                                 f"valid: {POLICY_NAMES}")
            target_policies = tuple(requested)
        else:
            target_policies = POLICY_NAMES

    # ===== 4. LLM client + ranker policy =====
    # Default AsyncOpenAI без custom httpx — SDK сам управляет pool;
    # max_retries=0 — отключает SDK auto-retry (который создавал
    # retry-deadlock на 16 параллельных streams при transient 5xx,
    # подтверждено инцидентами 2026-05-08).
    # Per-call hard timeout через asyncio.wait_for(call, 90) — hung
    # call отменяется → нет deadlock семафора и connection pool.
    api_key = load_api_key()
    p_in, p_out = PRICING.get(args.model, (1.0, 5.0))
    sem = asyncio.Semaphore(args.concurrency)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
    )

    # Для GPT-5.x моделей через OpenRouter: явно отключаем reasoning,
    # иначе бьются reasoning_tokens (биллятся как output) и взрывается
    # latency. Subagent research 2026-05-08 подтвердил, что gpt-5.4-nano
    # должен запускаться с reasoning.enabled=False.
    needs_reasoning_off = args.model.startswith("openai/gpt-5")
    extra_body = (
        {"reasoning": {"enabled": False}} if needs_reasoning_off else {}
    )

    # Live diagnostics: глобальный счётчик in-flight + статистика по
    # latency + heartbeat (печатает каждые 30 сек).
    diag_state = {
        "calls_started": 0, "calls_completed": 0, "calls_timeout": 0,
        "calls_error": 0, "in_flight": 0,
        "latencies": [],  # последние 100, для median
        "pbar_decisions": None,  # set after pbar created
    }

    PER_CALL_HARD_TIMEOUT = 90.0  # asyncio.wait_for cancel ceiling

    async def llm_call(system, user, max_tokens=200):
        async with sem:
            diag_state["calls_started"] += 1
            diag_state["in_flight"] += 1
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.3,
                        max_tokens=max_tokens,
                        timeout=60,
                        extra_body=extra_body,
                    ),
                    timeout=PER_CALL_HARD_TIMEOUT,
                )
            except asyncio.TimeoutError:
                diag_state["calls_timeout"] += 1
                diag_state["in_flight"] -= 1
                return ('{"choice":"skip","reason":"api-error: '
                        f'wait_for-timeout-{PER_CALL_HARD_TIMEOUT}s"}}', 0.0)
            except Exception as e:
                diag_state["calls_error"] += 1
                diag_state["in_flight"] -= 1
                return (f'{{"choice":"skip","reason":"api-error: {e}"}}',
                        0.0)
            finally:
                duration = time.perf_counter() - t0
                if len(diag_state["latencies"]) >= 100:
                    diag_state["latencies"].pop(0)
                diag_state["latencies"].append(duration)
        diag_state["calls_completed"] += 1
        diag_state["in_flight"] -= 1
        if diag_state["pbar_decisions"] is not None:
            diag_state["pbar_decisions"].update(1)
        msg = resp.choices[0].message.content or ""
        u = resp.usage
        cost = 0.0
        if u is not None:
            cost = (u.prompt_tokens / 1e6) * p_in
            cost += (u.completion_tokens / 1e6) * p_out
        return msg, cost

    async def heartbeat_loop():
        """Печатает live diagnostics каждые 30 секунд."""
        while True:
            await asyncio.sleep(30)
            lats = diag_state["latencies"]
            med = float(np.median(lats)) if lats else 0.0
            tqdm.write(
                f"[V] heartbeat: in_flight={diag_state['in_flight']} "
                f"completed={diag_state['calls_completed']} "
                f"timeout={diag_state['calls_timeout']} "
                f"errors={diag_state['calls_error']} "
                f"median_latency={med:.2f}s "
                f"cost=${cumulative_cost_ref[0]:.3f}"
            )

    capacity_aware_pol = CapacityAwarePolicy()
    llm_ranker_pol = None
    if "llm_ranker" in target_policies:
        from src.policies.llm_ranker_policy import LLMRankerPolicy
        # warm cache используется по умолчанию: LLMRankerPolicy._load_cache()
        # читает из CACHE_PATH = experiments/logs/llm_ranker_cache.json
        llm_ranker_pol = LLMRankerPolicy(
            model="openai/gpt-4o-mini",
            budget_usd=args.budget_cap,  # общий cap
        )
        if not args.use_warm_cache:
            llm_ranker_pol.cache = {}

    # ===== 5. Iterate LHS rows × policies =====
    # ИНВАРИАНТ:
    # - SEQUENTIAL внутри одного слота (gossip-инвариант — slot_choice_count
    #   обновляется per agent decision; ПАРАЛЛЕЛИТЬ ЗАПРЕЩЕНО);
    # - SEQUENTIAL по слотам внутри одной (LHS, policy) пары;
    # - PARALLEL по политикам внутри одной LHS-row (CRN audience общий —
    #   prepare(lhs) выполняется ДО gather(policies));
    # - PARALLEL по LHS-row через --parallel-lhs N: разные LHS-row не
    #   делят state (audience/программа/seeds свои), общий ресурс — только
    #   cumulative_cost_ref[0] (atomic-increment под Python GIL) и
    #   LLMRankerPolicy.cache (разные cache-keys на разных LHS).
    cumulative_cost_ref = [0.0]
    long_rows: list[dict] = []
    aborted_outer = False
    overall_status = "ok"
    parallel_lhs = max(1, int(args.parallel_lhs))

    # ===== 5a. Persistent partial save / resume =====
    suffix_resume = "_smoke" if args.smoke else ""
    partial_path = Path(
        f"{args.output_prefix}{suffix_resume}.partial.jsonl"
    ).resolve()
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    completed_pairs: set[tuple[int, str]] = set()
    if args.resume and partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed_pairs.add((rec["lhs_row_id"], rec["policy"]))
                long_rows.append(rec)
                if rec.get("llmagent_cost_usd"):
                    cumulative_cost_ref[0] += float(rec["llmagent_cost_usd"])
                if rec.get("llmranker_cost_delta_usd"):
                    cumulative_cost_ref[0] += float(
                        rec["llmranker_cost_delta_usd"]
                    )
        print(f"[V] resume: loaded {len(completed_pairs)} completed "
              f"(lhs_row_id, policy) pairs from {partial_path.name}; "
              f"cumulative_cost so far = ${cumulative_cost_ref[0]:.4f}",
              flush=True)
    else:
        if (not args.resume) and partial_path.exists():
            partial_path.unlink()
            print(f"[V] no-resume: removed {partial_path.name}", flush=True)

    def append_partial(rec: dict) -> None:
        """Append одной long-format строки в JSONL с явным fsync.

        Это критически важно: при kill / sleep / network-hang мы должны
        сохранить уже выполненные evals. Append-only формат + fsync даёт
        атомарность на уровне строки.
        """
        with open(partial_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            try:
                import os as _os
                _os.fsync(f.fileno())
            except OSError:
                pass

    total_evals = len(target_lhs_ids) * len(target_policies)
    # Two pbars:
    # - outer "V evals" — 48 evals, обновляется когда (lhs, policy) done
    # - inner "V decisions" — total = sum(audience × 16 slots × policies),
    #   обновляется на КАЖДЫЙ agent decision → live прогресс с самого начала
    total_decisions = sum(
        next(r["audience_size"] for r in lhs_rows_meta
             if r["lhs_row_id"] == lid)
        * len(base_conf.slots) * len(target_policies)
        for lid in target_lhs_ids
    )
    pbar_evals = tqdm(total=total_evals, desc="V evals", unit="eval",
                      dynamic_ncols=True, position=0)
    pbar_decisions = tqdm(total=total_decisions, desc="V decisions",
                          unit="dec", dynamic_ncols=True, position=1,
                          leave=True, mininterval=1.0)

    # Связываем pbar_decisions с diag_state — llm_call обновляет его
    # на каждом successful decide.
    diag_state["pbar_decisions"] = pbar_decisions

    t_total_start = time.perf_counter()
    heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def process_lhs(lhs_id: int) -> list[dict]:
        """Обрабатывает одну LHS-row: prepare + gather(4 policy).

        Возвращает list of long-format records (по политикам).
        ВАЖНО: per-policy save выполняется ВНУТРИ asyncio task для
        каждой политики, чтобы партиал на диске появлялся сразу при
        завершении любой политики, а не после всей LHS. Это даёт
        прогресс-восстановление при зависаниях / sleep mode.
        """
        row_meta = next(r for r in lhs_rows_meta
                        if r["lhs_row_id"] == lhs_id)
        seeds = derive_seeds(lhs_id, replicate=1)
        actual_cfg_seed = args.cfg_seed

        scaled_conf = scale_capacity(
            base_conf, row_meta["capacity_multiplier"],
        )
        q_pair_sample = q_results_by_pair.get((lhs_id, "no_policy"))
        q_swap = q_pair_sample.get("swap_descriptor") if q_pair_sample else None
        program_conf, prog_meta = apply_program_variant_with_q_descriptor(
            scaled_conf, row_meta["program_variant"],
            q_swap_descriptor=q_swap,
            phi_seed=seeds["phi_seed"],
        )
        audience = select_audience(
            all_users, row_meta["audience_size"],
            seeds["audience_seed"],
        )

        # Per-policy inner pbars — только при parallel_lhs == 1, чтобы
        # терминал не превращался в хаос из 16+ перекрывающихся баров.
        decisions_per_policy = (
            row_meta["audience_size"] * len(program_conf.slots)
        )
        per_policy_inner_bars: dict[str, tqdm | None] = {}
        if parallel_lhs == 1:
            for pi in target_policies:
                per_policy_inner_bars[pi] = tqdm(
                    total=decisions_per_policy,
                    desc=f"  LHS#{lhs_id} {pi}", unit="dec",
                    leave=False, position=1, dynamic_ncols=True,
                )
        else:
            for pi in target_policies:
                per_policy_inner_bars[pi] = None

        async def _run_policy(pi):
            # Skip уже посчитанные (lhs, policy) пары при resume
            if (lhs_id, pi) in completed_pairs:
                return None
            res = await run_lhs_policy(
                lhs_row=row_meta,
                policy_name=pi,
                program_conf=program_conf,
                audience=audience,
                audience_emb_map={u.id: u.embedding for u in audience},
                talk_emb_map=talk_emb_map,
                K=args.K,
                llm_call=llm_call,
                cumulative_cost_ref=cumulative_cost_ref,
                budget_cap=args.budget_cap,
                cfg_seed=actual_cfg_seed,
                capacity_aware_pol=capacity_aware_pol,
                llm_ranker_pol=(llm_ranker_pol if pi == "llm_ranker"
                                else None),
                pbar=per_policy_inner_bars[pi],
                language=args.language,
            )
            metrics = res["metrics"]
            rec = {
                "lhs_row_id": lhs_id,
                "capacity_multiplier": row_meta["capacity_multiplier"],
                "popularity_source": row_meta["popularity_source"],
                "w_rel": row_meta["w_rel"],
                "w_rec": row_meta["w_rec"],
                "w_gossip": row_meta["w_gossip"],
                "audience_size": row_meta["audience_size"],
                "program_variant": row_meta["program_variant"],
                "policy": pi,
                "replicate": 1,
                "audience_seed": seeds["audience_seed"],
                "phi_seed": seeds["phi_seed"],
                "cfg_seed": actual_cfg_seed,
                "is_maximin_point": True,
                "fallback_to_p0": prog_meta.get("fallback_to_p0", False),
                "swap_descriptor": prog_meta.get("swap_descriptor"),
                "gossip_level": res["gossip_level"],
                "status": res["status"],
                "n_parse_errors": res["n_parse_errors"],
                "llmagent_cost_usd": res["llmagent_cost_usd"],
                "llmagent_calls": res["llmagent_calls"],
                "llmagent_total_time_s": res["llmagent_total_time_s"],
                "llmranker_total_time_s": res["llmranker_total_time_s"],
                "llmranker_calls_delta": res["llmranker_calls_delta"],
                "llmranker_cache_hits_delta":
                    res["llmranker_cache_hits_delta"],
                "llmranker_cost_delta_usd": res["llmranker_cost_delta_usd"],
                **metrics,
            }
            # Сохраняем сразу — НЕ ждём все 4 policy этой LHS
            append_partial(rec)
            completed_pairs.add((lhs_id, pi))
            tqdm.write(
                f"[V] saved (lhs={lhs_id}, policy={pi}) "
                f"cost=${cumulative_cost_ref[0]:.3f} "
                f"overload={metrics['metric_mean_overload_excess']:.4f}"
            )
            return rec

        results = await asyncio.gather(
            *(_run_policy(pi) for pi in target_policies)
        )
        for bar in per_policy_inner_bars.values():
            if bar is not None:
                bar.close()

        return [r for r in results if r is not None]

    # Filter target_lhs_ids: skip LHS, у которых ВСЕ ожидаемые политики
    # уже сохранены в partial JSONL.
    expected_policies_per_lhs = set(target_policies)

    def lhs_fully_done(lhs_id: int) -> bool:
        done = {pol for (lid, pol) in completed_pairs if lid == lhs_id}
        return expected_policies_per_lhs.issubset(done)

    pending_lhs_ids = [
        lid for lid in target_lhs_ids if not lhs_fully_done(lid)
    ]
    skipped = len(target_lhs_ids) - len(pending_lhs_ids)
    if skipped:
        print(f"[V] resume: skipping {skipped} fully-done LHS-row, "
              f"{len(pending_lhs_ids)} remaining", flush=True)
        pbar_evals.update(
            sum(len(target_policies) for _ in range(skipped))
        )

    # Батчим LHS-row по parallel_lhs; внутри батча — gather, между
    # батчами — sequential (для прозрачного abort при budget_exceeded и
    # для reset client против HTTP connection-pool exhaustion).
    for batch_start in range(0, len(pending_lhs_ids), parallel_lhs):
        if aborted_outer:
            break
        batch = pending_lhs_ids[batch_start:batch_start + parallel_lhs]
        if parallel_lhs > 1 and len(batch) > 1:
            tqdm.write(f"[V] batch LHS-row parallel: {batch}")
        batch_results = await asyncio.gather(
            *(process_lhs(lhs_id) for lhs_id in batch)
        )
        # records уже сохранены в partial.jsonl внутри _run_policy.
        # Здесь только обновляем in-memory long_rows и outer pbar.
        for records in batch_results:
            for rec in records:
                long_rows.append(rec)
                pbar_evals.update(1)
                pbar_evals.set_postfix({
                    "cost": f"${cumulative_cost_ref[0]:.3f}",
                    "lhs": rec["lhs_row_id"],
                    "pol": rec["policy"],
                })
                if rec["status"] != "ok":
                    overall_status = rec["status"]
                    aborted_outer = True

    pbar_evals.close()
    pbar_decisions.close()
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    t_total_end = time.perf_counter()

    # Save llm_ranker cache (если использовался)
    if llm_ranker_pol is not None:
        llm_ranker_pol._flush()

    # ===== 6. Sha256 invariant =====
    sha_after = {p.name: sha256_of_file(p)
                 for p in q_artifact_paths + s_artifact_paths if p.exists()}
    qs_invariant_violations = [
        name for name in sha_before
        if sha_before[name] != sha_after.get(name)
    ]

    # ===== 7. Write artifacts =====
    suffix = "_smoke" if args.smoke else ""
    out_json = Path(f"{args.output_prefix}{suffix}.json").resolve()
    out_csv = Path(f"{args.output_prefix}{suffix}.csv").resolve()
    out_md = Path(f"{args.output_prefix}{suffix}.md").resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    aggregated = {
        "etap": "V" if not args.smoke else "V-smoke",
        "conference": args.conference,
        "status": overall_status,
        "elapsed_total_s": t_total_end - t_total_start,
        "params": {
            "date": time.strftime("%Y-%m-%d"),
            "model": args.model,
            "K": args.K,
            "cfg_seed": args.cfg_seed,
            "budget_cap_usd": args.budget_cap,
            "concurrency": args.concurrency,
            "use_warm_cache": args.use_warm_cache,
            "parallel_lhs": parallel_lhs,
            "personas_source": args.personas,
            "input_q_artifact": str(q_path.name),
            "target_lhs_ids": list(target_lhs_ids),
            "target_policies": list(target_policies),
            "smoke": bool(args.smoke),
        },
        "results": long_rows,
        "n_results": len(long_rows),
        "cumulative_cost_usd": float(cumulative_cost_ref[0]),
        "llmagent_cost_usd": float(sum(r["llmagent_cost_usd"]
                                        for r in long_rows)),
        "llmagent_calls": int(sum(r["llmagent_calls"] for r in long_rows)),
        "llmagent_total_time_s": float(sum(r["llmagent_total_time_s"]
                                           for r in long_rows)),
        "llmranker_total_time_s": float(sum(r["llmranker_total_time_s"]
                                            for r in long_rows)),
        "llmranker_calls_total": int(sum(r["llmranker_calls_delta"]
                                         for r in long_rows)),
        "llmranker_cache_hits_total": int(sum(r["llmranker_cache_hits_delta"]
                                              for r in long_rows)),
        "llmranker_cost_total_usd": float(sum(r["llmranker_cost_delta_usd"]
                                              for r in long_rows)),
        "n_parse_errors_total": int(sum(r["n_parse_errors"]
                                        for r in long_rows)),
        "qs_invariant_violations": qs_invariant_violations,
        "qs_sha256_before": sha_before,
        "qs_sha256_after": sha_after,
    }
    out_json.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2))

    # Long-format CSV
    csv_columns = [
        "lhs_row_id", "capacity_multiplier", "popularity_source",
        "w_rel", "w_rec", "w_gossip", "audience_size", "program_variant",
        "policy", "replicate", "audience_seed", "phi_seed", "cfg_seed",
        "is_maximin_point", "fallback_to_p0", "gossip_level", "status",
        "metric_mean_overload_excess", "metric_mean_user_utility",
        "metric_overflow_rate_slothall",
        "metric_hall_utilization_variance",
        "metric_n_skipped", "metric_n_users",
        "n_parse_errors", "llmagent_cost_usd", "llmagent_calls",
        "llmagent_total_time_s", "llmranker_total_time_s",
        "llmranker_calls_delta", "llmranker_cache_hits_delta",
        "llmranker_cost_delta_usd",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns,
                                extrasaction="ignore")
        writer.writeheader()
        for r in long_rows:
            writer.writerow(r)

    # Markdown report
    md_lines = []
    md_lines.append(f"# Этап V: LLM-симулятор на 12 maximin "
                    f"{'(SMOKE)' if args.smoke else ''}")
    md_lines.append("")
    md_lines.append(f"Дата: {time.strftime('%Y-%m-%d')}")
    md_lines.append(f"Конференция: `{args.conference}`")
    md_lines.append(f"Модель: `{args.model}`")
    md_lines.append(f"Source Q (read-only): `{q_path.name}`")
    md_lines.append(f"Status: **{overall_status}**")
    md_lines.append("")
    md_lines.append("## Параметры")
    md_lines.append("")
    md_lines.append(f"- target LHS rows: {target_lhs_ids}")
    md_lines.append(f"- target policies: {list(target_policies)}")
    md_lines.append(f"- cfg_seed: {args.cfg_seed}")
    md_lines.append(f"- budget hard cap: ${args.budget_cap}")
    md_lines.append(f"- concurrency: {args.concurrency}")
    md_lines.append(f"- warm cache LLM-ranker: {args.use_warm_cache}")
    md_lines.append(f"- smoke: {args.smoke}")
    md_lines.append("")
    md_lines.append("## Cost / time breakdown")
    md_lines.append("")
    md_lines.append(f"- Cumulative cost (всё): "
                    f"**${cumulative_cost_ref[0]:.4f}**")
    md_lines.append(f"- LLMAgent cost: "
                    f"**${aggregated['llmagent_cost_usd']:.4f}**")
    md_lines.append(f"- LLMAgent calls: "
                    f"{aggregated['llmagent_calls']}")
    md_lines.append(f"- LLMAgent total time: "
                    f"{aggregated['llmagent_total_time_s']:.1f}s")
    md_lines.append(f"- LLMRankerPolicy cost (delta): "
                    f"**${aggregated['llmranker_cost_total_usd']:.4f}**")
    md_lines.append(f"- LLMRankerPolicy API calls (new): "
                    f"{aggregated['llmranker_calls_total']}")
    md_lines.append(f"- LLMRankerPolicy cache hits: "
                    f"{aggregated['llmranker_cache_hits_total']}")
    md_lines.append(f"- LLMRankerPolicy total time: "
                    f"{aggregated['llmranker_total_time_s']:.1f}s")
    md_lines.append(f"- Total wallclock: "
                    f"{aggregated['elapsed_total_s']:.1f}s")
    md_lines.append(f"- Avg time per eval: "
                    f"{aggregated['elapsed_total_s'] / max(1, len(long_rows)):.1f}s")
    md_lines.append(f"- Parse errors total: "
                    f"{aggregated['n_parse_errors_total']}")
    md_lines.append("")
    md_lines.append("## Q/S invariant")
    md_lines.append("")
    if qs_invariant_violations:
        md_lines.append(
            f"**FAIL:** sha256 различается у файлов: "
            f"{qs_invariant_violations}"
        )
    else:
        md_lines.append("**PASS:** sha256 всех Q/S артефактов совпадают "
                        "до и после прогона.")
    md_lines.append("")
    md_lines.append("## Per-eval table (overload / utility)")
    md_lines.append("")
    md_lines.append(
        "| LHS | policy | gossip | overload | utility | overflow | hall_var |"
    )
    md_lines.append("|---:|---|---|---:|---:|---:|---:|")
    for r in long_rows:
        md_lines.append(
            f"| {r['lhs_row_id']} | {r['policy']} | {r['gossip_level']} | "
            f"{r['metric_mean_overload_excess']:.4f} | "
            f"{r['metric_mean_user_utility']:.4f} | "
            f"{r['metric_overflow_rate_slothall']:.4f} | "
            f"{r['metric_hall_utilization_variance']:.4f} |"
        )
    md_lines.append("")
    out_md.write_text("\n".join(md_lines))

    print(f"\n[V] WROTE: {out_json}", flush=True)
    print(f"[V] WROTE: {out_csv}", flush=True)
    print(f"[V] WROTE: {out_md}", flush=True)
    print(f"[V] cumulative cost = ${cumulative_cost_ref[0]:.4f}", flush=True)
    print(f"[V] LLMAgent cost   = ${aggregated['llmagent_cost_usd']:.4f}",
          flush=True)
    print(f"[V] LLMRanker cost  = ${aggregated['llmranker_cost_total_usd']:.4f}",
          flush=True)
    print(f"[V] LLMAgent calls  = {aggregated['llmagent_calls']}", flush=True)
    print(f"[V] LLMRanker calls = {aggregated['llmranker_calls_total']}",
          flush=True)
    print(f"[V] LLMRanker hits  = {aggregated['llmranker_cache_hits_total']}",
          flush=True)
    print(f"[V] elapsed         = {aggregated['elapsed_total_s']:.1f}s",
          flush=True)
    if qs_invariant_violations:
        print(f"[V] [!] Q/S invariant FAIL: {qs_invariant_violations}",
              flush=True)
        return 2
    if overall_status != "ok":
        print(f"[V] [!] status = {overall_status}", flush=True)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Этап V: LLM cross-validation на 12 maximin LHS-точках"
    )
    ap.add_argument("--input", default=DEFAULT_LHS_INPUT,
                    help="Q JSON (read-only)")
    ap.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    ap.add_argument("--conference", default=DEFAULT_CONFERENCE)
    ap.add_argument("--personas", default=DEFAULT_PERSONAS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--cfg-seed", type=int, default=DEFAULT_CFG_SEED)
    ap.add_argument("--budget-cap", type=float, default=DEFAULT_BUDGET_CAP)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--use-warm-cache", action="store_true", default=True,
                    help="использовать llm_ranker_cache.json (по умолчанию)")
    ap.add_argument("--no-warm-cache", action="store_false",
                    dest="use_warm_cache",
                    help="игнорировать существующий cache LLM-ranker")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke-режим: 1 LHS-row × 2 policy")
    ap.add_argument("--parallel-lhs", type=int, default=2,
                    help="число LHS-row, обрабатываемых параллельно "
                         "(default 2 → 8 streams при concurrency=8). "
                         "С 16 streams (parallel-lhs=4 × 4 policy) "
                         "наблюдались зависания на macOS asyncio + httpx "
                         "0.27 (см. инциденты 2026-05-08). Gossip invariant "
                         "сохраняется при любом значении: у каждой LHS свой "
                         "slot_choice_count.")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="resume из partial JSONL (по умолчанию)")
    ap.add_argument("--no-resume", action="store_false", dest="resume",
                    help="игнорировать partial JSONL, начать с нуля")
    ap.add_argument("--policies", default=None,
                    help="comma-separated policies; default: all from POLICY_NAMES")
    ap.add_argument("--language", default="ru", choices=["ru", "en"],
                    help="язык промптов LLMAgent: 'ru' (default, RU-прогон) или "
                         "'en' (EN-пайплайн с BGE+ABTT, паритет каналов)")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
