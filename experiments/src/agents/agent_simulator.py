"""Симулятор слоя 2: прогон LLM-агентов с памятью на программе конференции
для каждой из политик.

Отличие от src/simulator.py:
- пользователь — LLM-агент с памятью, не softmax-выбор;
- решение принимается через LLM-вызов с reasoning;
- агенты обрабатываются последовательно внутри слота (sequential greedy),
  но между слотами memory сохраняется.

Кэширование: ключ — (agent_id, slot_id, set_кандидатов, top_K, hall_loads_bucketed).
Поскольку memory влияет на decision, агентское решение зависит от истории — кэш-ключ
учитывает hash памяти.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from .generative_agent import GenerativeAgent, AgentDecision, _bucket


@dataclass
class AgentSimResult:
    """Результат симуляции слоя 2."""
    decisions: list = field(default_factory=list)  # [{agent_id, slot_id, decision, reason}]
    hall_load_per_slot: Dict[str, Dict[int, int]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_errors: int = 0


async def simulate_agents(
    conf,  # Conference object
    agents: List[GenerativeAgent],
    policy: Callable,  # signature (user, slot, conf, state) -> List[talk_id]
    user_profiles: list,  # list of UserProfile compatible objects (for policy input)
    K: int = 2,
    concurrency: int = 30,
    seed: int = 42,
) -> AgentSimResult:
    """Прогоняет всех агентов через всю программу конференции с одной политикой.

    agents и user_profiles параллельны: agents[i] соответствует user_profiles[i]
    (LLM-агент vs объект для запроса политики).
    """
    rng = np.random.default_rng(seed)
    sem = asyncio.Semaphore(concurrency)

    # init hall load
    hall_load: Dict[tuple, int] = {}
    for s in conf.slots:
        for h in conf.halls.values():
            hall_load[(s.id, h.id)] = 0

    result = AgentSimResult()
    slot_num = 0

    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        candidates = [conf.talks[tid] for tid in slot.talk_ids]
        slot_num += 1

        # Перемешиваем порядок агентов внутри слота (как в исходном симуляторе)
        order = list(range(len(agents)))
        rng.shuffle(order)

        # Для каждого агента: получаем top-K от политики, затем LLM-вызов
        # Параллельно для всего слота через asyncio.gather
        async def process_agent(idx):
            agent = agents[idx]
            user = user_profiles[idx]
            # state для политики
            state = {
                "hall_load": dict(hall_load),
                "slot_id": slot.id,
                "K": K,
            }
            recs = policy(user=user, slot=slot, conf=conf, state=state)
            recs = [r for r in recs if r in slot.talk_ids][:K]

            # текущая загрузка залов в слоте (фракции)
            hall_load_fractions = {}
            halls_in_slot = {t.hall for t in candidates}
            for hid in halls_in_slot:
                cap = conf.halls[hid].capacity
                occ = hall_load[(slot.id, hid)]
                hall_load_fractions[hid] = occ / max(1.0, cap)

            decision = await agent.decide(
                slot=slot,
                slot_num=slot_num,
                candidates=candidates,
                hall_loads=hall_load_fractions,
                recommendation=recs,
                sem=sem,
            )
            return idx, decision

        # Запускаем всех параллельно для одного слота
        # ВАЖНО: hall_load обновляется СИНХРОННО в порядке order, поэтому делать
        # параллельно нельзя без потери эффекта congestion. Делаем партиями по 10.
        BATCH = 5
        for i in range(0, len(order), BATCH):
            batch_idx = order[i : i + BATCH]
            tasks = [process_agent(idx) for idx in batch_idx]
            results_batch = await asyncio.gather(*tasks)

            # обновляем hall_load по результатам батча (sequential commit)
            for idx, dec in results_batch:
                result.decisions.append({
                    "agent_id": agents[idx].id,
                    "slot_id": slot.id,
                    "decision": dec.decision,
                    "reason": dec.reason,
                })
                result.total_cost += dec.cost
                if dec.decision != "skip":
                    talk = conf.talks[dec.decision]
                    hall_load[(slot.id, talk.hall)] += 1

        result.total_errors += sum(a.errors for a in agents)

    # упаковка hall_load_per_slot
    per_slot: Dict[str, Dict[int, int]] = {}
    for (sid, hid), n in hall_load.items():
        per_slot.setdefault(sid, {})[hid] = n
    result.hall_load_per_slot = per_slot
    return result


def compute_agent_metrics(conf, sim_result, K=2):
    """Совместимые с src/metrics.py метрики на основе AgentSimResult."""
    # overflow_choice: только слоты с >=2 параллельными докладами
    overfull_choice = 0
    total_choice = 0
    overfull_all = 0
    total_all = 0
    variance_per_slot = []

    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        halls_in_slot = sorted({conf.talks[tid].hall for tid in slot.talk_ids})
        loads = []
        for hid in halls_in_slot:
            cap = conf.halls[hid].capacity
            occ = sim_result.hall_load_per_slot.get(slot.id, {}).get(hid, 0)
            total_all += 1
            if occ > cap:
                overfull_all += 1
            if len(slot.talk_ids) > 1:
                total_choice += 1
                if occ > cap:
                    overfull_choice += 1
            loads.append(occ / max(1.0, cap))
        if len(halls_in_slot) >= 2:
            variance_per_slot.append(float(np.var(loads)))

    n_decisions = len(sim_result.decisions)
    n_skip = sum(1 for d in sim_result.decisions if d["decision"] == "skip")

    # mean utility approximation: для агентов нет численной cosine, но можно посчитать
    # cosine между профилем агента и выбранным докладом, если они доступны
    return {
        "overflow_rate_all": overfull_all / max(1, total_all),
        "overflow_rate_choice": overfull_choice / max(1, total_choice),
        "hall_utilization_variance": float(np.mean(variance_per_slot)) if variance_per_slot else 0.0,
        "skip_rate": n_skip / max(1, n_decisions),
        "n_decisions": n_decisions,
        "total_cost_usd": sim_result.total_cost,
        "total_errors": sim_result.total_errors,
    }
