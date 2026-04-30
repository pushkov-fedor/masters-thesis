"""Agent simulator v2 — слой 2 валидации с эмерджентным поведением.

Расширения:
- использует GenerativeAgentV2 (память, личность, усталость, рефлексия)
- социальный граф между агентами (peer decisions передаются в промпт)
- agents видят друзей в текущем слоте перед решением
- параллелизация внутри слота batch'ами по 10 (баланс параллелизма и эффекта socila)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .generative_agent_v2 import GenerativeAgentV2
from .social_graph import SocialGraph


@dataclass
class AgentSimResultV2:
    decisions: list = field(default_factory=list)
    hall_load_per_slot: Dict[str, Dict[int, int]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_errors: int = 0
    # для тестов гипотез
    skip_rate_per_slot: Dict[str, float] = field(default_factory=dict)
    fatigue_per_agent_per_slot: Dict[str, Dict[int, float]] = field(default_factory=dict)
    personality_per_agent: Dict[int, dict] = field(default_factory=dict)


async def simulate_agents_v2(
    conf,
    agents: List[GenerativeAgentV2],
    policy: Callable,
    user_profiles: list,
    social_graph: SocialGraph,
    K: int = 2,
    concurrency: int = 30,
    seed: int = 42,
    relevance_fn: Optional[Callable] = None,
) -> AgentSimResultV2:
    """Прогон агентов через программу конференции с эмерджентным поведением.

    Внутри слота агенты обрабатываются батчами 8-10: первая партия видит initial
    hall_load и пустой social_signal, последующие видят кумулятивно обновлённое
    состояние. Это создаёт реалистичный congestion+social-эффект.
    """
    rng = np.random.default_rng(seed)
    sem = asyncio.Semaphore(concurrency)

    hall_load: Dict[tuple, int] = {}
    for s in conf.slots:
        for h in conf.halls.values():
            hall_load[(s.id, h.id)] = 0

    result = AgentSimResultV2()
    # Записываем personality каждого агента (для гипотез)
    for a in agents:
        result.personality_per_agent[a.idx] = {
            "openness": a.personality.openness,
            "conscientiousness": a.personality.conscientiousness,
            "extraversion": a.personality.extraversion,
            "agreeableness": a.personality.agreeableness,
            "neuroticism": a.personality.neuroticism,
        }

    slot_num = 0
    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        candidates = [conf.talks[tid] for tid in slot.talk_ids]
        slot_num += 1

        # query_emb — среднее эмбеддингов кандидатов слота (контекст)
        slot_query_emb = np.mean([t.embedding for t in candidates], axis=0)
        slot_query_emb = slot_query_emb / max(1e-9, np.linalg.norm(slot_query_emb))

        halls_in_slot = sorted({t.hall for t in candidates})

        # Перемешиваем порядок (детерминированно по seed)
        order = list(range(len(agents)))
        rng.shuffle(order)

        async def process_agent(idx_in_order):
            agent_pos = order[idx_in_order]
            agent = agents[agent_pos]
            user = user_profiles[agent_pos]

            # social signal — что уже сделали друзья (видно из social_graph)
            social_signal = social_graph.render_signal_for_agent(
                agent_idx=agent.idx,
                slot_id=slot.id,
                halls_in_slot=halls_in_slot,
            )
            # current hall_load для каждого зала
            hall_load_fractions = {}
            for hid in halls_in_slot:
                cap = conf.halls[hid].capacity
                occ = hall_load[(slot.id, hid)]
                hall_load_fractions[hid] = occ / max(1.0, cap)

            # state для политики
            policy_state = {
                "hall_load": dict(hall_load),
                "slot_id": slot.id,
                "K": K,
                "relevance_fn": relevance_fn,
            }
            recs = policy(user=user, slot=slot, conf=conf, state=policy_state)
            recs = [r for r in recs if r in slot.talk_ids][:K]

            decision = await agent.decide(
                slot=slot,
                slot_num=slot_num,
                slot_query_emb=slot_query_emb,
                candidates=candidates,
                hall_loads=hall_load_fractions,
                recommendation=recs,
                social_signal=social_signal,
                sem=sem,
            )
            return agent_pos, decision

        # Батчи внутри слота: 10 агентов параллельно, между батчами обновляется hall_load
        # и social_graph — это даёт эффект «более поздние агенты видят, кто уже пошёл»
        BATCH = 10
        for batch_start in range(0, len(order), BATCH):
            batch_end = min(batch_start + BATCH, len(order))
            tasks = [process_agent(i) for i in range(batch_start, batch_end)]
            results_batch = await asyncio.gather(*tasks)

            for agent_pos, dec in results_batch:
                agent = agents[agent_pos]
                result.decisions.append({
                    "agent_id": agent.id,
                    "agent_idx": agent.idx,
                    "slot_id": slot.id,
                    "slot_num": slot_num,
                    "decision": dec.decision,
                    "reason": dec.reason,
                    "fatigue_level": agent.fatigue.level,
                })
                result.total_cost += dec.cost

                # Обновляем social_graph и hall_load
                if dec.decision != "skip":
                    talk = conf.talks[dec.decision]
                    hall_load[(slot.id, talk.hall)] += 1
                    social_graph.record(slot.id, agent.idx, talk.hall)
                else:
                    social_graph.record(slot.id, agent.idx, None)

                # Записываем fatigue за этот слот
                result.fatigue_per_agent_per_slot.setdefault(agent.id, {})[slot_num] = agent.fatigue.level

        # Skip rate в этом слоте
        slot_decisions = [d for d in result.decisions if d["slot_id"] == slot.id]
        skips = sum(1 for d in slot_decisions if d["decision"] == "skip")
        result.skip_rate_per_slot[slot.id] = skips / max(1, len(slot_decisions))

    # упаковка hall_load_per_slot
    per_slot: Dict[str, Dict[int, int]] = {}
    for (sid, hid), n in hall_load.items():
        per_slot.setdefault(sid, {})[hid] = n
    result.hall_load_per_slot = per_slot
    result.total_errors = sum(a.errors for a in agents)
    return result
