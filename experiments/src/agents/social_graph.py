"""Социальный граф агентов через Watts-Strogatz и трекинг peer decisions.

Каждый агент имеет ~5% знакомых. После каждого слота граф знает, кто куда пошёл.
В decide() агент видит «X% друзей идут в зал Y» — этот сигнал влияет
на его решение через personality.agreeableness.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


class SocialGraph:
    """Граф связей между агентами + трекинг peer-решений по слотам."""

    def __init__(self, n_agents: int, k: int = 6, p_rewire: float = 0.1, seed: int = 42):
        """Watts-Strogatz: каждый агент связан с k ближайшими + p_rewire вероятность перепривязки.

        Args:
            n_agents: число агентов
            k: средняя степень узла (число друзей)
            p_rewire: вероятность перепривязки ребра — даёт small-world
            seed: для воспроизводимости
        """
        self.n_agents = n_agents
        if nx is None:
            # Fallback без networkx — просто соседние агенты по индексу
            self.adjacency = {i: set([(i-j) % n_agents for j in range(1, k//2+1)] +
                                     [(i+j) % n_agents for j in range(1, k//2+1)])
                              for i in range(n_agents)}
        else:
            G = nx.watts_strogatz_graph(n_agents, k, p_rewire, seed=seed)
            self.adjacency = {i: set(G.neighbors(i)) for i in range(n_agents)}
        # peer_decisions[slot_id][agent_idx] -> hall_id (или None если skip)
        self.peer_decisions: Dict[str, Dict[int, Optional[int]]] = defaultdict(dict)

    def record(self, slot_id: str, agent_idx: int, hall_id: Optional[int]):
        """Зафиксировать решение агента в слоте."""
        self.peer_decisions[slot_id][agent_idx] = hall_id

    def friends(self, agent_idx: int) -> List[int]:
        return list(self.adjacency.get(agent_idx, set()))

    def friends_in_hall(self, agent_idx: int, slot_id: str, hall_id: int) -> int:
        """Сколько друзей этого агента уже пошли в этот зал в этом слоте."""
        slot_decisions = self.peer_decisions.get(slot_id, {})
        return sum(1 for f in self.adjacency.get(agent_idx, []) if slot_decisions.get(f) == hall_id)

    def friends_attending_share(self, agent_idx: int, slot_id: str, hall_id: int) -> float:
        """Доля друзей, посетивших данный зал в данном слоте."""
        n_friends = len(self.adjacency.get(agent_idx, set()))
        if n_friends == 0:
            return 0.0
        return self.friends_in_hall(agent_idx, slot_id, hall_id) / n_friends

    def render_signal_for_agent(self, agent_idx: int, slot_id: str,
                                halls_in_slot: List[int]) -> str:
        """Текстовое описание social signal для промпта."""
        n_friends = len(self.adjacency.get(agent_idx, set()))
        if n_friends == 0:
            return "(нет данных о друзьях)"
        slot_decisions = self.peer_decisions.get(slot_id, {})
        # сколько уже определились
        decided_friends = [self.adjacency[agent_idx] & set(slot_decisions.keys())]
        if not decided_friends or not decided_friends[0]:
            return "(друзья ещё не выбрали)"
        parts = []
        for hid in halls_in_slot:
            count = self.friends_in_hall(agent_idx, slot_id, hid)
            if count > 0:
                parts.append(f"в зал {hid} идут {count} из {n_friends} коллег")
        skip_count = sum(1 for f in self.adjacency[agent_idx]
                         if f in slot_decisions and slot_decisions[f] is None)
        if skip_count > 0:
            parts.append(f"{skip_count} коллег пропускают слот")
        return "; ".join(parts) if parts else "(никто из коллег пока не выбрал)"
