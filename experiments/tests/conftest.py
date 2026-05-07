"""Pytest-фикстуры для этапа I PIVOT_IMPLEMENTATION_PLAN r5.

Состав:
- путь репозитория и `experiments/` в sys.path для `from src.simulator import ...`;
- builder для синтетических конференций (полностью контролируемые эмбединги);
- загрузчики `toy_microconf_2slot` + `personas_100` для EC tests;
- фабрики SimConfig / активных политик.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.simulator import (  # noqa: E402
    Conference,
    Hall,
    SimConfig,
    Slot,
    Talk,
    UserProfile,
)


# ---------- Базовые пути ----------

@pytest.fixture(scope="session")
def experiments_root() -> Path:
    return EXPERIMENTS_ROOT


# ---------- Загрузчики реальных toy/personas ----------

@pytest.fixture(scope="session")
def toy_2slot_conf(experiments_root: Path) -> Conference:
    """toy_microconf_2slot (этап H): 2 слота × 2 зала × 2 доклада в каждом слоте."""
    return Conference.load(
        experiments_root / "data/conferences/toy_microconf_2slot.json",
        experiments_root / "data/conferences/toy_microconf_2slot_embeddings.npz",
    )


@pytest.fixture(scope="session")
def personas_100_users(experiments_root: Path) -> List[UserProfile]:
    pers_path = experiments_root / "data/personas/personas_100.json"
    emb_path = experiments_root / "data/personas/personas_100_embeddings.npz"
    pers = json.loads(pers_path.read_text())
    npz = np.load(emb_path, allow_pickle=False)
    emb_map = {pid: npz["embeddings"][i] for i, pid in enumerate(npz["ids"])}
    return [
        UserProfile(
            id=p["id"],
            text=p.get("background", ""),
            embedding=emb_map[p["id"]],
        )
        for p in pers
    ]


@pytest.fixture(scope="session")
def personas_50_users(personas_100_users: List[UserProfile]) -> List[UserProfile]:
    """Первые 50 персон — для ускорения EC-тестов."""
    return personas_100_users[:50]


# ---------- Builder для synthetic conf (для toy-cases) ----------

@pytest.fixture
def make_synth_conf() -> Callable:
    """Builder синтетической конференции с полностью контролируемыми эмбедингами.

    Параметры:
        slots: int                      — число слотов
        talks_per_slot: int             — talks в каждом слоте
        hall_capacities: List[int] |
            Dict[(slot,hall), int]      — capacities; len = halls; hall id = 1..halls
        emb_dim: int                    — размерность эмбединга
        seed: int                       — random seed для эмбедингов
        equal_talks: bool               — все talks делаем равно-релевантными
        talk_emb_per_slot: Optional[
            List[np.ndarray]]           — явные эмбединги per slot (override)

    Возвращает (conf, talk_emb_map: dict[talk_id -> np.ndarray]).
    """
    def _build(
        slots: int = 1,
        talks_per_slot: int = 2,
        hall_capacities: List[int] = (50, 50),
        emb_dim: int = 8,
        seed: int = 0,
        equal_talks: bool = False,
        talk_emb_per_slot=None,
    ):
        n_halls = len(hall_capacities)
        assert talks_per_slot <= n_halls, "talks_per_slot ≤ n_halls"
        rng = np.random.default_rng(seed)

        halls = {i + 1: Hall(id=i + 1, capacity=int(hall_capacities[i]))
                 for i in range(n_halls)}

        talks = {}
        slot_objs = []
        for s in range(slots):
            sid = f"slot_{s:02d}"
            slot_talk_ids = []
            for h in range(talks_per_slot):
                tid = f"t_s{s}_h{h+1}"
                if talk_emb_per_slot is not None and s < len(talk_emb_per_slot):
                    emb = np.asarray(talk_emb_per_slot[s][h], dtype=np.float32)
                elif equal_talks:
                    base = rng.standard_normal(emb_dim).astype(np.float32)
                    base /= np.linalg.norm(base) + 1e-9
                    emb = base
                else:
                    base = rng.standard_normal(emb_dim).astype(np.float32)
                    base /= np.linalg.norm(base) + 1e-9
                    emb = base
                talks[tid] = Talk(
                    id=tid,
                    title=f"Talk {tid}",
                    hall=h + 1,
                    slot_id=sid,
                    category=f"cat_{h}",
                    abstract="",
                    embedding=emb,
                    fame=0.0,
                )
                slot_talk_ids.append(tid)
            slot_objs.append(Slot(
                id=sid,
                datetime=f"2026-01-0{s+1}T10:00:00",
                talk_ids=slot_talk_ids,
                hall_capacities=None,
            ))

        conf = Conference(
            name="synth",
            talks=talks,
            halls=halls,
            slots=slot_objs,
        )
        emb_map = {tid: t.embedding for tid, t in talks.items()}
        return conf, emb_map

    return _build


@pytest.fixture
def make_synth_users() -> Callable:
    """Builder синтетических пользователей с заданными или сгенерированными эмбедингами."""
    def _build(
        n: int = 100,
        emb_dim: int = 8,
        seed: int = 0,
        embeddings=None,
    ) -> List[UserProfile]:
        rng = np.random.default_rng(seed)
        users = []
        for i in range(n):
            if embeddings is not None:
                emb = np.asarray(embeddings[i % len(embeddings)], dtype=np.float32)
            else:
                emb = rng.standard_normal(emb_dim).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-9
            users.append(UserProfile(
                id=f"u_{i:04d}",
                text=f"synthetic user {i}",
                embedding=emb,
            ))
        return users

    return _build


# ---------- Активные политики ----------

@pytest.fixture
def active_pols_no_llm():
    """active_policies(include_llm=False): {no_policy, cosine, capacity_aware}."""
    from src.policies.registry import active_policies
    return active_policies(include_llm=False)


@pytest.fixture
def base_cfg() -> SimConfig:
    """Базовый SimConfig для smoke / EC: w_rel=0.7, w_rec=0.3, K=2, p_skip=0.10."""
    return SimConfig(
        tau=0.7,
        p_skip_base=0.10,
        K=2,
        seed=1,
        w_rel=0.7,
        w_rec=0.3,
        w_fame=0.0,
    )
