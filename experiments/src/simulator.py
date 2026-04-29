"""Параметрический симулятор Constrained MDP для задачи рекомендации программы конференции.

Постановка по §2.1 Главы 2:
- состояние s_t — вектор заполненности залов в текущем тайм-слоте;
- действие a_t — список рекомендаций L_{i,t} для каждого пользователя i;
- модель выбора — мультиномиальная логит с температурой τ + альтернатива «отказ»;
- штраф за переполнение в полезности: u_ij = rel(p_i, j) - λ · overflow_penalty(j, s_t);
- action masking: переполненные залы исключаются из выдачи (опционально, контролируется через политику).

Релевантность: косинус между эмбеддингом профиля пользователя и эмбеддингом доклада.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Talk:
    id: str
    title: str
    hall: int
    slot_id: str
    category: str
    abstract: str
    embedding: np.ndarray  # 1D, нормализован


@dataclass
class Hall:
    id: int
    capacity: int


@dataclass
class Slot:
    id: str
    datetime: str
    talk_ids: List[str]


@dataclass
class Conference:
    name: str
    talks: Dict[str, Talk]
    halls: Dict[int, Hall]
    slots: List[Slot]  # упорядоченный список

    @classmethod
    def load(cls, prog_path: Path, emb_path: Path) -> "Conference":
        with open(prog_path, encoding="utf-8") as f:
            prog = json.load(f)
        npz = np.load(emb_path, allow_pickle=False)
        emb_ids = list(npz["ids"])
        emb_vec = npz["embeddings"]
        emb_map = {tid: emb_vec[i] for i, tid in enumerate(emb_ids)}

        talks = {}
        for t in prog["talks"]:
            talks[t["id"]] = Talk(
                id=t["id"],
                title=t["title"],
                hall=int(t["hall"]),
                slot_id=t["slot_id"],
                category=t.get("category", "Other"),
                abstract=t.get("abstract", ""),
                embedding=emb_map[t["id"]],
            )

        halls = {h["id"]: Hall(id=h["id"], capacity=h["capacity"]) for h in prog["halls"]}

        slot_index: Dict[str, List[str]] = {}
        for tid, t in talks.items():
            slot_index.setdefault(t.slot_id, []).append(tid)

        slots = [
            Slot(id=s["id"], datetime=s["datetime"], talk_ids=slot_index.get(s["id"], []))
            for s in prog["slots"]
        ]

        return cls(name=prog["name"], talks=talks, halls=halls, slots=slots)


@dataclass
class UserProfile:
    id: str
    text: str
    embedding: np.ndarray  # нормализован


@dataclass
class SimConfig:
    """Параметры симуляции."""
    tau: float = 0.7              # температура softmax для выбора пользователя
    lambda_overflow: float = 1.0  # штраф за заполненность зала (видимый пользователю)
    p_skip_base: float = 0.10     # базовая вероятность отказа в каждом слоте
    K: int = 3                    # размер выдачи (top-K рекомендаций)
    seed: int = 0


@dataclass
class StepRecord:
    slot_id: str
    user_id: str
    recommended: List[str]      # список id из выдачи политики
    chosen: Optional[str]        # выбранный доклад или None (отказ)
    chosen_relevance: float
    chosen_hall_load_before: float  # загрузка зала до выбора, [0, 1]


@dataclass
class SimResult:
    steps: List[StepRecord] = field(default_factory=list)
    # per-(slot, hall) -> число посетителей (после симуляции)
    hall_load_per_slot: Dict[str, Dict[int, int]] = field(default_factory=dict)


def cosine_relevance(user_emb: np.ndarray, talk_emb: np.ndarray) -> float:
    # эмбеддинги уже нормализованы
    return float(np.dot(user_emb, talk_emb))


def overflow_fraction(occupied: int, capacity: int) -> float:
    """Доля переполнения, [0, +inf). Если ≤ capacity → 0; > capacity → > 0."""
    if occupied <= capacity:
        return 0.0
    return (occupied - capacity) / max(1.0, capacity)


def utilization(occupied: int, capacity: int) -> float:
    """Заполненность [0, +inf), может быть > 1 при переполнении."""
    return occupied / max(1.0, capacity)


def simulate(
    conf: Conference,
    users: Sequence[UserProfile],
    policy: Callable,  # signature: (user, slot, conf, state) -> List[talk_id] (length K)
    cfg: SimConfig,
) -> SimResult:
    """Прогоняет одну симуляцию по конференции для всех пользователей.

    state передаётся в политику между вызовами как dict с ключами:
      - hall_load: dict[(slot_id, hall_id) -> int] (пользователи уже вошедшие)
      - slot_id: текущий слот
    """
    rng = np.random.default_rng(cfg.seed)
    # рандомизированный порядок пользователей чтобы порядок прихода не влиял систематически
    user_order = list(users)
    rng.shuffle(user_order)

    # Initialize hall load per slot
    hall_load: Dict[tuple, int] = {}
    for s in conf.slots:
        for h in conf.halls.values():
            hall_load[(s.id, h.id)] = 0

    result = SimResult()

    for slot in conf.slots:
        if not slot.talk_ids:
            continue
        candidates = [conf.talks[tid] for tid in slot.talk_ids]
        if not candidates:
            continue

        for user in user_order:
            state = {
                "hall_load": dict(hall_load),
                "slot_id": slot.id,
                "K": cfg.K,
            }
            recs = policy(user=user, slot=slot, conf=conf, state=state)
            # ограничить K и валидность
            recs = [r for r in recs if r in slot.talk_ids][: cfg.K]
            if not recs:
                # пустая выдача → отказ
                result.steps.append(StepRecord(
                    slot_id=slot.id, user_id=user.id, recommended=[],
                    chosen=None, chosen_relevance=0.0, chosen_hall_load_before=0.0,
                ))
                continue

            # формируем utility-вектор для softmax-выбора
            utils = []
            for tid in recs:
                t = conf.talks[tid]
                rel = cosine_relevance(user.embedding, t.embedding)
                hall = conf.halls[t.hall]
                load_frac = utilization(hall_load[(slot.id, hall.id)], hall.capacity)
                # Пользователь видит штраф за переполненный зал (косвенный сигнал)
                u = rel - cfg.lambda_overflow * max(0.0, load_frac - 0.85)
                utils.append(u)
            utils = np.array(utils, dtype=np.float64)

            # выбор: softmax + альтернатива "отказ" с полезностью log(p_skip / (1-p_skip)) * tau
            # Используем явное скалирование
            # P(j) ∝ exp(u_j / tau); P(skip) — отдельная альтернатива с фиксированной базовой вероятностью
            scaled = utils / max(cfg.tau, 1e-6)
            scaled = scaled - scaled.max()  # стабильность
            exps = np.exp(scaled)
            sum_exp = exps.sum()
            # отдельная вероятность отказа: p_skip_base + поправка от среднего штрафа
            p_skip = cfg.p_skip_base
            # нормализуем: рекомендации делят (1 - p_skip) пропорционально exps
            probs_recs = (1.0 - p_skip) * (exps / sum_exp)
            probs = np.concatenate([probs_recs, [p_skip]])
            probs = probs / probs.sum()

            choice_idx = rng.choice(len(probs), p=probs)
            if choice_idx == len(probs) - 1:
                # отказ
                result.steps.append(StepRecord(
                    slot_id=slot.id, user_id=user.id, recommended=recs,
                    chosen=None, chosen_relevance=0.0, chosen_hall_load_before=0.0,
                ))
                continue

            chosen_id = recs[choice_idx]
            chosen_talk = conf.talks[chosen_id]
            chosen_hall = conf.halls[chosen_talk.hall]
            load_before = utilization(hall_load[(slot.id, chosen_hall.id)], chosen_hall.capacity)
            hall_load[(slot.id, chosen_hall.id)] += 1
            result.steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=recs,
                chosen=chosen_id,
                chosen_relevance=float(cosine_relevance(user.embedding, chosen_talk.embedding)),
                chosen_hall_load_before=load_before,
            ))

    # упаковка hall_load_per_slot
    per_slot: Dict[str, Dict[int, int]] = {}
    for (sid, hid), n in hall_load.items():
        per_slot.setdefault(sid, {})[hid] = n
    result.hall_load_per_slot = per_slot
    return result
