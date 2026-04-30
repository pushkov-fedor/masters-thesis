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
    fame: float = 0.0  # популярность доклада [0, 1] — для star-speaker effect


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
    def load(cls, prog_path, emb_path, fame_path=None) -> "Conference":
        prog_path = Path(prog_path)
        emb_path = Path(emb_path)
        with open(prog_path, encoding="utf-8") as f:
            prog = json.load(f)
        npz = np.load(emb_path, allow_pickle=False)
        emb_ids = list(npz["ids"])
        emb_vec = npz["embeddings"]
        emb_map = {tid: emb_vec[i] for i, tid in enumerate(emb_ids)}

        # Optional fame scores (если файл существует или указан явно)
        fame_map = {}
        if fame_path is None:
            # автоопределение по имени конференции
            auto = prog_path.with_name(prog_path.stem + "_fame.json")
            if auto.exists():
                fame_path = auto
        if fame_path:
            fame_path = Path(fame_path)
        if fame_path and fame_path.exists():
            with open(fame_path, encoding="utf-8") as f:
                fame_data = json.load(f)
            fame_map = fame_data.get("fame", {})

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
                fame=float(fame_map.get(t["id"], 0.0)),
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
    # Star-speaker effect: вес fame в effective utility пользователя
    w_fame: float = 0.0  # 0 = без fame (старое поведение); 0.3 = умеренный star effect
    # Compliance: насколько пользователь следует подсказке системы
    # 1.0 = строго из top-K (старое поведение)
    # 0.5 = с вероятностью 50% выбирает из ВСЕХ кандидатов слота независимо от подсказки
    user_compliance: float = 1.0


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


class LearnedPreferenceFn:
    """Обёртка над обученной моделью предпочтений f(persona_emb, talk_emb) → [0, 1].

    Внутри использует sklearn HistGradientBoostingRegressor, обученный на
    LLM-сгенерированной матрице оценок (см. scripts/train_preference_model.py).

    Кэширует предсказания по hash эмбеддингов для повторного использования.
    Поддерживает векторный batch_call.
    """

    def __init__(self, model_path):
        import pickle
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self._cache = {}  # (persona_hash, talk_hash) -> score

    @staticmethod
    def _hash_emb(emb: np.ndarray) -> int:
        # Хэш по полному массиву байтов — нет коллизий
        return hash(emb.astype(np.float32).tobytes())

    def __call__(self, persona_emb: np.ndarray, talk_emb: np.ndarray) -> float:
        ph = self._hash_emb(persona_emb)
        th = self._hash_emb(talk_emb)
        key = (ph, th)
        if key in self._cache:
            return self._cache[key]
        cosine = float(np.dot(persona_emb, talk_emb))
        feat = np.concatenate([persona_emb, talk_emb, [cosine]]).reshape(1, -1)
        pred = float(self.model.predict(feat)[0])
        result = max(0.0, min(1.0, pred))
        self._cache[key] = result
        return result

    def batch_call(self, persona_emb: np.ndarray, talk_embs: np.ndarray) -> np.ndarray:
        """Векторный вариант: один пользователь × M докладов."""
        cosines = talk_embs @ persona_emb  # (M,)
        n_talks = talk_embs.shape[0]
        persona_tile = np.tile(persona_emb, (n_talks, 1))
        feats = np.concatenate([
            persona_tile,
            talk_embs,
            cosines.reshape(-1, 1),
        ], axis=1)
        preds = self.model.predict(feats)
        return np.clip(preds, 0.0, 1.0)

    def precompute_all(self, personas: dict, talks: dict):
        """Заполняет кэш предсказаниями для всех (persona, talk) пар.

        Args:
            personas: {persona_id: persona_emb}
            talks: {talk_id: talk_emb}
        """
        # Build feature matrix for ALL pairs at once
        persona_ids = list(personas.keys())
        talk_ids = list(talks.keys())
        p_arr = np.stack([personas[pid] for pid in persona_ids])
        t_arr = np.stack([talks[tid] for tid in talk_ids])
        n_p = len(persona_ids)
        n_t = len(talk_ids)
        # Cartesian product
        p_repeat = np.repeat(p_arr, n_t, axis=0)  # (n_p * n_t, 384)
        t_tile = np.tile(t_arr, (n_p, 1))  # (n_p * n_t, 384)
        cosines = (p_repeat * t_tile).sum(axis=1, keepdims=True)  # (n_p * n_t, 1)
        feats = np.concatenate([p_repeat, t_tile, cosines], axis=1)
        preds = np.clip(self.model.predict(feats), 0.0, 1.0)
        # Fill cache
        idx = 0
        for pid in persona_ids:
            for tid in talk_ids:
                ph = self._hash_emb(personas[pid])
                th = self._hash_emb(talks[tid])
                self._cache[(ph, th)] = float(preds[idx])
                idx += 1


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
    relevance_fn: Optional[Callable] = None,  # (user_emb, talk_emb) -> float
) -> SimResult:
    """Прогоняет одну симуляцию по конференции для всех пользователей.

    state передаётся в политику между вызовами как dict с ключами:
      - hall_load: dict[(slot_id, hall_id) -> int] (пользователи уже вошедшие)
      - slot_id: текущий слот

    relevance_fn — функция вычисления релевантности; по умолчанию cosine.
    Может быть LearnedPreferenceFn или другая совместимая по интерфейсу.
    """
    if relevance_fn is None:
        relevance_fn = cosine_relevance
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
                "relevance_fn": relevance_fn,
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

            # Compliance: с вероятностью (1 - compliance) пользователь "не слушает" и
            # рассматривает ВСЕ доклады в слоте, не только top-K от политики.
            ignore_recommendation = (cfg.user_compliance < 1.0
                                     and rng.random() > cfg.user_compliance)
            consider_ids = list(slot.talk_ids) if ignore_recommendation else recs

            # формируем utility-вектор для softmax-выбора
            utils = []
            for tid in consider_ids:
                t = conf.talks[tid]
                rel = relevance_fn(user.embedding, t.embedding)
                hall = conf.halls[t.hall]
                load_frac = utilization(hall_load[(slot.id, hall.id)], hall.capacity)
                # Effective attractiveness = (1-w_fame)*relevance + w_fame*fame
                # Это создаёт star-speaker effect: даже не-релевантные звёздные доклады тянут пользователя.
                effective_rel = (1 - cfg.w_fame) * rel + cfg.w_fame * t.fame
                # Пользователь видит штраф за переполненный зал (косвенный сигнал)
                u = effective_rel - cfg.lambda_overflow * max(0.0, load_frac - 0.85)
                utils.append(u)
            utils = np.array(utils, dtype=np.float64)
            # Используем consider_ids вместо recs дальше
            recs = consider_ids

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
            chosen_rel_for_record = relevance_fn(user.embedding, chosen_talk.embedding)
            # Если у политики есть update_history (Sequential) — записываем фактический выбор
            if hasattr(policy, "update_history"):
                policy.update_history(user.id, chosen_id)
            result.steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=recs,
                chosen=chosen_id,
                chosen_relevance=chosen_rel_for_record,
                chosen_hall_load_before=load_before,
            ))

    # упаковка hall_load_per_slot
    per_slot: Dict[str, Dict[int, int]] = {}
    for (sid, hid), n in hall_load.items():
        per_slot.setdefault(sid, {})[hid] = n
    result.hall_load_per_slot = per_slot
    return result
