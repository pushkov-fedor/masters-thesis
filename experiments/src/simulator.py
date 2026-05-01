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

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

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
    hall_capacities: Optional[Dict[int, int]] = None  # per-slot override; None → fallback на Hall.capacity


@dataclass
class Conference:
    name: str
    talks: Dict[str, Talk]
    halls: Dict[int, Hall]
    slots: List[Slot]  # упорядоченный список

    def __post_init__(self):
        self._slot_by_id = {s.id: s for s in self.slots}

    def capacity_at(self, slot_id: str, hall_id: int) -> int:
        """Вместимость зала в конкретном слоте.

        Если у слота задано переопределение `hall_capacities` (контролируемая
        постановка эксперимента, см. Demo Day), используется оно; иначе —
        глобальная вместимость Hall.capacity (Mobius, ITC, Meetup).
        """
        s = self._slot_by_id.get(slot_id)
        if s is not None and s.hall_capacities and hall_id in s.hall_capacities:
            return int(s.hall_capacities[hall_id])
        return int(self.halls[hall_id].capacity)

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
            Slot(
                id=s["id"],
                datetime=s["datetime"],
                talk_ids=slot_index.get(s["id"], []),
                hall_capacities=(
                    {int(k): int(v) for k, v in s["hall_capacities"].items()}
                    if s.get("hall_capacities") else None
                ),
            )
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
    # Compliance: насколько пользователь следует подсказке системы (legacy mode).
    # 1.0 = строго из top-K (старое поведение)
    # 0.5 = с вероятностью 50% выбирает из ВСЕХ кандидатов слота независимо от подсказки
    user_compliance: float = 1.0
    # Калиброванная трёхтипная модель compliance (B/C/A).
    # Доли получены из калибровки на Meetup RSVPs (scripts/calibrate_compliance_meetup.py):
    #   compliant 71.7% / star-chaser 21.3% / curious 7.0% (на non-tie слотах);
    #   55.4% / 32.5% / 12.2% (на всех парах с tie 50/50).
    # Если use_calibrated_compliance=True, перекрывает поле user_compliance:
    #   compliant пользователь идёт по top-K от политики (как user_compliance=1.0),
    #   star-chaser игнорирует политику и идёт на argmax по fame в слоте,
    #   curious игнорирует политику и выбирает softmax по effective_utility.
    use_calibrated_compliance: bool = False
    alpha_compliant: float = 0.717
    alpha_starchaser: float = 0.213
    alpha_curious: float = 0.070


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
    policy,
    cfg: SimConfig,
    relevance_fn: Optional[Callable] = None,
) -> SimResult:
    """Sync-обёртка над `simulate_async` для совместимости с прежним кодом.

    Использует `asyncio.run` — нельзя вызывать из уже активного event loop
    (из юпитера, например). В таком случае использовать `simulate_async` напрямую.
    """
    return asyncio.run(simulate_async(conf, users, policy, cfg, relevance_fn))


async def _process_one_slot(
    conf: Conference,
    slot: Slot,
    slot_idx: int,
    user_order: List[UserProfile],
    policy,
    cfg: SimConfig,
    relevance_fn: Callable,
    parallel_safe: bool,
):
    """Обрабатывает один слот: пользователи внутри строго последовательно.

    Возвращает (slot_id, steps_list, local_load_dict).

    parallel_safe=True (несколько слотов параллельно) → отключается update_history,
    т.к. история политики при параллельных слотах перепутается.
    """
    slot_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx)
    local_load: Dict[int, int] = {h.id: 0 for h in conf.halls.values()}
    slot_steps: List[StepRecord] = []

    if not slot.talk_ids:
        return slot.id, slot_steps, local_load

    n_cands = len(slot.talk_ids)
    effective_K = max(1, min(cfg.K, n_cands - 1)) if n_cands >= 2 else 1

    for user in user_order:
        state = {
            "hall_load": {(slot.id, hid): occ for hid, occ in local_load.items()},
            "slot_id": slot.id,
            "K": effective_K,
            "relevance_fn": relevance_fn,
        }
        recs = await policy.acall(user=user, slot=slot, conf=conf, state=state)
        recs = [r for r in recs if r in slot.talk_ids][:effective_K]
        if not recs:
            slot_steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=[],
                chosen=None, chosen_relevance=0.0, chosen_hall_load_before=0.0,
            ))
            continue

        consider_ids = recs
        forced_choice_id: Optional[str] = None
        if cfg.use_calibrated_compliance:
            roll = slot_rng.random()
            if roll < cfg.alpha_compliant:
                consider_ids = recs
            elif roll < cfg.alpha_compliant + cfg.alpha_starchaser:
                fame_scores = [(tid, conf.talks[tid].fame) for tid in slot.talk_ids]
                max_fame = max(s for _, s in fame_scores)
                if max_fame > 0:
                    forced_choice_id = max(fame_scores, key=lambda x: x[1])[0]
                else:
                    forced_choice_id = max(
                        slot.talk_ids,
                        key=lambda tid: relevance_fn(user.embedding,
                                                    conf.talks[tid].embedding),
                    )
            else:
                consider_ids = list(slot.talk_ids)
        else:
            ignore_recommendation = (cfg.user_compliance < 1.0
                                     and slot_rng.random() > cfg.user_compliance)
            consider_ids = list(slot.talk_ids) if ignore_recommendation else recs

        if forced_choice_id is not None:
            chosen_id = forced_choice_id
            chosen_talk = conf.talks[chosen_id]
            chosen_hall = conf.halls[chosen_talk.hall]
            load_before = utilization(local_load[chosen_hall.id],
                                      conf.capacity_at(slot.id, chosen_hall.id))
            local_load[chosen_hall.id] += 1
            chosen_rel_for_record = relevance_fn(user.embedding, chosen_talk.embedding)
            if not parallel_safe and hasattr(policy, "update_history"):
                policy.update_history(user.id, chosen_id)
            slot_steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=recs,
                chosen=chosen_id,
                chosen_relevance=chosen_rel_for_record,
                chosen_hall_load_before=load_before,
            ))
            continue

        utils = []
        for tid in consider_ids:
            t = conf.talks[tid]
            rel = relevance_fn(user.embedding, t.embedding)
            hall = conf.halls[t.hall]
            load_frac = utilization(local_load[hall.id],
                                    conf.capacity_at(slot.id, hall.id))
            effective_rel = (1 - cfg.w_fame) * rel + cfg.w_fame * t.fame
            u = effective_rel - cfg.lambda_overflow * max(0.0, load_frac - 0.85)
            utils.append(u)
        utils = np.array(utils, dtype=np.float64)
        recs = consider_ids

        scaled = utils / max(cfg.tau, 1e-6)
        scaled = scaled - scaled.max()
        exps = np.exp(scaled)
        sum_exp = exps.sum()
        p_skip = cfg.p_skip_base
        probs_recs = (1.0 - p_skip) * (exps / sum_exp)
        probs = np.concatenate([probs_recs, [p_skip]])
        probs = probs / probs.sum()

        choice_idx = slot_rng.choice(len(probs), p=probs)
        if choice_idx == len(probs) - 1:
            slot_steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=recs,
                chosen=None, chosen_relevance=0.0, chosen_hall_load_before=0.0,
            ))
            continue

        chosen_id = recs[choice_idx]
        chosen_talk = conf.talks[chosen_id]
        chosen_hall = conf.halls[chosen_talk.hall]
        load_before = utilization(local_load[chosen_hall.id],
                                  conf.capacity_at(slot.id, chosen_hall.id))
        local_load[chosen_hall.id] += 1
        chosen_rel_for_record = relevance_fn(user.embedding, chosen_talk.embedding)
        if not parallel_safe and hasattr(policy, "update_history"):
            policy.update_history(user.id, chosen_id)
        slot_steps.append(StepRecord(
            slot_id=slot.id, user_id=user.id, recommended=recs,
            chosen=chosen_id,
            chosen_relevance=chosen_rel_for_record,
            chosen_hall_load_before=load_before,
        ))

    return slot.id, slot_steps, local_load


async def simulate_async(
    conf: Conference,
    users: Sequence[UserProfile],
    policy,
    cfg: SimConfig,
    relevance_fn: Optional[Callable] = None,
    slot_concurrency: int = 1,
) -> SimResult:
    """Единый async-симулятор.

    slot_concurrency:
      = 1 — слоты обрабатываются последовательно (поведение совпадает с прежней
            sync-версией). Поддерживает update_history (Sequential).
      > 1 — слоты обрабатываются параллельно с семафором (полезно для LLM-политик
            с сетевой латентностью). Внутри слота пользователи остаются
            последовательными → state-awareness сохранена. update_history НЕ
            вызывается, чтобы история не перепутывалась между параллельными слотами.

    Любая политика, наследующая `BasePolicy`, работает через свой `acall`. Для
    локальных политик `acall` тривиально оборачивает `__call__`.
    """
    if relevance_fn is None:
        relevance_fn = cosine_relevance

    rng_global = np.random.default_rng(cfg.seed)
    user_order = list(users)
    rng_global.shuffle(user_order)

    parallel_safe = slot_concurrency > 1
    if slot_concurrency <= 1:
        # Последовательная обработка слотов — детерминистично, поддерживает update_history.
        per_slot_results = []
        for i, slot in enumerate(conf.slots):
            res = await _process_one_slot(
                conf, slot, i, user_order, policy, cfg, relevance_fn,
                parallel_safe=False,
            )
            per_slot_results.append(res)
    else:
        sem = asyncio.Semaphore(slot_concurrency)

        async def gated(slot_idx, slot):
            async with sem:
                return await _process_one_slot(
                    conf, slot, slot_idx, user_order, policy, cfg, relevance_fn,
                    parallel_safe=True,
                )

        tasks = [gated(i, slot) for i, slot in enumerate(conf.slots)]
        per_slot_results = await asyncio.gather(*tasks)

    result = SimResult()
    for sid, steps, load in per_slot_results:
        result.steps.extend(steps)
        result.hall_load_per_slot[sid] = load
    return result


# Backward-compatible alias
async def simulate_async_slots(
    conf: Conference,
    users: Sequence[UserProfile],
    policy,
    cfg: SimConfig,
    relevance_fn: Optional[Callable] = None,
    concurrency: int = 10,
) -> SimResult:
    """Совместимостный псевдоним: вызывает `simulate_async` со slot_concurrency."""
    return await simulate_async(conf, users, policy, cfg, relevance_fn,
                                 slot_concurrency=concurrency)
