"""Параметрический симулятор для задачи поддержки формирования программы конференции.

Постановка по PROJECT_DESIGN §7 + accepted decision спайка модели поведения
(``docs/spikes/spike_behavior_model.md``):

- состояние s_t — вектор заполненности залов в текущем тайм-слоте;
- действие a_t — список рекомендаций L_{i,t} для каждого пользователя i;
- consider_ids = весь slot всегда; политика НЕ ограничивает choice set;
- модель выбора — multinomial logit с температурой τ + outside option (skip);
- utility: U = w_rel * effective_rel + w_rec * 1{t in recs};
- capacity-effect живёт ТОЛЬКО в политике П3 (CapacityAwarePolicy), не в utility;
- gossip — отдельный плановый инкремент этапов J–L PIVOT_IMPLEMENTATION_PLAN.

Common random numbers (CRN). Внутри одного слота используются два независимых
RNG-потока: ``choice_rng`` для финального softmax-choice и legacy compliance
roll, ``policy_rng`` (передаётся в state) для стохастичности самих политик.
Это обеспечивает EC3 (PROJECT_DESIGN §11): при w_rec = 0 финальный выбор не
зависит от политики, потому что utility сводится к ``w_rel * effective_rel``,
а ``choice_rng`` идёт по той же траектории независимо от стохастичности
политики.

Релевантность: косинус между эмбеддингом профиля пользователя и эмбеддингом
доклада (см. ``cosine_relevance``); ML-альтернатива через ``LearnedPreferenceFn``.
"""
from __future__ import annotations

import asyncio
import json
import math
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
    # Список спикеров (etap N spike_program_modification accepted Q-M7).
    # Парсится из JSON-поля `speakers` (comma-separated string) или []
    # при отсутствии. Используется оператором Φ для hard-валидации
    # speaker-конфликтов; default [] делает validation no-op для toy / ITC /
    # Meetup, у которых поля speakers нет.
    speakers: List[str] = field(default_factory=list)


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
            # speakers: comma-separated string в Mobius / Demo Day; список в
            # будущих fixtures; отсутствует в toy / ITC / Meetup. При
            # отсутствии — пустой список (speaker-validation становится no-op,
            # но это НЕ доказательство отсутствия конфликтов).
            speakers_raw = t.get("speakers", "")
            if isinstance(speakers_raw, str):
                speakers_list = [s.strip() for s in speakers_raw.split(",")
                                 if s.strip()]
            elif isinstance(speakers_raw, list):
                speakers_list = [str(s).strip() for s in speakers_raw if s]
            else:
                speakers_list = []
            talks[t["id"]] = Talk(
                id=t["id"],
                title=t["title"],
                hall=int(t["hall"]),
                slot_id=t["slot_id"],
                category=t.get("category", "Other"),
                abstract=t.get("abstract", ""),
                embedding=emb_map[t["id"]],
                fame=float(fame_map.get(t["id"], 0.0)),
                speakers=speakers_list,
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
    """Параметры симуляции (модель поведения участника).

    Базовая utility-форма (PROJECT_DESIGN §7, accepted в spike_behavior_model
    и spike_gossip §8 / §9):

        U(t | i, hat_pi) = w_rel * effective_rel(i, t)
                         + w_rec * 1{t in recs}
                         + w_gossip * gossip(t, L_t)
        effective_rel(i, t) = (1 - w_fame) * cos(profile_i, t) + w_fame * t.fame
        gossip(t, L_t)      = log(1 + count_t) / log(1 + N_users)   (V5 log_count)
                              где count_t = local_choice_count[t] — счётчик
                              пользователей в текущем слоте, выбравших доклад t,
                              до текущего юзера; N_users = len(user_order).
        consider_ids = весь slot   (политика НЕ ограничивает choice set)
        capacity-effect — только в политике П3 capacity_aware (НЕ в utility)
        outside option — вероятность skip = p_skip_base

    Гибкость по весам не задаётся жёстко в коде: для основной матрицы
    эксперимента пользователь подтвердил симплексную нормировку
    w_rel + w_rec + w_gossip = 1, но SimConfig не запрещает другие комбинации
    (например, для sensitivity-sweep). Контракт CRN-инвариантности EC3
    формулируется как «при w_rec = 0 политика не влияет на utility», что
    выполняется при любых значениях w_rel и w_gossip независимо от нормировки.

    Gossip-инвариант (accepted в spike_gossip §11.L.1, проверка в этапе L):
    при w_gossip = 0 траектории `chosen_id` пословно совпадают с траекториями
    реализации этапа E (без gossip-канала); реализуется тривиально, потому что
    cfg.w_gossip * gossip(...) = 0 тождественно при w_gossip = 0, и gossip-член
    не использует RNG (не сдвигает choice_rng).
    """
    tau: float = 0.7              # температура softmax для выбора пользователя
    p_skip_base: float = 0.10     # вероятность skip как outside option (LCM4Rec-канон)
    K: int = 3                    # размер выдачи (top-K рекомендаций)
    seed: int = 0
    # Веса базовой модели поведения (rel + rec + gossip). Default: w_rel = 0.7,
    # w_rec = 0.3, w_gossip = 0 (default — gossip-канал отключён, baseline E).
    w_rel: float = 0.7
    w_rec: float = 0.3
    w_gossip: float = 0.0
    # Star-speaker effect: вес fame внутри effective_rel.
    w_fame: float = 0.0  # 0 = без fame; 0.3 = умеренный star effect
    # ----- DEPRECATED поля (этап E PIVOT_IMPLEMENTATION_PLAN) -----
    # lambda_overflow в utility больше НЕ используется: capacity-effect вынесен
    # в политику CapacityAwarePolicy (П3). Поле сохранено только для обратной
    # совместимости с legacy-скриптами одномерных sweep'ов (capacity_sensitivity,
    # run_compliance_sweep, run_experiments, make_plots), которые не входят в
    # основной эксперимент с этапа F. Любое значение игнорируется в utility.
    # Не использовать в новом коде.
    lambda_overflow: float = 0.0
    # ----- Legacy-параметры compliance (вне основного эксперимента) -----
    # Default: оба выключены. Активны только при явном указании; в основной
    # матрице LHS-конфигураций не используются. Назначение: distribution-match
    # на Meetup-RSVPs как distribution-level якорь (PROJECT_STATUS §8).
    # Контракт: при default-значениях (use_calibrated_compliance=False,
    # user_compliance=1.0) основной path игнорирует обе ветки compliance.
    user_compliance: float = 1.0
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

    parallel_safe=True (несколько слотов параллельно) → отключается
    update_history, т.к. история политики при параллельных слотах
    перепутается.

    Модель поведения участника:

        consider_ids = list(slot.talk_ids)            # весь slot всегда
        rec_indicator(t) = 1 if t in recs else 0
        effective_rel(t) = (1 - w_fame) * rel + w_fame * t.fame
        gossip(t)        = log(1 + local_choice_count[t]) / log(1 + N_users)
                           # V5 log_count, accepted spike_gossip §8
        U(t) = w_rel * effective_rel(t)
             + w_rec * rec_indicator(t)
             + w_gossip * gossip(t)
        P(choose t) = (1 - p_skip) * softmax(U / tau) ⊕ p_skip   (skip = no-choice)

    Capacity-effect в utility отсутствует. Capacity-канал — ответственность
    политики П3 (CapacityAwarePolicy.score = sim - alpha * load_frac).

    Gossip-канал считает по `local_choice_count[talk_id]` (per-talk счётчик),
    параллельный `local_load[hall_id]` (per-hall счётчик для capacity-канала
    политики). На текущих программах (один доклад на зал в слот) численно они
    совпадают, но семантически разные — это сознательное разделение каналов
    (см. spike_gossip §6.V2 risk).

    Common random numbers (CRN):
      - ``choice_rng``  — финальный softmax-choice и legacy compliance roll;
      - ``policy_rng``  — стохастичность самой политики (передаётся в state).
    Разделение потоков обеспечивает: при ``cfg.w_rec == 0`` финальный choice
    не зависит от политики (rec_indicator * 0 = 0), и ``choice_rng`` идёт по
    той же траектории независимо от того, потребляла ли политика ``policy_rng``.

    Legacy compliance:
      - ``cfg.use_calibrated_compliance`` (default False) — трёхтипная B/C/A
        модель на Meetup-RSVPs;
      - ``cfg.user_compliance`` (default 1.0) — Bernoulli compliance.
    Обе ветки активируются ТОЛЬКО при явных значениях, отличных от default;
    в основной матрице эксперимента не используются (PROJECT_STATUS §8).
    Когда они активны, они работают на уровне ``consider_ids`` поверх новой
    utility-формулы (compliant — ограничивает выбор top-K от политики,
    star_chaser — forced choice по argmax fame, curious / non-compliant —
    consider_ids = весь slot).
    """
    choice_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx)
    policy_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx + 31)
    local_load: Dict[int, int] = {h.id: 0 for h in conf.halls.values()}
    # V5 log_count gossip: per-talk счётчик выбравших, параллельный local_load
    # (per-hall). При cfg.w_gossip = 0 значения не используются.
    local_choice_count: Dict[str, int] = {tid: 0 for tid in slot.talk_ids}
    n_users_in_slot = len(user_order)
    slot_steps: List[StepRecord] = []

    if not slot.talk_ids:
        return slot.id, slot_steps, local_load

    n_cands = len(slot.talk_ids)
    effective_K = max(1, min(cfg.K, n_cands - 1)) if n_cands >= 2 else 1

    legacy_compliance_active = (
        cfg.use_calibrated_compliance or cfg.user_compliance < 1.0
    )

    for user in user_order:
        state = {
            "hall_load": {(slot.id, hid): occ for hid, occ in local_load.items()},
            "slot_id": slot.id,
            "K": effective_K,
            "relevance_fn": relevance_fn,
            "policy_rng": policy_rng,  # CRN: отдельный поток для стохастических политик
        }
        recs = await policy.acall(user=user, slot=slot, conf=conf, state=state)
        recs = [r for r in recs if r in slot.talk_ids][:effective_K]

        # Default (новая модель): consider_ids = весь slot всегда; политика
        # влияет на utility через rec_indicator, не через ограничение
        # consideration set.
        consider_ids: List[str] = list(slot.talk_ids)
        forced_choice_id: Optional[str] = None

        if legacy_compliance_active:
            # Legacy ветка для distribution-match Meetup; вне основного
            # эксперимента (PROJECT_STATUS §8). Активируется только при
            # явном указании флагов (use_calibrated_compliance=True или
            # user_compliance<1.0). RNG этой ветки — choice_rng, чтобы
            # стохастичность политики не сдвигала её результаты.
            if cfg.use_calibrated_compliance:
                roll = choice_rng.random()
                if roll < cfg.alpha_compliant:
                    if recs:
                        consider_ids = recs
                elif roll < cfg.alpha_compliant + cfg.alpha_starchaser:
                    fame_scores = [(tid, conf.talks[tid].fame)
                                   for tid in slot.talk_ids]
                    max_fame = max(s for _, s in fame_scores)
                    if max_fame > 0:
                        forced_choice_id = max(fame_scores, key=lambda x: x[1])[0]
                    else:
                        forced_choice_id = max(
                            slot.talk_ids,
                            key=lambda tid: relevance_fn(
                                user.embedding, conf.talks[tid].embedding
                            ),
                        )
            else:  # cfg.user_compliance < 1.0
                ignore_recommendation = choice_rng.random() > cfg.user_compliance
                if not ignore_recommendation and recs:
                    consider_ids = recs

        if forced_choice_id is not None:
            chosen_id = forced_choice_id
            chosen_talk = conf.talks[chosen_id]
            chosen_hall = conf.halls[chosen_talk.hall]
            load_before = utilization(local_load[chosen_hall.id],
                                      conf.capacity_at(slot.id, chosen_hall.id))
            local_load[chosen_hall.id] += 1
            local_choice_count[chosen_id] = local_choice_count.get(chosen_id, 0) + 1
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

        # Новая utility-формула:
        #   U = w_rel * effective_rel + w_rec * 1{t in recs} + w_gossip * gossip(t).
        # gossip(t) = log(1 + count_t) / log(1 + N_users)   (V5 log_count, accepted
        # spike_gossip §8). Capacity-канал в utility отсутствует.
        recs_set = set(recs)
        # Знаменатель log(1 + N_users) фиксируется один раз; деления на 0 не
        # бывает, потому что слот без юзеров не доходит до этой ветки (early
        # return при `not slot.talk_ids`), а условие `n_users_in_slot >= 1`
        # обеспечивает log(1 + N) > 0 при N ≥ 1.
        log_n_plus_one = math.log1p(n_users_in_slot) if n_users_in_slot >= 1 else 0.0
        utils_list: List[float] = []
        for tid in consider_ids:
            t = conf.talks[tid]
            rel = relevance_fn(user.embedding, t.embedding)
            effective_rel = (1.0 - cfg.w_fame) * rel + cfg.w_fame * t.fame
            rec_indicator = 1.0 if tid in recs_set else 0.0
            count_t = local_choice_count.get(tid, 0)
            gossip = (math.log1p(count_t) / log_n_plus_one) if log_n_plus_one > 0 else 0.0
            u = (cfg.w_rel * effective_rel
                 + cfg.w_rec * rec_indicator
                 + cfg.w_gossip * gossip)
            utils_list.append(u)
        utils = np.array(utils_list, dtype=np.float64)

        scaled = utils / max(cfg.tau, 1e-6)
        scaled = scaled - scaled.max()
        exps = np.exp(scaled)
        sum_exp = exps.sum()
        p_skip = cfg.p_skip_base
        probs_recs = (1.0 - p_skip) * (exps / sum_exp)
        probs = np.concatenate([probs_recs, [p_skip]])
        probs = probs / probs.sum()

        choice_idx = int(choice_rng.choice(len(probs), p=probs))
        if choice_idx == len(probs) - 1:
            slot_steps.append(StepRecord(
                slot_id=slot.id, user_id=user.id, recommended=recs,
                chosen=None, chosen_relevance=0.0, chosen_hall_load_before=0.0,
            ))
            continue

        chosen_id = consider_ids[choice_idx]
        chosen_talk = conf.talks[chosen_id]
        chosen_hall = conf.halls[chosen_talk.hall]
        load_before = utilization(local_load[chosen_hall.id],
                                  conf.capacity_at(slot.id, chosen_hall.id))
        local_load[chosen_hall.id] += 1
        local_choice_count[chosen_id] = local_choice_count.get(chosen_id, 0) + 1
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
