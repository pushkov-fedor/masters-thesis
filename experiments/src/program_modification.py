"""Оператор локальных модификаций программы Φ (этап N PIVOT_IMPLEMENTATION_PLAN r5).

Реализация строго по принятому memo M (`docs/spikes/spike_program_modification.md`).

Форма Φ — V1 + V5 + V0 (accepted Q-M1 — Q-M8 от 2026-05-07):

- **V1.** Pairwise swap двух докладов между **разными временными слотами одного
  дня** (терминологическое уточнение: PROJECT_DESIGN §7 «параллельные слоты
  одного дня» = в нашей реализации «разные временные слоты одного дня»).
  Меняем `slot_id` у двух талков; `hall` НЕ меняем; состав докладов сохраняется.
- **V5.** Если число валидных swap-кандидатов больше `k_max` — random subsample
  с фиксированным rng-seed (детерминизм при одинаковом seed).
- **V0.** P_0 (исходная программа) **не входит** в выдачу `enumerate_modifications`
  — её клеит вызывающий (LHS-генератор) как `program_variant = 0`.

Hard-validation speaker-конфликтов: если swap создаёт конфликт (один спикер в
двух разных талках одного слота), кандидат отклоняется. На конференциях, где
поле `speakers` в JSON отсутствует, `Talk.speakers = []` после загрузки и
validation становится **no-op** (см. memo M §3, accepted Q-M2). Это НЕ
доказательство отсутствия конфликтов — это отсутствие данных.

Хранение: in-memory `Conference` объекты + `SwapDescriptor` (accepted Q-M6).
Никаких отдельных JSON-файлов на диск.

API:

    enumerate_modifications(conf, k_max, rng, same_day_only=True)
        -> List[Tuple[Conference, SwapDescriptor]]

    has_speaker_conflict(conf) -> bool
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .simulator import Conference, Slot


@dataclass(frozen=True)
class SwapDescriptor:
    """Описание одной локальной перестановки между двумя временными слотами.

    Семантика swap: `t1` перемещается из `slot_a` в `slot_b`,
    `t2` — из `slot_b` в `slot_a`. `hall` обоих талков сохраняется.
    """
    slot_a: str  # slot_id, откуда приходит t1
    slot_b: str  # slot_id, откуда приходит t2
    t1: str      # talk_id, изначально в slot_a → перемещается в slot_b
    t2: str      # talk_id, изначально в slot_b → перемещается в slot_a


def has_speaker_conflict(conf: Conference) -> bool:
    """True, если в каком-либо слоте у двух разных талков есть общий спикер.

    На конференциях без данных о спикерах (`Talk.speakers = []`) функция
    тривиально возвращает False — это **no-op-валидация**, не подтверждение
    отсутствия конфликтов.
    """
    for slot in conf.slots:
        seen: set = set()
        for tid in slot.talk_ids:
            for sp in conf.talks[tid].speakers:
                if sp in seen:
                    return True
                seen.add(sp)
    return False


def _slot_day(slot: Slot) -> str:
    """Извлекает день из ISO datetime: '2026-01-01T10:00:00' → '2026-01-01'.

    Для не-ISO форматов или коротких строк возвращает первые 10 символов
    либо саму строку. На реальных датасетах (Mobius, Demo Day, toy_*) формат
    ISO 8601 — этого достаточно.
    """
    dt = slot.datetime or ""
    return dt[:10]


def _enumerate_all_pairs(
    conf: Conference, same_day_only: bool = True,
) -> List[SwapDescriptor]:
    """Все candidate-pairs swap'ов между разными временными слотами.

    Если `same_day_only=True` — только в пределах одного дня
    (PROJECT_DESIGN §7, accepted Q-M4). Если `False` — между любыми разными
    timeslot'ами (опция оставлена в API на будущее, в этапе N не активируется).
    """
    pairs: List[SwapDescriptor] = []
    if same_day_only:
        slots_by_day: dict = {}
        for s in conf.slots:
            slots_by_day.setdefault(_slot_day(s), []).append(s)
        for slots in slots_by_day.values():
            # PROJECT_DESIGN §7: «параллельные слоты одного дня» — в нашей
            # терминологии «разные временные слоты одного дня». См. memo M §3.
            for i, s_a in enumerate(slots):
                for s_b in slots[i + 1:]:
                    for t1 in s_a.talk_ids:
                        for t2 in s_b.talk_ids:
                            pairs.append(SwapDescriptor(
                                slot_a=s_a.id, slot_b=s_b.id, t1=t1, t2=t2,
                            ))
    else:
        for i, s_a in enumerate(conf.slots):
            for s_b in conf.slots[i + 1:]:
                for t1 in s_a.talk_ids:
                    for t2 in s_b.talk_ids:
                        pairs.append(SwapDescriptor(
                            slot_a=s_a.id, slot_b=s_b.id, t1=t1, t2=t2,
                        ))
    return pairs


def _apply_swap(conf: Conference, desc: SwapDescriptor) -> Conference:
    """Применяет swap к глубокой копии конференции.

    Меняет `slot_id` у двух талков; `hall` сохраняется. Перестраивает
    `Slot.talk_ids` обратным индексом по обновлённым `Talk.slot_id`.
    """
    cloned = copy.deepcopy(conf)
    t1 = cloned.talks[desc.t1]
    t2 = cloned.talks[desc.t2]
    t1.slot_id, t2.slot_id = desc.slot_b, desc.slot_a
    # Перестраиваем Slot.talk_ids
    talk_ids_by_slot: dict = {s.id: [] for s in cloned.slots}
    for tid, t in cloned.talks.items():
        talk_ids_by_slot.setdefault(t.slot_id, []).append(tid)
    for s in cloned.slots:
        s.talk_ids = sorted(talk_ids_by_slot.get(s.id, []))
    return cloned


def enumerate_modifications(
    conf: Conference,
    k_max: int,
    rng: np.random.Generator,
    same_day_only: bool = True,
) -> List[Tuple[Conference, SwapDescriptor]]:
    """Возвращает до k_max валидных одиночных swap-модификаций программы.

    P_0 (исходная) НЕ включается в результат — её добавляет вызывающий
    (LHS-генератор) как `program_variant = 0`.

    Каждый возвращаемый Conference имеет:
    - тот же набор `talks` (id-set неизменен);
    - те же `halls`;
    - те же `slots` по id и размерам (`len(slot.talk_ids)` сохраняется);
    - изменён `slot_id` ровно у двух талков относительно исходного.

    Speaker-конфликты в результате swap отсекаются (hard validation).
    Если `len(valid_pool) <= k_max` — возвращается весь pool.
    Если `len(valid_pool) > k_max` — `rng.choice(...)` сэмплирует k_max
    кандидатов без повторов; результат детерминистичен при одном rng-seed.

    Если `k_max <= 0` или нет ни одной валидной пары (например, программа
    из одного timeslot) — возвращается пустой список.
    """
    if k_max <= 0:
        return []
    candidates = _enumerate_all_pairs(conf, same_day_only=same_day_only)
    valid: List[Tuple[Conference, SwapDescriptor]] = []
    for desc in candidates:
        modified = _apply_swap(conf, desc)
        if not has_speaker_conflict(modified):
            valid.append((modified, desc))
    if not valid:
        return []
    if len(valid) <= k_max:
        return valid
    idx = rng.choice(len(valid), size=k_max, replace=False)
    # sorted чтобы порядок был стабильным при одном seed (rng.choice уже даёт
    # детерминированный набор, но сортировка делает выдачу более читаемой
    # в логах / тестах).
    return [valid[i] for i in sorted(idx)]
