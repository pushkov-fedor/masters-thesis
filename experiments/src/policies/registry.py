"""Реестр активных политик основного эксперимента.

Зафиксированный состав (PROJECT_DESIGN §9, PROJECT_STATUS §7):

- ``no_policy``       — П1, контрольная;
- ``cosine``          — П2, по релевантности;
- ``capacity_aware``  — П3, с учётом загрузки;
- ``llm_ranker``      — П4, на основе языковой модели.

Старые политики (``random``, ``mmr``, ``dpp``, ``gnn``, ``ppo``/``ppo_v2``,
``sequential``, ``calibrated``, ``random_capacity_aware``,
``capacity_aware_mmr``, ``llm_ranker_state_aware``) физически не удаляются и
не переносятся, но в реестр не входят: они не участвуют в основном
эксперименте.

Параметры ``user_compliance`` / ``calibrated_compliance`` остаются legacy
вне основного эксперимента и здесь не упоминаются.

Lazy import LLMRankerPolicy. Класс импортируется только внутри
``active_policies()`` и только при ``include_llm=True``. Это даёт
возможность запускать smoke-прогоны (этап F), unit-тесты (этап I) и
toy-проверки без установленных ``openai`` / ``python-dotenv``, которые
нужны самой LLM-политике, но не нужны для П1-П3.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .base import BasePolicy
from .capacity_aware_policy import CapacityAwarePolicy
from .cosine_policy import CosinePolicy
from .no_policy import NoPolicy


ACTIVE_POLICY_NAMES: Tuple[str, ...] = (
    "no_policy",
    "cosine",
    "capacity_aware",
    "llm_ranker",
)


def active_policies(
    seed: int = 0,
    llm_ranker_kwargs: Optional[Mapping[str, Any]] = None,
    include_llm: bool = True,
) -> Dict[str, BasePolicy]:
    """Возвращает словарь ``{name: policy}`` для активных политик.

    Параметры
    ---------
    seed : int
        Зарезервирован для будущих стохастических политик основного состава.
        Текущие политики детерминированы по входу или внешнему кэшу,
        поэтому ``seed`` пока не используется. Сигнатура зафиксирована,
        чтобы вызывающий код не менялся при последующих расширениях.
    llm_ranker_kwargs : Mapping[str, Any] | None
        Опциональные параметры конструктора ``LLMRankerPolicy`` (модель,
        budget, путь к кэшу). Если ``None`` — используются дефолты политики.
    include_llm : bool, default True
        Если ``False``, в результат попадают только три детерминированные
        политики (no_policy, cosine, capacity_aware), без LLMRankerPolicy
        и без её транзитивных зависимостей (``openai``, ``python-dotenv``).
        Используется для smoke-прогонов и юнит-тестов, где LLM не нужен.
    """
    del seed  # явно: текущие активные политики его не потребляют
    pols: Dict[str, BasePolicy] = {
        "no_policy":      NoPolicy(),
        "cosine":         CosinePolicy(),
        "capacity_aware": CapacityAwarePolicy(),
    }
    if include_llm:
        # Lazy import: класс LLMRankerPolicy тянет openai + python-dotenv.
        # Импорт здесь, а не в module scope, чтобы импорт пакета policies
        # не требовал этих зависимостей для пути без LLM (smoke / тесты).
        from .llm_ranker_policy import LLMRankerPolicy
        llm_kwargs: Dict[str, Any] = dict(llm_ranker_kwargs) if llm_ranker_kwargs else {}
        pols["llm_ranker"] = LLMRankerPolicy(**llm_kwargs)
    return pols
