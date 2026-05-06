"""П1 — контрольная политика основного эксперимента.

См. PROJECT_DESIGN §9: рекомендация участнику не предъявляется, компонент
``w_rec`` из функции полезности исключён. Политика играет роль точки отсчёта
при попарном сравнении в каждой точке гиперкуба.

На этапе B политика возвращает пустой список. На этапе E связь с моделью
поведения участника фиксируется на стороне ``simulator.py``: для ``no_policy``
рекомендательный канал не вкладывается в utility.
"""
from __future__ import annotations

from typing import List

from .base import BasePolicy


class NoPolicy(BasePolicy):
    name = "no_policy"

    def __init__(self, **_):
        pass

    def __call__(self, *, user, slot, conf, state) -> List[str]:
        return []
