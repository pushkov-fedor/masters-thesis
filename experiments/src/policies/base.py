"""Базовый класс рекомендательной политики.

Единый контракт всех политик: метод `__call__(user, slot, conf, state) → list[talk_id]`
для обычных (локально-вычислимых) политик, и опциональный `acall(...)` для
асинхронных политик (LLM-ranker и подобных).

По умолчанию `acall` оборачивает `__call__` в `async def`, что позволяет
прогонять любую политику через единый асинхронный симулятор `simulate_async`.

Sequential-политика (использующая `update_history`) сохраняет этот хук как
дополнительный — он опциональный, симулятор проверяет наличие через `hasattr`.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


class BasePolicy:
    """Базовый класс. Любая политика обязана определить `__call__`.

    Дефолтная реализация `acall` тривиальна и вызывает `__call__`. LLM-политики
    переопределяют `acall` под реальный async-вызов.
    """

    name: str = "base"

    def __call__(self, *, user, slot, conf, state) -> List[str]:
        raise NotImplementedError

    async def acall(self, *, user, slot, conf, state) -> List[str]:
        return self.__call__(user=user, slot=slot, conf=conf, state=state)


@runtime_checkable
class PolicyProtocol(Protocol):
    """Тайп-чек контракт: всё, что можно вызвать как политику.

    Используется в аннотациях функций симулятора для статической проверки.
    `BasePolicy` и его наследники удовлетворяют этому Protocol.
    """
    name: str

    def __call__(self, *, user, slot, conf, state) -> List[str]: ...
