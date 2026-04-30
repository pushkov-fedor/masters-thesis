"""Простая модель усталости агента."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FatigueState:
    """Усталость растёт с каждым слотом, восстанавливается между днями."""
    level: float = 0.0  # [0, 1]
    last_date: str = ""

    def update_after_decision(self, decision: str, current_date: str):
        """Обновить уровень усталости после решения в слоте.

        - attended: +0.10 (внимание на докладе утомительно)
        - skipped: +0.05 (просто отдохнул в кулуарах, тоже немного устал)
        - между днями: восстановление 60%
        """
        if self.last_date and current_date != self.last_date:
            # Между днями восстановление
            self.level = max(0.0, self.level * 0.4)
        increment = 0.05 if decision == "skip" else 0.10
        self.level = min(1.0, self.level + increment)
        self.last_date = current_date

    def render(self) -> str:
        if self.level < 0.25:
            return "свежий"
        elif self.level < 0.5:
            return "слегка устал"
        elif self.level < 0.75:
            return "заметно устал"
        else:
            return "очень устал"
