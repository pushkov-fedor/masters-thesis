"""Big Five personality для агентов — адаптация PUB (arXiv:2506.04551).

Каждый агент имеет личностный профиль (O, C, E, A, N), который модулирует
его выбор в decide(). Это создаёт неоднородную популяцию — без personality
все агенты с одной персоной ведут себя одинаково.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class BigFive:
    """Большая пятёрка [0, 1] значений."""
    openness: float        # склонность к новому, нишевому
    conscientiousness: float  # дисциплина, склонность к топ-релевантному
    extraversion: float    # склонность к толпе, популярным докладам
    agreeableness: float   # склонность копировать выбор друзей (social)
    neuroticism: float     # тревожность, склонность пропускать слот

    @classmethod
    def from_seed(cls, seed: int) -> "BigFive":
        """Сэмплинг из приближённо-нормального распределения через seed."""
        rng = np.random.default_rng(seed)
        # Усреднение трёх uniform даёт приближённо-нормальное в [0, 1]
        vals = rng.uniform(0, 1, size=(5, 3)).mean(axis=1)
        # Растягиваем чтобы покрыть [0.1, 0.9] — избегаем экстремумов
        vals = 0.1 + vals * 0.8
        return cls(
            openness=float(vals[0]),
            conscientiousness=float(vals[1]),
            extraversion=float(vals[2]),
            agreeableness=float(vals[3]),
            neuroticism=float(vals[4]),
        )

    @classmethod
    def from_persona_text(cls, text: str, seed: int) -> "BigFive":
        """Эвристическая инференция Big Five из текста персоны.

        Это упрощённая версия PUB-style inference: ищем ключевые слова,
        корректируем base-распределение seed'ом.
        """
        base = cls.from_seed(seed)
        text_lower = text.lower()

        # Простые лексические паттерны
        # Openness: открытость к новому
        if re.search(r"любозн|новых|изуча|исследова|эксперимент", text_lower):
            base.openness = min(1.0, base.openness + 0.2)
        if re.search(r"консервативн|стабильн|проверенн", text_lower):
            base.openness = max(0.1, base.openness - 0.2)

        # Conscientiousness: дисциплина
        if re.search(r"архитект|тимлид|сениор|senior|lead", text_lower):
            base.conscientiousness = min(1.0, base.conscientiousness + 0.2)
        if re.search(r"джун|junior|начинающ|стажёр", text_lower):
            base.conscientiousness = max(0.1, base.conscientiousness - 0.15)

        # Extraversion: тяга к толпе
        if re.search(r"девре|devrel|митап|спикер|комьюнити", text_lower):
            base.extraversion = min(1.0, base.extraversion + 0.25)
        if re.search(r"индивидуа|самостоятель|сольно", text_lower):
            base.extraversion = max(0.1, base.extraversion - 0.2)

        # Neuroticism: тревожность
        if re.search(r"осторожн|тревожн|стесняет|перфекциониз", text_lower):
            base.neuroticism = min(1.0, base.neuroticism + 0.2)

        return base

    def render(self) -> str:
        """Краткое описание для промпта."""
        traits = []
        if self.openness > 0.65:
            traits.append("любит новое и нишевое")
        elif self.openness < 0.35:
            traits.append("предпочитает проверенные темы")
        if self.conscientiousness > 0.65:
            traits.append("дисциплинирован")
        if self.extraversion > 0.65:
            traits.append("тянется к большим залам")
        elif self.extraversion < 0.35:
            traits.append("избегает толп")
        if self.agreeableness > 0.65:
            traits.append("часто следует за коллегами")
        if self.neuroticism > 0.65:
            traits.append("склонен делать перерыв")
        return "; ".join(traits) if traits else "уравновешенный"

    def relevance_modifier(self, base_relevance: float, hall_load_frac: float,
                           friends_attending_frac: float) -> float:
        """Personality модулирует базовую релевантность.

        Возвращает модифицированную relevance с учётом черт.
        """
        modified = base_relevance
        # Высокая extraversion → bonus за большие залы
        modified += 0.15 * self.extraversion * (hall_load_frac - 0.5)
        # Высокая agreeableness → bonus за social copying
        modified += 0.20 * self.agreeableness * friends_attending_frac
        # High openness → bonus к нишевым (когда зал малозагружен — индикатор «не popular» темы)
        if hall_load_frac < 0.4:
            modified += 0.10 * self.openness
        # High conscientiousness → стабильнее держится релевантности
        modified += 0.05 * self.conscientiousness * base_relevance
        return float(np.clip(modified, 0.0, 1.5))

    def skip_propensity_modifier(self, base_skip: float, current_fatigue: float) -> float:
        """Personality + fatigue → итоговая склонность пропустить слот."""
        modified = base_skip
        # High neuroticism → выше skip
        modified += 0.10 * self.neuroticism
        # Fatigue effect
        modified += 0.30 * current_fatigue
        # Conscientiousness снижает skip
        modified -= 0.05 * self.conscientiousness
        return float(np.clip(modified, 0.0, 0.95))
