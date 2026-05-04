"""Простой LLM-агент для симуляции выбора участника на параллельных слотах конференции.

Минимальная архитектура (Уровень 2 по обзору):
- Профиль участника как текст.
- История посещённых сегодня докладов (для тематической инерции и усталости).
- Текущая загрузка залов передаётся в промпт.
- Прямой запрос к LLM: «что выбираешь?» → id доклада или skip.

Без Big Five, без графа знакомств, без рефлексии — это вкладывается в эффект через
естественный приор LLM, а не явный механизм. Это позволит проверить, какие эффекты
LLM-симулятор воспроизводит «бесплатно» по сравнению с параметрическим.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT = """Ты — реальный участник IT-конференции в зале. Тебе нужно решить, на какой из параллельных докладов сейчас идти. На реальной конференции ты учитываешь:
1. Свой профиль и интересы.
2. Что уже посещал сегодня (не повторять одни темы, к концу дня усталость).
3. Рекомендацию системы, если она есть (не обязан слушать).

Ответ строго в формате JSON: {"choice": "<talk_id или 'skip'>", "reason": "<1 короткое предложение>"}
Никакого текста до или после JSON."""


USER_PROMPT_TEMPLATE = """Профиль:
{profile}

Уже посетил сегодня ({n_visited}):
{history}

Сейчас параллельно ({n_options} вариантов в слоте {slot_id}):
{options_block}
{rec_block}

Что выбираешь? Только JSON."""


@dataclass
class LLMAgentDecision:
    agent_id: str
    slot_id: str
    chosen: str | None  # talk_id или None если skip
    reason: str
    cost_usd: float = 0.0


@dataclass
class LLMAgent:
    """Один участник конференции, делающий выбор через LLM."""
    agent_id: str
    profile: str
    history: list[dict] = field(default_factory=list)  # [{slot_id, talk_id, title}]

    def render_history(self) -> str:
        if not self.history:
            return "  (пока ничего)"
        return "\n".join(
            f"  - [{h['slot_id']}] {h.get('title', h['talk_id'])[:80]}"
            for h in self.history
        )

    async def decide(self, slot_id, talks, hall_loads_pct, recommendation, llm_call):
        """Один шаг: спрашиваем у LLM, куда идти.

        Args:
            slot_id: id текущего слота.
            talks: list[dict(id, title, hall, abstract, category)]
            hall_loads_pct: dict[hall_id -> float in 0..>1] — НЕ передаётся агенту,
                реальный участник % загрузки не знает; capacity-логика — в политике.
            recommendation: list[talk_id] от политики (или None если no-policy)
            llm_call: async (system, user, max_tokens) -> (text, cost_usd)
        """
        options_lines = []
        for i, t in enumerate(talks):
            short_abs = (t.get("abstract", "") or "")[:200]
            options_lines.append(
                f"  {i + 1}. [{t['id']}] зал {t['hall']}, тема {t.get('category', '?')}\n"
                f"     {t['title'][:120]}\n"
                f"     {short_abs}"
            )
        rec_block = ""
        if recommendation:
            rec_block = f"\nРекомендация системы (не обязан слушать): {recommendation}"

        user_msg = USER_PROMPT_TEMPLATE.format(
            profile=self.profile[:600],
            n_visited=len(self.history),
            history=self.render_history(),
            n_options=len(talks),
            slot_id=slot_id,
            options_block="\n".join(options_lines),
            rec_block=rec_block,
        )

        text, cost = await llm_call(SYSTEM_PROMPT, user_msg, max_tokens=200)
        chosen, reason = self._parse_response(text, valid_ids=[t["id"] for t in talks])

        return LLMAgentDecision(
            agent_id=self.agent_id,
            slot_id=slot_id,
            chosen=chosen,
            reason=reason,
            cost_usd=cost,
        )

    def commit(self, slot_id, chosen_talk):
        """Запоминаем посещение в историю (для следующих слотов)."""
        if chosen_talk is None:
            return
        self.history.append({
            "slot_id": slot_id,
            "talk_id": chosen_talk["id"],
            "title": chosen_talk.get("title", ""),
            "category": chosen_talk.get("category", ""),
        })

    @staticmethod
    def _parse_response(text: str, valid_ids: list[str]):
        text = (text or "").strip()
        # вынимаем JSON
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j == -1:
            return None, "parse-error"
        try:
            d = json.loads(text[i:j + 1])
            choice = str(d.get("choice", "")).strip()
            reason = str(d.get("reason", ""))[:300]
        except Exception:
            return None, "parse-error"
        if choice == "skip" or choice == "" or choice.lower() == "none":
            return None, reason
        if choice in valid_ids:
            return choice, reason
        # иногда LLM возвращает индекс или префикс — пытаемся match'ить
        for vid in valid_ids:
            if choice in vid or vid.startswith(choice):
                return vid, reason
        return None, f"invalid-choice: {choice[:50]}"
