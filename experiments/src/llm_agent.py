"""Простой LLM-агент для симуляции выбора участника на параллельных слотах конференции.

Минимальная архитектура (Уровень 2 по обзору + L2 gossip из spike_gossip_llm_amendment):
- Профиль участника как текст.
- История посещённых сегодня докладов (для тематической инерции и усталости).
- (Опционально, при w_gossip > 0) социальный сигнал по докладам: реальные числа
  count_t / N_users того же слота — agent видит, сколько участников уже выбрали
  каждый из вариантов. Управляется уровнем gossip ('off'/'moderate'/'strong'),
  определяемым на стороне run_llm_spike.py из cfg.w_gossip.
- НЕ передаётся в промпт: % загрузки залов (capacity), Big Five, граф знакомств,
  рефлексия. Capacity — ответственность политики П3, не агента (Q-G accepted в
  spike_llm_simulator).

L2 gossip — engineering choice на базе литературы (S12 OASIS log-popularity,
S13 Agent4Rec, S20 social proof, S6/S7 recsys feedback loops). Не воспроизводит
полную OASIS-архитектуру; берётся только идея controlled aggregated social
signal в text-form промпте. См. docs/spikes/spike_gossip_llm_amendment.md.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT_BASE = """Ты — реальный участник IT-конференции в зале. Тебе нужно решить, на какой из параллельных докладов сейчас идти. На реальной конференции ты учитываешь:
1. Свой профиль и интересы.
2. Что уже посещал сегодня (не повторять одни темы, к концу дня усталость).
3. Рекомендацию системы, если она есть (не обязан слушать).

Ответ строго в формате JSON: {"choice": "<talk_id или 'skip'>", "reason": "<1 короткое предложение>"}
Никакого текста до или после JSON."""


# Дополнения системного промпта для разных уровней L2 gossip.
# Соответствие cfg.w_gossip → уровень фиксируется на стороне run_llm_spike.py
# (Q-J8 accepted): off (w=0), moderate (0 < w < 0.4), strong (w >= 0.4).
SYSTEM_PROMPT_GOSSIP_MODERATE = (
    "\n4. Выбор других участников в этом же слоте — учитывай как умеренный "
    "социальный фактор (но не определяющий)."
)
SYSTEM_PROMPT_GOSSIP_STRONG = (
    "\n4. Выбор других участников в этом же слоте — учитывай как заметный "
    "социальный фактор: толпа сигнализирует, что доклад хорош."
)


def build_system_prompt(gossip_level: str = "off") -> str:
    """Собирает SYSTEM_PROMPT под уровень L2 gossip.

    gossip_level: 'off' (w_gossip = 0; Q-J9 accepted: блок не добавляется),
                  'moderate' (0 < w_gossip < 0.4), 'strong' (w_gossip >= 0.4).
    """
    if gossip_level == "off":
        return SYSTEM_PROMPT_BASE
    if gossip_level == "moderate":
        return SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_GOSSIP_MODERATE
    if gossip_level == "strong":
        return SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_GOSSIP_STRONG
    raise ValueError(
        f"unknown gossip_level={gossip_level!r}; must be off|moderate|strong"
    )


# Backward-compatible: код, импортирующий SYSTEM_PROMPT, получает baseline
# (gossip_level='off') — это эквивалент поведения этапа H V3 без gossip.
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE


USER_PROMPT_TEMPLATE = """Профиль:
{profile}

Уже посетил сегодня ({n_visited}):
{history}

Сейчас параллельно ({n_options} вариантов в слоте {slot_id}):
{options_block}
{rec_block}{gossip_block}

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

    async def decide(
        self,
        slot_id,
        talks,
        hall_loads_pct,
        recommendation,
        llm_call,
        gossip_counts=None,
        gossip_n_total=None,
        gossip_level="off",
    ):
        """Один шаг: спрашиваем у LLM, куда идти.

        Args:
            slot_id: id текущего слота.
            talks: list[dict(id, title, hall, abstract, category)]
            hall_loads_pct: dict[hall_id -> float in 0..>1] — НЕ передаётся агенту,
                реальный участник % загрузки не знает; capacity-логика — в политике
                (Q-G accepted в spike_llm_simulator). Параметр сохранён в сигнатуре
                как историческое поле.
            recommendation: list[talk_id] от политики (или None если no-policy)
            llm_call: async (system, user, max_tokens) -> (text, cost_usd)
            gossip_counts: Optional[Dict[talk_id -> int]] — счётчики выбравших
                каждый доклад в текущем слоте на момент обработки текущего
                агента. None или {} ⇒ блок отсутствует.
            gossip_n_total: Optional[int] — общее число участников в run; нужно
                для формулировки «X из N». None при gossip_level='off'.
            gossip_level: 'off' | 'moderate' | 'strong' — уровень L2 gossip
                (определяется в run_llm_spike.py из cfg.w_gossip).
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

        # L2 gossip: блок попадает в промпт только при gossip_level != 'off' И
        # переданных gossip_counts/gossip_n_total. При gossip_level='off' блок
        # ПОЛНОСТЬЮ отсутствует (Q-J9 accepted: не пишем «игнорируй gossip»,
        # просто не передаём блок).
        gossip_block = ""
        if gossip_level != "off" and gossip_counts and gossip_n_total:
            lines = []
            for t in talks:
                count = int(gossip_counts.get(t["id"], 0))
                lines.append(
                    f"  - [{t['id']}] {t['title'][:80]}: {count} из {gossip_n_total}"
                )
            gossip_block = (
                "\nЧто уже выбрали другие участники в этом слоте:\n"
                + "\n".join(lines)
            )

        user_msg = USER_PROMPT_TEMPLATE.format(
            profile=self.profile[:600],
            n_visited=len(self.history),
            history=self.render_history(),
            n_options=len(talks),
            slot_id=slot_id,
            options_block="\n".join(options_lines),
            rec_block=rec_block,
            gossip_block=gossip_block,
        )

        system_prompt = build_system_prompt(gossip_level)
        text, cost = await llm_call(system_prompt, user_msg, max_tokens=200)
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
