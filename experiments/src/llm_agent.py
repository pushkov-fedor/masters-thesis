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

Bilingual templates ru/en — для паритета каналов с параметрическим симулятором
на EN-пайплайне (BGE-large-en + ABTT-1). Язык выбирается полем
``LLMAgent.language`` и параметром ``build_system_prompt(..., language=)``.
По умолчанию ``"ru"`` (обратная совместимость с RU-прогоном 08.05.2026).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------- Bilingual templates ----------

_TEMPLATES: dict[str, dict[str, str]] = {
    "ru": {
        "system_base": (
            "Ты — реальный участник IT-конференции в зале. Тебе нужно решить, "
            "на какой из параллельных докладов сейчас идти. На реальной "
            "конференции ты учитываешь:\n"
            "1. Свой профиль и интересы.\n"
            "2. Что уже посещал сегодня (не повторять одни темы, к концу дня "
            "усталость).\n"
            "3. Рекомендацию системы, если она есть (не обязан слушать).\n\n"
            "Ответ строго в формате JSON: "
            "{\"choice\": \"<talk_id или 'skip'>\", \"reason\": \"<1 короткое "
            "предложение>\"}\n"
            "Никакого текста до или после JSON."
        ),
        "system_gossip_moderate": (
            "\n4. Выбор других участников в этом же слоте — учитывай как "
            "умеренный социальный фактор (но не определяющий)."
        ),
        "system_gossip_strong": (
            "\n4. Выбор других участников в этом же слоте — учитывай как "
            "заметный социальный фактор: толпа сигнализирует, что доклад хорош."
        ),
        "user_template": (
            "Профиль:\n{profile}\n\n"
            "Уже посетил сегодня ({n_visited}):\n{history}\n\n"
            "Сейчас параллельно ({n_options} вариантов в слоте {slot_id}):\n"
            "{options_block}{rec_block}{gossip_block}\n\n"
            "Что выбираешь? Только JSON."
        ),
        "history_empty": "  (пока ничего)",
        "history_line": "  - [{slot_id}] {title}",
        "option_line": (
            "  {i}. [{tid}] зал {hall}, тема {cat}\n"
            "     {title}\n"
            "     {abstract}"
        ),
        "rec_block": "\nРекомендация системы (не обязан слушать): {rec}",
        "gossip_header": "\nЧто уже выбрали другие участники в этом слоте:",
        "gossip_line": "  - [{tid}] {title}: {count} из {total}",
    },
    "en": {
        "system_base": (
            "You are a real attendee at an IT conference. You need to decide "
            "which of the parallel talks to attend right now. At a real "
            "conference you consider:\n"
            "1. Your profile and interests.\n"
            "2. What you have already attended today (avoid repeating the same "
            "topics; fatigue builds up by the end of the day).\n"
            "3. The system's recommendation, if any (you are not required to "
            "follow it).\n\n"
            "Answer strictly as JSON: "
            "{\"choice\": \"<talk_id or 'skip'>\", \"reason\": \"<one short "
            "sentence>\"}\n"
            "No text before or after the JSON."
        ),
        "system_gossip_moderate": (
            "\n4. The choices of other attendees in the same slot — treat as "
            "a moderate social signal (but not decisive)."
        ),
        "system_gossip_strong": (
            "\n4. The choices of other attendees in the same slot — treat as "
            "a strong social signal: a crowd suggests the talk is good."
        ),
        "user_template": (
            "Profile:\n{profile}\n\n"
            "Already attended today ({n_visited}):\n{history}\n\n"
            "Currently parallel ({n_options} options in slot {slot_id}):\n"
            "{options_block}{rec_block}{gossip_block}\n\n"
            "Which one do you choose? JSON only."
        ),
        "history_empty": "  (none yet)",
        "history_line": "  - [{slot_id}] {title}",
        "option_line": (
            "  {i}. [{tid}] hall {hall}, topic {cat}\n"
            "     {title}\n"
            "     {abstract}"
        ),
        "rec_block": "\nSystem recommendation (not required to follow): {rec}",
        "gossip_header": "\nWhat other attendees in this slot have already chosen:",
        "gossip_line": "  - [{tid}] {title}: {count} of {total}",
    },
}


def _get_templates(language: str) -> dict[str, str]:
    if language not in _TEMPLATES:
        raise ValueError(
            f"unknown language={language!r}; must be one of {list(_TEMPLATES)}"
        )
    return _TEMPLATES[language]


def build_system_prompt(gossip_level: str = "off", language: str = "ru") -> str:
    """Собирает SYSTEM_PROMPT под уровень L2 gossip и язык.

    gossip_level: 'off' (w_gossip = 0; Q-J9 accepted: блок не добавляется),
                  'moderate' (0 < w_gossip < 0.4), 'strong' (w_gossip >= 0.4).
    language: 'ru' (default) | 'en' — паритет с языком текстов программы.
    """
    t = _get_templates(language)
    if gossip_level == "off":
        return t["system_base"]
    if gossip_level == "moderate":
        return t["system_base"] + t["system_gossip_moderate"]
    if gossip_level == "strong":
        return t["system_base"] + t["system_gossip_strong"]
    raise ValueError(
        f"unknown gossip_level={gossip_level!r}; must be off|moderate|strong"
    )


# Backward-compatible: код, импортирующий SYSTEM_PROMPT, получает RU baseline
# (gossip_level='off') — это эквивалент поведения этапа H V3 без gossip.
SYSTEM_PROMPT = _TEMPLATES["ru"]["system_base"]
SYSTEM_PROMPT_BASE = _TEMPLATES["ru"]["system_base"]
SYSTEM_PROMPT_GOSSIP_MODERATE = _TEMPLATES["ru"]["system_gossip_moderate"]
SYSTEM_PROMPT_GOSSIP_STRONG = _TEMPLATES["ru"]["system_gossip_strong"]
USER_PROMPT_TEMPLATE = _TEMPLATES["ru"]["user_template"]


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
    language: str = "ru"

    def render_history(self) -> str:
        t = _get_templates(self.language)
        if not self.history:
            return t["history_empty"]
        return "\n".join(
            t["history_line"].format(
                slot_id=h["slot_id"],
                title=h.get("title", h["talk_id"])[:80],
            )
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
        t = _get_templates(self.language)
        options_lines = []
        for i, talk in enumerate(talks):
            short_abs = (talk.get("abstract", "") or "")[:200]
            options_lines.append(t["option_line"].format(
                i=i + 1,
                tid=talk["id"],
                hall=talk["hall"],
                cat=talk.get("category", "?"),
                title=talk["title"][:120],
                abstract=short_abs,
            ))
        rec_block = ""
        if recommendation:
            rec_block = t["rec_block"].format(rec=recommendation)

        # L2 gossip: блок попадает в промпт только при gossip_level != 'off' И
        # переданных gossip_counts/gossip_n_total. При gossip_level='off' блок
        # ПОЛНОСТЬЮ отсутствует (Q-J9 accepted: не пишем «игнорируй gossip»,
        # просто не передаём блок).
        gossip_block = ""
        if gossip_level != "off" and gossip_counts and gossip_n_total:
            lines = [
                t["gossip_line"].format(
                    tid=talk["id"],
                    title=talk["title"][:80],
                    count=int(gossip_counts.get(talk["id"], 0)),
                    total=gossip_n_total,
                )
                for talk in talks
            ]
            gossip_block = t["gossip_header"] + "\n" + "\n".join(lines)

        user_msg = t["user_template"].format(
            profile=self.profile[:600],
            n_visited=len(self.history),
            history=self.render_history(),
            n_options=len(talks),
            slot_id=slot_id,
            options_block="\n".join(options_lines),
            rec_block=rec_block,
            gossip_block=gossip_block,
        )

        system_prompt = build_system_prompt(gossip_level, language=self.language)
        text, cost = await llm_call(system_prompt, user_msg, max_tokens=200)
        chosen, reason = self._parse_response(text, valid_ids=[talk["id"] for talk in talks])

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
