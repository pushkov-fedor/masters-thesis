"""GenerativeAgent — Park2023-стиль агент-симулятор пользователя для слоя 2 валидации.

Каждый агент:
- имеет богатую персону (текстовый профиль из LLM-генерации);
- ведёт хронологическую память выбора и впечатлений;
- получает текущий слот, кандидатов, загрузку залов и top-K рекомендаций политики;
- сам решает, идти ли (и куда) или отказаться через LLM;
- записывает решение и краткое впечатление в память для следующих слотов.

Это качественно другая модель пользователя, чем мультиномиальная логит-модель в
src/simulator.py. Здесь мы получаем согласованное во времени поведение
с явной reasoning chain.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI


SYSTEM_PROMPT = """Ты — участник IT-конференции Mobius (мобильная разработка) в Санкт-Петербурге. Конференция длится 2 дня, в каждом тайм-слоте идёт 1-3 параллельных доклада в разных залах.

Твоя личность задана текстом «Профиль». Веди себя в соответствии с интересами, опытом и стилем этого профиля.

В каждом слоте ты получаешь:
1. Список кандидатов — доклады, которые сейчас идут параллельно.
2. Текущую заполненность залов (low/medium/high/overflow).
3. Подсказку рекомендательной системы (top-K).
4. Свою прошлую память (что ты уже посетил и что чувствовал).

Решение:
- Можешь пойти на любой из доступных докладов (даже не на тот, что предложила система).
- Можешь отказаться («устал», «нужен перерыв», «не интересно»).
- Учитывай свою усталость, разнообразие тем, вместимость залов.

Возвращай строго JSON: {"decision": "<id|skip>", "reason": "одно предложение"}
Если идёшь — поле decision содержит id выбранного доклада. Если отказываешься — "skip"."""


USER_TEMPLATE = """Профиль:
{persona}

Память (последние 5 слотов):
{memory}

Тайм-слот #{slot_num} ({slot_time}). Кандидаты:
{candidates}

Подсказка системы (top-{K}, в порядке приоритета):
{recommendation}

Решение?"""


def _bucket(load_frac: float) -> str:
    if load_frac >= 1.0:
        return "overflow"
    if load_frac >= 0.8:
        return "high"
    if load_frac >= 0.5:
        return "medium"
    return "low"


@dataclass
class AgentMemory:
    """История агента: хронологический список решений."""
    entries: list = field(default_factory=list)

    def add(self, slot_id, slot_time, decision, talk_title=None, reason=""):
        self.entries.append({
            "slot_id": slot_id,
            "slot_time": slot_time,
            "decision": decision,
            "talk_title": talk_title,
            "reason": reason,
        })

    def render_recent(self, n=5) -> str:
        if not self.entries:
            return "(пока пусто — это первый слот конференции)"
        recent = self.entries[-n:]
        lines = []
        for e in recent:
            if e["decision"] == "skip":
                lines.append(f"- {e['slot_time']}: пропустил ({e['reason']})")
            else:
                lines.append(f"- {e['slot_time']}: пошёл на «{e['talk_title'][:60]}» ({e['reason']})")
        return "\n".join(lines)


@dataclass
class AgentDecision:
    decision: str  # talk_id или 'skip'
    reason: str
    cost: float = 0.0
    cached: bool = False


def parse_decision(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("{")
    j = text.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(text[i : j + 1])
    except Exception:
        return None


PRICING = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
    "anthropic/claude-sonnet-4.6": {"prompt": 3.0, "completion": 15.0},
}


class GenerativeAgent:
    """Один LLM-агент с памятью."""

    def __init__(
        self,
        agent_id: str,
        persona: str,
        client: AsyncOpenAI,
        model: str = "anthropic/claude-haiku-4.5",
    ):
        self.id = agent_id
        self.persona = persona
        self.client = client
        self.model = model
        self.memory = AgentMemory()
        self.cumulative_cost = 0.0
        self.errors = 0

    async def decide(
        self,
        slot,  # Slot object
        slot_num: int,
        candidates: list,  # list of Talk objects
        hall_loads: dict,  # {hall_id: load_frac}
        recommendation: list,  # ranked list of talk_ids
        sem: asyncio.Semaphore,
    ) -> AgentDecision:
        # Render candidates
        cand_lines = []
        for t in candidates:
            load = hall_loads.get(t.hall, 0.0)
            cand_lines.append(
                f"  id={t.id[:8]}: {t.title}\n"
                f"    зал {t.hall}, загрузка: {_bucket(load)} ({load*100:.0f}%)\n"
                f"    категория: {t.category}\n"
                f"    тема: {t.abstract[:200]}"
            )
        candidates_text = "\n".join(cand_lines)

        # Render recommendation
        rec_lines = []
        for i, tid in enumerate(recommendation):
            t_obj = next((t for t in candidates if t.id == tid), None)
            if t_obj is None:
                continue
            rec_lines.append(f"  {i+1}. id={tid[:8]} — «{t_obj.title}»")
        rec_text = "\n".join(rec_lines) if rec_lines else "(пусто)"

        user_msg = USER_TEMPLATE.format(
            persona=self.persona,
            memory=self.memory.render_recent(5),
            slot_num=slot_num,
            slot_time=slot.datetime,
            candidates=candidates_text,
            recommendation=rec_text,
            K=len(recommendation),
        )

        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
            except Exception as e:
                self.errors += 1
                # fallback: skip
                return AgentDecision(decision="skip", reason=f"api_error: {e}")

            usage = resp.usage
            pricing = PRICING.get(self.model, {"prompt": 1.0, "completion": 5.0})
            cost = (
                (usage.prompt_tokens or 0) / 1e6 * pricing["prompt"]
                + (usage.completion_tokens or 0) / 1e6 * pricing["completion"]
            )
            self.cumulative_cost += cost

            msg = resp.choices[0].message.content or ""
            parsed = parse_decision(msg)

            if not parsed:
                self.errors += 1
                return AgentDecision(decision="skip", reason="parse_fail", cost=cost)

            decision = str(parsed.get("decision", "skip"))
            reason = str(parsed.get("reason", ""))[:120]

            # Resolve short prefix to full talk id if needed
            if decision != "skip":
                # Could be 8-char prefix or full id
                full_id = None
                for t in candidates:
                    if t.id == decision or t.id.startswith(decision):
                        full_id = t.id
                        break
                if full_id is None:
                    return AgentDecision(decision="skip", reason=f"invalid_id_{decision}", cost=cost)
                decision = full_id

            # Update memory
            if decision == "skip":
                self.memory.add(slot.id, slot.datetime, "skip", reason=reason)
            else:
                t_chosen = next((t for t in candidates if t.id == decision), None)
                self.memory.add(
                    slot.id, slot.datetime, decision,
                    talk_title=t_chosen.title if t_chosen else "?",
                    reason=reason,
                )

            return AgentDecision(decision=decision, reason=reason, cost=cost)
