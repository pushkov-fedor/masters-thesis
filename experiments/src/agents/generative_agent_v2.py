"""GenerativeAgent v2 — OASIS-style агент с эмерджентным поведением.

Расширения относительно v1 (generative_agent.py):
- Memory с retrieval (Park2023): relevance × recency × importance
- Big Five personality (PUB-style)
- Fatigue model (накапливается по слотам)
- Social graph (агент видит, куда идут друзья)
- Reflection (LLM агрегирует наблюдения в insights)

Это ставит наш слой 2 валидации на уровень serious agent-based recsys-симулятора.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from .memory_with_retrieval import MemoryWithRetrieval
from .personality import BigFive
from .fatigue import FatigueState


SYSTEM_PROMPT = """Ты — участник IT-конференции в Санкт-Петербурге. Веди себя как настоящий человек: учитывай свою личность, усталость, что делают коллеги, и свои воспоминания о прошедших слотах.

В каждом слоте получаешь:
1. Свой профиль и черты личности.
2. Свой текущий уровень усталости.
3. Релевантные воспоминания (выбраны по семантической близости к этому слоту).
4. Insights рефлексии — твои выводы по ходу конференции.
5. Социальный сигнал — кто из коллег куда уже пошёл в этом слоте.
6. Кандидатов в слоте + их загрузка залов.
7. Рекомендацию системы.

Твоё решение должно учитывать ВСЕ эти факторы:
- Если ты экстраверт и зал заполнен — это плюс (атмосфера); если интроверт — минус.
- Если ты любишь новое (high openness) — нишевые темы тебе интереснее.
- Если ты устал — выше шанс пропустить.
- Если коллеги идут на доклад — у тебя есть слабая склонность присоединиться.
- Если есть воспоминания об усталости от похожих тем — учти их.

Верни строго JSON: {"decision": "<id|skip>", "reason": "одно предложение"}"""


USER_TEMPLATE = """Профиль:
{persona}

Личность (Big Five):
{personality}

Усталость: {fatigue} (уровень {fatigue_level:.2f})

Релевантные воспоминания:
{memory_render}

Высокоуровневые выводы по конференции:
{reflection_insights}

Тайм-слот #{slot_num} ({slot_time}). Кандидаты:
{candidates}

Социальный сигнал: {social_signal}

Чат конференции (мнения коллег после прошлых слотов):
{chat_signal}

Подсказка системы (top-{K}):
{recommendation}

Решение?"""


REFLECTION_SYSTEM = """Ты обобщаешь опыт участника конференции в 1-2 высокоуровневых вывода.
Из набора недавних воспоминаний вычлени паттерны: что нравится, что утомляет, какие темы интереснее.
Возвращай строго JSON: {"insights": ["короткая строка вывода 1", "вывод 2"]}"""


PRICING = {
    # Цены на 01.05.2026 по openrouter.ai/<slug>
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
    "anthropic/claude-sonnet-4.6": {"prompt": 3.0, "completion": 15.0},
    "anthropic/claude-opus-4.7": {"prompt": 5.0, "completion": 25.0},
    "google/gemini-3-flash-preview": {"prompt": 0.50, "completion": 3.0},
    "google/gemini-2.5-flash": {"prompt": 0.30, "completion": 2.50},
    "google/gemini-2.5-flash-lite": {"prompt": 0.10, "completion": 0.40},
    "deepseek/deepseek-v3.2-exp": {"prompt": 0.27, "completion": 0.41},
    "deepseek/deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28},
    "openai/gpt-oss-120b": {"prompt": 0.039, "completion": 0.18},
    "openai/gpt-5.4": {"prompt": 2.50, "completion": 15.0},
    "x-ai/grok-4.1-fast": {"prompt": 0.20, "completion": 0.50},
    "moonshotai/kimi-k2.6": {"prompt": 0.74, "completion": 3.49},
    "minimax/minimax-m2.7": {"prompt": 0.30, "completion": 1.20},
}


def parse_json_object(text: str) -> Optional[dict]:
    """Извлекает JSON из ответа модели.

    Поддерживает форматы:
    - Чистый JSON.
    - JSON в markdown-блоке ```json ... ```.
    - JSON после <think>...</think> reasoning-блока (DeepSeek и др.).
    - JSON с trailing-комментариями или текстом до/после.

    Стратегия: сначала снимаем reasoning-обёртки, затем сканируем все возможные
    JSON-объекты в тексте и возвращаем последний валидный, в котором есть
    ключ "decision".
    """
    text = (text or "").strip()
    if not text:
        return None

    # Снимаем <think>...</think> блок (DeepSeek и многие reasoning-модели)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Снимаем reasoning-обёртки других форматов
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)

    # Снимаем markdown-обёртки
    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Прямая попытка распарсить как JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "decision" in obj:
            return obj
    except Exception:
        pass

    # Сканируем все возможные {...} блоки и возвращаем последний валидный
    # с ключом "decision"
    candidates: List[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict) and "decision" in obj:
                        candidates.append(obj)
                except Exception:
                    pass
                start = -1

    if candidates:
        return candidates[-1]

    return None


@dataclass
class AgentDecision:
    decision: str  # talk_id или 'skip'
    reason: str
    cost: float = 0.0


class GenerativeAgentV2:
    """OASIS-style агент с памятью, личностью, усталостью, социальной информацией."""

    def __init__(
        self,
        agent_id: str,
        agent_idx: int,            # порядковый индекс для социального графа
        persona_text: str,
        personality: BigFive,
        client: AsyncOpenAI,
        model: str = "anthropic/claude-haiku-4.5",
        reflection_threshold: float = 2.5,  # сумма importance, после которой триггерим reflection
    ):
        self.id = agent_id
        self.idx = agent_idx
        self.persona = persona_text
        self.personality = personality
        self.client = client
        self.model = model
        self.memory = MemoryWithRetrieval()
        self.fatigue = FatigueState()
        self.reflection_insights: List[str] = []
        self.last_reflection_slot = 0
        self.reflection_threshold = reflection_threshold
        self.cumulative_cost = 0.0
        self.errors = 0

    async def decide(
        self,
        slot,
        slot_num: int,
        slot_query_emb: np.ndarray,  # эмбеддинг "контекста слота" для retrieval
        candidates: list,
        hall_loads: dict,            # {hall_id: load_frac}
        recommendation: list,
        social_signal: str,
        sem: asyncio.Semaphore,
        chat_signal: str = "(чат конференции пока не используется)",
    ) -> AgentDecision:
        # candidates rendering
        cand_lines = []
        from .social_graph import SocialGraph  # for type hint only (used elsewhere)
        for t in candidates:
            load = hall_loads.get(t.hall, 0.0)
            load_label = (
                "очень загружен" if load > 0.85 else
                "загружен" if load > 0.6 else
                "комфортно" if load > 0.3 else
                "почти пуст"
            )
            cand_lines.append(
                f"  id={t.id[:8]}: {t.title}\n"
                f"    зал {t.hall} ({load_label}, {load*100:.0f}%) | категория: {t.category}\n"
                f"    тема: {t.abstract[:200]}"
            )
        candidates_text = "\n".join(cand_lines)

        rec_lines = []
        for i, tid in enumerate(recommendation):
            t_obj = next((t for t in candidates if t.id == tid), None)
            if t_obj is None:
                continue
            rec_lines.append(f"  {i+1}. id={tid[:8]} — «{t_obj.title}»")
        rec_text = "\n".join(rec_lines) if rec_lines else "(пусто)"

        # Retrieval из памяти
        memory_render = self.memory.render_for_prompt(query_emb=slot_query_emb,
                                                      now=slot_num, top_k=5)
        # Reflection insights
        reflection_render = ("• " + "\n• ".join(self.reflection_insights)) if self.reflection_insights \
                            else "(пока нет — ещё мало опыта)"

        user_msg = USER_TEMPLATE.format(
            persona=self.persona[:600],
            personality=self.personality.render(),
            fatigue=self.fatigue.render(),
            fatigue_level=self.fatigue.level,
            memory_render=memory_render,
            reflection_insights=reflection_render,
            slot_num=slot_num,
            slot_time=slot.datetime,
            candidates=candidates_text,
            social_signal=social_signal,
            chat_signal=chat_signal,
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
                return AgentDecision(decision="skip", reason=f"api_error")

            usage = resp.usage
            pricing = PRICING.get(self.model, {"prompt": 1.0, "completion": 5.0})
            cost = (
                (usage.prompt_tokens or 0) / 1e6 * pricing["prompt"]
                + (usage.completion_tokens or 0) / 1e6 * pricing["completion"]
            )
            self.cumulative_cost += cost

            msg = resp.choices[0].message.content or ""
            parsed = parse_json_object(msg)
            if not parsed:
                self.errors += 1
                return AgentDecision(decision="skip", reason="parse_fail", cost=cost)

            decision = str(parsed.get("decision", "skip"))
            reason = str(parsed.get("reason", ""))[:120]

            cand_ids = [c.id for c in candidates]
            if decision != "skip":
                full_id = next((c.id for c in candidates
                                if c.id == decision or c.id.startswith(decision)), None)
                if full_id is None:
                    decision = "skip"
                    reason = "invalid_id"
                else:
                    decision = full_id

            # Записываем в память
            current_date = slot.datetime[:10] if slot.datetime else ""
            if decision == "skip":
                obs = f"Пропустил слот {slot.datetime} ({reason})"
                obs_emb = slot_query_emb  # используем тот же контекст
                importance = 0.3 + 0.2 * self.fatigue.level
            else:
                t_chosen = next((t for t in candidates if t.id == decision), None)
                obs = f"Посетил «{t_chosen.title[:60]}» ({reason})"
                obs_emb = t_chosen.embedding if t_chosen is not None else slot_query_emb
                importance = 0.5
                # Если тема знакомая — низкая importance, новая — высокая
                # (proxy через cosine с предыдущими)
            self.memory.add(content=obs, content_emb=obs_emb,
                            timestamp=slot_num, importance=importance,
                            kind="decision")
            self.fatigue.update_after_decision(decision, current_date)

        # Триггер рефлексии — ВНЕ семафора (чтобы избежать deadlock)
        if (self.memory.aggregate_importance(since_timestamp=self.last_reflection_slot)
                >= self.reflection_threshold and len(self.memory) >= 3):
            await self._reflect(slot_num, sem)
            self.last_reflection_slot = slot_num

        return AgentDecision(decision=decision, reason=reason, cost=cost)

    async def _reflect(self, now: int, sem: asyncio.Semaphore):
        """Triggered reflection: обобщение recent observations в high-level insight."""
        recent = self.memory.fetch_recent(8)
        observations_text = "\n".join(f"- {e.content}" for e in recent)
        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REFLECTION_SYSTEM},
                        {"role": "user", "content": f"Профиль: {self.persona[:300]}\n\nНаблюдения:\n{observations_text}\n\nКакие 1-2 высокоуровневых вывода?"},
                    ],
                    temperature=0.5,
                    max_tokens=200,
                )
            except Exception:
                return
            usage = resp.usage
            pricing = PRICING.get(self.model, {"prompt": 1.0, "completion": 5.0})
            cost = (
                (usage.prompt_tokens or 0) / 1e6 * pricing["prompt"]
                + (usage.completion_tokens or 0) / 1e6 * pricing["completion"]
            )
            self.cumulative_cost += cost

            parsed = parse_json_object(resp.choices[0].message.content or "")
            if parsed and isinstance(parsed.get("insights"), list):
                new_insights = [str(s)[:120] for s in parsed["insights"][:3]]
                self.reflection_insights.extend(new_insights)
                # Сохраняем insight'ы как memory с высокой importance
                for ins in new_insights:
                    # Эмбеддинг insight'а — пока просто среднее эмбеддингов recent
                    avg_emb = np.mean([e.content_emb for e in recent], axis=0)
                    avg_emb = avg_emb / max(1e-9, np.linalg.norm(avg_emb))
                    self.memory.add(content=ins, content_emb=avg_emb,
                                    timestamp=now, importance=0.85, kind="reflection")
