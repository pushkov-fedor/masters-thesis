"""Inter-slot chat — упрощённая адаптация идеи MiroFish для конференции.

После каждого слота:
- N случайных активных агентов (которые посетили доклад) пишут краткий отзыв (1-2 предложения).
- Отзывы добавляются в общий пул `posts_pool`, индексируются по talk_id и tags.

В начале следующего слота:
- Каждый агент перед `decide()` читает 2-3 поста из пула, темы которых
  близки к кандидатам в текущем слоте (cosine между эмбеддингом talk_id поста и кандидатами).
- Эти посты попадают в его память как «впечатления коллег».

Это создаёт **информационную диффузию**: впечатления о популярных докладах
распространяются. Если многим понравился доклад про Kafka — в следующих слотах
агенты с ML-интересами могут изменить решение.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI


CHAT_SYSTEM_PROMPT = """Ты только что посетил доклад на IT-конференции Mobius. Напиши **очень краткий** отзыв (1-2 предложения, до 100 символов) — что понравилось/не понравилось.

Стиль — как пост в чате конференции. Без приветствий и подписей. Без эмодзи.

Возвращай только текст отзыва, без оформления."""


@dataclass
class ChatPost:
    """Один пост в общем пуле inter-slot chat."""
    agent_id: str
    talk_id: str
    talk_title: str
    talk_emb: np.ndarray
    text: str
    timestamp: int  # slot number


class InterSlotChatPool:
    """Пул отзывов агентов между слотами + retrieval по теме."""

    def __init__(self, client: AsyncOpenAI, model: str = "anthropic/claude-haiku-4.5"):
        self.posts: List[ChatPost] = []
        self.client = client
        self.model = model
        self.cumulative_cost = 0.0
        self.errors = 0

    async def generate_post_for_agent(
        self,
        agent_id: str,
        agent_persona: str,
        talk,  # conf.talks[chosen_id]
        slot_num: int,
        sem: asyncio.Semaphore,
    ) -> Optional[ChatPost]:
        """Запрашивает у LLM отзыв этого агента о посещённом докладе."""
        user_msg = (
            f"Профиль (кратко): {agent_persona[:200]}\n\n"
            f"Только что посетил: «{talk.title}»\n"
            f"Тема: {talk.abstract[:300]}\n\n"
            f"Твой отзыв (1-2 предложения):"
        )
        async with sem:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.85,
                    max_tokens=80,
                )
            except Exception:
                self.errors += 1
                return None
            usage = resp.usage
            # Haiku 4.5 pricing
            cost = ((usage.prompt_tokens or 0) / 1e6 * 1.0
                    + (usage.completion_tokens or 0) / 1e6 * 5.0)
            self.cumulative_cost += cost
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^[\"']+|[\"']+$", "", text)[:200]
            if not text:
                return None
            post = ChatPost(
                agent_id=agent_id,
                talk_id=talk.id,
                talk_title=talk.title,
                talk_emb=talk.embedding.copy(),
                text=text,
                timestamp=slot_num,
            )
            self.posts.append(post)
            return post

    def fetch_relevant_posts(self, query_emb: np.ndarray, top_k: int = 3,
                             max_age_slots: int = 4, now: int = 0) -> List[ChatPost]:
        """Топ-K постов наиболее релевантных query, не старше max_age_slots."""
        if not self.posts:
            return []
        candidates = [p for p in self.posts if (now - p.timestamp) <= max_age_slots]
        if not candidates:
            candidates = self.posts[-20:]  # fallback: последние 20
        scored = [(float(np.dot(query_emb, p.talk_emb)), p) for p in candidates]
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]

    def render_for_prompt(self, query_emb: np.ndarray, top_k: int = 3,
                          max_age_slots: int = 4, now: int = 0) -> str:
        relevant = self.fetch_relevant_posts(query_emb, top_k, max_age_slots, now)
        if not relevant:
            return "(в чате конференции пока тихо)"
        lines = []
        for p in relevant:
            lines.append(f"💬 {p.agent_id[:10]} о «{p.talk_title[:40]}»: «{p.text[:120]}»")
        return "\n".join(lines)


async def generate_posts_for_slot(
    pool: InterSlotChatPool,
    completed_decisions: list,  # список (agent_id, persona_text, chosen_talk) для текущего слота
    conf,
    slot_num: int,
    sem: asyncio.Semaphore,
    sample_fraction: float = 0.3,
):
    """После слота — выбираем подмножество посетивших, генерируем отзывы."""
    rng = np.random.default_rng(slot_num + 1000)
    # Только те, кто пошёл (decision != skip)
    active = [d for d in completed_decisions if d[2] is not None]
    if not active:
        return
    n_to_post = max(1, int(len(active) * sample_fraction))
    indices = rng.choice(len(active), size=min(n_to_post, len(active)), replace=False)
    tasks = []
    for i in indices:
        agent_id, persona_text, talk_id = active[int(i)]
        talk = conf.talks.get(talk_id)
        if talk is None:
            continue
        tasks.append(pool.generate_post_for_agent(agent_id, persona_text, talk, slot_num, sem))
    await asyncio.gather(*tasks)
