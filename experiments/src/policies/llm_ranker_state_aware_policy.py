"""State-aware LLM-ranker: 7-я политика, передающая в промпт текущую загрузку залов.

Отличие от обычного LLM-ranker (`llm_ranker_policy.py`): к описанию каждого
кандидата добавляется метка зала и его текущая загрузка (low / medium / high / overflow).

Это даёт LLM шанс выбирать менее загруженные альтернативы при близкой релевантности —
честное сравнение с rule-based capacity-aware политикой.

Кэш-ключ: (user_id_base, slot_id, candidates, load_bucket_signature),
чтобы одинаковые состояния по нагрузке давали одинаковый ответ.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Импортируем helpers из обычного LLM-ranker
from .llm_ranker_policy import (  # noqa: F401
    _load_api_key, _save_cache, _log_usage, _estimate_cost, _parse_array,
    PRICING, USAGE_LOG, ROOT,
)

CACHE_PATH = ROOT / "logs" / "llm_ranker_state_aware_cache.json"

SYSTEM_PROMPT = """Ты ранжируешь доклады IT-конференции под конкретного пользователя.
Тебе даётся профиль участника и список из 2-4 кандидатов в одном тайм-слоте.
Каждый кандидат сопровождён меткой загрузки зала: low / medium / high / overflow.

Цель: рекомендовать релевантные доклады с УЧЁТОМ загрузки залов.
Если зал отмечен overflow — туда направлять плохо: пользователь не попадёт.
Если зал отмечен high (>80%) — лучше выбрать менее загруженный близкий по релевантности доклад.
Если зал low (<50%) — он отличный кандидат при сопоставимой релевантности.

Верни строго JSON-массив id докладов в порядке убывания приоритета — самый релевантный И с подходящей загрузкой первым.

Без объяснений, без markdown, без комментариев. Только массив строк."""

USER_TEMPLATE = """Профиль:
{profile}

Кандидаты:
{candidates}

Верни JSON-массив всех id в порядке убывания приоритета. Учитывай и релевантность и загрузку залов.
Например: ["id_a", "id_c", "id_b"]"""


def _bucket(load_frac: float) -> str:
    if load_frac >= 1.0:
        return "overflow"
    if load_frac >= 0.8:
        return "high"
    if load_frac >= 0.5:
        return "medium"
    return "low"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


class LLMRankerStateAwarePolicy:
    name = "LLM-ranker (state-aware)"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        budget_usd: float = 3.0,
    ):
        self.model = model
        self.budget_usd = budget_usd
        self.cumulative_cost = 0.0
        self.client = OpenAI(api_key=_load_api_key(), base_url="https://openrouter.ai/api/v1")
        self.cache = _load_cache()
        self._dirty = False
        self._calls_since_save = 0
        self._save_every = 25

    def __del__(self):
        if getattr(self, "_dirty", False):
            try:
                with open(CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False)
            except Exception:
                pass

    def _flush(self):
        if self._dirty:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
            self._dirty = False
            self._calls_since_save = 0

    @staticmethod
    def _strip_replica(user_id: str) -> str:
        return re.sub(r"_r\d+$", "", user_id)

    def _cache_key(self, user_id, slot_id, candidate_ids, load_buckets):
        base_id = self._strip_replica(user_id)
        # сортируем кандидатов вместе с их бакетами для воспроизводимости
        pairs = sorted(zip(candidate_ids, load_buckets))
        sig = "|".join(f"{tid}:{b}" for tid, b in pairs)
        h = hashlib.sha1((base_id + "|" + slot_id + "|" + sig).encode("utf-8")).hexdigest()[:24]
        return h

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        hall_load = state["hall_load"]
        load_fracs = []
        load_buckets = []
        for tid in cand_ids:
            t = conf.talks[tid]
            cap = conf.halls[t.hall].capacity
            occ = hall_load.get((slot.id, t.hall), 0)
            lf = occ / max(1.0, cap)
            load_fracs.append(lf)
            load_buckets.append(_bucket(lf))

        key = self._cache_key(user.id, slot.id, cand_ids, load_buckets)
        cached = self.cache.get(key)
        if cached:
            return [tid for tid in cached if tid in cand_ids][:K]

        if self.cumulative_cost >= self.budget_usd:
            return self._fallback_capacity_aware(user, conf, cand_ids, load_fracs, K)

        candidates_text = "\n".join(
            f"- id={tid}\n  title: {conf.talks[tid].title}\n  category: {conf.talks[tid].category}\n  hall_load: {load_buckets[i]}\n  abstract: {conf.talks[tid].abstract[:300]}"
            for i, tid in enumerate(cand_ids)
        )
        user_msg = USER_TEMPLATE.format(profile=user.text, candidates=candidates_text)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=120,
            )
        except Exception as e:
            _log_usage({"ts": time.time(), "kind": "llm_ranker_sa_error", "error": str(e)})
            return self._fallback_capacity_aware(user, conf, cand_ids, load_fracs, K)

        usage = resp.usage
        cost = _estimate_cost(
            self.model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        self.cumulative_cost += cost

        msg = resp.choices[0].message.content or ""
        arr = _parse_array(msg)
        if not arr:
            return self._fallback_capacity_aware(user, conf, cand_ids, load_fracs, K)

        valid = [tid for tid in arr if tid in cand_ids]
        for tid in cand_ids:
            if tid not in valid:
                valid.append(tid)
        result = valid[:K]

        self.cache[key] = valid
        self._dirty = True
        self._calls_since_save += 1
        if self._calls_since_save >= self._save_every:
            self._flush()

        _log_usage({
            "ts": time.time(), "kind": "llm_ranker_state_aware",
            "user": user.id, "slot": slot.id,
            "load_buckets": load_buckets,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "cost": cost, "cumulative": self.cumulative_cost,
        })
        return result

    def _fallback_capacity_aware(self, user, conf, cand_ids, load_fracs, K):
        import numpy as np
        scored = []
        for tid, lf in zip(cand_ids, load_fracs):
            sim = float(np.dot(user.embedding, conf.talks[tid].embedding))
            score = sim - 0.5 * lf
            scored.append((score, tid))
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:K]]

    def stats(self) -> dict:
        return {
            "cumulative_cost_usd": self.cumulative_cost,
            "cache_size": len(self.cache),
        }
