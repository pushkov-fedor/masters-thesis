"""LLM-ranker как 6-я политика.

Дизайн:
- модель: openai/gpt-4o-mini (дёшево и достаточно для ранжирования);
- инпут: профиль пользователя + список из ~3 кандидатов с описаниями;
- аутпут: упорядоченный JSON-список id (top-K);
- кэш на диске: по хэшу (user_id, slot_id), без state — чистый семантический ранкер.

Семантика без state info — это сознательное упрощение для контроля стоимости.
LLM-ranker сравнивается с Cosine как «то же отображение профиль->релевантность,
но через семантическое понимание модели вместо ближайших соседей в эмбеддинге».
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
ENV_CANDIDATES = [
    ROOT.parent / ".env",
    ROOT.parent.parent / "party-of-one" / ".env",
]
CACHE_PATH = ROOT / "logs" / "llm_ranker_cache.json"
USAGE_LOG = ROOT / "logs" / "openrouter_usage.jsonl"

SYSTEM_PROMPT = """Ты ранжируешь доклады IT-конференции под конкретного пользователя.
Тебе даётся профиль участника и список из 2-4 кандидатов (доклады в одном тайм-слоте).
Верни строго JSON-массив id докладов в порядке убывания приоритета — самый релевантный первым.

Без объяснений, без markdown, без комментариев. Только массив строк."""

USER_TEMPLATE = """Профиль:
{profile}

Кандидаты:
{candidates}

Верни JSON-массив всех id в порядке убывания релевантности для этого участника. Например: ["id_a", "id_c", "id_b"]"""


def _load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            key = cfg.get("OPENROUTER_API_KEY")
            if key:
                return key
    raise RuntimeError("OPENROUTER_API_KEY not found")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _log_usage(record: dict):
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Pricing per 1M tokens, OpenRouter
PRICING = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
}


def _estimate_cost(model: str, p: int, c: int) -> float:
    pr = PRICING.get(model, {"prompt": 1.0, "completion": 5.0})
    return p / 1e6 * pr["prompt"] + c / 1e6 * pr["completion"]


def _parse_array(text: str) -> Optional[list]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M)
    i = text.find("[")
    j = text.rfind("]")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(text[i : j + 1])
    except Exception:
        return None


class LLMRankerPolicy:
    name = "LLM-ranker"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        budget_usd: float = 3.0,
        cache_path: Path = CACHE_PATH,
    ):
        self.model = model
        self.budget_usd = budget_usd
        self.cumulative_cost = 0.0
        self.client = OpenAI(api_key=_load_api_key(), base_url="https://openrouter.ai/api/v1")
        self.cache_path = cache_path
        self.cache = _load_cache()
        self._dirty = False
        self._save_every = 25  # сохранять кэш каждые N новых вызовов
        self._calls_since_save = 0
        self.n_api_calls = 0
        self.n_cache_hits = 0
        self._heartbeat_every = 200
        self._t_start = time.time()
        self._pbar = None

    def set_progress_total(self, total: int, desc: str = "LLM-ranker"):
        """Создаёт tqdm progress bar; вызывается из run_experiments.py."""
        from tqdm import tqdm
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = tqdm(total=total, desc=desc, unit="call", dynamic_ncols=True,
                          mininterval=1.0, file=__import__("sys").stdout)

    def close_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def __del__(self):
        if getattr(self, "_dirty", False):
            try:
                _save_cache(self.cache)
            except Exception:
                pass

    def _flush(self):
        if self._dirty:
            _save_cache(self.cache)
            self._dirty = False
            self._calls_since_save = 0

    @staticmethod
    def _strip_replica(user_id: str) -> str:
        """Превращает 'u_001_r2' в 'u_001'. Реплики разных копий разделяют LLM-ранжирование."""
        return re.sub(r"_r\d+$", "", user_id)

    def _cache_key(self, user_id: str, slot_id: str, candidate_ids: list) -> str:
        base_id = self._strip_replica(user_id)
        h = hashlib.sha1((base_id + "|" + slot_id + "|" + "|".join(sorted(candidate_ids))).encode("utf-8")).hexdigest()[:20]
        return h

    def __call__(self, *, user, slot, conf, state):
        K = state["K"]
        cand_ids = list(slot.talk_ids)
        if len(cand_ids) <= K:
            return cand_ids

        key = self._cache_key(user.id, slot.id, cand_ids)
        cached = self.cache.get(key)
        if cached:
            self.n_cache_hits += 1
            return [tid for tid in cached if tid in cand_ids][:K]

        if self.cumulative_cost >= self.budget_usd:
            # бюджет исчерпан — fallback на Cosine
            return self._fallback_cosine(user, conf, cand_ids, K)

        candidates_text = "\n".join(
            f"- id={tid}\n  title: {conf.talks[tid].title}\n  category: {conf.talks[tid].category}\n  abstract: {conf.talks[tid].abstract[:400]}"
            for tid in cand_ids
        )
        user_msg = USER_TEMPLATE.format(profile=user.text, candidates=candidates_text)
        try:
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=120,
                timeout=60,
            )
            self.n_api_calls += 1
            last_latency = time.time() - t0
            if self._pbar is not None:
                self._pbar.update(1)
                self._pbar.set_postfix({
                    "cost": f"${self.cumulative_cost:.2f}",
                    "last": f"{last_latency:.1f}s",
                    "hits": self.n_cache_hits,
                })
            elif self.n_api_calls % self._heartbeat_every == 0:
                elapsed = time.time() - self._t_start
                print(f"  [LLM-ranker] api={self.n_api_calls} cache_hits={self.n_cache_hits} "
                      f"cost=${self.cumulative_cost:.3f} last={last_latency:.1f}s elapsed={elapsed:.0f}s",
                      flush=True)
        except Exception as e:
            _log_usage({"ts": time.time(), "kind": "llm_ranker_error", "error": str(e)})
            return self._fallback_cosine(user, conf, cand_ids, K)

        usage = resp.usage
        cost = _estimate_cost(
            self.model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        self.cumulative_cost += cost
        if not resp.choices or resp.choices[0].message is None:
            return self._fallback_cosine(user, conf, cand_ids, K)
        msg = resp.choices[0].message.content or ""
        arr = _parse_array(msg)
        if not arr:
            _log_usage({"ts": time.time(), "kind": "llm_ranker_parse_fail",
                        "raw": msg[:200], "cost": cost})
            return self._fallback_cosine(user, conf, cand_ids, K)

        # фильтрация и удержание только валидных id из кандидатов
        valid = [tid for tid in arr if tid in cand_ids]
        # дополним недостающими, если LLM забыла
        for tid in cand_ids:
            if tid not in valid:
                valid.append(tid)
        result = valid[:K]

        self.cache[key] = valid  # храним полный порядок для возможного использования с другим K
        self._dirty = True
        self._calls_since_save += 1
        if self._calls_since_save >= self._save_every:
            self._flush()

        _log_usage({
            "ts": time.time(), "kind": "llm_ranker",
            "user": user.id, "slot": slot.id,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "cost": cost, "cumulative": self.cumulative_cost,
        })
        return result

    def _fallback_cosine(self, user, conf, cand_ids, K):
        import numpy as np
        scored = [(float(np.dot(user.embedding, conf.talks[tid].embedding)), tid) for tid in cand_ids]
        scored.sort(reverse=True)
        return [tid for _, tid in scored[:K]]

    def stats(self) -> dict:
        return {
            "cumulative_cost_usd": self.cumulative_cost,
            "cache_size": len(self.cache),
        }
