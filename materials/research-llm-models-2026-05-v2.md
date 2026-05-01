# Сравнение моделей OpenRouter для симулятора и LLM-ranker (01.05.2026)

Версия 2. Сбор данных через WebFetch по страницам моделей openrouter.ai по состоянию на 01.05.2026. Текущий выбор пользователя — `anthropic/claude-haiku-4.5` ($1/$5 за 1M токенов).

## 1. Сводная таблица 20 моделей из leaderboard'а недели

В колонке «structured» отмечено явное упоминание structured output / tool calling на странице модели (Y), косвенное упоминание (?) или отсутствие явного указания (—). Колонка «cache» — поддержка prompt caching на уровне OpenRouter согласно общему гайду по кэшированию (`https://openrouter.ai/docs/guides/best-practices/prompt-caching`).

| # | slug | input $/M | output $/M | контекст | structured | cache | репутация |
|---|------|-----------|------------|----------|------------|-------|-----------|
| 1 | `moonshotai/kimi-k2.6` | 0.74 | 3.49 | 262K | ? (агентская модель, явно не указано) | Y (Moonshot, авто) | Long-horizon coding, multi-agent orchestration, reasoning tokens |
| 2 | `tencent/hy3-preview:free` | 0 | 0 | 262K | — | — | MoE 295B/21B активных, agentic workflows; FREE до 08.05.2026 |
| 3 | `anthropic/claude-sonnet-4.6` | 3 | 15 | 1M | Y (tool calling, агенты) | Y (Anthropic, 1.25x/0.1x) | Frontier coding, агенты, релиз 17.02.2026 |
| 4 | `anthropic/claude-opus-4.7` | 5 | 25 | 1M | Y (агентская модель) | Y (Anthropic, 1.25x/0.1x) | Long-running async агенты, релиз 16.04.2026 |
| 5 | `deepseek/deepseek-v3.2-exp` | 0.27 | 0.41 | 163K | Y (по твиту OpenRouter — full tool calling) | Y (DeepSeek, авто, ~0.1x) | DeepSeek Sparse Attention, на уровне V3.1 по reasoning/agentic |
| 6 | `google/gemini-3-flash-preview` | 0.50 | 3 | 1M | **Y явно** (structured output + tool use + automatic context caching) | Y (Gemini, авто, 0.05x) | Near-Pro reasoning, релиз 17.12.2025 |
| 7 | `stepfun/step-3.5-flash` | 0.10 | 0.30 | 262K | — (только reasoning_details) | ? | Sparse MoE 196B/11B, reasoning, релиз 29.01.2026 |
| 8 | `minimax/minimax-m2.7` | 0.30 | 1.20 | 196K | — (явно не указано) | ? | 56.2% SWE-Pro, 1495 ELO GDPval-AA, productivity-агент |
| 9 | `x-ai/grok-4.1-fast` | 0.20 | 0.50 | 2M | Y («best agentic tool calling model» по странице) | Y (Grok, авто) | Agentic tool calling, customer support, deep research |
| 10 | `nvidia/nemotron-3-super-120b-a12b:free` | 0 | 0 | 262K | — | — | Hybrid Mamba-Transformer MoE 120B/12B, AIME 2025, SWE-bench Verified, FREE |
| 11 | `google/gemini-2.5-flash` | 0.30 | 2.50 | 1M | ? (явно не на странице, по факту поддерживает) | Y (Gemini, авто, 0.05x) | Workhorse reasoning/coding, миллиарды токенов в день |
| 12 | `anthropic/claude-opus-4.6` | 5 | 25 | 1M | Y (агентская модель) | Y (Anthropic, 1.25x/0.1x) | Strongest coding, sustained knowledge work, релиз 04.02.2026 |
| 13 | `google/gemini-2.5-flash-lite` | 0.10 | 0.40 | 1M | ? | Y (Gemini, авто, 0.05x) | Lightweight reasoning, ультранизкая латентность |
| 14 | `inclusionai/ling-2.6-1t` (free) | цена не найдена | цена не найдена | 262K | — | — | SOTA AIME26 + SWE-bench Verified, fast thinking |
| 15 | `deepseek/deepseek-v4-flash` | 0.14 | 0.28 | 1M | — (явно не указано) | Y (DeepSeek, авто) | MoE 284B/13B, hybrid attention, агентские workflow |
| 16 | `openai/gpt-5.4` | 2.50 | 15 | 1M+ | Y (OpenAI structured outputs стандартно) | Y (OpenAI, авто, 0.25-0.5x) | Frontier general-purpose + software engineering |
| 17 | `minimax/minimax-m2.5` | 0.15 | 1.15 | 196K | — | ? | 80.2% SWE-Bench Verified, 76.3% BrowseComp |
| 18 | `z-ai/glm-5.1` | 1.05 | 3.50 | 202K | — (явно не указано) | ? | Long-horizon (8+ часов автономной работы), coding |
| 19 | `openai/gpt-5.5` | 5 | 30 | 1M+ | Y (OpenAI structured outputs стандартно) | Y (OpenAI, авто) | Frontier reasoning, build over GPT-5.4 |
| 20 | `openai/gpt-oss-120b` | 0.039 | 0.18 | 131K | **Y явно** (structured output generation + native tool use) | ? | Open-weight MoE 117B/5.1B, single H100, JSON-надёжный |

Данные по ценам и контекстам — со страниц моделей `openrouter.ai/<slug>`, кроме `inclusionai/ling-2.6-1t` (на странице не указана цена) и Hy3-preview/Nemotron 3 Super (бесплатные через `:free`-маршрут).

Базовая шкала кэширования OpenRouter (`https://openrouter.ai/docs/guides/best-practices/prompt-caching`):

- **Anthropic**: write 1.25x (TTL 5 мин) или 2x (TTL 1 час), read 0.1x; требуется явный `cache_control` на сообщениях.
- **OpenAI**: write бесплатно, read 0.25-0.5x от input, авто при ≥1024 токенов.
- **Google Gemini 2.5+**: implicit caching, write = input + storage, read 0.05x.
- **DeepSeek / Moonshot / Grok / Groq**: автоматический cache с read ~0.1x.

## 2. Статус Claude Haiku 4.5 — актуальна

Поиск по openrouter.ai: страницы `anthropic/claude-haiku-4.6` и `anthropic/claude-haiku-5` возвращают «model is not available» (`https://openrouter.ai/anthropic/claude-haiku-4.6`, `https://openrouter.ai/anthropic/claude-haiku-5`). По состоянию на 30.04.2026 в семействе Anthropic у нижнего тарифа единственный актуальный вариант — `anthropic/claude-haiku-4.5` ($1/$5, 200K, релиз 15.10.2025). Преемника нет (`https://openrouter.ai/anthropic/claude-haiku-4.5`). Вверх по линейке: Sonnet 4.6 (17.02.2026, $3/$15) и Opus 4.7 (16.04.2026, $5/$25).

Это означает, что выбор Haiku 4.5 как baseline остаётся корректным; смотреть нужно за пределы Anthropic.

## 3. Top-3 для агентского симулятора (задача 1)

Профиль задачи: 1500-3000 токенов вход (профиль участника + 3-7 кандидатов + Big Five + усталость + граф знакомых), 50-200 токенов выход (JSON `{slot_id, choice}`), десятки тысяч вызовов на прогон, 3-10 прогонов (порядок 100-500 тысяч вызовов). Критично: надёжный structured output, разумный reasoning, низкая удельная цена, кэширование (общий кусок промпта — описание Big Five и инструкция формата — десятки тысяч раз повторяется неизменно).

### 1. `google/gemini-3-flash-preview` — рекомендация по умолчанию

- Цена: $0.50/$3 (`https://openrouter.ai/google/gemini-3-flash-preview`).
- Прямо на странице указано: «structured output, tool use, **automatic context caching**, configurable reasoning levels».
- Implicit caching Gemini имеет read-multiplier 0.05x (`https://openrouter.ai/docs/guides/best-practices/prompt-caching`) — самый агрессивный среди коммерческих провайдеров. Для нашего сценария с длинной системной частью промпта это режет input в ~20 раз после прогрева.
- Контекст 1M токенов закрывает любой граф знакомых.
- Качество reasoning близко к Pro-уровню по описанию страницы.

Эффективная цена при средней доле hit ~70% по кэшу: input ~$0.16/M, output $3/M. Для 100K вызовов с входом 2K и выходом 100 это ~$0.16·200 + $3·10 = $32 + $30 = $62 против Haiku 4.5 (~$200 + $50 = $250). Экономия ~4x.

### 2. `anthropic/claude-haiku-4.5` — оставить как safe fallback

- $1/$5 (`https://openrouter.ai/anthropic/claude-haiku-4.5`).
- Anthropic prompt caching с явным `cache_control` даёт read 0.1x (`https://openrouter.ai/docs/guides/best-practices/prompt-caching`). При длинной неизменной части промпта это снижает эффективную input-цену до ~$0.13-0.18/M при стабильном использовании.
- Tool calling через Anthropic SDK даёт надёжный structured output через `tool_use` (это де-факто стандарт у тебя в `generative_agent_v2.py`).
- Если столкнёшься с нестабильным JSON у Gemini 3 Flash Preview (preview-статус!) — Haiku 4.5 закрывает риск.

### 3. `openai/gpt-oss-120b` — экстремально-дешёвый вариант для масштаба

- $0.039/$0.18 (`https://openrouter.ai/openai/gpt-oss-120b`) — на порядок дешевле Haiku 4.5.
- Явное указание «structured output generation + native tool use, function calling» прямо на странице — единственная модель из топ-20, где OpenAI-стиль structured outputs прямо документирован для open-weight варианта.
- Ограничение: контекст 131K (нам хватает), знание мира до июня 2024 (не критично — наш промпт самодостаточен).
- Минус: модель открыто-весовая, маршрутизируется через сторонних провайдеров OpenRouter — возможна неконсистентность ответов между провайдерами. Перед прогоном нужен smoke-test.

При ~$0.039/M input можно позволить себе все 10 прогонов без боли по бюджету.

## 4. Top-3 для LLM-ranker (задача 2)

Профиль: 500-1500 токенов вход (3-7 кандидатов + краткий профиль), 50-100 токенов выход (упорядоченный список ID), ~50K вызовов на конфигурацию. Здесь reasoning менее критичен (короткая задача), главное — стабильный structured output и низкая удельная цена. Кэш менее эффективен, потому что для каждого участника префикс может варьироваться.

### 1. `openai/gpt-oss-120b` — оптимум по цене/качеству для ranker'а

- $0.039/$0.18 (`https://openrouter.ai/openai/gpt-oss-120b`).
- Явная поддержка structured output и function calling — для ranker'а это значит, что упорядоченный список ID можно затребовать через `response_format: json_schema` или через `tool_use`, и парсер не сломается.
- Простая задача ранжирования прекрасно справляется на 117B MoE с активными 5.1B.

50K вызовов с входом 1K и выходом 80 — $0.039·50 + $0.18·4 = $1.95 + $0.72 ≈ $2.7 на конфигурацию против Haiku 4.5 ~$1·50 + $5·4 = $70.

### 2. `deepseek/deepseek-v4-flash` — китайская альтернатива

- $0.14/$0.28 (`https://openrouter.ai/deepseek/deepseek-v4-flash`).
- DeepSeek славится стабильным tool calling и JSON-форматом (V3.2 уже поддерживает full tool calling в reasoner'е по твиту OpenRouter — `https://x.com/OpenRouterAI/status/1995511463386231012`); V4 Flash наследует это.
- Контекст 1M, hybrid attention — комфортно для batch-запросов.
- DeepSeek auto-caching read 0.1x — приятный бонус, если в ranker'е есть переиспользуемый префикс.

### 3. `google/gemini-2.5-flash-lite` — самый дешёвый «western» вариант

- $0.10/$0.40 (`https://openrouter.ai/google/gemini-2.5-flash-lite`).
- Поддерживает Gemini implicit caching (read 0.05x).
- Reasoning по умолчанию выключен — для коротких ranker-вызовов это плюс по латентности и цене.
- Минус: страница явно не подтверждает structured output, нужен smoke-test (Gemini API в целом поддерживает schema, но OpenRouter-страница не декларирует).

50K вызовов: $0.10·50 + $0.40·4 = $5 + $1.6 = $6.6 на конфигурацию.

## 5. Slug'и для подстановки в Python-код

В `generative_agent_v2.py` (агент-симулятор):

```python
MODEL_PRIMARY = "google/gemini-3-flash-preview"
MODEL_FALLBACK = "anthropic/claude-haiku-4.5"
MODEL_BUDGET = "openai/gpt-oss-120b"
```

В `llm_ranker_policy.py` (ranker):

```python
MODEL_PRIMARY = "openai/gpt-oss-120b"
MODEL_FALLBACK = "deepseek/deepseek-v4-flash"
MODEL_CHEAP_WESTERN = "google/gemini-2.5-flash-lite"
```

Текущий baseline `anthropic/claude-haiku-4.5` сохрани как контрольную точку для воспроизводимости результатов (например, в скрипте сравнения или в README запуска). Если нужно зафиксировать одну модель для финального прогона ВКР — это либо `google/gemini-3-flash-preview` (агент) и `openai/gpt-oss-120b` (ranker), либо обе — `anthropic/claude-haiku-4.5` ради единообразия со старыми экспериментами. На предзащите безопаснее показать одну модель в обеих ролях.

## 6. Замечания по риску

**Smoke-test обязателен для:**

1. **`google/gemini-3-flash-preview`** — preview-статус, релиз 17.12.2025. Preview-модели Google могут возвращать markdown-обёрнутый JSON. Smoke: 200 вызовов, доля парс-ошибок.

2. **`openai/gpt-oss-120b`** — open-weight, маршрутизируется через разных провайдеров (fireworks, together, groq). Зафиксировать `provider.order` и проверить consistency на 200 вызовах.

3. **`deepseek/deepseek-v3.2-exp`** — экспериментальный, tool calling подтверждён твитом, но не страницей. Если используется как fallback — обязательный smoke.

4. **`tencent/hy3-preview:free`** — бесплатный только до 08.05.2026, после slug пропадёт. В production не использовать.

5. **`inclusionai/ling-2.6-1t`** — цена на странице отсутствует. До выяснения через дашборд не использовать.

**Можно использовать без smoke-теста (стабильные провайдеры):**

- `anthropic/claude-haiku-4.5`, `anthropic/claude-sonnet-4.6`, `anthropic/claude-opus-4.7` — Anthropic SDK через OpenRouter работает идентично прямому Anthropic API.
- `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite` — Google production-модели.
- `openai/gpt-5.4`, `openai/gpt-5.5` — OpenAI production, structured outputs стандарт.

**Замечание про юрисдикцию:** в leaderboard'е на 5 из топ-10 позиций — китайские модели (Kimi, Hunyuan, DeepSeek, Step, MiniMax). По reasoning они уже сравнимы с western (DeepSeek V3.2 Speciale в твиттере OpenRouter сравнивается с Gemini 3 Pro). Для ВКР это не критично, но если научрук попросит обоснование выбора — можно ссылаться на то, что для воспроизводимости предпочтительнее модель с публичной документацией провайдера (Gemini, Anthropic, OpenAI), а не open-weight через сторонний провайдер.

**Кэширование как рычаг экономии:** агент-симулятор имеет длинную фиксированную часть промпта (инструкция + Big Five + формат ответа). Это идеальный кандидат для prompt caching: Anthropic — `cache_control: ephemeral`, write 1.25x, read 0.1x; Gemini — implicit, read 0.05x; DeepSeek/Moonshot/Grok — авто, read ~0.1x; OpenAI — авто, read 0.25-0.5x.

С кэшем эффективная input-цена Haiku 4.5 падает до ~$0.18/M, что почти ликвидирует разрыв с Gemini 3 Flash Preview. То есть выбор между ними — не «4x экономия», а «~30% экономия + риск preview vs стабильный Anthropic».

## Источники

Все цены, контексты, описания и release dates — со страниц `openrouter.ai/<slug>` (ссылки расставлены в тексте). Гайд по кэшированию — `https://openrouter.ai/docs/guides/best-practices/prompt-caching`. DeepSeek V3.2 tool calling — твит `https://x.com/OpenRouterAI/status/1995511463386231012`. Hy3-preview free до 08.05.2026 — `https://openrouter.ai/tencent/hy3-preview:free`. Линейка Anthropic 2026 — `https://benchlm.ai/blog/posts/claude-api-pricing`.
