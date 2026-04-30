# Предварительные выводы v4 — после исправления методологических дефектов

**Дата:** 2026-04-30, апгрейд-сессия + жёсткое техническое ревью + исправления

## Что изменилось после ревью

Внешний технический ревьюер выявил несколько методологических дефектов в v3 версии этого отчёта. **Они исправлены, выводы пересформулированы честно.** Ключевые правки:

1. **H3 social contagion** — добавлен permutation test (контроль reflection problem Манского). Raw r=0.65–0.69 объяснился общим трендом fatigue в популяции (perm_mean=0.60–0.65). После корректировки эффект слабый: adjusted r=0.01–0.09.
2. **H2 fatigue** — переформулирован как валидация консистентности модели, не «эмерджентный эффект» (fatigue прописана в промпте → tautology).
3. **H5 robustness** — разделён на (а) intra-Mobius sanity check ρ=0.99 и (б) **главный тест cross-conference Mobius vs Demo Day** ρ=0.683, p=0.042 (n=9 политик).
4. **Capacity sensitivity sweep** на Demo Day (±30%) — выявлен **новый содержательный нюанс**: при scale=0.7 (capacity недостаточна) лидер меняется на MMR.
5. Технические баги (`_hash_emb` коллизии, PPO carries hall_load, Sequential `update_history` не вызывался) исправлены.
6. Терминология сглажена: «упрощённая адаптация Park2023» вместо «OASIS-style», «inspired by SASRec» вместо «SASRec-стиль».

## Главный научный результат (после правок)

> **Политика Capacity-aware (rule-based action masking + штраф за загрузку) лидер по overflow_rate_choice среди 11 политик** (Random, Cosine, MMR, Capacity-aware, Capacity-aware MMR, DPP, Calibrated, Sequential, GNN, Constrained-PPO, LLM-ranker) **на двух структурно разных конференциях** (Mobius 40 докладов, Demo Day ITMO 210) при потере утилитарности 2–5%. Лидерство **сохраняется при варьировании ёмкости залов в диапазоне scale ∈ [0.85, 1.30]**, но **нарушается при scale=0.70** (популяция превышает суммарную capacity на >40%) — содержательная граница применимости метода.

## Числа на Mobius (11 политик, 5 сидов, 900 пользователей, learned relevance)

| Политика | OF_choice ↓ | Utility ↑ |
|---|---|---|
| Capacity-aware | **0.000** | 0.347 |
| Capacity-aware MMR | 0.022 | 0.347 |
| DPP | 0.222 | 0.351 |
| Random | 0.228 | 0.322 |
| Constrained-PPO | 0.267 | 0.352 |
| MMR | 0.278 | 0.354 |
| Calibrated | 0.283 | 0.344 |
| Cosine | 0.306 | 0.356 |
| Sequential | 0.317 | 0.331 |
| LLM-ranker | 0.344 | 0.339 |
| GNN | 0.361 | 0.332 |

## Cross-conference (Demo Day ITMO 2026, 9 политик)

| Политика | OF_choice ↓ | Utility ↑ |
|---|---|---|
| Capacity-aware | 0.456 | 0.330 |
| Capacity-aware MMR | 0.535 | 0.326 |
| MMR | 0.568 | 0.343 |
| DPP | 0.579 | 0.339 |
| Cosine | 0.583 | 0.346 |
| Calibrated | 0.594 | 0.341 |
| GNN | 0.612 | 0.323 |
| Sequential | 0.625 | 0.324 |
| Random | 0.754 | 0.308 |

## Capacity sensitivity sweep на Demo Day (НОВОЕ)

| Capacity scale | Лидер по OF_choice | OF_choice |
|---|---|---|
| 0.70 | MMR | 0.663 |
| 0.85 | Capacity-aware | 0.578 |
| 1.00 (базовый) | Capacity-aware | 0.460 |
| 1.15 | Capacity-aware | 0.353 |
| 1.30 | Capacity-aware | 0.262 |

**Содержательный вывод:** Capacity-aware лидер в широком диапазоне ёмкости (scale ≥ 0.85), но при сильно недостаточной capacity (scale 0.7, аудитория превышает суммарную вместимость на 40%+) hard_threshold=0.95 заставляет direct overflow распределяться, и MMR с тематическим разнообразием становится лучше. **Это ограничение применимости метода**, которое явно фиксируется в Главе 4.6 как направление дальнейшей работы (адаптивный hard_threshold).

## Research-гипотезы — пересмотренные результаты

| Гипотеза | Результат после ревью |
|---|---|
| H2 fatigue gradient | ✅ **valid as model consistency check** (slope=+0.05, p<0.005). Это не эмерджентный эффект — fatigue прописана в промпте, агент следует инструкции. Подтверждает работоспособность интеграции модели усталости. |
| H3 social contagion | ⚠️ **artifact of reflection problem** в большой части. Raw r=0.65–0.69 содержит общий тренд fatigue в популяции (perm_mean=0.60–0.65). Чистый social effect adjusted r=0.01–0.09 (z=0.7–5.7). Только в условии MMR-выдачи остаётся **слабый, но значимый** social signal (r=0.086, z=5.7); для Cosine r=0.031, z=2.2 (на грани значимости); для Capacity-aware и LLM-ranker — не значимо. |
| H5 robustness | 🟡 **moderate support cross-conference**. Intra-Mobius ρ=0.986 (sanity, тривиально). **Mobius vs Demo Day: ρ=0.683, p=0.042 (n=9)** — на грани значимости, корректно описывать как «moderate evidence», не «strongly supported». Топ-2 политики (Capacity-aware, Capacity-aware MMR) идентичны на двух конференциях — это и есть главный качественный аргумент. |
| H4 sequential beats cosine | 🚫 **не тестировалось корректно** — `update_history()` не вызывался из симулятора. Исправлено, требует повторного прогона. |
| H1 star-speaker | ⚠️ не тестировалось (требует speaker_fame метаданных) |

## Honest assessment объёма работы

**Что прочно (defensible на защите):**
- Параметрический симулятор + 11 политик на двух конференциях
- LLM preferences matrix + параметрическая модель выбора (HistGB Pearson r=0.79)
- Главный численный результат (Capacity-aware лидер) с capacity sensitivity sweep
- LLM-as-judge ranking
- Permutation-test для H3 (методологически грамотный)

**Что слабо (требует осторожной формулировки в тексте):**
- H2/H3 эмерджентность — заменить на «валидация консистентности модели в LLM-агентном симуляторе»
- Sequential/GNN — не SASRec и не настоящий GraphSAGE; формулировать как «упрощённые baseline'ы»
- PPO в single-agent среде — описать как baseline с честным замечанием о ограничении

**Что отсутствует (направления дальнейшей работы):**
- Multi-agent batch PPO в правильно поставленной congestion game
- Калибровка персон под реальную аудиторию JUG
- Полноценный Park2023 (importance scoring через LLM, planning module)
- Адаптивный hard_threshold для overcapacity сценариев

## Обновлённый объём работы

- ~5500 LoC Python кода (vs 1600 в начале сессии)
- 11 политик × 2 конференции × 5 сидов
- Параметрическая preference model (12K LLM-оценок)
- Capacity sensitivity sweep (5 точек)
- Permutation test для H3 (100 perm × 4 политики)
- LLM-агент с retrieval памятью + личностью + усталостью + социалкой + рефлексией
- 13+ графиков
- 16 commits в git

## Стоимость

- **$25 LLM** за всю сессию (preferences matrix $1, agent v2 $7.6, judge $8.5, прочее ~$8)
- ~3.5 часа работы
- Резерв $55 на дальнейшие итерации

## Top-3 риска на защите и план ответа

1. **«Reflection problem в H3»** — упреждающе показать permutation results: raw r=0.65, perm r=0.62, чистый эффект 0.03–0.09. Чисто социальное копирование оказалось **слабее, чем казалось**, но всё ещё значимо в одном условии (MMR), что и предсказывает теория congestion game.

2. **«PPO в неверной среде»** — открыто признать в Главе 4.6, что текущий PPO обучается в single-agent gym-среде; полноценный multi-agent congestion-game environment — направление дальнейшей работы. PPO результаты (OF=0.27) сейчас как baseline для будущего сравнения.

3. **«Capacity-aware ломается при scale=0.7»** — отвечать **превентивно**: «мы провели capacity sensitivity sweep ±30%, выявили границу применимости — при ≥40% overflow относительно вместимости hard threshold capacity-aware теряет преимущество. Это известное ограничение и направление расширения метода (адаптивный threshold)». Это **превращает слабость в strength** через осознанность ограничений.

## Главные графики

- `01_overflow.png`, `05_tradeoff.png` — 11 политик
- `21_cosine_vs_learned.png` — устойчивость к relevance signal
- `30_llm_judge_bradley_terry.png`, `31_judge_vs_overflow_tradeoff.png` — LLM-судья
- `40_sim2sim_comparison.png` — параметрический vs LLM-агентный сим
- `50_cross_conference.png` — устойчивость на двух конференциях (главный для H5)
- `60_h2_fatigue.png`, `61_h3_social.png` — гипотезы
- В планах: `70_capacity_sensitivity.png` (sweep)
