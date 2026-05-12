---
name: EN-pipeline rerun 2026-05-12
description: Перегон основного эксперимента на BGE-large-en + ABTT-1 (после аудита функции релевантности); пул 100 EN-персон; Q + V + cross-validation; Q-O7 пройден существенно сильнее
type: project
originSessionId: a26bf5f0-d8a0-458d-9d40-4e2278f55cf6
---
**Дата перегона**: 2026-05-12 (перед предзащитой 13.05).

**Почему перегнали.** Аудит функции релевантности (`docs/spikes/spike_relevance_function_audit.md`) показал, что `intfloat/multilingual-e5-small` даёт узкий конус на русскоязычных коротких текстах (Vendi 4.1% от max на personas_100, парные cos в [0.85, 0.96]). Победитель аудита — `BAAI/bge-large-en-v1.5 + ABTT-1` (gap +0.316 vs e5 +0.068; Spearman с LLM-judge 0.327 vs 0.240; Vendi пула 89% на 50 персон / 63% на 100).

**Why-detailed:** старая позиция «cosine ≈ capacity_aware по utility» была тривиальной — utility почти константна на узком конусе. Чтобы аргумент был содержательным, нужна функция релевантности, способная различать релевантный и нерелевантный доклад. BGE-large-en + ABTT-1 это даёт.

**Как применить (для будущих сессий).** Если работа касается основного эксперимента: верить `report_mobius_2025_autumn_en_full.md`, а не старому `report_mobius_2025_autumn_full.md` (RU snapshot из PDF). Все analysis_*.json — в `experiments/results/en/`. Bilingual LLM templates: `experiments/src/llm_agent.py`, флаг `--language en` в `experiments/scripts/run_llm_lhs_subset.py`.

**Главные числовые сдвиги (EN vs RU snapshot):**

- pairwise `cosine vs capacity_aware`: strict wins 20% (RU 26%); за ε 14% (RU 22%) — направление сохранилось, центральный тезис «cosine не выигрывает у capacity_aware ни на одной точке (0/50)» подтверждён
- risk-positive 11/50 (RU 13/50); на нём cap_aware строго снижает 8/11 = 73% (RU 11/13 = 85%)
- trade-off risk × utility 3/150 = 2% (RU 11/150 = 7.3%) — стало реже, лучше для защиты
- volatile points 72 (RU 24) — больше нестабильности (расширенный конус → меньше концентрация softmax)
- **Q-O7 overall median ρ = 0.769 (RU 0.554)** — pass пройден существенно сильнее
- `hall_utilization_variance` ρ 0.40 FAIL → **0.80 PASS на 12/12 LHS** — главное усиление методики

**Что НЕ менялось.** Текст ВКР (`thesis/*.tex`, PDF после антиплагиата 08.05) — не правится до защиты; корректировка в финальной версии после 13.05, чек-лист в корне `THESIS_CORRECTIONS_AFTER_DEFENSE.md`. Архитектурные решения (`PROJECT_OVERVIEW §14`): capacity вне utility, двусимуляторная архитектура с паритетом каналов, LHS + CRN, симплекс весов, gossip per-talk счётчик, Φ как ось не оптимизатор, EC как блокирующий фильтр — всё сохранено.

**Артефакты перегона** (всё в репо):

- Финальный отчёт: `experiments/results/report_mobius_2025_autumn_en_full.md`
- Q LHS (486 evals с П4): `experiments/results/lhs_parametric_mobius_2025_autumn_en_2026-05-12.{json,csv,md}`
- S postprocess + V + cross-validation: `experiments/results/en/` (8 analysis_*.json + V + cross-validation + 4 plot)
- Данные: `experiments/data/conferences/mobius_2025_autumn_en.{json,_embeddings.npz}`, `experiments/data/personas/personas_mobius_en.{json,_embeddings.npz}`
- Пул persons part2 + internal consistency: `experiments/data/personas/test_diversity/{personas_mobius_en_part2,internal_consistency_mobius_part2}.json`
- Скрипты: `experiments/scripts/{embed_bge_abtt,diagnose_mobius_personas_en_100,smoke_ec_mobius_en}.py`
- Bilingual LLM: `experiments/src/llm_agent.py` (поле `language`), `experiments/scripts/run_llm_lhs_subset.py` (флаг `--language`)

**Стоимость** V: $10.22 (44 160 calls на gpt-5.4-nano, cap $20); Q LLMRanker: $0.29 (gpt-4o-mini, кэш переиспользован в V на 100%). Всего $10.51, под капом 2×.

**Связанные обновления документов** (тоже 2026-05-12):

- `PROJECT_OVERVIEW.md` — блок «Update 2026-05-12 (EN pipeline pivot)» в начале; §12 (RU snapshot) не переписан
- `PROJECT_STATUS.md` §1.1 — расширен двумя абзацами; §11 + §13 обновлены
- `PREDZASHCHITA_PLAN.md` — Q1 CLOSED, R1 MITIGATED, R6 PARTIALLY MITIGATED, #23 DONE, добавлены #33 (перегон) и #34 (интеграция в ВКР)
- `presentation/speech-v3-for-kids.md` §5 + Дополнение — обновлены под EN-числа
- `presentation/slides.md` слайды результатов, cross-validation, выводы — обновлены под EN-числа
- `THESIS_CORRECTIONS_AFTER_DEFENSE.md` — новый чек-лист правок ВКР после защиты
- `docs/spikes/spike_relevance_function_audit.md` — добавлена implementation note

**Коммиты:**

- `5ae0a95` experiment(mobius): rerun parametric LHS with English BGE+ABTT-1 relevance
- `71b5628` experiment(mobius): add LLM cross-validation V on English pipeline
- (третий коммит про статус/презентацию — этим сообщением)
