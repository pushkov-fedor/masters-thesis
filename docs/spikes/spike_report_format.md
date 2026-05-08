# Design-spike: формат экспериментального отчёта (этап T)

Дата: 2026-05-08
Этап: T (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: design-spike, evidence-first; кода не меняет, экспериментов не запускает, к этапу U не переходит до отдельного сообщения.

> Memo evidence-first и compact. T — это **формат упаковки уже готовых результатов**, не новый экспериментальный или методический компонент. Все входы (Q/S/V) к моменту T закрыты и зафиксированы коммитами. Новый research-subagent не запускается.

---

## Accepted decision

Статус: предложено для подтверждения пользователем. Содержательных решений в T нет — только инженерный выбор формата отчёта.

Recommended:

1. **Compact technical Markdown** — без Quarto, HTML, PDF, notebook, pandoc, JS-зависимостей. Это технический экспериментальный артефакт, **не текст диплома**: цифры, таблицы, короткие выводы (1–2 предложения на блок), ограничения, ссылки на исходные JSON / графики.
2. **PNG-графики** опциональны: подключаются только если они уже есть в `experiments/results/plots/` и реально передают результат лучше таблицы. Никакой генерации новых графиков на T/U/W.
3. **Два файла отчёта**:
   - `experiments/results/report_mobius_2025_autumn_parametric.md` — этап **U**, без LLM-блока;
   - `experiments/results/report_mobius_2025_autumn_full.md` — этап **W**, U + LLM cross-validation блок из V.
4. **Размер**: ориентир **100–250 строк на U, 150–350 строк на W**. Если таблиц получается много — ссылаться на `analysis_*.json`, а не дублировать в markdown.
5. **Никакого собственного кода-генератора** (`build_report.py`) на этапе T не предполагается. Решение по `experiments/src/report.py` откладывается на U.
6. **Источник данных**: только уже закоммиченные `analysis_*.json` и `lhs_parametric_*.{json,md}` + `llm_agents_lhs_subset_12pts.{json,md}` + `analysis_cross_validation.json`. Никаких новых вычислений.

---

## 1. Проблема

После закрытых этапов Q/S/V у нас есть:

- 486 evals параметрического LHS;
- 9 `analysis_*.json` + 3 PNG-графика;
- 48 evals LLM-симулятора + cross-validation JSON + 1 PNG-график;
- 16 capacity / overload / utility / risk-positive / volatile-points выводов.

**Проблема**: эти артефакты — машинные. Они содержат всё необходимое для итогового сравнительного отчёта (`PROJECT_DESIGN §6: «Целевой пользователь системы — организатор программы конференции. Результат предъявляется в виде сравнительного отчёта»`), но не структурированы как один читаемый markdown.

**Цель T**: согласовать **схему упаковки** — без экспериментов, без новых вычислений, без research-subagent. T — это design-spike формата, не содержания.

PROJECT_DESIGN §6 явно требует от итогового отчёта: «карту ожидаемой загрузки залов, список горячих точек программы, сравнительные таблицы политик рекомендаций, перечень модификаций программы с количественной оценкой эффекта, графики чувствительности, раздел допущений».

---

## 2. Что уже готово (входные артефакты для U/W)

### 2.1. Q (полный параметрический LHS-прогон)

Закоммичено в `b4e0787` + `9ac6348`.

- `experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.json` — long-format 486 evals + LHS plan + maximin indices + acceptance + timings;
- `experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.csv` — long-format CSV;
- `experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.md` — Q acceptance отчёт + diagnostic-сводки.

Acceptance Q: **PASS** (все 7 чеков). Evals: П1=150, П2=150, П3=150, П4=36, total=486. Maximin indices: `[6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]`.

### 2.2. S (post-processing)

Закоммичено в `def0887`.

- `analysis_pairwise.json` — full-50 + maximin-12 pairwise win-rate / regret / per-policy distributions;
- `analysis_sensitivity.json` — 3-bucket OAT по `capacity_multiplier`/`w_rec`/`w_gossip` + дискретные оси;
- `analysis_program_effect.json` — per-PV агрегат + delta-vs-P0 + sign-test diagnostic-only;
- `analysis_gossip_effect.json` — 3-bucket × policy + conditional capacity × w_gossip + pairwise within bucket;
- `analysis_risk_utility.json` — per-LHS-row × policy points + trade-off markers;
- `analysis_llm_ranker_diagnostic.json` — restricted-to-maximin + ranking vectors (вход для V cross-validation);
- `analysis_stability.json` — std/mean per (lhs_row, policy) + volatile points;
- `analysis_capacity_audit.json` — per-slot capacity Mobius + LHS distribution × audience + risk-positive vs capacity-stress separation + capacity_aware effect + critical infeasible LHS;
- `analysis_lhs_parametric_2026-05-08.md` — markdown отчёт S с разделами 1–13 и capacity sanity;
- `plots/analysis_risk_utility_scatter.png`;
- `plots/analysis_gossip_bucket_bar.png`;
- `plots/analysis_ranking_heatmap_maximin.png`.

### 2.3. V (LLM cross-validation)

Закоммичено в `af1579c`.

- `experiments/results/llm_agents_lhs_subset_12pts.json` — long-format 48 evals + params + cost / time breakdown + Q/S sha256 invariant PASS;
- `experiments/results/llm_agents_lhs_subset_12pts.csv` — long-format CSV;
- `experiments/results/llm_agents_lhs_subset_12pts.md` — V acceptance + per-eval table + audit моделей;
- `experiments/results/analysis_cross_validation.json` — Spearman ρ + Kendall τ + top-1 Hamming на 12 maximin × 4 метрики + degenerate diagnostic;
- `experiments/results/analysis_cross_validation_2026-05-08.md` — V cross-validation отчёт с осторожной интерпретацией;
- `experiments/results/plots/cross_validation_rho_per_metric.png`.

V models (audit):
- LLMAgent (симулятор аудитории): **`openai/gpt-5.4-nano`**;
- LLMRankerPolicy (П4): **`openai/gpt-4o-mini`** (та же что в Q/S — warm cache 100 % hit, 0 новых ranker calls в V, ranker cost = $0).

V acceptance: overall median Spearman ρ = **0.5536 ≥ 0.5** → **PASS** (Q-O7).

---

## 3. Почему новых экспериментов на T не нужно

Этап T — это **выбор формата упаковки**, не методический компонент. Все артефакты для U/W уже:

- посчитаны (Q, S, V);
- сохранены на диск;
- зафиксированы коммитами;
- проверены acceptance gate'ами.

Любой research-subagent на T дублировал бы work этапа O (формат экспериментального отчёта в PIVOT § «Этап T» уже описан как 6 кандидат-вариантов 1. pure markdown, 2. Quarto/MyST, 3. HTML через nbconvert, 4. PDF через pandoc, 5. notebook, 6. hybrid). Из них только pure markdown и hybrid реализуемы без новых зависимостей и в timeline до 13.05 предзащиты.

Pure markdown выбирается как минимально-инвазивный.

---

## 4. Recommended decision: compact technical markdown

### 4.1. Жанр отчёта

U/W — это **технический экспериментальный артефакт**, не текст диплома и не литературный нарратив. Принципы:

- цифры, таблицы, acceptance, key findings, limitations;
- короткие выводы (1–2 предложения на блок);
- никаких «глав» и длинных объяснений;
- candidate claims = краткий список с числами, не готовый академический текст;
- ссылки на исходные JSON / графики, а не пересказ их содержимого.

Литературный текст и развёрнутые формулировки идут отдельно в `thesis/*.tex`, не сюда.

### 4.2. Что использовать

- **Markdown** в спецификации CommonMark + GFM-таблицы.
- **PNG-графики** уже готовы в `experiments/results/plots/`. Подключать **только если** график понятнее таблицы. По умолчанию — таблица; график опционален.
- **Цифры** копируются вручную или через python-однострочник из `analysis_*.json`. На T этот выбор не делается — он внутри U.

### 4.3. Что НЕ использовать

- Quarto / MyST / HTML / pandoc / Jupyter notebook — лишние зависимости и потеря git-readability.
- Дублирование содержимого больших таблиц из `analysis_*.json` — лучше ссылка на JSON.
- Литературный stream-of-thought, мотивации, экспозиции — это не сюда.

### 4.4. Почему compact markdown

1. **Минимум зависимостей**: pip-окружение `experiments/.venv` уже имеет всё нужное — markdown не требует ничего сверху.
2. **Git-readability**: научрук / рецензент читает прямо в репозитории; PNG показываются GitHub-вьювером.
3. **Простой diff**: при правках через PR видно построчно что изменилось.
4. **PROJECT_DESIGN §6**: «формат — markdown + графики, генерируется автоматически по результатам прогона» (буквальная формулировка из PROJECT_STATUS §10).

### 4.5. Файлы U и W

- **U** (parametric only): `experiments/results/report_mobius_2025_autumn_parametric.md` — ориентир **100–250 строк**.
- **W** (full = U-key + V cross-validation): `experiments/results/report_mobius_2025_autumn_full.md` — ориентир **150–350 строк**.

W не дублирует U полностью: W воспроизводит **только ключевые таблицы** из U (acceptance Q/S, capacity sanity, pairwise full-50 для overload, capacity_aware effect на risk-positive, risk × utility) и добавляет блок LLM cross-validation. Если рецензент хочет полный набор парадигм по программе — он читает U; для финального отчёта с LLM-валидацией — W.

---

## 5. Структура будущего U (compact)

Спецификация разделов; контент пишется на этапе U в **компактном** виде — таблицы, числа, 1–2 предложения вывода на блок, ссылки на JSON. Графики опциональны.

| § | Раздел | Что содержит | Источник |
|---|---|---|---|
| 1 | Metadata / source artifacts | даты, конференция, master_seed, ссылки на Q/S JSON, git-хеши коммитов Q/S | `lhs_parametric_*.json` params + git log |
| 2 | Experiment parameters | LHS n_points, replicates, audience grid, popularity_source grid, диапазоны w_rec/w_gossip, K | `lhs_parametric_*.json` params |
| 3 | Acceptance Q + S | Q acceptance таблица 7 чеков → PASS; S acceptance gate (8 чеков) → PASS | `lhs_parametric_*.md` + `analysis_lhs_parametric_2026-05-08.md` |
| 4 | Key numeric results | per-policy distribution `mean_overload_excess` / `utility` / `overflow_rate` / `hall_var` (full-50 + maximin-12); 1 строка вывода | `analysis_pairwise.json` distribution блок |
| 5 | Capacity sanity | per-slot capacity Mobius; LHS distribution stress / normal / loose × audience; risk-positive 13/50; critical infeasible 3 LHS-row | `analysis_capacity_audit.json` |
| 6 | Pairwise comparison | full-50 winrate таблица overload (3 пары × win_eps / ties_eps / loss_eps); maximin-12 winrate (6 пар) | `analysis_pairwise.json` |
| 7 | Risk-positive subset | где capacity_aware строго снижает риск: 11/13 strict-better; per-LHS таблица 13 строк | `analysis_capacity_audit.json` §11.4 |
| 8 | program_variant / gossip diagnostics | per-PV mean overload + delta-vs-P0 + sign-test (diagnostic-only); 3-bucket w_gossip × policy | `analysis_program_effect.json` + `analysis_gossip_effect.json` |
| 9 | Risk × utility | trade-off markers count; короткий вывод про utility ≈ const | `analysis_risk_utility.json` + опционально `plots/analysis_risk_utility_scatter.png` |
| 10 | Limitations | 8 пунктов из §7 этого memo (не causal по PV; не сравнивать П4 maximin с П1–П3 full-50; не выдавать absolute capacity за прогноз; стоп-лист тезисов) | spike_result_postprocessing §6 + §7 этого memo |
| 11 | Links to JSON / plots | список `experiments/results/analysis_*.json` + `plots/*.png` | (этот раздел — пути) |

Итого ~11 разделов; ориентир **100–250 строк markdown**. Если таблицы получаются длинные (per-LHS на 50 строк) — даём только агрегаты в markdown и ссылку на JSON.

---

## 6. Структура будущего W (compact)

W не дублирует U полностью. Воспроизводит **только ключевые таблицы U** (acceptance Q/S, capacity sanity, pairwise full-50 для overload, capacity_aware effect на risk-positive, risk × utility) + добавляет компактный блок LLM cross-validation.

| § | Раздел | Что содержит | Источник |
|---|---|---|---|
| 1 | Metadata + summary | даты, ссылки на Q/S/V JSON, git-хеши | params блоки JSON + git log |
| 2 | Key results from U (compressed) | acceptance Q/S, per-policy distribution overload + utility (full-50 и maximin-12), capacity sanity 1 таблица, pairwise full-50 overload 1 таблица, capacity_aware effect 1 таблица | агрегаты из `analysis_*.json` |
| 3 | LLM simulation setup | конференция, audience grid, persona pool, gossip levels off/moderate/strong, w_gossip mapping, hard cap | `llm_agents_lhs_subset_12pts.json` params + `spike_llm_simulator.md` |
| 4 | Model audit | LLMAgent = `openai/gpt-5.4-nano`; LLMRankerPolicy = `openai/gpt-4o-mini`; разные роли (предотвращает путаницу cost/calls/cache) | `llm_agents_lhs_subset_12pts.md` audit-блок |
| 5 | V acceptance + cost | 48 evals, 0 timeout / 0 errors, cost $11.55, wallclock 1ч 46мин, Q/S sha256 invariant PASS | `llm_agents_lhs_subset_12pts.json` |
| 6 | Cross-validation table | 1 таблица 4 метрики × `n_LHS_in_ρ` / median ρ / param_const / llm_const / acceptance | `analysis_cross_validation.json` |
| 7 | Overall PASS + per-metric caveat | overall median ρ=0.5536 PASS; utility/overflow_rate PASS; overload/hall_var weaker; degenerate diagnostic для overload (10/12 LHS константны) | `analysis_cross_validation.json.acceptance_overall` + per_metric |
| 8 | Interpretation (осторожная) | LLM cross-validation **partially supports** parametric conclusions, не полностью валидирует каждую метрику; budget cross-validation на nano, не сильная поведенческая валидация | `analysis_cross_validation_2026-05-08.md` § Interpretation |
| 9 | Final limitations | combined: 8 пунктов U §10 + 4 пункта V (degenerate cases в overload; budget cross-validation; модель различия; узкая выборка ρ) | §7 этого memo + cross-validation md |
| 10 | Links to JSON / plots | пути к Q/S/V JSON и графикам | (раздел — пути) |

Итого ~10 разделов; ориентир **150–350 строк markdown**.

---

## 7. Ограничения интерпретации (что нельзя делать в U/W)

Жёсткие запреты, вытекающие из accepted decisions Q-O9 / Q-R4 / Q-J7 и V cross-validation diagnostic. Эти запреты должны быть явно зафиксированы в § Limitations U и W.

1. **Не сравнивать П4 на 12 maximin с П1–П3 на full-50.** П4 присутствует только в restricted-to-maximin блоке. Любое сравнение средних П4 (n=36) со средними П1–П3 (n=150) — нарушение Q-O9.

2. **Не делать causal claims по program_variant.** LHS-точки с PV=k и PV=0 имеют разные значения остальных осей; conf-matching отсутствует. Sign-test для program_variant — diagnostic-only, не gate, не доказательство эффекта Φ. Это требование Q-R4.

3. **Не делать causal claims по bucket-графикам.** Bucket-агрегаты конфаундированы между осями; OAT scatter не даёт строгого causal cleavage. Это требование Q-O5.

4. **Не скрывать структуру выборки**: 74 % LHS-точек на mobius — безопасные сценарии (overload=0 у всех 4 политик). Основной эффект политик виден на risk-positive subset (26 % LHS). Capacity sanity §11.4 явно фиксирует: на 100 % risk-positive точек capacity_aware не хуже max(no_policy, cosine) за ε; на 85 % строго снижает риск.

5. **Не писать «LLM полностью подтвердил параметрический симулятор».** Cross-validation overall median Spearman ρ = 0.5536 — это **минимальный acceptance threshold пройден**, не сильная валидация. Per-metric картина: utility / overflow_rate PASS, overload_excess / hall_var FAIL. Для overload и overflow_rate ρ считается на 2 / 12 non-degenerate LHS-row — статистически малорепрезентативно.

6. **Не путать LLM-роли**:
   - LLMAgent (V) = `gpt-5.4-nano` — симулятор аудитории, новые API calls.
   - LLMRankerPolicy (Q + S + V) = `gpt-4o-mini` — политика П4, warm cache от Q.
   - Стоимость V = $11.55 относится к LLMAgent calls; ranker calls в V = 0 (cache hits).
   - Не выдавать nano за валидацию на дорогой модели — V это **budget cross-validation**, не сильная поведенческая валидация.

7. **Не выдавать absolute capacity-нагрузку Mobius** как прогноз реальной конференции. Это синтетическая аудитория, capacity_multiplier-sweep, сценарный анализ. PROJECT_STATUS §3 главный тезис: «абсолютные числа из системы не выдаются за прогноз реальной посещаемости конкретной конференции».

8. **Не возвращать стоп-лист тезисов** (PROJECT_STATUS §5): «лучший персональный рекомендатель», «B1 / accuracy@1 = 0.918 как внешняя валидация», «cross-domain Spearman как валидация реальности», Big Five / social graph / inter-slot chat как реализованный метод.

---

## 8. Acceptance этапа T

| # | Чек | Где проверяется |
|---|---|---|
| 1 | `docs/spikes/spike_report_format.md` создан | существование файла |
| 2 | Recommended format зафиксирован: pure markdown + PNG | §4 memo |
| 3 | Перечислены входные артефакты Q / S / V | §2 memo |
| 4 | Структура U зафиксирована (13 разделов) | §5 memo |
| 5 | Структура W зафиксирована (U + 8 разделов) | §6 memo |
| 6 | Ограничения интерпретации зафиксированы (8 пунктов) | §7 memo |
| 7 | Кода / результатов Q/S/V / тестов не трогали | git status: только новый T memo |
| 8 | Working tree содержит только новый T memo | git status |

Этап T **пройден**, если все 8 чеков зелёные.

---

## 9. Что НЕ входит в T

- Сам отчёт U (`report_mobius_2025_autumn_parametric.md`) — это этап U.
- Сам финальный отчёт W (`report_mobius_2025_autumn_full.md`) — это этап W.
- Реализация генератора `experiments/src/report.py` (если будет) — решение откладывается на U.
- Любые правки в `analysis_*.json`, `lhs_parametric_*.{json,csv,md}`, `llm_agents_lhs_subset_12pts.{json,csv,md}`, `analysis_cross_validation.{json,md}`. Все артефакты Q/S/V — read-only на этапах T/U/W.
- Любые новые экспериментальные прогоны.

---

## 10. После T

- К U/W не перехожу до отдельного сообщения пользователя.
- Если по итогам обсуждения появятся правки memo — точечные правки в этом же файле, без переименования.
- Если будет решение объединить T+U+W в один сжатый шаг — это меняет план PIVOT, требует отдельного user-decision и пометки в `PIVOT_IMPLEMENTATION_PLAN.md`.
