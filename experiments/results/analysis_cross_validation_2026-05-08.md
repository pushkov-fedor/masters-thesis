# Этап V: cross-validation parametric ↔ LLM

Дата: 2026-05-08
Parametric source: `analysis_llm_ranker_diagnostic.json`
LLM source: `llm_agents_lhs_subset_12pts.json`
LHS точек у LLM: 12
Evals у LLM: 48
Acceptance threshold (median Spearman ρ): ≥ 0.5

## Используемые LLM-модели (audit)

Этап V и Q/S задействуют LLM в **разных ролях**, и эти роли нельзя смешивать в интерпретации cost / calls / cache.

| Этап | LLM-роль | Модель | Кэш |
|---|---|---|---|
| Q/S (parametric) | `LLMRankerPolicy` (политика П4 — ranking docs) | `openai/gpt-4o-mini` (default класса) | `experiments/logs/llm_ranker_cache.json` |
| V | `LLMAgent` (симулятор аудитории) | `openai/gpt-5.4-nano` (params.model в JSON) | нет |
| V | `LLMRankerPolicy` (политика П4) | `openai/gpt-4o-mini` (явно прописано в `run_llm_lhs_subset.py`) | warm cache от Q (100 % hit) |

Из-за того, что V LLMRankerPolicy и Q LLMRankerPolicy используют **одну и ту же модель**, кэш ranker-вызовов от Q в V переиспользован 100 %: ноль новых API-вызовов и нулевой ranker-cost в V. Стоимость $11.55 в этапе V полностью относится к LLMAgent-вызовам.

## Acceptance per metric

| Метрика | n_LHS_in_ρ | median ρ | mean ρ | n_param_constant | n_llm_constant | passed |
|---|---:|---:|---:|---:|---:|---|
| mean_overload_excess | 2 / 12 | 0.300 | 0.300 | 10 | 5 | FAIL |
| overflow_rate_slothall | 2 / 12 | 0.741 | 0.741 | 10 | 5 | PASS |
| hall_utilization_variance | 12 / 12 | 0.400 | 0.400 | 0 | 0 | FAIL |
| mean_user_utility | 12 / 12 | 0.800 | 0.783 | 0 | 0 | PASS |

**Колонка `n_LHS_in_ρ` — критическое уточнение:** Spearman ρ пересчитывается только на тех LHS-row, где у обоих сторон ranks НЕ константны. Если все 4 политики у параметрика получают одинаковый ранг (пример: `overload_excess = 0` у всех в безопасных сценариях), Spearman undefined и LHS-row выпадает из агрегата. Колонки `n_param_constant` и `n_llm_constant` показывают сколько LHS-row дегенерированы у каждой стороны.

## Acceptance overall (median ρ across all metrics × LHS)

- median: **0.554** (PASS)
- mean: 0.581

## Top-1 match per metric

«Top-1» — политика с лучшим (минимальным) рангом. Совпало ли у параметрического и LLM, кто лучший на этой LHS-row.

**Важно:** при constant ranks `argmin` детерминированно возвращает первый индекс (`no_policy`), и top-1 у обоих сторон совпадёт автоматически — это **fake match**, не содержательное согласие. Колонка `non-degen match / n` ниже фильтрует такие случаи.

| Метрика | match all / n | match non-degen / n_non-degen | non-degen fraction |
|---|---:|---:|---:|
| mean_overload_excess | 7 / 12 | 2 / 2 | 1.00 |
| overflow_rate_slothall | 7 / 12 | 2 / 2 | 1.00 |
| hall_utilization_variance | 11 / 12 | 11 / 12 | 0.92 |
| mean_user_utility | 11 / 12 | 11 / 12 | 0.92 |

## Per-LHS-row breakdown (по `metric_mean_overload_excess`)

| LHS | param top-1 | LLM top-1 | match | ρ | τ |
|---:|---|---|---|---:|---:|
| 6 | no_policy | capacity_aware | no | — | — |
| 7 | no_policy | capacity_aware | no | — | — |
| 13 | no_policy | capacity_aware | no | — | — |
| 18 | capacity_aware | capacity_aware | yes | 0.20 | 0.00 |
| 23 | no_policy | no_policy | yes | — | — |
| 27 | no_policy | cosine | no | — | — |
| 31 | no_policy | cosine | no | — | — |
| 34 | no_policy | no_policy | yes | — | — |
| 36 | capacity_aware | capacity_aware | yes | 0.40 | 0.33 |
| 42 | no_policy | no_policy | yes | — | — |
| 45 | no_policy | no_policy | yes | — | — |
| 48 | no_policy | no_policy | yes | — | — |

## Wallclock breakdown

| Блок | сек |
|---|---:|
| read_inputs | 0.001 |
| compute_correlations | 0.014 |
| plots | 0.299 |
| total | 0.316 |

## Plots
- `plots/cross_validation_rho_per_metric.png`

## Interpretation (осторожная)

Overall median Spearman ρ = 0.554 ≥ 0.5 — **минимальный acceptance threshold пройден** (Q-O7 accepted). Это НЕ означает, что LLM полностью подтвердил параметрику; согласование умеренное и сильно неоднородное по метрикам.

**Структура согласования:**

- `mean_user_utility`: median ρ ≈ 0.80 на 12 / 12 LHS-row, top-1 match 11 / 12 — **сильное согласование**. И параметрик и LLM сходятся на том, что utility у политик почти равна (различия < 0.005).
- `overflow_rate_slothall`: median ρ ≈ 0.74, но **только на 2 / 12 non-degenerate LHS-row** — на остальных 10 параметрик даёт overflow=0 у всех 4 политик (constant ranks), Spearman undefined и пропускается. На том 2-точечном подмножестве согласование видно, но статистически малорепрезентативно.
- `mean_overload_excess`: median ρ ≈ 0.30, **только на 2 / 12 non-degenerate LHS-row**. Те же 10 параметрических safe-сценариев выпадают из ρ. Слабое согласование на ничтожной выборке — **содержательно неинформативно** для overload.
- `hall_utilization_variance`: median ρ ≈ 0.40 на 12 / 12 LHS-row (непрерывная мера, нет ties → нет skips). **Умеренно слабое согласование** — LLM и параметрик расходятся в том, как именно распределяется загрузка по залам, но top-1 совпадает 11 / 12.

**Что это значит для защиты:**

1. Overall PASS обязан в основном sustained agreement по utility и (на узкой выборке) overflow_rate.
2. Для overload и overflow_rate цифры ρ опираются на 2 LHS-row из 12 — это не валидация overload-семейства, а **диагностика**: большинство сценариев у параметрика безопасные (см. этап S §11.2), так что метрика overload не разделяет политики на этих 10 точках в принципе.
3. `gpt-5.4-nano` для LLMAgent — **бюджетная замена** более сильной модели (gpt-5.4-mini или gpt-4.1-mini). Полученное согласование — это budget cross-validation, не сильная поведенческая валидация. Стоимость full V на gpt-5.4-mini была бы ~$37 (vs $11.55 у nano) при тех же 44 160 calls.
4. Результат корректно интерпретировать как: «параметрический и LLM-симулятор сходятся по ranking-у политик в области релевантности (utility) и в небольшом подмножестве risk-positive LHS-row, но overload-семейство на mobius структурно дегенерировано и не даёт статистически содержательного ρ». Это не отказ от PROJECT_DESIGN §7 («второй независимый источник отклика») — это правдивая картина с честными ограничениями.

**Что в отчёт ВКР НЕ попадает:**

- «LLM полностью подтвердил параметрику» — это неверно;
- ρ-числа без указания `n_LHS_in_ρ` — без n=2 контекст теряется;
- top-1 match как сильное согласие на overload — половина случаев trivial при constant ranks.
