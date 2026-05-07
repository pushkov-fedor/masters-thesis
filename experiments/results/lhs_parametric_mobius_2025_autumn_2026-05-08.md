# Этап Q: полный параметрический LHS-прогон

Дата: 2026-05-08
Конференция: `mobius_2025_autumn`
Master seed: 2026

## Параметры

- n_points: **50**
- replicates: **3**
- maximin_k: **12**
- include_llm_ranker: **True**
- K (top-K): 3
- audience_size grid: {30, 60, 100}
- popularity_source grid: {cosine_only, fame_only, mixed}
- w_rec ∈ [0, 0.7], w_gossip ∈ [0, 0.7], симплекс w_rel + w_rec + w_gossip = 1
- program_variant ∈ {0..5}; P_0 control + до 5 swap-модификаций (Φ)

## Wallclock breakdown

| Блок | Время, сек |
|---|---:|
| load_conference | 0.01 |
| generate_lhs | 0.11 |
| maximin_subset | 0.00 |
| prep (capacity / Φ / audience) | 4.39 |
| П1–П3 evals | 6.00 |
| П4 (llm_ranker) evals | 2455.10 |
| **итого (run_lhs внутренний)** | **2465.61** |
| **wallclock полный** | **2465.96** |

## Сводка evals по политикам

| Политика | Evals |
|---|---:|
| no_policy | 150 |
| cosine | 150 |
| capacity_aware | 150 |
| llm_ranker | 36 |
| **итого** | **486** |

П4 LLMRankerPolicy cumulative cost: **$0.2236**

## Maximin subset

Indices (12): [6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]

## Acceptance

| Чек | Значение | Статус |
|---|---|---|
| П1–П3 evals == ожидаемое | 450 == 450 | PASS |
| П4 evals == ожидаемое | 36 == 36 | PASS |
| total evals == ожидаемое | 486 == 486 | PASS |
| П4 только на maximin | violations=0 | PASS |
| CRN audience/phi инвариант | violations=0 | PASS |
| cfg_seed = replicate | violations=0 | PASS |
| long-format ключи | missing=[] | PASS |

Дополнительно (диагностика, не блокатор):
- fallback_to_p0 случаев: **0** (когда `enumerate_modifications` вернула меньше swap-вариантов чем требовался индекс program_variant; не silent — флаг записан в каждой соответствующей строке)

### Итог: **PASS**

## Артефакты

- JSON: `lhs_parametric_mobius_2025_autumn_2026-05-08.json`
- CSV (long-format): `lhs_parametric_mobius_2025_autumn_2026-05-08.csv`
- этот отчёт: `lhs_parametric_mobius_2025_autumn_2026-05-08.md`

---

# Диагностические сводки по результату Q

> Эти сводки добавлены в отчёт постфактум по уже собранным CSV/JSON. Новых прогонов не запускалось. Это **диагностика**, не финальная аналитика — формальная постобработка / pairwise-сравнения / sensitivity / cross-validation = этапы R/S/V.

## 1. П1–П3 на всех 50 LHS-точках × 3 seed (n=150 каждая)

| Политика | Метрика | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|
| no_policy | overload | 0.0394 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | overflow_rate | 0.0492 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | hall_var | 0.0170 | 0.0076 | 0.0026 | 0.0164 |
| no_policy | utility | 0.7316 | 0.7320 | 0.7288 | 0.7370 |
| cosine | overload | 0.0454 | 0.0000 | 0.0000 | 0.0067 |
| cosine | overflow_rate | 0.0521 | 0.0000 | 0.0000 | 0.0252 |
| cosine | hall_var | 0.0236 | 0.0109 | 0.0038 | 0.0221 |
| cosine | utility | 0.7329 | 0.7336 | 0.7297 | 0.7384 |
| capacity_aware | overload | 0.0336 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | overflow_rate | 0.0448 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | hall_var | 0.0109 | 0.0034 | 0.0015 | 0.0088 |
| capacity_aware | utility | 0.7322 | 0.7324 | 0.7292 | 0.7373 |

**Наблюдение (диагностически):** в среднем `capacity_aware` имеет ниже `hall_var` (0.0109 vs 0.0236 у cosine, 0.0170 у no_policy) и ниже `mean_overload_excess` (0.0336 vs 0.0454 / 0.0394). Утилита у всех трёх практически одинакова (~0.732), то есть capacity-aware не платит видимой релевантностью за лучшую балансировку. Медианы `overload` и `overflow_rate` у всех трёх политик равны 0 — это значит, что при «лояльных» настройках LHS-точек (большие capacity_multiplier, малые w_rec) переполнения не возникает.

## 2. П4 llm_ranker (только 12 maximin-точках × 3 seed = 36 evals)

| Политика | Метрика | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|
| llm_ranker | overload | 0.0092 | 0.0000 | 0.0000 | 0.0000 |
| llm_ranker | overflow_rate | 0.0124 | 0.0000 | 0.0000 | 0.0000 |
| llm_ranker | hall_var | 0.0130 | 0.0102 | 0.0023 | 0.0163 |
| llm_ranker | utility | 0.7312 | 0.7321 | 0.7269 | 0.7372 |

**Важная оговорка:** П4 рассчитана **только на 12 maximin-точках × 3 seed = 36 evals**. **Эти числа НЕЛЬЗЯ напрямую сравнивать с П1–П3 на всех 50 LHS-точках** (n=150 каждая), потому что распределение конфигураций в 12-maximin subset отличается от распределения в полном LHS-50. Корректное сравнение П4 с другими политиками возможно только на тех же 12 maximin-точках — см. §3.

## 3. Все политики на 12 maximin-точках × 3 seed (n=36 каждая)

Это честная база для cross-validation на этапе V (LLM-симулятор).

| Политика | Метрика | mean | median |
|---|---|---:|---:|
| no_policy | overload | 0.0107 | 0.0000 |
| no_policy | overflow_rate | 0.0146 | 0.0000 |
| no_policy | hall_var | 0.0137 | 0.0080 |
| no_policy | utility | 0.7311 | 0.7316 |
| cosine | overload | 0.0110 | 0.0000 |
| cosine | overflow_rate | 0.0146 | 0.0000 |
| cosine | hall_var | 0.0164 | 0.0109 |
| cosine | utility | 0.7322 | 0.7333 |
| capacity_aware | overload | 0.0080 | 0.0000 |
| capacity_aware | overflow_rate | 0.0088 | 0.0000 |
| capacity_aware | hall_var | 0.0090 | 0.0050 |
| capacity_aware | utility | 0.7317 | 0.7323 |
| llm_ranker | overload | 0.0092 | 0.0000 |
| llm_ranker | overflow_rate | 0.0124 | 0.0000 |
| llm_ranker | hall_var | 0.0130 | 0.0102 |
| llm_ranker | utility | 0.7312 | 0.7321 |

**Наблюдение (диагностически):** на 12-maximin subset порядок политик по `mean_overload_excess` (mean):
- `capacity_aware` (0.0080) ≤ `llm_ranker` (0.0092) ≤ `no_policy` (0.0107) ≈ `cosine` (0.0110).

`capacity_aware` лучший на этом subset; `llm_ranker` близок к нему. По `hall_var` тот же порядок (П3 лидирует, П4 второй). Утилита у всех 4 практически одинакова (~0.731–0.732). Финальные ранжирования и pairwise win-rate — этап S; этот результат — только sanity-указатель.

## 4. Pairwise diagnostic: capacity_aware vs cosine (агрегат по 3 seed на каждую LHS-точку)

Сравнение по 50 LHS-точкам: для каждой `lhs_row_id` усреднили метрику по 3 replicate, потом посчитали попарные разницы.

| Чек | Значение |
|---|---:|
| n_lhs_points | 50 |
| `capacity_aware < cosine` (overload) | 14 / 50 |
| `capacity_aware ≤ cosine` (overload) | **50 / 50** |
| `mean(cosine − capacity_aware)` overload | +0.0118 |
| `median(cosine − capacity_aware)` overload | +0.0000 |
| `capacity_aware < cosine` (overflow_rate) | 11 / 50 |
| `capacity_aware ≤ cosine` (overflow_rate) | **49 / 50** |
| `mean(cosine − capacity_aware)` overflow_rate | +0.0073 |
| `median(cosine − capacity_aware)` overflow_rate | +0.0000 |

**Наблюдение (диагностически):** `capacity_aware` **никогда не хуже** `cosine` по overload (50/50 точек) и почти никогда не хуже по overflow_rate (49/50). Строго лучше на ~28% точек по overload и ~22% по overflow_rate. Средняя разница в пользу П3 (+0.0118 overload, +0.0073 overflow_rate). Это **diagnostic pairwise**, не финальная аналитика этапа S — формальные win-rate / regret / TPR будут считаться на этапе S по решениям memo R.

## 5. program_variant sanity (Φ ось)

Все 6 уровней представлены; наблюдается заметная вариация overload по уровням.

| program_variant | n_evals | n_lhs_points | mean_overload | mean_utility |
|---:|---:|---:|---:|---:|
| 0 (P_0 control) | 69 | 7 | 0.0097 | 0.7320 |
| 1 | 90 | 9 | 0.0095 | 0.7320 |
| 2 | 84 | 9 | 0.0509 | 0.7322 |
| 3 | 72 | 8 | 0.0000 | 0.7312 |
| 4 | 66 | 7 | 0.1636 | 0.7321 |
| 5 | 105 | 10 | 0.0142 | 0.7330 |

**Наблюдение (диагностически):** ось `program_variant` **реально варьирует** результат — `mean_overload_excess` колеблется от 0 (PV=3) до 0.16 (PV=4) на одинаковом ядре. Утилита почти одинаковая (~0.732). Это подтверждает, что Φ-перестановки нужны как ось эксперимента (без них ось 5 LHS была бы дегенеративной). **Никаких выводов «какая перестановка лучшая» здесь не делаем** — это конфаунд с другими осями LHS, разделение требует pairwise-анализа на этапе S.

## 6. w_gossip sanity (3 bucket)

| Bucket | Диапазон | n_evals | n_lhs | mean_overload | mean_hall_var | mean_utility |
|---|---|---:|---:|---:|---:|---:|
| low | [0, 0.25) | 228 | 23 | 0.0060 | 0.0106 | 0.7325 |
| mid | [0.25, 0.5) | 174 | 19 | 0.0887 | 0.0259 | 0.7317 |
| high | [0.5, 0.7] | 84 | 8 | 0.0153 | 0.0151 | 0.7321 |

**Наблюдение (диагностически):** `mean_overload` нелинейно по `w_gossip` — выше в `mid` (0.089), ниже в `low` (0.006) и `high` (0.015). Возможные причины (без вывода): mid-bucket сочетается с другими параметрами LHS, дающими стресс; high-bucket имеет всего 8 LHS-точек (мало); конфаунд с capacity_multiplier и w_rec. **Это только sanity** — финальный sensitivity-анализ (OAT по каждой оси с фиксированными остальными) — этап S.

## 7. Wallclock и cache (диагностическая заметка)

- Полный wallclock этапа Q: **2466 сек ≈ 41 минута**, что **выше ориентира 30 минут** PIVOT этап Q.
- Причина: П4 LLMRankerPolicy на cold cache занимает ~68 сек/eval (sync API-вызовы без concurrency).
- П1–П3 (450 evals): ~6 сек суммарно (~13 мс/eval) — на порядки быстрее, не лимитирующий фактор.
- Превышение ориентира **не ломает корректность данных** — это deviation по wallclock, не по acceptance.
- В этом прогоне 1200 cache miss + ~13540 cache hits. Cache закоммичен в `experiments/logs/llm_ranker_cache.json`.
- **Для следующих прогонов** (если они будут) wallclock ожидается ниже за счёт cache hits на тех же `(user_id, slot_id, candidate_ids)`-tuples; полностью warm cache даст ~10× ускорение П4. **Перед повторным прогоном явно решить, использовать ли существующий cache или начать с пустого**, чтобы избежать смешанной семантики результатов.

## 8. Cache artifact note (`experiments/logs/llm_ranker_cache.json`)

Проверка содержимого (выполнена скриптом, не вручную):

- размер: **316 820 байт** (~317 KB), 2170 entries;
- структура: `dict[str → list[str]]` — ключ это sha1[:20] хэш `(stripped_user_id | slot_id | sorted_candidate_ids)` (см. `llm_ranker_policy.py:_cache_key`); значение — список UUID-ов talk_id в порядке ранжирования от LLM;
- проверка regex по `sk-…`, `Bearer …`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, длинные base64-look-alike: **0 совпадений**;
- персональных данных нет: ни один user_id не является персональной информацией (это синтетические `personas_100.json` ID типа `u_001..u_100`); talk_id — UUID-ы из mobius программы.

**Рекомендация:** оставить файл в git как **артефакт воспроизводимости** этапа Q. Закоммиченный cache позволяет:
- в будущих прогонах с тем же набором (audience_seed, phi_seed, candidate_ids) получить идентичные ranking-выдачи без новых API-вызовов;
- не платить заново за уже сделанные $0.2236 cost.

Если в будущем cache начнёт расти неприемлемо большим (>1 MB) или появятся изменения в формате LLMRankerPolicy — рассмотреть вынос в `.gitignore` с явной фиксацией snapshot-cache в `experiments/results/`. Сейчас ~317 KB допустимо.
