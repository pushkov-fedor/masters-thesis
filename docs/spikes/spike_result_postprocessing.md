# Design-spike: постобработка результатов Q (этап R)

Дата: 2026-05-08
Этап: R (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: accepted by user 2026-05-08 with operational parameters Q-R1—Q-R5 confirmed (см. раздел Accepted decision); design-spike, evidence-first; кода не меняет, экспериментов не запускает, к этапу S не переходит до отдельного сообщения.

> Memo evidence-first. Структура повторяет принятые ранее `spike_behavior_model.md` (этап C), `spike_llm_simulator.md` (этап G), `spike_gossip.md` (этап J), `spike_program_modification.md` (этап M), `spike_experiment_protocol.md` (этап O).

> **О research-budget.** Литература по метрикам постобработки (LHS, CRN, OAT vs Sobol, Spearman, pairwise win-rate, regret, robust ranking) была реально изучена и зафиксирована в `spike_experiment_protocol.md` §5 — там 23 реально открытых внешних источника с цитатами (scipy.qmc, Wikipedia LHS / Variance reduction / K-medoids / Sensitivity analysis / RDM, Kleijnen 2005 abstract, EMA-workbench docs, V&V Sargent 2013, Spearman, Larooij & Törnberg 2025). Этап R — «согласовать формулы постобработки» (PIVOT строка 814), не повторный обзор; для решения этой задачи уже-собранной evidence-базы достаточно. Дополнительный research-subagent в этап R не запускается — это сознательное решение, а не пропуск процедуры. Если по результату обсуждения окажется, что нужна evidence по конкретному варианту (например, robust ranking literature), subagent можно поднять точечно.

---

## Accepted decision

Статус: принято пользователем для перехода к этапу S 2026-05-08. Методических блокеров нет — направление R (pairwise win-rate / regret / OAT-bucket / risk×relevance / Spearman-prep / 6 JSON + 1 markdown / 4-5 pytest invariants) подтверждено.

Q-R1—Q-R5 закрыты как операционные параметры этапа S:

1. **Q-R1 — aggregator:** **median** over 3 replicates (default для S; робастен к выбросам, согласован с Q-diagnostic-таблицами).
2. **Q-R2 — pairwise ε:** фиксированные пороги: **`ε = 0.005`** для overload-семейства (`mean_overload_excess`, `overflow_rate_slothall`, `hall_utilization_variance`), **`ε = 0.001`** для `mean_user_utility`. Адаптивный ε и ε=0 не используются.
3. **Q-R3 — bucket-границы непрерывных осей (фиксированные):**
   - `capacity_multiplier`: `[0.5, 1.0)`, `[1.0, 2.0)`, `[2.0, 3.0]`;
   - `w_rec`: `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 0.7]`;
   - `w_gossip`: `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 0.7]`.
   Quartile-based и 5-bucket варианты отвергнуты.
4. **Q-R4 — sign-test для program_variant:** **diagnostic-only**, не gate и не causal-доказательство. Строгого conf-matching между PV=k и PV=0 в LHS-плане нет (LHS-точки с PV=0 имеют другие значения остальных осей). Sign-test считается только при N ≥ 7 LHS-row на уровень PV; результат записывается в `analysis_program_effect.json` с явной пометкой `interpretation: "diagnostic only — no conf-matching"`.
5. **Q-R5 — графики:** **minimal** в этапе S — scatter `risk × utility`, bucket bar-chart `w_gossip × policy`, ranking heatmap 12×4 на maximin. Extended-графики (per-axis scatter × policy, program_variant boxplot) — на этапе U/W при необходимости.

Содержательные правки memo по итогам подтверждения:

- §5.2 чистая формулировка структуры (lhs_row_id, policy) пар: 150 (П1–П3) + 12 (П4) = **162** unique pairs;
- §10 переформулирован: open questions отсутствуют как методические блокеры; Q-R1—Q-R5 — операционные параметры, подтверждённые здесь;
- Q-R3 в §10: вариант (а) явно описан как fixed buckets (а не quartile-based);
- Q-R4 в §10: sign-test переведён в диагностический статус с явной оговоркой про отсутствие conf-matching.

После реализации этих параметров — переход к этапу S.

---

## 1. Проблема

После этапа Q есть полный long-format dataset (486 evals на mobius_2025_autumn, master_seed=2026), который не несёт никаких выводов сам по себе — это «таблица чисел», и вопрос «что эти числа говорят» решается на этапе S через пост-обработку.

PROJECT_DESIGN §16 положение 4 («количественные оценки относительного эффекта политик рекомендаций и локальных модификаций программы по показателям риска перегрузки залов, баланса нагрузки и релевантности выбора») и положение 5 («представление результатов сценарной оценки в виде сводных показателей и визуализаций») доказываются именно артефактами этапа S, не Q.

Цель memo R — **зафиксировать минимальный воспроизводимый протокол постобработки** для этапа S, удовлетворяющий:

1. **PROJECT_DESIGN §10** (метрики групп 1–6): загрузка / риск перегрузки / релевантность / устойчивость / эффект Φ / risk × relevance.
2. **PROJECT_DESIGN §11** (анализ): попарное сравнение политик внутри точек гиперкуба, не усреднение по точкам как главный вывод; сравнение программ попарно; sensitivity по осям; согласованность двух симуляторов (этап V).
3. **Accepted decisions предыдущих spike**: симплекс w_rel + w_rec + w_gossip = 1; П4 на 12 maximin-точках (Q-O9); Spearman ρ медиана ≥ 0.5 (Q-O7); OAT + scatter без Sobol (Q-O5); program_variant как ось с фиксированным k_max=5 (Q-M3).
4. **Q-артефакты как единственный вход**: новые эксперименты не запускаются; CSV/JSON/MD из этапа Q читаются read-only.
5. **Минимизация когнитивной нагрузки**: формулы простейшие из §10, не SOTA-обвес; finalная аналитика на этапе S, не R.

Без зафиксированного протокола постобработки этапы S/T/U/W заблокированы; формулы pairwise-сравнения, regret, sensitivity-bucket, cross-validation Spearman не определены; артефакты `analysis_*.json` не имеют схемы.

---

## 2. Текущие артефакты Q (вход для S)

### 2.1. Long-format CSV / JSON

`experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.csv`:

| # | колонка | тип | назначение |
|---|---|---|---|
| 1 | `lhs_row_id` | int [0..49] | уникальный ID LHS-точки |
| 2 | `capacity_multiplier` | float [0.5, 3.0] | ось 1 |
| 3 | `popularity_source` | str ∈ {cosine_only, fame_only, mixed} | ось 2 |
| 4 | `w_rel` | float | производный (1 − w_rec − w_gossip) |
| 5 | `w_rec` | float [0, 0.7] | ось 3a |
| 6 | `w_gossip` | float [0, 0.7] | ось 3b |
| 7 | `audience_size` | int ∈ {30, 60, 100} | ось 4 |
| 8 | `program_variant` | int [0..5] | ось 5 |
| 9 | `policy` | str ∈ {no_policy, cosine, capacity_aware, llm_ranker} | политика |
| 10 | `replicate` | int [1..3] | seed-реплика |
| 11 | `audience_seed` | int | CRN-маркер аудитории |
| 12 | `phi_seed` | int | CRN-маркер Φ-эффекта |
| 13 | `cfg_seed` | int | = `replicate` |
| 14 | `is_maximin_point` | bool | принадлежность к 12-subset |
| 15 | `fallback_to_p0` | bool | Φ-fallback (в Q равен False везде) |
| 16–19 | `swap_slot_a / swap_slot_b / swap_t1 / swap_t2` | str | swap-descriptor |
| 20 | `metric_mean_overload_excess` | float ≥ 0 | основная метрика риска (PROJECT_DESIGN §10 группа 2) |
| 21 | `metric_mean_user_utility` | float [-1, 1] | релевантность (§10 группа 3) |
| 22 | `metric_overflow_rate_slothall` | float [0, 1] | доля переполненных пар (§10 группа 2) |
| 23 | `metric_hall_utilization_variance` | float ≥ 0 | дисперсия загрузки (§10 группа 1) |
| 24 | `metric_n_skipped` | int | число skip-решений |
| 25 | `metric_n_users` | int | размер аудитории по факту |

JSON дополнительно содержит: `lhs_rows[50]` (с `u_raw` каждой LHS-точки), `maximin_indices[12]` (`[6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]`), `n_evals_by_policy`, `n_p4_evals`, `p4_cost_usd` (`$0.2236`), `timings`, `acceptance`.

### 2.2. Структура (lhs_row_id × policy × replicate)

| Группа | n_points | n_policies | n_replicates | n_evals |
|---|---:|---:|---:|---:|
| П1–П3 на всех 50 LHS-точках | 50 | 3 | 3 | 450 |
| П4 на 12 maximin-точках | 12 | 1 | 3 | 36 |
| **итого** | | | | **486** |

### 2.3. Diagnostic-сводки уже в md-отчёте Q

В разделе «Диагностические сводки по результату Q» отчёта Q (commit `9ac6348`) уже посчитаны:

- mean / median / p25 / p75 для П1–П3 на всех 50 × 3 (`§1`);
- mean / median / p25 / p75 для П4 (отдельно, n=36) (`§2`);
- 4 политики на 12 maximin-точках (`§3`);
- pairwise capacity_aware vs cosine (`§4`);
- program_variant sanity (`§5`);
- w_gossip 3-bucket sanity (`§6`).

Это **первый проход** (sanity-чеки на этапе Q). Этап S должен быть формальным расширением этого первого прохода с явными формулами, репрезентацией unknown через regret / win-rate, и сохранением выходов в JSON для T/U.

---

## 3. Исследовательские вопросы (что отвечает этап S)

Семь основных вопросов, на которые этап S должен дать ответ. Каждый явно связан с PROJECT_DESIGN §10 и тем, какая именно метрика / таблица его поддерживает.

### Q-S-Risk. Какая политика устойчивее по риску перегрузки?

**§10 группа 2 + группа 4.** Метрика — `mean_overload_excess` (среднее по парам слот×зал относительного превышения вместимости).

Что считать:
- pairwise win-rate по `mean_overload_excess` на агрегате-по-LHS-row (см. §5 ниже про contract);
- regret относительно лучшей политики в каждой LHS-точке;
- агрегат по политике: median + IQR через 3 replicate × 50 LHS-точек.

### Q-S-CapVsCos. Где и как `capacity_aware` лучше `cosine`?

Дополняет diagnostic §4 Q-отчёта формальным пер-точечным анализом:
- pairwise win-rate с порогом на разницу (значимость);
- разбиение по capacity_multiplier-bucket (когда capacity тесная — больше выигрыш П3);
- разбиение по w_rec-bucket (П3 эффективна при w_rec > 0; при w_rec = 0 политики не различаются по EC3).

### Q-S-Program. Как program_variant влияет на риск (эффект Φ)?

**§10 группа 5.** Таблица: для каждого `program_variant ∈ {0..5}` агрегировать `mean_overload_excess` по политикам. Sign-test Pₖ vs P₀ для k = 1..5 при фиксированной политике — **diagnostic-only** (Q-R4 accepted), не gate, не causal-доказательство; LHS-точки с PV=k и PV=0 не conf-matched по остальным осям.

Diagnostic Q §5 показал mean overload по PV ∈ [0.000, 0.164] — это **не** «PV=4 хуже всех», а «средняя по конфаундированному смешению». Этап S должен дать pairwise-разделение.

### Q-S-Gossip. Как `w_gossip` влияет на концентрацию и overload?

Diagnostic Q §6 показал нелинейность по 3-bucket (low / mid / high). Этап S должен:
- разбить `w_gossip` по 3 bucket (как в Q-отчёте) или скользящему окну;
- внутри bucket агрегировать по политикам;
- выделить «эффект gossip изолированно от других осей» — насколько возможно через scatter и bucket-таблицу. Полная Sobol-декомпозиция отвергнута Q-O5; OAT + scatter — рекомендованный подход.

### Q-S-RiskRelevance. Есть ли trade-off между риском и релевантностью?

**§10 группа 6.** Двумерное распределение `(mean_overload_excess, mean_user_utility)` по точкам и политикам. Q-отчёт уже показывает: utility у всех 4 политик ≈ 0.731–0.732 (разброс < 0.002). Это **отсутствие видимого trade-off** на mobius — но это сильная гипотеза, требующая формальной проверки на этапе S.

### Q-S-LLMRanker. Что можно честно сказать про `llm_ranker`?

П4 на 36 evals (12 maximin × 3 replicate). Этап S должен:
- **только** на 12 maximin-точках сравнивать с П1–П3 (одинаковая база);
- **никогда** не сравнивать средние П4 (n=36) со средними П1–П3 (n=150);
- посчитать pairwise win-rate против каждой из П1–П3 на 12 maximin-точках;
- честная formulировка для отчёта: «на 12 maximin-точках П4 даёт mean_overload = X, П3 — Y, П1 — Z»; экстраполяция на 50 LHS-точек **запрещена** этапом S.

### Q-S-Stability. Насколько устойчивы выводы к рандому?

3 replicate per (lhs_row_id × policy) → среднее vs std по этим 3 replicate. Если std/mean > некоторого порога — флагуем как «нестабильный сценарий». Это операционализирует §10 группа 4 (устойчивость политики при варьировании конфигурации).

---

## 4. Каталог метрик и таблиц для этапа S

PIVOT R перечисляет 7 семейств вариантов; ниже — конкретный choice для каждого с обоснованием.

### 4.1. Pairwise win-rate (вариант 1 PIVOT R)

PIVOT R предлагает:
- (a) строгое `<`;
- (b) не-строгое `≤`;
- (c) с порогом на разницу.

**Рекомендация:** **(a) строгое + (c) с порогом** одновременно — вычислить и записать оба:

```
win_rate_strict(A, B) = |{i : metric_A[i] < metric_B[i]}| / N
win_rate_eps(A, B; ε) = |{i : metric_A[i] < metric_B[i] - ε}| / N
ties(A, B; ε)         = |{i : |metric_A[i] - metric_B[i]| ≤ ε}| / N
```

где `i` — индекс по `lhs_row_id` (агрегат по 3 replicate; см. §5), `ε` — порог значимости (по умолчанию 0.005 для overload, 0.001 для utility). Не-строгое ≤ избыточно — выводится из `win_strict + ties`.

**Зачем оба:** строгое легко интерпретируется (как у Bradley-Terry); с порогом — отвечает на «где разница достаточно большая, чтобы её обсуждать», что важно для финального отчёта (часть медиан в Q-отчёте равны 0 — это семейства точек, где политики неотличимы).

### 4.2. Regret (вариант 2 PIVOT R)

PIVOT R предлагает:
- (a) `max(metric) − metric_π` (абсолютный);
- (b) `1 − metric_π / max` (относительный).

**Рекомендация:** **(a) абсолютный для overload-минимизации**. Для метрик вида «меньше — лучше» (overload, hall_var, overflow_rate):

```
regret_π[i] = metric_π[i] - min_{π' ∈ Π[i]} metric_{π'}[i]
```

где `Π[i]` — множество политик, оценённых на LHS-точке `i`. Для 12 maximin-точек `Π[i]` = все 4 политики; для остальных 38 точек `Π[i]` = только П1–П3. Это явно фиксируется в выходе.

Для `utility` (где «больше — лучше»):

```
regret_π[i] = max_{π'} metric_{π'}[i] - metric_π[i]
```

Относительный regret (b) отвергаем: при `min = 0` (медианы overload в Q-отчёте равны 0) делитель ноль, формула вырождается.

### 4.3. Устойчивость (вариант 3 PIVOT R)

PIVOT R: range / IQR / std / Sobol total-effect.

**Рекомендация:** **range + IQR** для агрегации между политиками; **std** для интра-политики между 3 replicate. Sobol откладывается (Q-O5 accepted).

```
range_metric(policies on i)        = max_π metric_π[i] - min_π metric_π[i]
IQR_metric(policies)               = p75 - p25 (по метрикам всех политик в LHS-row)
std_replicates(policy on i)        = std(metric_π[i, replicate ∈ {1,2,3}])
```

Если `std_replicates / |mean_replicates| > 0.5` — flag «волатильная LHS-точка», отдельный отчёт.

### 4.4. Эффект Φ (вариант 4 PIVOT R)

PIVOT R: абсолютная / относительная / знаковый тест.

**Рекомендация:** **абсолютная разница** как основной показатель + **sign-test как diagnostic-only** (Q-R4 accepted: не gate, не causal-доказательство).

```
delta_phi_k(policy, metric) = mean_{i : pv[i]=k} metric_π[i] - mean_{i : pv[i]=0} metric_π[i]

# Sign-test — diagnostic only (Q-R4 accepted 2026-05-08), не gate и не causal:
sign_test_pv_k(policy, metric):                  # only if N ≥ 7
    n_pos = |{i, j : pv[i]=k, pv[j]=0, metric_π[i] > metric_π[j]}|
    n_neg = аналогично с <
    p_value = binom_test(n_pos, n_pos+n_neg, p=0.5)
    # обязательная пометка в analysis_program_effect.json:
    #   "interpretation": "diagnostic only — no conf-matching"
```

**Conf-matching не существует** — сравнение Pₖ и P₀ при одинаковых остальных осях LHS строго **невозможно** (LHS-точки с PV=0 имеют другие capacity / w_rec / w_gossip / audience_size / popularity_source). Sign-test на mismatched парах смешивает эффект Φ с конфаундом остальных осей. Поэтому ограничиваемся:
- **агрегатной разницей** по PV-уровню относительно PV=0 (без conf-matching);
- **diagnostic**: mean_overload по PV, как в Q-отчёте §5;
- **sign-test (если N ≥ 7) — только diagnostic** в выходе, никогда не как «доказательство эффекта Φ».

Относительная не считается: при mean overload P₀ ≈ 0.01 (близко к 0) деление взрывает шум.

### 4.5. Sensitivity по осям (вариант 5 PIVOT R)

PIVOT R: Morris / Sobol / OAT + scatter.

**Рекомендация:** **OAT bucket + scatter**, no Sobol (Q-O5 accepted). По каждой непрерывной оси (`capacity_multiplier`, `w_rec`, `w_gossip`):

- разбить ось на 3-5 buckets с примерно равной мощностью;
- для каждого bucket агрегировать `mean_overload_excess` per-policy;
- построить bucket-таблицу + scatter-точки `(axis_value, metric_value)` per-policy.

По дискретным осям (`popularity_source`, `audience_size`, `program_variant`):
- агрегировать по уровням (как в Q-отчёте §5);
- pairwise win-rate per-уровень (где это имеет смысл).

### 4.6. Spearman / Kendall / Hamming (вариант 6 PIVOT R)

Это для **этапа V**, не S — сравнение параметрический ↔ LLM на 12 общих точках. На этапе S делаем только **подготовку**:

- per-LHS-row для 12 maximin-точек посчитать **ранг политик** по каждой ключевой метрике (sorted ascending для overload-семейства, descending для utility);
- сохранить эти рангинги как `parametric_rankings_on_maximin.json`;
- готовый artifact подхватит этап V для сравнения с LLM-симулятором (Spearman ρ медиана ≥ 0.5 — Q-O7 accepted).

Сам Spearman / Kendall здесь **не считается** — сравнивать параметрический с самим собой не имеет смысла.

### 4.7. Gossip-диагностика (вариант 7 PIVOT R)

Q-отчёт §6 показал нелинейность по 3-bucket. Этап S должен:
- 3-bucket таблицу повторить с расщеплением по политикам (4 × 3 = 12 ячеек);
- pairwise win-rate политик внутри каждого `w_gossip`-bucket;
- scatter `(w_gossip, mean_overload_excess)` per-policy.

**Cause (без вывода):** mid-bucket overload mean = 0.089 в Q-отчёте — конфаундирован capacity_multiplier и w_rec; этап S разделит через bucket-внутри-bucket (capacity × w_gossip).

### 4.8. Сводная таблица: что для каждого вопроса

| Вопрос | Главная метрика | Главная таблица | Опционально |
|---|---|---|---|
| Q-S-Risk | mean_overload_excess | pairwise_win_rate.json | regret_per_policy.json |
| Q-S-CapVsCos | mean_overload_excess | bucket_capacity_x_policy.json | bucket_w_rec_x_policy.json |
| Q-S-Program | mean_overload_excess | program_variant_effect.json | sign_test_pv.json |
| Q-S-Gossip | mean_overload_excess | bucket_w_gossip_x_policy.json | scatter_w_gossip.json |
| Q-S-RiskRelevance | (overload, utility) | risk_utility_scatter.json | пары policy |
| Q-S-LLMRanker | mean_overload_excess | maximin_only_4policies.json | win_rate_p4_vs_p123.json |
| Q-S-Stability | std_replicates | volatile_points.json | std distribution |

---

## 5. Контракт агрегации (CRN-aware)

### 5.1. Базовый принцип

**Сначала агрегируем 3 replicate внутри (lhs_row_id, policy), потом сравниваем политики внутри одной LHS-row.**

Семантика: `cfg_seed` (= replicate) — это репликация для усреднения, не отдельный сценарий. Сравнение `policy_A vs policy_B` имеет смысл при одной и той же LHS-row (одни и те же `audience_seed`/`phi_seed` — CRN). Сравнение между LHS-row — это сравнение между сценариями (разные конфигурации); агрегат по LHS-row — это **сценарное распределение**, не «средний выигрыш».

### 5.2. Конкретные шаги

1. **Уровень 0 (raw)**: 486 строк long-format CSV.
2. **Уровень 1 (per-(lhs_row_id, policy))**: **162 unique (lhs_row_id, policy) pairs**:
   - 50 LHS × 3 политики (П1–П3) = **150 пар** (на всех LHS-точках);
   - 12 maximin × 1 политика (П4 llm_ranker) = **12 пар** (только на maximin-subset);
   - итого 162.
   - Каждый агрегат: median, mean, std по 3 replicate. Default использует **median** (Q-R1 accepted; согласован с diagnostic Q-отчёта) для робастности к выбросам.
3. **Уровень 2 (per-LHS-row, политики бок о бок)**: 50 LHS-row × 3 (или 4, если is_maximin) колонки политик, для каждой политики — 1 агрегат метрики. Это база для pairwise-сравнений и regret.
4. **Уровень 3 (агрегат по политике)**: 1 строка per-policy с distribution-описанием (median, IQR, range over LHS-rows). Это финальная сводка.

### 5.3. Особенность П4

П4 (`llm_ranker`) **присутствует только** на 12 maximin LHS-row. Все pairwise-таблицы и regret должны иметь два варианта:

- **«полная» (на 50 LHS-row)** — содержит только П1–П3;
- **«restricted-to-maximin» (на 12 LHS-row)** — содержит все 4 политики, включая П4.

Их **никогда нельзя смешивать** в одной строке-выводе.

### 5.4. Структура выходных JSON (этап S, не сейчас)

`experiments/results/analysis_pairwise.json`:
```json
{
  "etap": "S",
  "input_file": "lhs_parametric_mobius_2025_autumn_2026-05-08.json",
  "metric": "mean_overload_excess",
  "aggregator": "median_over_replicates",
  "epsilon": 0.005,
  "full_50_lhs": {
    "no_policy_vs_cosine": {
      "win_strict": 0.28, "win_eps": 0.18, "ties_eps": 0.62,
      "loss_strict": 0.10
    },
    ...
  },
  "maximin_12_lhs": {
    "no_policy_vs_cosine": {...},
    "no_policy_vs_capacity_aware": {...},
    "no_policy_vs_llm_ranker": {...},
    "cosine_vs_capacity_aware": {...},
    "cosine_vs_llm_ranker": {...},
    "capacity_aware_vs_llm_ranker": {...}
  }
}
```

---

## 6. Что нельзя делать (limitations)

Жёсткие запреты для этапа S, вытекающие из CRN-контракта и accepted Q-O9:

1. **Нельзя напрямую сравнивать средние П4 (n=36) со средними П1–П3 (n=150).** Любое такое сравнение делается **только** на 12 maximin-точках, где все 4 политики оценены на одинаковой базе.
2. **Нельзя по bucket-графикам делать causal выводы.** Bucket — это маргинальное распределение, конфаундированное другими осями LHS. Если хочется исследовать «изолированный эффект w_gossip» — нужен conditional analysis (bucket внутри bucket capacity × w_rec), и даже это не строгий cause без full Sobol (Q-O5 отверг Sobol).
3. **Нельзя объявлять PV=4 «плохим» без pairwise-контроля остальных осей.** Mean overload PV=4 = 0.164 vs PV=0 = 0.010 в Q diagnostic — это маргинал по 7 LHS-row PV=4 vs 7 LHS-row PV=0; конфаунд capacity_multiplier и w_gossip не разделён.
4. **Нельзя трактовать среднее по всем 50 LHS-точкам как единственный итог.** PROJECT_DESIGN §11: «Усреднение показателя по точкам не используется». Главные выводы — **попарные внутри точки**.
5. **Нельзя экстраполировать П4 с 12 maximin-точек на полные 50.** Maximin subset покрывает «крайние и центральные» значения каждой оси, но не репрезентативен в распределенческом смысле.
6. **Нельзя делать выводы про absolute capacity-нагрузку Mobius.** Это синтетическая аудитория с фиксированным capacity_multiplier-sweep; «Mobius всегда переполняется» / «никогда не переполняется» — out-of-scope, см. PROJECT_DESIGN §13.
7. **Нельзя делать выводы про cross-validation параметрический ↔ LLM** в этапе S. Это этап V; у нас здесь только подготовка ranking-tabular для V.
8. **Нельзя писать текст диплома в этапе S.** Этап S создаёт `analysis_*.json` + опционально лёгкие графики; текст ВКР собирается отдельно (этапы U/W).

---

## 7. Артефакты этапа S (что создаётся)

### 7.1. Обязательные файлы

`experiments/scripts/analyze_lhs.py` — единый runner этапа S, читает Q JSON read-only.

`experiments/results/analysis_pairwise.json`:
- pairwise win-rate / regret для всех 6 пар политик (4 на maximin, 3 на full-50);
- агрегат по политике: median + IQR + range для всех 4 ключевых метрик.

`experiments/results/analysis_sensitivity.json`:
- 3-bucket таблицы по `capacity_multiplier`, `w_rec`, `w_gossip`;
- агрегат по уровням `popularity_source`, `audience_size`, `program_variant`;
- per-policy расщепление для каждой оси.

`experiments/results/analysis_program_effect.json`:
- per-`program_variant` (0..5) агрегат по политикам;
- delta-Pₖ-vs-P₀ по mean_overload_excess (с явной оговоркой про конфаунд);
- sign-test **diagnostic-only** (Q-R4 accepted), считается при N ≥ 7 на уровень PV; результат записывается с обязательной пометкой `interpretation: "diagnostic only — no conf-matching"`. Не gate, не causal.

`experiments/results/analysis_gossip_effect.json`:
- 3-bucket `w_gossip` × 4 политики;
- pairwise win-rate политик внутри каждого `w_gossip`-bucket;
- conditional bucket: `w_gossip × capacity_multiplier` (диагностика конфаунда).

`experiments/results/analysis_risk_utility.json`:
- per-LHS-row пара (overload, utility) per-policy;
- маркеры trade-off (где политика А выигрывает по overload и проигрывает по utility);
- общий вывод про trade-off на mobius (Q-отчёт показывает отсутствие trade-off; S формализует).

`experiments/results/analysis_llm_ranker_diagnostic.json`:
- restricted-to-maximin таблица всех 4 политик;
- pairwise win-rate П4 vs П1–П3 на 12 LHS-row;
- ranking-вектор политик per-LHS-row (вход для этапа V cross-validation).

`experiments/results/analysis_stability.json`:
- per (lhs_row_id, policy): mean / std / std_over_mean по 3 replicate;
- список «volatile» LHS-row (std/mean > 0.5);
- distribution std по политикам.

### 7.2. Markdown-отчёт этапа S

`experiments/results/analysis_lhs_parametric_<date>.md` — собирает таблицы из всех 6 JSON в человекочитаемый отчёт. Структура:

1. Параметры входа (Q-source, master_seed, etc.).
2. Q-S-Risk: pairwise win-rate (full-50 + maximin-12).
3. Q-S-CapVsCos: bucket анализ.
4. Q-S-Program: per-PV таблица + delta-vs-P₀.
5. Q-S-Gossip: 3-bucket × 4 policy.
6. Q-S-RiskRelevance: scatter + trade-off.
7. Q-S-LLMRanker: maximin-only сравнение.
8. Q-S-Stability: volatile points.
9. **Candidate claims**: список 5–10 формулировок, которые можно использовать в тексте ВКР (но не сам текст диплома). Каждый claim с конкретной таблицей-ссылкой.
10. Limitations.

### 7.3. Опциональные графики (PIVOT этап S «Графики в `experiments/results/plots/`»)

Минимально:
- scatter `risk × utility` (4 политики на одной плоскости, по LHS-row);
- bucket bar-chart для w_gossip (3 уровня × 4 политики);
- ranking-таблица per-LHS-row (heatmap 12 × 4) для maximin subset.

Используем `matplotlib` (уже в .venv); без heavy-dep `plotly`/`seaborn`.

### 7.4. Pytest для этапа S

`experiments/tests/test_analyze_lhs.py` (новый, ~150 LOC):
- `test_pairwise_win_rate_known_data` — на синтетических 4 LHS-row × 2 политики × 1 replicate проверить расчёт win/loss/ties;
- `test_regret_zero_for_best_policy` — для лучшей политики regret = 0;
- `test_aggregator_median_over_replicates` — median = средний из 3 replicate в синтетике;
- `test_p4_appears_only_on_maximin` — на Q-данных проверить, что П4 появляется только в maximin-таблицах;
- `test_no_silent_p4_in_full_50_table` — full-50 win-rate таблица содержит **только** П1–П3.

Это 4–5 тестов. Без них S может «проходить» с тихим багом смешения П4 в full-50.

---

## 8. Acceptance этапа S (gate)

Гейт для этапа S, выводимый из этого memo:

| # | Чек | Где проверяется |
|---|---|---|
| 1 | `analyze_lhs.py` запускается на Q-JSON read-only, не модифицирует input | git-status после прогона |
| 2 | Выходы `analysis_*.json` имеют схему §7.1 (валидируются ручной inspection) | inspection |
| 3 | П4 в full-50 таблицах **отсутствует** (только в maximin-таблицах) | pytest `test_p4_appears_only_on_maximin` |
| 4 | pairwise win-rate sums (`win_strict + ties_eps + loss_strict`) ≈ 1.0 для каждой пары | pytest |
| 5 | regret лучшей политики == 0 для каждой LHS-row | pytest |
| 6 | Markdown-отчёт собирается без exception | shell run |
| 7 | pytest 103 + новые ~5 тестов = ~108 passed | `pytest tests/ -q` |
| 8 | Wallclock этапа S — секунды, не минуты (это чистая агрегация in-memory) | таймер в analyze_lhs.py |

Этап S **пройден**, если все 8 чеков зелёные.

---

## 9. Recommended decision for S

Финальная сводка решений по PIVOT R-вариантам.

### Pairwise win-rate
- **Делать**: строгий + с порогом ε (default ε=0.005 для overload, 0.001 для utility).
- **Не делать**: не-строгое ≤ как separate output (выводится из других двух).

### Regret
- **Делать**: абсолютный `metric_π - min_{π'} metric_{π'}`.
- **Не делать**: относительный (нулевой делитель).

### Устойчивость
- **Делать**: range + IQR между политиками; std между 3 replicate (внутри политики).
- **Не делать**: Sobol total-effect (Q-O5 accepted).

### Эффект Φ
- **Делать**: per-PV агрегат + delta-Pₖ-vs-P₀ по mean_overload_excess (без conf-matching, с явной оговоркой про конфаунд).
- **Опционально (diagnostic-only, Q-R4 accepted)**: sign-test, если N ≥ 7 на уровень. Записывается в `analysis_program_effect.json` с пометкой `interpretation: "diagnostic only — no conf-matching"`. Не gate, не causal-доказательство.
- **Не делать**: относительная разница; полная conf-matched регрессия; использование sign-test как acceptance-критерия или claim в тексте ВКР.

### Sensitivity по осям
- **Делать**: OAT 3-bucket + scatter; per-policy расщепление.
- **Не делать**: Morris / Sobol / Saltelli (Q-O5 accepted).

### Spearman / cross-validation
- **Не делать в S**: это этап V.
- **Делать**: подготовку ranking-vectors per-LHS-row для maximin (вход в V).

### Gossip-диагностика
- **Делать**: 3-bucket × 4 policy + conditional bucket capacity × w_gossip.
- **Опционально**: scatter `w_gossip × mean_overload`.

### Aggregator
- **Default**: median over 3 replicate (робастен к выбросам).
- **Параллельно**: mean — для pairwise-сравнений с реплика-уровневой статистикой.

### Storage
- **JSON по семейству вопросов** (§7.1) — 6 файлов; не один раздутый.
- **Markdown отчёт** — единый, ссылается на JSON.

### Tests
- 4–5 pytest для invariants (аккуратное разделение П4 / full-50, sums = 1, regret-best=0).

---

## 10. Open questions

> Методических блокеров для перехода к этапу S **нет** — направление R закреплено accepted-decisions предыдущих spike (Q-O5 OAT/scatter, Q-O7 Spearman, Q-O9 П4 на maximin) + recommended decision этого memo. Q-R1—Q-R5 ниже — это **операционные параметры этапа S**, которые пользователь подтверждает перед запуском `analyze_lhs.py` (значения по умолчанию vs альтернативы, пороги, статус sign-test). Подтверждены пользователем 2026-05-08 (см. Accepted decision блок в начале memo).

### Q-R1. Aggregator default — median или mean? (accepted 2026-05-08)

- **(а)** **median** — робастен к выбросам, согласован с Q-diagnostic-таблицами (которые использовали median). **Принято.**
- (б) mean — проще для регрессий и Sobol-decomposition; не наш случай.

**Подтверждено пользователем 2026-05-08:** (а) median.

### Q-R2. Порог ε для pairwise-win-rate (accepted 2026-05-08)

- **(а)** **ε=0.005** для overload-семейства (`mean_overload_excess`, `overflow_rate_slothall`, `hall_utilization_variance`), **ε=0.001** для `mean_user_utility`. **Принято** (порядок шума на mobius из Q diagnostic).
- (б) ε по std среди replicate (адаптивный): ε = mean(std_replicates) — отвергнуто (даёт нестабильные пороги между прогонами).
- (в) ε=0 (только строгий) — отвергнуто (теряем «значимость»).

**Подтверждено пользователем 2026-05-08:** (а) фиксированные ε.

### Q-R3. Bucket-границы для непрерывных осей (accepted 2026-05-08)

- **(а) Фиксированные интервалы** для согласованности с Q-отчётом §6 (где `w_gossip` уже разбит по 0.25 / 0.5):
  - `capacity_multiplier`: `[0.5, 1.0)`, `[1.0, 2.0)`, `[2.0, 3.0]`;
  - `w_rec`: `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 0.7]`;
  - `w_gossip`: `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 0.7]`.
  **Принято.**
- (б) Quartile-based (p33, p67) — отвергнуто: даёт неровные границы между прогонами с разным master_seed; теряется согласованность с Q-отчётом §6.
- (в) 5-bucket — отвергнуто: на 50 LHS-точках 5-bucket даёт по 10 точек на bucket, маржинально для статистики; 3-bucket остаётся стандартом.

**Подтверждено пользователем 2026-05-08:** (а) fixed intervals.

### Q-R4. Знаковый тест для Φ-эффекта — статус (accepted 2026-05-08)

PIVOT R пункт 4: «знаковый тест по точкам».

**Принято: sign-test — diagnostic only, не gate и не causal-доказательство.**

Обоснование:
- LHS-точки с PV=k и PV=0 имеют **разные значения остальных осей** (capacity_multiplier, w_rec, w_gossip, audience_size, popularity_source). Conf-matching между ними **не гарантирован** в LHS-плане; sign-test на mismatched парах не оценивает «эффект Φ изолированно», а смешивает его с конфаундом остальных осей.
- Sign-test остаётся как **диагностика** распределения знака разницы Pₖ vs P₀ при условии N ≥ 7 LHS-row на уровень PV (минимум для биномиальной статистики).
- В выходе `analysis_program_effect.json` явно записать `interpretation: "diagnostic only — no conf-matching"` рядом с p-value.
- **Не использовать** sign-test как acceptance gate или causal claim в тексте ВКР.

Альтернативные варианты:
- (б) Не делать sign-test вообще — отвергнуто: distribution знака полезна для диагностики;
- (в) Делать sign-test всегда (на любом N) — отвергнуто: при N < 7 биномиальная статистика без значимости.

### Q-R5. Графики — minimal или extended? (accepted 2026-05-08)

- **(а)** **Minimal**: scatter `risk × utility` + bucket bar-chart `w_gossip × policy` + heatmap rankings 12×4 на maximin. **Принято.**
- (б) Extended: + scatter по каждой непрерывной оси × policy + program_variant boxplot — отвергнуто (overhead в S; лишние графики стоит делать на U/W при необходимости).

**Подтверждено пользователем 2026-05-08:** (а) minimal в S; extended — на U/W при необходимости.

---

## Recommended decision for S (краткая сводка)

| Параметр | Значение |
|---|---|
| Aggregator default | median over replicates |
| Pairwise win-rate | строгое + ε-пороговое (ε=0.005 для overload, 0.001 для utility) |
| Regret | абсолютный |
| Устойчивость | range + IQR между политиками; std между replicate |
| Эффект Φ | per-PV agg + delta-vs-P₀; sign-test **diagnostic-only** при N ≥ 7 (не gate, не causal) |
| Sensitivity | OAT 3-bucket + scatter; **без Sobol/Morris** |
| Cross-validation | в этапе V; в S — только подготовка ranking |
| Gossip-блок | 3-bucket × 4 policy + conditional capacity × w_gossip |
| Storage | 6 JSON по семействам + 1 markdown report |
| Графики | minimal: scatter, bucket bar, ranking heatmap |
| Tests | 4–5 pytest для invariants |
| Файлы для S | `experiments/scripts/analyze_lhs.py`, `experiments/tests/test_analyze_lhs.py`, 6 `analysis_*.json` + 1 markdown + 3 plots |

**Что не делается в S:** новые эксперименты, изменение Q/P-артефактов, текст диплома, cross-validation с LLM (этап V), Sobol/Morris, full integer programming.

**Q-R1—Q-R5 подтверждены пользователем 2026-05-08** (см. Accepted decision блок в начале memo). К этапу S не перехожу до отдельного сообщения пользователя.
