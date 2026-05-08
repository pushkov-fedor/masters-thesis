# Final experiment report — Mobius 2025 Autumn (parametric + LLM cross-validation)

Compact technical artefact (W-stage). Не текст диплома; только цифры, таблицы,
acceptance, ключевые выводы и ограничения. Параметрические подробности — в
[U-отчёте](report_mobius_2025_autumn_parametric.md). Этот файл — агрегат для
финальной точки чтения с LLM cross-validation блоком.

## 1. Metadata

| Поле | Значение |
|---|---|
| Конференция | `mobius_2025_autumn` |
| Дата отчёта | 2026-05-08 |
| master_seed | 2026 |
| Q (parametric LHS) | 50 LHS × {П1, П2, П3, П4 на maximin} × 3 replicate = **486 evals** |
| V (LLM cross-validation) | 12 maximin LHS × 4 политики × 1 seed = **48 evals**, 44 160 LLMAgent calls |

## 2. Inputs / source artefacts

Все read-only:

| Stage | Commit | Файлы |
|---|---|---|
| Q | `b4e0787`, `9ac6348` | `lhs_parametric_mobius_2025_autumn_2026-05-08.{json,csv,md}` |
| S | `def0887` | 8 × `analysis_*.json`, `analysis_lhs_parametric_2026-05-08.md`, 3 plot |
| T (memo) | `a40c3e5` | `docs/spikes/spike_report_format.md` |
| U | `7052e2f` | `experiments/results/report_mobius_2025_autumn_parametric.md` |
| V | `af1579c` | `llm_agents_lhs_subset_12pts.{json,csv,md}`, `analysis_cross_validation.{json,md}`, `plots/cross_validation_rho_per_metric.png` |

## 3. Q/S parametric summary

Q acceptance — **PASS** (7 чеков из `lhs_parametric_*.md`): П1–П3 evals = 450 == 450,
П4 evals = 36 == 36, total = 486 == 486, П4 только на maximin (violations=0),
CRN audience/phi инвариант (violations=0), cfg_seed = replicate (violations=0),
long-format ключи missing=[].

S acceptance — **PASS** (8 чеков по `spike_result_postprocessing.md` §8):
analyze_lhs.py не мутирует Q (sha256), 8 analysis JSON + markdown + 3 plot созданы,
П4 отсутствует в full-50 таблицах, win_eps + ties_eps + loss_eps ≈ 1, regret лучшей
политики = 0, markdown сборка без exception, pytest зелёный, wallclock S = 0.32 s.

Per-policy distribution (full-50, П1–П3):

| Policy | overload mean | overload median | overload p75 | utility mean | utility median |
|---|---:|---:|---:|---:|---:|
| no_policy | 0.0395 | 0.0000 | 0.0000 | 0.7308 | 0.7318 |
| cosine | 0.0449 | 0.0000 | 0.0066 | 0.7321 | 0.7331 |
| capacity_aware | 0.0336 | 0.0000 | 0.0000 | 0.7314 | 0.7323 |

## 4. Key numeric findings (parametric)

### 4.1. Pairwise full-50 overload (`mean_overload_excess`, ε=0.005)

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps |
|---|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.20 | 0.14 | 0.86 | 0.04 | 0.00 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.86 | 0.20 | 0.14 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.78 | 0.26 | 0.22 |

`capacity_aware` **ни на одной** из 50 LHS-row не показывает overload выше cosine
за ε. Подробности и `overflow_rate` / `hall_var` — в `analysis_pairwise.json`.

### 4.2. capacity_aware на risk-positive subset

13 / 50 LHS-row (26 %) — overload > 0 хотя бы у одной из П1–П3 (risk-positive).
На этих точках `capacity_aware`:

- **не хуже за ε (0.005)** vs `max(no_policy, cosine)`: **13 / 13 (100 %)**;
- **строго снижает риск за ε**: **11 / 13 (85 %)**.

Top reductions vs cosine (full breakdown — `analysis_capacity_audit.json`):

| LHS | cap_m | aud | overload no_policy | cosine | capacity_aware | Δ vs cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 49 | 0.774 | 100 | 0.4647 | 0.5449 | 0.3558 | +0.1891 |
| 26 | 0.629 | 60 | 0.0714 | 0.1706 | 0.0040 | +0.1667 |
| 35 | 0.963 | 100 | 0.0505 | 0.0606 | 0.0025 | +0.0581 |
| 18 | 1.040 | 60 | 0.0095 | 0.0214 | 0.0000 | +0.0214 |

### 4.3. Gossip / program_variant

Gossip 3-bucket × policy (`mean_overload_excess`, full-50):

| w_gossip bucket | n_lhs | no_policy | cosine | capacity_aware |
|---|---:|---:|---:|---:|
| [0, 0.25) low | 23 | 0.0061 | 0.0123 | 0.0003 |
| [0.25, 0.5) mid | 19 | 0.0891 | 0.0957 | 0.0822 |
| [0.5, 0.7] high | 8 | 0.0178 | 0.0178 | 0.0139 |

Нелинейный пик в среднем bucket. `capacity_aware` лучший во всех трёх.

`program_variant` (Φ): mean overload PV=4 = 0.176 vs PV=0 = 0.010 (cosine view), но
sign-test p ≈ 0.84 — **diagnostic only**, не causal (LHS-row с разными PV не conf-matched
по остальным осям). `analysis_program_effect.json` содержит явное поле
`interpretation: "diagnostic only — no conf-matching"` во всех ячейках sign-test.

### 4.4. Risk × utility, stability

Trade-off risk × utility проявляется редко: 11 маркеров (ε_o=0.005, ε_u=0.001) из
150 full-50 точек. Utility между политиками почти не меняется (разброс < 0.002).

Volatile points (`std/|mean| > 0.5`): 24 entries в full-50 П1–П3, 4 в maximin П4.

## 5. Capacity sanity / risk-positive subset

Mobius: 16 слотов (12 параллельных по 3 зала × 34 ≈ 102 на слот; 4 пленарных).
`audience_size ∈ {30, 60, 100}` согласован с per-slot capacity при cap_m=1.0.

| Bucket по cap_multiplier | n_lhs | a30 | a60 | a100 |
|---|---:|---:|---:|---:|
| stress [0.5, 1.0) | 10 | 4 | 3 | 3 |
| normal [1.0, 2.0) | 21 | 5 | 9 | 7 |
| loose [2.0, 3.0]  | 19 | 7 | 6 | 6 |

- 13 / 50 LHS (26 %) — risk-positive (overload > 0 у хотя бы одной П1–П3);
- 37 / 50 (74 %) — safe (все политики дают overload = 0);
- 3 critical infeasible LHS (`audience > per-slot capacity × cap_multiplier`):
  LHS=3 (cap_m=0.534, aud=100, effective=53.4); LHS=35 (effective=96.3); LHS=49
  (effective=77.4). Physical overload неустраним никакой политикой.

«Risk-positive» (по факту overload > 0) шире capacity-stress bucket — туда же
попадают tight-normal LHS с audience=100. Median overload = 0 в full-50 — структурное
свойство сценарного анализа, не отсутствие эффекта политик.

## 6. LLM simulation setup (V)

| Поле | Значение |
|---|---|
| target subset | maximin-12 (`[6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]`) |
| политики | П1, П2, П3, П4 (все 4) |
| evals | 48 (12 LHS × 4 policy × 1 seed) |
| cfg_seed | 1 (соответствует replicate=1 в Q для warm cache) |
| K (top-K) | 3 |
| concurrency | 16 |
| parallel-lhs | 4 |
| budget cap | $20 |
| gossip levels | off (w=0), moderate (0 < w < 0.4), strong (w ≥ 0.4) |
| persona pool | `personas_100` (KMeans-subsample по `audience_seed` per LHS) |

## 7. Model audit

LLM используется в **разных ролях** в Q/S и V — нельзя смешивать в интерпретации
cost / calls / cache:

| Stage | LLM-роль | Модель | Кэш |
|---|---|---|---|
| Q/S (parametric) | `LLMRankerPolicy` (политика П4 — ranking docs) | `openai/gpt-4o-mini` (default класса) | `experiments/logs/llm_ranker_cache.json` |
| V | `LLMAgent` (симулятор аудитории) | **`openai/gpt-5.4-nano`** (params.model в JSON) | нет |
| V | `LLMRankerPolicy` (П4) | `openai/gpt-4o-mini` (hard-coded в `run_llm_lhs_subset.py:681`) | warm cache от Q (100 % hit) |

Q LLMRankerPolicy и V LLMRankerPolicy используют **одну модель** → cache переиспользован
полностью: 0 новых API-вызовов и нулевая ranker-стоимость в V.

## 8. V acceptance

| Метрика | Значение |
|---|---:|
| evals | 48 / 48 ✓ |
| LLMAgent calls | 44 160 |
| LLMAgent cost | **$11.5488** (cap $20) |
| LLMRanker new API calls | 0 |
| LLMRanker new cache misses | 0 |
| LLMRanker source | ranking results reused from warm Q cache |
| LLMRanker cost (delta) | $0.0000 |
| Wallclock | 6385.4 s ≈ 1 ч 46 мин |
| timeout / errors | 0 / 0 |
| parse errors | 214 / 44 160 (0.48 %) |
| Q/S sha256 invariant | **PASS** (violations=[]) |
| Status | `ok` |

## 9. Cross-validation summary

Метод: per-LHS-row Spearman ρ ранжирований политик parametric ↔ LLM на 12 maximin
точках, для 4 ключевых метрик. Acceptance gate Q-O7: `median ρ ≥ 0.5`.

| Метрика | n_LHS_in_ρ / 12 | median ρ | param_const | llm_const | top1_nondegen | passed |
|---|---:|---:|---:|---:|---:|---|
| `mean_user_utility` | 12 | 0.80 | 0 | 0 | 11 / 12 | **PASS** |
| `overflow_rate_slothall` | 2 | 0.74 | 10 | 5 | 2 / 2 | PASS, but only 2 / 12 non-degenerate LHS rows |
| `mean_overload_excess` | 2 | 0.30 | 10 | 5 | 2 / 2 | FAIL |
| `hall_utilization_variance` | 12 | 0.40 | 0 | 0 | 11 / 12 | FAIL |

**Overall median Spearman ρ = 0.5536 ≥ 0.5 → PASS** (Q-O7).

`n_LHS_in_ρ` — критическое уточнение: для overload и overflow_rate из 12 maximin
LHS-row у параметрика 10 дают `overload = 0` у всех 4 политик (constant ranks),
Spearman undefined и LHS-row выпадает из ρ-summary. Top-1 match на degenerate LHS
автоматически совпадает через `argmin` tie-breaking — possible fake match;
поэтому колонка `top1_nondegen` фильтрует такие случаи.

Подробности и per-LHS — в `analysis_cross_validation.json`.

## 10. Combined interpretation

Overall acceptance Q-O7 пройдено (median ρ = 0.554 ≥ 0.5), но картина
**неоднородна по метрикам**:

- `mean_user_utility` — сильное согласование (median ρ ≈ 0.80 на 12 / 12 LHS,
  top-1 match 11 / 12). Параметрик и LLM сходятся на том, что utility у политик
  почти равна.
- `overflow_rate_slothall` — формально PASS (median ρ ≈ 0.74), но **только на
  2 / 12 non-degenerate LHS**. Узкая выборка, слабая статистическая поддержка.
- `mean_overload_excess` — FAIL (median ρ ≈ 0.30 на 2 / 12). Те же 10 параметрических
  safe-сценариев выпадают; на 2 точках статистически малорепрезентативно.
- `hall_utilization_variance` — FAIL (median ρ ≈ 0.40 на 12 / 12). Непрерывная
  мера, нет ties, но согласование умеренно слабое.

Корректная формулировка: **LLM cross-validation частично подтверждает выводы
параметрического симулятора** (utility и узко overflow_rate). Для overload-семейства
ρ опирается на 2 точки из 12 — это **не валидация, а диагностика** структурно
дегенерированной выборки (74 % LHS на mobius — safe сценарии). LLM-агент на
`gpt-5.4-nano` — **бюджетная модель**, поэтому V — это **budget cross-validation**,
не сильная поведенческая валидация дорогой моделью.

Это не отказ от PROJECT_DESIGN §7 («второй независимый источник отклика»);
это правдивая картина согласования с честными ограничениями.

## 11. Limitations

1. **П4 (`llm_ranker`) только на 12 maximin** — нельзя сравнивать средние П4 (n=36
   в Q или n=12 в V) со средними П1–П3 на full-50 (n=150 в Q).
2. **`program_variant` (Φ)** — sign-test и delta-vs-P0 — diagnostic only, не causal:
   LHS-row с разными PV не conf-matched по остальным осям.
3. **Bucket-агрегаты** конфаундированы между осями; OAT scatter ≠ строгое causal
   cleavage. Sobol / Morris отвергнуты (Q-O5).
4. **Cross-validation Spearman** для overload и overflow_rate считается на 2 / 12
   non-degenerate LHS-row — статистически малорепрезентативно. Top-1 match на
   degenerate LHS возможен как trivial совпадение через argmin tie-breaking.
5. **Budget cross-validation**: V на `gpt-5.4-nano` — не сильная поведенческая
   валидация. Сильная валидация на `gpt-5.4-mini` стоила бы ~$37 (vs $11.55 у nano)
   при тех же 44 160 calls.
6. **Абсолютные числа overload / overflow / utility** — это сравнительная
   характеристика политик в рамках единой синтетической модели. Не прогноз реальной
   посещаемости конкретной конференции (PROJECT_DESIGN §13).
7. **Mobius — сценарный стресс-тест**, не предсказание истины. Median overload = 0
   в full-50 — структурное свойство (74 % LHS-точек безопасные), не отсутствие
   эффекта политик.
8. **Стоп-лист тезисов** PROJECT_STATUS §5 действует: B1 / accuracy@1 не используется,
   cross-domain Spearman не выдается за валидацию реальности, MMR / inter-slot chat
   не подаются как защищаемые методы. Не выдавать V cross-validation как
   «LLM полностью подтвердил параметрический симулятор».

## 12. Links

- U (parametric report): [`report_mobius_2025_autumn_parametric.md`](report_mobius_2025_autumn_parametric.md)
- Q: `lhs_parametric_mobius_2025_autumn_2026-05-08.{json,csv,md}`
- S analysis JSONs (8 файлов): `experiments/results/analysis_*.json`
- S markdown: `analysis_lhs_parametric_2026-05-08.md`
- S plots:
  - `plots/analysis_risk_utility_scatter.png`
  - `plots/analysis_gossip_bucket_bar.png`
  - `plots/analysis_ranking_heatmap_maximin.png`
- V: `llm_agents_lhs_subset_12pts.{json,csv,md}`,
  `analysis_cross_validation.{json,md (analysis_cross_validation_2026-05-08.md)}`
- V plot: `plots/cross_validation_rho_per_metric.png`
- Spike memos: `docs/spikes/spike_experiment_protocol.md` (O),
  `docs/spikes/spike_result_postprocessing.md` (R),
  `docs/spikes/spike_report_format.md` (T)
