# Parametric experiment report — Mobius 2025 Autumn

Compact technical artefact (U-stage). Не текст диплома; только цифры,
таблицы, acceptance, ключевые выводы и ограничения. Источники — закоммиченные
`analysis_*.json` и `lhs_parametric_*.{json,md}`.

## 1. Metadata

| Поле | Значение |
|---|---|
| Конференция | `mobius_2025_autumn` |
| Дата отчёта | 2026-05-08 |
| Q artefact | `lhs_parametric_mobius_2025_autumn_2026-05-08.{json,csv,md}` |
| Q commit | `b4e0787` (full LHS run) + `9ac6348` (diagnostic summaries) |
| S artefacts | `analysis_pairwise.json`, `analysis_sensitivity.json`, `analysis_program_effect.json`, `analysis_gossip_effect.json`, `analysis_risk_utility.json`, `analysis_llm_ranker_diagnostic.json`, `analysis_stability.json`, `analysis_capacity_audit.json`, `analysis_lhs_parametric_2026-05-08.md` |
| S commit | `def0887` |
| master_seed | 2026 |
| n_points (LHS) | 50 |
| replicates | 3 |
| K (top-K) | 3 |
| Политики | П1=`no_policy`, П2=`cosine`, П3=`capacity_aware`, П4=`llm_ranker` (только на 12 maximin) |
| Evals | 486 (П1=150, П2=150, П3=150, П4=36) |
| Maximin indices | `[6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]` |

## 2. Acceptance

**Q (полный параметрический LHS-прогон) — PASS**, все 7 чеков:

| Чек | Значение | Статус |
|---|---|---|
| П1–П3 evals == ожидаемое | 450 == 450 | PASS |
| П4 evals == ожидаемое | 36 == 36 | PASS |
| total evals == ожидаемое | 486 == 486 | PASS |
| П4 только на maximin | violations=0 | PASS |
| CRN audience/phi инвариант | violations=0 | PASS |
| cfg_seed = replicate | violations=0 | PASS |
| long-format ключи | missing=[] | PASS |

**S (post-processing) — PASS**, все 8 чеков (см. `spike_result_postprocessing.md` §8):

- analyze_lhs.py не мутирует Q (sha256 идентичен);
- 8 `analysis_*.json` + markdown + 3 plot созданы;
- П4 отсутствует в full-50 таблицах;
- pairwise win_eps + ties_eps + loss_eps ≈ 1 для всех пар;
- regret лучшей политики = 0 на каждой LHS-row;
- markdown сборка без exception;
- pytest 117 passed (на момент S; сейчас 145 после V);
- wallclock S = 0.32 s.

**Q wallclock** (из `lhs_parametric_*.md`): П1–П3 evals 6.00 s; П4 (llm_ranker) 2455.10 s; полный 2465.96 s.

Q artefacts — read-only от этапа S и далее.

## 3. Per-policy distribution (full-50, П1–П3)

| Policy | overload mean | overload median | overload p75 | utility mean | utility median |
|---|---:|---:|---:|---:|---:|
| no_policy | 0.0395 | 0.0000 | 0.0000 | 0.7308 | 0.7318 |
| cosine | 0.0449 | 0.0000 | 0.0066 | 0.7321 | 0.7331 |
| capacity_aware | 0.0336 | 0.0000 | 0.0000 | 0.7314 | 0.7323 |

Utility разделяет политики в пределах < 0.002. Полный набор статистик +
`overflow_rate` / `hall_var` — в `analysis_pairwise.json` `distribution_full_50`.

## 4. Pairwise comparison

### 4.1. Full-50 (П1–П3 only), `mean_overload_excess`, ε=0.005

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps |
|---|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.20 | 0.14 | 0.86 | 0.04 | 0.00 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.86 | 0.20 | 0.14 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.78 | 0.26 | 0.22 |

`capacity_aware` ни на одной из 50 LHS-row не показывает overload выше cosine за ε
(`win_eps(cosine→capacity_aware)=0.00`).

### 4.2. Maximin-12, `mean_overload_excess` (ключевые пары)

| Пара (A vs B) | win_strict | ties_eps | loss_eps |
|---|---:|---:|---:|
| capacity_aware_vs_llm_ranker | 0.17 | 0.92 | 0.00 |
| no_policy_vs_capacity_aware | 0.00 | 0.83 | 0.17 |
| cosine_vs_capacity_aware | 0.00 | 0.83 | 0.17 |

Все 6 пар × 4 метрики — в `analysis_pairwise.json` `maximin_12`.

## 5. Capacity sanity

Mobius: 16 слотов (12 параллельных по 3 зала × 34 ≈ 102 места на слот; 4 пленарных
по 1 залу × 100). `audience_size` grid {30, 60, 100} согласован с per-slot capacity 102
при `capacity_multiplier = 1.0`.

LHS distribution × audience:

| Bucket по cap_multiplier | n_lhs | a30 | a60 | a100 |
|---|---:|---:|---:|---:|
| stress [0.5, 1.0) | 10 | 4 | 3 | 3 |
| normal [1.0, 2.0) | 21 | 5 | 9 | 7 |
| loose [2.0, 3.0]  | 19 | 7 | 6 | 6 |

**Risk-positive vs safe**:

- 13 / 50 LHS-row (26 %) — overload > 0 хотя бы у одной из П1–П3 (risk-positive);
- 37 / 50 (74 %) — все политики дают overload = 0 (безопасные сценарии);
- bucket `loose [2.0, 3.0]`: 0 / 19 overload-событий (как и должно быть).

**Risk-positive ⊃ stress[0.5,1.0)**: «risk-positive» (по факту overload > 0) шире
capacity-stress bucket — туда же попадают tight-normal LHS с audience=100. Подробности
в `analysis_capacity_audit.json` поле `subset_definition`.

**Critical infeasible LHS-row (3)**: `audience > per-slot capacity × cap_multiplier`,
physical overload неустраним:

| LHS | cap_m | aud | effective per-slot capacity | overshoot |
|---:|---:|---:|---:|---:|
| 3 | 0.534 | 100 | 53.4 | +46.6 |
| 35 | 0.963 | 100 | 96.3 | +3.7 |
| 49 | 0.774 | 100 | 77.4 | +22.6 |

Median overload = 0 в full-50 — это структурное свойство сценарного анализа
(74 % безопасны), не «эффекта нет». Главная ценность DSS — в risk-positive
подмножестве.

## 6. Capacity_aware effect on risk-positive subset

На 13 risk-positive LHS-row (overload > 0):

- **`capacity_aware` не хуже за ε (0.005)** относительно `max(no_policy, cosine)`:
  **13 / 13 (100 %)**;
- **строго снижает риск перегрузки за ε**: **11 / 13 (85 %)**.

Top-6 reductions (Δ overload vs cosine):

| LHS | cap_m | aud | overload no_policy | overload cosine | overload capacity_aware | Δ vs cosine | strict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 49 | 0.774 | 100 | 0.4647 | 0.5449 | 0.3558 | +0.1891 | yes |
| 26 | 0.629 |  60 | 0.0714 | 0.1706 | 0.0040 | +0.1667 | yes |
| 35 | 0.963 | 100 | 0.0505 | 0.0606 | 0.0025 | +0.0581 | yes |
|  3 | 0.534 | 100 | 1.1806 | 1.2222 | 1.1759 | +0.0463 | yes |
| 36 | 0.696 |  60 | 0.1424 | 0.1424 | 0.1111 | +0.0312 | yes |
| 18 | 1.040 |  60 | 0.0095 | 0.0214 | 0.0000 | +0.0214 | yes |

Полная таблица 13 строк — в `analysis_capacity_audit.json` `capacity_aware_effect.per_lhs_breakdown`.

## 7. program_variant Φ — diagnostic only

Per-PV agg на full-50 П1–П3, `mean_overload_excess` (cosine view):

| PV | n_lhs | mean overload |
|---:|---:|---:|
| 0 (P_0) | 7 | 0.0099 |
| 1 | 9 | 0.0211 |
| 2 | 9 | 0.0633 |
| 3 | 8 | 0.0000 |
| 4 | 7 | 0.1762 |
| 5 | 10 | 0.0183 |

**Diagnostic only — no conf-matching**: LHS-row с PV=k и PV=0 имеют разные значения
остальных осей; sign-test на cross-product помечен `interpretation: "diagnostic only"`
во всех ячейках `analysis_program_effect.json.sign_test_diagnostic_only`. Принятая
формулировка из `spike_result_postprocessing.md` Q-R4 — sign-test не gate, не causal.

## 8. Gossip — 3-bucket × policy

`mean_overload_excess` (full-50, П1–П3):

| w_gossip bucket | n_lhs | no_policy | cosine | capacity_aware |
|---|---:|---:|---:|---:|
| [0, 0.25) low | 23 | 0.0061 | 0.0123 | 0.0003 |
| [0.25, 0.5) mid | 19 | 0.0891 | 0.0957 | 0.0822 |
| [0.5, 0.7] high | 8 | 0.0178 | 0.0178 | 0.0139 |

Нелинейный пик в среднем bucket `[0.25, 0.5)`. `capacity_aware` лучший во всех трёх
bucket. Conditional `capacity × w_gossip` — в `analysis_gossip_effect.json`.

## 9. Risk × utility

| Subset | n_points | tradeoff_markers (ε_o=0.005, ε_u=0.001) |
|---|---:|---:|
| full-50 П1–П3 | 150 | 11 |
| maximin-12 (все 4 политики) | 48 | 3 |

Trade-off (политика A лучше по overload и одновременно хуже по utility за пределами ε)
на mobius — редкое событие. Utility у политик практически идентична. Подробности
и per-LHS — в `analysis_risk_utility.json`.

## 10. Stability / volatile points

Threshold `std / |mean| > 0.5` по 3 replicate per (lhs_row, policy, metric):

| Subset | n_volatile entries |
|---:|---:|
| full-50 П1–П3 | 24 |
| maximin-12 П4 only | 4 |

Список volatile + std distribution — в `analysis_stability.json`.

## 11. Candidate claims

1. На full-50 LHS `cosine_vs_capacity_aware`: `loss_eps = 0.22`, `ties_eps = 0.78`,
   `win_eps = 0` — `capacity_aware` ни на одной из 50 LHS-row не показывает overload
   выше cosine за ε.
2. `capacity_aware` mean overload `0.0336` против cosine `0.0449` (≈ −25 % относительно)
   на full-50; median у обеих 0.
3. На risk-positive подмножестве (13 LHS-row из 50) `capacity_aware` не хуже за ε
   на 13 / 13 точек, строго снижает риск за ε на 11 / 13 (85 %).
4. Trade-off risk × utility проявляется редко: 11 маркеров из 150 full-50 точек.
   Utility между политиками почти не меняется (разброс < 0.002 во всех buckets),
   поэтому улучшение балансировки у `capacity_aware` не сопровождается заметной
   потерей релевантности.
5. На 12 maximin: ordering по mean overload `capacity_aware (0.0093) ≤ llm_ranker
   (0.0109) ≤ no_policy (0.0127) ≈ cosine (0.0136)`.
6. `w_gossip` нелинеен по 3-bucket: `mid [0.25, 0.5)` mean overload ≈ 0.089 — явный
   пик; в low / high bucket — 0.006 / 0.018.
7. PV-эффект Φ → diagnostic only: mean overload PV=4 = 0.176 vs PV=0 = 0.010, но
   sign-test p-value ≈ 0.84–0.85; LHS-точки не conf-matched по остальным осям.

## 12. Limitations

1. П4 (`llm_ranker`) присутствует только на 12 maximin LHS-row; нельзя сравнивать
   средние П4 (n=36) со средними П1–П3 (n=150). Pairwise по П4 — только в restricted
   maximin-12 блоке.
2. `program_variant` (Φ) — все sign-test и delta-vs-P0 — diagnostic only, не causal:
   LHS-row с разными PV не conf-matched по остальным осям.
3. Bucket-агрегаты (`capacity_multiplier`, `w_rec`, `w_gossip`) конфаундированы между
   осями; OAT scatter ≠ строгое causal cleavage. Sobol / Morris отвергнуты (Q-O5).
4. Абсолютные числа overload / overflow / utility — это сравнительная характеристика
   политик в рамках единой синтетической модели. Не прогноз реальной посещаемости
   конкретной конференции (PROJECT_DESIGN §13).
5. Mobius — сценарный стресс-тест, не предсказание истины. Median overload = 0 в
   full-50 — структурное свойство (74 % LHS-точек безопасные), не отсутствие эффекта.
6. Стоп-лист тезисов из PROJECT_STATUS §5 действует: B1 / accuracy@1 не используется,
   cross-domain Spearman не выдается за валидацию реальности, MMR / inter-slot chat
   не подаются как защищаемые методы.

## 13. Links

- Q: `experiments/results/lhs_parametric_mobius_2025_autumn_2026-05-08.{json,csv,md}`
- S analysis JSONs (8 файлов): `experiments/results/analysis_*.json`
- S markdown: `experiments/results/analysis_lhs_parametric_2026-05-08.md`
- S plots:
  - `experiments/results/plots/analysis_risk_utility_scatter.png`
  - `experiments/results/plots/analysis_gossip_bucket_bar.png`
  - `experiments/results/plots/analysis_ranking_heatmap_maximin.png`
- Spike memos: `docs/spikes/spike_experiment_protocol.md` (O), `docs/spikes/spike_result_postprocessing.md` (R), `docs/spikes/spike_report_format.md` (T)
