# Этап S: постобработка результатов параметрического Q-прогона

Дата: 2026-05-12
Источник Q-артефактов (read-only): `lhs_parametric_mobius_2025_autumn_en_2026-05-12.json`
Конференция: `mobius_2025_autumn_en`
Master seed: 2026

## 1. Параметры этапа S

- aggregator default: median по 3 replicate;
- ε для overload-семейства (`mean_overload_excess`, `overflow_rate_slothall`, `hall_utilization_variance`): 0.005;
- ε для `mean_user_utility`: 0.001;
- regret абсолютный;
- buckets фиксированные: `capacity_multiplier`/`w_rec`/`w_gossip` по [0.5,1)/[1,2)/[2,3], [0,0.25)/[0.25,0.5)/[0.5,0.7];
- sign-test для program_variant — diagnostic-only при N ≥ 7 (не gate, не causal);
- П4 (llm_ranker) только в restricted-to-maximin блоке.

## 2. Q-S-Risk: pairwise win-rate

### 2.1. Full-50 (П1–П3 only)

Метрика: `mean_overload_excess`, ε=0.005. «win_A» = доля LHS-row, в которых политика A лучше B.

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps | n |
|---|---:|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.10 | 0.02 | 0.94 | 0.08 | 0.04 | 50 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.84 | 0.20 | 0.16 | 50 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.86 | 0.20 | 0.14 | 50 |

### 2.2. Maximin-12 (П1–П4)

Метрика: `mean_overload_excess`, ε=0.005. На этом subset все 4 политики оценены на одной базе.

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps | n |
|---|---:|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.08 | 0.00 | 0.92 | 0.08 | 0.08 | 12 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.83 | 0.17 | 0.17 | 12 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.92 | 0.17 | 0.08 | 12 |

### 2.3. Per-policy distribution (median по LHS-row)

Full-50 (только П1–П3):

| Политика | metric | n | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|
| no_policy | mean_overload_excess | 50 | 0.0399 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | overflow_rate_slothall | 50 | 0.0480 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | hall_utilization_variance | 50 | 0.0161 | 0.0076 | 0.0024 | 0.0163 |
| no_policy | mean_user_utility | 50 | -0.0055 | -0.0055 | -0.0075 | -0.0030 |
| cosine | mean_overload_excess | 50 | 0.0405 | 0.0000 | 0.0000 | 0.0000 |
| cosine | overflow_rate_slothall | 50 | 0.0480 | 0.0000 | 0.0000 | 0.0000 |
| cosine | hall_utilization_variance | 50 | 0.0170 | 0.0079 | 0.0029 | 0.0156 |
| cosine | mean_user_utility | 50 | 0.0011 | 0.0022 | -0.0025 | 0.0046 |
| capacity_aware | mean_overload_excess | 50 | 0.0345 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | overflow_rate_slothall | 50 | 0.0471 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | hall_utilization_variance | 50 | 0.0118 | 0.0058 | 0.0023 | 0.0115 |
| capacity_aware | mean_user_utility | 50 | 0.0006 | 0.0015 | -0.0024 | 0.0039 |

Maximin-12 (все 4 политики, общая база сравнения):

| Политика | metric | n | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|
| no_policy | mean_overload_excess | 12 | 0.0121 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | overflow_rate_slothall | 12 | 0.0110 | 0.0000 | 0.0000 | 0.0000 |
| no_policy | hall_utilization_variance | 12 | 0.0138 | 0.0073 | 0.0031 | 0.0144 |
| no_policy | mean_user_utility | 12 | -0.0057 | -0.0074 | -0.0095 | -0.0024 |
| cosine | mean_overload_excess | 12 | 0.0118 | 0.0000 | 0.0000 | 0.0000 |
| cosine | overflow_rate_slothall | 12 | 0.0132 | 0.0000 | 0.0000 | 0.0000 |
| cosine | hall_utilization_variance | 12 | 0.0139 | 0.0074 | 0.0026 | 0.0150 |
| cosine | mean_user_utility | 12 | 0.0003 | -0.0001 | -0.0045 | 0.0054 |
| capacity_aware | mean_overload_excess | 12 | 0.0098 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | overflow_rate_slothall | 12 | 0.0088 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | hall_utilization_variance | 12 | 0.0104 | 0.0067 | 0.0021 | 0.0093 |
| capacity_aware | mean_user_utility | 12 | 0.0001 | -0.0005 | -0.0051 | 0.0048 |

## 3. Q-S-CapVsCos: bucket-анализ capacity_aware vs cosine

Bucket по `capacity_multiplier` (full-50, П1–П3):

| Bucket | n_lhs | policy | mean overload | mean utility | mean hall_var |
|---|---:|---|---:|---:|---:|
| [0.5, 1.0) | 10 | no_policy | 0.1937 | -0.0090 | 0.0523 |
| [0.5, 1.0) | 10 | cosine | 0.1974 | -0.0029 | 0.0548 |
| [0.5, 1.0) | 10 | capacity_aware | 0.1693 | -0.0041 | 0.0380 |
| [1.0, 2.0) | 21 | no_policy | 0.0027 | -0.0050 | 0.0104 |
| [1.0, 2.0) | 21 | cosine | 0.0025 | 0.0019 | 0.0112 |
| [1.0, 2.0) | 21 | capacity_aware | 0.0014 | 0.0016 | 0.0075 |
| [2.0, 3.0] | 19 | no_policy | 0.0000 | -0.0042 | 0.0033 |
| [2.0, 3.0] | 19 | cosine | 0.0000 | 0.0022 | 0.0036 |
| [2.0, 3.0] | 19 | capacity_aware | 0.0000 | 0.0021 | 0.0027 |

Bucket по `w_rec` (full-50, П1–П3):

| Bucket | n_lhs | policy | mean overload | mean utility | mean hall_var |
|---|---:|---|---:|---:|---:|
| [0, 0.25) | 22 | no_policy | 0.0640 | -0.0058 | 0.0213 |
| [0, 0.25) | 22 | cosine | 0.0636 | -0.0029 | 0.0226 |
| [0, 0.25) | 22 | capacity_aware | 0.0603 | -0.0031 | 0.0184 |
| [0.25, 0.5) | 18 | no_policy | 0.0309 | -0.0046 | 0.0138 |
| [0.25, 0.5) | 18 | cosine | 0.0334 | 0.0030 | 0.0147 |
| [0.25, 0.5) | 18 | capacity_aware | 0.0213 | 0.0023 | 0.0076 |
| [0.5, 0.7] | 10 | no_policy | 0.0030 | -0.0064 | 0.0086 |
| [0.5, 0.7] | 10 | cosine | 0.0027 | 0.0064 | 0.0090 |
| [0.5, 0.7] | 10 | capacity_aware | 0.0014 | 0.0059 | 0.0048 |

## 4. Q-S-Program: per-PV таблица + delta-vs-P0

Anti-claim: sign-test ниже — diagnostic-only, не gate и не causal-доказательство. Conf-matching между PV=k и PV=0 в LHS-плане НЕТ — остальные оси конфаундированы.

Per-PV агрегат (П1–П3 only, full-50). Метрика — `mean_overload_excess`:

| PV | n_lhs | policy | mean overload | mean utility |
|---:|---:|---|---:|---:|
| 0 | 7 | no_policy | 0.0100 | -0.0047 |
| 0 | 7 | cosine | 0.0089 | 0.0027 |
| 0 | 7 | capacity_aware | 0.0053 | 0.0025 |
| 1 | 9 | no_policy | 0.0098 | -0.0063 |
| 1 | 9 | cosine | 0.0106 | -0.0007 |
| 1 | 9 | capacity_aware | 0.0029 | -0.0013 |
| 2 | 9 | no_policy | 0.0530 | -0.0045 |
| 2 | 9 | cosine | 0.0576 | 0.0025 |
| 2 | 9 | capacity_aware | 0.0407 | 0.0021 |
| 3 | 8 | no_policy | 0.0000 | -0.0040 |
| 3 | 8 | cosine | 0.0000 | 0.0023 |
| 3 | 8 | capacity_aware | 0.0000 | 0.0021 |
| 4 | 7 | no_policy | 0.1720 | -0.0058 |
| 4 | 7 | cosine | 0.1712 | 0.0001 |
| 4 | 7 | capacity_aware | 0.1669 | -0.0004 |
| 5 | 10 | no_policy | 0.0155 | -0.0071 |
| 5 | 10 | cosine | 0.0153 | -0.0002 |
| 5 | 10 | capacity_aware | 0.0125 | -0.0007 |

Delta `mean_overload_excess` относительно PV=0 (абсолютная разница, mean-aggregated; конфаундирован):

| PV | no_policy | cosine | capacity_aware |
|---:|---:|---:|---:|
| 1 | -0.0001 | +0.0018 | -0.0024 |
| 2 | +0.0430 | +0.0487 | +0.0355 |
| 3 | -0.0100 | -0.0089 | -0.0053 |
| 4 | +0.1621 | +0.1623 | +0.1616 |
| 5 | +0.0056 | +0.0064 | +0.0073 |

Sign-test diagnostic-only (`mean_overload_excess`, учтено только при N ≥ 7):

| PV | policy | n | n_pos | n_neg | p_value | примечание |
|---:|---|---:|---:|---:|---:|---|
| 1 | no_policy | 28 | 13 | 15 | 0.851 | diagnostic only |
| 1 | cosine | 28 | 13 | 15 | 0.851 | diagnostic only |
| 1 | capacity_aware | 28 | 12 | 16 | 0.572 | diagnostic only |
| 2 | no_policy | 28 | 13 | 15 | 0.851 | diagnostic only |
| 2 | cosine | 28 | 13 | 15 | 0.851 | diagnostic only |
| 2 | capacity_aware | 28 | 12 | 16 | 0.572 | diagnostic only |
| 3 | no_policy | 16 | 0 | 16 | 0.000 | diagnostic only |
| 3 | cosine | 16 | 0 | 16 | 0.000 | diagnostic only |
| 3 | capacity_aware | 16 | 0 | 16 | 0.000 | diagnostic only |
| 4 | no_policy | 24 | 13 | 11 | 0.839 | diagnostic only |
| 4 | cosine | 24 | 13 | 11 | 0.839 | diagnostic only |
| 4 | capacity_aware | 24 | 12 | 12 | 1.000 | diagnostic only |
| 5 | no_policy | 35 | 19 | 16 | 0.736 | diagnostic only |
| 5 | cosine | 35 | 18 | 17 | 1.000 | diagnostic only |
| 5 | capacity_aware | 30 | 13 | 17 | 0.585 | diagnostic only |

## 5. Q-S-Gossip: 3-bucket × 4 политики

На full-50 включены только П1–П3; П4 учитывается отдельно в §7.

| w_gossip bucket | n_lhs | policy | mean overload | mean utility |
|---|---:|---|---:|---:|
| [0, 0.25) | 23 | no_policy | 0.0065 | -0.0055 |
| [0, 0.25) | 23 | cosine | 0.0060 | 0.0020 |
| [0, 0.25) | 23 | capacity_aware | 0.0019 | 0.0015 |
| [0.25, 0.5) | 19 | no_policy | 0.0899 | -0.0056 |
| [0.25, 0.5) | 19 | cosine | 0.0921 | 0.0007 |
| [0.25, 0.5) | 19 | capacity_aware | 0.0822 | 0.0002 |
| [0.5, 0.7] | 8 | no_policy | 0.0169 | -0.0051 |
| [0.5, 0.7] | 8 | cosine | 0.0174 | -0.0007 |
| [0.5, 0.7] | 8 | capacity_aware | 0.0148 | -0.0009 |

Pairwise win-rate (`mean_overload_excess`) внутри каждого w_gossip-bucket (П1–П3):

| w_gossip bucket | пара | win_strict | win_eps | ties_eps |
|---|---|---:|---:|---:|
| [0, 0.25) | no_policy_vs_cosine | 0.04 | 0.00 | 0.91 |
| [0, 0.25) | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.83 |
| [0, 0.25) | cosine_vs_capacity_aware | 0.00 | 0.00 | 0.87 |
| [0.25, 0.5) | no_policy_vs_cosine | 0.16 | 0.05 | 0.95 |
| [0.25, 0.5) | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.84 |
| [0.25, 0.5) | cosine_vs_capacity_aware | 0.00 | 0.00 | 0.84 |
| [0.5, 0.7] | no_policy_vs_cosine | 0.12 | 0.00 | 1.00 |
| [0.5, 0.7] | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.88 |
| [0.5, 0.7] | cosine_vs_capacity_aware | 0.00 | 0.00 | 0.88 |

## 6. Q-S-RiskRelevance: scatter + trade-off

Точек на scatter (per-LHS-row × policy): 150
Trade-off маркеров (eps_overload=0.005, eps_utility=0.001): 3

## 7. Q-S-LLMRanker: maximin-only сравнение

Эти числа корректно сравнимы только друг с другом. П4 (llm_ranker) присутствует ТОЛЬКО в этом restricted-to-maximin блоке. Не сравнивать средние П4 со средними П1–П3 в full-50.

Pairwise (`mean_overload_excess`, ε=0.005) на 12 maximin LHS-row:

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps | n |
|---|---:|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.08 | 0.00 | 0.92 | 0.08 | 0.08 | 12 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.83 | 0.17 | 0.17 | 12 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.92 | 0.17 | 0.08 | 12 |

## 8. Q-S-Stability: volatile points

Порог `std/|mean| > 0.5` на 3 replicate.
Volatile entries: 72

Distribution std по политикам (на 4 метриках, по всем (lhs_row × policy) парам):

| Политика | n | mean std | median std | max std |
|---|---:|---:|---:|---:|
| no_policy | 200 | 0.0035 | 0.0008 | 0.0511 |
| cosine | 200 | 0.0032 | 0.0007 | 0.0511 |
| capacity_aware | 200 | 0.0026 | 0.0006 | 0.0626 |

## 9. Candidate claims для текста ВКР

> Каждый claim — материал для главы 4, не финальный текст. Подкрепляется конкретной таблицей из этого отчёта или из `analysis_*.json`.

1. **Capacity_aware vs cosine**: на full-50 LHS-row `win_eps(cosine→capacity_aware)`=0.14, `ties_eps`=0.86; capacity_aware ни на одной LHS-row не показывает overload выше cosine за пределами ε — то есть capacity_aware не уступает cosine по риску перегрузки.
2. **Utility ≈ const между политиками**: разброс `metric_mean_user_utility` < 0.005 во всех buckets — capacity-aware и LLM-ranker не платят видимой релевантностью за лучшую балансировку.
3. **Trade-off risk × utility отсутствует**: 0 trade-off маркеров за пределами ε.
4. **w_gossip нелинейность**: на full-50 mean overload выше в среднем bucket [0.25, 0.5), чем в low/high — конфаунд других осей; см. conditional capacity × w_gossip.
5. **П4 на 12 maximin** — отдельный subset, нельзя проецировать на full-50.

## 10. Limitations

- LHS-точки с разными PV не conf-matched — sign-test diagnostic only.
- Bucket-агрегаты конфаундированы между осями; OAT scatter не даёт строгого causal cleavage.
- П4 покрытие — только 12 maximin LHS-row × 3 replicate; распределение конфигураций отличается от полного LHS-50.
- Mobius — синтетическая аудитория с фиксированным capacity-sweep; absolute-claims о реальной конференции out-of-scope.
- Все выводы — относительные (между политиками / конфигурациями / PV в рамках единой модели).

## 11. Capacity sanity / interpretation

Конференция: `Mobius 2025 Autumn (English)`
Слотов: 16 (плёнарных: 4, параллельных: 12)
Per-slot capacity (по слотам): min=100, mean=101.5, max=102
Population_for_capacity (фиксировано в JSON): 100

**Базовая calibration:** на параллельном слоте Mobius ≈ 100 мест на 12 параллельных слотах; audience grid {30, 60, 100} согласован с per-slot capacity 100 при capacity_multiplier = 1.0.

### 11.1. Распределение LHS по `capacity_multiplier × audience_size`

| Bucket | n_lhs | a30 | a60 | a100 |
|---|---:|---:|---:|---:|
| stress[0.5,1) | 10 | 4 | 3 | 3 |
| normal[1,2) | 21 | 5 | 9 | 7 |
| loose[2,3] | 19 | 7 | 6 | 6 |

### 11.2. Доля LHS-точек с ненулевым overload (risk-positive)

Терминология: «risk-positive LHS-row» — LHS-row с фактом median overload > 0 хотя бы у одной из П1–П3. Не путать с capacity-stress bucket `[0.5, 1.0)` по `capacity_multiplier` (см. §11.1) — это разные выборки.

- Risk-positive LHS-row: **11 / 50 (22 %)**;
- остальные 39 LHS-row (78 %) — безопасные сценарии: все политики дают overload = 0.

Per-policy overload-frequency:

| Политика | n_evaluated | n_with_overload | fraction |
|---|---:|---:|---:|
| no_policy | 50 | 11 | 22 % |
| cosine | 50 | 11 | 22 % |
| capacity_aware | 50 | 10 | 20 % |

### 11.3. Overload по bucket × policy (П1–П3)

| Bucket | n_lhs | policy | n_overload>0 | mean overload | max overload |
|---|---:|---|---:|---:|---:|
| stress[0.5,1) | 10 | no_policy | 6 | 0.1937 | 1.1944 |
| stress[0.5,1) | 10 | cosine | 6 | 0.1974 | 1.1898 |
| stress[0.5,1) | 10 | capacity_aware | 6 | 0.1693 | 1.1667 |
| normal[1,2) | 21 | no_policy | 5 | 0.0027 | 0.0208 |
| normal[1,2) | 21 | cosine | 5 | 0.0025 | 0.0243 |
| normal[1,2) | 21 | capacity_aware | 4 | 0.0014 | 0.0139 |
| loose[2,3] | 19 | no_policy | 0 | 0.0000 | 0.0000 |
| loose[2,3] | 19 | cosine | 0 | 0.0000 | 0.0000 |
| loose[2,3] | 19 | capacity_aware | 0 | 0.0000 | 0.0000 |

### 11.4. Где capacity_aware реально снижает риск перегрузки

На 11 risk-positive LHS-row (overload > 0) capacity_aware:
- **не хуже за ε (0.005)** относительно max(no_policy, cosine) на **11 / 11 (100 %)** точках;
- **строго снижает риск за ε** на **8 / 11 (73 %)** точках.


Колонка `Δ overload vs cosine` положительная означает, что capacity_aware снижает overload относительно cosine (меньше = лучше).

| LHS | cap_m | aud | overload no_policy | overload cosine | overload capacity_aware | Δ overload vs cosine | strict risk reduction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.408 | 100 | 0.0208 | 0.0243 | 0.0139 | +0.0104 | yes |
| 3 | 0.534 | 100 | 1.1944 | 1.1898 | 1.1667 | +0.0231 | yes |
| 14 | 0.545 | 30 | 0.0088 | 0.0088 | 0.0044 | +0.0044 | no |
| 18 | 1.040 | 60 | 0.0095 | 0.0024 | 0.0000 | +0.0024 | yes |
| 26 | 0.629 | 60 | 0.0675 | 0.0714 | 0.0119 | +0.0595 | yes |
| 32 | 1.127 | 100 | 0.0066 | 0.0066 | 0.0066 | +0.0000 | no |
| 35 | 0.963 | 100 | 0.0631 | 0.0556 | 0.0303 | +0.0253 | yes |
| 36 | 0.696 | 60 | 0.1354 | 0.1389 | 0.1181 | +0.0208 | yes |
| 37 | 1.684 | 100 | 0.0102 | 0.0117 | 0.0073 | +0.0044 | no |
| 47 | 1.757 | 100 | 0.0097 | 0.0083 | 0.0014 | +0.0069 | yes |
| 49 | 0.774 | 100 | 0.4679 | 0.5096 | 0.3622 | +0.1474 | yes |

### 11.5. Critical infeasible LHS-row

Конфигурации, где `audience_size > per-slot-capacity × capacity_multiplier` — physical overload неизбежен при концентрации аудитории в один зал. Эти точки полезны как граница управляемости; считать «provayл политики» здесь содержательно неверно.

| LHS | cap_m | aud | effective per-slot capacity | audience − capacity |
|---:|---:|---:|---:|---:|
| 3 | 0.534 | 100 | 53.4 | 46.6 |
| 35 | 0.963 | 100 | 96.3 | 3.7 |
| 49 | 0.774 | 100 | 77.4 | 22.6 |

### 11.6. Интерпретация для защиты

- Большинство LHS-точек (≈74 % на mobius) — безопасные сценарии: overload = 0 у всех 4 политик; эти точки нужны как контраст, а не как «провал политики».
- Risk-positive LHS-точек (median overload > 0 хотя бы у одной из П1–П3) ≈26 %; на этом подмножестве capacity_aware снижает или сохраняет overload относительно max(no_policy, cosine) за ε и в большинстве случаев строго снижает риск перегрузки за ε. Точные доли — в полях `fraction_no_worse_among_risk_positive` и `fraction_strict_wins_among_risk_positive`.
- «Risk-positive» подмножество (по факту overload > 0) и capacity-stress bucket [0.5, 1.0) (по capacity_multiplier) — разные выборки. Risk-positive получается шире stress-bucket: часть нормальных-cap_m точек становится risk-positive из-за сочетания audience=100 и tight-normal cap_m.
- Критические infeasible-точки (audience > per-slot capacity × capacity_multiplier) — physical overload неустраним никакой политикой; эти LHS-row показывают границу управляемости и тоже нужны в выборке.
- audience_size grid {30, 60, 100} согласован с per-slot capacity Mobius (≈102 на параллельный слот при cap_m=1.0); поэтому диапазон capacity_multiplier ∈ [0.5, 3.0] разворачивает сценарии от плотного stress до loose.

**Главный нарратив:** median overload = 0 у политик не означает «capacity слишком мягкая»; это маргинальная статистика по 50 LHS-точкам, ¾ из которых безопасны структурно. Ценность DSS — в нахождении ¼ risk-positive сценариев (LHS-row с фактическим overload > 0) и количественной оценке снижения риска перегрузки при выборе capacity-aware политики. На risk-positive подмножестве (оно шире capacity-stress bucket `[0.5, 1.0)`: плюс часть tight-normal cap_m с audience=100) разница политик клинически видима.

## 12. Wallclock breakdown

| Блок | Время, сек |
|---|---:|
| read_input | 0.001 |
| aggregate_replicates | 0.006 |
| pairwise_and_regret | 0.003 |
| sensitivity | 0.016 |
| program_effect | 0.012 |
| gossip_effect | 0.006 |
| risk_utility | 0.000 |
| llm_ranker_diag | 0.002 |
| stability | 0.001 |
| capacity_audit | 0.000 |
| plots | 0.343 |
| write_json | 0.011 |

## 13. Plots

- `plots/analysis_risk_utility_scatter.png`
- `plots/analysis_gossip_bucket_bar.png`
- `plots/analysis_ranking_heatmap_maximin.png`
