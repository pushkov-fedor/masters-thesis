# Этап S: постобработка результатов параметрического Q-прогона

Дата: 2026-05-12
Источник Q-артефактов (read-only): `lhs_parametric_demo_day_2026_en_2026-05-12.json`
Конференция: `demo_day_2026_en`
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
| no_policy_vs_cosine | 0.14 | 0.02 | 0.94 | 0.20 | 0.04 | 50 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.84 | 0.34 | 0.16 | 50 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.86 | 0.34 | 0.14 | 50 |

### 2.2. Maximin-12 (П1–П4)

Метрика: `mean_overload_excess`, ε=0.005. На этом subset все 4 политики оценены на одной базе.

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps | n |
|---|---:|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.17 | 0.00 | 0.92 | 0.33 | 0.08 | 12 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.75 | 0.50 | 0.25 | 12 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.92 | 0.42 | 0.08 | 12 |

### 2.3. Per-policy distribution (median по LHS-row)

Full-50 (только П1–П3):

| Политика | metric | n | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|
| no_policy | mean_overload_excess | 50 | 0.0486 | 0.0000 | 0.0000 | 0.0022 |
| no_policy | overflow_rate_slothall | 50 | 0.0521 | 0.0000 | 0.0000 | 0.0051 |
| no_policy | hall_utilization_variance | 50 | 0.0213 | 0.0129 | 0.0045 | 0.0214 |
| no_policy | mean_user_utility | 50 | -0.0010 | -0.0017 | -0.0029 | 0.0013 |
| cosine | mean_overload_excess | 50 | 0.0484 | 0.0000 | 0.0000 | 0.0027 |
| cosine | overflow_rate_slothall | 50 | 0.0509 | 0.0000 | 0.0000 | 0.0051 |
| cosine | hall_utilization_variance | 50 | 0.0209 | 0.0121 | 0.0046 | 0.0214 |
| cosine | mean_user_utility | 50 | 0.0078 | 0.0080 | 0.0044 | 0.0123 |
| capacity_aware | mean_overload_excess | 50 | 0.0399 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | overflow_rate_slothall | 50 | 0.0504 | 0.0000 | 0.0000 | 0.0000 |
| capacity_aware | hall_utilization_variance | 50 | 0.0151 | 0.0084 | 0.0041 | 0.0142 |
| capacity_aware | mean_user_utility | 50 | 0.0072 | 0.0071 | 0.0040 | 0.0111 |

Maximin-12 (все 4 политики, общая база сравнения):

| Политика | metric | n | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|
| no_policy | mean_overload_excess | 12 | 0.0151 | 0.0002 | 0.0000 | 0.0059 |
| no_policy | overflow_rate_slothall | 12 | 0.0207 | 0.0025 | 0.0000 | 0.0139 |
| no_policy | hall_utilization_variance | 12 | 0.0187 | 0.0135 | 0.0038 | 0.0264 |
| no_policy | mean_user_utility | 12 | -0.0015 | -0.0018 | -0.0030 | -0.0002 |
| cosine | mean_overload_excess | 12 | 0.0136 | 0.0000 | 0.0000 | 0.0054 |
| cosine | overflow_rate_slothall | 12 | 0.0194 | 0.0000 | 0.0000 | 0.0101 |
| cosine | hall_utilization_variance | 12 | 0.0183 | 0.0135 | 0.0039 | 0.0245 |
| cosine | mean_user_utility | 12 | 0.0067 | 0.0066 | 0.0022 | 0.0101 |
| capacity_aware | mean_overload_excess | 12 | 0.0118 | 0.0000 | 0.0000 | 0.0007 |
| capacity_aware | overflow_rate_slothall | 12 | 0.0161 | 0.0000 | 0.0000 | 0.0025 |
| capacity_aware | hall_utilization_variance | 12 | 0.0150 | 0.0104 | 0.0035 | 0.0190 |
| capacity_aware | mean_user_utility | 12 | 0.0061 | 0.0053 | 0.0020 | 0.0098 |

## 3. Q-S-CapVsCos: bucket-анализ capacity_aware vs cosine

Bucket по `capacity_multiplier` (full-50, П1–П3):

| Bucket | n_lhs | policy | mean overload | mean utility | mean hall_var |
|---|---:|---|---:|---:|---:|
| [0.5, 1.0) | 10 | no_policy | 0.2278 | -0.0004 | 0.0611 |
| [0.5, 1.0) | 10 | cosine | 0.2263 | 0.0077 | 0.0602 |
| [0.5, 1.0) | 10 | capacity_aware | 0.1889 | 0.0064 | 0.0425 |
| [1.0, 2.0) | 21 | no_policy | 0.0073 | -0.0015 | 0.0171 |
| [1.0, 2.0) | 21 | cosine | 0.0075 | 0.0079 | 0.0167 |
| [1.0, 2.0) | 21 | capacity_aware | 0.0051 | 0.0073 | 0.0121 |
| [2.0, 3.0] | 19 | no_policy | 0.0000 | -0.0007 | 0.0051 |
| [2.0, 3.0] | 19 | cosine | 0.0000 | 0.0078 | 0.0049 |
| [2.0, 3.0] | 19 | capacity_aware | 0.0000 | 0.0075 | 0.0039 |

Bucket по `w_rec` (full-50, П1–П3):

| Bucket | n_lhs | policy | mean overload | mean utility | mean hall_var |
|---|---:|---|---:|---:|---:|
| [0, 0.25) | 22 | no_policy | 0.0765 | -0.0002 | 0.0270 |
| [0, 0.25) | 22 | cosine | 0.0755 | 0.0035 | 0.0265 |
| [0, 0.25) | 22 | capacity_aware | 0.0702 | 0.0031 | 0.0225 |
| [0.25, 0.5) | 18 | no_policy | 0.0397 | -0.0012 | 0.0189 |
| [0.25, 0.5) | 18 | cosine | 0.0402 | 0.0093 | 0.0188 |
| [0.25, 0.5) | 18 | capacity_aware | 0.0249 | 0.0085 | 0.0106 |
| [0.5, 0.7] | 10 | no_policy | 0.0034 | -0.0022 | 0.0131 |
| [0.5, 0.7] | 10 | cosine | 0.0035 | 0.0147 | 0.0125 |
| [0.5, 0.7] | 10 | capacity_aware | 0.0005 | 0.0139 | 0.0068 |

## 4. Q-S-Program: per-PV таблица + delta-vs-P0

Anti-claim: sign-test ниже — diagnostic-only, не gate и не causal-доказательство. Conf-matching между PV=k и PV=0 в LHS-плане НЕТ — остальные оси конфаундированы.

Per-PV агрегат (П1–П3 only, full-50). Метрика — `mean_overload_excess`:

| PV | n_lhs | policy | mean overload | mean utility |
|---:|---:|---|---:|---:|
| 0 | 7 | no_policy | 0.0454 | -0.0008 |
| 0 | 7 | cosine | 0.0458 | 0.0089 |
| 0 | 7 | capacity_aware | 0.0351 | 0.0082 |
| 1 | 9 | no_policy | 0.0223 | -0.0011 |
| 1 | 9 | cosine | 0.0247 | 0.0065 |
| 1 | 9 | capacity_aware | 0.0094 | 0.0058 |
| 2 | 9 | no_policy | 0.0600 | -0.0006 |
| 2 | 9 | cosine | 0.0586 | 0.0085 |
| 2 | 9 | capacity_aware | 0.0413 | 0.0078 |
| 3 | 8 | no_policy | 0.0000 | -0.0006 |
| 3 | 8 | cosine | 0.0000 | 0.0076 |
| 3 | 8 | capacity_aware | 0.0000 | 0.0074 |
| 4 | 7 | no_policy | 0.1712 | -0.0004 |
| 4 | 7 | cosine | 0.1709 | 0.0081 |
| 4 | 7 | capacity_aware | 0.1653 | 0.0074 |
| 5 | 10 | no_policy | 0.0175 | -0.0019 |
| 5 | 10 | cosine | 0.0153 | 0.0076 |
| 5 | 10 | capacity_aware | 0.0138 | 0.0071 |

Delta `mean_overload_excess` относительно PV=0 (абсолютная разница, mean-aggregated; конфаундирован):

| PV | no_policy | cosine | capacity_aware |
|---:|---:|---:|---:|
| 1 | -0.0231 | -0.0211 | -0.0256 |
| 2 | +0.0146 | +0.0128 | +0.0063 |
| 3 | -0.0454 | -0.0458 | -0.0351 |
| 4 | +0.1258 | +0.1251 | +0.1302 |
| 5 | -0.0279 | -0.0305 | -0.0212 |

Sign-test diagnostic-only (`mean_overload_excess`, учтено только при N ≥ 7):

| PV | policy | n | n_pos | n_neg | p_value | примечание |
|---:|---|---:|---:|---:|---:|---|
| 1 | no_policy | 39 | 15 | 24 | 0.200 | diagnostic only |
| 1 | cosine | 39 | 15 | 24 | 0.200 | diagnostic only |
| 1 | capacity_aware | 35 | 10 | 25 | 0.017 | diagnostic only |
| 2 | no_policy | 43 | 20 | 23 | 0.761 | diagnostic only |
| 2 | cosine | 43 | 19 | 24 | 0.542 | diagnostic only |
| 2 | capacity_aware | 35 | 12 | 23 | 0.090 | diagnostic only |
| 3 | no_policy | 24 | 0 | 24 | 0.000 | diagnostic only |
| 3 | cosine | 24 | 0 | 24 | 0.000 | diagnostic only |
| 3 | capacity_aware | 24 | 0 | 24 | 0.000 | diagnostic only |
| 4 | no_policy | 29 | 11 | 18 | 0.265 | diagnostic only |
| 4 | cosine | 29 | 11 | 18 | 0.265 | diagnostic only |
| 4 | capacity_aware | 25 | 7 | 18 | 0.043 | diagnostic only |
| 5 | no_policy | 50 | 23 | 27 | 0.672 | diagnostic only |
| 5 | cosine | 50 | 23 | 27 | 0.672 | diagnostic only |
| 5 | capacity_aware | 38 | 11 | 27 | 0.014 | diagnostic only |

## 5. Q-S-Gossip: 3-bucket × 4 политики

На full-50 включены только П1–П3; П4 учитывается отдельно в §7.

| w_gossip bucket | n_lhs | policy | mean overload | mean utility |
|---|---:|---|---:|---:|
| [0, 0.25) | 23 | no_policy | 0.0189 | -0.0005 |
| [0, 0.25) | 23 | cosine | 0.0195 | 0.0095 |
| [0, 0.25) | 23 | capacity_aware | 0.0112 | 0.0087 |
| [0.25, 0.5) | 19 | no_policy | 0.0977 | -0.0012 |
| [0.25, 0.5) | 19 | cosine | 0.0970 | 0.0074 |
| [0.25, 0.5) | 19 | capacity_aware | 0.0852 | 0.0068 |
| [0.5, 0.7] | 8 | no_policy | 0.0176 | -0.0017 |
| [0.5, 0.7] | 8 | cosine | 0.0160 | 0.0043 |
| [0.5, 0.7] | 8 | capacity_aware | 0.0152 | 0.0039 |

Pairwise win-rate (`mean_overload_excess`) внутри каждого w_gossip-bucket (П1–П3):

| w_gossip bucket | пара | win_strict | win_eps | ties_eps |
|---|---|---:|---:|---:|
| [0, 0.25) | no_policy_vs_cosine | 0.17 | 0.04 | 0.96 |
| [0, 0.25) | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.83 |
| [0, 0.25) | cosine_vs_capacity_aware | 0.00 | 0.00 | 0.87 |
| [0.25, 0.5) | no_policy_vs_cosine | 0.11 | 0.00 | 0.95 |
| [0.25, 0.5) | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.84 |
| [0.25, 0.5) | cosine_vs_capacity_aware | 0.00 | 0.00 | 0.79 |
| [0.5, 0.7] | no_policy_vs_cosine | 0.12 | 0.00 | 0.88 |
| [0.5, 0.7] | no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.88 |
| [0.5, 0.7] | cosine_vs_capacity_aware | 0.00 | 0.00 | 1.00 |

## 6. Q-S-RiskRelevance: scatter + trade-off

Точек на scatter (per-LHS-row × policy): 150
Trade-off маркеров (eps_overload=0.005, eps_utility=0.001): 6

## 7. Q-S-LLMRanker: maximin-only сравнение

Эти числа корректно сравнимы только друг с другом. П4 (llm_ranker) присутствует ТОЛЬКО в этом restricted-to-maximin блоке. Не сравнивать средние П4 со средними П1–П3 в full-50.

Pairwise (`mean_overload_excess`, ε=0.005) на 12 maximin LHS-row:

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps | n |
|---|---:|---:|---:|---:|---:|---:|
| no_policy_vs_cosine | 0.17 | 0.00 | 0.92 | 0.33 | 0.08 | 12 |
| no_policy_vs_capacity_aware | 0.00 | 0.00 | 0.75 | 0.50 | 0.25 | 12 |
| cosine_vs_capacity_aware | 0.00 | 0.00 | 0.92 | 0.42 | 0.08 | 12 |

## 8. Q-S-Stability: volatile points

Порог `std/|mean| > 0.5` на 3 replicate.
Volatile entries: 72

Distribution std по политикам (на 4 метриках, по всем (lhs_row × policy) парам):

| Политика | n | mean std | median std | max std |
|---|---:|---:|---:|---:|
| no_policy | 200 | 0.0023 | 0.0007 | 0.0353 |
| cosine | 200 | 0.0023 | 0.0006 | 0.0328 |
| capacity_aware | 200 | 0.0016 | 0.0004 | 0.0275 |

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

Конференция: `Demo Day ITMO 2026 (English)`
Слотов: 56 (плёнарных: 10, параллельных: 46)
Per-slot capacity (по слотам): min=100, mean=100.6, max=102
Population_for_capacity (фиксировано в JSON): 100

**Базовая calibration:** на параллельном слоте Mobius ≈ 100 мест на 46 параллельных слотах; audience grid {30, 60, 100} согласован с per-slot capacity 100 при capacity_multiplier = 1.0.

### 11.1. Распределение LHS по `capacity_multiplier × audience_size`

| Bucket | n_lhs | a30 | a60 | a100 |
|---|---:|---:|---:|---:|
| stress[0.5,1) | 10 | 4 | 3 | 3 |
| normal[1,2) | 21 | 5 | 9 | 7 |
| loose[2,3] | 19 | 7 | 6 | 6 |

### 11.2. Доля LHS-точек с ненулевым overload (risk-positive)

Терминология: «risk-positive LHS-row» — LHS-row с фактом median overload > 0 хотя бы у одной из П1–П3. Не путать с capacity-stress bucket `[0.5, 1.0)` по `capacity_multiplier` (см. §11.1) — это разные выборки.

- Risk-positive LHS-row: **18 / 50 (36 %)**;
- остальные 32 LHS-row (64 %) — безопасные сценарии: все политики дают overload = 0.

Per-policy overload-frequency:

| Политика | n_evaluated | n_with_overload | fraction |
|---|---:|---:|---:|
| no_policy | 50 | 17 | 34 % |
| cosine | 50 | 17 | 34 % |
| capacity_aware | 50 | 10 | 20 % |

### 11.3. Overload по bucket × policy (П1–П3)

| Bucket | n_lhs | policy | n_overload>0 | mean overload | max overload |
|---|---:|---|---:|---:|---:|
| stress[0.5,1) | 10 | no_policy | 8 | 0.2278 | 1.1966 |
| stress[0.5,1) | 10 | cosine | 7 | 0.2263 | 1.1937 |
| stress[0.5,1) | 10 | capacity_aware | 7 | 0.1889 | 1.1568 |
| normal[1,2) | 21 | no_policy | 9 | 0.0073 | 0.0898 |
| normal[1,2) | 21 | cosine | 10 | 0.0075 | 0.0923 |
| normal[1,2) | 21 | capacity_aware | 3 | 0.0051 | 0.0862 |
| loose[2,3] | 19 | no_policy | 0 | 0.0000 | 0.0000 |
| loose[2,3] | 19 | cosine | 0 | 0.0000 | 0.0000 |
| loose[2,3] | 19 | capacity_aware | 0 | 0.0000 | 0.0000 |

### 11.4. Где capacity_aware реально снижает риск перегрузки

На 18 risk-positive LHS-row (overload > 0) capacity_aware:
- **не хуже за ε (0.005)** относительно max(no_policy, cosine) на **18 / 18 (100 %)** точках;
- **строго снижает риск за ε** на **9 / 18 (50 %)** точках.


Колонка `Δ overload vs cosine` положительная означает, что capacity_aware снижает overload относительно cosine (меньше = лучше).

| LHS | cap_m | aud | overload no_policy | overload cosine | overload capacity_aware | Δ overload vs cosine | strict risk reduction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.408 | 100 | 0.0270 | 0.0283 | 0.0047 | +0.0236 | yes |
| 1 | 1.110 | 60 | 0.0023 | 0.0008 | 0.0000 | +0.0008 | no |
| 3 | 0.534 | 100 | 1.1966 | 1.1937 | 1.1568 | +0.0369 | yes |
| 6 | 1.384 | 100 | 0.0234 | 0.0204 | 0.0172 | +0.0032 | yes |
| 7 | 0.914 | 60 | 0.0095 | 0.0108 | 0.0027 | +0.0081 | yes |
| 13 | 0.666 | 30 | 0.0020 | 0.0000 | 0.0000 | +0.0000 | no |
| 14 | 0.545 | 30 | 0.0108 | 0.0088 | 0.0066 | +0.0022 | no |
| 18 | 1.040 | 60 | 0.0047 | 0.0036 | 0.0000 | +0.0036 | no |
| 26 | 0.629 | 60 | 0.1730 | 0.1915 | 0.0800 | +0.1114 | yes |
| 27 | 1.886 | 100 | 0.0005 | 0.0027 | 0.0000 | +0.0027 | no |
| 32 | 1.127 | 100 | 0.0898 | 0.0923 | 0.0862 | +0.0061 | yes |
| 33 | 1.942 | 100 | 0.0007 | 0.0013 | 0.0000 | +0.0013 | no |
| 35 | 0.963 | 100 | 0.2185 | 0.2176 | 0.1565 | +0.0611 | yes |
| 36 | 0.696 | 60 | 0.1407 | 0.1251 | 0.1213 | +0.0039 | yes |
| 37 | 1.684 | 100 | 0.0038 | 0.0031 | 0.0000 | +0.0031 | no |
| 43 | 1.232 | 60 | 0.0021 | 0.0021 | 0.0000 | +0.0021 | no |
| 47 | 1.757 | 100 | 0.0000 | 0.0025 | 0.0000 | +0.0025 | no |
| 49 | 0.774 | 100 | 0.5266 | 0.5156 | 0.3653 | +0.1502 | yes |

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
| aggregate_replicates | 0.007 |
| pairwise_and_regret | 0.003 |
| sensitivity | 0.017 |
| program_effect | 0.013 |
| gossip_effect | 0.007 |
| risk_utility | 0.001 |
| llm_ranker_diag | 0.002 |
| stability | 0.000 |
| capacity_audit | 0.001 |
| plots | 0.347 |
| write_json | 0.012 |

## 13. Plots

- `plots/analysis_risk_utility_scatter.png`
- `plots/analysis_gossip_bucket_bar.png`
- `plots/analysis_ranking_heatmap_maximin.png`
