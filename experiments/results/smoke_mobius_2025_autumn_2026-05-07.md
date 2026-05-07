# Smoke-прогон этапа F: `mobius_2025_autumn`

Дата: 2026-05-07
Ядро: `experiments/src/simulator.py` (этап E).
Реестр политик: `active_policies(include_llm=False)` → ['capacity_aware', 'cosine', 'no_policy'].
Параметры: K=2, τ=0.7, p_skip=0.10, seeds=[1, 2, 3], w_rec=[0.0, 0.5, 1.0].
Capacity-сценарии: ['natural', 'stress_x0_5', 'loose_x3_0'].

## Ключевые метрики (mean over seeds)

### Capacity scenario: `natural` (×1.0)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 0.0302 | 0.7333 | 0.1000 | 0.0159 | 174.0 |
| 0.00 | cosine         | 0.0302 | 0.7333 | 0.1000 | 0.0159 | 174.0 |
| 0.00 | capacity_aware | 0.0302 | 0.7333 | 0.1000 | 0.0159 | 174.0 |
| 0.50 | no_policy      | 0.0327 | 0.7332 | 0.1000 | 0.0162 | 174.0 |
| 0.50 | cosine         | 0.0899 | 0.7353 | 0.2333 | 0.0377 | 174.0 |
| 0.50 | capacity_aware | 0.0000 | 0.7341 | 0.0000 | 0.0032 | 174.0 |
| 1.00 | no_policy      | 0.0335 | 0.7331 | 0.1083 | 0.0164 | 174.0 |
| 1.00 | cosine         | 0.1626 | 0.7365 | 0.3167 | 0.0756 | 174.0 |
| 1.00 | capacity_aware | 0.0000 | 0.7346 | 0.0000 | 0.0015 | 174.0 |

### Capacity scenario: `stress_x0_5` (×0.5)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 1.0000 | 0.7333 | 1.0000 | 0.0637 | 174.0 |
| 0.00 | cosine         | 1.0000 | 0.7333 | 1.0000 | 0.0637 | 174.0 |
| 0.00 | capacity_aware | 1.0000 | 0.7333 | 1.0000 | 0.0637 | 174.0 |
| 0.50 | no_policy      | 1.0016 | 0.7332 | 1.0000 | 0.0648 | 174.0 |
| 0.50 | cosine         | 1.1454 | 0.7353 | 0.9833 | 0.1508 | 174.0 |
| 0.50 | capacity_aware | 0.9624 | 0.7343 | 1.0000 | 0.0404 | 174.0 |
| 1.00 | no_policy      | 1.0033 | 0.7331 | 1.0000 | 0.0656 | 174.0 |
| 1.00 | cosine         | 1.3105 | 0.7365 | 0.8917 | 0.3023 | 174.0 |
| 1.00 | capacity_aware | 1.0147 | 0.7351 | 1.0000 | 0.0609 | 174.0 |

### Capacity scenario: `loose_x3_0` (×3.0)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 0.0000 | 0.7333 | 0.0000 | 0.0018 | 174.0 |
| 0.00 | cosine         | 0.0000 | 0.7333 | 0.0000 | 0.0018 | 174.0 |
| 0.00 | capacity_aware | 0.0000 | 0.7333 | 0.0000 | 0.0018 | 174.0 |
| 0.50 | no_policy      | 0.0000 | 0.7332 | 0.0000 | 0.0018 | 174.0 |
| 0.50 | cosine         | 0.0000 | 0.7353 | 0.0000 | 0.0042 | 174.0 |
| 0.50 | capacity_aware | 0.0000 | 0.7346 | 0.0000 | 0.0008 | 174.0 |
| 1.00 | no_policy      | 0.0000 | 0.7331 | 0.0000 | 0.0018 | 174.0 |
| 1.00 | cosine         | 0.0000 | 0.7365 | 0.0000 | 0.0084 | 174.0 |
| 1.00 | capacity_aware | 0.0000 | 0.7354 | 0.0000 | 0.0007 | 174.0 |

## Acceptance: 5 ожиданий этапа D через ядро

Stress-сценарий для MC3 / EC2 / TC-D3: `stress_x0_5`.

- **EC3 strict** (w_rec=0, natural): range(utility)=0.00e+00, range(overload)=0.00e+00 → **PASS**
- **MC3 monotone** на `stress_x0_5`: [w=0.00: 0.0000, w=0.50: 0.1830, w=1.00: 0.3072] → **PASS**
- **EC1** (loose ×3.0): max overload = 0.0000 → **PASS**
- **EC2** (`stress_x0_5`): max overload = 1.3105 → **PASS**
- **TC-D3** (П3 vs П2 на `stress_x0_5`):
    - w_rec=0.00: Δoverload(cos-cap)=+0.0000, util_ratio(cap/cos)=1.000
    - w_rec=0.50: Δoverload(cos-cap)=+0.1830, util_ratio(cap/cos)=0.999
    - w_rec=1.00: Δoverload(cos-cap)=+0.2958, util_ratio(cap/cos)=0.998
  - П3 не хуже П2 по overload @ w_rec≥0.5: **PASS**
  - util_ratio(П3/П2) > 0.6 @ w_rec≥0.5: **PASS**

### Итог: **OK — все ожидания выполнены**
