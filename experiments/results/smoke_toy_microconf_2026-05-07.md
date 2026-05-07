# Smoke-прогон этапа F: `toy_microconf`

Дата: 2026-05-07
Ядро: `experiments/src/simulator.py` (этап E).
Реестр политик: `active_policies(include_llm=False)` → ['capacity_aware', 'cosine', 'no_policy'].
Параметры: K=2, τ=0.7, p_skip=0.10, seeds=[1, 2, 3], w_rec=[0.0, 0.5, 1.0].
Capacity-сценарии: ['natural', 'stress_x0_5', 'loose_x3_0'].

## Ключевые метрики (mean over seeds)

### Capacity scenario: `natural` (×1.0)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 0.0600 | 0.6624 | 0.1667 | 0.0195 | 11.0 |
| 0.00 | cosine         | 0.0600 | 0.6624 | 0.1667 | 0.0195 | 11.0 |
| 0.00 | capacity_aware | 0.0600 | 0.6624 | 0.1667 | 0.0195 | 11.0 |
| 0.50 | no_policy      | 0.0600 | 0.5554 | 0.1667 | 0.0194 | 11.0 |
| 0.50 | cosine         | 0.0600 | 0.6632 | 0.1667 | 0.0196 | 11.0 |
| 0.50 | capacity_aware | 0.0267 | 0.6488 | 0.1667 | 0.0070 | 11.0 |
| 1.00 | no_policy      | 0.0200 | 0.4158 | 0.1667 | 0.0068 | 11.0 |
| 1.00 | cosine         | 0.0467 | 0.6738 | 0.1667 | 0.0139 | 11.0 |
| 1.00 | capacity_aware | 0.0133 | 0.6581 | 0.1667 | 0.0039 | 11.0 |

### Capacity scenario: `stress_x0_5` (×0.5)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 0.9600 | 0.6624 | 1.0000 | 0.0780 | 11.0 |
| 0.00 | cosine         | 0.9600 | 0.6624 | 1.0000 | 0.0780 | 11.0 |
| 0.00 | capacity_aware | 0.9600 | 0.6624 | 1.0000 | 0.0780 | 11.0 |
| 0.50 | no_policy      | 0.9600 | 0.5554 | 1.0000 | 0.0775 | 11.0 |
| 0.50 | cosine         | 0.9733 | 0.6632 | 1.0000 | 0.0785 | 11.0 |
| 0.50 | capacity_aware | 0.9467 | 0.6498 | 1.0000 | 0.0471 | 11.0 |
| 1.00 | no_policy      | 0.9333 | 0.4158 | 1.0000 | 0.0273 | 11.0 |
| 1.00 | cosine         | 0.9467 | 0.6738 | 1.0000 | 0.0556 | 11.0 |
| 1.00 | capacity_aware | 0.9200 | 0.6538 | 1.0000 | 0.0300 | 11.0 |

### Capacity scenario: `loose_x3_0` (×3.0)

| w_rec | policy | overload_excess | user_utility | overflow_rate | hall_var | n_skip |
|------:|--------|----------------:|-------------:|--------------:|---------:|-------:|
| 0.00 | no_policy      | 0.0000 | 0.6624 | 0.0000 | 0.0022 | 11.0 |
| 0.00 | cosine         | 0.0000 | 0.6624 | 0.0000 | 0.0022 | 11.0 |
| 0.00 | capacity_aware | 0.0000 | 0.6624 | 0.0000 | 0.0022 | 11.0 |
| 0.50 | no_policy      | 0.0000 | 0.5554 | 0.0000 | 0.0022 | 11.0 |
| 0.50 | cosine         | 0.0000 | 0.6632 | 0.0000 | 0.0022 | 11.0 |
| 0.50 | capacity_aware | 0.0000 | 0.6632 | 0.0000 | 0.0022 | 11.0 |
| 1.00 | no_policy      | 0.0000 | 0.4158 | 0.0000 | 0.0008 | 11.0 |
| 1.00 | cosine         | 0.0000 | 0.6738 | 0.0000 | 0.0015 | 11.0 |
| 1.00 | capacity_aware | 0.0000 | 0.6738 | 0.0000 | 0.0015 | 11.0 |

## Acceptance: 5 ожиданий этапа D через ядро

Stress-сценарий для MC3 / EC2 / TC-D3: `stress_x0_5`.

- **EC3 strict** (w_rec=0, natural): range(utility)=0.00e+00, range(overload)=0.00e+00 → **PASS**
- **MC3 monotone** на `stress_x0_5`: [w=0.00: 0.0000, w=0.50: 0.0267, w=1.00: 0.0267] → **PASS**
- **EC1** (loose ×3.0): max overload = 0.0000 → **PASS**
- **EC2** (`stress_x0_5`): max overload = 0.9733 → **PASS**
- **TC-D3** (П3 vs П2 на `stress_x0_5`):
    - w_rec=0.00: Δoverload(cos-cap)=+0.0000, util_ratio(cap/cos)=1.000
    - w_rec=0.50: Δoverload(cos-cap)=+0.0267, util_ratio(cap/cos)=0.980
    - w_rec=1.00: Δoverload(cos-cap)=+0.0267, util_ratio(cap/cos)=0.970
  - П3 не хуже П2 по overload @ w_rec≥0.5: **PASS**
  - util_ratio(П3/П2) > 0.6 @ w_rec≥0.5: **PASS**

### Итог: **OK — все ожидания выполнены**
