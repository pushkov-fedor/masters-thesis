# Этап Q: полный параметрический LHS-прогон

Дата: 2026-05-12
Конференция: `mobius_2025_autumn_en`
Master seed: 2026

## Параметры

- n_points: **50**
- replicates: **3**
- maximin_k: **12**
- include_llm_ranker: **False**
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
| prep (capacity / Φ / audience) | 4.05 |
| П1–П3 evals | 5.56 |
| П4 (llm_ranker) evals | 0.00 |
| **итого (run_lhs внутренний)** | **9.72** |
| **wallclock полный** | **9.80** |

## Сводка evals по политикам

| Политика | Evals |
|---|---:|
| no_policy | 150 |
| cosine | 150 |
| capacity_aware | 150 |
| llm_ranker | 0 |
| **итого** | **450** |

## Maximin subset

Indices (12): [6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]

## Acceptance

| Чек | Значение | Статус |
|---|---|---|
| П1–П3 evals == ожидаемое | 450 == 450 | PASS |
| П4 evals == ожидаемое | 0 == 0 | PASS |
| total evals == ожидаемое | 450 == 450 | PASS |
| П4 только на maximin | violations=0 | PASS |
| CRN audience/phi инвариант | violations=0 | PASS |
| cfg_seed = replicate | violations=0 | PASS |
| long-format ключи | missing=[] | PASS |

Дополнительно (диагностика, не блокатор):
- fallback_to_p0 случаев: **0** (когда `enumerate_modifications` вернула меньше swap-вариантов чем требовался индекс program_variant; не silent — флаг записан в каждой соответствующей строке)

### Итог: **PASS**

## Артефакты

- JSON: `lhs_parametric_mobius_2025_autumn_en_2026-05-12.json`
- CSV (long-format): `lhs_parametric_mobius_2025_autumn_en_2026-05-12.csv`
- этот отчёт: `lhs_parametric_mobius_2025_autumn_en_2026-05-12.md`
