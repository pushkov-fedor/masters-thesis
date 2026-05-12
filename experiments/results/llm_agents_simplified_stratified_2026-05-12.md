# Этап V: LLM-симулятор на 12 maximin 

Дата: 2026-05-13
Конференция: `mobius_2025_autumn_en`
Модель: `openai/gpt-5.4-nano`
Source Q (read-only): `lhs_parametric_simplified_2026-05-12_mobius_2025_autumn_en_for_v.json`
Status: **ok**

## Параметры

- target LHS rows: [0, 1, 2, 6, 12, 14, 17, 31, 33, 34, 36, 39]
- target policies: ['no_policy', 'cosine', 'capacity_aware']
- cfg_seed: 1
- budget hard cap: $15.0
- concurrency: 16
- warm cache LLM-ranker: True
- smoke: False

## Cost / time breakdown

- Cumulative cost (всё): **$13.2310**
- LLMAgent cost: **$13.2310**
- LLMAgent calls: 57600
- LLMAgent total time: 78634.7s
- LLMRankerPolicy cost (delta): **$0.0000**
- LLMRankerPolicy API calls (new): 0
- LLMRankerPolicy cache hits: 0
- LLMRankerPolicy total time: 0.0s
- Total wallclock: 13408.1s
- Avg time per eval: 372.4s
- Parse errors total: 233

## Q/S invariant

**PASS:** sha256 всех Q/S артефактов совпадают до и после прогона.

## Per-eval table (overload / utility)

| LHS | policy | gossip | overload | utility | overflow | hall_var |
|---:|---|---|---:|---:|---:|---:|
| 0 | no_policy | strong | 0.3216 | 0.0269 | 0.4250 | 0.5527 |
| 0 | cosine | strong | 0.3159 | 0.0349 | 0.4750 | 0.4738 |
| 0 | capacity_aware | strong | 0.1751 | 0.0319 | 0.5250 | 0.1135 |
| 1 | no_policy | moderate | 0.3764 | 0.0336 | 0.5250 | 0.4633 |
| 1 | cosine | moderate | 0.3849 | 0.0402 | 0.6500 | 0.3183 |
| 1 | capacity_aware | moderate | 0.2945 | 0.0345 | 0.7500 | 0.1268 |
| 2 | no_policy | strong | 0.0643 | 0.0274 | 0.1750 | 0.2278 |
| 2 | cosine | strong | 0.0423 | 0.0364 | 0.2000 | 0.1404 |
| 2 | capacity_aware | strong | 0.0219 | 0.0316 | 0.0750 | 0.0833 |
| 6 | no_policy | moderate | 0.2526 | 0.0294 | 0.4000 | 0.3341 |
| 6 | cosine | moderate | 0.2326 | 0.0419 | 0.5000 | 0.2412 |
| 6 | capacity_aware | moderate | 0.1604 | 0.0328 | 0.5750 | 0.0839 |
| 12 | no_policy | moderate | 0.2094 | 0.0329 | 0.4250 | 0.3216 |
| 12 | cosine | moderate | 0.1896 | 0.0404 | 0.4250 | 0.2337 |
| 12 | capacity_aware | moderate | 0.1089 | 0.0333 | 0.4500 | 0.0998 |
| 14 | no_policy | moderate | 0.7266 | 0.0324 | 0.7000 | 0.7596 |
| 14 | cosine | moderate | 0.8057 | 0.0405 | 0.8000 | 0.7036 |
| 14 | capacity_aware | moderate | 0.7595 | 0.0360 | 0.9000 | 0.3144 |
| 17 | no_policy | moderate | 0.0198 | 0.0340 | 0.2000 | 0.1097 |
| 17 | cosine | moderate | 0.0234 | 0.0399 | 0.1250 | 0.0969 |
| 17 | capacity_aware | moderate | 0.0031 | 0.0357 | 0.0750 | 0.0450 |
| 31 | no_policy | moderate | 0.0342 | 0.0361 | 0.2000 | 0.1179 |
| 31 | cosine | moderate | 0.0359 | 0.0387 | 0.1750 | 0.1145 |
| 31 | capacity_aware | moderate | 0.0114 | 0.0329 | 0.0500 | 0.0567 |
| 33 | no_policy | moderate | 0.1230 | 0.0340 | 0.3250 | 0.2071 |
| 33 | cosine | moderate | 0.1108 | 0.0373 | 0.3500 | 0.1695 |
| 33 | capacity_aware | moderate | 0.0568 | 0.0355 | 0.2750 | 0.0864 |
| 34 | no_policy | moderate | 0.1596 | 0.0322 | 0.3250 | 0.2479 |
| 34 | cosine | moderate | 0.1449 | 0.0396 | 0.4000 | 0.1913 |
| 34 | capacity_aware | moderate | 0.0750 | 0.0351 | 0.3500 | 0.0776 |
| 36 | no_policy | strong | 0.7010 | 0.0287 | 0.5750 | 1.0881 |
| 36 | cosine | strong | 0.6762 | 0.0373 | 0.7750 | 0.6951 |
| 36 | capacity_aware | strong | 0.6078 | 0.0343 | 0.8500 | 0.3123 |
| 39 | no_policy | moderate | 0.0098 | 0.0333 | 0.1500 | 0.1008 |
| 39 | cosine | moderate | 0.0191 | 0.0385 | 0.1250 | 0.0854 |
| 39 | capacity_aware | moderate | 0.0049 | 0.0353 | 0.0750 | 0.0521 |
