# Этап V: LLM-симулятор на 12 maximin 

Дата: 2026-05-12
Конференция: `mobius_2025_autumn_en`
Модель: `openai/gpt-5.4-nano`
Source Q (read-only): `lhs_parametric_mobius_2025_autumn_en_2026-05-12.json`
Status: **ok**

## Параметры

- target LHS rows: [6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]
- target policies: ['no_policy', 'cosine', 'capacity_aware', 'llm_ranker']
- cfg_seed: 1
- budget hard cap: $20.0
- concurrency: 16
- warm cache LLM-ranker: True
- smoke: False

## Cost / time breakdown

- Cumulative cost (всё): **$10.2227**
- LLMAgent cost: **$10.2227**
- LLMAgent calls: 44160
- LLMAgent total time: 51815.4s
- LLMRankerPolicy cost (delta): **$0.0000**
- LLMRankerPolicy API calls (new): 0
- LLMRankerPolicy cache hits: 0
- LLMRankerPolicy total time: 0.0s
- Total wallclock: 8580.2s
- Avg time per eval: 178.8s
- Parse errors total: 133

## Q/S invariant

**PASS:** sha256 всех Q/S артефактов совпадают до и после прогона.

## Per-eval table (overload / utility)

| LHS | policy | gossip | overload | utility | overflow | hall_var |
|---:|---|---|---:|---:|---:|---:|
| 6 | no_policy | moderate | 0.0495 | 0.0315 | 0.1750 | 0.1643 |
| 6 | cosine | moderate | 0.0207 | 0.0389 | 0.1500 | 0.0896 |
| 6 | capacity_aware | moderate | 0.0096 | 0.0332 | 0.0500 | 0.0560 |
| 6 | llm_ranker | moderate | 0.0968 | 0.0226 | 0.2250 | 0.2558 |
| 7 | no_policy | moderate | 0.0073 | 0.0313 | 0.1000 | 0.1090 |
| 7 | cosine | moderate | 0.0081 | 0.0340 | 0.0500 | 0.0802 |
| 7 | capacity_aware | moderate | 0.0040 | 0.0355 | 0.0500 | 0.0520 |
| 7 | llm_ranker | moderate | 0.0540 | 0.0207 | 0.2000 | 0.1805 |
| 13 | no_policy | moderate | 0.0000 | 0.0279 | 0.0000 | 0.0615 |
| 13 | cosine | moderate | 0.0000 | 0.0293 | 0.0000 | 0.0340 |
| 13 | capacity_aware | moderate | 0.0000 | 0.0311 | 0.0000 | 0.0283 |
| 13 | llm_ranker | moderate | 0.0069 | 0.0177 | 0.0526 | 0.0994 |
| 18 | no_policy | moderate | 0.0226 | 0.0370 | 0.1842 | 0.1129 |
| 18 | cosine | moderate | 0.0128 | 0.0430 | 0.0789 | 0.0689 |
| 18 | capacity_aware | moderate | 0.0023 | 0.0394 | 0.0526 | 0.0397 |
| 18 | llm_ranker | moderate | 0.0466 | 0.0277 | 0.2368 | 0.1682 |
| 23 | no_policy | strong | 0.0000 | 0.0290 | 0.0000 | 0.0148 |
| 23 | cosine | strong | 0.0000 | 0.0443 | 0.0000 | 0.0074 |
| 23 | capacity_aware | strong | 0.0000 | 0.0419 | 0.0000 | 0.0040 |
| 23 | llm_ranker | strong | 0.0000 | 0.0191 | 0.0000 | 0.0280 |
| 27 | no_policy | strong | 0.0276 | 0.0244 | 0.1282 | 0.1425 |
| 27 | cosine | strong | 0.0056 | 0.0411 | 0.0256 | 0.0663 |
| 27 | capacity_aware | strong | 0.0000 | 0.0371 | 0.0000 | 0.0471 |
| 27 | llm_ranker | strong | 0.0605 | 0.0169 | 0.2308 | 0.2352 |
| 31 | no_policy | moderate | 0.0000 | 0.0311 | 0.0000 | 0.0525 |
| 31 | cosine | moderate | 0.0000 | 0.0396 | 0.0000 | 0.0240 |
| 31 | capacity_aware | moderate | 0.0000 | 0.0366 | 0.0000 | 0.0149 |
| 31 | llm_ranker | moderate | 0.0000 | 0.0204 | 0.0000 | 0.0669 |
| 34 | no_policy | moderate | 0.0000 | 0.0289 | 0.0000 | 0.0396 |
| 34 | cosine | moderate | 0.0000 | 0.0390 | 0.0000 | 0.0224 |
| 34 | capacity_aware | moderate | 0.0000 | 0.0404 | 0.0000 | 0.0157 |
| 34 | llm_ranker | moderate | 0.0000 | 0.0236 | 0.0000 | 0.0555 |
| 36 | no_policy | strong | 0.2127 | 0.0194 | 0.3158 | 0.4478 |
| 36 | cosine | strong | 0.1371 | 0.0352 | 0.3421 | 0.2717 |
| 36 | capacity_aware | strong | 0.0636 | 0.0295 | 0.2632 | 0.0910 |
| 36 | llm_ranker | strong | 0.2917 | 0.0163 | 0.3158 | 0.7214 |
| 42 | no_policy | moderate | 0.0000 | 0.0259 | 0.0000 | 0.0061 |
| 42 | cosine | moderate | 0.0000 | 0.0358 | 0.0000 | 0.0048 |
| 42 | capacity_aware | moderate | 0.0000 | 0.0328 | 0.0000 | 0.0029 |
| 42 | llm_ranker | moderate | 0.0000 | 0.0242 | 0.0000 | 0.0081 |
| 45 | no_policy | moderate | 0.0000 | 0.0401 | 0.0000 | 0.0043 |
| 45 | cosine | moderate | 0.0000 | 0.0472 | 0.0000 | 0.0032 |
| 45 | capacity_aware | moderate | 0.0000 | 0.0466 | 0.0000 | 0.0024 |
| 45 | llm_ranker | moderate | 0.0000 | 0.0333 | 0.0000 | 0.0059 |
| 48 | no_policy | strong | 0.0000 | 0.0216 | 0.0000 | 0.0061 |
| 48 | cosine | strong | 0.0000 | 0.0349 | 0.0000 | 0.0025 |
| 48 | capacity_aware | strong | 0.0000 | 0.0358 | 0.0000 | 0.0029 |
| 48 | llm_ranker | strong | 0.0000 | 0.0153 | 0.0000 | 0.0090 |
