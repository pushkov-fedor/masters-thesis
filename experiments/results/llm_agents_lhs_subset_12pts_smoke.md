# Этап V: LLM-симулятор на 12 maximin (SMOKE)

Дата: 2026-05-08
Конференция: `mobius_2025_autumn`
Модель: `openai/gpt-5.4-nano`
Source Q (read-only): `lhs_parametric_mobius_2025_autumn_2026-05-08.json`
Status: **ok**

## Параметры

- target LHS rows: [13]
- target policies: ['no_policy', 'capacity_aware']
- cfg_seed: 1
- budget hard cap: $1.0
- concurrency: 16
- warm cache LLM-ranker: True
- smoke: True

## Cost / time breakdown

- Cumulative cost (всё): **$0.2482**
- LLMAgent cost: **$0.2482**
- LLMAgent calls: 960
- LLMAgent total time: 1386.4s
- LLMRankerPolicy cost (delta): **$0.0000**
- LLMRankerPolicy API calls (new): 0
- LLMRankerPolicy cache hits: 0
- LLMRankerPolicy total time: 0.0s
- Total wallclock: 704.4s
- Avg time per eval: 352.2s
- Parse errors total: 3

## Q/S invariant

**PASS:** sha256 всех Q/S артефактов совпадают до и после прогона.

## Per-eval table (overload / utility)

| LHS | policy | gossip | overload | utility | overflow | hall_var |
|---:|---|---|---:|---:|---:|---:|
| 13 | no_policy | moderate | 0.0309 | 0.8263 | 0.1579 | 0.1943 |
| 13 | capacity_aware | moderate | 0.0034 | 0.8263 | 0.0526 | 0.0731 |
