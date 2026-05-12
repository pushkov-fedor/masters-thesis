# Этап V: LLM-симулятор на 12 maximin 

Дата: 2026-05-13
Конференция: `mobius_2025_autumn_en`
Модель: `openai/gpt-5.4-nano`
Source Q (read-only): `ec_smoke_llm_for_v.json`
Status: **ok**

## Параметры

- target LHS rows: [100, 101, 102, 103]
- target policies: ['no_policy', 'cosine', 'capacity_aware']
- cfg_seed: 1
- budget hard cap: $5.0
- concurrency: 32
- warm cache LLM-ranker: True
- smoke: False

## Cost / time breakdown

- Cumulative cost (всё): **$1.1744**
- LLMAgent cost: **$1.1744**
- LLMAgent calls: 5760
- LLMAgent total time: 7451.8s
- LLMRankerPolicy cost (delta): **$0.0000**
- LLMRankerPolicy API calls (new): 0
- LLMRankerPolicy cache hits: 0
- LLMRankerPolicy total time: 0.0s
- Total wallclock: 639.4s
- Avg time per eval: 53.3s
- Parse errors total: 41

## Q/S invariant

**PASS:** sha256 всех Q/S артефактов совпадают до и после прогона.

## Per-eval table (overload / utility)

| LHS | policy | gossip | overload | utility | overflow | hall_var |
|---:|---|---|---:|---:|---:|---:|
| 100 | no_policy | off | 0.0000 | 0.0221 | 0.0000 | 0.0038 |
| 100 | cosine | off | 0.0000 | 0.0385 | 0.0000 | 0.0028 |
| 100 | capacity_aware | off | 0.0000 | 0.0354 | 0.0000 | 0.0018 |
| 101 | no_policy | off | 0.0000 | 0.0199 | 0.0000 | 0.0268 |
| 101 | cosine | off | 0.0000 | 0.0348 | 0.0000 | 0.0242 |
| 101 | capacity_aware | off | 0.0000 | 0.0298 | 0.0000 | 0.0182 |
| 102 | no_policy | off | 0.0176 | 0.0161 | 0.0500 | 0.1250 |
| 102 | cosine | off | 0.0044 | 0.0337 | 0.0500 | 0.0779 |
| 102 | capacity_aware | off | 0.0000 | 0.0297 | 0.0000 | 0.0343 |
| 103 | no_policy | off | 0.0000 | 0.0328 | 0.0000 | 0.0300 |
| 103 | cosine | off | 0.0000 | 0.0454 | 0.0000 | 0.0222 |
| 103 | capacity_aware | off | 0.0000 | 0.0438 | 0.0000 | 0.0188 |
