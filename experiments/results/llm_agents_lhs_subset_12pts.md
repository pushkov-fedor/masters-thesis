# Этап V: LLM-симулятор на 12 maximin 

Дата: 2026-05-08
Конференция: `mobius_2025_autumn`
Модель LLMAgent (симулятор аудитории): `openai/gpt-5.4-nano`
Модель LLMRankerPolicy (П4 ranking docs): `openai/gpt-4o-mini`
Source Q (read-only): `lhs_parametric_mobius_2025_autumn_2026-05-08.json`
Status: **ok**

## Используемые LLM-модели (audit)

Этап V и Q/S задействуют LLM в **разных ролях** на **разных моделях** —
смешивать их нельзя в интерпретации cost / calls / cache.

| Этап | LLM-роль | Модель | Источник модели |
|---|---|---|---|
| Q/S (parametric) | `LLMRankerPolicy` (политика П4) | `openai/gpt-4o-mini` | default класса `LLMRankerPolicy` (`experiments/src/policies/llm_ranker_policy.py:114`) |
| V | `LLMAgent` (симулятор аудитории) | `openai/gpt-5.4-nano` | CLI `--model`, записан в `params.model` |
| V | `LLMRankerPolicy` (политика П4) | `openai/gpt-4o-mini` | hard-coded в `run_llm_lhs_subset.py:682` |

Параметрический Q/S запускался ровно с той же ranker-моделью
(`gpt-4o-mini`), поэтому warm cache `experiments/logs/llm_ranker_cache.json`
полностью переиспользован: 0 новых API-вызовов и нулевая ranker-стоимость
в V (`llmranker_calls_delta=0` для всех 12 maximin LHS-row × П4). Стоимость
$11.5488 в этапе V полностью относится к LLMAgent-вызовам на `gpt-5.4-nano`
(44 160 calls × ~$0.000262/call).

## Параметры

- target LHS rows: [6, 7, 13, 18, 23, 27, 31, 34, 36, 42, 45, 48]
- target policies: ['no_policy', 'cosine', 'capacity_aware', 'llm_ranker']
- cfg_seed: 1
- budget hard cap: $20.0
- concurrency: 16
- warm cache LLM-ranker: True
- smoke: False

## Cost / time breakdown

- Cumulative cost (всё): **$11.5488**
- LLMAgent cost: **$11.5488**
- LLMAgent calls: 44160
- LLMAgent total time: 66360.5s
- LLMRankerPolicy cost (delta): **$0.0000**
- LLMRankerPolicy API calls (new): 0
- LLMRankerPolicy cache hits: 0
- LLMRankerPolicy total time: 0.1s
- Total wallclock: 6385.4s
- Avg time per eval: 133.0s
- Parse errors total: 214

## Q/S invariant

**PASS:** sha256 всех Q/S артефактов совпадают до и после прогона.

## Per-eval table (overload / utility)

| LHS | policy | gossip | overload | utility | overflow | hall_var |
|---:|---|---|---:|---:|---:|---:|
| 6 | no_policy | moderate | 0.1527 | 0.8261 | 0.2750 | 0.3899 |
| 6 | cosine | moderate | 0.0995 | 0.8276 | 0.2500 | 0.2710 |
| 6 | capacity_aware | moderate | 0.0543 | 0.8259 | 0.2250 | 0.1741 |
| 6 | llm_ranker | moderate | 0.1516 | 0.8261 | 0.3250 | 0.3905 |
| 7 | no_policy | moderate | 0.1121 | 0.8258 | 0.2250 | 0.3228 |
| 7 | cosine | moderate | 0.0669 | 0.8273 | 0.2000 | 0.1970 |
| 7 | capacity_aware | moderate | 0.0339 | 0.8262 | 0.1500 | 0.1223 |
| 7 | llm_ranker | moderate | 0.0798 | 0.8245 | 0.2000 | 0.2456 |
| 13 | no_policy | moderate | 0.0263 | 0.8262 | 0.1316 | 0.1718 |
| 13 | cosine | moderate | 0.0057 | 0.8278 | 0.0789 | 0.1034 |
| 13 | capacity_aware | moderate | 0.0000 | 0.8263 | 0.0000 | 0.0689 |
| 13 | llm_ranker | moderate | 0.0206 | 0.8267 | 0.1316 | 0.1851 |
| 18 | no_policy | moderate | 0.0752 | 0.8262 | 0.2895 | 0.2436 |
| 18 | cosine | moderate | 0.0308 | 0.8279 | 0.1842 | 0.1392 |
| 18 | capacity_aware | moderate | 0.0128 | 0.8262 | 0.1053 | 0.0977 |
| 18 | llm_ranker | moderate | 0.0782 | 0.8264 | 0.2632 | 0.2500 |
| 23 | no_policy | strong | 0.0000 | 0.8251 | 0.0000 | 0.0328 |
| 23 | cosine | strong | 0.0000 | 0.8287 | 0.0000 | 0.0210 |
| 23 | capacity_aware | strong | 0.0000 | 0.8272 | 0.0000 | 0.0139 |
| 23 | llm_ranker | strong | 0.0000 | 0.8264 | 0.0000 | 0.0356 |
| 27 | no_policy | strong | 0.0817 | 0.8253 | 0.2564 | 0.2937 |
| 27 | cosine | strong | 0.0220 | 0.8280 | 0.1026 | 0.1338 |
| 27 | capacity_aware | strong | 0.0260 | 0.8263 | 0.1026 | 0.1235 |
| 27 | llm_ranker | strong | 0.0757 | 0.8259 | 0.2051 | 0.2699 |
| 31 | no_policy | moderate | 0.0006 | 0.8262 | 0.0256 | 0.1198 |
| 31 | cosine | moderate | 0.0000 | 0.8282 | 0.0000 | 0.0716 |
| 31 | capacity_aware | moderate | 0.0000 | 0.8267 | 0.0000 | 0.0522 |
| 31 | llm_ranker | moderate | 0.0028 | 0.8264 | 0.0513 | 0.1189 |
| 34 | no_policy | moderate | 0.0000 | 0.8256 | 0.0000 | 0.0897 |
| 34 | cosine | moderate | 0.0000 | 0.8267 | 0.0000 | 0.0464 |
| 34 | capacity_aware | moderate | 0.0000 | 0.8255 | 0.0000 | 0.0364 |
| 34 | llm_ranker | moderate | 0.0000 | 0.8247 | 0.0000 | 0.0779 |
| 36 | no_policy | strong | 0.2708 | 0.8282 | 0.3158 | 0.5680 |
| 36 | cosine | strong | 0.2138 | 0.8304 | 0.3421 | 0.4631 |
| 36 | capacity_aware | strong | 0.1458 | 0.8285 | 0.2632 | 0.3122 |
| 36 | llm_ranker | strong | 0.3081 | 0.8286 | 0.3158 | 0.6981 |
| 42 | no_policy | moderate | 0.0000 | 0.8237 | 0.0000 | 0.0119 |
| 42 | cosine | moderate | 0.0000 | 0.8260 | 0.0000 | 0.0095 |
| 42 | capacity_aware | moderate | 0.0000 | 0.8242 | 0.0000 | 0.0060 |
| 42 | llm_ranker | moderate | 0.0000 | 0.8234 | 0.0000 | 0.0129 |
| 45 | no_policy | moderate | 0.0000 | 0.8232 | 0.0000 | 0.0103 |
| 45 | cosine | moderate | 0.0000 | 0.8256 | 0.0000 | 0.0063 |
| 45 | capacity_aware | moderate | 0.0000 | 0.8238 | 0.0000 | 0.0053 |
| 45 | llm_ranker | moderate | 0.0000 | 0.8226 | 0.0000 | 0.0079 |
| 48 | no_policy | strong | 0.0000 | 0.8278 | 0.0000 | 0.0116 |
| 48 | cosine | strong | 0.0000 | 0.8293 | 0.0000 | 0.0064 |
| 48 | capacity_aware | strong | 0.0000 | 0.8294 | 0.0000 | 0.0052 |
| 48 | llm_ranker | strong | 0.0000 | 0.8282 | 0.0000 | 0.0130 |
