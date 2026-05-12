---
name: EC smoke на LLM-симуляторе (2026-05-13)
description: Граничная верификация LLM-симулятора на 4 точках × 3 политики Mobius EN. Все три проверки (EC1, EC2, EC4) проходят. Артефакты в experiments/results/ec_smoke_llm_2026-05-13.*.
type: project
originSessionId: 2010cee2-1a6b-460e-a9b1-a452d64b57fc
---
**Что.** Smoke граничная верификация LLM-симулятора на программе Mobius EN. До этого на LLM никаких EC-проверок не делалось — был только smoke параметрика на Mobius EN через `smoke_ec_mobius_en.py`. Слайд 11 ранее ошибочно описывал «smoke 3/3 на LLM» — этот прогон закрыл фактическую дыру.

**Когда.** 13.05.2026, ночь перед предзащитой.

**Параметры прогона:**
- 4 LHS-точки: cap=3.0 (EC1), cap=1.0 (EC2 mid), cap=0.5 (EC2 end), cap=1.0/w_rec=0.95 (EC4).
- 3 политики (no_policy, cosine, capacity_aware), audience_size=30 (минимум для скорости), gossip=off (w_gossip=0).
- Модель `gpt-5.4-nano`, concurrency=32, parallel-lhs=4, budget-cap=$5.
- Запуск через `run_llm_lhs_subset.py` с синтетическим `ec_smoke_llm_for_v.json` (4 LHS-rows + 12 evals).

**Затраты:**
- Wallclock: 639 с (10:39).
- Cost: $1.17 (12 evals, 5760 LLM-вызовов).
- Parse errors: 41 / 5760 = 0.7%.
- Q/S invariant: PASS.

**Результаты:**
- **EC1 (cap=3.0):** overload = 0 у всех 3 политик — PASS.
- **EC2 (монотонность по cap):** cap=3.0 → 1.0 даёт 0/0/0 (LLM не разрешает разницу при отсутствии gossip и средних весах); cap=1.0 → 0.5 даёт рост (no_policy 0.018, cosine 0.004, cap_aware 0.000). Монотонный неубывающий рост — PASS, но различимость только на самой стрессовой точке.
- **EC4 (cap=1.0, w_rec=0.95):** политики различаются по utility (no_policy 0.033, cosine 0.045, cap_aware 0.044) — PASS.
- **EC3 не делается:** требует пословного совпадения протоколов, что для стохастической LLM невозможно в принципе.

**Побочное наблюдение (полезно для защиты).** На LHS 102 (cap=0.5) тройка политик по overload расположена в защитном направлении: no_policy 0.018 > cosine 0.004 > cap_aware 0.000. Тот же паттерн, что и главный численный результат работы. Можно использовать в Q&A как «направление сохраняется и на граничной верификации».

**Артефакты:** `experiments/results/ec_smoke_llm_2026-05-13.{json,csv,md,partial.jsonl}`. Input: `experiments/results/ec_smoke_llm_for_v.json`. Лог: `experiments/logs/ec_smoke_llm_2026-05-13.log`.

**Where used.** Слайд 11 (Граничная верификация модели) — формулировка «Все три проверки на LLM проходят». Если в Q&A спросят «как проверяли LLM-симулятор» — карточка готова.
