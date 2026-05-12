---
name: Результаты двух дополнительных прогонов 12.05.2026 — Mobius simplified + Demo Day
description: Накануне предзащиты прогнаны (а) Mobius на упрощённой сетке 3 осей и (б) Demo Day на полной 6-осевой сетке как второй инстанс. Центральный тезис подтверждён, ВКР-числа усилены.
type: project
originSessionId: 8088bf28-d328-45d5-97c8-e30324d87cb5
---
12 мая 2026 (вечер, накануне предзащиты) проведены два дополнительных прогона для усиления защиты.

## 1. Mobius simplified (план #35)

**Сетка:** 3 оси (capacity_multiplier ∈ [0.5, 1.5], w_rec, w_gossip), audience_size=100 фикс, popularity_source=cosine_only фикс, program_variant=0 фикс. 50 LHS × 3 политики (no_policy, cosine, capacity_aware) × 3 replicate = 450 evals. Без llm_ranker, без V (V запущен отдельно — см. ниже).

**Скрипт:** `experiments/scripts/run_lhs_parametric_simplified.py` (новый, не трогает основной код).

**Результаты:**
- Wallclock: 9 секунд.
- risk-positive точек: **48/50 (96%)** — против 11/50 (22%) на старом 6-осевом EN-прогоне.
- `cap_aware` строго лучше `cosine` в **72%** evals (108/150) — против 20% на старом.
- `cap_aware` за ε лучше `cosine` в **64%** evals.
- `cosine` не выигрывает у `cap_aware` **ни в одной** evals (0% strict, 0% за ε) — центральный тезис **сохраняется**.
- Per-policy mean overload: no_policy 0.225, cosine 0.229 (худший), cap_aware 0.191 (лучший).

**Артефакты:** `experiments/results/lhs_parametric_simplified_2026-05-12_mobius_2025_autumn_en.{json,csv,md}`. Также `..._stratified_subset.json` — 12 точек по 4 корзинам overload для V (3 точки safe, 3 light, 3 moderate, 3 severe).

**V запущен в фоне** (`logs/v_simplified_2026-05-12.log`), ETA ~3 часа на момент завершения этой записи; артефакты будут в `results/llm_agents_simplified_stratified_2026-05-12*`.

## 2. Demo Day full (план #26)

**Сетка:** та же 6-осевая, что для Mobius EN в основном эксперименте (capacity_multiplier ∈ [0.5, 3.0], polularity_source 3 уровня, audience_size {30, 60, 100}, program_variant 0..5, и веса). 50 LHS × 3 политики × 3 replicate = 450 evals. Без llm_ranker.

**Сделано subagent'ом** автономно (1612 сек = 27 минут wallclock):
- Перевод программы Demo Day RU → EN (210 talks) через claude-haiku-4.5, $0.23.
- Генерация 150 EN-персон через claude-haiku-4.5, $0.21.
- LLM-judge audit consistency 148/150 (98.7%) PASS.
- BGE+ABTT эмбеддинги (fame.json скопирован из RU, id совпадают).
- LHS-прогон Q (907s, из них 886s — deepcopy bottleneck в `enumerate_modifications`, см. отдельную заметку).
- EC smoke 3/3 PASS.
- Постобработка `analyze_lhs.py`.

**Результаты Demo Day (полное сравнение с Mobius EN):**

| метрика (full 50) | Mobius EN | Demo Day EN |
|---|---:|---:|
| cosine strict wins vs cap_aware | 0% | **0%** |
| cap_aware strict wins vs cosine | 20% | **34%** |
| eps cosine wins | 0% | 0% |
| eps cap_aware wins | 14% | 14% |
| ε-equivalent | 86% | 86% |

Центральный тезис **подтверждён и усилен**: на более крупной программе (210 talks vs 40, 7 залов vs 3) cap_aware строго побеждает cosine в **34%** точек против 20% на Mobius. Cosine по-прежнему не выигрывает у cap_aware ни в одной точке.

**Acceptance персон:**
- internal consistency: 148/150 (98.7%) PASS
- coverage: 0 dead docs из 210 при τ=0.5
- EC smoke: 3/3 PASS
- distributions exp/company matched targets

**Слабое место Demo Day:** Vendi Score пула 36% против 89% у Mobius — пул из 150 персон под широкую программу плотнее в embedding-пространстве. Дублей нет (max ABTT cos 0.785), coverage отличный. Это известное ограничение генерации больших пулов под широкие домены.

**Бюджет:** $0.52 LLM total, 27 минут wallclock.

**Артефакты:** см. PROJECT_OVERVIEW и `experiments/results/demo_day_en/report_demoday_en_summary.md`.

## Итог для защиты 13.05

**Усиление позиции уверенное:**
1. Центральный тезис «cosine не выигрывает у cap_aware ни в одной точке» подтверждён на **двух разных инстансах конференций** на одной и той же сетке параметров (Mobius EN 6-осевой и Demo Day EN 6-осевой) — прямое cross-instance подтверждение.
2. На упрощённой сетке (Mobius simplified) тезис усилен количественно: 72% vs 20% strict wins для cap_aware. Это даёт «второй слой» аргумента — на стрессовых сценариях разница между политиками выражена ещё ярче.

**How to apply (стратегия подачи 13.05):**
- В основной речи слайд 9: подавать **Demo Day как второй инстанс**, прямое сравнение 20% → 34% с Mobius. Это самое сильное и однозначное сообщение.
- В Q&A: Mobius simplified — backup-аргумент «дополнительно проверено на узком стрессовом диапазоне, 72% wins».
- НЕ выводить оба простой парой 72% и 34% — сетки разные, объяснить расхождение методологий за 7 минут сложно.
- Если спросят про Vendi 36% Demo Day — отвечать «coverage программы полный (0 dead docs), пул структурно покрывает программу; Vendi ниже из-за более широкого домена; это известное свойство, не влияет на главный результат».
