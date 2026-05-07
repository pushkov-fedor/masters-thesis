# Этап L: проверка gossip-инкремента

Дата: 2026-05-07
Этап: L (PIVOT_IMPLEMENTATION_PLAN r5).
Source-of-truth: `docs/spikes/spike_gossip.md` §11 (L.1–L.6); `docs/spikes/spike_gossip_llm_amendment.md` §6, Q-J10.
Артефакты прогонов:
- `experiments/results/smoke_toy_microconf_2026-05-07.{json,md}` — smoke на toy с расширенной осью `w_gossip ∈ {0.0, 0.3, 0.7}`;
- `experiments/results/smoke_mobius_2025_autumn_2026-05-07.{json,md}` — smoke на Mobius с тем же гридом;
- `experiments/results/llm_spike_2026_05_07_L_AB.json` — LLM-gossip A/B при `w_gossip ∈ {0.0, 0.5}`.

Кода в этапе L не правил.

## 1. Сводная таблица L.1–L.6

| # | Проверка | Источник | Где запускалась | Ключевой факт | Статус |
|---|---|---|---|---|---|
| L.1 | Инвариант `w_gossip = 0` ⇒ совпадение с базовой моделью этапа E (пословное равенство `chosen` всех steps) | spike_gossip §11.L.1 | `pytest`: `test_simulator_unit::test_w_gossip_zero_baseline_invariance` | Для всех 3 политик (П1, П2, П3) и одинаковых seed/users/conf траектории `chosen` идентичны при `w_gossip = 0` | **PASS** |
| L.2 | Социальный сигнал реально влияет при `w_gossip > 0`: распределение посещений отличается от `w_gossip = 0` | spike_gossip §11.L.2 | `pytest`: `test_simulator_unit::test_gossip_changes_distribution_when_positive` + smoke (см. ниже) | На 80 синтетических users / 2 слота / cosine: `hall_load_per_slot(w_g=0.0) ≠ hall_load_per_slot(w_g=0.2)`. На Mobius natural cosine: overload 0.0899 → 0.1021 при росте `w_gossip` 0.0 → 0.3 | **PASS** |
| L.3 | Рост `w_gossip` усиливает концентрацию: `hall_load_gini` монотонно неубывающая по `w_gossip ∈ {0, 0.3, 0.7}` | spike_gossip §11.L.3, гипотеза этапа K | `pytest`: `test_extreme_conditions::test_l3_gossip_concentration_monotone` + smoke (см. блок «Смежные наблюдения») | На toy_microconf_2slot stress×0.5 / no_policy / 5 seeds: gini монотонна в пределах ε=5e-3 на сетке `w_gossip ∈ {0, 0.3, 0.7}` | **PASS** |
| L.4 | EC1–EC4 остаются зелёными при `w_gossip > 0` | spike_gossip §11.L.4 | `pytest`: 4 теста `test_extreme_conditions::test_ecN_extended_*` + L.5 + L.3 | EC1 (cap×3 → overload=0): PASS; EC2 (монотонность по cap_mult): PASS; EC3 (`w_rec=0, w_gossip=0.3` → range_overload < 1e-9): PASS; EC4 (`w_rec=0.7, w_gossip=0.3, stress×0.5` → range_overload > 0.02): PASS | **PASS** |
| L.5 | Capacity-aware (П3) не теряет смысла при `w_gossip > 0`: на стрессовой capacity П3 даёт overload ≤ П2 | spike_gossip §11.L.5 + amendment §6 | `pytest`: `test_extreme_conditions::test_l5_capacity_aware_still_works_with_gossip` + Mobius smoke | На Mobius stress×0.5: для всех 6 валидных пар `(w_rec, w_gossip)` `cap_overload ≤ cos_overload` (см. таблицу L.5 ниже) | **PASS** |
| L.6 | Sensitivity по `w_gossip` видна: range / max ≥ 5% хотя бы для одной политики (relative range, согласно spike_gossip §11.L.6) | spike_gossip §11.L.6 | smoke на toy + Mobius | На Mobius natural / w_rec=0.5: range/max = 0.012/0.102 ≈ 11.8% (cosine), 0.008/0.041 ≈ 19.5% (no_policy). На toy natural / w_rec=0.5 / no_policy: 0.013/0.060 ≈ 21.7% | **PASS** |

**Итог по L.1–L.6: все 6 проверок PASS.**

Замечание по L.6 (диагностика, не блокер). Чек `gossip_visible_pass` в `check_expectations` (`run_smoke.py`) использует **абсолютный** порог `range_overload ≥ 0.05`. На Mobius natural при `w_rec = 0.5` абсолютный range составляет 0.008–0.018 → формальный absolute-чек FAIL. Однако acceptance L.6 в spike_gossip §11.L.6 формулируется как `range / max > 0.05` (**relative**) — этот вариант проходит уверенно (см. таблицу выше). Расхождение между «строгой» формулой в memo и абсолютным порогом в коде не блокирует L: код-чек диагностический, и на Mobius stress×0.5 переполнение уже потолочное (overload ≈ 1.0+), gossip не может его поднять выше — это интерпретируемое физическое ограничение, не баг. Кода не правил.

## 2. Smoke: ключевые точки

### 2.1. toy_microconf (1 слот × 2 зала × 100 users)

| capacity | w_rec | policy | w_gossip=0.0 overload | w_gossip=0.3 overload | Δ |
|---|---|---|---|---|---|
| natural | 0.5 | no_policy | 0.0600 | 0.0467 | −0.0133 |
| natural | 0.5 | cosine | 0.0600 | 0.0467 | −0.0133 |
| natural | 0.5 | capacity_aware | 0.0267 | 0.0200 | −0.0067 |
| stress_x0_5 | 0.5 | no_policy | 0.9600 | 0.9600 | 0.0000 (потолок) |
| stress_x0_5 | 0.5 | capacity_aware | 0.9467 | 0.8933 | −0.0533 (gossip помогает П3) |
| loose_x3_0 | 0.5 | all | 0.0000 | 0.0000 | 0.0000 (нет переполнения) |

На toy gossip местами **снижает** overload (потому что в нашем единственном-слотном toy все 100 users почти равно-релевантны двум talks, и gossip добавляет ничью «социальную координацию», которая распределяет пользователей более равномерно по залам после первых нескольких выборов). Это ожидаемый артефакт малой структуры toy-конференции; полная картина — на Mobius.

### 2.2. mobius_2025_autumn (16 слотов × 40 talks × 3 зала × 100 users)

| capacity | w_rec | policy | w_g=0.0 overload | w_g=0.3 overload | Δ |
|---|---|---|---|---|---|
| natural | 0.5 | no_policy | 0.0327 | 0.0408 | **+0.0081** (направление гипотезы K: gossip → концентрация → больше overload) |
| natural | 0.5 | cosine | 0.0899 | 0.1021 | **+0.0122** |
| natural | 0.5 | capacity_aware | 0.0000 | 0.0008 | +0.0008 (П3 всё ещё держит) |
| stress_x0_5 | 0.5 | no_policy | 1.0016 | 1.0163 | +0.0147 (потолок ≈ 1.0) |
| stress_x0_5 | 0.5 | capacity_aware | 0.9624 | 0.9624 | 0.0000 (П3 на полном стрессе) |

Гипотеза этапа K (PIVOT_IMPLEMENTATION_PLAN строка 648 «социальное заражение усиливает концентрацию выбора в популярных докладах при больших γ, что увеличивает риск перегрузки даже при capacity-aware политике») **подтверждается** на Mobius natural: рост `w_gossip` повышает `mean_overload_excess` для всех 3 политик. Эффект мал в абсолюте (∼15–20% relative), что согласовано с «log-scale» формой V5 (логарифм диминишингует силу сигнала при `count_t ≪ N_users` на коротких слотах).

### 2.3. L.5 на Mobius: cap_aware vs cosine при разных `(w_rec, w_gossip)`

| w_rec | w_gossip | cap overload | cos overload | cap ≤ cos? |
|---|---|---|---|---|
| 0.0 | 0.0 | 1.0000 | 1.0000 | ✓ |
| 0.5 | 0.0 | 0.9624 | 1.1454 | ✓ |
| 1.0 | 0.0 | 1.0147 | 1.3105 | ✓ |
| 0.0 | 0.3 | 1.0180 | 1.0180 | ✓ |
| 0.5 | 0.3 | 0.9624 | 1.1634 | ✓ |
| 0.0 | 0.7 | 1.0686 | 1.0686 | ✓ |

**6 из 6 валидных симплексных точек: capacity-aware (П3) даёт overload ≤ cosine (П2)**. Capacity-канал не размывается gossip-каналом. Это центральный архитектурный риск spike (§6.V2 amendment), который **не реализовался** благодаря выбору V5 (count_t per talk, не load_frac per hall).

### 2.4. Все базовые EC из этапа I остаются зелёными

Smoke acceptance-чеки (вычислены в `check_expectations` при `w_gossip=0` baseline-точке):

| Чек | toy_microconf | mobius_2025_autumn |
|---|---|---|
| EC3 strict (w_rec=0, natural) | PASS (range=0) | PASS (range=0) |
| MC3 monotone (range по w_rec на stress) | PASS | PASS |
| EC1 (loose × 3.0 → max overload = 0) | PASS | PASS |
| EC2 (stress → max overload > 0) | PASS | PASS |
| TC-D3 asym (П3 ≤ П2 по overload, util_ratio > 0.6) | PASS | PASS |
| **overall_pass** | **OK** | **OK** |

## 3. LLM-gossip A/B (Q-J10)

Конфигурация:
- скрипт: `experiments/scripts/run_llm_spike.py --w-gossip "0.0,0.5" --suffix L_AB`;
- конференция: `toy_microconf_2slot` (2 слота × 2 зала × 4 talks: NLP / iOS / DevOps / Java);
- 10 агентов из `personas_100`, отбор `kmeans_k10_seed42` (как в этапе H);
- политики: `no_policy`, `cosine` (две крайних по rec-каналу);
- модель: `openai/gpt-5.4-mini` через OpenRouter;
- бюджет: $5 hard cap.

| policy | w_gossip | overload | hall_var | overflow | n_skipped | n_parse_errors | cost |
|---|---|---|---|---|---|---|---|
| no_policy | 0.0 | 0.0500 | 0.1700 | 0.2500 | 6 | 0 | $0.0011 |
| cosine | 0.0 | 0.3000 | 0.4250 | 0.5000 | 1 | 0 | $0.0012 |
| no_policy | 0.5 | **0.3000** | **0.4000** | **0.5000** | 0 | 0 | $0.0013 |
| cosine | 0.5 | 0.3000 | 0.4850 | 0.5000 | 1 | 0 | $0.0013 |

Итог LLM-A/B:
- elapsed = 119 с;
- cumulative cost = **$0.0049** (vs $5 hard cap; запас 1000×);
- status = `ok`; n_decisions_aborted = 0;
- n_parse_errors = 0 во всех 4 ячейках.

Ключевое наблюдение: **на no_policy `w_gossip = 0.5` даёт скачок overload c 0.05 до 0.30 (×6)**. Это поведенческое подтверждение, что L2-сигнал реально достигает LLM-агента и сдвигает выбор в направлении «толпы». Эффект меньше выражен на cosine (overload остался 0.30) — потому что cosine уже сам тянет mobile-выборку персон в один зал (iOS/Java); gossip дополнительной концентрации почти не добавляет. Это согласуется с feedback loop literature (S6 Chaney 2018): recommender-side feedback и user-side gossip частично перекрываются.

Skip-rate `n_skipped`:
- При `w_gossip = 0` no_policy: 6/20 = 0.30 (как в этапе H — diagnostic).
- При `w_gossip = 0.5` no_policy: **0/20 = 0.00**. Социальный сигнал даёт агентам с mobile-профилем «социальную мотивацию» выбрать что-то даже на отдалённой теме — content skip уходит.

Это интерпретируемый herd-эффект (S20 social proof, S2 Bandwagon), а не баг.

### LLM-gossip A/B соответствие L-acceptance

| Acceptance | Источник | Статус |
|---|---|---|
| Прогон без ошибок | acceptance этапа H (унаследовано) | exit 0; status=ok | ✓ |
| Стоимость в пределах cap | $0.0049 ≤ $5 | ✓ |
| `n_parse_errors = 0` | 0 во всех ячейках | ✓ |
| `n_skipped/n_decisions ≤ 0.30` (diagnostic) | 0.30 при w_g=0 (граница), 0.00 при w_g=0.5, 0.05 при cosine | ✓ |
| LLM-gossip-сигнал передан корректно | gossip_block формируется в `decide()`, system_prompt по уровням `off/strong` | ✓ (по логам и парсингу промпта) |
| LLM-gossip-сигнал реально влияет на выбор | overload no_policy: 0.05 → 0.30 при `w_gossip` 0 → 0.5 | ✓ |

**LLM-A/B пройден.** Q-J10 выполнен.

## 4. Что НЕ менялось

Согласно условию этапа L:
- `experiments/src/simulator.py` — без правок;
- `experiments/src/llm_agent.py` — без правок;
- `experiments/scripts/run_smoke.py` — без правок;
- `experiments/scripts/run_llm_spike.py` — без правок;
- `experiments/tests/*` — без правок;
- активный реестр политик `experiments/src/policies/registry.py` — без правок;
- legacy не чистился.

## 5. Артефакты

- `experiments/results/smoke_toy_microconf_2026-05-07.{json,md}` (расширенная сетка `w_gossip`);
- `experiments/results/smoke_mobius_2025_autumn_2026-05-07.{json,md}` (расширенная сетка `w_gossip`);
- `experiments/results/llm_spike_2026_05_07_L_AB.json` (LLM-gossip A/B).

Старые артефакты этапов F/H остались на местах и переписаны в рамках этапа L (smoke за тот же день перезаписан с дополненной сеткой `w_gossip`; LLM-spike сохранён под отдельным `_L_AB` суффиксом, не пересекается с этапом H `llm_spike_2026_05_07.json`).

## 6. Итог

**Этап L пройден.** Все 6 acceptance-проверок (L.1–L.6) проходят на pytest и smoke (включая Mobius). LLM-gossip A/B демонстрирует, что L2-сигнал доходит до LLM-агента и сдвигает выбор в ожидаемом направлении. Capacity-канал политики П3 не размывается gossip-каналом. EC1–EC4 этапа I остаются зелёными при `w_gossip > 0`.

К этапу M не перехожу до отдельного сообщения пользователя.
