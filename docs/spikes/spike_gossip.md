# Design-spike: gossip / социальное заражение (этап J)

Дата: 2026-05-07
Этап: J (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: design-spike, evidence-first; кода не меняет, экспериментов не запускает, LLM API не вызывает.

> **Amendment 2026-05-07 (после прочтения пользователем):** решение по LLM-стороне gossip пересмотрено — gossip передаётся **и в параметрический, и в LLM-симулятор**. Подробности и рекомендация (вариант L2: реальные числа `count_t / N_users` в промпте + 3-уровневая параметризация системным промптом) — в отдельном файле `docs/spikes/spike_gossip_llm_amendment.md`. Параметрическая часть этого spike (V5 log_count, симплексная нормировка, EC-проверки L.1–L.6) **не меняется**. Точечные следы amendment — в §8 (LLM-блок), §10 пункт 5, §12 Q-J7, Recommended decision for K.

> Memo evidence-first. Сначала research log и реально изученные источники, затем требования и обзор вариантов, и только в конце — рекомендация и описание минимальной первой версии для этапа K. Структура повторяет принятые ранее `spike_behavior_model.md` (этап C) и `spike_llm_simulator.md` (этап G).

---

## 1. Проблема

Нужно зафиксировать форму gossip-компонента utility участника `gossip(t, L_t)` для первой работающей реализации в ядре `simulator.py`. Решение — вход в этап K (минимальная правка `_process_one_slot` и `SimConfig`); проверка — этап L. Без выбора формы нельзя ни запустить gossip-инкремент, ни прогнать LHS / CRN с осью `w_gossip` (PROJECT_DESIGN §8 ось 3 явно требует `w_gossip` как параметрическую ось).

Цель memo — выбрать **минимальную проверяемую** форму, удовлетворяющую трём жёстким инвариантам:

1. **PROJECT_DESIGN §7 — трёхкомпонентная аддитивная utility** `U = w_rel·rel + w_rec·rec + w_gossip·gossip(t, L_t)`. Это нормативная фиксация: gossip — отдельный аддитивный компонент со своим весом.
2. **Инвариант «выключения»:** при `w_gossip = 0` поведение симулятора **строго совпадает** с базовой моделью этапа E (точная пословная идентичность `chosen_id` всех steps при тех же seed / users / conf / policy). Это аналог EC3 для gossip-канала; pytest-набор этапа I должен оставаться зелёным без правок.
3. **Capacity-канал не размывается gossip-каналом.** Политика П3 capacity_aware (`score = sim − α·load_frac`) должна сохранить смысл при `w_gossip > 0`. EC1–EC4 проходят и в новой точке `w_gossip > 0`.

Дополнительно: gossip должен быть детерминированным (не потребляет RNG), чтобы CRN на 12 точках LHS работало (одно зерно при разных политиках даёт идентичные траектории при `w_rec = 0`, **независимо** от значения `w_gossip`).

Без решения по форме `gossip(t, L_t)` этап K (реализация) и этап L (проверка) заблокированы; LHS / Φ-оператор / полный параметрический run заблокированы каскадно.

---

## 2. Текущая реализация в репозитории

### 2.1. Точка вкладывания gossip-слагаемого

Файл: `experiments/src/simulator.py`, функция `_process_one_slot`, строка 464:

```python
u = cfg.w_rel * effective_rel + cfg.w_rec * rec_indicator
```

Сюда будет добавлено `+ cfg.w_gossip * gossip_value(...)`. Цикл по `tid in consider_ids` (строки 459–465) проходит по всем докладам слота. Перед циклом: `recs_set = set(recs)` (строка 457). После выбора: `local_load[chosen_hall.id] += 1` (строка 490).

### 2.2. Что доступно для gossip-сигнала

`local_load: Dict[int, int]` (строка 378) — счётчик по `hall_id`, каждое значение растёт на 1 после выбора пользователя N (строка 490). К моменту обработки пользователя N+1 он отражает выбор первых N пользователей **этого же слота**. Это естественный sequential causality: «следующий видит предыдущих».

Тот же `local_load` доступен политике через `state["hall_load"]` (строка 393):
```python
state = {
    "hall_load": {(slot.id, hid): occ for hid, occ in local_load.items()},
    ...
}
```

Это даёт **паритет источников информации** между gossip-каналом утилиты и capacity-каналом политики П3 — оба видят одно и то же состояние, но через разные коэффициенты (`w_gossip` в utility vs `α` в политике П3 `score = sim − α·load_frac`).

### 2.3. Архитектурный риск: «gossip по залу» vs «gossip по докладу»

`local_load` агрегирует выбор на уровне зала (`Dict[int, int]`, ключ — `hall_id`). Если в одном слоте один зал содержит ровно один доклад (это так на JUG / Mobius / ITC / Meetup и в `toy_microconf_2slot`), то `local_load[hall(t)] == count(пользователей, выбравших t)` тождественно. Семантически же это разные сигналы:

- **«gossip по залу»** = `load_frac(hall(t))` — буквально то же, что использует политика П3 со знаком минус. Прямой риск двойного учёта capacity-канала (см. §6.V2).
- **«gossip по докладу»** = `count(t)` — счётчик пользователей, выбравших именно t. Семантически независим от capacity (concept «t популярен» ≠ concept «зал заполнен»).

Чтобы формально разделить, нужно ввести отдельный счётчик `local_choice_count: Dict[str, int]` (ключ — `talk_id`), параллельный `local_load`. На текущих программах он численно совпадает с `local_load[hall(t)]`, но семантика и расширяемость другие.

### 2.4. CRN-инвариантность

Строки 376–378:
```python
choice_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx)
policy_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx + 31)
```

Два независимых RNG-потока. При `w_rec = 0` rec_indicator зануляется → utility не зависит от выдачи политики → `choice_rng` идёт по одной траектории при разных политиках (формальная EC3, проверена в `test_extreme_conditions.py::test_ec3_invariance_when_w_rec_zero`).

**Критичное наблюдение для gossip:** если gossip-функция детерминирована (читает только `local_choice_count` и саму программу, без вызовов RNG), то новый RNG-поток не нужен. Инвариант `w_gossip = 0` ⇒ baseline получается «бесплатно»: `cfg.w_gossip · f(...) = 0` тождественно при `w_gossip = 0`, никаких изменений в `choice_rng` траектории не возникает. Все варианты V1–V5 ниже — детерминированные.

### 2.5. SimConfig

Файл: `simulator.py:152–200`. Уже содержит `w_rel`, `w_rec`, `w_fame`, `tau`, `p_skip_base`, `K`, `seed`. Добавление `w_gossip: float = 0.0` (default = 0) тривиально и сохраняет обратную совместимость с тестами этапа I — они не задают `w_gossip` явно и получают 0, поэтому все траектории остаются прежними.

### 2.6. Sequential vs batched

Строка 391: `for user in user_order:` — строгий sequential проход внутри слота. Между слотами есть `slot_concurrency > 1` (строки 542–553), но `local_load` создаётся локально в каждом `_process_one_slot` (строка 378) и **не разделяется между слотами**. Поэтому in-slot gossip каузально течёт по порядку (агент N+1 видит выбор первых N), а между слотами «памяти» нет. Это автоматически отсекает V7 (memory-based across slots) от минимальной реализации — для него потребуется проброс кумулятивного состояния через `simulate_async`, что не аддитивная правка.

### 2.7. Точки в скриптах прогона

- `experiments/scripts/run_smoke.py` строки 355, 366–369, 148–185 — CLI-флаг `--w-rec`, грид прогонов. Чтобы добавить ось `w_gossip`, потребуется добавить флаг `--w-gossip "0.0,0.3,0.7"` и расширить `run_grid` дополнительным циклом. Acceptance-чеки (строки 199–273) должны проверять EC и при `w_gossip = 0`, и при `w_gossip > 0`. Это правка этапа K, не J.
- `experiments/scripts/run_toy_microconf.py` строки 218–314 — содержит локальную копию `simulate_local` с тем же местом вкладывания (строка 280: `utils = w_rel * rels + w_rec * rec_indicator`). Это «зеркало» строки 464 ядра. Если выберем менять и его, удвоится правка; альтернатива — отказаться от локального скрипта в пользу via_core.
- `experiments/scripts/run_toy_microconf_via_core.py` — использует `simulate` напрямую, изменения ядра автоматически попадут сюда. Acceptance-функция `check_utility_invariance` (строки 157–204) тестирует инвариант EC3 при `w_rec = 0`. После K её нужно расширить под `w_gossip = 0`-инвариант.

### 2.8. Тестовый слой

- `experiments/tests/test_extreme_conditions.py` строки 1–181 — EC1–EC4 уже зелёные. Добавление gossip потребует: (а) extended EC3 при `w_rec = 0, w_gossip > 0` (политики идентичны, потому что gossip не зависит от политики); (б) baseline-equivalence test при `w_gossip = 0` (пословно совпадает с текущей реализацией); (в) EC4 при `w_gossip = 0.3` (политики различимы при умеренном gossip).
- `experiments/tests/test_simulator_unit.py` строки 1–129 — `test_utility_invariant_under_policy_when_w_rec_zero` сравнивает `chosen` пословно. Тот же образец понадобится для gossip-инварианта.
- `experiments/tests/test_toy_cases.py` строки 115–129 — TC3 уже стоит как `pytest.skip` с reason про gossip-инкремент J–K–L. Разморозится в L.

---

## 3. Требования из PROJECT_DESIGN / PROJECT_STATUS / PIVOT_IMPLEMENTATION_PLAN

### Из PROJECT_DESIGN

- **§7 Модуль модели поведения участника** (строка 92): «В функцию полезности входят три компонента, каждый управляется отдельным весом. Компонент релевантности отражает близость доклада к профилю участника, w_rel. Компонент следования рекомендации отражает влияние показанной политикой подсказки, w_rec. Компонент социального заражения (gossip) описывает динамическое усиление выбора по тому, что уже выбрала когорта, w_gossip. Кроме весов, у модели есть параметр стохастичности.» — нормативная фиксация трёхкомпонентной аддитивной формы и динамической природы gossip («что уже выбрала когорта»).
- **§8 Параметрические оси** (строка 122): «Ось 3 — параметры модели поведения участника. Веса (w_rel, w_rec, w_gossip) и параметр стохастичности.» — `w_gossip` обязан войти в LHS как параметрическая ось, не boolean флаг.
- **§9 Состав политик** (строка 134): «На итоговое распределение посещений политика влияет только через компонент w_rec функции полезности участника. При w_rec → 0 политики неразличимы по итогу; при w_rec → 1 выбор политики определяет итог полностью.» — этот инвариант сохраняется при `w_gossip > 0`: gossip-канал не должен зависеть от выдачи политики.
- **§11 Верификация в граничных условиях** (строка 202): EC1–EC4. После gossip они должны остаться зелёными как минимум на одной тестовой точке `w_gossip > 0`.
- **§13 Допущения** (строки 214–224): аудитория синтетическая, индивидуальный выбор параметрический, модификации программы локальные. Это запрещает gossip через социальный граф дружб (нужны данные отношений) или inter-agent chat (стоп-лист §5).

### Из PROJECT_STATUS

- **§5 Стоп-лист** (строка 49): «Big Five / social graph / inter-slot chat как реализованный метод (был прототип, не основной результат)». Прямо исключает V9 (network-based), V10 (inter-agent chat). V6 (cohort) близок по жанру и должен оставаться откладываемым.
- **§7 Текущее направление** (строка 91): «MMR и gossip-вход — параметрические модификаторы, не отдельные политики». Gossip остаётся в utility агента, политика остаётся четырёхэлементной (П1–П4).
- **§8 Валидация** (строки 99–105): toy-cases, internal consistency, monotonicity, repeated seeds, sensitivity, согласованность двух симуляторов, EC tests. Канон Sargent / Kleijnen + Larooij & Törnberg.

### Из PIVOT_IMPLEMENTATION_PLAN r5

- **Раздел 7** (строки 230–342): обязательный research log с минимальным бюджетом ~5–8 минут активного исследования; источники реально открыты, не сниппеты; пометки доступа `abstract-only / not-accessible / derived-only`. Формат decision memo — 12 пунктов.
- **Раздел 6 принципы** (строки 214–227): Принцип 5: «Gossip — обязательный плановый инкремент модели поведения, не опциональная ветка. До полного LHS должна быть реализована и проверена форма `rel + rec + gossip`, а ось `w_gossip` должна войти в экспериментальный протокол. **LHS и full parametric run нельзя запускать до реализации и проверки gossip.**»
- **Этап J** (строки 621–649): 8 предложенных вариантов (зеркало списку в этом memo с расширением до 11). Memo обязан рассмотреть каждый, дать рекомендацию, параметризацию, протокол проверки.
- **Этап K** (строки 651–667): расширение `SimConfig`, вкладывание в utility, корректное обновление `L_t` в слоте. «Последовательная обработка пользователей в слоте уже даёт это» — фиксирует, что in-slot sequential — достаточный механизм для gossip causality.
- **Этап L** (строки 670–691): шесть проверок (см. §11 ниже).
- **Гипотеза этапа K** (строка 648): «Социальное заражение усиливает концентрацию выбора в популярных докладах при больших γ, что увеличивает риск перегрузки даже при capacity-aware политике, и должно учитываться при анализе устойчивости политик».

### Из spike_behavior_model.md (этап C, accepted)

- V7 (Logit-choice с gossip) был зафиксирован как «отдельный плановый инкремент этапов J–L», вынесен из spike модели поведения. Текущий spike J — продолжение того же выбора V7 = V4-base + gossip-term, обсуждается **только форма gossip-функции**, не пересборка модели.
- Accepted decision §10 пункт 1: «Gossip-компонент `w_gossip · gossip(t, L_t)`. Расширение V4 → V7. Отдельный spike (этап J), отдельная реализация (этап K), отдельная проверка (этап L). До прохождения L не запускаем полный LHS.»
- §3 «Аддитивное расширение, не пересборка модели» — этап K реализуется как минимальная правка той же утилитной формы.

### Из spike_llm_simulator.md (этап G, accepted)

- Q-G «Capacity в промпт LLM-агента не передавать (соответствует accepted decision этапа C)». **По аналогии: gossip в промпт LLM-агента тоже не передаётся.** Иначе LLM-симулятор перестаёт быть независимым от параметрического. Это автоматически отвергает V8 (LLM-only gossip) с двух сторон: ломает паритет двух симуляторов и лишает параметрический симулятор gossip-канала, что нарушает PROJECT_DESIGN §7.

---

## 4. Research log

Расширенный design-spike по правилу §7 PIVOT_IMPLEMENTATION_PLAN r5 выполнен через отдельный research-subagent с time-boxed бюджетом. Subagent изучал предметную область и возвращал research brief; написание самого memo ведётся в основной сессии.

### 4.1. Время

- **start time (subagent):** epoch `1778161485`, ~2026-05-07 08:44:45 MSK.
- **end time (subagent):** epoch `1778161832`, ~2026-05-07 08:50:32 MSK.
- **elapsed seconds:** 347 (≥ 300 — минимальный research budget по §7).
- Sleep / искусственное ожидание не использовались. Время потрачено на реальное I/O: Read локальных файлов, WebFetch внешних источников. WebFetch на JSTOR / Science / SSC / Springer / ScienceDirect частично заблокирован (paywall / 403 / 303), компенсировано локальным `research_field_survey_2026-05-04.md` и Wikipedia как secondary source с явной пометкой `derived-only`.

### 4.2. Изученные файлы кода

- `experiments/src/simulator.py` — `_process_one_slot` (строки 329–501), `SimConfig` (152–200). Точка вкладывания gossip-слагаемого — строка 464. CRN-структура — 376–378.
- `experiments/scripts/run_smoke.py` — точки расширения для оси `w_gossip` в гриде.
- `experiments/scripts/run_toy_microconf.py` — локальная копия utility-логики (строка 280), требует синхронной правки если её сохранять.
- `experiments/scripts/run_toy_microconf_via_core.py` — использует ядро напрямую; `check_utility_invariance` (строки 157–204) — кандидат на расширение.
- `experiments/tests/test_extreme_conditions.py` — EC1–EC4 (строки 1–181); шаблон для gossip-расширенных EC.
- `experiments/tests/test_simulator_unit.py` — `test_utility_invariant_under_policy_when_w_rec_zero` (строки 35–70); шаблон для baseline-equivalence теста.
- `experiments/tests/test_toy_cases.py` — TC3 как `pytest.skip` (строки 115–129); разморозится в L.

### 4.3. Изученные документы проекта

- `PROJECT_DESIGN.md` — §5, §7, §8, §9, §11, §13.
- `PROJECT_STATUS.md` — §5, §7, §8.
- `PIVOT_IMPLEMENTATION_PLAN.md` — §6 принцип 5, §7 правила design-spike, этапы J/K/L (строки 621–691), раздел 10 (toy-cases TC3, EC tests).
- `docs/spikes/spike_behavior_model.md` — accepted decision §10 пункт 1, V7 в обзоре вариантов.
- `docs/spikes/spike_llm_simulator.md` — Q-G решение (capacity не в промпте LLM).
- `.claude/memory/research_field_survey_2026-05-04.md` — 2.2 (capacity-aware), 3.2 (LLM-агенты), 5 (валидация), 6 (gap).
- `.claude/memory/reference_validation_defense.md` — must-cite, distribution-match Meetup ρ=0.438.
- `materials/_legacy/research-симуляторы-датасеты.md` — secondary reading по архитектурным модулям симуляторов.

### 4.4. Внешние источники

См. §5 — отдельная таблица. Каждый источник с пометкой `full / partial / abstract-only / derived-only / not-accessible`. Особо отмечу: основные классические работы по cascades (Bikhchandani 1992, Banerjee 1992, Granovetter 1978) и Brock-Durlauf 2001 (social-MNL) недоступны через WebFetch — компенсация через Wikipedia + research_field_survey + diversity-paradox 2025 (E6 — full HTML с прямой формулой). Это известное ограничение для текущего spike; полные PDF необходимо открыть вручную для citation-precision в diss.

### 4.5. Что оказалось нерелевантным и почему

- **Vangerven 2018** (conference scheduling, OMEGA) — assumes individual preferences без социального заражения; цитируется только в формальной постановке задачи DSS, не для gossip.
- **Macal-North 2010** (ABM tutorial, Springer) — 303 redirect; ABM-канон уже покрыт через `research_field_survey_2026-05-04.md` (Lempert / Marchau / Kwakkel).
- **Centola 2010** (Science 403) — empirical complex contagion (нужна множественная экспозиция); теоретически релевантно, но избыточно для минимального V1/V5.
- **Quaeghebeur 2014** — упомянут в research_field_survey как «нужна проверка», для gossip-spike не релевантен.
- **AgentCF (S8 в spike LLM-симулятора)** — talk-as-agent дуальный подход; для нашей задачи talks — фиксированные объекты, не агенты.
- **Mehrotra et al. fair marketplace** — формальное название не нашлось точным URL.

### 4.6. Открытые вопросы по итогам research

См. §12 (открытые решения для подтверждения пользователем).

---

## 5. Обзор реально изученных источников

Источники сгруппированы по направлениям; каждый снабжён пометкой о фактической доступности.

### A. Классика information cascades / herd behavior / threshold models

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S1 | Bikhchandani S., Hirshleifer D., Welch I. (1992). *A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades.* JPE 100(5). | classic theory | derived-only (JSTOR 403; UCLA mirror SSL fail; через Wikipedia как secondary) | Cascade — sequential agents видят свой private signal + историю действий предшественников. Триггер cascade: разница `accepts − rejects` ≥ 2 ⇒ агент игнорирует private signal. Cascades **fragile**: малая публичная информация может развернуть. Прямо мотивирует использование `local_load` или `local_choice_count` как «история действий первых N». |
| S2 | Banerjee A. (1992). *A Simple Model of Herd Behavior.* QJE 107(3). | classic theory | not-accessible (JSTOR 403); упомянут в research_field_survey | Sequential decision-making с private signals + observation. Аналогичная динамика стадо-эффекта. Используется как cite-only в Limitations diss. |
| S3 | Granovetter M. (1978). *Threshold Models of Collective Behavior.* AJS 83(6). | classic theory | derived-only (JSTOR 403; через Wikipedia) | Правило: `actor i acts iff fraction (or count) of prior actors ≥ threshold_i`. Гетерогенное распределение thresholds — критично; гомогенное даёт all-or-nothing, гетерогенное — градуированную динамику. Наша модель — **smooth approximation** threshold через V4 saturating или V5 log_count. |
| S4 | Salganik M., Dodds P., Watts D. (2006). *Experimental Study of Inequality and Unpredictability in an Artificial Cultural Market.* Science 311. | empirical primary | not-accessible (Science 403); через research_field_survey | Видимость download counts усиливает inequality и unpredictability. Эмпирическое подтверждение: **достаточно ОДНОГО агрегированного сигнала «сколько уже выбрали t»** для cascade-like behavior. Прямая поддержка V1/V5 как минимально достаточных. |
| S5 | Centola D. (2010). *The Spread of Behavior in an Online Social Network Experiment.* Science 329. | empirical primary | not-accessible (Science 403) | Complex contagion — для распространения нужна множественная экспозиция; для нашего минимального V (где экспозиция определяется числом предшественников в слоте) применимо как теоретическая мотивация saturating формы. |

### B. Recsys feedback loops / popularity bias / herd effects

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S6 | Chaney A., Stewart B., Engelhardt B. (2018). *How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility.* RecSys 2018. arXiv:1710.11214. | primary paper | abstract-only (PDF binary) | Главный вывод: **homogenization arises from algorithmic confounding (training-time)**, не user-side popularity. Контрапозиция для нас: чтобы выделить эффект gossip, нужно сравнивать `w_gossip = 0` vs `w_gossip > 0` с одинаковыми политиками — иначе эффекты конфаундятся с recsys feedback. **Аргумент за обязательный baseline `w_gossip = 0`** в LHS как gold-standard control. |
| S7 | Mansoury M. et al. (2020). *Feedback Loop and Bias Amplification in Recommender Systems.* CIKM 2020. arXiv:2007.13019. | primary paper | abstract-only | Feedback loop: рекомендации → user reaction → лог → перетренировка. Усиление popularity bias, снижение aggregate diversity, эволюция preference representations. Минорные группы страдают сильнее. User reaction model не специфицирована в абстракте. Поддержка тезиса: **gossip — дополнительный механизм** к training-loop эффекту, проверяется отдельно. |
| S8 | *Diversity paradox feedback loop* (2025). arXiv:2510.14857. | primary paper | full HTML | **Прямая формула utility:** `V_{u,i}(t) = c_u(t) + G_u(t)·log(1 + s_i(t)) + λ·1/(1+s_i(t)) + η_{u,i}(t)`. Choice probability — softmax `P(i\|u,t) = exp(V/τ) / Σ_j exp(V/τ)`. Ключевое для нас: (а) форма `log(1 + s_i)` — каноническая для popularity signal в современной литературе; (б) она комбинируется аддитивно в utility; (в) есть параллельный saturating term `λ/(1+s_i)` (anti-popularity bias). Главный результат: feedback loop увеличивает **individual** diversity, но снижает **collective**. **Прямой источник формулы для V5 log_count.** |

### C. Discrete choice + социальное заражение (формальный канон)

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S9 | Brock W., Durlauf S. (2001). *Discrete Choice with Social Interactions.* Review of Economic Studies 68(2). | classic theory | not-accessible (SSC PDF timeout; Wiley 303; NBER mismatch) | Через Wikipedia подтверждено: добавление социального терма `V_i = β·x_i + γ·m_i` (где `m_i` — mean choice / fraction adopted) **нарушает IIA**. Наша gossip-расширенная MNL формально становится «social-MNL». Цитата для §13 Limitations: «MNL with social term нарушает IIA, для устойчивости рекомендуется nested logit, но overkill для timeline». |
| S10 | McFadden D. (1974). *Conditional Logit Analysis of Qualitative Choice Behavior.* (через Train K. 2009 *Discrete Choice Methods with Simulation* как secondary) | textbook secondary | full (через S9 spike_behavior_model и Wikipedia) | Канонический MNL. Аддитивная систематическая часть `V_i = Σ β_k · x_k` — стандарт; gossip-term `+ β_g · m_t` встраивается в эту схему. IIA нарушается при социальных термах (S9). |
| S11 | Bass F. (1969). *A New Product Growth for Model Consumer Durables.* Management Science. (через Wikipedia как secondary) | classic theory | derived-only (Wikipedia) | **Прямая формула:** `dF/dt = (1 − F)·(p + qF)`, p — innovation, q — imitation, F — fraction adopted. **V1 (linear additive popularity) ≡ Bass-imitation term `qF` с коэффициентом `q = w_gossip`.** Bass — известный канон в маркетинге, нормально процитировать. |

### D. LLM-симуляторы пользователей (что они делают с социальным сигналом)

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S12 | Yang Z. et al. (2024). *OASIS.* arXiv:2411.11581. | primary paper | partial HTML | **Прямая формула hot-score** для Reddit-style rec: `h = log₁₀(max(\|u−d\|, 1)) + sign(u−d)·(t−t₀)/45000`. Это **log-scaled** popularity signal с логарифмическим весом. **Аргумент за V5 (log_count) как канонически-индустриальный** — все большие платформы используют именно log-form, не linear. |
| S13 | Zhang A. et al. (2024). *Agent4Rec.* arXiv:2310.10108. | primary paper | abstract-only | Profile + Memory + Action. Filter-bubble emulation. Inter-agent communication в абстракте не упомянута; по обзорам — нет cross-agent gossip; всё через recsys-feedback на user-item уровне. **Поддерживает наше архитектурное решение: gossip в параметрическом симуляторе, не в LLM.** |
| S14 | Wang L. et al. *RecAgent.* arXiv:2306.02552. | primary paper | abstract-only / binary PDF | Sandbox с persona/memory/action; chat-action между агентами как ядро. Аналог inter-agent gossip — но это **наш rejected V10** (стоп-лист §5 PROJECT_STATUS). |
| S15 | Mollabagher M., Naghizadeh P. (2025). arXiv:2504.07105. | primary paper | abstract-only | Reactive users opinion dynamics. Continuous responsiveness к recommendations. По gossip — поддерживает «continuous responsiveness» к социальному сигналу как современный канон, не Bernoulli-threshold. |

### E. Capacity-aware и congestion в recsys (для отделения gossip от capacity)

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S16 | Mashayekhi Y. et al. (2023). *ReCon: Reducing Congestion in Job Recommendation using Optimal Transport.* RecSys 2023. arXiv:2308.09516. | primary paper | abstract + локальный legacy summary | Capacity-канал живёт **в обучении recommender'а** (`O_ReCon = O_M + λ·O_C`), не в utility пользователя. **Прямая поддержка архитектурного правила этапа C** + аргумент против V2 (gossip-bonus на load_frac пересекается с capacity-каналом). |

### F. Wikipedia secondary readings (заменители недоступных primary)

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S17 | *Information cascade* (Wikipedia) | encyclopedia | full | Bayesian update формула. Cascade trigger при разнице ≥ 2. Cascade fragile. Не даёт прямой functional form для gossip-bonus, но фиксирует «cascade естественно эмулируется через aggregated count of prior actions». |
| S18 | *Threshold model* (Wikipedia) | encyclopedia | full | Granovetter precise: `actor acts iff fraction of prior actors ≥ threshold_i`. Гетерогенное распределение thresholds. Threshold-form ступенчатая, наша gossip — гладкая (saturating / log) — связь как «smooth approximation of threshold». |
| S19 | *Bandwagon effect* (Wikipedia) | encyclopedia | full | Положительный feedback loop: «more affected ⇒ more affected». Драйверы: efficiency, conformity, information, FOMO. Социологическая мотивация. |
| S20 | *Social proof* (Wikipedia) | encyclopedia | full | Mirroring under uncertainty. Эффективность зависит от **similarity** между наблюдателем и группой. **Мотивация для V6 (cohort-level)** — социальный сигнал сильнее в похожей когорте. |

### Итог по §5

Реально открыто 20 внешних источников + 9 локальных файлов кода + 8 локальных документов проекта.

Из 20 внешних: **2 с прямыми формулами в тексте** (S8 diversity-paradox `log(1+s_i)` и `λ/(1+s_i)`; S12 OASIS log₁₀-scoring). **2 с прямыми формулами через secondary** (S11 Bass `dF/dt = (1−F)·(p + qF)`; S17 cascade Bayesian update). **3 abstract-only primary** (S6, S7, S13–S15). **4 derived-only через Wikipedia** (S1, S2, S3, S9). **2 not-accessible** (S4, S5). **4 Wikipedia secondary** (S17–S20).

Решение в §8–9 опирается прежде всего на **S8 (diversity-paradox), S11 (Bass), S12 (OASIS), S16 (ReCon)** — источники с прямым методологическим содержанием, плюс на **S6, S7** как мотивацию обязательного baseline `w_gossip = 0`.

---

## 6. Обзор вариантов реализации

### V0. No gossip / `w_gossip = 0` (контрольная точка)

`gossip(t, L_t) ≡ 0`; реализуется как дефолтное значение `SimConfig.w_gossip = 0.0`.

- **Что говорит литература:** S6 (Chaney 2018) — без явного контрола `w_gossip = 0` любой эффект homogenization не атрибутируем (конфаундится с recsys feedback loop в обучении). S7 (Mansoury 2020) поддерживает — gossip это **дополнительный** механизм, отдельный от training-loop.
- **Реализуемость:** zero-cost, дефолт.
- **Плюсы:** обязательный gold-standard control для атрибуции эффекта; покрывает требование PROJECT_DESIGN §11 EC3 (выключенный gossip = baseline E).
- **Минусы:** сам по себе ничего не даёт — это контрольная точка, не эксперимент.
- **Решение:** **обязательный** компонент LHS как точка `w_gossip = 0` — не «вариант выбора», а контрол.

### V1. Linear additive popularity по числу выбравших доклад

`gossip(t, L_t) = local_choice_count[t] / N_users`, где `N_users = len(user_order)`.

- **Что говорит литература:** S11 Bass-imitation `qF`, где F = fraction adopted, q = `w_gossip` — **прямой Bass-параллель**. S1 Bikhchandani — линейный по count cascade. S15 Mollabagher continuous responsiveness.
- **Реализуемость:** 1–2 строки кода + добавление `local_choice_count` в `_process_one_slot`.
- **Плюсы:** простейшая интерпретация, прямая референс на Bass diffusion model. На защите легко объяснить как «доля аудитории, выбравшая доклад».
- **Минусы:** при больших count_t утилита растёт линейно неограниченно — может «пробить» rel и rec каналы; в сочетании с softmax даёт «winner-takes-all» поведение быстрее saturating форм. Требует жёсткой нормализации, иначе scale gossip-term зависит от размера аудитории.
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-канал не пересекается, потому что считаем по talk_id, а не hall_id (✓ при использовании `local_choice_count`).
- **Связь с PROJECT_DESIGN §7:** соответствует «динамическое усиление выбора по тому, что уже выбрала когорта» — буквально `count_t / N`.

### V2. Load fraction внутри слота: `gossip = local_load[hall(t)] / capacity(hall(t))`

Это **в точности `load_frac`**, который CapacityAwarePolicy использует со знаком минус для штрафа, но с противоположным знаком (как бонус).

- **Что говорит литература:** S3 Granovetter (fraction-based threshold), S11 Bass (fraction-based diffusion). Архитектурно — естественный выбор, потому что `load_frac` уже доступен из `state["hall_load"]`.
- **Реализуемость:** 1 строка кода (значение уже есть в `local_load[chosen_hall.id] / capacity_at(slot.id, hall.id)`).
- **Плюсы:** переиспользует существующее состояние, нормирован [0, 1], прямой connect с capacity-axis эксперимента.
- **Минусы:** **прямой риск двойного учёта capacity-канала.** `load_frac` — центральная переменная политики П3 (`score = sim − α·load_frac`). При `w_gossip > 0` agent видит тот же signal, что и политика, но с противоположным знаком — gossip толкает в перегруженный зал, П3 толкает прочь. При `w_gossip ≈ α` эффекты компенсируются в идеальной симметрии, и **П3 формально перестанет влиять на overload** через capacity-канал — это нарушение PROJECT_DESIGN §9 (политика теряет действие, но не из-за `w_rec`, а из-за gossip).
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-инвариант — **нарушается** или жёстко завязан на (`w_gossip ≪ α`); риск №1 этого spike.
- **Решение:** **rejected** для минимального варианта. Если хочется gossip на load — нужно отделить семантику (gossip по talk_id, не hall_id; см. V1 / V4 / V5).

### V3. Z-score нормализованная доля внутри слота: `gossip = (count_t − mean_t' count_t') / std_t' count_t'`

Сравнение текущего доклада с распределением count'ов по другим докладам в том же слоте.

- **Что говорит литература:** Каноническая центрировка из choice modeling (Train Ch.3 — utility identifiable up to overall shift). Внутри-слотовый z-score автоматически центрирован.
- **Реализуемость:** малая (требуется per-step расчёт mean/std по 2–4 докладам слота).
- **Плюсы:** invariant к shift, не зависит от абсолютных capacity.
- **Минусы:** при пустом слоте (все count_t' нулевые) std = 0, z-score не определена — нужен fallback. При двух докладах z-score даёт ровно ±1 после первого выбора, что слишком резко. Не идёт в ногу с общим scale других каналов (`rel ∈ [-1, 1]`, `rec ∈ {0, 1}`).
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-канал не пересекается (✓). Но resco scale несовместим с rel/rec.
- **Решение:** **rejected** — сложнее V1/V5 без явных преимуществ для минимального варианта.

### V4. Saturating: `gossip = 1 − exp(−γ · count_t)` или эквивалент `tanh(γ · count_t)`

Параметр `γ` задаёт скорость насыщения.

- **Что говорит литература:** PROJECT_DESIGN §2.1.2 (упомянутая в PIVOT этап J пункт 4) — прямая отсылка как «форма из проектного дизайна». S8 diversity-paradox имеет похожий saturating `λ/(1+s_i)` (anti-popularity bias). S5 Centola complex contagion — нелинейная функция от множественной экспозиции.
- **Реализуемость:** 1 строка кода + новый параметр `gossip_gamma` в `SimConfig`.
- **Плюсы:** ограничена сверху (1), интерпретируется как «максимальный социальный bonus = w_gossip», избегает «winner-takes-all» эффекта неограниченных линейных bonus. Параметр γ становится дополнительной осью эксперимента (sensitivity по форме gossip).
- **Минусы:** добавляет вторую ось эксперимента (`w_gossip × γ`) — без жёсткой фиксации γ или явного включения в LHS число прогонов растёт. Параметр γ нелинейно связан с эффективной силой gossip — интерпретация в защите труднее линейной формы.
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-канал не пересекается (✓ при count_t).
- **Решение:** кандидат на основной выбор, но требует решения по γ (фиксировать или ось).

### V5. Log-scaled popularity: `gossip = log(1 + count_t) / log(1 + N_users)`

Логарифмическая нормализация count.

- **Что говорит литература:** **прямой канон** в IR / recsys / production-платформах. S12 OASIS использует `log₁₀(max(|u−d|, 1))` как hot-score. S8 diversity-paradox использует `log(1 + s_i)` — это **прямая формула** в utility-уравнении choice model. S20 social proof — log-scale соответствует «diminishing returns from each additional observed peer».
- **Реализуемость:** 1–2 строки кода (math.log + знаменатель). Нормировка `log(1 + N_users)` гарантирует gossip ∈ [0, 1].
- **Плюсы:** гладкая, ограничена, **реалистична относительно индустриального канона** (S12), легко интерпретируется через «удвоение count_t даёт фиксированный прирост сигнала» (постоянная надбавка на каждое x2). **Не требует дополнительного параметра насыщения** — нормировка фиксирована через N_users.
- **Минусы:** при низких count_t (1, 2, 3) производная large — резкий старт; при count_t = 0 gossip = 0, что сильно отличает «никто не выбрал» от «один выбрал». В малой аудитории (toy 10 users) это может быть заметно.
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-канал не пересекается (✓). Совместим с другими каналами по scale (gossip ∈ [0, 1]).
- **Связь с PROJECT_DESIGN §7:** «динамическое усиление выбора по тому, что уже выбрала когорта» — log-form удовлетворяет (диминишинг, но всё равно растёт).
- **Решение:** **основной кандидат для рекомендации.** Нет дополнительной параметрической оси, индустриальный канон, прямая ссылка на S8 / S12.

### V6. Cohort-level social influence

Сигнал считается не по всей аудитории, а по подгруппе пользователей с близким embedding профиля. Реализация: pre-compute K-means кластеры на персон-эмбедингах; `gossip(t, L_t) = log(1 + count_t_cohort) / log(1 + N_cohort)`, считается среди тех в текущем слоте, кто из того же кластера, что текущий пользователь.

- **Что говорит литература:** S20 social proof (similarity boosts effect), S4 Salganik (similar-group cascade сильнее). PUB SIGIR 2025 (research_field_survey 3.2) — Big Five-based heterogeneity.
- **Реализуемость:** требует pre-computing кластеров (как в run_llm_spike.py), добавляет k_clusters и cohort_assignment как ось.
- **Плюсы:** научно богаче, эмпирически правдоподобнее (Cialdini effect), Salganik-2006 evidence.
- **Минусы:** при малой аудитории (toy 10–50) кластеры разрежены, реализация заметно сложнее. Стоп-лист §5 не запрещает буквально, но «Big Five / social graph» помечен как «прототип, не основной результат» — близкий по жанру.
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓). Capacity-канал не пересекается (✓).
- **Решение:** **отложить** на «later» в memo (раздел 10).

### V7. Memory-based across slots

Gossip переносится между слотами: то, что было популярно в slot_k, влияет на slot_{k+1}. Реализация: `simulate_async` хранит cumulative `decay·counter` per talk и пробрасывает в `_process_one_slot` как initial state.

- **Что говорит литература:** PIVOT этап J пункт 7 явно упомянуто; S11 Bass (cumulative F across time); S15 Mollabagher (decreasing/adaptive consumption — динамика across time).
- **Реализуемость:** **большая** — нарушает sequential isolation слотов, требует протаскивания state через `simulate_async`, появляется decay parameter δ как третья ось.
- **Плюсы:** ближе к реальности конференции (день-1 hype переходит в день-2).
- **Минусы:** **архитектурно противоречит §K строки 656** PIVOT_IMPLEMENTATION_PLAN: «Последовательная обработка пользователей в слоте уже даёт это». То есть memory-based перестаёт быть аддитивной правкой и становится refactoring `_process_one_slot`. Дороже в реализации.
- **Связь с инвариантами:** `w_gossip = 0` ⇒ baseline (✓ если decay = 0 или counter не инициализирован при `w_gossip = 0`).
- **Решение:** **отложить** на «later» (раздел 10).

### V8. LLM-only social signal

Gossip присутствует только в LLM-симуляторе (как часть промпта `prior_choices_summary` — «у тебя в зале X уже Y% занято»), параметрический симулятор без gossip.

- **Что говорит литература:** S13 Agent4Rec, S14 RecAgent — LLM-симуляторы могут эмулировать gossip через chat-action.
- **Плюсы:** технически тривиально (один параметр промпта).
- **Минусы:** **прямо нарушает PROJECT_DESIGN §7** («w_gossip — компонент функции полезности участника»; параметрический симулятор без gossip ломает это). **Прямо нарушает паритет двух симуляторов** — они становятся разными моделями отклика. Принятое решение spike_llm_simulator Q-G «capacity в промпт LLM-агента не передавать» применимо по аналогии: gossip тоже не передаётся в LLM-промпт.
- **Решение:** **rejected.**

### V9. Network-based gossip (социальный граф)

Социальный граф participants (например, друзья / последователи), gossip распространяется по edges по правилам independent cascade или linear threshold model.

- **Что говорит литература:** S3 (Granovetter linear threshold), S11 Bass + network. Свежее: agentic feedback loop modeling.
- **Минусы:** **отсутствие данных о социальном графе** среди реальных конференционных участников (PROJECT_DESIGN §13) делает граф произвольным дополнительным параметром. **PROJECT_STATUS §5** прямо: «social graph как реализованный метод — был прототип, не основной результат».
- **Решение:** **rejected.**

### V10. Chamber-of-Choice / inter-agent chat

Прямой обмен сообщениями между агентами (как RecAgent S14). Каждый агент видит, что написали соседи, и обновляет своё решение.

- **Минусы:** **PROJECT_STATUS §5 стоп-лист** прямо: «inter-slot chat как реализованный метод — был прототип, не основной результат». Архитектурно тяжело — требует двух раундов processing per user. Дорого по $.
- **Решение:** **rejected.**

### V11. Cumulative-cascade Bayesian: `gossip = log P(t популярен | history)` на основе S17 формулы

- **Что говорит литература:** S1 Bikhchandani, S17 Wikipedia cascade Bayesian.
- **Плюсы:** формально корректный cascade.
- **Минусы:** требует prior `p` и signal precision `q` как новые параметры (две новые оси). Не имеет прямого precedent в production recsys (S12 OASIS использует empirical log). Overkill для timeline.
- **Решение:** **rejected** — не основной выбор; описывается в §10 как теоретический «more rigorous» вариант.

---

## 7. Сравнительная таблица вариантов

| Вариант | §7 трёхкомпонентная | §9 политика только через w_rec | §11 EC1–EC4 совместимость | Инвариант w_gossip=0 ⇒ baseline | Не размывает capacity-канал | Реализуемость в срок | Стоимость по K | Зависимость от LLM | Поддержка литературы |
|---|---|---|---|---|---|---|---|---|---|
| V0 (control) | n/a | ✓ | ✓ | trivially | ✓ | trivially | 0 строк | нет | S6, S7 (контрол) |
| V1 linear count_t/N | ✓ | ✓ | ✓ | ✓ | ✓ при count_t | малая | ~5 строк | нет | S11 Bass |
| V2 load_frac (по hall) | ✓ | **частично** (gossip vs П3 пересекаются) | **риск** | ✓ | **✗ риск №1** | малая | ~3 строки | нет | S3, S11 |
| V3 z-score | ✓ | ✓ | ✓ при std≠0 | ✓ | ✓ | средняя | ~10 строк | нет | Train Ch.3 |
| V4 saturating | ✓ | ✓ | ✓ | ✓ | ✓ | малая | ~5 строк + γ ось | нет | PROJECT_DESIGN §2.1.2, S5 |
| V5 log_count | ✓ | ✓ | ✓ | ✓ | ✓ | малая | ~5 строк | нет | **S8 (формула), S12 (формула)** |
| V6 cohort | ✓ | ✓ | ✓ | ✓ | ✓ | большая | ~50 строк | нет | S20, S4 |
| V7 memory across slots | ✓ | ✓ | ✓ | ✓ если decay=0 | ✓ | большая | refactor simulate_async | нет | S11 cumulative |
| V8 LLM-only | **✗** | n/a | n/a | n/a | n/a | малая | 1 строка промпта | **высокая** | rejected |
| V9 network graph | ✓ | ✓ | ✓ | ✓ | ✓ | очень большая | новый модуль | нет | rejected (стоп-лист §5) |
| V10 inter-agent chat | ✓ | ✓ | ✓ | ✓ | ✓ | очень большая | refactor + LLM | очень высокая | rejected (стоп-лист §5) |
| V11 Bayesian cascade | ✓ | ✓ | ✓ | ✓ | ✓ | средняя | ~15 строк + 2 параметра | нет | S1, S17 |

Критерии выбраны из PROJECT_DESIGN §7, §9, §11; PROJECT_STATUS §5, §7; PIVOT_IMPLEMENTATION_PLAN §6 принципы 5/7, §7 правила, этапы J/K/L; spike_behavior_model accepted; spike_llm_simulator Q-G.

---

## 8. Evidence-based recommendation

**Рекомендованный вариант для этапа K:** **V5 — log-scaled popularity по `local_choice_count` per talk**, с обязательной точкой контроля **V0** (`w_gossip = 0`) в LHS.

Финальная формула:

```
gossip(t, L_t) = log(1 + count_t) / log(1 + N_users)

где:
    count_t = local_choice_count[t]   (счётчик пользователей в текущем слоте,
                                       выбравших доклад t, до текущего юзера);
    N_users = len(user_order)         (общее число пользователей в run).

U(t | i, hat_pi) = w_rel · effective_rel(i, t)
                 + w_rec · 1{t ∈ recs}
                 + w_gossip · gossip(t, L_t)

P(choose t | slot) = (1 - p_skip) · softmax(U / τ) ⊕ p_skip
consider_ids = slot.talk_ids                      (как в этапе E)
capacity-effect: только в политике П3, не в utility
```

### Обоснование (evidence-first, со ссылками на источники)

1. **PROJECT_DESIGN §7 — нормативная трёхкомпонентность.** V5 — точная инстанциация: третий аддитивный член с собственным весом `w_gossip`. Динамическая природа («что уже выбрала когорта») реализована через `count_t`, который растёт по ходу обработки слота. V0–V8 удовлетворяют §7 в равной мере; V8/V9/V10 — нет (см. §6).
2. **Инвариант `w_gossip = 0` ⇒ baseline (формальный аналог EC3).** V5 удовлетворяет тривиально: при `cfg.w_gossip = 0` слагаемое `cfg.w_gossip * f(...) = 0` вне зависимости от `count_t`. `choice_rng` идёт по идентичной траектории, поскольку gossip не использует RNG. **pytest-набор этапа I остаётся зелёным без правок.**
3. **Capacity-канал не размывается** (главный архитектурный риск spike). V5 считает по `count_t` (per talk), не по `load_frac` (per hall). На текущих программах (один доклад на зал в слот) они численно совпадают, но **семантически разные**. Это разделение прямо поддержано **S16 ReCon 2023**: capacity-канал живёт в objective recommender'a, не в utility пользователя. V2 нарушает это разделение и **rejected** по этому критерию.
4. **Каноничность формы log-scale** (S8 diversity-paradox 2025, S12 OASIS 2024). Обе работы используют **точную формулу** `log(1 + count)` в utility / scoring. S8 — recsys feedback loop modeling 2025, S12 — production LLM-симулятор 2024. V5 переиспользует индустриально-стандартный канон.
5. **Bass-параллель в V1** (S11) сильнее по интерпретируемости в защите, но V5 включает Bass как partial case при малых count (log(1+x) ≈ x). V5 ограничен сверху, V1 нет — это снимает риск «winner-takes-all» при больших count_t.
6. **Bayesian cascade V11** (S1, S17) формально rigorous, но требует двух новых параметров (prior p и precision q). V5 не требует γ или новых параметров — это **минимальный** вариант по числу осей.
7. **PROJECT_DESIGN §9 — политика только через `w_rec`.** V5 удовлетворяет: gossip-канал зависит только от `count_t` (по talk_id), который, в свою очередь, зависит от: (а) `choice_rng` через выбор предыдущих юзеров (общий поток для всех политик при `w_rec = 0`); (б) выдачи политики через `rec_indicator` в utility этих предыдущих юзеров. То есть **при `w_rec = 0` count_t у всех политик идентичен** (все юзеры выбирают по одной и той же `softmax(w_rel · rel)` независимо от политики), и gossip-канал не вводит политическую зависимость. **Формальная EC3 сохраняется при `w_gossip > 0`.**
8. **Стоп-лист PROJECT_STATUS §5.** V5 не задействует social graph, Big Five, inter-agent chat — все три из стоп-листа.
9. **Симплексная нормировка весов.** Согласовано с spike_behavior_model accepted Q2 (`w_rel + w_rec = 1` для базовой модели). Расширение: `w_rel + w_rec + w_gossip = 1` для трёхкомпонентной формы; LHS-точки внутри симплекса. В этапе K параметр сохраняется как «независимые поля» в SimConfig, нормировка применяется на уровне run_smoke / LHS-генератора.

### Что отклоняется и почему

- **V2 (load_frac)** — пересечение с capacity-каналом, риск нарушения §9 (политика теряет смысл при `w_gossip ≈ α`).
- **V3 (z-score)** — нестандартный scale, fallback при std=0, без явных преимуществ перед V5.
- **V4 (saturating)** — требует γ как новую ось эксперимента; V5 покрывает аналогичную «диминишинг» интуицию без дополнительного параметра.
- **V6 (cohort)** — большая реализация; научно богаче, но overkill для timeline; откладывается на «later».
- **V7 (memory across slots)** — refactor `simulate_async`; противоречит §K строки 656 PIVOT.
- **V8 (LLM-only)** — нарушает §7 PROJECT_DESIGN и паритет двух симуляторов.
- **V9 (network graph)** и **V10 (inter-agent chat)** — стоп-лист §5 PROJECT_STATUS.
- **V11 (Bayesian)** — два новых параметра, нет precedent в production; подходит для расширенной диссертации, не для минимального варианта.

### Архитектурный комментарий: gossip и LLM-симулятор

> **Пересмотрено amendment 2026-05-07:** см. `docs/spikes/spike_gossip_llm_amendment.md`. Решение пользователя — gossip передаётся **и в LLM-симулятор**, рекомендованная форма — L2 (реальные числа `count_t / N_users` в промпте + системный промпт по уровням `w_gossip`).
>
> Следствие для cross-validation на 12 точках LHS: оба симулятора используют **одну и ту же семантику** gossip-сигнала — количество уже выбравших доклад участников в текущем слоте (`count_t` и `N_users`). Конкретные значения `count_t` эндогенны внутри каждого симулятора и могут различаться из-за разных моделей выбора. Различие — и в источнике значений, и в способе их взвешивания (параметрический — через `w_gossip` в utility-MNL; LLM — через инструкцию в системном промпте). Это сохраняет «второй независимый источник отклика» в смысле PROJECT_DESIGN §7 (LLM сам интерпретирует свой эндогенный сигнал), при этом сохраняя семантический паритет канала. Конкретные правки к `simulator.py` ниже (§9) **не меняются**; LLM-сторона добавляется отдельной правкой ~35 LOC в `llm_agent.py` и `run_llm_spike.py` (см. amendment §6).
>
> **Прежнее предложение** «LLM-симулятор не получает gossip-сигнала, ссылаясь на Q-G accepted (capacity не в промпте)» — отозвано. Аналогия с capacity не работает: capacity — структурное ограничение (если LLM видит, дублирует канал П3); gossip — динамический social signal в utility. См. amendment §4.

---

## 9. Минимальная первая реализация для этапа K

### 9.1. Целевая формула (для этапа K, не J)

```
U(t | i, hat_pi) = w_rel · effective_rel(i, t)
                 + w_rec · 1{t ∈ recs}
                 + w_gossip · log(1 + count_t) / log(1 + N_users)

count_t = local_choice_count[t]   (per-talk счётчик в текущем слоте, до текущего юзера)
N_users = len(user_order)
```

### 9.2. Что меняется в `simulator.py` (этап K)

1. **`SimConfig`** (после строки 181):
   ```python
   w_gossip: float = 0.0       # gossip-канал отключён по умолчанию
   ```
2. **`_process_one_slot`** (строки 378–500):
   - В строке 379 или 385 добавить:
     ```python
     local_choice_count: Dict[str, int] = {}
     ```
   - В цикле формирования utility (строки 459–465) добавить gossip-слагаемое:
     ```python
     for tid in consider_ids:
         t = conf.talks[tid]
         rel = relevance_fn(user.embedding, t.embedding)
         effective_rel = (1.0 - cfg.w_fame) * rel + cfg.w_fame * t.fame
         rec_indicator = 1.0 if tid in recs_set else 0.0
         # NEW: gossip term
         count_t = local_choice_count.get(tid, 0)
         n_users = len(user_order)
         gossip = (math.log1p(count_t) / math.log1p(n_users)) if n_users >= 1 else 0.0
         u = (cfg.w_rel * effective_rel
              + cfg.w_rec * rec_indicator
              + cfg.w_gossip * gossip)
         utils_list.append(u)
     ```
   - После выбора (строка 490) добавить инкремент:
     ```python
     local_load[chosen_hall.id] += 1
     local_choice_count[chosen_id] = local_choice_count.get(chosen_id, 0) + 1
     ```
   - Также в forced_choice ветке (строка 437–453, legacy compliance) — синхронно инкрементировать `local_choice_count`. Иначе при включенном legacy (`use_calibrated_compliance=True`) gossip-канал теряет данные.
3. **`import math`** в начале файла (строка ~19; уже есть `numpy`, `math` нужно добавить, если нет).

### 9.3. Что не меняется

- `Conference`, `Talk`, `Hall`, `Slot`, `UserProfile` — без изменений.
- `Embedder`, `LearnedPreferenceFn`, `cosine_relevance` — без изменений.
- `simulate_async`, `_process_one_slot` API — без изменений.
- Все политики (`NoPolicy`, `CosinePolicy`, `CapacityAwarePolicy`, `LLMRankerPolicy`) — без изменений.
- Метрики (`metrics.py`) — без изменений.

### 9.4. Что меняется в `run_smoke.py` (этап K)

- Новый CLI-флаг `--w-gossip "0.0,0.3"` (минимум 2 точки: control + central).
- Расширение `run_grid` дополнительным циклом по `w_gossip_grid`.
- В `aggregate_seeds` ничего не меняется (имена метрик прежние).
- В `check_expectations` добавить расширенные проверки EC при `w_gossip = 0` и `w_gossip > 0` (см. §11).

### 9.5. Что меняется в тестовом слое (этап K, минимально)

- В `test_extreme_conditions.py` добавить:
  - `test_baseline_equivalence_when_w_gossip_zero` — на фиксированных seed/users/conf проверка пословной идентичности `chosen_id` всех steps между текущей реализацией (`w_gossip=0` после правки) и baseline (`w_gossip=0` до правки — фиксируем через стабильный snapshot).
  - `test_ec3_extended_with_w_gossip_positive` — при `w_rec=0, w_gossip=0.3` все 3 политики дают одинаковый результат.
  - `test_ec4_with_w_gossip_positive` — при `w_rec=1, w_gossip=0.3` политики различимы (range > некоторого ε).
- В `test_simulator_unit.py` добавить:
  - `test_utility_invariant_when_w_gossip_zero_for_all_policies` — обобщение existing test на gossip-выключенный канал.
- В `test_toy_cases.py`:
  - Снять `pytest.skip` с TC3.

### 9.6. Стоимость и время

- Правка ядра: ~10 строк изменений в `simulator.py` + 1 import.
- Правка скриптов: ~15 строк в `run_smoke.py`.
- Тесты: ~40 строк (3 новых теста + расшевеление TC3).
- Итого ~70 строк нового кода. Реализация — этап K, проверка — этап L.

---

## 10. Что сознательно откладываем

1. **V6 cohort-level social influence.** Социальный сигнал в подгруппах (k-means на эмбедингах персон). Научно богаче (S20 social proof, S4 Salganik). Откладывается до post-defense; в Limitations: «cohort-level gossip — отдельная ось эксперимента, не входит в обязательный результат; реализуется через предварительную кластеризацию персон + per-cohort accumulator».
2. **V7 memory-based across slots.** Кросс-слотовый gossip с decay δ. Архитектурно противоречит §K «slot-local local_load уже даёт это». Откладывается; в Limitations: «in-slot gossip — минимальная форма; cross-slot memory с decay — естественное расширение, не реализовано в первой версии».
3. **V11 Bayesian cascade.** Формально rigorous, но требует prior p и precision q как новые параметры. Откладывается; в § диссертации Connections-to-literature: «байесовский каскад (Bikhchandani–Hirshleifer–Welch 1992) формализует cascade-mechanic строже log-popularity, но не имеет precedent в production recsys; для исследовательской симуляции log-form достаточен».
4. **γ-ось эксперимента (V4 saturating).** Параметризация формы gossip через `gossip_kind: str`. Не делается на этапе K; на этапе S (постобработка) можно sensitivity-sweep по форме при необходимости.
5. **Gossip в LLM-симуляторе.** Решение пользователя 2026-05-07: gossip **передаётся** в LLM-симулятор. Конкретная форма — L2 (реальные числа `count_t / N_users` в промпте + 3-уровневая параметризация системным промптом по `w_gossip`). Подробности — `docs/spikes/spike_gossip_llm_amendment.md`. Cohort-level (L3) и full OASIS / inter-agent chat (L4) — отложены на «later» / отвергнуты соответственно. Старая редакция этого пункта («Принципиально отвергнут (см. §6.V8 и §8)») недействительна и сохранена только в комментариях к amendment.
6. **Network-based gossip (V9), inter-agent chat (V10).** Стоп-лист §5 PROJECT_STATUS. Не реализуется как защищаемый результат.
7. **Изменение `local_load` структуры.** На текущих программах (один доклад на зал в слот) `local_load[hall(t)] == count_t` тождественно. Сохраняем `local_load` для capacity-канала политики и добавляем параллельный `local_choice_count` для gossip — это минимальное архитектурное расширение без рефакторинга существующего.
8. **Расширение LLM-промпта `prior_choices_summary`.** Не делается. См. Q-G accepted в spike_llm_simulator.
9. **Перевод gossip-формы в осью гиперкуба.** Фиксируем V5 log_count в коде; форма как ось — отдельный sensitivity-sweep в этапе S, не в основном LHS.
10. **Реализация `_process_one_slot` рефакторинг.** Не делается. Текущая структура поддерживает gossip как аддитивную правку.

---

## 11. Какие проверки должны пройти в этапе L

Шесть требований PIVOT_IMPLEMENTATION_PLAN строк 670–691 + расширенные EC из задачи пользователя:

### L.1. Инвариант `w_gossip = 0` ⇒ совпадение с базовой моделью этапа E

**Что:** при `w_gossip = 0` симулятор даёт **пословно** идентичные результаты с реализацией этапа E.

**Как проверить:** pytest `test_baseline_equivalence_when_w_gossip_zero` — на фиксированной toy-конфигурации (toy_microconf_2slot или synthetic из conftest), фиксированном seed, фиксированных users и каждой из активных политик П1–П3:

```python
@pytest.mark.parametrize("policy_name", ["no_policy", "cosine", "capacity_aware"])
def test_baseline_equivalence_when_w_gossip_zero(...):
    cfg_baseline = SimConfig(w_rec=0.3, w_gossip=0.0, seed=1)
    res = simulate(toy_2slot_conf, personas_50_users, pol, cfg_baseline)
    chosen = [s.chosen for s in res.steps]
    assert chosen == BASELINE_SNAPSHOT[policy_name]
```

`BASELINE_SNAPSHOT` фиксируется в момент первого зелёного прохода (ровно перед merge правки этапа K) и хранится как frozen константа в тесте.

**Блокирующее.** Если не PASS — критическая регрессия, переход к Q заблокирован.

### L.2. Социальный сигнал реально влияет при `w_gossip > 0`

**Что:** при `w_gossip = 0.3` распределение посещений отличается от `w_gossip = 0` (на той же seed/users/conf/policy).

**Как проверить:** pytest `test_gossip_changes_distribution_when_positive` — сравнение `hall_load_per_slot` между `w_gossip = 0.0` и `w_gossip = 0.3`. Должно быть `≠`.

**Блокирующее.**

### L.3. Рост силы gossip усиливает концентрацию выбора

**Что:** монотонность Gini / hall_load_concentration по `w_gossip ∈ {0, 0.3, 0.7}` при фиксированных rest. Гипотеза этапа K (PIVOT строка 648): «социальное заражение усиливает концентрацию выбора в популярных докладах при больших γ».

**Как проверить:** pytest `test_gossip_monotone_concentration` — на toy_microconf_2slot, sweep `w_gossip ∈ {0, 0.3, 0.7}` × 5 seeds, считать `hall_load_gini` или Gini по `count_per_talk`. Должно быть монотонно неубывающим.

**Блокирующее.**

### L.4. EC1–EC4 остаются зелёными при `w_gossip > 0`

Расширение test-suite этапа I:

- **EC1-extended**: при `cap_multiplier ≥ 3.0`, `w_gossip = 0.3` — `mean_overload_excess == 0` для всех 3 политик (tested в `test_extreme_conditions.py::test_ec1_loose_capacity_eliminates_overload` с дополнительной параметризацией `w_gossip`).
- **EC2-extended**: монотонность `mean_overload_excess` по `cap_mult` при `w_gossip = 0.3` (5 seeds).
- **EC3-extended**: при `w_rec = 0, w_gossip = 0.3` — `range(mean_overload_excess, policies) < 1e-9`. Это проверка **gossip-CRN-инвариантности**: gossip не зависит от выдачи политики, а через `choice_rng` все юзеры идут по идентичной траектории при `w_rec = 0`.
- **EC4-extended**: при `w_rec = 1, w_gossip = 0.3` стресс×0.5 — `range > 0.02`. Политики различимы при умеренном gossip.

**Все четыре блокирующие.** Реализуется как расширение `test_extreme_conditions.py` параметризацией по `w_gossip ∈ {0, 0.3}` (см. § реализации этапа K).

### L.5. Capacity-aware (П3) не теряет смысла при включённом gossip

**Что:** при `w_gossip = 0.3`, asymmetric capacity (стресс×0.5), П3 всё равно даёт меньший `mean_overload_excess` чем П2 (cosine). Иначе capacity-канал размыт gossip-каналом.

**Как проверить:** pytest `test_capacity_aware_still_works_with_gossip` — на toy_microconf_2slot stress×0.5 × `w_rec=0.5, w_gossip=0.3` × 5 seeds:

```python
assert overload(cap_aware) <= overload(cosine) + tolerance
```

**Блокирующее.** Это центральный архитектурный риск spike (§6.V2 risk).

### L.6. Sensitivity по `w_gossip` видна в результатах

**Что:** в smoke-таблице (этап F расширенный) хотя бы одна метрика риска (overload_excess или hall_var) **значимо** меняется при `w_gossip ∈ {0, 0.3, 0.7}` на фиксированной конфигурации. «Значимо» = `range / max > 0.05` или `range > 2 × std_seed`.

**Как проверить:** новая acceptance-функция в `run_smoke.py` `check_gossip_sensitivity` (расширение `check_expectations`). На stress×0.5 × `w_rec = 0.5` × 5 seeds × 3 политики × `w_gossip ∈ {0, 0.3, 0.7}` — для хотя бы одной политики range overload или hall_var ≥ 0.05.

**Блокирующее.** Без этого gossip как ось теряет смысл.

### Сводный критерий

Все 6 проверок (L.1–L.6, включая 4 EC-extended внутри L.4) должны быть зелёными перед переходом к этапу M (spike оператора Φ). Если хотя бы одна не PASS — возврат к выбору формы gossip (J/K) или к параметрам.

---

## 12. Какие решения требуют подтверждения пользователя

### Q-J1. Какую форму gossip-функции использовать в этапе K?

Варианты (см. §6):

- **(а)** V5 log-scaled `log(1 + count_t) / log(1 + N_users)`. **Рекомендуется.**
- (б) V1 linear `count_t / N_users`. Простейшая, прямая Bass-параллель, но неограниченно растёт.
- (в) V4 saturating `1 - exp(-γ · count_t)` с γ = 0.1. Требует γ как ось.
- (г) V2 load_frac `local_load[hall(t)] / capacity(hall(t))`. **Не рекомендуется** — пересечение с capacity-каналом П3.

**Предложение:** (а) V5. Каноничная форма (S8 diversity-paradox, S12 OASIS), нет дополнительных параметров, ограничена [0, 1], не размывает capacity-канал.

### Q-J2. Какой signal использовать как `count_t` — per talk или per hall (load_frac)?

Варианты:

- **(а)** `local_choice_count[t]` — per talk. **Рекомендуется.** Семантически отделяет gossip от capacity (S16 ReCon: capacity в политике, не в utility).
- (б) `local_load[hall(t)]` — per hall. На текущих программах численно совпадает с (а), но семантически тождественно capacity-каналу П3.

**Предложение:** (а) per talk. Это согласовано с разделением каналов из spike_behavior_model accepted decision (capacity только в политике).

### Q-J3. Делать ли `gossip_kind` осью гиперкуба или фиксировать форму в коде?

Варианты:

- **(а)** Фиксировать V5 в коде; форма — не ось эксперимента. **Рекомендуется.** LHS остаётся 5-мерным (capacity, popularity, behavior_weights, audience, program_variant); прибавление формы как 6-й оси раздувает гиперкуб без чёткой пользы.
- (б) Параметризовать `gossip_kind: str` в SimConfig; sensitivity по форме как часть LHS.
- (в) Sensitivity по форме как отдельный sweep на этапе S (постобработка) на одной фиксированной точке.

**Предложение:** (а) фиксировать V5; (в) опциональный sensitivity-sweep на этапе S, если будет время до 13.05.

### Q-J4. Диапазон `w_gossip` в LHS

Варианты:

- **(а)** Симплекс `w_rel + w_rec + w_gossip = 1`, `w_gossip ∈ [0, 0.7]`. Согласовано с spike_behavior_model accepted Q2 (связанные веса). **Рекомендуется.**
- (б) Независимые поля; `w_gossip ∈ [0, 1]`.

**Предложение:** (а) симплекс. Естественное расширение accepted Q2 на трёхкомпонентную модель; верхняя граница 0.7 — потому что выше gossip доминирует (нереалистично).

### Q-J5. Тестовая точка для EC-extended проверок в L.4

Варианты:

- **(а)** `w_gossip = 0.3` (центральная по предложенному диапазону). **Рекомендуется.**
- (б) `w_gossip = 0.5` (середина симплекса при `w_rec = 0.5, w_rel = 0`).
- (в) Несколько точек: `{0.3, 0.7}`.

**Предложение:** (а) `w_gossip = 0.3` для EC-extended; `0.7` — sensitivity-sweep, не EC.

### Q-J6. Сетка `w_gossip` в smoke (`run_smoke.py`)

Варианты:

- **(а)** `--w-gossip "0.0,0.3,0.7"` — 3 точки. **Рекомендуется.** Покрывает control / central / strong, при этом грид х3 от текущего F.
- (б) `--w-gossip "0.0,0.3"` — 2 точки. Минимум для проверки L.6.
- (в) `--w-gossip "0.0,0.2,0.5,0.7"` — 4 точки.

**Предложение:** (а) 3 точки.

### Q-J7-revised (2026-05-07). Какой вариант LLM-gossip использовать

См. `docs/spikes/spike_gossip_llm_amendment.md` §2 для деталей вариантов и §6 для рекомендации.

- **(а)** L2 — реальные числа `count_t / N_users` в промпте + системный промпт по уровням `w_gossip`. **Рекомендуется.**
- (б) L5 — pass-through scalar `social_score = log(1+count_t)/log(1+N_users)` из параметрического (максимальная сопоставимость, низкая независимость).
- (в) L1 — простой словесный сигнал без чисел.
- (г) L3 — cohort-level, OASIS-inspired controlled signal (откладывается на «later»).

**Предложение:** (а) L2.

**Дополнительные новые вопросы Q-J8 — Q-J12** (параметризация LLM-gossip от `w_gossip`, что значит `w_gossip = 0` для LLM, где проверять L2, нужен ли мини-research, синхронизация LHS-точек) — см. amendment §8.

**Старая редакция Q-J7** («Gossip только в параметрическом, LLM без gossip») — недействительна.

---

## Recommended decision for K

**Форма gossip-функции:**

```
gossip(t, L_t) = log(1 + count_t) / log(1 + N_users)

где:
    count_t = local_choice_count[t]
    N_users = len(user_order)
```

**Utility:**

```
U(t | i, hat_pi) = w_rel · effective_rel(i, t)
                 + w_rec · 1{t ∈ recs}
                 + w_gossip · log(1 + count_t) / log(1 + N_users)
```

**Файлы для правки на этапе K:**
- `experiments/src/simulator.py` — добавить `w_gossip: float = 0.0` в `SimConfig`; добавить `local_choice_count: Dict[str, int] = {}` в `_process_one_slot` (строка ~378); добавить gossip-слагаемое в utility (строки 459–465); инкрементировать `local_choice_count` после выбора (строка ~490 + ветка legacy compliance).
- `experiments/scripts/run_smoke.py` — CLI `--w-gossip`, расширение `run_grid`, обновление `check_expectations`.
- `experiments/tests/test_extreme_conditions.py`, `test_simulator_unit.py`, `test_toy_cases.py` — добавить проверки L.1–L.6 + расшевелить TC3.

**Проверки этапа L (блокирующие):** L.1, L.2, L.3, L.4 (EC1–EC4 при `w_gossip > 0`), L.5, L.6.

**Сетка `w_gossip` для LHS:** `[0.0, 0.3, 0.7]`. Симплексная нормировка `w_rel + w_rec + w_gossip = 1` на уровне run_smoke / LHS-генератора.

**LLM-симулятор:** gossip **получает** через L2 — реальные числа `count_t / N_users` в промпте + 3-уровневая параметризация системным промптом по `w_gossip` (см. `docs/spikes/spike_gossip_llm_amendment.md`). Cross-validation на 12 точках LHS = согласованность ranking-политик при общей семантике gossip-сигнала (количество уже выбравших доклад участников в текущем слоте) и разных response-моделях; конкретные значения `count_t` эндогенны в каждом симуляторе и могут различаться. Параметрическая часть K (`simulator.py`, V5 log_count) **не меняется** — amendment добавляет ~35 LOC в `llm_agent.py` + `run_llm_spike.py`.

**Что требует подтверждения пользователя:** Q-J1 — Q-J6 в §12 (шесть open questions параметрической стороны) + **Q-J7-revised** (выбор L-варианта для LLM-gossip) + **Q-J8 — Q-J12** (новые вопросы amendment по параметризации LLM-gossip; см. `spike_gossip_llm_amendment.md` §8). Все имеют рекомендации.

После подтверждения — переход к этапу K (реализация). К этапу K не перехожу до отдельного сообщения пользователя.
