# Design-spike: оператор локальных модификаций программы Φ (этап M)

Дата: 2026-05-07
Этап: M (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: accepted by user 2026-05-07 with one terminological clarification (см. раздел Accepted decision); design-spike, evidence-first; кода не меняет, экспериментов не запускает, к этапу N не переходит до отдельного сообщения.

> Memo evidence-first. Сначала research log и реально изученные источники, затем требования и обзор вариантов, и только в конце — рекомендация и описание минимальной первой версии для этапа N. Структура повторяет принятые ранее `spike_behavior_model.md` (этап C), `spike_llm_simulator.md` (этап G), `spike_gossip.md` (этап J).

---

## Accepted decision

Статус: принято пользователем для перехода к этапу N с одним терминологическим уточнением (2026-05-07).

### Терминологическое уточнение (2026-05-07)

В исходной редакции memo местами использовалась формулировка «swap между **параллельными слотами** одного дня» (заимствована из PROJECT_DESIGN §7). По фактической структуре данных `Slot` — это **временной** слот, внутри которого уже находятся параллельные доклады по разным залам. Поэтому корректная фиксация выбранного оператора:

> **«pairwise swap двух докладов между разными временными слотами одного дня».**

Не «между параллельными слотами одного дня» (это терминологически некорректно — параллельны не слоты, а доклады/залы внутри одного timeslot); и **не** swap между залами внутри одного slot.

Точечная семантика V1:
- меняем `slot_id` у двух докладов;
- `hall` НЕ меняем;
- состав докладов сохраняется;
- размеры слотов сохраняются;
- speaker-conflict проверяется после swap;
- `P_0` входит как `program_variant = 0`;
- модификации `P_1..P_k` генерируются in-memory через `SwapDescriptor`;
- `same_day_only = True` по умолчанию;
- `k_max = 5` для LHS-50, `k_max = 3` для LLM-12.

Цитата из PROJECT_DESIGN §7 («обменивает пары докладов между параллельными слотами одного дня») сохранена в memo как нормативный текст; ниже §3 фиксируется, что в нашей терминологии это означает «между разными временными слотами одного дня».

### Wording про отсутствие speakers (2026-05-07)

Раньше в memo встречалось «нет speakers → конфликтов нет тривиально». Корректная формулировка:

> **«speaker-validation становится no-op, потому что данных о спикерах нет».**

Это важно для интерпретации: отсутствие конфликта — следствие отсутствия информации, не свойство программы. Для Mobius / Demo Day, где `speakers` есть, validation реально проверяет; для toy / ITC / Meetup validation no-op (не «зелёный свет», а «информации недостаточно»).

### Подтверждённые решения по open questions раздела 12

1. **Q-M1.** Форма Φ — **V1 + V5 + V0**: pairwise swap двух докладов между разными временными слотами одного дня + random subsample до `k_max` штук + `P_0` как control.
2. **Q-M2.** При отсутствии `speakers` в JSON — `speakers=[]`; speaker-validation становится no-op.
3. **Q-M3.** `k_max = 5` для LHS-50, `k_max = 3` для LLM-12.
4. **Q-M4.** Только внутри дня; `same_day_only = True` по умолчанию.
5. **Q-M5.** `P_0` как `program_variant = 0` (control-точка LHS).
6. **Q-M6.** In-memory descriptors; LHS-row хранит `(phi_seed, k_index, swap_descriptor)`. Отдельных JSON-файлов на диск не пишем.
7. **Q-M7.** Расширить `Talk.speakers: List[str] = field(default_factory=list)` напрямую в dataclass + парсинг в `Conference.load`.
8. **Q-M8.** Создать `experiments/data/conferences/toy_speaker_conflict.json` для TC5 (минимальный fixture: 2 слота × 2 зала × 4 талка, один speaker в двух слотах).

После перехода к этапу N эти решения становятся обязательными для реализации.

---

## 1. Проблема

Нужно зафиксировать форму оператора локальных модификаций программы Φ (PROJECT_DESIGN §7 «Модуль локальных модификаций программы») для первой работающей реализации. Решение — вход в этап N (создание `experiments/src/program_modification.py`); проверка — этапы N + Q (вход варианта программы в LHS как ось 5). Без выбора формы нельзя ни запустить N, ни прогнать LHS / Q.

Цель memo — выбрать **минимальную проверяемую** форму, удовлетворяющую трём ограничениям из PROJECT_DESIGN:

1. **Состав докладов фиксирован.** Talks-set не меняется; меняются только привязки `talk → (slot, hall)`. PROJECT_DESIGN §5 (строка 48): «состав докладов фиксирован».
2. **Конфликты спикеров не допускаются.** PROJECT_DESIGN §5 + §7 (строка 100): «конфликты спикеров не допускаются».
3. **Локальные перестановки**, не полная оптимизация. PROJECT_DESIGN §13 (строка 224): «Полная задача оптимизации программы (целочисленное программирование на уровне всей решётки слотов и залов) к работе не относится; она разработана в существующей литературе по conference scheduling (Vangerven 2018, Pylyavskyy 2024, Bulhões 2022)».

Дополнительно: оператор Φ — **не оптимизатор расписания**, а инструмент стресс-теста. Он генерирует набор $\{P_0, P_1, \dots, P_{k_\max}\}$ для оси 5 LHS; численный анализ эффекта локальных модификаций — отдельная задача этапов Q/S.

Без решения по форме Φ:
- этап N (реализация) заблокирован;
- ось 5 в LHS-протоколе вырождается в константу `P_0`;
- защищаемое положение №4 («количественные оценки … локальных модификаций программы», §16.4) теряет содержание;
- сравнение исходной программы с модификациями (PROJECT_DESIGN §6 «выходные данные», группа 5 §10) технически невозможно.

---

## 2. Текущая реализация в репозитории

### 2.1. Класс `Talk` и загрузчик `Conference.load`

Файл: `experiments/src/simulator.py`, строки 36–46:

```python
@dataclass
class Talk:
    id: str
    title: str
    hall: int
    slot_id: str
    category: str
    abstract: str
    embedding: np.ndarray
    fame: float = 0.0
```

**Главное:** в dataclass `Talk` поля `speaker` / `speakers` НЕТ. Привязка доклада к (зал, слот) — через `hall: int` и `slot_id: str`.

`Conference.load(prog_path, emb_path, fame_path)` (строки 84–141): читает `prog["talks"]`, оставляет только заявленные поля dataclass (`id, title, hall, slot_id, category, abstract`) и подмешивает embedding + fame. **Поле `speakers` JSON-а игнорируется при загрузке**, даже если оно физически есть в источнике (Mobius/Demo Day; см. §2.3).

`Slot.talk_ids` строится как обратный индекс по `slot_id` (строки 124–127). `Conference.capacity_at(slot_id, hall_id)` (строки 72–82) возвращает per-slot override либо глобальный `Hall.capacity`.

### 2.2. Прецедент in-memory модификации Conference

`experiments/scripts/run_smoke.py`, строки 108–119:

```python
def scale_capacity(conf: Conference, mult: float) -> Conference:
    cloned = copy.deepcopy(conf)
    for h in cloned.halls.values():
        h.capacity = max(1, int(round(h.capacity * mult)))
    for s in cloned.slots:
        if s.hall_capacities:
            s.hall_capacities = {
                hid: max(1, int(round(c * mult)))
                for hid, c in s.hall_capacities.items()
            }
    return cloned
```

Это работающий шаблон: `copy.deepcopy(conf) → правка → возврат нового Conference`. **Φ должен следовать тому же паттерну** — без записи на диск, без mutex'а, без рефакторинга политик.

### 2.3. Фактические поля JSON-программ

Инспекция через `python3 -c "import json; ..."` восьми датасетов в `experiments/data/conferences/`:

| conference | поля talks | n_talks / n_halls / n_slots | есть `speakers`? |
|---|---|---|---|
| `mobius_2025_autumn.json` | `id, title, speakers, hall, date, start_time, end_time, category, abstract, slot_id` | 40 / 3 / 16 | **да** (строка ФИО через `, `) |
| `demo_day_2026.json` | то же | 210 / 7 / 56 | **да** |
| `toy_microconf.json` | `id, title, hall, slot_id, category, abstract` | 2 / 2 / 1 | нет |
| `toy_microconf_2slot.json` | то же | 4 / 2 / 2 | нет |
| `itc2007_t1_exam_comp_set1.json` | `id, title, hall, slot_id, category, abstract, fame` | 600 / 7 / 54 | нет |
| `itc2019_bet_spr18.json` | то же | 600 / 29 / 3 | нет |
| `itc2019_mary_fal18.json` | то же | 600 / 44 / 7 | нет |
| `meetup_rsvp.json` | то же | 381 / 225 / 171 | нет |

**Главный вывод (важная корректировка PIVOT R3):** R3 в `PIVOT_IMPLEMENTATION_PLAN.md` (строка 1051: «В данных Mobius/Demo Day нет явного `speakers` — Φ не проверишь — Зафиксировать прокси в memo M; запасной вариант — мягкое ограничение по category») — **устарело**. В Mobius и Demo Day поле `speakers` физически присутствует в JSON; проблема в том, что `Conference.load` его не читает в `Talk` dataclass. Прокси-канал по category не нужен. Достаточно расширить `Talk.speakers` и обновить загрузчик.

**Реальное состояние Mobius по спикерам** (из subagent inspection): 47 уникальных лиц на 40 талков; 5 спикеров в ≥2 талках (Юрий Дубовой — 3, остальные — по 2). В исходной программе Mobius 0/16 слотов имеют конфликт. **Demo Day:** 210 талков, 313 уникальных спикеров (включая ко-авторов), 5 в ≥2 талках. Это даёт реалистичный test-bed.

**Toy / ITC / Meetup:** поле `speakers` в JSON отсутствует → `Talk.speakers = []` после загрузки → speaker-validation становится **no-op** (не «нет конфликта», а «данных о спикерах нет»). Φ применяется без speaker-проверок.

### 2.4. Старые попытки оператора Φ

Subagent выполнил `grep -rE "(permute|swap_talk|swap.*hall|swap.*slot|modify_schedule|program_modification)"` по `experiments/` и `experiments/scripts/_legacy_scripts/` — **нет совпадений**. Реализация Φ — с чистого листа, без переиспользования legacy.

`experiments/scripts/patch_capacities.py` — это admin-скрипт пере-вычисления `hall_capacities` под целевую популяцию N, запись в JSON на диск. Не релевантно для in-memory Φ.

### 2.5. Что не нужно трогать

- Все политики (`policies/*.py`) — `BasePolicy.__call__(*, user, slot, conf, state)` принимает `Conference` целиком; политике не нужно знать, исходная это программа или $P_k$.
- Симулятор (`simulator.py:_process_one_slot`, `simulate_async`) — не трогается.
- Метрики (`metrics.py`) — работают по слотам, не зависят от того, как $P_k$ сгенерирована.
- Активный реестр П1–П4 (`registry.py`) — без изменений.

---

## 3. Требования из PROJECT_DESIGN / PROJECT_STATUS / PIVOT_IMPLEMENTATION_PLAN

### Из PROJECT_DESIGN

- **§5 Постановка задачи** (строка 48): «оператор $\Phi$ выдаёт набор допустимых вариантов программы $\{P_0, P_1, \dots\}$ при двух ограничениях: состав докладов фиксирован, конфликты спикеров не допускаются».
- **§7 Модуль локальных модификаций программы** (строка 100): «Реализует оператор $\Phi$. Оператор обменивает пары докладов между параллельными слотами одного дня; состав докладов фиксирован, конфликты спикеров не допускаются. Полученные варианты программы поступают в симулятор так же, как исходная программа.» — **прямо указывает pairwise swap внутри одного дня** как референсную форму.

  **Терминологическое уточнение** (Accepted decision 2026-05-07): по фактической структуре данных `Slot` — это **временной** слот, внутри которого уже находятся параллельные доклады по разным залам. Поэтому в нашей реализации «параллельные слоты одного дня» из PROJECT_DESIGN §7 практически реализуется как «**разные временные слоты одного дня**». Не как swap между залами внутри одного `slot_id`. Меняем `slot_id` двух талков, `hall` не трогаем.
- **§8 Параметрические оси** (строка 126): «Ось 5 — вариант программы. Исходная программа $P_0$ или одна из модификаций $P_k$, полученных оператором $\Phi$ при фиксированном бюджете локальных перестановок. Реализуется как индекс варианта в множестве $\{P_0, P_1, \dots, P_k\}$.» — фраза «фиксированный бюджет локальных перестановок» намекает, что $P_k$ может быть результатом нескольких swap'ов, хотя §7 говорит про одну пару. Это open question (см. §12 Q-M5).
- **§10 Группа 5. Эффект модификации программы** (строка 170): «Относительное снижение показателей риска от перестановки относительно исходной программы при фиксированной политике и фиксированной конфигурации». То есть $P_k$ должна быть сравнима с $P_0$ по той же метрике в той же конфигурации.
- **§13 Допущения** (строка 224): «Модификации программы рассматриваются как локальные перестановки в окрестности исходной программы. Полная задача оптимизации программы … к работе не относится». Жёсткий out-of-scope для V7 / V8.
- **§16 Положения, выносимые на защиту, пункт 4**: «Получены количественные оценки относительного эффекта политик рекомендаций и локальных модификаций программы по показателям риска перегрузки залов, баланса нагрузки и релевантности выбора». Без работающего Φ этот пункт не доказывается.

### Из PROJECT_STATUS

- **§5 Стоп-лист**: о Φ ничего напрямую — но в духе §5 Φ не должен превратиться в «полноценный schedule optimizer».
- **§7 Текущее направление**: «MMR и gossip-вход — параметрические модификаторы» — Φ тоже параметрический модификатор уровня данных.
- **§11 Следующий практический шаг** (строки 144–146): «1. Латинский гиперкуб … 2. Оператор локальных перестановок программы $\Phi$ с проверкой конфликтов спикеров. 3. Ось «вариант программы» в плане эксперимента.» — прямое требование implement Φ перед LHS.

### Из PIVOT_IMPLEMENTATION_PLAN r5

- **Раздел 7** правила design-spike (строки 230–342): обязательный research log, source-tracking, recommendation с минимальной первой реализацией.
- **§6 принцип 9**: design-spike обязателен перед методически рискованными компонентами; Φ — один из перечисленных (вторая половина списка).
- **Этап M** (строки 695–718): предложенный набор вариантов (V1 = pairwise swap внутри дня — рекомендован; V5 = random sample; V7 = жадная оптимизация — отвергнут; V8 = IP — отвергнут).
- **Этап N** (строки 727–745): артефакт — `experiments/src/program_modification.py` с функцией `enumerate_modifications(P0, k_max, rng)`. Юнит-тест TC5 на синтетический speaker-конфликт.
- **R3** (строка 1051): устаревшая запись (см. §2.3 выше) — поле `speakers` есть, прокси по category не нужен.

### Из spike_behavior_model.md / spike_gossip.md / spike_gossip_llm_amendment.md

- Φ **не должен** трогать utility-формулу (`U = w_rel·rel + w_rec·rec + w_gossip·gossip`). Изменяется только `Talk.slot_id` (и опционально `Talk.hall`, см. §6 V2).
- Φ **не должен** менять контракт `Slot.talk_ids` — он только перестраивается из обновлённых `Talk.slot_id` после swap.
- Φ **не должен** влиять на capacity-канал политики П3 — `local_load[hall_id]` остаётся в семантике слот×зал.
- gossip-канал зависит от `count_t` per talk; talk-id не меняется, поэтому gossip-канал работает на $P_k$ так же, как на $P_0$.

**Вывод:** Φ ортогонален всем уже принятым спайкам. Никаких пересмотров C/G/J/K/L не требуется.

---

## 4. Research log

Расширенный design-spike по правилу §7 PIVOT_IMPLEMENTATION_PLAN r5 выполнен через отдельный research-subagent с time-boxed бюджетом.

### 4.1. Время

- **start time (subagent):** epoch `1778170750`, ~2026-05-07 22:19 MSK.
- **end time (subagent):** epoch `1778171056`, ~2026-05-07 22:24 MSK.
- **elapsed seconds:** 306 (≥ 300 — минимальный research budget по §7).
- Sleep / искусственное ожидание не использовались. Время потрачено на реальное I/O: Read локальных файлов + WebFetch внешних источников. Subagent был явно проинструктирован минимум 8 успешных WebFetch с реальными цитатами; выполнено **12 успешных WebFetch** (см. §5).

### 4.2. Изученные файлы кода

- `experiments/src/simulator.py` — классы `Talk` / `Hall` / `Slot` / `Conference` (строки 36–141), `Conference.load` — точное место расширения для `speakers`.
- `experiments/src/policies/base.py` — контракт BasePolicy, не требует изменений.
- `experiments/scripts/run_smoke.py` — `scale_capacity` (строки 108–119) как прецедент in-memory deepcopy.
- `experiments/scripts/build_demo_day_dataset.py` — как создаётся `demo_day_2026.json` со speakers.
- `experiments/scripts/patch_capacities.py` — admin-скрипт, не Φ.
- `experiments/scripts/_legacy_scripts/*.py` — пустой grep по permute/swap/modify_schedule.
- `experiments/data/conferences/*.json` — 8 датасетов, инспекция полей через `python3 -c`.

### 4.3. Изученные документы проекта

- `PROJECT_DESIGN.md` — §5, §7, §8, §10, §13, §16.
- `PROJECT_STATUS.md` — §5, §7, §11.
- `PIVOT_IMPLEMENTATION_PLAN.md` — §6, §7, §9 (этапы M/N), §10 (TC5), R3.
- `docs/spikes/spike_behavior_model.md` — accepted decisions.
- `docs/spikes/spike_gossip.md` + amendment.

### 4.4. Локальные research-файлы

- `.claude/memory/research_field_survey_2026-05-04.md` § 2.4 (Vangerven, Rezaeinia, Pylyavskyy/Kheiri/Jacko, Bulhões, Stidsen 2018, **Manda 2019** — ближайший open-source аналог, CoSPLib 2025).
- `.claude/memory/reference_validation_defense.md` — методическая рамка Sargent / Kleijnen.
- `materials/_legacy/research-conference-recsys-deep-2026-05.md` — нет Φ-релевантных деталей.
- `thesis/bibliography.bib` — пуст по conference scheduling.

### 4.5. Внешние источники

12 успешных WebFetch с реальными цитатами — см. таблицу §5.

### 4.6. Что оказалось нерелевантным

- **Full integer programming** (Vangerven 2018, Pylyavskyy 2024, Bulhões 2022) — out-of-scope §13.
- **Tabu search / hyper-heuristics** (Pylyavskyy/Kheiri DASA 2024) — overkill.
- **Kempe-chain neighborhood** — стандартный timetabling-neighborhood, но избыточно для V1 (см. §6 V6).
- **Classroom assignment / course allocation** (Budish, Kornbluth) — другая задача.
- **Manda 2019** (PeerJ topic-model + simulated annealing) — методически близко, но 403 при попытке открыть PDF; конкретные swap-операторы не извлечены.
- **2-opt / TSP analogy** — полезно как vocabulary, не алгоритмическая основа.

### 4.7. Открытые вопросы

См. §12 Q-M1 — Q-M6.

---

## 5. Обзор реально изученных источников

12 внешних источников с реальными цитатами, плюс 8 локальных JSON + 5+ локальных файлов кода / документов.

### A. Conference scheduling (специальная литература)

| № | Источник | Тип | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S1 | Vangerven et al. (2018). *Conference scheduling — A personalized approach.* PATAT 2016 preprint. URL: https://www.patatconference.org/patat2016/files/proceedings/paper_30.pdf | primary paper | full (PDF text extracted) | «To the best of our knowledge, our approach is the first to deal with session hopping. ... used an assignment based formulation which features 11 variables ... The model assigns every block to a timeslot, minimizing speaker availability violations.» | Three-phase IP для MAPSP 2015. Полная оптимизация — out-of-scope §13. **Speaker availability** у них soft (objective term), у нас — hard. Разница архитектурная, фиксируется в Limitations. |
| S2 | Bulhões et al. (2022). *Optimization Online preprint.* URL: https://optimization-online.org/2020/08/7988/ | primary paper | abstract | «basic yet sufficiently general version of the problem that aims at maximizing the benefit of clustering papers with common topics in the same session, while leaving the particularities of the event to be addressed by means of side constraints.» | Branch-and-cut / B&P для NP-hard clustering. Cite-only. |
| S3 | Pylyavskyy, Jacko, Kheiri (2024) EJOR. URL: https://ideas.repec.org/a/eee/ejores/v317y2024i2p487-499.html | primary paper | abstract | «We present a penalty system that allows organisers to set up scheduling preferences for tracks and submissions regarding sessions and rooms. ... Two integer programming models are presented: an exact model and an extended model.» | Generic IP framework; **penalty system** — soft-constraints. Подкрепляет, что speaker-related ограничения в литературе обычно мягкие; у нас — жёсткие. |
| S4 | Pylyavskyy/Kheiri/Jacko (DASA 2024) Manchester research portal. URL: https://research.manchester.ac.uk/en/publications/exact-and-hyper-heuristic-methods-for-solving-the-conference-sche | primary paper | abstract | «relaxed integer programming model, a decomposed two-phase matheuristic solution approach, and a selection hyper-heuristic algorithm ... extended formulations of previously proposed mathematical models to handle constraints at the time slot level» | Hyper-heuristic + exact IP. Не наш путь, методическая опора. |
| S5 | PATAT 2024 proceedings. URL: https://patat.cs.kuleuven.be/patat-conferences/patat24/proceedings | full (page) | full | «Adaptive Large Neighborhood Search is an invited talk by Stefan Røpke ... covers local search methodologies relevant to timetabling problems»; «Metaheuristic optimization of Danish High-School Timetables» | Свежее (2024) подтверждение, что swap / large-neighborhood-search активно используется в timetabling. |
| S6 | CoSPLib README (GitHub). URL: https://github.com/ahmedkheiri/CoSPLib | full | full | «The Conference Scheduler is an advanced tool designed to optimise the process of scheduling conferences ... »; instance fields include «Presenters - Author names, comma-separated (optional)» | **Прямое подтверждение**: канонический способ хранить спикеров в conference scheduling — строка ФИО через запятую. Совпадает с тем, как это уже сделано в Mobius / Demo Day. |

### B. Local search / pairwise swap / neighborhood structures

| № | Источник | Тип | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S7 | Wikipedia: *Local search (optimization).* URL: https://en.wikipedia.org/wiki/Local_search_(optimization) | encyclopedia | full | «In computer science, local search is a heuristic method for solving computationally hard optimization problems.»; «local optimization with neighborhoods that involve changing up to k components of the solution is often referred to as k-opt» | Базовое определение local search. Подтверждает терминологию «k-opt» / «pairwise swap» как стандартную. |
| S8 | Wikipedia: *Tabu search.* URL: https://en.wikipedia.org/wiki/Tabu_search | encyclopedia | full | «Tabu search (TS) is a metaheuristic search method employing local search methods used for mathematical optimization.» | Tabu — расширение local search памятью; в нашей постановке избыточно (нам не нужен поиск оптимума, только перечисление допустимых модификаций). |
| S9 | Wikipedia: *Simulated annealing.* URL: https://en.wikipedia.org/wiki/Simulated_annealing | encyclopedia | full | «Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function.»; «The neighbors of any state are the set of permutations produced by swapping any two of these cities.» | Подтверждает swap как стандартный neighbor-оператор. Мы НЕ ищем оптимум — задача sweep, не optimisation. |
| S10 | Wikipedia: *Kempe chain.* URL: https://en.wikipedia.org/wiki/Kempe_chain | encyclopedia | full | «Kempe chains have also been used as neighborhood operators in local search algorithms for problems related to graph coloring, including timetabling and seating allocation.» | Прямое подтверждение: Kempe-chain — стандартный timetabling-neighborhood. Слишком сложный для V1 (см. §6 V6). |
| S11 | Wikipedia: *2-opt.* URL: https://en.wikipedia.org/wiki/2-opt | encyclopedia | full | «A 2-opt operation involves removing 2 edges and adding 2 different edges. ... 2-opt is a simple local search algorithm for solving the traveling salesman problem.» | 2-opt — каноническая «pairwise swap». В timetable-постановке вырождается в swap двух assignments. |

### C. Vangerven extracted text (продолжение)

| № | Источник | Тип | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S12 | Vangerven et al. (тот же PDF, p.2 problem statement) | primary paper | full | «Typically, a conference schedule groups presentations into sessions»; «it is the responsibility of the organizers to develop a schedule that allows participants to attend the presentations of their interest» | Описание проблемы. У нас сессия = (slot, hall). |

### Не открыты напрямую (компенсация)

ScienceDirect (Burke-Petrovic 2002, Carter-Laporte 1996), TandFonline (Rezaeinia 2024), PeerJ (Manda 2019), Cardiff zonal mirror Lewis 2008 PDF — все недоступны через WebFetch (403/404). Компенсация: WebFetch успешно открыл Wikipedia, arXiv, GitHub, Optimization-Online, IDEAS-RePEc, PATAT proceedings (12 источников), что покрыло терминологию (S7, S10, S11), методический канон (S1, S5), penalty-vs-hard разницу (S3), формат данных (S6).

### Итог по §5

Реально открыто 12 внешних источников (с цитатами), 8 локальных JSON-программ (через `python3 -c`), 5+ локальных файлов кода/документов проекта.

Решение в §8–9 опирается прежде всего на:
- **S1 Vangerven 2018** — подтверждение out-of-scope для full IP;
- **S6 CoSPLib** — формат `speakers` как comma-separated string;
- **S7, S9, S11** — терминология pairwise swap / 2-opt / canonical neighborhood;
- **PROJECT_DESIGN §7** — прямое нормативное указание «обменивает пары докладов между параллельными слотами одного дня».

---

## 6. Обзор вариантов реализации

### V0. Не менять программу (P_0 only, control)

`k_max = 0`, ось «вариант программы» в LHS вырождается в константу `P_0`.

- **Что говорит литература:** контроль необходим для атрибуции эффекта (по аналогии с `w_gossip = 0` baseline-control в spike_gossip §6 V0).
- **Реализуемость:** zero-cost.
- **Плюсы:** обязательная контрольная точка для оси 5; в LHS точка `program_variant = 0` всегда есть.
- **Минусы:** сам по себе не закрывает PROJECT_DESIGN §10 группа 5 и §16.4.
- **Решение:** **обязательный** компонент LHS как `program_variant = 0` — не «вариант выбора», а control. **P_0 включается**.

### V1. Pairwise swap двух докладов между РАЗНЫМИ ВРЕМЕННЫМИ слотами одного дня

Нормативная форма из PROJECT_DESIGN §7 (с терминологическим уточнением — см. §3 и Accepted decision: «параллельные слоты одного дня» в §7 практически реализуется как «разные временные слоты одного дня»). Алгоритм:

1. Выбираем два **временных** слота `s1, s2` одного дня (`s1.datetime[:10] == s2.datetime[:10]`, `s1.id ≠ s2.id`).
2. Выбираем доклад `t1 ∈ s1.talk_ids` и доклад `t2 ∈ s2.talk_ids`.
3. Меняем `slot_id`: `t1.slot_id := s2.id`, `t2.slot_id := s1.id`. **`hall` не меняем** — это сознательное разделение каналов: ось 5 (program_variant) → swap между timeslot'ами, ось 1 (capacity) → отдельная.
4. Перестраиваем `Slot.talk_ids` обратным индексом.
5. Валидируем `not has_speaker_conflict(P_k)` для всех слотов.

- **Реализация:** ~50 строк в `experiments/src/program_modification.py`. Прецедент — `scale_capacity` (run_smoke.py:108–119): `copy.deepcopy(conf)` + правка + return.
- **Литературная поддержка:** S7 (k-opt), S9 (swap canonical), S11 (2-opt). PROJECT_DESIGN §7 — прямая нормативная формулировка (с уточнением «временные слоты»).
- **Плюсы:** точное соответствие §7 (в нашей терминологической интерпретации); не ломает `len(slot.talk_ids)`; не ломает invariant #docs; не требует трогать политики; интерпретируется в защите как «one local swap между двумя timeslot'ами одного дня».
- **Минусы:** на одно-слотных программах не активируется (`toy_microconf` имеет 1 timeslot → нет другого слота для swap). Для toy ось 5 вырождается.
- **Связь с §3 инвариантами:** ✓ состав докладов фиксирован; ✓ конфликты валидируются hard; ✓ локальная перестановка.
- **Решение:** **рекомендуется как минимум**.

### V2. Pairwise swap (talk, hall) внутри одного слота

При фиксированном `slot_id` меняем `hall` у двух талков.

- **Минусы:** **НЕ соответствует §7** — там swap между **разными временными слотами** одного дня, а здесь swap залов внутри одного timeslot'а. Если в слоте `n` докладов и `n` залов одинаковой ёмкости, swap по hall'ам тождественно ничего не меняет (на Mobius hall_capacities ровно одинаковы, см. patch_capacities.py). На Mobius эффект V2 = ноль.
- **Решение:** **rejected** как single-form.

### V3. Pairwise swap внутри дня + межслотный сдвиг

V1 + перенос одного талка в свободный слот того же дня (без обмена).

- **Минусы:** «свободный слот» в реальной программе обычно не существует (timeslot обычно полностью заполнен по всем halls); потребуется концепт «вакантного hall-в-slot», которого в текущем `Slot.talk_ids` нет. Большой риск переусложнения.
- **Решение:** **rejected**.

### V4. Block swap: обмен двух блоков по K докладов между слотами одного дня

При K=1 совпадает с V1; при K>1 это уже не «локальная перестановка» в смысле §13; растёт risk speaker-conflict.

- **Решение:** **rejected** как single-form. Опционально как enhancement (V1 + K=2 как одна swap-операция).

### V5. Random sample из множества допустимых V1-перестановок

Это **не альтернатива V1, а способ обузданного применения V1** при больших программах.

- **Алгоритм:** перечислить все candidate-pairs `(slot_a, slot_b, t1, t2)` (полный $O(D \cdot S^2 \cdot N^2)$ для $D$ дней, $S$ слотов в день, $N$ талков в слоте); валидировать speaker-conflict; если `len(pool) > k_max` — `rng.choice(pool, k_max, replace=False)`.
- **Совместимость с PIVOT §9.N (строка 727):** прямо указана сигнатура `enumerate_modifications(P0, k_max, rng)` — это и есть V1+V5.
- **Решение:** **рекомендуется в комбинации с V1** как механизм контроля `k_max`.

### V6. Kempe-chain swap

Расширение V1 — обмен не одной пары, а связанной «цепи» (по аналогии с graph coloring).

- **Литературная поддержка:** S10 Wikipedia подтверждает как стандартный timetabling-neighborhood.
- **Плюсы:** сильнее меняет распределение, при корректной реализации гарантированно не создаёт speaker-conflict.
- **Минусы:** реализация и тесты дороже; формулировка §7 говорит «обменивает пары», не «цепи».
- **Решение:** **отложить** на «later» (раздел 10) как future enhancement.

### V7. Жадная оптимизация / hill-climbing по риску

На каждом шаге выбираем swap, минимизирующий suspected-overload-metric.

- **Минусы:** PROJECT_DESIGN §13 явно отклоняет «полную задачу оптимизации»; PIVOT §9.M строка 708 явно помечает как отклонённый.
- **Решение:** **rejected**.

### V8. IP a la Vangerven 2018 / Pylyavskyy 2024 / Bulhões 2022

Three-phase IP / branch-and-cut / penalty-system.

- **Минусы:** требует solver (Gurobi / CPLEX); PROJECT_DESIGN §13 + PIVOT §9.M строка 707 явно отклоняют.
- **Решение:** **rejected**. Цитируется как related work.

### V9. Capacity-only modification

Модификация только `Hall.capacity` (как `scale_capacity` в run_smoke.py).

- **Минусы:** capacity уже отдельная ось 1 LHS. Объединять её с осью 5 «вариант программы» — концептуальная путаница (это разные оси с разной семантикой).
- **Решение:** **rejected** как форма Φ. Capacity-axis работает отдельно.

---

## 7. Сравнительная таблица вариантов

| Вариант | Соответствие §7 (pairwise swap внутри дня) | §13 (не полная оптимизация) | Состав докладов сохранён | Speaker hard-validation возможна | Реализуемость в срок | Стоимость по N |
|---|---|---|---|---|---|---|
| V0 (control P_0) | n/a | ✓ | ✓ | ✓ (нет swap — состояние не меняется) | trivially | 0 LOC |
| **V1 pairwise swap (slot, day)** | **✓ нормативная форма** | **✓** | **✓** | **✓ через `has_speaker_conflict`** | **малая** | **~50 LOC** |
| V2 swap hall в слоте | ✗ (нарушает §7) | ✓ | ✓ | n/a (slot не меняется) | малая | ~30 LOC, но эффект 0 на Mobius |
| V3 + межслотный сдвиг | частично | ✓ | ✓ | требует дополнительной логики | средняя | ~80 LOC |
| V4 block swap K>1 | частично | граничит с §13 | ✓ | ✓ | средняя | ~70 LOC |
| **V5 random sample поверх V1** | **✓** | **✓** | **✓** | **✓** | **малая (часть V1)** | **~10 LOC поверх V1** |
| V6 Kempe-chain | расширение | граничит | ✓ | сложнее | средняя-большая | ~150 LOC |
| V7 hill-climbing | n/a | ✗ (нарушает §13) | ✓ | ✓ | средняя | ~200 LOC + objective |
| V8 IP / branch-and-cut | n/a | ✗ | ✓ | ✓ | очень большая | новый модуль + solver |
| V9 capacity-only | ✗ (другая ось) | ✓ | ✓ | n/a | trivial (уже есть) | дублирует ось 1 |

Критерии выбраны из PROJECT_DESIGN §5, §7, §13; PROJECT_STATUS §11; PIVOT_IMPLEMENTATION_PLAN §6/§7/§9 этапов M/N.

**Победитель:** **V1 + V5 + V0 (control)** — единственная комбинация, которая (а) соответствует §7 буквально, (б) удовлетворяет §13 явно, (в) реализуется за ~60 LOC в одном модуле без рефакторинга политик / симулятора / тестов этапа I.

---

## 8. Evidence-based recommendation

**Рекомендованная форма для этапа N:** **V1 + V5 + V0** — детерминированное перечисление V1-pairwise-swap внутри дня с random subsampling до `k_max` штук, плюс P_0 как control.

Содержательно:

```
Φ(P_0, k_max, rng, same_day_only=True) -> [P_1, P_2, ..., P_{k'}]
    где k' = min(k_max, |valid_swap_pool|)
    и valid_swap_pool = {(s_a, s_b, t1, t2) : s_a, s_b — разные временные слоты
                                              одного дня (s_a.id ≠ s_b.id,
                                              s_a.datetime[:10] == s_b.datetime[:10]);
                                              t1 ∈ s_a.talk_ids,
                                              t2 ∈ s_b.talk_ids;
                                              swap не создаёт speaker-conflict
                                              ни в одном слоте P_k}
```

P_0 не входит в выдачу `Φ` — она клеится отдельно вызывающим (LHS-генератор), индекс `k = 0` в LHS-row соответствует `P_0`.

### Обоснование (evidence-first)

1. **PROJECT_DESIGN §7 — прямая нормативная формулировка** «обменивает пары докладов между параллельными слотами одного дня» (с терминологическим уточнением 2026-05-07: «параллельные слоты одного дня» в нашей реализации = «разные временные слоты одного дня»). V1 — буквальная инстанциация. V2 нарушает (swap по hall внутри одного timeslot'а, не между timeslot'ами). V3, V4 расширяют в недопустимое.
2. **PROJECT_DESIGN §13 — out-of-scope для full IP.** V7, V8 явно отклонены. Vangerven 2018 (S1), Pylyavskyy 2024 (S3, S4), Bulhões 2022 (S2) цитируются только как related work. Pylyavskyy «penalty system» (S3) — soft-constraints, у нас hard; разница архитектурная фиксируется в Limitations.
3. **Канон local search.** S7 (k-opt), S9 (swap as canonical neighborhood), S11 (2-opt) подтверждают V1 как стандартную форму neighborhood. Kempe-chain (S10) — известное расширение, отложено.
4. **CoSPLib (S6) — формат speakers совпадает с нашим.** Comma-separated string в JSON Mobius/Demo Day — канонический формат de-facto.
5. **Прецедент в репо.** `scale_capacity` (run_smoke.py:108–119) — рабочий шаблон in-memory `copy.deepcopy(conf) + правка + return`. Φ следует тому же.
6. **Ничего не ломается.** Φ ортогонален всем уже принятым спайкам (C/G/J/K/L). EC1–EC4, gossip-инварианты, capacity-канал П3 — все продолжают работать на $P_k$ как на $P_0$.
7. **Минимум по LOC.** ~60 строк на новый модуль + ~10 строк расширение `Talk.speakers` + загрузчика. Без новых модулей политик, без правки симулятора, без правки тестов этапа I (новые тесты добавляются как TC5 fixture, не правка).
8. **Стоп-лист §5.** Не задействуется (Φ — модификация программы, не social ABM, не recommender optimizer).

### Что отклоняется

- **V2** — нарушает §7 + эффект 0 на Mobius.
- **V3, V4** — превышают «локальную» границу.
- **V6 (Kempe)** — overkill для V1; future enhancement.
- **V7 (hill-climbing)** — out-of-scope §13.
- **V8 (IP)** — out-of-scope §13.
- **V9 (capacity-only)** — дублирует ось 1.

---

## 9. Минимальная первая реализация для этапа N

### 9.1. Расширение модели (`experiments/src/simulator.py`)

Точечная правка в dataclass `Talk` (строки 36–46):

```python
@dataclass
class Talk:
    id: str
    title: str
    hall: int
    slot_id: str
    category: str
    abstract: str
    embedding: np.ndarray
    fame: float = 0.0
    speakers: List[str] = field(default_factory=list)   # NEW
```

В `Conference.load`, после строки 119, добавить парсинг:

```python
speakers_raw = t.get("speakers", "")
speakers_list = (
    [s.strip() for s in speakers_raw.split(",") if s.strip()]
    if isinstance(speakers_raw, str) else list(speakers_raw or [])
)
talks[t["id"]] = Talk(
    ...
    speakers=speakers_list,
)
```

**Backwards-compat:** для toy / ITC / Meetup `speakers` отсутствует в JSON → `speakers=[]` → speaker-validation становится **no-op** (не «нет конфликтов», а «данных о спикерах нет»; в отчёте этапа N это явно фиксируется).

### 9.2. Новый модуль `experiments/src/program_modification.py`

API:

```python
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .simulator import Conference, Talk, Slot


@dataclass
class SwapDescriptor:
    """Описание одной локальной перестановки (для воспроизводимости и логов)."""
    slot_a: str       # slot_id, откуда приходит t1
    slot_b: str       # slot_id, откуда приходит t2
    t1: str           # talk_id, который перемещается из slot_a в slot_b
    t2: str           # talk_id, который перемещается из slot_b в slot_a


def has_speaker_conflict(conf: Conference) -> bool:
    """True, если в каком-либо слоте у двух разных талков есть общий спикер."""
    for slot in conf.slots:
        seen: set = set()
        for tid in slot.talk_ids:
            for sp in conf.talks[tid].speakers:
                if sp in seen:
                    return True
                seen.add(sp)
    return False


def _slot_day(slot: Slot) -> str:
    """Извлекает день из ISO datetime: '2026-01-01T10:00:00' → '2026-01-01'."""
    return slot.datetime[:10]


def _enumerate_all_pairs(conf: Conference) -> List[SwapDescriptor]:
    """Все candidate-pairs разных временных слотов одного дня."""
    pairs: List[SwapDescriptor] = []
    slots_by_day: dict = {}
    for s in conf.slots:
        slots_by_day.setdefault(_slot_day(s), []).append(s)
    for day, slots in slots_by_day.items():
        # PROJECT_DESIGN §7 говорит «параллельные слоты одного дня». В нашей
        # терминологии (Slot — это временной слот, параллельность реализуется
        # внутри slot'а через разные halls) корректное прочтение: «два РАЗНЫХ
        # временных слота одного дня». См. Accepted decision 2026-05-07.
        for i, s_a in enumerate(slots):
            for s_b in slots[i + 1:]:
                for t1 in s_a.talk_ids:
                    for t2 in s_b.talk_ids:
                        pairs.append(SwapDescriptor(
                            slot_a=s_a.id, slot_b=s_b.id, t1=t1, t2=t2
                        ))
    return pairs


def _apply_swap(conf: Conference, desc: SwapDescriptor) -> Conference:
    """Применяет swap к копии конференции."""
    cloned = copy.deepcopy(conf)
    t1, t2 = cloned.talks[desc.t1], cloned.talks[desc.t2]
    t1.slot_id, t2.slot_id = desc.slot_b, desc.slot_a
    # Перестраиваем Slot.talk_ids
    talk_ids_by_slot: dict = {s.id: [] for s in cloned.slots}
    for tid, t in cloned.talks.items():
        talk_ids_by_slot.setdefault(t.slot_id, []).append(tid)
    for s in cloned.slots:
        s.talk_ids = talk_ids_by_slot.get(s.id, [])
    return cloned


def enumerate_modifications(
    conf: Conference,
    k_max: int,
    rng: np.random.Generator,
    same_day_only: bool = True,
) -> List[Tuple[Conference, SwapDescriptor]]:
    """Возвращает до k_max валидных одиночных swap-модификаций программы.

    Не включает P_0 (исходную) — её добавляет вызывающий.
    Каждый возвращаемый Conference имеет тот же набор talks и halls;
    изменены только slot_id двух талков.

    Speaker-конфликты в результате swap отсекаются (hard validation).
    Если len(valid_pool) < k_max — возвращается всё, что есть.
    """
    if k_max <= 0:
        return []
    candidates = _enumerate_all_pairs(conf)
    valid: List[Tuple[Conference, SwapDescriptor]] = []
    for desc in candidates:
        modified = _apply_swap(conf, desc)
        if not has_speaker_conflict(modified):
            valid.append((modified, desc))
    if len(valid) <= k_max:
        return valid
    idx = rng.choice(len(valid), size=k_max, replace=False)
    return [valid[i] for i in sorted(idx)]
```

**Что делает модуль:**
- ~80 LOC.
- Не правит политики, симулятор, метрики, тесты этапа I.
- Использует прецедент `copy.deepcopy(conf) + правка + return` (run_smoke.py).
- Hard-validation через `has_speaker_conflict`. Для конференций без `speakers` (toy / ITC / Meetup) validation становится **no-op** (set() пустой, проверять нечего; это **не** свидетельство отсутствия конфликта, а отсутствие данных).

### 9.3. Тестовая поддержка (этап N, не сейчас)

- Новый fixture `experiments/data/conferences/toy_speaker_conflict.json` — минимальный test-fixture с явным конфликтом спикера: 2 слота × 2 зала × 4 талка, один speaker в двух слотах.
- Новый файл `experiments/tests/test_program_modification.py`:
  - TC5: `test_phi_rejects_speaker_conflict` — на toy_speaker_conflict при попытке swap, создающего конфликт, swap отклоняется.
  - `test_phi_preserves_talk_set` — `set(P_k.talks.keys()) == set(P_0.talks.keys())`.
  - `test_phi_preserves_slot_sizes` — `[len(s.talk_ids) for s in P_k.slots] == [len(s.talk_ids) for s in P_0.slots]`.
  - `test_phi_changes_slot_id_for_two_talks` — ровно 2 талка имеют изменённый `slot_id`.
  - `test_phi_returns_empty_for_single_slot_program` — для toy_microconf (1 слот) возвращает `[]`.
  - `test_phi_deterministic_under_fixed_rng` — два прогона с одним rng-seed дают идентичный результат.

### 9.4. Что не меняется в этапе N

- `simulator.py` — единственная правка `Talk.speakers` + парсинг в `Conference.load`. Всё остальное не трогается.
- Политики — не правятся.
- Метрики — не правятся.
- `run_smoke.py`, `run_llm_spike.py` — не правятся в N (расширятся в Q под ось 5).
- Активный реестр П1–П4 — не трогается.

### 9.5. Что нужно сделать до перехода к Q

- Реализовать §9.1 + §9.2 (этап N).
- Все тесты §9.3 проходят.
- pytest-набор этапа I + добавки этапа K остаются зелёными (51 → 56 тестов).

### 9.6. Бюджет

- Этап N: ~80 LOC модуль + ~10 LOC расширение `Talk` + ~150 LOC тестов + 1 JSON-фикстура → ~250 LOC и ~3–4 часа. Вписывается в одну сессию.

---

## 10. Что сознательно откладываем

1. **V6 Kempe-chain swap.** Расширение V1, поддержано S10. Откладывается; в Limitations: «Kempe-chain — стандартный timetabling-neighborhood; в нашей минимальной первой реализации не используется, оставляется как future enhancement».
2. **V4 K>1 block swap.** Многошаговые модификации с budget B>1. PROJECT_DESIGN §8 («фиксированный бюджет локальных перестановок») формально допускает, но V1+B=1 — минимум; B≤2 как опциональный sensitivity-sweep.
3. **V7 hill-climbing.** Out-of-scope §13.
4. **V8 IP.** Out-of-scope §13. Цитируется как related work.
5. **V3 межслотный сдвиг.** Требует концепта вакантного hall-в-slot, которого нет в текущем `Slot`.
6. **Storage модифицированных программ как отдельных JSON-файлов.** In-memory descriptors предпочтительнее для воспроизводимости (см. §11 Q-M6 ниже).
7. **Soft-penalty система** (как в Pylyavskyy 2024 S3). У нас hard validation; soft нарушает PROJECT_DESIGN §5.
8. **Multi-day swap.** PROJECT_DESIGN §7 явно «одного дня». Внутри-дня достаточно для стресс-теста.
9. **Visualization Φ (heatmap до/после).** Делается в этапе T (отчёт), не в N.

---

## 11. Какие проверки должны пройти в этапе N

Минимум, после реализации §9.1–9.3:

| Проверка | Где | Тип |
|---|---|---|
| `set(P_k.talks.keys()) == set(P_0.talks.keys())` для всех k | pytest | invariant |
| `[len(s.talk_ids) for s in P_k.slots] == [len(s.talk_ids) for s in P_0.slots]` | pytest | invariant |
| Ровно 2 талка имеют изменённый `slot_id` относительно P_0 | pytest | invariant |
| `not has_speaker_conflict(P_k)` для всех возвращаемых $P_k$ | pytest | hard validation |
| TC5 на `toy_speaker_conflict.json`: swap, создающий конфликт, отклоняется | pytest | TC5 PIVOT §10.1 |
| Для toy_microconf (1 слот) `enumerate_modifications` возвращает `[]` | pytest | edge case |
| `enumerate_modifications` детерминистична при фиксированном `rng-seed` | pytest | reproducibility |
| `pytest experiments/tests/ -v` — все 51 + новые тесты зелёные | pytest | regression |

Не входит в N (откладывается на Q):
- Применение Φ внутри LHS-генератора.
- Расчёт метрик группы 5 (эффект модификации программы).
- Sensitivity-sweep `k_max ∈ {0, 3, 5, 10}`.

---

## 12. Какие решения требуют подтверждения пользователя

### Q-M1. Какая форма Φ?

Варианты:
- **(а)** V1 pairwise swap **между разными временными слотами одного дня** + V5 random subsample + V0 P_0 control. **Рекомендуется.**
- (б) V1 + V6 Kempe (расширение). Дороже, не соответствует §7 буквально.
- (в) V4 block swap K>1.
- (г) V7 hill-climbing — отвергнуто §13.

**Предложение:** (а). **Подтверждено пользователем 2026-05-07** с терминологическим уточнением: «разные временные слоты одного дня» (а не «параллельные слоты»).

### Q-M2. Что делать при отсутствии `speakers` в JSON?

Варианты:
- **(а)** Считать `speakers=[]` (`Conference.load` возвращает пустой список). Speaker-validation становится **no-op** (данных о спикерах нет, проверять нечего; это не равно «конфликтов нет»). Для toy/ITC/Meetup Φ работает без speaker-проверок; в отчётах этот факт явно фиксируется как ограничение. **Рекомендуется.**
- (б) Использовать прокси по category / title — устаревшая идея R3, см. §2.3.
- (в) Hard-fail при отсутствии — слишком жёстко.

**Предложение:** (а). Расширение `Talk.speakers: List[str] = field(default_factory=list)` обеспечивает backwards-compat. **Подтверждено пользователем 2026-05-07.**

### Q-M3. Сколько вариантов программы генерировать (`k_max`)?

Варианты:
- **(а)** `k_max = 5` для LHS-50; `k_max = 3` для LLM-12. **Рекомендуется.** Разумное число для интерпретации.
- (б) `k_max = 10` — слишком много для интерпретации в отчёте.
- (в) `k_max = 1` — слишком мало для sensitivity.
- (г) Полное перечисление — на Mobius (40 талков, 16 слотов / 2 дня) candidate-pool ~500 пар; не нужно столько.

**Предложение:** (а) `k_max = 5` (LHS) / `3` (LLM). **Подтверждено пользователем 2026-05-07.**

### Q-M4. Разрешены ли перестановки между днями или только внутри дня?

PROJECT_DESIGN §7 прямо: «между параллельными слотами **одного дня**» (в нашей терминологии — между разными временными слотами одного дня; §3 уточнение). PIVOT-кандидат V3 (между днями) отклонён по §7.

Варианты:
- **(а)** Только внутри дня (`same_day_only=True` default). **Рекомендуется.** Соответствует §7.
- (б) Опциональный флаг `same_day_only=False` — расширение для будущего sensitivity.

**Предложение:** (а). Параметр оставить в API на будущее, default `True`. **Подтверждено пользователем 2026-05-07.**

### Q-M5. Включать ли P_0 как отдельный `program_variant`?

Варианты:
- **(а)** Да, `program_variant = 0` соответствует P_0 (control); `program_variant ∈ {1, ..., k_max}` соответствует $P_k$ из `enumerate_modifications`. **Рекомендуется.** PROJECT_DESIGN §8 строка 126 явно: «индекс варианта в множестве $\{P_0, P_1, \dots, P_k\}$».
- (б) Нет, ось 5 — только модификации (без control). Нарушает §8.

**Предложение:** (а). P_0 — обязательная control-точка LHS. **Подтверждено пользователем 2026-05-07.**

### Q-M6. Хранить ли модифицированные программы как отдельные JSON-файлы или in-memory?

Варианты:
- **(а)** **In-memory** descriptors. Φ — генератор; LHS-row хранит `(phi_seed, k_index, swap_descriptor)`; $P_k$ восстанавливается детерминированно. **Рекомендуется.** Соответствует прецеденту `scale_capacity` (in-memory deepcopy) и memory-правилу `feedback_no_backup_files`.
- (б) Отдельные JSON-файлы (`mobius_2025_autumn_p1.json`, ...). Создаёт мусор; PROJECT_STATUS §11 не упоминает; pytest fixture усложняется.

**Предложение:** (а) in-memory. **Подтверждено пользователем 2026-05-07.**

### Дополнительные решения, требующие подтверждения

#### Q-M7. Расширение `Talk.speakers: List[str]` в `simulator.py`

Это **изменение модели**, не только реализация Φ. Минимальная правка ~5 строк, backwards-compat сохраняется через default `[]`.

Варианты:
- **(а)** Расширить `Talk` напрямую. **Рекомендуется.**
- (б) Отдельный `speakers_map: Dict[talk_id, List[str]]` в `Conference`. Менее естественно.

**Предложение:** (а). **Подтверждено пользователем 2026-05-07.**

#### Q-M8. TC5 test-fixture

PIVOT §10.1 строка 955 описывает TC5 как «программа с одним и тем же спикером в двух параллельных слотах» (= в двух разных временных слотах одного дня, §3 уточнение). В наших данных такой программы нет.

Варианты:
- **(а)** Создать `experiments/data/conferences/toy_speaker_conflict.json` (минимальный test-fixture: 2 слота × 2 зала × 4 талка, один speaker в двух слотах). **Рекомендуется.**
- (б) Динамически модифицировать `toy_microconf_2slot` в pytest fixture.

**Предложение:** (а) — отдельный JSON-файл явный и документируемый. **Подтверждено пользователем 2026-05-07.**

---

## Recommended decision for N

Финальная конфигурация после спайка M (с терминологическим уточнением и подтверждениями Q-M1—Q-M8 от 2026-05-07).

**Форма Φ:** V1 pairwise swap двух докладов между **разными временными слотами одного дня** (терминологическое уточнение Accepted decision 2026-05-07; PROJECT_DESIGN §7 «параллельные слоты одного дня» в нашей реализации = «разные временные слоты одного дня») + V5 random subsample до `k_max` штук + V0 (P_0 control).

Семантика swap (зафиксировано):
- меняем `slot_id` у двух докладов;
- `hall` НЕ меняем;
- состав докладов сохраняется;
- размеры слотов сохраняются;
- speaker-conflict проверяется после swap (hard validation; на конференциях без `speakers` validation становится no-op).

**Файл:** `experiments/src/program_modification.py` (~80 LOC).

**API:**

```python
def enumerate_modifications(
    conf: Conference,
    k_max: int,
    rng: np.random.Generator,
    same_day_only: bool = True,
) -> List[Tuple[Conference, SwapDescriptor]]:
    """Возвращает до k_max валидных одиночных swap-модификаций.
    P_0 не входит — её клеит вызывающий (LHS-генератор).
    Hard validation: speaker-конфликты отсекаются."""

def has_speaker_conflict(conf: Conference) -> bool:
    """True, если у двух разных талков в одном слоте есть общий спикер."""
```

**Расширение модели:** `Talk.speakers: List[str] = field(default_factory=list)` + парсинг в `Conference.load` («, »-split + strip).

**Параметры по умолчанию:**
- `k_max = 5` (LHS-50), `k_max = 3` (LLM-12).
- `same_day_only = True`.
- `rng = np.random.default_rng(cfg.phi_seed)` (новое поле в LHS-row).

**Storage:** in-memory; LHS-row хранит `(phi_seed, k_index, swap_descriptor)` — этого достаточно для детерминированного восстановления любой $P_k$.

**Test-fixture:** новый `experiments/data/conferences/toy_speaker_conflict.json` для TC5.

**Acceptance этапа N (блокирующие, gate):**
1. `set(P_k.talks.keys()) == set(P_0.talks.keys())` для всех k.
2. Ровно 2 талка имеют изменённый `slot_id` относительно P_0 на каждый swap.
3. `[len(s.talk_ids) for s in P_k.slots] == [len(s.talk_ids) for s in P_0.slots]`.
4. `not has_speaker_conflict(P_k)` для всех возвращаемых $P_k$.
5. TC5 (`toy_speaker_conflict`): swap, создающий конфликт, отклоняется.
6. `enumerate_modifications` возвращает `[]` для одно-слотной toy.
7. Детерминизм при фиксированном `rng-seed`.
8. pytest 51 + новые тесты — все зелёные.

**Все Q-M1 — Q-M8 подтверждены пользователем 2026-05-07** (см. раздел Accepted decision в начале memo).

К этапу N не перехожу до отдельного сообщения пользователя.
