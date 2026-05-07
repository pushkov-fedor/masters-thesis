# Design-spike: LLM-симулятор (этап G)

Дата: 2026-05-07
Этап: G (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: accepted by user 2026-05-07 with three clarifications (см. раздел Accepted decision); design-spike, evidence-first; кода не меняет, экспериментов не запускает, LLM API не вызывает.

> Memo evidence-first. Сначала research log и реально изученные источники, затем требования и обзор вариантов, и только в конце — рекомендация и описание минимальной первой версии для этапа H.

---

## Accepted decision

Статус: принято пользователем для перехода к этапу H с тремя уточнениями (2026-05-07).

Подтверждённые решения по open questions раздела 13:

1. **Q-A.** Состав агента — V3 status quo `experiments/src/llm_agent.py` (Profile + Memory + Action). Изменений в `LLMAgent` на этапе H нет.
2. **Q-B.** Политики в LLM-spike — минимальные inline `no_policy` + `cosine` в новом `run_llm_spike.py`. Активный реестр П1–П4 в LLM-симуляторе выравнивается на этапе V, не в H.
3. **Q-C.** Параметры toy: 10 агентов; 2 слота; политики `no_policy` + `cosine`; K = 1.
4. **Q-D.** Бюджет — для H hard cap $5; полный бюджет V пока не фиксируется.
5. **Q-E.** Полный V к 13.05 — опционален. E2 (`llm_agents_mobius_2025_autumn_n50_falsification_4pol.json`) при необходимости — иллюстрация в W.
6. **Q-F.** Источник персон — `personas_100.json`. **10 агентов отбираются детерминированно и разнообразно** (равномерное прореживание индекса с фиксированным шагом либо k-means по эмбедингам, k=10), а не «первые 10», если первые 10 явно однотипны.
7. **Q-G.** Capacity в промпт LLM-агента не передавать (соответствует accepted decision этапа C).

Уточнения от пользователя 2026-05-07, изменяющие или дополняющие исходное предложение memo:

1. **Toy-данные для H.** Зафиксировано: H создаёт **отдельный** `toy_microconf_2slot` — 2 слота × 2 зала × 2 доклада в каждом слоте, **отдельно** от `toy_microconf` этапов D–F (там 1 слот). На исходном 1-слотном toy агент с history фактически не отличается от агента без history — это обнуляет разницу V2 / V3 на этапе H. Решение: новый файл данных в этапе H.
2. **Acceptance этапа H.** Сравнение `no_policy` vs `cosine` по overload — **diagnostic observation, не блокирующий gate**. На 10 LLM-агентах сравнение шумно и недоказательно; даже несовпадение знака — не основание блокировать переход. Блокирующие чеки — только operational (см. §9.5).
3. **Поведение при превышении бюджета.** При достижении hard cap $5 прогон **прерывается** со статусом `budget_exceeded` / `incomplete`, частичные результаты сохраняются, но **acceptance не считается пройденным**. Оставшиеся решения **не превращаются в skip** — иначе `skip-rate` и `mean_overload_excess` искажаются искусственно (агенты, которые «не пришли» из-за пустого fallback, ложно снижают переполнение).

### Дополнительные решения после прогона этапа H (2026-05-07)

После выполнения H (`experiments/results/llm_spike_2026_05_07.{json,md}`) пользователь принял этап и зафиксировал две дополнительные правки:

- **Q-H1 → (в).** Расщепление чека 6 §9.5 на два:
  - **gate:** `n_parse_errors == 0` и отсутствие API/JSON parsing failures;
  - **diagnostic:** `n_skipped / n_decisions`, **включая содержательные skip**.
  
  Содержательный skip («оба доклада далеки от моего фокуса») — это нормальное поведение V3-агента, а не технический провал. parse / API сбои — отдельная сущность и остаются gate.
- **Q-H2 → (2.4).** Текущий прогон H принимается как пройденный. `toy_microconf_2slot` сейчас не расширяется, промпт `LLMAgent` не правится, выборка персон сейчас не переделывается.
- **Замечание для V и дополнительных LLM-smoke.** Для этапа V и любых будущих LLM-прогонов на полных конференциях (не toy) обязательно контролировать стратификацию персон (например, по `preferred_topics` в `personas_100.json`) и покрытие тем талков, чтобы doc-rate не зависел от случайной mobile-heavy выборки. См. §10 пункт 11.

---

## 1. Проблема

Нужно зафиксировать минимальный проверяемый состав LLM-агента до начала этапа H (ранний LLM-spike: 5–10 агентов, 1–2 слота, 2 политики, та же toy-микроконференция). Цель этапа G — не построить наиболее умный агент, а выбрать конфигурацию, которая:

1. Удовлетворяет PROJECT_DESIGN §7 («второй независимый источник отклика», «профиль-персона и история выборов в текущем слоте»).
2. Сохраняет независимость LLM-симулятора от параметрического (`U = w_rel·rel + w_rec·rec`) для содержательной перекрёстной проверки на 12 точках LHS (§11, этап V).
3. Не противоречит уже принятому в этапе C решению («capacity-эффект только в политике, не в utility пользователя»).
4. Реализуема в срок, проверяема, недорогая по бюджету $.
5. Совместима по формату результата со smoke (этап F) и будущим LHS (этап V) для попарного сравнения политик и Spearman-корреляции ранжирований.

Без этого решения нельзя ни запускать H (поля результата заранее не определены), ни планировать V (не известно, что прогоняется).

---

## 2. Текущая реализация в репозитории

### 2.1. `experiments/src/llm_agent.py`

Минимальная архитектура, явно описанная в docstring (строки 1–12): «Профиль участника как текст. История посещённых сегодня докладов (для тематической инерции и усталости). Текущая загрузка залов передаётся в промпт. Прямой запрос к LLM: что выбираешь? → id доклада или skip. Без Big Five, без графа знакомств, без рефлексии — это вкладывается в эффект через естественный приор LLM, а не явный механизм».

Класс `LLMAgent` (строки 53–146) хранит:
- `agent_id: str`,
- `profile: str` — текстовая персона (до 600 символов в промпте),
- `history: list[dict]` — список посещённых сегодня докладов (`slot_id`, `talk_id`, `title`, `category`).

Метод `decide` (строки 68–110) принимает `slot_id`, `talks`, `hall_loads_pct`, `recommendation`, `llm_call`. Существенно: docstring строк 75–77 явно фиксирует, что `hall_loads_pct` **не передаётся в промпт** (это аргумент сигнатуры по историческим причинам, но в `USER_PROMPT_TEMPLATE` строки 31–41 поле для загрузки отсутствует). Это согласуется с принятым в этапе C решением «capacity-эффект только в политике П3».

Системный промпт (строки 22–28) даёт три фактора решения: профиль участника, история сегодняшних докладов (тематическая инерция + усталость к концу дня), рекомендация системы («не обязан слушать»). Никакого gossip, никакого межагентского общения, никакой социальной информации. Память — однодневный список без reflection. Парсер ответа (строки 123–146) терпим к небольшим отклонениям формата JSON и к индексам/префиксам id.

Это архитектура «Profile + Memory + Action», точная инстанциация канонической минимальной сборки simulation-oriented LLM-recsys по обзорной литературе (см. §5).

### 2.2. `experiments/scripts/run_llm_agents.py`

Скрипт прогона LLM-агентов на конференции. Существенные особенности:

1. **Inline-политики** (строки 60–131): четыре функции `policy_no_policy`, `policy_cosine`, `policy_cosine_capacity_filter`, `policy_cap_aware_mmr`. Это не активный реестр; см. §2.3.
2. **Динамический capacity** (строки 282–293): `cap = ceil(N_agents / halls_in_slot)` — слот-уровневое равномерное деление аудитории по залам, перекрывает структурное capacity конференции в JSON. Это конструкция specifically для того, чтобы переполнения **гарантированно были возможны**, иначе sanity-эффект политик П3 не виден.
3. **Sequential / batched processing** в слоте (строки 182–221): агенты идут батчами размера `slot_concurrency` (default 1, строгий sequential). Внутри батча параллельно, между батчами обновляется `slot_loads`. Это даёт правильную capacity-логику политики (политика видит накопительную загрузку), но смешивает политику и симулятор в одном цикле.
4. **Глобальная переменная `agent_emb_map`** (строка 304) — code smell, но работает.
5. **Метрики** (строки 261–272): `OF_choice` (доля выборов с переполнением «по итогам слота»), `hall_var_mean`, `mean_overload_excess`, `skip_rate`, `cost_usd`. Совпадает по смыслу со smoke, но имена другие (см. §11).
6. **Формат результата** (строки 354–365): `{config, elapsed_total_s, results: {policy: {decisions, slot_loads, metrics}}}`. Flat по политике; нет осей `point_id`, `w_rec`, `seeds`, `capacity_scenario`.

### 2.3. Расхождение inline-политик с активным реестром П1–П4

Активный реестр зафиксирован в `experiments/src/policies/registry.py` (строки 35–40):

```python
ACTIVE_POLICY_NAMES = ("no_policy", "cosine", "capacity_aware", "llm_ranker")
```

В `run_llm_agents.py` используются:

| inline-политика в скрипте | соответствие активному реестру | замечание |
|---|---|---|
| `policy_no_policy` | П1 `no_policy` | концептуально совпадает (рекомендация = None) |
| `policy_cosine` | П2 `cosine` | концептуально совпадает (top-K по cos) |
| `policy_cosine_capacity_filter` | **не входит в реестр** | capacity-mask `if load < 1.0`, не включена в П1–П4 |
| `policy_cap_aware_mmr` | **legacy, в стоп-листе** `registry.py:11` | MMR-разнообразие + capacity-penalty, не входит в основной эксперимент |

Активная П3 `capacity_aware` (`experiments/src/policies/capacity_aware_policy.py`) и активная П4 `llm_ranker` (`llm_ranker_policy.py`) в `run_llm_agents.py` **не вызываются**. PIVOT_IMPLEMENTATION_PLAN этап V (строки 902–903) явно требует выровнять состав в финальном LLM-прогоне на 12 точках. Этап H (строки 575–587) требует только две политики (например, no_policy и cosine), поэтому полное приведение к П1–П4 не обязательно для H, но желательно — иначе появляется разрыв «sky vs reality», который придётся закрывать в V.

### 2.4. Старый LLM-результат: формат

`experiments/results/llm_agents_mobius_2025_autumn_n50_falsification_4pol.json`:

```json
{
  "config": {"conference": "...", "personas": "...", "n_agents": 50, "K": 2, "model": "...",
             "policies": [...], "seed": 42},
  "elapsed_total_s": 1184.0,
  "results": {
    "<policy_name>": {
      "decisions": [{"agent_id": "...", "slot_id": "...", "chosen": "...|null",
                     "reason": "...", "cost_usd": 0.001}],
      "slot_loads": {"<slot>": {"<hall>": <int>}},
      "metrics": {"n_decisions": 800, "skip_rate": 0.13, "OF_choice": 0.50,
                  "hall_var_mean": 0.37, "mean_overload_excess": 0.21, "cost_usd": 0.07}
    }
  }
}
```

`results` — flat dict по политикам. Нет осей `seed`, `w_rec`, `point_id`, `capacity_scenario`. Это историческая E2-форма (один прогон ровно одной конфигурации), несовместима со структурой smoke (этап F) `results: [{capacity_scenario, policy, w_rec, seeds, agg, per_seed}]`.

### 2.5. Артефакты, которые остаются как есть

Нижеследующее не правится в этапе G (правило: spike не меняет код):

- `LLMAgent` — оставляется в текущем виде (Profile + Memory + Action — наш V3, согласно §5 он же канонический минимум).
- `LLMRankerPolicy` (`llm_ranker_policy.py`) — это П4, политика, не симулятор. Её роль — ranker внутри активного реестра, не часть LLM-симулятора отклика. Кэш на диске, бюджет $3, fallback на cosine — всё уже работает.
- inline-политики `run_llm_agents.py` — менять только в этапе V; в этапе H сохранить как минимальный адаптер (см. §9 и §13).

---

## 3. Требования из PROJECT_DESIGN / PROJECT_STATUS / PIVOT_IMPLEMENTATION_PLAN

### Из PROJECT_DESIGN

- **§7 Модуль симуляции:** «Существует в двух независимых вариантах. Параметрический вариант реализует модель индивидуального выбора замкнутой формы — мультиномиальный логит. Агентский вариант построен на языковой модели: каждому участнику соответствует отдельный экземпляр языковой модели с профилем-персоной и историей выборов в текущем слоте. Агентский симулятор играет роль второго независимого источника отклика и нужен для перекрёстной проверки выводов параметрического».
- **§9 Состав политик:** ровно П1–П4 в основном сравнении. На итог политика влияет ТОЛЬКО через `w_rec` функции полезности.
- **§11 Экспериментальный протокол / LLM-симулятор:** «12 точек, покрывающих крайние и центральные значения каждой оси. Внутри каждой точки также перебираются все четыре политики. Случайное зерно одно: внутри одной точки все четыре политики прогоняются на одной и той же синтетической аудитории — это известный приём общих случайных чисел. Суммарный объём — порядка 48 прогонов; временной бюджет — часы».
- **§11 Согласованность двух симуляторов:** «Сопоставляются ранжирования политик в LLM-эксперименте и в параметрическом эксперименте. Качественное совпадение ранжирования трактуется как взаимная проверка двух независимых источников отклика».

### Из PROJECT_STATUS

- **§5 Стоп-лист:** «Big Five / social graph / inter-slot chat как реализованный метод (был прототип, не основной результат)». Социальный граф и межагентский чат не подаются как защищаемые элементы LLM-симулятора.
- **§7:** активный состав политик П1–П4. MMR/gossip — параметрические модификаторы, не отдельные политики.
- **§8:** валидация многоуровневая, distribution-match Meetup ρ=0.438. B1/accuracy@1 не используется.

### Из PIVOT_IMPLEMENTATION_PLAN r5

- **§6 принцип 6:** «LLM-симулятор — ранний spike, поздний полный прогон. Spike на 5–10 агентах, 1–2 слотах, 2 политиках. Полный прогон на 12 точках — поздний этап».
- **§7 правило design-spike:** обязательный research log; источники реально открыты, не сниппеты; пометки доступа (`abstract-only`, `not-accessible`, `derived-only`).
- **Этап G (542–567):** список из 8 кандидат-вариантов; обязательное покрытие литературой (Park 2023/2024, Agent4Rec, OASIS, AgentSociety, SimUSER, PUB, Larooij & Törnberg).
- **Этап H (571–587):** минимальный прогон 5–10 агентов × 1–2 слота × 2 политики на toy-микроконференции; стоимость $ соответствует G; поля результата совместимы с форматом параметрического симулятора.
- **Этап V (898–916):** полный LLM-прогон 12 точек × 4 политики (П1–П4 активного реестра, не cap_aware_mmr/cosine_capacity_filter); метрика согласованности — Spearman.
- **Этап C accepted decision (`docs/spikes/spike_behavior_model.md`):** capacity-эффект полностью удалён из utility пользователя; живёт только в политике П3. Это инвариант, который LLM-симулятор обязан соблюсти под угрозой потери EC3 и нарушения §9.

---

## 4. Research log

Расширенный design-spike по правилу §7 PIVOT_IMPLEMENTATION_PLAN r5 выполнен через отдельный research-subagent с time-boxed бюджетом. Subagent изучал предметную область и возвращал research brief; написание самого memo ведётся в основной сессии.

### 4.1. Время

- **start time (subagent):** 1778112638 (epoch seconds) — 2026-05-07, ~07:10:38 MSK.
- **end time (subagent):** 1778112948 (epoch seconds) — 2026-05-07, ~07:15:48 MSK.
- **elapsed seconds:** 310 (≥ 300 — минимальный research budget по §7).
- Sleep / искусственное ожидание не использовались. Время потрачено на реальное I/O: Read локальных файлов, WebFetch / WebSearch внешних источников.

### 4.2. Изученные файлы кода

- `experiments/src/llm_agent.py` — текущая архитектура агента (Profile + Memory + Action), `hall_loads_pct` не передаётся в промпт.
- `experiments/scripts/run_llm_agents.py` — inline-политики, capacity = ceil(N/halls), batched sequential, формат результата.
- `experiments/src/policies/registry.py` — активный реестр П1–П4, lazy import LLMRankerPolicy.
- `experiments/src/policies/no_policy.py` — П1 (пустой recs).
- `experiments/src/policies/cosine_policy.py` — П2 (top-K cos).
- `experiments/src/policies/capacity_aware_policy.py` — П3 (sim − α·load_frac, hard threshold).
- `experiments/src/policies/llm_ranker_policy.py` — П4 (LLM-ranker через OpenRouter, кэш, бюджет, fallback).
- `experiments/src/policies/base.py` — контракт `BasePolicy.__call__(*, user, slot, conf, state)`.
- `experiments/results/llm_agents_mobius_2025_autumn_n50_falsification_4pol.json` — формат старого LLM-результата.
- `experiments/results/smoke_mobius_2025_autumn_2026-05-07.json` — формат параметрического smoke.

### 4.3. Изученные документы проекта

- `PROJECT_DESIGN.md` — §7 (два симулятора), §9 (состав политик), §11 (LLM-симулятор, согласованность).
- `PROJECT_STATUS.md` — §5 (стоп-лист), §7 (направление эксперимента), §8 (валидация).
- `PIVOT_IMPLEMENTATION_PLAN.md` — §6 (принципы), §7 (правила design-spike), G/H/V (этапы 542–587, 898–916).
- `docs/spikes/spike_behavior_model.md` — accepted decision этапа C: capacity вне utility, `consider_ids = slot.talk_ids`.
- `.claude/memory/research_field_survey_2026-05-04.md` — карта подходов LLM-симуляторов и канон валидации.
- `.claude/memory/reference_validation_defense.md` — must-cite, distribution-match Meetup.
- `materials/_legacy/research-симуляторы-датасеты.md` — таблица архитектурных модулей по симуляторам.
- `materials/_legacy/research-llm-models-2026-05-v2.md` — поверхностный аудит LLM-моделей.
- `materials/_legacy/research-conference-recsys-deep-2026-05.md` — старый deep research по conference recsys.
- `thesis/bibliography.bib` — пуст (только комментарии).

### 4.4. Внешние источники, реально открытые

См. §5 — отдельная таблица. Каждый источник снабжён пометкой о фактической доступности (`full`, `abstract-only`, `metadata-only`, `derived-only`, `not-accessible`).

### 4.5. Что оказалось нерелевантным и почему

- **AgentCF** (talk-as-agent) — у нас доклады фиксированные объекты программы, layer item-as-agent избыточен.
- **RecoWorld** (Meta 2025) — agentic recsys env для RL-обучения policy; мы не тренируем policy, политики у нас фиксированные.
- **iAgent / i2Agent** — LLM-shield между user и recommender; парадигма interaction-oriented; у нас LLM-as-ranker уже есть в П4.
- **SUBER** — LLM как rater для RL; не наш кейс.
- **MiroFish** — не peer-reviewed, инженерный продукт; cite-only максимум.
- **Sim4Rec / T-RECS** — параметрические recsys-симуляторы; не источник по LLM-агенту, но валидируют разделение «параметрический симулятор отдельный модуль».
- **LLM-emulated MNL** (концептуальный вариант): LLM решает по формуле `U = w_rel·rel + w_rec·rec` в промпте — отвергнут архитектурно (см. §6, V-extra-2).

### 4.6. Открытые вопросы по итогам research

См. §13 (открытые решения для подтверждения пользователем).

---

## 5. Обзор реально изученных источников

Источники сгруппированы по направлениям; каждый снабжён реальной пометкой о доступности.

### A. Generative agents и LLM-симуляторы пользователя

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S1 | Park J. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST '23. arXiv:2304.03442. | primary paper | abstract + GitHub repo (joonspk-research/generative_agents) | Архитектура: memory stream + reflection + planning. **Ablation:** observation, planning, reflection — каждый critically contributes to believability. Минимальная сборка в репо — 3 агента. История агента — CSV semicolon-separated memories. |
| S2 | Park J. et al. (2024). *Generative Agent Simulations of 1000 People.* arXiv:2411.10109. | primary paper | abstract-only | Три варианта инициализации: 2-hour interview / GSS+Big Five surveys / combined. **Точность относительно two-week test-retest:** 74% (demographics-only baseline) / 82% (surveys-only) / 83% (interview-only) / 86% (combined). Демография как baseline даёт 74%, +12 п.п. до полного — нелинейный возврат от глубины персоны. |
| S3 | Zhang A. et al. (2024). *On Generative Agents in Recommendation* (Agent4Rec). SIGIR '24. arXiv:2310.10108. | primary paper | full HTML + GitHub repo | Три модуля: Profile (social traits + tastes from 25 sample items) + Memory (factual + emotional + reflection) + Action (taste-driven + emotion-driven). **Стоимость:** $16 на 1000 агентов (ChatGPT-3.5) = **$0.016 на агента-полную-сессию**. Quick demo: 3 агента ~3 мин. **Ablation: social traits significantly influence agent behavior**. Filter-bubble воспроизведён успешно; **65% precision / 75% recall** на preference discrimination — «LLM hallucinations cause agents to consistently select fixed item counts». |
| S4 | Yang Z. et al. (2024). *OASIS: Open Agent Social Interaction Simulations with One Million Agents.* arXiv:2411.11581. | primary paper | full HTML | Архитектура из 5 компонентов: Environment Server, RecSys, Agent Module, Time Engine, Scalable Inferencer. **21 действие**. Ablation: «Removing the RecSys leads to premature end of information spread». Воспроизведение спреда — **~30% normalized error**. Агенты показывают «stronger herd effects than humans, particularly regarding dislikes». |
| S5 | Piao J. et al. (2025). *AgentSociety.* arXiv:2502.08691. | primary paper | full HTML | Четыре обязательных модуля: Profile + Status, Mental Processes (emotions / needs / cognition), Social Behaviors, Memory (Event Flow + Perception Flow). **MQTT-powered messaging system** — обязательный архитектурный слой для inter-agent / user-agent / surveys. Тяжёлая «full social» парадигма. |
| S6 | Bougie N., Watanabe N. (2025). *SimUSER: Simulating User Behavior with LLMs for Recommender System Evaluation.* ACL '25. arXiv:2504.12722. | primary paper | full HTML methodology | Четыре модуля: Persona (Big Five + age + occupation + pickiness) + Memory (episodic + knowledge-graph) + Perception (image captioning) + Brain (5-step CoT). **Ablation:** SimUSER(sim·persona) RMSE=0.502 на MovieLens; w/o persona — RMSE=0.666 (degradation **33%**). Human-likeness (5-pt): **SimUSER 4.41±0.16 vs Agent4Rec 3.04±0.12**. Micro-level accuracy 0.7912. |
| S7 | Wang L. et al. RecAgent. arXiv:2306.02552 + GitHub RUC-GSAI/YuLan-Rec. TOIS 12.2024. | primary paper | abstract + repo | Three-tier memory: sensory → short-term → long-term. **Inter-agent chat и broadcasting на «социальную сеть» — ядро архитектуры**. До 1000 агентов. Действия: смотреть фильм / chat с другим агентом / постить публично. Это V6 (межагентское общение). |
| S8 | Zhang J. et al. (2024). *AgentCF.* WWW '24. arXiv:2310.09233. | primary paper | abstract-only | **Дуальный подход:** пользователи И айтемы — агенты; mutual reflection через несоответствия с реальными данными. Не наш кейс — мы не делаем talk-as-agent. |
| S9 | Wu Z. et al. PUB. SIGIR '25. arXiv:2506.04551. | primary paper | abstract-only | LLM + Big Five inferred from logs + item metadata. Amazon reviews. Synthetic logs «closely align with authentic user behavior»; «meaningful correlations between personality traits and recommendation outcomes». Распределительная валидация, не accuracy@1. |
| S10 | Survey on LLM-powered Agents for Recommender Systems. arXiv:2502.10050. | survey | abstract + html | **Канонические 4 модуля:** Profile Construction + Memory Management + Strategic Planning + Action Execution. **«Simulation-oriented systems consistently exclude the Planning Module»**. Минимум для simulation = Profile + Memory + Action. Это формальный канон, в который укладывается V3 status quo `llm_agent.py`. |

### B. Валидация LLM-агентских симуляций

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S11 | Larooij M., Törnberg P. (2025). *Validation is the central challenge for generative social simulation.* AI Review (Springer). arXiv:2504.03274. | review | abstract-only | **«Many studies relying solely on subjective assessments of model believability»** — operational validity не доказывается. **«LLMs will exacerbate rather than resolve the long-standing challenges of ABMs»** — black-box. Прямая поддержка отказа от accuracy@1 и distribution-match как канона. |
| S12 | Tomasevic et al. (2025). *Operational Validation of an LLM-based Voat Simulation.* arXiv:2508.21740. | primary paper | abstract-only | **Stateless agents (Dolphin Mistral 24B)** без persistent memory между взаимодействиями. Operational validity по 5 dimensions: activity / network structure / toxicity / topical / stylistic. **99% confidence intervals** совпадают по unique users / root posts / DAU. Систематические divergences именно из-за statelessness — **persona+history даёт измеримую пользу** на сетевых метриках. |
| S13 | Beyond Believability (2025). *Promoting Behavioural Realism in Web Agents.* arXiv:2503.20749. | primary paper | abstract + html | **Prompt-only LLMs (DeepSeek-R1, Llama, Claude): 11.86% individual action accuracy. Fine-tuned: 17.26%, F1 33.86%**. Причины: LLM optimized for task completion, not behavior fidelity; modality gap; inherent strict-match difficulty. **Recommendation: distributional measures comparing aggregate action statistics** — каноническое подкрепление нашего тезиса. |
| S14 | Sargent R. (2013). *Verification and Validation of Simulation Models.* Journal of Simulation 7(1). | survey | metadata-only (использован через локальный обзор) | Канон extreme condition tests; operational validity = «accuracy required for the model's intended purpose over the domain of intended applicability». Используется для V и для §12. |

### C. Capacity-aware и congestion в recsys

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S15 | Mashayekhi Y. et al. (2023). *ReCon: Reducing Congestion in Job Recommendation using Optimal Transport.* RecSys '23. arXiv:2308.09516. | primary paper | abstract + локальный legacy summary | **Capacity-канал живёт в обучении recommender'а** (`O_ReCon = O_M + λ·O_C`), не в utility пользователя. Прямо подкрепляет архитектурное правило этапа C. |
| S16 | Diversity paradox feedback loop (2025). arXiv:2510.14857. | primary paper | abstract-only | **«Feedback loop increases individual diversity but simultaneously reduces collective diversity and concentrates demand on a few popular items»**. Прямая поддержка тезиса «без capacity-маски популярные доклады переполняются». |

### D. LLM-as-policy / LLM-as-ranker

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S17 | iAgent / i2Agent (2025). arXiv:2502.14662. | primary paper | abstract + html | iAgent: Parser + Reranker + Self-reflection. i2Agent добавляет dynamic memory + profile generator. **+16.6%** на ranking metrics. LLM-as-ranker — отдельная парадигма (recommender-oriented по S10), у нас уже инстанциирована в П4. |
| S18 | SUBER (2024). arXiv:2406.01631. | primary paper | abstract-only | Modular framework, LLM as synthetic users for RL recommender training. Близок по стеку. |

### Итог по §5

Реально открыто ~10 локальных файлов кода + ~10 локальных документов проекта + 18 внешних источников. Из 18 внешних: 6 с full HTML / methodology section (S3 Agent4Rec, S4 OASIS, S5 AgentSociety, S6 SimUSER, S10 survey, S17 iAgent), 2 с abstract + repo (S1 Park 2023, S7 RecAgent), 10 abstract-only / metadata-only.

Источники с пометкой `abstract-only` и `metadata-only` (S2, S7, S8, S9, S11, S12, S13, S14, S15, S16, S18) используются индикативно: их выводы — поддерживающие, не определяющие. Решение в §7–9 опирается прежде всего на **S1, S3, S4, S6, S10, S15** (источники с реально полученным методологическим содержанием).

---

## 6. Обзор вариантов реализации

### V1. Агент без персоны, без истории

LLM получает только список вариантов в слоте, выбирает один. Нет profile, нет history, нет hall_loads.

- **Что говорит литература:** S12 (Tomasevic) показывает, что stateless-агенты воспроизводят agregate-метрики (activity, DAU, root posts) в 99% CI, но систематически расходятся по network structure и stylistic. S2 (Park 2024) даёт нижнюю границу 74% accuracy для demographics-only baseline — без персоны теряется ~12 п.п. до combined 86%.
- **Реализуемость:** trivial — удалить из текущего `LLMAgent` всё, кроме `slot+talks`. ~1 час.
- **Плюсы:** дёшево, минимальный confound, чистый baseline для ablation.
- **Минусы:** все агенты выглядят одинаково для LLM → выбирают «лучший в среднем» доклад. В нашей задаче это ровно эффект E2 falsification — все 50 агентов выбирают один «обзор рынка». Не воспроизводит расщепление спроса между параллельными залами, что напрямую противоречит требованию §7 PROJECT_DESIGN.
- **Риски для нашей задачи:** без персон LLM-симулятор не выдаст meaningful распределение нагрузки → сравнение политик по `mean_overload_excess` обнуляется (всё переполняется одинаково).

### V2. Агент с персоной, без истории

`profile` передаётся в промпт; история между слотами не хранится.

- **Что говорит литература:** S12 — это V2 без памяти; statelessness — главный источник divergence в сетевой структуре. S6 (SimUSER): ablation w/o persona даёт RMSE 0.666 vs 0.502 (degradation 33%) — персона обязательна. S10 survey: simulation-oriented systems по канону требуют Profile + Memory.
- **Реализуемость:** ~1 час (передавать `history=[]` всегда).
- **Плюсы:** дёшево; персонализация выбора (DevOps-агент выбирает DevOps-доклад). Воспроизводит heterogeneity спроса.
- **Минусы:** агент не «помнит» что слушал на предыдущих слотах → нет тематической инерции / усталости, каждый слот независим. PROJECT_DESIGN §7 явно требует «историю выборов в текущем слоте».
- **Риски:** на toy с одним слотом эквивалентен V3, на полном Mobius (~6 слотов) теряет дискриминативность по последовательности выборов.

### V3. Агент с персоной и историей выборов в день (status quo `llm_agent.py`)

`profile` + список посещённых сегодня докладов в промпте. `hall_loads_pct` агенту не показываем (capacity — в политике П3).

- **Что говорит литература:** S1 (Park 2023) — ablation: memory + reflection + planning каждый critically contributes; для simulation памяти-как-логи может быть достаточно без явного reflection. S10 survey: «simulation-oriented systems consistently exclude the Planning Module»; **минимум для simulation = Profile + Memory + Action — это ровно V3**. S3 (Agent4Rec): Profile + Memory + Action — формальный канон simulation-oriented LLM-recsys.
- **Реализуемость:** **уже сделано** в `llm_agent.py`. Изменений в коде агента не требует.
- **Плюсы:** соответствует PROJECT_DESIGN §7 буква-в-букву; соответствует каноническому минимуму simulation-oriented (S10); воспроизводит heterogeneity спроса (через персоны) и тематическую инерцию (через history). Стоимость согласована со старым LLM-прогоном (~$0.07 на 50 агентов × 16 слотов на gpt-5.4-mini).
- **Минусы:** требует эмбедингов персон (`personas_100_embeddings.npz`) для outer cosine-политик; больше токенов в промпте. Это уже работает.
- **Риски:** нет (status quo).

### V4. Агент с персоной, историей и информацией о загрузке залов

Добавляем `hall_loads_pct` в промпт.

- **Что говорит литература:** S15 (ReCon) — congestion penalty в обучении recommender'а / в политике, не в utility пользователя.
- **Что говорит проект:** spike поведения этапа C (`docs/spikes/spike_behavior_model.md` accepted decision 5–6) принят пользователем: capacity-effect полностью удалён из utility. PROJECT_DESIGN §9: политика влияет только через `w_rec`. Если агент видит capacity напрямую — политика влияет помимо `w_rec`, EC3 ломается.
- **Реализуемость:** легко (~1 час), но архитектурно противоречит уже принятому решению этапа C.
- **Плюсы:** видимо реалистично («очередь у зала») — но это narrative, не строгое требование.
- **Минусы:** прямо нарушает spike поведения этапа C; обнуляет cross-validation на политике П3 (LLM-агент копирует поведение П3, между симуляторами исчезает разница на политике П3).
- **Риски:** ломает архитектурный паритет двух симуляторов. **Не делать.**

### V5. Агент с социальной информацией / gossip-сигналом

В промпт добавляется «соседи говорят X хорош» / рейтинги соседей / индикатор популярности.

- **Что говорит литература:** S5 (AgentSociety) — social interactions обязательный модуль, MQTT layer. S16 (diversity paradox) — feedback loops искажают коллективные распределения.
- **Что говорит проект:** PROJECT_STATUS §5 явно: «Big Five / social graph / inter-slot chat как реализованный метод (был прототип, не основной результат)». Gossip — отдельный плановый инкремент **этапов J–L** с собственным spike, не входит в G/H.
- **Реализуемость:** требует mechanism для генерации gossip-сигнала (peer ratings? leakage от уже выбравших? популярность по итогам предыдущего слота?).
- **Плюсы:** воспроизводит социальное заражение.
- **Минусы:** делает LLM-симулятор тяжелее параметрического (там тоже gossip ещё не реализован — этапы D–F прошли без него); **нарушает архитектурный паритет** двух симуляторов; gossip — отдельный spike с правилом §7. Если ввести gossip в LLM раньше, чем в параметрический, cross-validation на 12 точках теряет смысл (сравнение политик в одном симуляторе — с gossip, в другом — без).
- **Риски:** нарушение последовательности этапов плана.
- **Решение:** **отложить до этапов J–L**, как и предписывает план.

### V6. Межагентское общение (chat между слотами)

Агенты обмениваются сообщениями между слотами, влияют на чужие выборы.

- **Что говорит литература:** S7 (RecAgent) — inter-agent chat ядро архитектуры; до 1000 агентов; принят в TOIS 12.2024. S5 (AgentSociety) — MQTT message layer обязателен. S4 (OASIS) — chat не главный канал, главное — RecSys feed + 21 действие.
- **Что говорит проект:** PROJECT_STATUS §5 — inter-slot chat в стоп-листе как защищаемый результат.
- **Реализуемость:** очень тяжело — нужна асинхронная message passing, persistence между слотами, decision когда читать чужие сообщения, как они влияют на решение.
- **Плюсы:** наиболее «реалистично» в narrative-смысле.
- **Минусы:** дорого по $ (LLM-вызовов в N×N); сложно по коду; **прямой стоп-лист**.
- **Риски:** нарушение защитной логики; превышение бюджета.
- **Решение:** **не делать.**

### V7. LLM только как policy / ranker, не как симулятор

LLM не симулятор пользователя, а только ранкер (П4). Симулятор — параметрический MNL.

- **Что говорит литература:** S17 (iAgent) — LLM-as-ranker отдельная парадигма (recommender-oriented по S10).
- **Что говорит проект:** `llm_ranker_policy.py` это уже сделано — П4 в активном реестре. Но это политика, не симулятор отклика.
- **Реализуемость:** сделано.
- **Плюсы:** простой, дешёвый (бюджет $3 кэшируется).
- **Минусы:** **не покрывает требование PROJECT_DESIGN §7** о двух независимых источниках отклика. Если ВСЁ свести к V7, нет защищаемого LLM-симулятора, защищаемое утверждение «реализованы два независимых механизма» (PROJECT_DESIGN §16, положение 3) недоказуемо.
- **Риски:** срыв одного из положений на защиту.
- **Решение:** **не подменять LLM-симулятор на LLM-ranker.** V7 как политика П4 — оставить, как симулятор отклика — не годится.

### V8. Гибрид: параметрический пользователь + LLM-ranker как П4

Комбинация V7 + smoke (этап F): MNL-агент, политика — П4 LLM-ranker. Оба «симулятора» — на самом деле один MNL.

- **Реализуемость:** уже работает; smoke использует `include_llm=False`, полный гиперкуб включит П4.
- **Плюсы:** минимум новой работы. П4 как «интеллектуальная политика» сохраняется.
- **Минусы:** **не закрывает требование §7 о двух независимых симуляторах**. Cross-validation на 12 точках обнуляется (сравниваем MNL c MNL).
- **Риски:** то же, что V7.
- **Решение:** **не подменять LLM-симулятор на гибрид.** V8 описывает текущее состояние полного LHS на параметрическом симуляторе с П4 — это нормально для этапа Q, но это не LLM-симулятор отклика для этапа V.

### V9. Минимальный LLM-spike поверх текущего `llm_agent.py` без большого рефакторинга

`LLMAgent` остаётся как есть (V3 = persona + history). В новом `experiments/scripts/run_llm_spike.py` — минимальный прогон toy (1–2 слота × 5–10 агентов × 2 политики). inline-политики `run_llm_agents.py` пока не трогаем; формат результата приводим к smoke-совместимому канону.

- **Реализуемость:** новый скрипт `run_llm_spike.py` с правкой ~3–4 часа; `run_llm_agents.py` не трогается; `LLMAgent` не трогается.
- **Плюсы:** соответствует PIVOT_IMPLEMENTATION_PLAN H буква-в-букву; status quo `llm_agent.py` уже = V3 (Profile + Memory + Action — канонический минимум по S10); бюджет — единицы долларов на toy. Поля результата совместимы со smoke (требование H:581).
- **Минусы:** нужна унификация наименований метрик (`OF_choice` vs `overflow_rate_slothall` и т.п.). Это — постобработка / mapper; реализуется в скрипте без правки ядра.
- **Риски:** низкие.
- **Решение:** **это рекомендуемая форма этапа H** (см. §7–9).

### V-extra-1. Persona-from-personas без history

Промежуточный между V2 и V3: использовать структурированную персону из `personas_100.json`, но всегда передавать `history=[]`.

- **Литература:** S6 (SimUSER) ablation w/o memory не публиковалась явно; S12 — stateless+persona даёт divergence в сетевых метриках.
- **Реализуемость:** trivial.
- **Минусы:** прямо нарушает PROJECT_DESIGN §7 («история выборов в текущем слоте» — явно описанный компонент).
- **Решение:** **не делать.**

### V-extra-2. LLM-emulated MNL

Агент = LLM, но в промпте вытащен «вычисли U = w_rel·rel + w_rec·rec и выбери top-1». Это путь репликации параметрического симулятора через LLM.

- **Литература:** S11 (Larooij-Törnberg) — believability ≠ validity; такой подход прямо опасен — LLM «играет» в наш параметрический MNL, теряется независимость.
- **Минусы:** убивает требование §7 «второй независимый источник отклика».
- **Решение:** **не делать.**

---

## 7. Сравнительная таблица вариантов

| Вариант | Соответствие §7 (два независимых симулятора) | Соответствие §9 (политика только через w_rec) | Сохраняет inv. этапа C (capacity вне utility) | Реализуемость в срок | Стоимость $ для H | Стоимость $ для V (12 точек × 4 политики) | Поддержка литературой |
|---|---|---|---|---|---|---|---|
| V1 без персоны / истории | частично (агенты неотличимы) | да | да | trivial | < $1 на toy | ~$5 для V | S12 stateless даёт agregate ok |
| V2 персона без истории | частично | да | да | малая | $1–2 | ~$10 | S12, S6 ablation |
| V3 персона + история (status quo) | **да** | да | **да** | **0 (уже есть)** | $1–3 | ~$30–50 | **S1, S3, S6, S10 канон** |
| V4 + hall_loads в промпт | нет (копирует П3) | **нет** | **нет** | малая | как V3 | как V3 | S15 против |
| V5 + gossip-сигнал | да | да | да, но с gossip-каналом | средняя, отдельный spike | как V3 + gossip | дороже | S5, S16; **отдельный spike J** |
| V6 inter-agent chat | да | да | да | большая, дорого | дорого | очень дорого | S7; **стоп-лист** |
| V7 LLM-as-ranker only | **нет** (нет второго симулятора) | да | да | сделано | $0 | $3 (бюджет) | S17, **не закрывает §7** |
| V8 гибрид MNL + П4 | **нет** (один MNL) | да | да | сделано | $0 | $3 | то же; **не закрывает §7** |
| V9 минимальный LLM-spike поверх V3 | **да** | **да** | **да** | **малая** (~3–4 часа) | $1–3 | ~$30–50 | **наследует от V3** |
| V-extra-1 persona без истории | частично | да | да | trivial | $1 | $10 | прямо против §7 PROJECT_DESIGN |
| V-extra-2 LLM-emulated MNL | **нет** | да | да | trivial | $1 | $10 | S11 против |

Критерии выбраны из PROJECT_DESIGN §7, §9, §11; PROJECT_STATUS §5, §7; PIVOT_IMPLEMENTATION_PLAN §6 принципы 6, 7; §7 правила spike; этапы G/H/V; spike поведения этапа C accepted.

---

## 8. Evidence-based recommendation

Рекомендованный вариант для этапа H: **V9 — минимальный LLM-spike поверх текущего `llm_agent.py` (= V3), без рефакторинга агента**.

Содержательно: агент = Profile + Memory + Action (status quo), 5–10 агентов из `personas_100`, 1–2 слота из toy-микроконференции, политики `no_policy` и `cosine` (две крайних по rec-каналу), формат результата — расширение текущего LLM-result-формата под smoke-совместимый канон. inline-политики `run_llm_agents.py` в этапе H **не правим**, новый скрипт `run_llm_spike.py` использует свой минимальный набор. Активный реестр П1–П4 в LLM-симуляторе включается только в этапе V (см. §10).

**Обоснование (evidence-first, со ссылками на источники):**

1. **PROJECT_DESIGN §7 — два независимых симулятора.** V3 — единственная конфигурация в §6, кроме V5 (gossip), которая буква-в-букву соответствует требованию «отдельный экземпляр LLM с профилем-персоной и историей выборов в текущем слоте». V5 откладывается до J–L по плану.
2. **Канон simulation-oriented (S10 survey).** «Simulation-oriented systems consistently exclude the Planning Module» — минимум Profile + Memory + Action. V3 — точная инстанциация. V1, V2 — недостаточны по канону.
3. **Ablation персоны (S6 SimUSER).** Без персоны RMSE 0.666 vs 0.502, degradation 33%; без персоны индивидуальная фидельность теряется быстро. V1 → не годится для содержательного сравнения политик.
4. **Распределительный канон валидации (S2, S11, S13).** Park 2024: 74% baseline / 86% combined — distribution-match — каноническая метрика, не accuracy@1. Larooij-Törnberg: believability ≠ validity, distribution required. Beyond Believability: 11.86% individual accuracy, distributional measures recommended. **Отказ от accuracy@1 уже зафиксирован в PROJECT_STATUS §5; LLM-симулятор у нас — не для accuracy, а для второго распределительного ответа.**
5. **Capacity-инвариант этапа C.** V4 (capacity в промпт) прямо нарушает accepted decision этапа C. S15 (ReCon) подтверждает: capacity — в политике, не в утилите пользователя. Включение `hall_loads_pct` в LLM-промпт обнуляет cross-validation на политике П3.
6. **Стоп-лист (PROJECT_STATUS §5).** Inter-slot chat и social graph — не защищаемый результат. V5, V6 уходят за рамки этапа G.
7. **План §6 принцип 6.** «LLM-симулятор — ранний spike, поздний полный прогон. Spike на 5–10 агентах, 1–2 слотах, 2 политиках». V9 — буквальное соответствие.
8. **Стоимость (S3 Agent4Rec).** $0.016 на агента-полную-сессию (ChatGPT-3.5) → ×3–5 на gpt-5.4-mini / claude-haiku → **$1–3 для H** (10 агентов × 2 слота × 2 политики); **~$30–50 для V** (12 точек × 4 политики × ~50 агентов × ~6 слотов на gpt-5.4-mini). Это вписывается в «временной бюджет — часы» (PROJECT_DESIGN §11).

**Архитектурные следствия:**

- LLM-агент в этапе H остаётся «нем по capacity» — `hall_loads_pct` приходит в сигнатуру `decide`, но в промпт не попадает (status quo).
- Политики в `run_llm_spike.py` (новый файл) могут быть либо inline-копиями (как сейчас в `run_llm_agents.py`), либо лёгким адаптером поверх `BasePolicy` из `experiments/src/policies/`. Активный реестр П1–П4 в LLM-симуляторе включается только в этапе V — это вынужденный отложенный рефакторинг, см. §13 Q-B.
- Формат результата H пишется сразу с расширяемой структурой `{etap, conference, params, results: [{policy, point_id?, w_rec?, seed, agg, per_decision: [...]}]}` — flat в осях, но с явными полями `point_id` и `w_rec` (на этапе H заполнены константами), чтобы V не пере-форматировать.

**Что отклоняется:**

- V1, V2: недостаточны по канону simulation-oriented (S10) и по PROJECT_DESIGN §7.
- V4: ломает invариант этапа C, обнуляет cross-validation на П3.
- V5: gossip — отдельный spike J, нарушение последовательности этапов.
- V6: стоп-лист PROJECT_STATUS §5; дорого; не закрывает требование «минимальный».
- V7, V8: не закрывают §7 (нет второго независимого симулятора отклика).
- V-extra-1: нарушает явное требование §7 «история выборов».
- V-extra-2: подрывает независимость, S11.

---

## 9. Минимальная первая реализация для этапа H

### 9.1. Состав агента

Точно V3 status quo `experiments/src/llm_agent.py`. Изменений в агенте на этапе H **нет**.

```
LLMAgent:
    agent_id: str
    profile: str                    # текстовая персона из personas_100.json
    history: list[{slot_id, talk_id, title, category}]   # пуста на старте, обновляется commit() по итогу слота

decide(slot_id, talks, hall_loads_pct, recommendation, llm_call):
    # hall_loads_pct в сигнатуре, но НЕ в промпте — capacity только в политике
    user_prompt = render(profile, history, talks, recommendation)
    return LLM-decision (chosen_id | None, reason, cost)
```

Системный и пользовательский промпты — как сейчас. `hall_loads_pct` агенту не показываем.

### 9.2. Конфигурация прогона

| Параметр | Значение |
|---|---|
| Конференция | **`toy_microconf_2slot`** — новый файл данных для этапа H: 2 слота × 2 зала × 2 доклада в каждом слоте; **отдельно** от `toy_microconf` этапов D–F (там 1 слот). Создаётся в самом этапе H. |
| Количество агентов | **10** (из `personas_100`, **детерминированный разнообразный отбор**: равномерное прореживание индекса с фиксированным шагом либо k-means по эмбедингам, k=10; не «первые 10») |
| Количество слотов | **2** (содержательная проверка history-канала; на 1-слотном toy V3 = V2 на этапе H) |
| Политики | `no_policy` + `cosine` (две крайних по rec-каналу — даёт чистый sanity «есть ли вообще эффект политики») |
| Параметр K (top-K рекомендаций) | 1 (на toy этого достаточно) |
| Модель | `openai/gpt-5.4-mini` через OpenRouter (как в текущем `run_llm_agents.py`) |
| Случайное зерно | 1 (toy-spike, не статистика) |
| Capacity | `ceil(n_agents / halls_in_slot)` (как в `run_llm_agents.py`) — стресс по умолчанию |

**Бюджет (грубая оценка из S3 Agent4Rec, ×3–5 для gpt-5.4-mini):** $1–3 на H. **Hard cap $5 в скрипте.**

### 9.2.1. Поведение при превышении бюджета

При достижении `cumulative_cost ≥ $5` (hard cap) скрипт **не превращает оставшиеся решения в `skip`**. Логика:

1. Скрипт прекращает дальнейшую отправку запросов в LLM API.
2. Сохраняет частичные результаты в JSON-файл; в корне результата ставится поле `status: "budget_exceeded"` (или `"incomplete"`); количество не сделанных решений выписывается отдельным полем `n_decisions_aborted`.
3. Возвращает non-zero exit code или явный флаг `incomplete = true`.
4. Acceptance check «Прогон завершён без ошибок» в этом случае **не считается пройденным** (см. §9.5).

Причина: если оставшиеся решения превратить в `skip`, исказятся и `n_skipped / n_decisions`, и `mean_overload_excess` (агенты, которые «не пришли» из-за пустого fallback, ложно снижают переполнение). Корректнее зафиксировать неполноту прогона явно и пере-запустить с увеличенным cap или с меньшим числом агентов / политик.

### 9.3. Что делает скрипт `experiments/scripts/run_llm_spike.py` (новый файл, реализация — этап H)

1. Загружает `toy_microconf_2slot.json` + эмбединги (создаётся в этапе H), `personas_100.json` + эмбединги. Из `personas_100` отбирает 10 агентов **детерминированно и разнообразно** (равномерное прореживание индекса с фиксированным шагом либо k-means по эмбедингам, k=10), не «первые 10».
2. Создаёт LLMAgents (через текущий `LLMAgent`, без изменений).
3. Для каждой политики прогоняет агентов по слотам (sequential внутри слота для корректной capacity-логики политики; параллелизм между политиками — как в `run_llm_agents.py`).
4. Считает метрики (`mean_overload_excess`, `hall_var_mean`, `OF_choice`, `skip_rate`, `cost_usd`, `mean_user_utility` если возможно — иначе только первые пять).
5. Пишет результат в `experiments/results/llm_spike_<date>.json` в smoke-совместимом каноне.

### 9.4. Формат результата (canon `experiments/results/llm_spike_<date>.json`)

```json
{
  "etap": "H",
  "conference": "toy_microconf_2slot",
  "status": "ok",                          // или "budget_exceeded" / "incomplete"
  "n_decisions_aborted": 0,                // > 0 если budget_exceeded
  "params": {
    "n_agents": 10, "n_slots": 2, "K": 1, "model": "openai/gpt-5.4-mini",
    "seeds": [1], "policies": ["no_policy", "cosine"],
    "w_rec_values": [1.0],
    "capacity_scenarios": ["natural"],
    "personas_selection": "deterministic_diverse"   // как отбирали 10 из personas_100
  },
  "results": [
    {
      "capacity_scenario": "natural",
      "policy": "no_policy",
      "w_rec": 1.0,
      "seed": 1,
      "agg": {
        "mean_overload_excess": 0.0,
        "hall_utilization_variance": 0.0,
        "overflow_rate_slothall": 0.0,
        "n_decisions": 10,
        "n_skipped": 1,
        "cost_usd": 0.05
      },
      "per_decision": [
        {"agent_id": "u_001", "slot_id": "s_00", "chosen": "t_a", "reason": "...", "cost_usd": 0.005}
      ]
    }
  ]
}
```

Поля `w_rec` и `capacity_scenario` на этапе H заполнены константами (`1.0` и `"natural"` соответственно). На этапе V они станут переменными (12 точек × 4 политики × set of `w_rec`). Имена метрик в `agg` — канонические (соответствуют smoke):

| Метрика | smoke (этап F) | старый LLM (E2) | канон H/V |
|---|---|---|---|
| Превышение вместимости | `mean_overload_excess_mean` | `mean_overload_excess` | `mean_overload_excess` |
| Дисперсия загрузки залов | `hall_utilization_variance_mean` | `hall_var_mean` | `hall_utilization_variance` |
| Доля переполненных пар | `overflow_rate_slothall_mean` | `OF_choice` (другая семантика) | `overflow_rate_slothall` |
| Средняя utility пользователя | `mean_user_utility_mean` | — | `mean_user_utility` (если считается) |
| Доля отказов | `n_skipped_mean / n_decisions` | `skip_rate` | `n_skipped`, `n_decisions` |
| Стоимость | — | `cost_usd` | `cost_usd` |

Старая `OF_choice` («доля выборов с переполнением по итогам слота») — другая семантика, не идентична `overflow_rate_slothall` («доля пар слот×зал с превышением»). На этапе H писать `overflow_rate_slothall` (smoke-совместимый канон), `OF_choice` не писать.

### 9.5. Acceptance checks этапа H

#### Блокирующие (gate)

Все 6 чеков должны пройти. Если хотя бы один не пройден — этап H не считается выполненным, переход к I не разрешён.

| Чек | Проверка | Источник требования |
|---|---|---|
| Прогон завершён без ошибок | exit code 0; raised exceptions отсутствуют; статус **не** `budget_exceeded` / `incomplete` | PIVOT H:582 |
| Стоимость в пределах H-бюджета | `sum(cost_usd) ≤ $5` **и прогон не остановлен по cap** (см. §9.2.1) | PIVOT H:582 |
| Поля результата smoke-совместимы | в `agg` есть `mean_overload_excess`, `hall_utilization_variance`, `overflow_rate_slothall`; структура `etap / conference / status / params / results: [{...}]` | PIVOT H:581 |
| Время выполнения toy-spike | `elapsed_total_s ≤ 600` (10 минут) | здравый смысл |
| Parse / JSON / API failures отсутствуют | `n_parse_errors == 0`; ни одно решение не записано как `api-error` / `invalid-choice` / `parse-error` | sanity, fault rate; **не** включает содержательный skip от LLM |

#### Diagnostic observations (не блокируют)

Эти наблюдения фиксируются в выходном файле для отчёта, но **не влияют** на acceptance. На 10 LLM-агентах они шумны и не доказательны; полная содержательная проверка согласованности — этап V (12 точек × 4 политики, Spearman ранжирований).

| Наблюдение | Зачем смотрим |
|---|---|
| `n_skipped / n_decisions` (содержательный skip-rate) | Доля решений, в которых LLM-агент содержательно отказался выбирать («оба доклада далеки от моего фокуса»). На 10 агентах с однородной mobile-выборкой может уходить за 0.30; это **не** parse-fault. Возможный сигнал плохого покрытия тем докладов или однородной выборки персон, но не блокатор этапа H. |
| `sign(mean_overload_excess[no_policy] − mean_overload_excess[cosine])` | Согласованность с параметрическим smoke по знаку. На 10 агентах × 2 слота шум большой; даже несовпадение знака — не доказательство ошибки модели. |
| Распределение `chosen` по докладам и залам | Качественная проверка, что LLM-агенты не «коллапсируют» на одном докладе для всех 10 агентов и реагируют на персону. |
| Длина и качество поля `reason` | Sanity, что LLM-агент содержательно объясняет выбор, не выдаёт пустую заглушку. |
| Стоимость в расчёте на агента-полную-сессию | Проверка соответствия оценке S3 Agent4Rec ($0.016/агент на ChatGPT-3.5 → ×3–5 на gpt-5.4-mini); сильное отклонение — сигнал переоценки бюджета V. |

**Что НЕ проверяем на этапе H:**

- Никаких EC1–EC4 (это этап I для параметрического симулятора).
- Никакого Spearman / pairwise win-rate (это V).
- Никакой согласованности с параметрическим smoke по абсолютным числам.
- Никакой согласованности по знаку как блокирующий чек — только diagnostic.

### 9.6. Что фиксируется в коде этапа H

- Новый файл `experiments/scripts/run_llm_spike.py`.
- Возможны мелкие правки `experiments/src/llm_agent.py` для совместимости (например, exposing token usage). Status quo достаточно.
- Никаких изменений в `simulator.py`, `policies/registry.py`, `run_llm_agents.py`, активном реестре политик.

---

## 10. Что сознательно откладываем

1. **Полное приведение `run_llm_agents.py` к активному реестру П1–П4** (замена inline-политик на `active_policies()` или адаптер поверх `BasePolicy`). Это сделать в этапе V (PIVOT_IMPLEMENTATION_PLAN строки 902–903), не в H. Причина: H — sanity-spike, не финальный прогон; время лучше потратить на корректность формата и независимость симуляторов, не на refactoring адаптера.
2. **Добавление политик П3 `capacity_aware` и П4 `llm_ranker` в LLM-spike**. На H достаточно `no_policy` + `cosine`. П3 и П4 включаются в этапе V, когда есть полная LHS-структура.
3. **Gossip-сигнал в LLM-агенте (V5).** Отдельный spike J, отдельная реализация K, отдельная проверка L. До этого LLM-симулятор и параметрический симулятор симметричны: оба без gossip. После этапа K оба обновляются параллельно. Если gossip окажется реализован только в одном — cross-validation на 12 точках теряет смысл.
4. **Социальная информация / inter-agent chat (V6).** Стоп-лист §5 PROJECT_STATUS. Не реализуем как защищаемый результат.
5. **Big Five / структурированная персона.** Текущая персона — текст из `personas_100.json` (свободная форма). S6 (SimUSER) использует Big Five + occupation + age + pickiness — это богаче. На этап H избыточно; в Limitations диссертации (глава 4): «персоны заданы как свободно-текстовые описания; SimUSER-стиль структурированной персоны (Big Five) — отдельная ось эксперимента, не входит в обязательный результат».
6. **Reflection / planning модуль (S1 Park 2023).** S10 survey: simulation-oriented systems exclude planning. Status quo `LLMAgent` без reflection корректен по канону.
7. **Распределительная калибровка LLM-агента под Meetup ρ=0.438.** PROJECT_STATUS §8: Meetup — distribution-level якорь, для LLM прогона не обязателен на предзащиту. Если будет время — отдельный sanity на смежной задаче.
8. **Multiple seeds для LLM-симулятора.** PROJECT_DESIGN §11: «Случайное зерно одно: внутри одной точки все четыре политики прогоняются на одной и той же синтетической аудитории — common random numbers». На H seed = 1; на V seed = 1 на точку. Это согласовано с §11 и снижает шум сравнения политик.
9. **Финальный конвертер «smoke-формат ↔ LLM-формат» для общего pandas DataFrame.** Реализуется в этапе S (постобработка). На H достаточно того, чтобы формат LLM был расширяемым в будущую структуру.
10. **Замена модели (`gpt-5.4-mini` → `claude-haiku-4.5` или `deepseek/deepseek-v3.2-exp`).** Параметр `--model` уже есть. На H остаётся `gpt-5.4-mini` для совместимости со старым прогоном; в V — отдельное решение по бюджету.
11. **Стратификация персон и контроль покрытия тем для V и будущих LLM-smoke** (замечание зафиксировано пользователем 2026-05-07 после H). На toy `toy_microconf_2slot` (4 темы — NLP / iOS / DevOps / Java) на 10 персонах из `personas_100`, отобранных через KMeans-разнообразие, content skip-rate составил 0.35 у `no_policy`: персоны систематически mobile-ориентированные, тогда как 2 из 4 тем (NLP, DevOps) для них слабо релевантны. Для V и любых дальнейших LLM-прогонов на полных конференциях (не toy) **обязательно**:
    - использовать стратифицированный отбор персон по `preferred_topics`, а не только KMeans по эмбедингам;
    - контролировать покрытие тем талков (доля талков из тем, по которым в выборке есть хотя бы одна персона);
    - следить за content skip-rate как diagnostic-метрикой, и если он систематически зашкаливает — пересматривать выборку, не промпт `LLMAgent`.
    Это не правка ядра, а ограничение на параметры запуска V и любых дополнительных smoke. Сейчас не делается, фиксируется как inheritable требование.

---

## 11. Какие проверки должны пройти до перехода дальше

### Этап G → этап H

- Memo подписан пользователем по разделам 11–13 (явно подтверждены open questions Q-A — Q-G).
- Решение по форме персоны / памяти / capacity-в-промпте зафиксировано.
- Бюджет H согласован.

### Этап H → этап I (extreme conditions параметрического)

- Ранний LLM-spike прошёл acceptance checks §9.5.
- Формат результата `experiments/results/llm_spike_<date>.json` валиден по схеме §9.4.
- Качественное согласие LLM-spike и параметрического smoke на toy: `sign(no_policy − cosine)` в `mean_overload_excess` согласован.

### Этап I → этап V (полный LLM-прогон)

- EC1–EC4 параметрического симулятора прошли (см. spike поведения C, §11).
- inline-политики `run_llm_agents.py` приведены к активному реестру П1–П4 либо адаптированы через лёгкий адаптер.
- Спайк J выполнен; gossip-инкремент K реализован; проверка L пройдена.

### Этап V (acceptance)

- 48 прогонов на 12 точках × 4 политики × 1 seed.
- Spearman-корреляция между ранжированиями политик параметрического и LLM-симулятора по каждой ключевой метрике рассчитана.
- Стоимость в рамках согласованного бюджета.

---

## 12. Какие решения требуют подтверждения пользователя

### Q-A. Состав агента для этапа H

Вариант:

- **(а)** V3 status quo `llm_agent.py` (Profile + Memory + Action), без изменений.
- (б) V2 (без истории) — упростить, но потерять соответствие §7 PROJECT_DESIGN.
- (в) V1 (без персоны и истории) — баseline-only, не для основного сравнения.

**Предложение:** (а) V3. Соответствие PROJECT_DESIGN §7, канон simulation-oriented S10, ablation S6 (без персоны −33% RMSE). Изменений в коде не требует.

**Подтверждено пользователем 2026-05-07:** (а) V3.

### Q-B. Активный реестр политик в LLM-spike

Вариант:

- **(а)** На этапе H — оставить минимальный inline-адаптер `policy_no_policy` + `policy_cosine` в `run_llm_spike.py` (новый файл). `run_llm_agents.py` не трогать. Активный реестр П1–П4 в LLM-симуляторе включить в этапе V (отдельный adaptor над `BasePolicy`, см. ниже).
- (б) Сразу в H использовать `active_policies(include_llm=False)` через адаптер: преобразовать `BasePolicy.__call__(*, user, slot, conf, state)` в сигнатуру inline-политики (`user_emb, talk_embs, talk_ids, hall_loads, K`).

**Предложение:** (а). Этап H — sanity, не финальный прогон. Адаптер требует решения по контракту `state` (где брать `hall_load`, `K`, `relevance_fn`) — это отдельная мини-задача с риском обнаружить расхождение между LLM-симулятором и параметрическим в самом адаптере. Делать в V, когда нужны все 4 политики.

**Подтверждено пользователем 2026-05-07:** (а). Реестр П1–П4 в LLM-симуляторе выравнивается на этапе V.

### Q-C. Параметры toy-конфигурации этапа H

- **Q-C-1.** Количество агентов: 5, 10 или другое? Предложение: **10**. **Подтверждено пользователем 2026-05-07:** 10.
- **Q-C-2.** Количество слотов: 1 (одно решение на агента) или 2 (память между слотами становится содержательной)? Предложение: **2** — иначе history не вступает в игру, и V3 = V2 на toy. **Подтверждено пользователем 2026-05-07:** 2 слота через **новый отдельный** `toy_microconf_2slot` (2 слота × 2 зала × 2 доклада в каждом слоте; не расширение `toy_microconf` этапов D–F, а самостоятельный файл данных).
- **Q-C-3.** Политики: `no_policy + cosine` (два крайних) или `no_policy + capacity_aware` (тестируем capacity-канал)? Предложение: **no_policy + cosine** — это ровно то, что предлагает PIVOT H:577. capacity_aware включится в V. **Подтверждено пользователем 2026-05-07:** `no_policy + cosine`.
- **Q-C-4.** K (top-K): 1 или 2? Предложение: **1** на toy (с 2 талками в слоте top-2 == все варианты, политика обнуляется). **Подтверждено пользователем 2026-05-07:** K = 1.

### Q-D. Бюджет $ для V

- **(а)** $50 hard cap (позволяет gpt-5.4-mini).
- (б) $20 hard cap (требует deepseek-v3.2-exp или урезание агентов до 25 на точку).
- (в) Отказаться от V к 13.05; использовать E2 (один прогон) как иллюстрацию, и Spearman не считать. PROJECT_STATUS §13 уже это допускает.

**Предложение:** дать оценку бюджета пользователю и услышать его решение. Без бюджетного потолка V не запускать.

**Подтверждено пользователем 2026-05-07:** для H — hard cap **$5**; полный бюджет V на данный момент **не фиксируется** (решение откладывается до прохождения этапов H, I, J, K, L и предварительной оценки времени до 13.05).

### Q-E. Обязательность V к предзащите 13.05

- **(а)** Обязателен — Spearman-согласованность входит в защищаемые результаты (положение 3 §16 PROJECT_DESIGN).
- (б) Опционален — LLM-симулятор показывается через E2-иллюстрацию `experiments/results/llm_agents_mobius_2025_autumn_n50_falsification_4pol.json`; раздел согласованности помечен «LLM-этап выполнен на одной точке» (PIVOT W:925).

**Предложение:** к предзащите 13.05 V — **опционален**, при условии, что E2 явно помечен как иллюстрация. К антиплагиату 08.05 V не нужен. Зависит от Q-D и от того, успеют ли пройти этапы I, J, K, L, M, N, O, P, Q до V.

**Подтверждено пользователем 2026-05-07:** (б) опционален.

### Q-F. Источник персон для LLM-spike

- **(а)** `personas_100.json` (текущий, согласован со старым LLM-прогоном).
- (б) Сгенерировать `toy_personas_100.json` (как в этапе D — может быть мельче и проще для toy).

**Предложение:** **(а)** `personas_100.json` для консистентности и бесплатных эмбедингов в `personas_100_embeddings.npz`.

**Подтверждено пользователем 2026-05-07:** (а) `personas_100.json`. Дополнительное требование: **отбор 10 агентов детерминированный и разнообразный** (равномерное прореживание индекса с фиксированным шагом либо k-means по эмбедингам, k=10), не «первые 10», если первые 10 явно однотипны. Способ отбора фиксируется в `params.personas_selection` выходного JSON.

### Q-G. Capacity в промпте LLM-агента

- **(а)** Не передавать `hall_loads_pct` агенту; capacity-канал только в политике (status quo, accepted в этапе C).
- (б) Передать `hall_loads_pct` как narrative-сигнал («очередь у зала»).

**Предложение:** **(а)**. Это accepted decision этапа C; (б) ломает EC3 и обнуляет cross-validation на политике П3.

**Подтверждено пользователем 2026-05-07:** (а) не передавать.

---

## Recommended decision for H

Финальная конфигурация после трёх уточнений пользователя 2026-05-07.

**Состав агента:** V3 status quo `experiments/src/llm_agent.py` — Profile + Memory + Action. Изменений в `LLMAgent` нет.

**Политики для H:** `no_policy` + `cosine` (inline-копии в новом скрипте `run_llm_spike.py`). Активный реестр П1–П4 в LLM-симуляторе — отложен до V.

**Данные:**
- Конференция: **`toy_microconf_2slot`** — новый файл данных, создаётся в этапе H: 2 слота × 2 зала × 2 доклада в каждом слоте; **отдельно** от `toy_microconf` этапов D–F. На 1-слотном toy V3 = V2, поэтому новый файл обязателен.
- Персоны: `personas_100.json` + `personas_100_embeddings.npz`. **10 агентов отбираются детерминированно и разнообразно** (равномерное прореживание индекса с фиксированным шагом либо k-means по эмбедингам, k=10), не «первые 10». Способ отбора фиксируется в `params.personas_selection` выходного JSON.
- Программа: 2 слота × 2 зала × 2 доклада, capacity = `ceil(n_agents / halls_in_slot)`.

**Размер spike:** 10 агентов × 2 слота × 2 политики × 1 seed.

**Бюджет:** hard cap **$5** на скрипт (грубо $1–3 ожидается на gpt-5.4-mini). При достижении cap прогон **прерывается**, не превращает оставшиеся решения в `skip`; в результат записывается `status: "budget_exceeded"`, `n_decisions_aborted: > 0`, exit code non-zero. Acceptance в этом случае **не считается пройденным** (см. §9.2.1, §9.5).

**Метрики (smoke-совместимый канон):**
- `mean_overload_excess`,
- `hall_utilization_variance`,
- `overflow_rate_slothall`,
- `n_decisions`, `n_skipped`, `n_parse_errors`, `n_decisions_aborted`,
- `cost_usd`.

(`mean_user_utility` — опционально, если можно посчитать по relevance из эмбедингов.)

**Acceptance checks (блокирующие, gate):**
1. Прогон завершён без ошибок; exit code 0; `status` **не** `budget_exceeded` / `incomplete`.
2. `sum(cost_usd) ≤ $5` и прогон **не остановлен по cap**.
3. Формат результата `etap / conference / status / params / results: [{...}]` с каноническими именами метрик в `agg`.
4. `elapsed_total_s ≤ 600`.
5. `n_parse_errors == 0`; ни одно решение не записано как `api-error` / `invalid-choice` / `parse-error`. Содержательный skip от LLM сюда **не** входит — он переехал в diagnostic (см. ниже, по решению пользователя 2026-05-07 после этапа H, Q-H1 (в)).

**Diagnostic observations (не блокируют переход):**

- `n_skipped / n_decisions` — содержательный skip от LLM-агента; на однородной mobile-выборке + узком покрытии тем может превышать 0.30, это не блокер, но сигнал к стратификации персон / расширению тем для V.
- `sign(mean_overload_excess[no_policy] − mean_overload_excess[cosine])` — на 10 агентах шум большой; даже несовпадение знака — не блокатор, не повод возвращаться к C/G.
- Распределение `chosen` по докладам и залам (sanity, не «коллапс на одном докладе»).
- Качество поля `reason` (содержательные мотивации, не пустая заглушка).
- Стоимость на агента-полную-сессию vs S3 Agent4Rec ($0.016 / агент на ChatGPT-3.5 → ×3–5 на gpt-5.4-mini); сильное отклонение — сигнал переоценки бюджета V.

**Все open questions §13 подтверждены пользователем 2026-05-07.** Этап G принят. Этап H принят пользователем 2026-05-07 после прогона `experiments/results/llm_spike_2026_05_07.{json,md}` с расщеплением чека 6 на gate (parse_errors) и diagnostic (skip-rate). Переход к этапу I — отдельным сообщением пользователя.
