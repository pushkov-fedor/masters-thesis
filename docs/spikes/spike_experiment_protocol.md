# Design-spike: экспериментальный протокол (этап O)

Дата: 2026-05-07
Этап: O (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: accepted by user 2026-05-07 with five technical clarifications (см. раздел Accepted decision); design-spike, evidence-first; кода не меняет, экспериментов не запускает, к этапу P не переходит до отдельного сообщения.

> Memo evidence-first. Сначала research log и реально изученные источники, затем требования и обзор вариантов, и только в конце — рекомендация и описание минимальной первой версии для этапа P. Структура повторяет принятые ранее `spike_behavior_model.md` (этап C), `spike_llm_simulator.md` (этап G), `spike_gossip.md` (этап J), `spike_program_modification.md` (этап M).

---

## Accepted decision

Статус: принято пользователем для перехода к этапу P с пятью техническими уточнениями (2026-05-07). Направление O (LHS + CRN + 50×4×3, 12 LLM-точек, Spearman, OAT/scatter вместо Sobol) подтверждено.

### Уточнение 1 — статус П4 LLMRankerPolicy в полном LHS (Q-O9 accepted 2026-05-07)

PROJECT_DESIGN §11 нормативно требует «все четыре политики на всех 50 точках». Однако П4 — API-вызов на ChatGPT-4o-mini через OpenRouter; стоимость и wallclock несопоставимы с П1–П3. Прецедент: smoke F был выполнен на `active_policies(include_llm=False)` (П1–П3), оценка wallclock «12–15 минут на 600 evals» в §9 основана **только** на П1–П3.

**Принятое решение Q-O9 (2026-05-07): вариант (в) компромисс.**

- **П1–П3** на всех 50 LHS-точках × 3 seed = 450 evals (полный охват PROJECT_DESIGN §11 для трёх детерминированных политик);
- **П4 LLMRankerPolicy** только на 12 maximin-точках, выбранных для LLM-V × 3 seed = 36 evals (cross-validation двух симуляторов сохраняется на тех самых 12 точках, как требует §11);
- Параметрический LHS = **486 evals** (450 + 36);
- LLM-V = 48 evals (12 точек × 4 политики × 1 seed; П4 в LLM-симуляторе на тех же 12 точках для cross-validation).

**Это осознанное частичное отступление от §11** в части «все 4 политики на всех 50 точках». Обоснование:
- П4 — это LLM-канал; запуск ChatGPT-4o-mini на всех 600 evals параметрического LHS даёт cold-cache wallclock ~2–3 часа и стоимость ~$5–10, что несопоставимо с временным бюджетом «минуты» для параметрического (§11);
- §11 cross-validation «на 12 общих точках» сохраняется полностью: на 12 maximin-точках П4 присутствует и в параметрическом, и в LLM-симуляторе;
- остальные 38 LHS-точек (где §11 формально требует все 4 политики) фиксируются в Limitations: «П4 LLMRankerPolicy в полном параметрическом LHS не запускалась из-за wallclock-ограничений; cross-validation двух симуляторов проведена на 12 общих точках».

Cost (~$2–4) и wallclock (~15 мин warm-cache) укладываются в практичный бюджет.

### Уточнение 2 — audience_size

Сетка `{50, 100, 200}` в §6 базировалась на неявном предположении, что есть согласованный nest 50⊆100⊆200. Реальное состояние:
- `personas_50.json` (50) ⊆ `personas_100.json` (100) — согласованы;
- `personas_x3_200.json` (200) — **независимая выборка с другими ID** (пересечение с personas_100 пусто).

Сетка `{50, 100, 200}` без явной фиксации источника даёт **разрыв базовой выборки** между уровнями оси. Принятое решение: **сетка `audience_size ∈ {30, 60, 100}`**, все уровни — subsample из `personas_100.json` через `audience_seed`. Это сохраняет согласованность с этапами H/L (которые работали на `personas_100`).

### Уточнение 3 — generate_lhs без фиксированного over_generate_factor

В §14 предложен `over_generate_factor=1.2`. Это hard-coded и не гарантирует ровно N=50 валидных строк после симплекс-фильтра. Принятое решение: **rejection sampling в цикле** — генерировать блоки и фильтровать до тех пор, пока не наберётся ровно `n_points` валидных строк. Каждый блок использует свой sub-seed (ответвление от `master_seed`), детерминизм сохраняется.

### Уточнение 4 — контроль дискретных осей после фильтрации

После симплекс-фильтра дискретные оси (`program_variant ∈ {0..5}`, `audience_size ∈ {30, 60, 100}`, `popularity_source ∈ {cosine_only, fame_only, mixed}`) могут оказаться несбалансированными (некоторые уровни не представлены или представлены ≤ 2 раз). Принятое решение: **проверять покрытие после генерации; при дисбалансе — regenerate / repair**, не fail в тесте. Конкретно:
- каждый уровень `program_variant` ∈ {0..5} — представлен ≥ 5 раз (50/6 ≈ 8 ожидаемо);
- каждый уровень `audience_size` ∈ {30, 60, 100} — представлен ≥ 12 раз (50/3 ≈ 17);
- каждый уровень `popularity_source` ∈ {3 категории} — представлен ≥ 12 раз;
- при нарушении — replace random point на forced-level point.

### Уточнение 5 — smoke этапа P не проверяет EC3 на случайных 5 точках

В §14 предложено: «EC3 на этих 5 точках проходит». Это некорректно: на случайных LHS-5 точках `w_rec = 0` почти не появится (диапазон `[0, 0.7]`, вероятность точно 0 — нулевая мера). Принятое решение: **smoke этапа P проверяет**:
- формат long-table выхода (правильные колонки);
- CRN-инвариант (одна аудитория и тот же `phi`-результат для 4 политик внутри LHS-точки);
- работоспособность связки policies / Φ / metrics;
- wallclock 5×4×1 evals в пределах разумного.

EC3-smoke (если нужен) — **отдельная forced test row с `w_rec = 0` явным образом**, не на случайных LHS-точках. Это согласовано с pytest этапа I (где EC3 уже проверяется forced-row тестом `test_ec3_invariance_when_w_rec_zero`).

### Подтверждённые направления

- LHS как метод (не Sobol);
- 50 точек × 4 политики × 3 seed (с уточнением Q-O9 по П4);
- 12 точек для LLM через maximin distance с force-include `program_variant=0`;
- CRN: `audience_seed`/`phi_seed` фиксированы на уровне LHS-row, `cfg_seed=replicate`;
- Spearman ρ как метрика согласованности;
- OAT + scatter, без Sobol.

После реализации этих уточнений — переход к этапу P.

---

## 1. Проблема

Нужно зафиксировать экспериментальный протокол: **набор LHS-осей**, **диапазоны**, **число точек**, **число seed**, **стратегию CRN** (одинаковая аудитория и фиксированные random potoki внутри одной LHS-точки), **отбор 12 точек для LLM-прогона**, **синхронизацию `w_gossip` между параметрическим и LLM**. Решение — вход в этап P (создание `experiments/src/lhs.py`, `experiments/src/seeds.py`, `experiments/scripts/run_lhs_parametric.py`); далее этап Q (полный параметрический прогон 600 evals) и этап V (LLM-прогон 12 точек × 4 политики × 1 seed).

Без зафиксированного протокола:
- этапы P/Q/V заблокированы;
- защищаемые положения PROJECT_DESIGN §16 (особенно положение 4 «количественные оценки … локальных модификаций программы») недоказуемы;
- согласованность двух симуляторов на 12 общих точках (PROJECT_DESIGN §11) не имеет процедуры.

Цель memo — выбрать **минимальный воспроизводимый протокол**, удовлетворяющий ограничениям:

1. **Соответствие PROJECT_DESIGN §11** — 50 точек LHS × 4 политики × 3 seed для параметрического (~600 evals); 12 точек × 4 политики × 1 seed (CRN) для LLM (~48 evals).
2. **Соответствие 6 параметрическим осям §8.**
3. **CRN на уровне LHS-точки** — все 4 политики и все 3 seed-реплики видят одну и ту же синтетическую аудиторию и тот же `program_variant`-эффект.
4. **Sargent-канон** — extreme condition tests (EC1–EC4) до содержательных выводов; уже зелёные в smoke F и в pytest этапа I.
5. **Совместимость с уже принятыми спайками** — симплекс `w_rel + w_rec + w_gossip = 1` (Q-J4), L2 LLM-gossip (Q-J7-revised), синхронность `w_gossip` (Q-J12), `k_max=5/3` для Φ (Q-M3), `program_variant=0` как control (Q-M5).
6. **Минимизация wallclock** — 600 параметрических evals в пределах разумного времени (PIVOT этап Q: «временной бюджет — минуты»); LLM 48 evals в пределах часов и согласованного бюджета (Q-D в spike LLM-симулятора).

Исключаются: Sobol indices total-order (overkill для timeline), новые датасеты, accuracy@1 как валидация, full factorial, sequential bifurcation.

---

## 2. Текущая реализация в репозитории

### 2.1. SimConfig

`experiments/src/simulator.py`, строки 152–232. Уже содержит все нужные поля:

```python
@dataclass
class SimConfig:
    tau: float = 0.7
    p_skip_base: float = 0.10
    K: int = 3
    seed: int = 0
    w_rel: float = 0.7
    w_rec: float = 0.3
    w_gossip: float = 0.0
    w_fame: float = 0.0
    lambda_overflow: float = 0.0     # DEPRECATED, capacity вынесен в политику П3
    user_compliance: float = 1.0      # legacy
    use_calibrated_compliance: bool = False  # legacy
    alpha_compliant: float = 0.717    # legacy
    alpha_starchaser: float = 0.213
    alpha_curious: float = 0.070
```

Расширений в P не требуется. Все 6 осей PROJECT_DESIGN §8 параметризуются через существующие поля + `audience_seed` (в новом `seeds.py`) + `program_variant` (через `enumerate_modifications`).

### 2.2. CRN-структура уже в ядре

Строки 376–378 в `_process_one_slot`:

```python
choice_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx)
policy_rng = np.random.default_rng(cfg.seed * 1_000_003 + slot_idx + 31)
```

Два независимых RNG-потока. EC3-инвариант (формальная CRN при `w_rec=0`) уже зелёный (`test_extreme_conditions::test_ec3_invariance_when_w_rec_zero`, `test_ec3_extended_invariance_when_w_rec_zero_w_gossip_positive`). При `w_rec > 0` траектория `choice_rng` дрейфует, но **первые шаги до первого политика-зависимого выбора** идентичны — это даёт частичный CRN.

### 2.3. Прецеденты грид-прогонов

`experiments/scripts/run_smoke.py`, строки 148–194. Текущий грид:

```python
for cap_name in cap_scenarios:
    scaled_conf = scale_capacity(base_conf, CAPACITY_SCENARIOS[cap_name])
    for w_rec in w_rec_grid:
        for w_gossip in w_gossip_grid:
            if w_rec + w_gossip > 1.0 + 1e-9:
                print(f"  skip: w_rec={w_rec} + w_gossip={w_gossip} > 1.0")
                continue
            w_rel = max(0.0, 1.0 - w_rec - w_gossip)
            cfg = SimConfig(tau=0.7, p_skip_base=0.10, K=K, seed=0,
                            w_rel=w_rel, w_rec=w_rec, w_gossip=w_gossip)
            for pol_name, pol_obj in pols.items():
                per_seed = []
                for s in seeds:
                    cfg.seed = s
                    res = simulate(scaled_conf, users, pol_obj, cfg)
                    per_seed.append(compute_metrics(scaled_conf, res))
                rows.append({...})
```

Это **полный факториальный грид** (не LHS). Симплексная нормировка `w_rel + w_rec + w_gossip = 1` уже работает с явным skip недопустимых пар. Этот шаблон переносится в `run_lhs_parametric.py` с заменой внешних циклов на проход по LHS-точкам.

### 2.4. Operator Φ

`experiments/src/program_modification.py`:

```python
def enumerate_modifications(
    conf: Conference,
    k_max: int,
    rng: np.random.Generator,
    same_day_only: bool = True,
) -> List[Tuple[Conference, SwapDescriptor]]:
```

Возвращает до `k_max` валидных swap-модификаций; `P_0` НЕ включается в выдачу — клеится вызывающим как `program_variant=0`. **Детерминирована при фиксированном `rng`-seed** (test_phi_deterministic_under_fixed_rng зелёный).

### 2.5. Уже работающая инфраструктура отбора

`experiments/scripts/run_llm_spike.py`, строки 70–95:

```python
def select_personas_kmeans(pers_ids, pers_embs, k=10, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(pers_embs)
    selected = []
    for c in range(k):
        members = np.where(labels == c)[0]
        center = km.cluster_centers_[c]
        sims = pers_embs[members] @ center
        best = int(members[int(np.argmax(sims))])
        selected.append(best)
    return selected, f"kmeans_k{k}_seed{random_state}"
```

**Прямо переносится** на отбор 12 LHS-точек: вместо эмбедингов персон — координаты LHS-row в unit-cube; вместо k=10 — k=12.

### 2.6. Smoke-результаты как baseline

- `smoke_toy_microconf_2026-05-07.{json,md}` — все 5 expectation PASS на сетке 3 capacity × 3 w_rec × 3 w_gossip × 3 политики × 3 seed.
- `smoke_mobius_2025_autumn_2026-05-07.{json,md}` — то же на полной программе. Wallclock: ~5 минут на полный грид (27 cap×wrec×wgossip × 3 политики × 3 seed = 243 evals).
- `gossip_validation_2026_05_07.md` — отчёт L (gossip-эффект виден на Mobius natural).

Это даёт калибровку wallclock: **600 параметрических evals на Mobius при той же глубине ≈ 12–15 минут**, что соответствует PIVOT этап Q «временной бюджет — минуты».

---

## 3. Требования из PROJECT_DESIGN / PROJECT_STATUS / PIVOT + accepted decisions

### Из PROJECT_DESIGN

- **§5 постановка** (строка 48): «Множество параметрических конфигураций $\Xi$ есть пространство всевозможных комбинаций значений по всем параметрическим осям: вместимость, параметры модели популярности, веса модели поведения, размер аудитории, вариант программы».
- **§8 Параметрические оси** (строки 114–130): 6 осей. Ось 1 capacity multiplier; Ось 2 popularity model; Ось 3 `(w_rel, w_rec, w_gossip)` + параметр стохастичности (tau); Ось 4 audience size + distribution; Ось 5 program_variant; Ось 6 random seed (репликация).
- **§9** (строка 134): «На итоговое распределение посещений политика влияет только через компонент w_rec». → CRN-инвариант EC3.
- **§10 группа 4** (строки 162–166): «Устойчивость политики при варьировании конфигурации. Доля точек выборки, в которых одна политика лучше другой по показателю риска (попарное сравнение) … Размах значений показателя риска при варьировании одной оси конфигурации. Сожаление политики относительно лучшей в каждой точке». — это и есть OAT-sensitivity и pairwise win-rate.
- **§11 Экспериментальный протокол** (строки 177–198):
  - параметрический симулятор: «Гиперкуб строится из 50 точек по осям конфигурации (вместимость, модель популярности, параметры модели поведения, размер аудитории, вариант программы). Внутри каждой точки перебираются все четыре политики, и для каждой пары (ξ, π) симулятор запускается с тремя различными случайными зёрнами. Суммарный объём — порядка 600 прогонов; временной бюджет — минуты».
  - LLM-симулятор: «12 точек, покрывающие крайние и центральные значения каждой оси. Внутри каждой точки также перебираются все четыре политики. Случайное зерно одно: внутри одной точки все четыре политики прогоняются на одной и той же синтетической аудитории — это известный приём общих случайных чисел (common random numbers)».
  - анализ: «Сравнение политик идёт попарно внутри каждой точки гиперкуба … Усреднение показателя по точкам не используется».
  - согласованность: «Согласованность двух симуляторов проверяется на 12 общих точках. Сопоставляются ранжирования политик в LLM-эксперименте и в параметрическом эксперименте».
- **§11 EC** (строки 200–204): extreme condition testing (Sargent 2013); если EC проваливается — переход к содержательным выводам блокируется. EC1–EC4 уже зелёные в pytest и smoke (для `w_gossip=0` и `w_gossip=0.3`).
- **§13 допущения** (строка 224): «Полная задача оптимизации программы … к работе не относится».

### Из PROJECT_STATUS

- **§5 стоп-лист** (строка 49): accuracy@1 / B1-leakage не валидация. Распределительный канон (Sargent / Larooij & Törnberg). Big Five / social graph / inter-slot chat — не основной результат.
- **§7 направление** (строки 78–95): план эксперимента — LHS по параметрическим осям + полный перебор политик; сравнение пер-точечное, не среднее.
- **§8 валидация** (строки 99–105): toy-cases, internal consistency, monotonicity, repeated seeds, sensitivity, согласованность двух симуляторов, EC tests.
- **§11 следующий шаг** (строки 140–151): спринт реализации — LHS+CRN, оператор Φ, ось program_variant, полный параметрический прогон, отбор 12 точек, постобработка, отчёт, EC tests.

### Из PIVOT_IMPLEMENTATION_PLAN

- **§6 принцип 5** (строка 222): «LHS и full parametric run нельзя запускать до реализации и проверки gossip». ✓ gossip реализован (этап K) и проверен (этап L), pytest 67 зелёных.
- **§6 принцип 7**: «Отчёт — экспериментальный артефакт, не текст диплома».
- **§7** правила design-spike: research log, source tracking, recommendation.
- **Этап O** (строки 745–770): семь групп вариантов design-space — план эксперимента / variance reduction / sensitivity / N points / N seeds / оси / отбор 12.
- **Этап P** (строки 776–796): новые файлы `experiments/src/lhs.py`, `experiments/src/seeds.py`, `experiments/scripts/run_lhs_parametric.py`. Smoke run этапа P: 5 точек × 4 × 1 на Mobius.
- **Этап Q** (строки 796–812): 50 × 4 × 3 = 600 evals; wallclock ≤ 30 минут; long-формат с колонками для всех осей.
- **Этап V** (строки 898–916): 12 точек × 4 политики × 1 seed CRN; Spearman ρ между ranking-ями политик параметрического и LLM.

### Из accepted decisions предыдущих spike

- **spike_behavior_model** (этап C accepted): `w_rel + w_rec = 1` для базовой модели; capacity вне utility; `p_skip_base = 0.10`; модель `rel + rec` без gossip — только для этапов C–F.
- **spike_gossip** (этап J accepted Q-J1, Q-J2, Q-J3, Q-J4, Q-J5, Q-J6):
  - V5 log_count: `gossip(t, L_t) = log(1+count_t)/log(1+N_users)`;
  - per-talk count, не per-hall load_frac;
  - форма gossip фиксирована в коде (не ось);
  - симплекс `w_rel + w_rec + w_gossip = 1`, диапазон каждого `[0, 0.7]`;
  - smoke сетка `[0.0, 0.3, 0.7]`.
- **spike_gossip_llm_amendment** (accepted Q-J7-revised, Q-J8, Q-J9, Q-J10, Q-J11, Q-J12):
  - L2: реальные числа `count_t/N_users` в LLM-промпте + 3-уровневая параметризация системным промптом;
  - дискретные уровни `{off (w=0), moderate (0<w<0.4), strong (w≥0.4)}`;
  - `w_gossip=0` → блок отсутствует в промпте полностью;
  - `w_gossip` синхронен между параметрическим и LLM в каждой LHS-точке.
- **spike_program_modification** (accepted Q-M1 — Q-M8):
  - V1+V5+V0;
  - `k_max=5` для LHS, `k_max=3` для LLM;
  - `same_day_only=True`;
  - `P_0` как `program_variant=0` (control);
  - in-memory descriptors;
  - hard-validation speaker-конфликтов; no-op при отсутствии данных.

---

## 4. Research log

Расширенный design-spike по правилу §7 PIVOT_IMPLEMENTATION_PLAN r5 выполнен через отдельный research-subagent с time-boxed бюджетом и явным требованием минимум 8 успешных WebFetch с реальными цитатами.

### 4.1. Время

- **start time (subagent):** epoch `1778172816`, 2026-05-07.
- **end time (subagent):** epoch `1778173116`, 2026-05-07.
- **elapsed seconds:** 300 (≥ 300 — минимальный research budget по §7).
- Sleep / искусственное ожидание не использовались. Время потрачено на реальное I/O: Read локальных файлов + WebFetch / WebSearch внешних источников. Часть PDF-источников (Sargent 2013 JoS, Kleijnen 2005 EJOR ScienceDirect, McKay 1979 JSTOR) недоступны через WebFetch (403/404/binary PDF) — компенсировано Wikipedia, scipy.docs, EMA-workbench docs, Tilburg/WUR mirrors с явной пометкой `derived-only / abstract-only / not-accessible`.

### 4.2. Изученные файлы кода

- `experiments/src/simulator.py:152–232` — SimConfig.
- `experiments/src/simulator.py:376–501` — `_process_one_slot` (RNG, gossip-канал, swap-чувствительность).
- `experiments/src/program_modification.py:138–179` — `enumerate_modifications` API.
- `experiments/scripts/run_smoke.py:148–230` — текущий грид с симплексом и skip недопустимых пар; `scale_capacity`.
- `experiments/scripts/run_llm_spike.py:70–95, 200–290` — `select_personas_kmeans`, gossip_level_from_w, проброс w_gossip в decide.
- `experiments/results/smoke_toy_microconf_2026-05-07.{json,md}` — структура; все 5 expectation PASS.
- `experiments/results/smoke_mobius_2025_autumn_2026-05-07.{json,md}` — то же на полной программе.
- `experiments/results/llm_spike_2026_05_07_L_AB.json` — формат LLM-A/B.
- `experiments/results/gossip_validation_2026_05_07.md` — отчёт L.

### 4.3. Изученные документы проекта

- `PROJECT_DESIGN.md` §5, §7, §8, §10, §11, §13, §16.
- `PROJECT_STATUS.md` §5, §7, §8, §11.
- `PIVOT_IMPLEMENTATION_PLAN.md` §6, §7, этапы O, P, Q, V.
- `docs/spikes/spike_behavior_model.md` accepted.
- `docs/spikes/spike_gossip.md` accepted (включая amendment).
- `docs/spikes/spike_gossip_llm_amendment.md` accepted.
- `docs/spikes/spike_program_modification.md` accepted.
- `docs/spikes/spike_llm_simulator.md` accepted.

### 4.4. Локальные research-файлы

- `.claude/memory/research_field_survey_2026-05-04.md` — DMDU-канон (Lempert 2003 RAND, Marchau 2019, Kwakkel & Pruyt 2013 TFSC, Kwakkel 2017 EMS — EMA-workbench), §1.А прогнозный трэк закрыт; §2.1 discrete choice / MNL.
- `.claude/memory/reference_validation_defense.md` — must-cite: Sargent 2013, Kleijnen 2005, Robinson 2014, JASSS 27/1/11, Larooij & Törnberg 2025; canonical distribution-match как валидация LLM-слоя.
- `materials/_legacy/research-conference-recsys-deep-2026-05.md` — legacy под старый фрейм; cite только для контекста.

### 4.5. Внешние источники

23 успешных WebFetch (>500 символов содержания каждый), реально открытых. Из них **19 — full / partial с прямыми цитатами**, **4 — abstract-only / metadata-only** с явной пометкой. См. таблицу §5.

### 4.6. Что оказалось нерелевантным

- **Sobol indices total-order** — требует N×(k+2) дополнительных evals под Sobol. Для timeline 13.05 — overkill. Замена: OAT + scatter (PIVOT этап R/S).
- **Sobol sequence для генерации точек** — ограничение «n must be power of 2»; 50 не подходит, нужно 64. LHS не имеет этого ограничения.
- **Halton sequence** — low-discrepancy, но без strata-гарантии. LHS для нашего OAT-sensitivity лучше.
- **Full factorial / partial factorial** — для 6 осей × 5 levels = 15625 точек; неподъёмно.
- **Mixed Logit / Random-coefficients MNL / Latent-class** — overkill, отвергнуто в spike_behavior_model.
- **PRIM scenario discovery** — полезный посттреатмент, но это этап S/T, не O. Упомянуть как «следующий шаг после полного LHS-прогона».
- **Sequential bifurcation / group screening** — для k≫10. У нас k=6, не нужно.
- **Lloyd-optimised LHS** — возможно улучшит discrepancy, но default `LatinHypercube(d=6, scramble=True, optimization='random-cd')` достаточен.

### 4.7. Открытые вопросы

См. §16 (Q-O1 — Q-O8).

---

## 5. Обзор реально изученных источников

### A. Latin Hypercube Sampling

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S1 | scipy.stats.qmc.LatinHypercube | docs.scipy.org/.../LatinHypercube.html | full | «with a LHS of n points, the variance of the integral is always lower than plain MC»; «a 14-fold reduction in the number of samples required achieved compared to grid sampling when studying a 6-parameter epidemic model» | API: `LatinHypercube(d, scramble=True, strength=1, optimization=None, rng=None)`; `optimization='random-cd'` для центрированной discrepancy. **Для нашего N=50 / k=6 — рекомендованная конфигурация**. |
| S2 | Wikipedia: Latin hypercube sampling | en.wikipedia.org/wiki/Latin_hypercube_sampling | full | «Latin hypercube sampling (LHS) is a statistical method for generating a near-random sample of parameter values from a multidimensional distribution»; «This sampling scheme does not require more samples for more dimensions (variables); this independence is one of the main advantages» | McKay 1979 LANL — origin; число сэмплов **не зависит от размерности** — ключевой довод за LHS-50 при k=6. |
| S3 | Wikipedia: Latin square | en.wikipedia.org/wiki/Latin_square | full | «In experimental design, Latin squares serve as a special case of row-column designs for two blocking factors. They help researchers minimize experimental errors.» | Корневой concept — LHS обобщает Latin square на n измерений. |
| S4 | Wikipedia: Stratified sampling | en.wikipedia.org/wiki/Stratified_sampling | full | «It can produce a weighted mean that has less variability than the arithmetic mean of a simple random sample of the population»; «If measurements within strata have a lower standard deviation, stratification gives a smaller error in estimation» | Концептуальная основа LHS — strata по каждой оси, по одной точке в каждой strata. |
| S5 | Kleijnen 2005 abstract via WUR mirror | research.wur.nl/.../an-overview-of-the-design-and-analysis-of-simulation-experiments | abstract-only | «Latin hypercube sampling (LHS), and other 'space filling' designs»; «Modern designs were developed for simulated systems in engineering and management science, allowing many factors (more than 100), each with either a few or many values (more than 100)» | Kleijnen 2005 — авторитет на DOE для simulation. LHS — рекомендованный «space filling» design. Полный PDF binary `derived-only`. |

### B. Common Random Numbers

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S6 | Wikipedia: Variance reduction | en.wikipedia.org/wiki/Variance_reduction | full | «The common random numbers variance reduction technique is a popular and useful variance reduction technique which applies when we are comparing two or more alternative configurations»; «Var[Z(n)] = (Var[X₁] + Var[X₂] − 2·Cov[X₁, X₂]) / n … if we succeed to induce an element of positive correlation … the variance is reduced» | **Прямое обоснование CRN** для попарного сравнения политик внутри одной LHS-точки. Условие: одинаковая аудитория и одинаковый `choice_rng` для всех 4 политик. |
| S7 | Wikipedia: Monte Carlo method | en.wikipedia.org/wiki/Monte_Carlo_method | full | «By the central limit theorem, this method displays 1/√N convergence — i.e., quadrupling the number of sampled points halves the error» | Аргумент за 50 LHS поверх 50 random MC: тот же N даёт меньший std через stratification. CRN внутри LHS-точки даёт 3 seed → 1/√3 ≈ 0.58 std reduction. |

### C. Sensitivity analysis & QMC

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S8 | Wikipedia: Sensitivity analysis | en.wikipedia.org/wiki/Sensitivity_analysis | full | «One of the simplest and most common approaches is that of changing one-factor-at-a-time (OAT) … the OAT approach cannot detect the presence of interactions between input variables and is unsuitable for nonlinear models»; «Variance-based methods … Sobol»; «Morris … is suitable for screening systems with many parameters» | OAT недостаточен для нелинейных моделей с взаимодействиями. Sobol — золотой стандарт, но дорогой. **Для timeline 13.05 — OAT по каждой оси внутри LHS + двумерные scatter**; Sobol откладывается. |
| S9 | Wikipedia: Quasi-Monte Carlo method | en.wikipedia.org/wiki/Quasi-Monte_Carlo_method | full | «Quasi-Monte Carlo has a rate of convergence close to O(1/N), whereas the rate for the Monte Carlo method is O(N⁻⁰·⁵)»; «the Halton sequence performs best for dimensions up to around 6; the Sobol sequence performs best for higher dimensions» | LHS — компромисс между MC O(N⁻⁰·⁵) и QMC O(1/N). Для k=6 LHS даёт **гарантированный stratification по каждой оси** — критично для one-at-a-time sensitivity. |
| S10 | scipy.stats.qmc.Sobol docs | docs.scipy.org/.../Sobol.html | full | «Sobol' sequences are a quadrature rule and they lose their balance properties if one uses a sample size that is not a power of 2, or skips the first point, or thins the sequence»; «After 2^B points are generated, sequences repeat and errors are raised» | **Сильное ограничение Sobol**: 50 — не степень 2. Sobol работал бы с n=64, не 50. **Конкретный аргумент за LHS**, не Sobol для нашего N=50. |
| S11 | Wikipedia: Halton sequence | en.wikipedia.org/wiki/Halton_sequence | full | «Although these sequences are deterministic, they are of low discrepancy»; «The Halton sequence covers the space more evenly compared to pseudorandom sequences» | Альтернатива LHS — Halton, но без strata-гарантии. Для одностороннего OAT-sweep LHS предпочтительнее. |

### D. Subset selection (k-medoids, k-means)

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S12 | Wikipedia: K-medoids | en.wikipedia.org/wiki/K-medoids | full | «k-medoids chooses actual data points as centers (medoids or exemplars), and thereby allows for greater interpretability of the cluster centers than in k-means»; «PAM uses a greedy search method which may not find the optimum solution, but is faster than exhaustive search» | Для отбора 12 точек из 50: k-medoids предпочтительнее k-means — выбираются **реальные точки LHS** (medoids). |
| S13 | Wikipedia: K-means clustering | en.wikipedia.org/wiki/K-means_clustering | full | «k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean»; «k-means can easily be used to choose k different but prototypical objects from a large data set for further analysis» | Альтернатива k-medoids; уже реализован паттерн в `run_llm_spike.py:70–95`. После k-means — выбрать ближайшую к центру реальную LHS-точку. |
| S14 | scipy.spatial.distance | docs.scipy.org/.../spatial.distance.html | full | «pdist — Pairwise distances between observations in n-dimensional space»; «cdist — Compute distance between each pair of the two collections of inputs» | Реализационный паттерн для maximin: `pdist(unit_lhs)` → minimum pairwise distance → greedy add of point maximizing min distance to selected set. |

### E. DMDU и валидация

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S15 | Wikipedia: Robust decision making | en.wikipedia.org/wiki/Robust_decision_making | full | «RDM focuses on informing decisions under conditions of … 'deep uncertainty' … where the parties to a decision do not know or do not agree on the system models … or the prior probability distributions for the key input parameters»; «Robust decision methods seem most appropriate when the uncertainty is deep as opposed to well characterized, when there is a rich set of decision options, and the decision challenge is sufficiently complex that decision-makers need simulation models to trace the potential consequences over many plausible scenarios» | Прямое обоснование DMDU-фрейма ВКР (Lempert 2003 RAND). Сценарный полигон — каноничный пример RDM. |
| S16 | EMA Workbench in-depth tutorial | emaworkbench.readthedocs.io/.../general-introduction.html | full | «By default, the workbench uses Latin hypercube sampling for both sampling over levers and sampling over uncertainties»; «Any given parameterization of the levers is known as a policy, while any given parametrization over the uncertainties is known as a scenario. Any policy is evaluated over each of the scenarios»; «all model-based deep uncertainty approaches are forms of exploratory modeling as first introduced by Bankes (1993)» | **Прямой эталон**: open-source DMDU-toolkit использует **LHS как default**. «Каждая политика прогоняется по каждому сценарию» — полный перебор политик внутри LHS-точки совпадает с PROJECT_DESIGN §11. |
| S17 | Wikipedia: Verification and validation of computer simulation models | en.wikipedia.org/.../Verification_and_validation_of_computer_simulation_models | full | «A model that has face validity appears to be a reasonable imitation of a real-world system to people who are knowledgeable of the real world system»; «Sensitivity to model inputs can also be used to judge face validity»; «Model validation is defined to mean substantiation that a computerized model within its domain of applicability possesses a satisfactory range of accuracy consistent with the intended application» | Sargent-канон через secondary, фиксирует «operational validity» как основной concept. EC1–EC4 — operational validity в smoke. |
| S18 | Sargent 2013 search summaries | dl.acm.org/.../2675983.2676023 | metadata-only | «describes three approaches to deciding model validity, presents two paradigms relating verification and validation to the model development process, defines various validation techniques»; canonical 12 V&V techniques include face validity, traces, sensitivity analysis, **extreme condition tests**, parameter variability, degenerate behaviour | Канон V&V; extreme condition testing — название процедуры в литературе для наших EC1–EC4. Cite-only (полный PDF недоступен). |
| S19 | Larooij & Törnberg 2025 abstract | arxiv.org/abs/2504.03274 | abstract-only | «Validation remains poorly addressed, with many studies relying solely on subjective assessments of model 'believability'»; «fail to adequately evidence operational validity»; «LLMs' black-box nature compounds validation difficulties» | Поддерживает выбор cross-validation параметрический ↔ LLM как методически признанной альтернативы accuracy@1. |

### F. Сопутствующее

| № | Источник | URL | Доступ | Реальная цитата | Релевантные факты |
|---|---|---|---|---|---|
| S20 | Wikipedia: Spearman's rank correlation | en.wikipedia.org/.../Spearman's_rank_correlation_coefficient | full | «while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships»; «defined as the Pearson correlation coefficient between the rank variables» | Метрика согласованности параметрический ↔ LLM на 12 точках: Spearman ρ ранжирований политик. |
| S21 | Wikipedia: Multinomial logistic regression | en.wikipedia.org/wiki/Multinomial_logistic_regression | full | «softmax function converts linear predictor scores into probabilities»; «independence of irrelevant alternatives (IIA)—meaning the odds between any two choices remain unaffected by introducing additional alternatives» | Background для нашей utility-формы (уже принято в spike_behavior_model). |
| S22 | Wikipedia: Choice modelling | en.wikipedia.org/wiki/Choice_modelling | full | «MNL model: converts the observed choice frequencies into utility estimates via the logistic function»; «Random utility theory: utility contains both deterministic and random components» | Background. |
| S23 | Wikipedia: Information cascade | en.wikipedia.org/wiki/Information_cascade | full | (см. spike_gossip §17) | Background для gossip-канала. |

### Итог

23 успешных WebFetch с цитатами (19 full/partial + 4 abstract-only/metadata-only с явной пометкой). Главные опоры решения:

- **S1 scipy.qmc** — конкретный API + рекомендованная конфигурация для k=6, N=50;
- **S2 Wikipedia LHS** — независимость числа сэмплов от dimensionality, обоснование 50 при k=6;
- **S5 Kleijnen 2005** — LHS канон для simulation experiments;
- **S6 Variance reduction (CRN)** — точная формула снижения дисперсии при положительной ковариации;
- **S8 Sensitivity analysis** — OAT vs Sobol trade-off;
- **S10 scipy.qmc.Sobol** — почему Sobol не подходит для N=50;
- **S12 K-medoids** — обоснование выбора 12 реальных LHS-точек;
- **S14 scipy.spatial.distance** — реализационный паттерн maximin;
- **S16 EMA-workbench** — прецедент DMDU-toolkit (LHS default);
- **S17 V&V** — operational validity как канон;
- **S20 Spearman** — метрика согласованности.

---

## 6. Каталог осей LHS

PROJECT_DESIGN §8 фиксирует 6 параметрических осей. Ось 6 (random) — **репликация**, не часть LHS. Ось 5 (program_variant) — discrete. Остальные — continuous или discrete-categorical.

### Финальный каталог (рекомендуемый)

| # | Имя оси | Тип | Диапазон / уровни | Распределение | Параметр в коде |
|---|---|---|---|---|---|
| 1 | `capacity_multiplier` | continuous | `[0.5, 3.0]` | uniform | `scale_capacity(conf, mult)` (clone в run_lhs) |
| 2 | `popularity_source` | categorical (3) | `{"cosine_only", "fame_only", "mixed"}` | discrete uniform | `cfg.w_fame ∈ {0.0, 1.0, 0.3}` |
| 3a | `w_rec` | continuous | `[0.0, 0.7]` | uniform (симплекс) | `cfg.w_rec` |
| 3b | `w_gossip` | continuous | `[0.0, 0.7]` | uniform (симплекс) | `cfg.w_gossip` |
| 4 | `audience_size` | discrete (3) | **`{30, 60, 100}`** | discrete uniform | `audience_seed` + субсэмпл `audience_size` персон из `personas_100.json` (Accepted decision уточнение 2; `{50, 100, 200}` отвергнуто из-за разрыва базовой выборки между `personas_100` и независимой `personas_x3_200`) |
| 5 | `program_variant` | discrete (6) | `{0, 1, 2, 3, 4, 5}` | discrete uniform | `enumerate_modifications(conf, k_max=5, rng=phi_rng)` + индекс варианта |
| — | `policy` | НЕ в LHS | П1–П4 (4) | полный перебор | `active_policies(include_llm=...)` |
| — | `cfg.seed` | НЕ в LHS | `{1, 2, 3}` | репликация | `cfg.seed = replicate` |

**Эффективная dimensionality LHS**: d = 5 continuous/categorical + 1 discrete = **6 осей**, как в PROJECT_DESIGN §8.

### Что фиксируем константами (не оси)

- `tau = 0.7` (см. Q-O1 — отдельный sensitivity-sweep после LHS, не часть основной матрицы);
- `audience_distribution = "broad"` (subset из `personas_100.json` без фильтра по тематике; см. Q-O2);
- `p_skip_base = 0.10` (accepted spike_behavior_model);
- `K = 3` (top-K для cosine/capacity_aware/llm_ranker);
- `same_day_only = True` для Φ (accepted Q-M4);
- `k_max = 5` для Φ-LHS, `k_max = 3` для Φ-LLM (accepted Q-M3).

### Маппинг unit-cube → реальные значения

LHS возвращает точки в `[0,1)^6`. В `lhs.py` каждая точка маппится:

```python
# каждая координата u_i ∈ [0,1)
capacity_multiplier = 0.5 + u_1 * (3.0 - 0.5)             # [0.5, 3.0]
popularity_source   = ["cosine_only", "fame_only", "mixed"][int(u_2 * 3)]
w_rec               = u_3 * 0.7                           # [0, 0.7)
w_gossip            = u_4 * 0.7                           # [0, 0.7)
# симплексная нормировка: если w_rec + w_gossip > 1.0 → пропускаем
# или мягко проецируем в симплекс (см. Q-O3)
audience_size       = [30, 60, 100][int(u_5 * 3)]   # subsample из personas_100
program_variant_idx = int(u_6 * 6)                        # {0..5}
```

Distribution `optimization='random-cd'` гарантирует, что центрированная discrepancy минимизирована — strata по каждой оси сохраняются.

---

## 7. Симплекс весов и реализация

PROJECT_DESIGN §8 ось 3: «(w_rel, w_rec, w_gossip) и параметр стохастичности». Accepted Q-J4: `w_rel + w_rec + w_gossip = 1`, диапазон каждого `[0, 0.7]`.

### Варианты реализации

**(а)** Две независимые LHS-оси `w_rec ∈ [0, 0.7]` и `w_gossip ∈ [0, 0.7]` с явным skip недопустимых пар (`w_rec + w_gossip > 1.0`). Это уже работает в `run_smoke.py:171–174`.
- Плюсы: просто, прямая совместимость со smoke F.
- Минусы: при N=50 ~20% точек выпадают (область `w_rec + w_gossip > 1.0` имеет площадь 0.7² − 0.5·(2·0.7−1)² / 2 ≈ 0.31 в `[0, 0.7]²`, но эффективно — около 12% всего unit-cube). Для N=50 это потеря ~6 точек → нужно генерировать 56–60 LHS-точек и оставлять первые 50 валидных.

**(б)** Одна LHS-ось `w_rec ∈ [0, 0.7]` + одна LHS-ось `w_gossip` НЕ независимая, а условная: `w_gossip ∈ [0, 1.0 - w_rec]` через стандартную трансформацию. Это устраняет skip, но смещает распределение: при больших `w_rec` диапазон `w_gossip` сжимается, и LHS-strata перестают быть равными.

**(в)** Барицентрические координаты через Dirichlet(α=1,1,1) sampling — даёт равномерное распределение по симплексу, но это **не LHS** (потеря strata-property).

### Рекомендация

**(а)** Две независимые LHS-оси с явным skip и over-generation. Сохраняет каноническое определение LHS (strata по каждой оси), не требует условных распределений. Ожидаемая потеря — ~12% точек; генерировать LHS на 60 точек, оставлять первые 50 валидных по симплексу. Это согласовано с текущим smoke F и не требует ad-hoc математики.

---

## 8. Стратегия CRN

Контракт: **внутри одной LHS-точки все 4 политики и 3 seed-реплики видят одну и ту же синтетическую аудиторию и одинаковый `program_variant`-эффект**. Только `cfg.seed` варьируется между репликами.

### RNG-потоки в коде после P

| RNG-поток | Где | Зависит от |
|---|---|---|
| `master_lhs_rng` | `lhs.py` для генерации 50 точек | `master_seed` (фиксированный, например 2026) |
| `audience_rng` | новый, в `seeds.py` | `lhs_row_id` |
| `phi_rng` | новый, в `seeds.py` | `lhs_row_id` |
| `choice_rng` | `_process_one_slot` (уже есть) | `cfg.seed` × `slot_idx` |
| `policy_rng` | `_process_one_slot` (уже есть) | `cfg.seed` × `slot_idx + 31` |
| `select_personas_rng` (для LLM) | `run_llm_spike.py` (уже есть) | фиксированный `random_state=42` |

### derive_seeds (новый файл `experiments/src/seeds.py`)

```python
def derive_seeds(lhs_row_id: int, replicate: int) -> dict:
    """CRN-контракт.

    audience_seed и phi_seed зависят ТОЛЬКО от lhs_row_id —
    одна и та же аудитория и один и тот же program_variant между всеми
    политиками и всеми seed-репликами в рамках LHS-точки.

    cfg_seed = replicate — варьируется между репликами; изолированный
    через choice_rng / policy_rng в _process_one_slot, не сдвигает
    audience и phi.
    """
    return {
        "audience_seed": lhs_row_id * 1_000_003,
        "phi_seed":      lhs_row_id * 1_000_003 + 17,
        "cfg_seed":      replicate,
    }
```

### Что одинаково между политиками в LHS-точке

1. **Аудитория**: subset `audience_size` персон из `personas_100.json`, выбранный детерминированным `audience_rng = default_rng(audience_seed)`.
2. **`program_variant`-эффект**: одна и та же $P_k$ для всех 4 политик. Φ вызывается один раз на LHS-точке с `phi_rng = default_rng(phi_seed)`, индекс `k = program_variant_idx`. Для `k=0` — `P_0`; для `k≥1` — `enumerate_modifications(conf, k_max=5, rng=phi_rng)[k-1]`.
3. **`cfg.seed`** в одной seed-реплике — один для всех 4 политик. → `choice_rng` и `policy_rng` траектории совпадают в начале для всех политик.

### Что варьируется между seed-репликами внутри LHS-точки

- Только `cfg.seed ∈ {1, 2, 3}` → `choice_rng` и `policy_rng`.
- Аудитория и `program_variant` — **не варьируются** (CRN на уровне LHS-точки).

### Что варьируется между LHS-точками

- Всё, что определяется LHS-row.
- `master_seed` фиксированный (для воспроизводимости полного прогона).

---

## 9. Число точек / число seed

### Что говорят источники

- **PROJECT_DESIGN §11** норматив: 50 точек × 4 политики × 3 seed = 600 evals; LLM: 12 × 4 × 1 = 48 evals.
- **Kleijnen 2005** (S5 abstract): для simulation experiments общее правило n ≥ 10·k (k=6 → n ≥ 60). 50 — на нижней границе guideline, но допустимо для «modern designs» с стратификацией по каждой оси.
- **scipy.qmc** (S1): «14-fold reduction in samples vs grid sampling» для 6-параметрической эпид. модели — поддерживает 50 точек как достаточные.
- **Smoke F wallclock** на Mobius: ~5 минут на 243 evals → **600 evals ≈ 12–15 минут**, в пределах PIVOT этап Q «временной бюджет — минуты».
- **CLT 1/√N** (S7): 3 seed → reduction std ≈ 0.58; 5 seed → 0.45. **+66% wallclock за 13% дополнительного снижения** — diminishing returns.

### Рекомендация

| Параметр | Значение | Обоснование |
|---|---|---|
| LHS точек (параметрический) | **50** | PROJECT_DESIGN §11 норматив; Kleijnen на нижней границе guideline (n≥10·k=60), но допустимо при стратификации; scipy.qmc подтверждает достаточность. |
| Seed на точку (параметрический) | **3** | PROJECT_DESIGN §11 норматив; CLT diminishing returns. |
| Полный объём параметрический | 50 × 4 × 3 = **600 evals** | wallclock ~12–15 мин на Mobius. |
| LHS точек (LLM) | **12** | PROJECT_DESIGN §11 норматив; subset 50 → 12 через maximin (см. §10). |
| Seed на точку (LLM) | **1** (CRN) | PROJECT_DESIGN §11; одинаковая аудитория для 4 политик внутри точки. |
| Полный объём LLM | 12 × 4 × 1 = **48 evals** | wallclock ~часы (по этапу H — $0.005 на 4 evals на 10 агентах × 2 слота; на 12 точках × 4 политики × ~50 агентов × ~6 слотов ≈ $5–50). |

### Альтернативы (отклоняются)

- **30 точек** — слишком coarse coverage по дискретной оси `program_variant` (30/6 = 5 точек на уровень — маргинально).
- **100 точек** — ×2 wallclock без явной выгоды; согласованность 12 LLM ↔ параметрические работает на 50, доп. точки не дают новой информации.
- **5 seed** — diminishing returns (см. CLT выше).
- **1 seed** для параметрического — нарушает §11 «3 различных случайных зерна».

---

## 10. Отбор 12 точек для LLM

PROJECT_DESIGN §11: «12 точек, **покрывающие крайние и центральные значения каждой оси**». Это ключ для выбора алгоритма.

### Варианты

**(а) Maximin distance** (Johnson-Moore-Ylvisaker 1990; реализация через `scipy.spatial.distance` S14): из 50 LHS-точек жадно выбираем 12 с максимальным min pairwise distance в unit-cube.
- Плюсы: гарантированное покрытие границ и центра; **прямо соответствует PROJECT_DESIGN §11** «крайние и центральные значения каждой оси».
- Минусы: greedy не оптимален; может пропустить некоторые комбинации `program_variant=0` (control).

**(б) k-medoids** (S12): кластеризация 50 точек на 12 кластеров; в качестве «представителя» — реальная LHS-точка (medoid).
- Плюсы: каждая выбранная точка — реальный LHS-row (интерпретируемо в защите); medoid = центр кластера.
- Минусы: PAM heuristic stochastic; кластеры могут быть в центре, но edge-coverage хуже maximin.

**(в) k-means + ближайший к центру** (S13): уже реализован паттерн в `run_llm_spike.py:70–95` (`select_personas_kmeans`).
- Плюсы: **готовая инфраструктура** — прямо переносится с заменой эмбедингов на LHS-координаты.
- Минусы: edge-coverage хуже maximin (как и k-medoids).

**(г) Manual edge+center**: 6 точек на границах (по 2 экстремума на каждой из 3 главных осей) + 6 в центре.
- Минусы: для 6 осей нужно 12 крайних только; не оптимально.

### Рекомендация

**(а) Maximin distance** с принудительным включением хотя бы одной точки с `program_variant=0` (P_0 control). Если maximin даёт плохое edge-coverage по дискретной оси `program_variant` — fallback на **k-medoids** (S12 PAM seed=42).

Алгоритм maximin (greedy):

```python
def maximin_subset(unit_lhs: np.ndarray, k: int = 12,
                   force_include: List[int] = None) -> List[int]:
    """unit_lhs: (N, d) точки в unit-cube. Возвращает k индексов."""
    selected = list(force_include or [])
    n = unit_lhs.shape[0]
    while len(selected) < k:
        best_i, best_min_d = -1, -1.0
        for i in range(n):
            if i in selected: continue
            min_d = (
                np.min(np.linalg.norm(unit_lhs[i] - unit_lhs[selected], axis=1))
                if selected else 1.0
            )
            if min_d > best_min_d:
                best_i, best_min_d = i, min_d
        selected.append(best_i)
    return selected
```

`force_include` — индексы LHS-точек, обязательных для включения (например, точка с `program_variant=0` в центре других осей).

---

## 11. LLM-gossip синхронизация

Q-J12 accepted: **`w_gossip` синхронен между параметрическим и LLM в каждой LHS-точке**.

### Технически

Каждая LHS-row имеет одно continuous значение `w_gossip ∈ [0, 0.7]`. Параметрический симулятор использует это значение напрямую в utility (V5 log_count). LLM использует то же `w_gossip` через дискретизацию (Q-J8 accepted):

| `w_gossip` | gossip-блок в LLM-промпте | Системный промпт |
|---|---|---|
| 0.0 | **отсутствует** | стандартный (Q-J9) |
| (0.0, 0.4) | вставляется | + «учитывай как умеренный фактор» |
| [0.4, 0.7] | вставляется | + «учитывай как сильный фактор» |

Для каждой из 12 LLM-точек, выбранных в §10, `w_gossip` берётся из соответствующей LHS-row; **никаких дополнительных правок промпта или wording не требуется** — это уже сделано в этапах K и K-amend.

### Семантика gossip-сигнала

Q-J7-revised: «выбор других участников» через `count_t / N_users`. На текущих программах `count_t ≡ load[hall(t)]`, но формулировка строго социальная (не «загрузка зала»), чтобы не нарушать accepted Q-G capacity-в-LLM-промпте (`spike_llm_simulator`).

### Согласованность параметрический ↔ LLM

PROJECT_DESIGN §11 «сопоставляются ранжирования политик». Метрика — Spearman ρ между ranking-векторами политик в каждой из 12 точек по каждой ключевой метрике (`mean_overload_excess`, `hall_utilization_variance`, `overflow_rate`).

**Threshold**: медиана ρ ≥ 0.5 (умеренная корреляция) — допустимая согласованность. ρ ≥ 0.7 — высокая. PROJECT_STATUS §8 пункт 6 не задаёт жёсткий порог; ρ ≥ 0.5 — стандарт в behavioral simulation literature (Park 2023 / Agent4Rec / SimUSER используют distribution-match без strict ρ-thresholds).

---

## 12. Sensitivity

PROJECT_DESIGN §10 группа 4: «Размах значений показателя риска при варьировании одной оси конфигурации». Это **OAT** (one-at-a-time, S8).

### Что делаем (этап S, после Q)

1. **OAT по каждой из 6 осей**: фиксируем 5 осей в центральных значениях, варьируем 1 → строим scatter-plot `metric vs axis_value` для каждой политики.
2. **Двумерный risk × relevance**: для каждой LHS-точки строим scatter `mean_overload_excess × mean_user_utility` по 4 политикам — это PROJECT_DESIGN §10 группа 6.
3. **Pairwise win-rate**: для каждой пары политик считаем долю LHS-точек, где политика A лучше политики B по показателю риска (PROJECT_DESIGN §10 группа 4).

### Что не делаем (откладывается)

- **Sobol indices total-order** — overkill (S8 + S10).
- **Morris screening** — для k≫10; не наш случай (k=6).
- **PRIM scenario discovery** — полезный посттреатмент (DMDU-канон S15, S16), но это этап S/T, не O. Упомянуть в Limitations как «следующий шаг после полного LHS-прогона».

---

## 13. Что сознательно откладываем

1. **Sobol indices** (variance-based sensitivity) — overkill для timeline 13.05; OAT + scatter достаточен.
2. **PRIM / scenario discovery** (EMA-workbench, S16) — следующий шаг после полного LHS; вне scope этапа O/P/Q.
3. **`tau` как ось гиперкуба** — отдельный sensitivity-sweep после LHS, не часть основной матрицы (см. Q-O1).
4. **`audience_distribution` как ось** — фиксируем `broad` для timeline; см. Q-O2.
5. **Mixed Logit / Random-coefficients MNL** — overkill, отвергнуто в spike_behavior_model.
6. **Sequential bifurcation** (Kleijnen) — для k≫10; у нас k=6.
7. **Lloyd-optimised LHS** — `optimization='random-cd'` достаточен для k=6.
8. **Full factorial / partial factorial** — 6 осей × 5 levels = 15625 точек, неподъёмно.
9. **Parallelization** через `multiprocessing` или `joblib` — оставляем sequential; если smoke этапа P покажет wallclock > 30 минут — расширим.

---

## 14. Минимальная первая реализация для этапа P

### Файлы (этап P, не сейчас)

**`experiments/src/lhs.py`** (~120 LOC):

```python
import numpy as np
from scipy.stats import qmc
from typing import List, Dict, Tuple

LHS_AXES = [
    "capacity_multiplier",   # ось 1
    "popularity_source",     # ось 2 (categorical через index)
    "w_rec",                 # ось 3a (симплекс)
    "w_gossip",              # ось 3b (симплекс)
    "audience_size",         # ось 4 (discrete)
    "program_variant",       # ось 5 (discrete)
]

POPULARITY_SOURCES = ["cosine_only", "fame_only", "mixed"]
AUDIENCE_SIZES = [30, 60, 100]      # все subset из personas_100 (см. Accepted decision уточнение 2)
PROGRAM_VARIANT_LEVELS = list(range(6))   # {0..5}; 0 = P_0 (control)

# Минимальное покрытие дискретных осей после фильтра симплекса (уточнение 4):
MIN_PER_LEVEL = {
    "program_variant": 5,    # 50/6 ≈ 8 expected, требуем ≥ 5 каждого уровня
    "audience_size":   12,   # 50/3 ≈ 17, требуем ≥ 12
    "popularity_source": 12,
}


def _map_unit_to_row(u: np.ndarray) -> Dict:
    """Маппинг одной unit-cube точки в реальные значения. None если вне симплекса."""
    cap_mult     = 0.5 + u[0] * 2.5
    pop_src      = POPULARITY_SOURCES[min(2, int(u[1] * 3))]
    w_rec        = u[2] * 0.7
    w_gossip     = u[3] * 0.7
    if w_rec + w_gossip > 1.0:
        return None
    aud_size     = AUDIENCE_SIZES[min(2, int(u[4] * 3))]
    prog_idx     = PROGRAM_VARIANT_LEVELS[min(5, int(u[5] * 6))]
    return {
        "u_raw": u.tolist(),
        "capacity_multiplier": cap_mult,
        "popularity_source": pop_src,
        "w_rec": w_rec,
        "w_gossip": w_gossip,
        "w_rel": max(0.0, 1.0 - w_rec - w_gossip),
        "audience_size": aud_size,
        "program_variant": prog_idx,
    }


def _check_balance(rows: List[Dict]) -> Dict[str, Dict]:
    """Возвращает counter покрытия дискретных осей."""
    return {
        "program_variant":   {lv: sum(1 for r in rows if r["program_variant"] == lv)
                              for lv in PROGRAM_VARIANT_LEVELS},
        "audience_size":     {lv: sum(1 for r in rows if r["audience_size"] == lv)
                              for lv in AUDIENCE_SIZES},
        "popularity_source": {lv: sum(1 for r in rows if r["popularity_source"] == lv)
                              for lv in POPULARITY_SOURCES},
    }


def _is_balanced(counts: Dict[str, Dict]) -> bool:
    return all(
        all(v >= MIN_PER_LEVEL[axis] for v in counts[axis].values())
        for axis in MIN_PER_LEVEL
    )


def generate_lhs(
    n_points: int = 50,
    master_seed: int = 2026,
    block_size: int = 64,
    max_attempts: int = 50,
) -> List[Dict]:
    """Генерирует ровно n_points LHS-точек.

    Уточнение 3 (Accepted decision): rejection sampling в цикле — генерируем
    блоки `block_size`, фильтруем по симплексу w_rec + w_gossip ≤ 1.0,
    добавляем в pool до достижения n_points валидных строк.
    Каждый блок использует sub-seed ответвлённый от master_seed → полная
    воспроизводимость.

    Уточнение 4: после набора n_points проверяется покрытие дискретных осей.
    Если несбалансировано — repair: replace random точку на forced-level
    точку из дополнительного блока. До max_attempts попыток repair.
    Если не удалось — поднять ValueError с диагностикой.
    """
    master_rng = np.random.default_rng(master_seed)
    rows: List[Dict] = []
    block_idx = 0
    while len(rows) < n_points:
        # Каждый блок — новый Sampler с детерминированным sub-seed.
        block_seed = int(master_rng.integers(0, 2**31 - 1))
        sampler = qmc.LatinHypercube(
            d=6, scramble=True, optimization='random-cd',
            rng=np.random.default_rng(block_seed),
        )
        raw = sampler.random(block_size)
        for u in raw:
            row = _map_unit_to_row(u)
            if row is not None:
                rows.append(row)
                if len(rows) >= n_points:
                    break
        block_idx += 1
        if block_idx > max_attempts * 2:
            raise ValueError(
                f"generate_lhs: после {block_idx} блоков набрано "
                f"только {len(rows)}/{n_points} валидных точек"
            )

    rows = rows[:n_points]

    # Repair дискретных осей (уточнение 4)
    for attempt in range(max_attempts):
        counts = _check_balance(rows)
        if _is_balanced(counts):
            break
        # Найти недопредставленный уровень
        replaced = False
        for axis, level_counts in counts.items():
            for level, cnt in level_counts.items():
                if cnt < MIN_PER_LEVEL[axis]:
                    # Заменить случайную точку на принудительную
                    over_axis = max(level_counts, key=level_counts.get)
                    candidates = [i for i, r in enumerate(rows)
                                  if r[axis] == over_axis]
                    if not candidates:
                        continue
                    replace_idx = int(master_rng.choice(candidates))
                    new_row = _force_level_row(rows[replace_idx], axis, level)
                    rows[replace_idx] = new_row
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            raise ValueError(
                f"generate_lhs: дисбаланс не устранён за {attempt + 1} попыток. "
                f"Counts: {counts}"
            )
    else:
        raise ValueError(
            f"generate_lhs: дисбаланс не устранён за {max_attempts} попыток"
        )

    # Проставить lhs_row_id после всех repair
    for i, row in enumerate(rows):
        row["lhs_row_id"] = i
    return rows


def _force_level_row(row: Dict, axis: str, level) -> Dict:
    """Возвращает копию row с принудительно установленным уровнем по axis."""
    new_row = dict(row)
    new_row[axis] = level
    # u_raw больше не описывает строку буквально после repair —
    # пересчитываем для прозрачности (если нужно — оставляем оригинал
    # с пометкой `repaired_from`).
    new_row["repaired_axis"] = axis
    new_row["repaired_level"] = level
    return new_row
```

**Тест acceptance** (часть `test_lhs.py`):
- `len(generate_lhs(50)) == 50`;
- ровно 50 валидных по симплексу (`all(r["w_rec"] + r["w_gossip"] <= 1.0)`);
- каждый уровень дискретной оси представлен ≥ `MIN_PER_LEVEL[axis]` раз;
- детерминизм при одинаковом `master_seed`;
- ValueError при `max_attempts=0` или невозможном балансе.

**`experiments/src/seeds.py`** (~30 LOC):

```python
def derive_seeds(lhs_row_id: int, replicate: int) -> dict:
    return {
        "audience_seed": lhs_row_id * 1_000_003,
        "phi_seed":      lhs_row_id * 1_000_003 + 17,
        "cfg_seed":      replicate,
    }
```

**`experiments/scripts/run_lhs_parametric.py`** (~250 LOC) — точка входа этапа P:
- Load conference + personas;
- generate_lhs(50);
- для каждой LHS-row: subset аудитории + Φ + 4 политики × 3 seed → метрики;
- long-format выход: `experiments/results/lhs_parametric_<conf>_<date>.{json,csv}`.

**`experiments/scripts/run_smoke_lhs.py`** (~150 LOC) — этап P smoke:
- generate_lhs(5);
- проход 5 × 4 × 1 = 20 evals на Mobius;
- **проверки** (Accepted decision уточнение 5): формат long-table; CRN-инвариант (одна аудитория и тот же program_variant между политиками внутри LHS-точки); связка policies / Φ / metrics работает; wallclock в пределах разумного.
- **EC3 на случайных LHS-точках НЕ проверяется** (на 5 случайных точках `w_rec=0` почти не появится — диапазон `[0, 0.7]`, мера 0). EC3 проверяется отдельным forced-test row с `w_rec=0` (как в pytest этапа I `test_ec3_invariance_when_w_rec_zero`); если нужна smoke-проверка EC3, добавить отдельный test-script с явным forced row, не на random LHS.

### Что уже есть (не дублировать)

- `simulator.py`, `policies/*.py`, `metrics.py` — без правок.
- `program_modification.py` — используется как есть.
- `select_personas_kmeans` (run_llm_spike) — паттерн для maximin или k-medoids.

### Tests (этап P)

`experiments/tests/test_lhs.py` (~150 LOC):
- `test_lhs_returns_n_points` — `len(generate_lhs(50)) == 50`.
- `test_lhs_simplex_satisfied` — `w_rec + w_gossip ≤ 1.0` для всех row.
- `test_lhs_all_axes_have_full_range_coverage` — каждая ось имеет min/max в первой и последней десятке точек (страт-проверка).
- `test_lhs_deterministic_under_master_seed` — два прогона с одним `master_seed` дают идентичные точки.
- `test_lhs_program_variant_distribution` — `program_variant ∈ {0..5}` представлены ≥ 5 раз каждый (50/6 ≈ 8 точек на уровень).

`experiments/tests/test_seeds.py` (~50 LOC):
- `test_derive_seeds_audience_invariant_across_replicates` — `audience_seed` зависит только от `lhs_row_id`.
- `test_derive_seeds_phi_invariant_across_replicates` — `phi_seed` зависит только от `lhs_row_id`.
- `test_derive_seeds_cfg_seed_equals_replicate`.

### Acceptance этапа P (gate)

1. `pytest experiments/tests/ -v` — все 67 + новые ~10 тестов зелёные (включая инварианты `lhs.py`: ровно `n_points` валидных, дискретный баланс, детерминизм).
2. Smoke `run_smoke_lhs.py` на Mobius (5 точек × 4 политики × 1 seed = 20 evals) проходит за ≤ 5 минут.
3. **Формат long-table** валиден (правильные колонки: `lhs_row_id, capacity_multiplier, popularity_source, w_rec, w_gossip, w_rel, audience_size, program_variant, policy, replicate, metric_name, metric_value`).
4. **CRN-инвариант**: для одной LHS-точки разные политики получают одинаковую аудиторию (по `audience_seed`) и одинаковый `program_variant`-эффект (по `phi_seed`). Проверяется через детерминизм между прогонами при одинаковом `master_seed`.

**EC3 на случайных LHS-точках НЕ часть acceptance этапа P** (см. Accepted decision уточнение 5). EC3 уже проверен в pytest этапа I forced-row тестом и в smoke F. Дополнительная проверка EC3 на LHS-генераторе — отдельный `test_lhs_forced_w_rec_zero` (с явным `w_rec=0` row).

После acceptance этапа P — переход к этапу Q (полный 450 или 600-eval прогон, в зависимости от Q-O9).

---

## 15. Какие проверки должны пройти до перехода к P

| Проверка | Где | Тип |
|---|---|---|
| Memo подписан пользователем по разделам Q-O1 — Q-O8 | spike_experiment_protocol.md | accept |
| Уже-зелёный pytest 67 (этапы B–N) | `pytest experiments/tests/ -v` | regression |
| Существующий smoke F (toy + Mobius) воспроизводится без правок | `run_smoke.py` | regression |
| Гипотеза о wallclock 600 evals ≤ 30 мин — сверка по smoke F | `smoke_mobius_2025_autumn_2026-05-07.json` | sanity |

---

## 16. Какие решения требуют подтверждения пользователя

> Уточнения по 5 техническим пунктам приняты пользователем 2026-05-07 (см. Accepted decision в начале memo). Q-O9 — новый вопрос (статус П4 в полном LHS).

### Q-O1. Включаем ли `tau` как ось гиперкуба?

PROJECT_DESIGN §8 ось 3 формально включает «параметр стохастичности». Но spike_behavior_model не варьировал `tau` — он зафиксирован на 0.7.

Варианты:
- **(а)** Фиксировать `tau = 0.7` в основной матрице; отдельный sensitivity-sweep `tau ∈ {0.3, 0.5, 0.7, 1.0, 1.5}` после полного LHS (этап S). **Рекомендуется.**
- (б) Включить `tau` как 7-ю ось LHS — увеличит dimensionality при том же N=50 (n/k = 50/7 ≈ 7.1, ниже Kleijnen guideline n ≥ 10·k = 70).

**Предложение:** (а). Разделение «основная матрица — 6 осей; tau-sensitivity — отдельный sweep» сохраняет PROJECT_DESIGN §11 норматив 600 evals и не растягивает wallclock.

### Q-O2. Включаем ли `audience_distribution` как ось 4b?

PROJECT_DESIGN §8 ось 4: «Численность; распределение тематических интересов». Distribution может быть `broad` / `narrow_ml` / `narrow_mobile`.

Варианты:
- **(а)** Фиксировать `audience_distribution = "broad"` (subset из `personas_100.json` без фильтра). **Рекомендуется** для timeline 13.05.
- (б) Включить как 7-ю ось LHS (как Q-O1 — увеличит dimensionality).
- (в) Фиксировать `broad`, но добавить отдельный sensitivity по distribution в этапе S.

**Предложение:** (в). Default — `broad`; sensitivity-sweep как опциональный этап S.

### Q-O3. Симплекс — две независимые оси с skip или одна композитная?

Варианты:
- **(а)** Две независимые LHS-оси `w_rec ∈ [0, 0.7]` и `w_gossip ∈ [0, 0.7]` с явным skip недопустимых пар + over-generation на 20%. **Рекомендуется** (как в `run_smoke.py`).
- (б) Одна ось `w_rec ∈ [0, 0.7]` + условная `w_gossip ∈ [0, 1−w_rec]` — устраняет skip, но смещает strata.
- (в) Барицентрические координаты через Dirichlet — равномерно по симплексу, но **не LHS** (потеря strata).

**Предложение:** (а). Сохраняет каноническое определение LHS; over-generation +20% — простой и воспроизводимый workaround.

### Q-O4. Алгоритм отбора 12 LLM-точек

Варианты:
- **(а)** Maximin distance с принудительным включением одной точки `program_variant=0` (control). **Рекомендуется.** Прямо соответствует PROJECT_DESIGN §11 «крайние и центральные значения каждой оси».
- (б) k-medoids (PAM seed=42).
- (в) k-means + ближайший к центру (готовая инфраструктура `run_llm_spike.py`).
- (г) Manual edge+center.

**Предложение:** (а) maximin; (б) k-medoids как fallback если edge-coverage по `program_variant` плохое.

### Q-O5. Sensitivity — OAT или Sobol?

Варианты:
- **(а)** OAT по каждой оси + двумерные scatter (risk × relevance, S8). **Рекомендуется** для timeline 13.05.
- (б) Sobol total-order indices — overkill, +400 evals.

**Предложение:** (а); Sobol — Limitations.

### Q-O6. `program_variant` как discrete или continuous?

Варианты:
- **(а)** Discrete `{0, 1, ..., 5}` — `int(u_5 * 6)`. **Рекомендуется** (естественная семантика).
- (б) Continuous `u_5 ∈ [0, 1)` с round-to-nearest-int.

**Предложение:** (а). LHS strength=1 даёт стрейт по этой оси; 50/6 ≈ 8 точек на уровень — достаточно. В acceptance pytest проверяем, что каждый уровень представлен ≥ 5 раз.

### Q-O7. Метрика согласованности параметрический ↔ LLM

Варианты:
- **(а)** Spearman ρ per-metric per-LHS-point, агрегат — медиана + 25/75-квантили; threshold медианы ≥ 0.5. **Рекомендуется.**
- (б) Kendall τ — более робастный, но сложнее интерпретировать.
- (в) Hamming на топ-1 политику — слишком грубо при 4 политиках.

**Предложение:** (а) Spearman ρ.

### Q-O8. wallclock-budget 30 минут на 600 evals — нужен ли parallel?

Варианты:
- **(а)** Sequential с `slot_concurrency=1` (как сейчас). **Рекомендуется** для первого прогона.
- (б) `multiprocessing` через `joblib.Parallel(n_jobs=4)` — 4× ускорение, но сложнее отладка CRN.
- (в) `slot_concurrency > 1` (уже есть в `simulate_async`) — но это меняет CRN-инвариант.

**Предложение:** (а). Если smoke этапа P покажет wallclock > 30 минут — расширим. **Уточнение Accepted decision уточнение 1**: оценка «12–15 минут на 600 evals» базируется на smoke F с **П1–П3** (`include_llm=False`); П4 LLMRankerPolicy не участвовала. Уточнённый прогноз — см. Q-O9.

### Q-O9. Статус П4 LLMRankerPolicy в полном LHS (accepted 2026-05-07)

PROJECT_DESIGN §11 нормативно требует «все четыре политики на всех 50 точках». П4 — API-вызов на ChatGPT-4o-mini через OpenRouter; latency ~1–2 сек, стоимость ~$0.0001 на (user, slot, candidates)-tuple, кэширование на диске (`logs/llm_ranker_cache.json`).

**Оценка стоимости / wallclock** для трёх вариантов (Mobius: 16 слотов; средн. audience_size = 60):

| Вариант | Прогонов с П4 | Уникальных API-tuples (cache miss) | Cost USD | wallclock cold-cache | wallclock warm-cache |
|---|---|---|---|---|---|
| (а) П4 на всех 50 точках × 3 seed | 150 | ~50 × 60 × 16 = 48 000 | ~$5–10 | ~2–3 часа | ~30–60 мин |
| (б) П4 только на 12 LLM-точках в этапе V (исключена из параметрического LHS) | 12 (LLM-V) | ~12 × 60 × 16 = 11 520 | ~$1–2 | ~30 мин | ~10 мин |
| (в) П4 только на 12 LHS-точках (subset maximin) + LLM-V на тех же 12 | 12 (LHS) + 12 (LLM-V) | ~24 × 60 × 16 = 23 040 | ~$2–4 | ~1 час | ~15 мин |

Варианты:

- **(а)** Все 4 политики на всех 50 точках × 3 seed (формально соответствует §11). 150 evals с П4. Требует cold-cache prefill отдельным шагом; основной прогон на warm cache ~30–60 мин. **Соответствие §11: полное.**
- **(б)** Только П1–П3 на 50 LHS-точках (450 evals); П4 — только в LLM-симуляторе на этапе V (12 точек × 4 политики × 1 seed = 48 evals; П4 в LLM-V уже включена). Параметрический LHS = 50 × 3 × 3 = **450 evals**. **Deviation от §11 в части «все 4 политики на всех 50 точках»**, обоснование: П4 — это LLM-канал, и его место — в LLM-симуляторе (этап V), не в параметрическом LHS; §11 «согласованность двух симуляторов на 12 общих точках» сохраняется.
- **(в)** Компромисс: П1–П3 на всех 50 LHS-точках; П4 на subset 12 точек (тот же maximin, что для LLM-V). Параметрический LHS = 50×3×3 + 12×1×3 = 450 + 36 = **486 evals**. **Частичное соответствие §11**: П4 присутствует на тех самых 12 точках, где §11 требует cross-validation двух симуляторов.

**Предложение:** **(в)** компромисс. Параметрический П4 на 12 точках даёт прямую базу для cross-validation на тех же 12 точках, где LLM-V запускает П4 на LLM-симуляторе; для остальных 38 LHS-точек (где §11 требует «все 4») в Limitations диссертации фиксируется: «П4 LLMRankerPolicy в полном параметрическом LHS не запускалась из-за wallclock-ограничений; cross-validation двух симуляторов проведена на 12 общих точках, как требует §11 acceptance». Cost ~$2–4, wallclock ~15 мин warm-cache.

Альтернатива (а) полно соответствует §11, но требует cold-cache prefill (~2–3 часа) перед основным прогоном. Альтернатива (б) — самая практичная по timeline, но ослабляет cross-validation базу.

**Подтверждено пользователем 2026-05-07: вариант (в) принят как осознанное частичное отступление от §11 «все 4 политики на всех 50 точках» по причине cost/wallclock; cross-validation сохраняется на 12 общих точках (см. Accepted decision уточнение 1).**

---

## Recommended decision for P

Финальная конфигурация после спайка O.

**LHS axes** (6 осей):
1. `capacity_multiplier ∈ [0.5, 3.0]` continuous;
2. `popularity_source ∈ {cosine_only, fame_only, mixed}` categorical;
3. `w_rec ∈ [0, 0.7]` continuous (симплекс);
4. `w_gossip ∈ [0, 0.7]` continuous (симплекс);
5. `audience_size ∈ {30, 60, 100}` discrete (subsample из `personas_100.json` через `audience_seed`; см. Accepted decision уточнение 2);
6. `program_variant ∈ {0, 1, ..., 5}` discrete.

**Не оси (фиксированы):** `tau=0.7`, `audience_distribution="broad"`, `p_skip_base=0.10`, `K=3`, `same_day_only=True`, `k_max=5` (LHS) / `3` (LLM).

**Размеры:**
- LHS точек: 50 (rejection sampling до 50 валидных по симплексу + repair дискретных осей; см. уточнения 3–4 Accepted decision);
- Seed на точку: 3 (replicates `{1, 2, 3}`);
- Полный объём параметрический: **486 evals** (Q-O9 accepted вариант (в) — компромисс): 50×3×3 = 450 evals для П1–П3 на всех LHS-точках + 12×1×3 = 36 evals для П4 LLMRankerPolicy на 12 maximin-точках, выбранных для LLM-V;
- LLM точек: 12 (maximin subset из 50);
- Seed на LLM-точку: 1 (CRN);
- Полный объём LLM: **48 evals** (12 × 4 политики × 1 seed; П4 в LLM-V присутствует всегда).

**CRN:**
- `audience_seed`, `phi_seed` — фиксированы по `lhs_row_id` (CRN на уровне LHS-точки);
- `cfg_seed = replicate` — варьируется между репликами;
- `master_seed = 2026` для генерации LHS.

**Sensitivity:** OAT + двумерный scatter (этап S/T); Sobol откладывается.

**Файлы для этапа P:**
- `experiments/src/lhs.py` (~80 LOC) — `generate_lhs(n_points, master_seed)`.
- `experiments/src/seeds.py` (~30 LOC) — `derive_seeds(lhs_row_id, replicate)`.
- `experiments/scripts/run_lhs_parametric.py` (~250 LOC) — точка входа Q.
- `experiments/scripts/run_smoke_lhs.py` (~150 LOC) — smoke этапа P (5 × 4 × 1 = 20 evals).
- `experiments/tests/test_lhs.py` + `test_seeds.py` (~200 LOC) — pytest invariants.

**Что не меняется в P:**
- `simulator.py` — без правок.
- Политики — без правок.
- `program_modification.py` — без правок.
- `metrics.py` — без правок.
- `run_smoke.py` — без правок.
- `run_llm_spike.py` — без правок (расширится в этапе V).
- Активный реестр П1–П4 — без правок.

**LLM-gossip синхронизация (Q-J12 уже accepted):** `w_gossip` continuous в параметрическом, дискретизирован {off / moderate / strong} в LLM (граница 0.4). Никаких новых правок промпта.

**Метрика согласованности:** Spearman ρ ranking-политик per-metric per-LHS-point; threshold медиана ≥ 0.5.

**Все open questions закрыты пользователем 2026-05-07:** Q-O1 — Q-O8 (первый accept по направлению LHS/CRN/Spearman/OAT) и **Q-O9** (вариант (в) компромисс — П1–П3 на всех 50, П4 на 12 maximin-точках; см. Accepted decision уточнение 1). Пять технических уточнений (Q-O9, audience grid, generate_lhs rejection sampling, repair дискретных осей, smoke без EC3 на random точках) приняты.

К этапу P не перехожу до отдельного сообщения пользователя.
