# Design-spike: модель поведения участника

Дата: 2026-05-07
Этап: C (PIVOT_IMPLEMENTATION_PLAN r5).
Статус: accepted for implementation in stages D–E.

> Memo evidence-first. Сначала идёт research log и реально изученные источники, затем требования и обзор вариантов, и только в конце — рекомендация. Это сознательное отличие от первой попытки этого spike (`_failed_spike_behavior_model_2026-05-07.md`), в которой web-search использовался только по сниппетам, а ключевые статьи не были реально открыты.

---

## Accepted decision

Статус: принято пользователем для этапов D–E.

Принятые решения:

1. Первая модель поведения для этапов D–E: `U = w_rel * rel + w_rec * rec(t, hat_pi)`.
2. `consider_ids = slot.talk_ids` всегда; политика больше не ограничивает choice set через top-K-фильтр.
3. `rec(t, hat_pi) = 1{t ∈ recs}` для первой реализации.
4. Для базовой модели без gossip используем нормировку `w_rel + w_rec = 1`.
5. Capacity-effect удаляется из utility участника полностью.
6. Capacity-effect живёт только в политике П3 `capacity_aware`.
7. `user_compliance` / `calibrated_compliance` остаются как legacy под флагами, default off, вне основного эксперимента.
8. `p_skip_base = 0.10` оставляем для первой реализации; в отдельных sanity / EC-тестах можно фиксировать skip = 0, если нужно проверить чистую логику выбора.
9. Gossip не включается в этапы D–E; он остаётся обязательным отдельным инкрементом этапов J–L.

---

## 1. Проблема

Нужно зафиксировать форму функции полезности участника `U(t | i, slot, hat_pi, L_t)` для первой работающей реализации параметрического симулятора. Решение — вход в этап E (минимальное изменение `simulator.py`). Без решения по форме utility нельзя ни запускать toy-cases (этап D), ни проходить extreme condition tests (этап I).

Ключевое требование PROJECT_DESIGN §9: «На итоговое распределение посещений политика влияет только через компонент w_rec функции полезности участника». Это требование операционализируется через extreme conditions §11:

- **EC3.** При `w_rec → 0` различие политик по показателям загрузки → 0.
- **EC4.** При `w_rec → 1` различие политик по показателям загрузки максимально.

Если хотя бы одно из этих свойств нарушено, переход к содержательным выводам блокируется (PROJECT_DESIGN §11, в традиции Sargent 2013).

---

## 2. Текущая реализация в репозитории

Файл: `experiments/src/simulator.py`, функция `_process_one_slot` (строки 292–423).

Текущая utility (строки 384–386):

```python
effective_rel = (1 - cfg.w_fame) * rel + cfg.w_fame * t.fame
u = effective_rel - cfg.lambda_overflow * max(0.0, load_frac - 0.85)
```

Свойства текущей реализации:

1. **Capacity-эффект зашит в utility пользователя** через `-λ·max(0, load_frac - 0.85)`. Параллельно политика `CapacityAwarePolicy` (`experiments/src/policies/capacity_aware_policy.py`, строка 36) тоже штрафует загрузку: `score = sim - α·load_frac`. Capacity-канал работает дважды — и в utility агента, и в скоринге политики.
2. **Рекомендация работает через top-K-фильтр consideration set** (строки 335–357): `consider_ids = recs` (если compliance не активен). Это эквивалентно `w_rec ≈ 1` всегда; градуированной зависимости от `w_rec` нет.
3. **Compliance моделируется двумя legacy-механизмами**: `cfg.user_compliance` (бернуллиевский) и `cfg.use_calibrated_compliance` (трёхтипная B/C/A модель из `calibrate_compliance_meetup.py`). Ни тот, ни другой не дают `w_rec`-канала; они только переключают consider_ids между `recs` и `slot.talk_ids`.
4. **Outside option / no-choice альтернатива** реализован через `p_skip_base = 0.10` (строки 394–404): доля массы 0.10 всегда уходит на «отказ».

Свойства, которые ломаются текущей формой:

- **EC3 не выполнима.** При `w_rec → 0`, в текущей формуле `w_rec` отсутствует как параметр — top-K-фильтр действует жёстко. Различие политик не устраняется.
- **PROJECT_DESIGN §9 нарушено.** Политика П3 имеет двойной канал влияния: через свой top-K и через capacity-term, который агент видит сам.
- **Двойное определение compliance.** `user_compliance` и `calibrated_compliance` — два не-ортогональных механизма, оба не выводимы из `w_rec`.

---

## 3. Требования

### Из PROJECT_DESIGN
- **§7 Модуль модели поведения участника:** «В функцию полезности входят три компонента, каждый управляется отдельным весом: w_rel, w_rec, w_gossip. Кроме весов — параметр стохастичности.»
- **§8 Параметрические оси:** Ось 3 — `(w_rel, w_rec, w_gossip)` и параметр стохастичности.
- **§9 Состав политик:** «На итоговое распределение посещений политика влияет только через компонент w_rec функции полезности участника. При w_rec → 0 политики неразличимы по итогу; при w_rec → 1 выбор политики определяет итог полностью.»
- **§11 Верификация в граничных условиях:** EC1–EC4, в традиции Sargent 2013.
- **§13 Допущения:** «Модель индивидуального выбора имеет параметрическую форму. Калибровка её параметров на внешних данных производится в одном из вариантов реализации, но в обязательный результат работы не входит.»

### Из PROJECT_STATUS
- **§5 Стоп-лист:** B1/accuracy@1 = 0.918 — не валидация (utility leakage); cap-aware MMR — не центральный результат.
- **§7 Текущее направление:** 4 политики П1–П4, MMR/gossip-вход — параметрические модификаторы.
- **§8 Валидация:** Meetup используется только как distribution-level якорь, не B1.

### Из PIVOT_IMPLEMENTATION_PLAN r5
- **Принцип 1:** первая минимальная модель — `rel + rec`. Gossip — отдельный плановый инкремент (этапы J–K).
- **Этап E:** минимальная правка `simulator.py` под `U = w_rel·rel + w_rec·rec`, `consider_ids = slot.talk_ids` всегда.

---

## 4. Research log

Расширенный design-spike по правилу раздела 7 PIVOT_IMPLEMENTATION_PLAN r5 выполнен через отдельный research-subagent с time-boxed бюджетом. Subagent изучал предметную область и возвращал research brief; написание самого memo ведётся в основной сессии.

### 4.1 Время

- **start time (subagent):** 2026-05-07 02:12:27 MSK
- **start timestamp:** 1778109147 (epoch seconds)
- **end time (subagent):** 2026-05-07 02:19:08 MSK
- **end timestamp:** 1778109548
- **elapsed seconds:** 401 (≥ 300 — минимальный research budget)
- Sleep / искусственное ожидание не использовались. Время потрачено на реальное I/O (Read локальных файлов и WebFetch / WebSearch внешних источников).

### 4.2 Изученные файлы кода

- `experiments/src/simulator.py` — текущая utility, top-K-фильтр через consider_ids, p_skip как outside option.
- `experiments/src/policies/registry.py` — фиксированный реестр П1–П4.
- `experiments/src/policies/no_policy.py` — П1 возвращает пустой recs.
- `experiments/src/policies/cosine_policy.py` — П2 top-K по cosine.
- `experiments/src/policies/capacity_aware_policy.py` — П3 уже включает `score = sim - α·load_frac`; с текущим simulator capacity-канал работает дважды.
- `experiments/src/policies/calibrated_policy.py` — Steck-style re-ranker (KL по категориям). Терминологически совпадает с `calibrated_compliance` из simulator.py, но это **другая сущность** (re-ranker, не модель compliance).

### 4.3 Изученные документы проекта

- `PROJECT_DESIGN.md` — постановка (§7, §9, §11, §13).
- `PROJECT_STATUS.md` — стоп-лист (§5), валидация (§8).
- `.claude/memory/research_field_survey_2026-05-04.md` — карта подходов.
- `.claude/memory/reference_validation_defense.md` — must-cite, distribution-match.
- `materials/_legacy/research-conference-recsys-deep-2026-05.md` — старый deep research по conference recsys.
- `materials/_legacy/research-recon-deep-2026-05.md` — старый deep research по ReCon.

### 4.4 Внешние источники, реально открытые (с пометками о доступности)

См. раздел 5 — отдельная таблица. Каждый источник снабжён пометкой о фактической доступности: `full / abstract-only / metadata-only / derived-only / not-accessible`.

### 4.5 Что оказалось нерелевантным и почему

- **Cascade click model / position bias** (стандарт IR) — не применимо: доклад выбирается заранее на день, не «прокручивается» в ленте.
- **Generative agents с памятью / reflection** (Park 2023, RecAgent) — это путь LLM-агентского симулятора, который у нас уже есть как параллельный второй симулятор. Параметрический МДП-симулятор должен оставаться замкнутой формы.
- **Mixed Logit / Random-coefficients MNL** — научно богаче, но требует калибровки β-распределений на real preference data, которой у нас нет; гетерогенность профилей уже учтена через `personas`.

### 4.6 Открытые вопросы по итогам research

См. раздел 12 (открытые решения для подтверждения пользователем).

---

## 5. Обзор реально изученных источников

Источники сгруппированы по направлениям; каждый снабжён реальной пометкой о доступности.

### A. Discrete choice / MNL — фундаментальная база

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S1 | Train K. (2009). *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge UP. URL: https://eml.berkeley.edu/books/choice2.html, Ch.3 PDF: https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf | textbook | partial (binary PDF, формулы извлечены) | Каноническая форма `P(i) = exp(V_i)/Σ_j exp(V_j)` под `U_i = V_i + ε_i`, `ε ~ iid Gumbel`. IIA свойство. Линейная (additive) систематическая часть V_i — стандарт. Масштаб (температура) идентифицируется только относительно дисперсии ε. |
| S2 | McFadden D. (1974). *Conditional Logit Analysis of Qualitative Choice Behavior.* В: Frontiers in Econometrics, ed. Zarembka. URL: https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf | primary paper | metadata-only (binary PDF не распарсился; факты — через Train, PyMC docs, citation summaries) | Оригинальное доказательство, что iid Type I extreme value на ε даёт логит-форму. Footnote-цитата для §3.1 диссертации. |
| S3 | Krause T. et al. (2024). *Mitigating Exposure Bias in Recommender Systems — A Comparative Analysis of Discrete Choice Models.* ACM TORS. DOI: 10.1145/3641291. GitHub: https://github.com/krauthorDFKI/DiscreteChoiceForBiasMitigation | primary paper | derived-only (полный PDF 403; метаданные + search summary + repo) | MNL/GEV/nested/mixed сравниваются как механизмы для recommender simulation. MNL «underestimated overexposed items' popularity slightly» — IIA-ограничение. MNL — рабочая baseline. |
| S4 | Krause T. et al. (2025). *LCM4Rec: A Non-Parametric Choice Model.* RecSys 2025. arXiv:2507.20035. URL: https://arxiv.org/html/2507.20035v1 | primary paper | full HTML (abstract + first sections) | Additive utility `U_ij = V_ij + ε_ij` сохраняется, ε параметризуется не Gumbel. Users choose из top-K consideration set + «no-choice alternative» как escape. MNL допустим как baseline, но с осознанием ограничений. |

### B. Recommender influence / user choice — отдельный канал utility

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S5 | Tudoran L., Ricci F. (2023). *Choice models and recommender systems effects on users' choices.* UMUAI 33(5). URL: https://link.springer.com/article/10.1007/s11257-023-09366-x | primary paper | derived-only (303 redirect; метаданные + search summary) | Choice model и recommender — два независимо варьируемых канала. Distribution выборов зависит от обоих. Прямая формальная база для разделения `rel` и `rec` в utility. |
| S6 | Mollabagher M., Naghizadeh P. (2025). *The Feedback Loop Between Recommendation Systems and Reactive Users.* arXiv:2504.07105. URL: https://arxiv.org/abs/2504.07105 | primary paper | abstract | Reactive users моделируются как continuous responsiveness к рекомендации, а не Bernoulli. Поддерживает `w_rec` как continuous knob. |
| S7 | Adomavicius G., Bockstedt J., Curley S., Zhang J. (2013). *Do Recommender Systems Manipulate Consumer Preferences?* Information Systems Research. DOI: 10.1287/isre.2013.0497 | primary paper | not-accessible (403 на PubsOnline; через search summary) | «Recommendation acts as anchor for constructed preference» — поддерживает additive shift в utility, не hard filter. |
| S8 | Yao S. et al. (2021). *Measuring Recommender System Effects with Simulated Users.* arXiv:2101.04526. URL: https://arxiv.org/pdf/2101.04526 | primary paper | abstract-only (binary PDF) | Общий нарратив: recommender effect нужно formally изолировать от intrinsic preference в симуляции. |

### C. Compliance / adherence

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S9 | Steck H. (2018). *Calibrated Recommendations.* RecSys 2018. DOI: 10.1145/3240323.3240372 | primary paper | derived-only (PDF binary не распарсился; формулы — через туториал http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html и survey-обзоры) | Steck — это **post-processing re-ranker** поверх существующего recommender, формула `argmax (1-λ)·s(I) - λ·KL(p||q̃)`. Это **не модель compliance**; имя `calibrated_compliance` в `simulator.py` — **ложный друг** относительно `CalibratedPolicy`. |
| S10 | Bougie N., Watanabe N. (2025). *SimUSER.* ACL 2025. arXiv:2504.12722. URL: https://arxiv.org/html/2504.12722v1 | primary paper | partial HTML (methodology section) | В LLM-симуляторах compliance не tunable parameter, эмерджентен из persona + Chain-of-Thought. В **параметрическом** симуляторе compliance обязан быть явным parameter. |

### D. Capacity-aware / congestion-aware

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S11 | Mashayekhi Y. et al. (2023). *ReCon: Reducing Congestion in Job Recommendation using Optimal Transport.* RecSys 2023. arXiv:2308.09516. URL: https://arxiv.org/abs/2308.09516 | primary paper | abstract + local legacy summary (`materials/_legacy/research-recon-deep-2026-05.md`) | Congestion penalty живёт **в обучении recommender'а** через optimal transport: `O_ReCon = O_M + λ·O_C`, не в utility пользователя. User моделируется только через scoring `p_ui`. |
| S12 | Li N. et al. (2024). *FEIR: Quantifying and Reducing Envy and Inferiority for Fair Recommendation of Limited Resources.* ACM TIST. arXiv:2311.04542. URL: https://arxiv.org/abs/2311.04542 | primary paper | abstract + local summary | FEIR — post-processing для recommendation. Capacity-aware дисциплина живёт в *recommender ranking*, не в user choice. |
| S13 | Capacity-aware fair POI recommendation (ESWA 2023). DOI: 10.1016/j.eswa.2023.120488. URL: https://www.sciencedirect.com/science/article/pii/S156849462300738X | primary paper | not-accessible (403 на статью); search snippets | Capacity-aware allocation как over-demand cut / under-demand add policy. Capacity живёт в **post-processing allocation шаге**, не в user utility. |

### E. Валидация (рамочная база)

| № | Источник | Тип | Доступ | Что вытащено |
|---|---|---|---|---|
| S14 | Sargent R. (2013). *Verification and Validation of Simulation Models.* Journal of Simulation 7(1). | survey | metadata-only (PDF недоступен на public mirror) | Extreme condition tests — каноническая техника валидации. Operational validity = «accuracy required for the model's intended purpose over the domain of intended applicability». |
| S15 | Larooij M., Törnberg P. (2025). *Validation is the central challenge for generative social simulation.* AI Review (Springer). arXiv:2504.03274. URL: https://arxiv.org/abs/2504.03274 | review | abstract | «Believability ≠ operational validity». Cross-validation между параметрическим и LLM-симулятором — методически рекомендуемый ответ. EC tests — стандарт. |

### Итог по 5

Реально открыты 6 локальных файлов кода + 6 локальных документов проекта + 15 внешних источников. Из 15 внешних — 1 textbook (partial), 4 primary papers (full / abstract / partial HTML), 8 derived-only / abstract-only / metadata-only / not-accessible.

Источники с пометкой `not-accessible` (S7, S13) и `metadata-only` (S2, S14) используются только индикативно: их выводы — поддерживающие, не определяющие. Решение в разделах 8–9 опирается прежде всего на S1, S4, S5, S6, S11, S12 (источники с реально полученным содержанием).

---

## 6. Обзор вариантов реализации

### V1. Top-K-фильтр (status quo, текущая реализация)

`consider_ids = recs`; политика жёстко ограничивает choice set top-K рекомендациями. Параметра `w_rec` нет.

- **Плюсы:** уже работает в коде.
- **Минусы:** нарушает PROJECT_DESIGN §9 (политика влияет не через w_rec); EC3 невыполнима (нет параметра, который можно отправить в 0); EC4 формально выполнена, но неуправляемо.
- **Связь с литературой:** жёсткий choice set без continuous control противоречит S5 (две независимые dials), S6 (continuous responsiveness), S7 (anchor, не filter).

### V2. Bernoulli compliance (legacy `cfg.user_compliance`)

С вероятностью `α` consider_ids = recs, иначе consider_ids = slot.talk_ids. Уже частично реализовано (`simulator.py` строки 354–357).

- **Плюсы:** простой, интерпретируемый («доля участников, слушающих рекомендацию»).
- **Минусы:** не даёт continuous gradient по `w_rec`. EC3 не выполнима гладко (резкий переход α=0 → α>0). Compliance действует на consideration set, не на utility — несовместимо с MNL-формой PROJECT_DESIGN.
- **Связь с литературой:** S6 явно против Bernoulli compliance в пользу continuous responsiveness.

### V3. Calibrated 3-type compliance (legacy `cfg.use_calibrated_compliance`)

Трёхтипная модель B/C/A на Meetup-долях 71.7/21.3/7.0. Латентный класс выбирается перед choice; для compliant — consider_ids = recs, для star_chaser — argmax по fame, для curious — softmax по relevance в slot.

- **Плюсы:** distribution-level калибровка на Meetup; научно богаче (latent-class choice model).
- **Минусы:** доли зашиты на Meetup и не варьируются по гиперкубу; star_chaser игнорирует политику безусловно; ломает EC3 (compliant класс всегда следует политике независимо от `w_rec`); конфликтует с PROJECT_STATUS §5 («Big Five / social graph как реализованный метод — был прототип, не основной результат») по духу — это не основной механизм.
- **Связь с литературой:** формально это latent-class choice model (Train Ch.6), которую мы ранее не открывали; для целей предзащиты overkill.

### V4. Additive bonus через `w_rec`: `U = w_rel·rel + w_rec·rec(t, hat_pi)`

Рекомендация — отдельный additive компонент utility, score `rec(t, hat_pi) ∈ [0, 1]` зависит от выдачи политики. `consider_ids = slot.talk_ids` всегда.

- **Плюсы:** соответствует PROJECT_DESIGN §7 (3 компонента, каждый с своим весом). EC3 выполнима гладко: при `w_rec → 0` rec-term исчезает, все политики дают одинаковую utility. EC4 выполнима: при `w_rec → 1` rec-term доминирует. Совместима с MNL-формулировкой §2.1.2.
- **Минусы:** требует определить форму `rec(t, hat_pi)` (индикатор / ранг / probabilistic). Меняет формулу utility в `simulator.py`.
- **Связь с литературой:** прямо поддержано S5, S6, S7, S8. Каноническая additive utility из S1 (Train Ch.3), MNL-baseline из S3, S4.

### V5. Multiplicative bonus через `w_rec`: `U = rel · (1 + w_rec · rec)`

Рекомендация модулирует relevance, не складывается с ней.

- **Плюсы:** интуитивно «рекомендация усиливает интерес».
- **Минусы:** Multiplicative form не даёт чистого `w_rec → 0` (при `w_rec=0` всё равно остаётся `U = rel`, что хорошо, но при `w_rec → ∞` rel компонент не исчезает — он умножается). EC4 не выполнима в чистом виде. Не соответствует PROJECT_DESIGN §7 (там аддитивная декомпозиция). Не каноническая форма для MNL.
- **Связь с литературой:** в discrete choice mainstream (S1) аддитивная форма каноническая; multiplicative — экзотика.

### V6. Logit-choice без gossip (= V4 в softmax-форме)

То же, что V4, явно завёрнутое в softmax с температурой τ. По сути это просто более явная формулировка V4.

- **Плюсы:** явно совпадает с PROJECT_DESIGN §2.1.2. Не отдельный вариант, а способ записи V4.
- **Минусы:** —
- **Связь с литературой:** канонический MNL (S1).

### V7. Logit-choice с gossip: `U = w_rel·rel + w_rec·rec + w_gossip·gossip(t, L_t)`

Полная форма §2.1.2, включая социальное заражение.

- **Плюсы:** полностью соответствует PROJECT_DESIGN §7 (3 компонента).
- **Минусы:** gossip — отдельный плановый инкремент (этапы J–L PIVOT_IMPLEMENTATION_PLAN), требует своего spike. Не делать в одной правке.
- **Связь с литературой:** information cascades (Bikhchandani 1992), social influence in choice models — отдельный объём литературы; должен быть рассмотрен в spike J.

### V8. Смесь compliance + `w_rec`

`w_rec` как continuous, поверх — Bernoulli или calibrated compliance как мультипликативный модификатор «слушает / не слушает».

- **Плюсы:** комбинирует preference signal и behavior.
- **Минусы:** две оси параметров, ортогональные по смыслу но не по эффекту; гиперкуб распухает; интерпретация результата усложняется без видимой пользы.
- **Связь с литературой:** S6 утверждает, что continuous responsiveness покрывает функцию compliance; добавлять Bernoulli поверх — избыточно.

---

## 7. Сравнительная таблица вариантов

| Вариант | EC3 (`w_rec→0`) | EC4 (`w_rec→1`) | §9 (политика только через w_rec) | §7 (3 компонента) | Совместим с MNL §2.1.2 | Сложность правки | Поддержка литературы |
|---|---|---|---|---|---|---|---|
| V1 top-K filter (status quo) | ✗ | частично | ✗ | ✗ | ✗ | 0 | против (S5–S8) |
| V2 Bernoulli compliance | разрывно | да | ✗ | ✗ | частично | малая | против (S6) |
| V3 calibrated 3-type | ✗ | да | ✗ | ✗ | частично | средняя | overkill для срока |
| V4 additive `rel + rec` | ✓ гладко | ✓ | ✓ | частично (без gossip) | ✓ | малая | прямая (S1, S4, S5, S6) |
| V5 multiplicative | частично | ✗ | частично | ✗ | ✗ | малая | нет |
| V6 logit без gossip | = V4 | = V4 | ✓ | частично | ✓ | = V4 | = V4 |
| V7 logit с gossip | ✓ | ✓ | ✓ | ✓ | ✓ | большая, отдельный spike | прямая, но другой spike |
| V8 mix compliance + w_rec | размывается | да | частично | ✗ | частично | большая | против (S6) |

Критерии выбраны из PROJECT_DESIGN §7, §9, §11 и из анализа источников S1–S15.

---

## 8. Evidence-based recommendation

Рекомендованный вариант: **V4 (additive `U = w_rel·rel + w_rec·rec(t, hat_pi)`)**, эквивалентно **V6** (logit-choice без gossip).

**Обоснование (evidence-first, с явными ссылками на источники):**

1. **EC3/EC4 PROJECT_DESIGN §11.** Только V4/V6 удовлетворяют их гладко. V1 их структурно не реализует (нет параметра `w_rec`), V2 даёт разрывный переход, V5 ломает EC4, V8 размывает обе.
2. **Канонический MNL (S1, Train Ch.3).** Аддитивная систематическая часть `V_i = Σ β_k · x_k` — стандарт discrete choice. V4 — прямая инстанциация.
3. **Recommender effect как отдельный канал (S5 Tudoran-Ricci 2023, S6 Mollabagher 2025).** Recommendation должна моделироваться как отдельный варьируемый компонент utility, не как фильтр choice set. V4 — единственный вариант, где это так.
4. **Anchor-on-utility (S7 Adomavicius 2013).** Recommendation действует как additive shift, поддерживает V4.
5. **Не lock-in по compliance (S6).** Continuous `w_rec` покрывает функцию compliance; Bernoulli (V2) и латентный класс (V3) — лишний механизм без выгоды.
6. **Соответствие PROJECT_DESIGN §7.** V4 — двухкомпонентный частный случай (`w_gossip = 0`) трёхкомпонентной формы §7. Это согласовано с PIVOT_IMPLEMENTATION_PLAN принцип 4: «Первая минимальная модель — `rel + rec`. Это не означает отказ от gossip».
7. **Совместимость с этапами J–L.** V4 → V7 — аддитивное расширение `+ w_gossip·gossip(t, L_t)`, не пересборка модели. Этап K (gossip) реализуется как минимальная правка той же утилитной формы.

**Capacity-effect — отдельное архитектурное решение.**

Параллельно с выбором V4 необходимо принять, **где живёт capacity-effect**. Текущая формула содержит `-λ·max(0, load_frac - 0.85)` в utility пользователя. Research brief (раздел 6 brief) даёт три прямых довода за перенос capacity в политику П3, не в utility:

- **S11 (ReCon 2023).** Congestion penalty живёт **в обучении recommender'а** через optimal transport: `O_ReCon = O_M + λ·O_C`, не в utility пользователя. User моделируется только через scoring `p_ui`.
- **S12 (FEIR 2024).** Capacity-aware дисциплина живёт в *recommender ranking*, не в user choice.
- **S13 (capacity-aware POI ESWA 2023).** Capacity в **post-processing allocation шаге**, не в utility.

К этому добавляются формальные доводы:

- **PROJECT_DESIGN §9** запрещает политике влиять помимо `w_rec`. Capacity-term в utility означает, что *все* политики (включая П1 без рекомендаций) видят capacity-сигнал — это не «через `w_rec`».
- **EC3 ломается capacity-в-utility.** При `w_rec → 0` пользователь продолжает видеть capacity, и поведение зависит от `load_frac`, формирующегося под действием **всей** аудитории. Различие политик не очищается — capacity-эффект работает «снизу», даже без рекомендации.

**Decision (capacity):** capacity-эффект убирается из utility пользователя. Capacity-aware канал реализуется **только в политике П3** (CapacityAwarePolicy уже это делает через `score = sim - α·load_frac`).

**Compliance.**

`cfg.user_compliance` и `cfg.use_calibrated_compliance` остаются в коде как **legacy-параметры вне основного эксперимента**. Это согласовано с:

- PROJECT_STATUS §5: «Big Five / social graph как реализованный метод — был прототип, не основной результат»;
- PIVOT_IMPLEMENTATION_PLAN Q2: «Текущие user_compliance / calibrated_compliance оставляем как legacy-параметры вне основного эксперимента (флаг по умолчанию выключен, не входит в реестр конфигураций)»;
- PROJECT_STATUS §8: «Meetup используется как внешний якорь только для compliance-калибровки и distribution-level sanity».

Calibrated_compliance можно отдельно (вне 50-точечного гиперкуба) запускать на Meetup для distribution-match как distribution-level якорь.

**Outside option / no-choice альтернатива.**

`p_skip_base = 0.10` остаётся. Это согласовано с S4 (LCM4Rec): users могут выбрать «no-choice alternative» как escape. PROJECT_DESIGN явно outside option не упоминает, но и не запрещает. Минимальная правка — оставить как есть.

---

## 9. Минимальная первая реализация для этапов D–E

### Целевая формула

```
U(t | i, slot, hat_pi) = w_rel · rel(t, profile_i)
                       + w_rec · rec(t, hat_pi)
P(choose t | slot) = (1 - p_skip) · softmax(U / τ) ⊕ p_skip   (skip = no-choice)
consider_ids = slot.talk_ids                       (всегда; не recs)
capacity-effect:    только в политике П3, не в utility
```

где:
- `rel(t, profile_i) ∈ [-1, 1]` — нормализованный cosine, как сейчас.
- `rec(t, hat_pi) ∈ {0, 1}` — индикатор: `1`, если `t ∈ recs`, `0` иначе. Простейшая форма для первой реализации.
- `w_rel + w_rec` нормализуются вместе (см. open question Q1) либо варьируются независимо (выбор пользователя).
- `τ` — `cfg.tau`, как сейчас.
- `p_skip` — `cfg.p_skip_base`, как сейчас.

### Что меняется в `simulator.py` (этап E)

1. **`SimConfig`:** убрать `lambda_overflow` (capacity больше не в utility); добавить `w_rel: float = 0.7`, `w_rec: float = 0.3`. `w_fame` остаётся как есть (это часть rel-source, не отдельный компонент). `tau`, `p_skip_base`, `K`, `seed`, `user_compliance`, `use_calibrated_compliance` (последние — legacy-флаги, default off) остаются.
2. **`_process_one_slot`:** убрать ветку `cfg.use_calibrated_compliance` и Bernoulli compliance из основного path (оставить опционально под флагами для legacy distribution-match); зафиксировать `consider_ids = list(slot.talk_ids)`; новая utility-формула с `rel` и `rec` без capacity.
3. **`NoPolicy`:** уже возвращает пустой recs — для П1 `rec(t, hat_pi) = 0` для всех t. Никаких специальных branches не требуется.

### Что не меняется

- `Conference`, `Talk`, `Hall`, `Slot`, `UserProfile` — без изменений.
- `Embedder`, `LearnedPreferenceFn` — без изменений.
- `simulate_async`, async-инфраструктура — без изменений.
- Метрики (`metrics.py`) — без изменений.
- `CapacityAwarePolicy` — без изменений (capacity уже там).
- Все остальные политики (cosine, llm_ranker) — без изменений.

### Toy-кейсы для этапа D (предварительный smoke до правки simulator.py)

Все на 1 слоте × 2 залах × 2 равно-релевантных докладах × 100 пользователей, 3 seed:

- **TC-D1 (`w_rec = 0`).** Все 4 политики дают близкие распределения посещений (CV между политиками < 5%). Это операционализация EC3 на toy.
- **TC-D2 (`w_rec = 1.0`).** При П1 (no_policy) — баланс ~ 50/50 (rel-only выбор). При П2 (cosine) — концентрация на single talk per user (top-1 = top-K при K=1). При П3 — баланс уходит к менее загруженному залу. Различие политик выражено.
- **TC-D3 (capacity 20 vs 80, `w_rec = 0.5`).** При П2 — переполнение маленького зала. При П3 — переполнение исчезает. Базовая проверка, что capacity-канал в политике П3 работает.
- **TC-D4 (`w_rec` от 0 до 1 шагом 0.2, П1 vs П3).** Различие метрик растёт монотонно по `w_rec`. Это операционализация MC3.

### Этап E (правка ядра)

После прохождения toy-кейсов D — минимальная правка `simulator.py:_process_one_slot`. Никаких других модулей не трогаем.

---

## 10. Что сознательно откладываем

1. **Gossip-компонент `w_gossip · gossip(t, L_t)`.** Расширение V4 → V7. Отдельный spike (этап J), отдельная реализация (этап K), отдельная проверка (этап L). До прохождения L не запускаем полный LHS.
2. **Калибровка `w_rec`** на distribution-match с Meetup. Отдельная задача, не блокатор для EC3/EC4 и не для предзащиты. Может войти в анализ чувствительности (этап S).
3. **Nested logit / mixed logit / latent-class.** Научно богаче, но overkill для timeline предзащиты. В Limitations диссертации (§4 главы 4): «MNL предполагает IIA; для слотов с тематически близкими докладами nested logit был бы корректнее, но требует калибровки nest structure на real data».
4. **Generative agents с памятью.** Это путь LLM-симулятора, который у нас уже есть параллельным треком.
5. **Bernoulli и calibrated compliance** в основной матрице эксперимента. Остаются в коде как legacy под флагами; запускаются отдельно для distribution-match на Meetup.
6. **Изменения формы `rec(t, hat_pi)`** на ранг или probabilistic. На первой итерации — индикатор. Если будет видно, что индикаторная форма даёт слишком резкий gradient на гиперкубе по `w_rec` — переходим к ранговой форме `(K - rank(t)) / K` без переписывания utility.

---

## 11. Какие проверки должны пройти до перехода дальше

Этап C → этап D:
- Memo подписан пользователем по разделам 11–12 (явно подтверждены open questions Q1–Q5).

Этап D → этап E:
- Все 4 toy-кейса TC-D1…TC-D4 выполнены качественно (см. раздел 9). Если хотя бы один не проходит — возврат в C.

Этап E → этап F:
- Юнит-тест: `utility(t, hat_pi_A) == utility(t, hat_pi_B)` при `w_rec = 0` для любых hat_pi_A, hat_pi_B (формальная EC3 на уровне функции).
- Toy-кейсы TC-D1…TC-D4 воспроизводятся через ядро `simulator.py` (а не через локальную функцию в скрипте D) с теми же качественными выводами.

Этап F → этап I (extreme conditions):
- На Mobius (24 talks, K=2, 3 seed) воспроизводятся качественные выводы D. Время прогона ≤ 5 минут.

Этап I (extreme conditions) — обязательный фильтр перед содержательными выводами:
- EC1: `capacity_multiplier ≥ 3.0` ⇒ `mean_overload_excess == 0` для всех 4 политик.
- EC2: монотонность по `capacity_multiplier ∈ {0.5, 0.7, 1.0, 1.5, 3.0}`, усреднение по 5 seed.
- EC3: `w_rec → 0` ⇒ `CV(metric, policies) < 5%`.
- EC4: `w_rec → 1` ⇒ `range(metric, policies) > 10× CV(EC3)`.

Если хотя бы один EC проваливается, содержательные выводы блокируются (PROJECT_DESIGN §11).

---

## 12. Какие решения требуют подтверждения пользователя

### Q1. Форма `rec(t, hat_pi)` для первой реализации

Варианты:
- **(a) Индикатор:** `rec(t, hat_pi) = 1` если `t ∈ recs`, иначе `0`. Простейшая форма.
- **(b) Ранговая:** `rec(t, hat_pi) = (K - rank(t)) / K` для t в top-K, `0` иначе. Гладкая, но даёт другие эффекты в softmax.
- **(c) Probabilistic:** `rec(t, hat_pi) = score(t, hat_pi) / max_t score`, нормированный score политики.

**Предложение:** (a) индикатор для первой реализации. Решение должно быть подтверждено.

### Q2. Веса `w_rel`, `w_rec` — связаны или независимы?

Варианты:
- **(a) Связаны:** `w_rel + w_rec = 1`, варьируется один параметр `w_rec ∈ [0, 1]`. Интерпретируется как convex mix.
- **(b) Независимы:** оба варьируются по гиперкубу как отдельные оси.

**Предложение:** (a) связаны. Это упрощает гиперкуб (одна ось вместо двух) и согласуется с интуицией «доля влияния рекомендации». PROJECT_DESIGN §8 ось 3 не запрещает связку. Решение должно быть подтверждено.

### Q3. Capacity-эффект из utility удаляется полностью?

PROJECT_DESIGN §9 + EC3 + literature (S11, S12, S13) поддерживают **да**. Текущая реализация `u = effective_rel - λ·max(0, load_frac - 0.85)` удаляется. Capacity-канал остаётся только в `CapacityAwarePolicy`.

**Предложение:** да, удаляется. Решение должно быть подтверждено.

### Q4. Что делать с `cfg.user_compliance` и `cfg.use_calibrated_compliance`?

Варианты:
- **(a) Оставить как legacy под флагами,** default off, не входит в основной реестр конфигураций. Может быть использован отдельно для distribution-match на Meetup.
- **(b) Удалить полностью** из `simulator.py`.

**Предложение:** (a) оставить как legacy. Это согласуется с PIVOT_IMPLEMENTATION_PLAN Q2 и с PROJECT_STATUS §8 («Meetup как distribution-level якорь»). Решение должно быть подтверждено.

### Q5. Outside option `p_skip_base`?

Варианты:
- **(a) Оставить** `p_skip_base = 0.10` как сейчас. Поддержано S4 (LCM4Rec).
- **(b) Убрать** (упрощение).
- **(c) Сделать осью гиперкуба.**

**Предложение:** (a) оставить как есть. Не блокатор. Решение должно быть подтверждено.

---

## Финальное резюме (отдельно — то, что просил пользователь)

### 1. Рекомендованная первая реализация для этапов D–E

```
U(t | i, slot, hat_pi) = w_rel · rel(t, profile_i) + w_rec · rec(t, hat_pi)
P(choose t | slot)     = (1 - p_skip) · softmax(U / τ) ⊕ p_skip
consider_ids           = slot.talk_ids   (всегда)
rec(t, hat_pi)         = 1{t ∈ recs}     (индикатор)
capacity-effect        = только в политике П3
```

### 2. Что делать с `user_compliance` / `calibrated_compliance`

Оставить как legacy-параметры под флагами в `SimConfig`, default `False / 1.0`. Не входят в основной реестр конфигураций гиперкуба. Calibrated_compliance можно отдельно прогонять для distribution-match на Meetup как distribution-level sanity якорь. PROJECT_STATUS §8 это подтверждает.

### 3. Что делать с capacity-effect в utility

Удалить полностью. Строка `u = effective_rel - cfg.lambda_overflow * max(0.0, load_frac - 0.85)` заменяется на `u = cfg.w_rel * rel + cfg.w_rec * rec_indicator`. Параметр `cfg.lambda_overflow` помечается как deprecated или удаляется. Capacity-канал остаётся в `CapacityAwarePolicy` (где он уже есть как `score = sim - α·load_frac`).

### 4. Какую форму `rec(t, hat_pi)` выбрать

Для первой реализации — **индикатор `1{t ∈ recs}`** (Q1 option a). Если в этапе F окажется, что индикаторная форма даёт слишком резкий gradient на оси `w_rec`, переход на ранговую `(K - rank(t)) / K` — без переписывания utility.

### 5. Какие toy-кейсы и acceptance checks должны пройти

Toy-кейсы (этап D, до правки simulator.py):
- TC-D1: при `w_rec = 0` все 4 политики дают близкие распределения (CV < 5%).
- TC-D2: при `w_rec = 1` различия политик выражены.
- TC-D3: capacity-перекос (20 vs 80) — П3 устраняет переполнение.
- TC-D4: монотонность различий политик по `w_rec ∈ [0, 1]`.

Acceptance checks (этап E):
- Юнит-тест: utility инвариантна к политике при `w_rec = 0`.
- Toy-кейсы воспроизводятся через ядро.

Extreme conditions (этап I, обязательный фильтр):
- EC1, EC2, EC3, EC4 из PROJECT_DESIGN §11.

### 6. Какие решения требуют моего подтверждения

См. раздел 12. Пять открытых решений Q1–Q5:
- Q1. Форма `rec` (предложение: индикатор).
- Q2. `w_rel`, `w_rec` связаны или независимы (предложение: связаны, `w_rel + w_rec = 1`).
- Q3. Capacity-эффект удаляется из utility полностью (предложение: да).
- Q4. `user_compliance` / `calibrated_compliance` остаются как legacy под флагами (предложение: да).
- Q5. `p_skip_base` остаётся как outside option (предложение: да).

После подтверждения этих решений — переход к этапу D.
