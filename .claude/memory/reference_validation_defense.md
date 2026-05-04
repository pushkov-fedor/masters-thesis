---
name: Литература и канон валидации для защиты
description: Must-cite список и канон валидации LLM-симуляторов / DSS под текущий фрейм (сценарный аналитический полигон)
type: reference
originSessionId: 2ab16009-c3da-4bf3-88ce-b864d0acdf4a
---
Главный документ проекта — `/Users/fedor/Study/masters-degree/PROJECT_STATUS.md` (§9 «Роль литературы»). Здесь — расширенные цитаты и единственный канонический пункт валидации LLM-слоя на Meetup.

## Must-cite по группам

### DSS / Robust Decision Making / DMDU (методический фундамент)

- **Lempert R.J., Popper S.W., Bankes S.C.** *Shaping the Next One Hundred Years*. RAND, 2003 — XLRM-фрейм (X uncertainties, L levers, R relations, M measures), Robust Decision Making.
- **Marchau V., Walker W.E., Bloemen P.J.T.M., Popper S.W.** *Decision Making under Deep Uncertainty: From Theory to Practice*. Springer, **open access**, 2019.
- **Kwakkel J.H.** The Exploratory Modeling Workbench. *Environ. Modell. Softw.*, 2017.

### Simulation V&V

- **Sargent R.G.** Verification and Validation of Simulation Models. *J. Simulation*, 2013 — канонические 12 методов V&V (face validity, extreme condition, parameter variability, sensitivity, degenerate behaviour и др.).
- **Robinson S.** *Simulation: The Practice of Model Development and Use*, 2nd ed., 2014 — структура simulation-проекта.
- **Kleijnen J.P.C.** Design and Analysis of Simulation Experiments for Sensitivity Analysis. *EJOR*, 2005 — DOE для sensitivity analysis.
- **JASSS 27/1/11**, 2024 — методы валидации ABM без real data (face / operational / data validity).

### Capacity-aware / congestion-aware recsys

- **Mashayekhi et al.** ReCon: Reducing Congestion in Job Recommendation. *RecSys*, 2023 — congestion как первичная метрика.
- **Wang & Joachims** FEIR: Quantifying and Reducing Envy and Inferiority for Fair Recommendation of Limited Resources. *ACM TIST*, 2024.
- Capacity-aware fair POI re-ranking. *Expert Systems with Applications*, 2024.

### Conference scheduling (родственный жанр)

- **Vangerven et al.** Scheduling conferences using attendees' preferences. *JORS*, 2024.
- **Pylyavskyy, Kheiri, Jacko** A generic approach to conference scheduling with integer programming. *EJOR*, 2024.
- **Kheiri, Pylyavskyy, Jacko** *CoSPLib — A benchmark library for conference scheduling problems*. GECCO Companion, 2025. (только cite-only — в инстансах нет attendance / preference / capacity / talk text).
- **Bulhões, Correia, Subramanian.** Clustering-based conference scheduling. *EJOR*, 2022.
- **Manda et al.** PeerJ CS, 2019 — open-source ближайший аналог.
- **Stidsen, Pisinger, Vigo.** Scheduling EURO-k conferences. *EJOR*, 2018.

### LLM-agent simulation

- **Park J.S. et al.** Generative Agents. *UIST*, 2023.
- **Zhang et al.** Agent4Rec. *SIGIR*, 2024.
- **Yang et al.** OASIS. *NeurIPS*, 2024.
- **Piao et al.** AgentSociety. arXiv:2502.08691, 2025.
- **Wang et al.** RecAgent. *ACM TOIS*, 2024.
- SimUSER, arXiv:2504.12722, 2025.

### Критика LLM-ABM (защитная подкладка)

- **Larooij M., Törnberg P.** *Validation is the central challenge for generative social simulation*. Artif. Intell. Rev. (Springer), 2025 — «believability ≠ validity».
- arXiv:2504.07105, 2025 — reactive users feedback loop.
- **Chaney, Stewart, Engelhardt.** Algorithmic confounding. *RecSys*, 2018.

### Negative-references (для оправдания узкой постановки)

- **Quaeghebeur** Evolution 2014 (N=29) — единственный публичный per-attendee per-talk attendance dataset.
- **CoSPLib (Kheiri 2025)** — нет attendance, capacity, preferences, talk text.
- **OpenReview bidding** — публичных дампов нет.
- **Wharton CourseMatch (Budish 2017)** — только summary statistics, raw bids приватны.
- **SocioPatterns SFHH/Hypertext-2009** — только face-to-face proximity, не per-talk attendance.

### Русскоязычные

- **Бурый А.С., Цаплина О.С.** «Правовая информатика», 2025 — агентные DSS на LLM.
- **Абрамов В.И.** «Актуальные проблемы экономики и права», т.12 №1, 2018 — ABM + сценарный анализ + валидация.
- **Буянов Б.Б., Лубков Н.В., Поляк Г.Л.** «Проблемы управления», 2006 — СППР + имитационное моделирование.
- **Лычкина Н.Н.** *Имитационное моделирование экономических процессов*, ВШЭ — учебник.

## Канон валидации LLM-симулятора

Никто из канона (Park / Agent4Rec / RecAgent / SimUSER / OASIS / AgentSociety) не валидирует индивидуальный choice реального юзера. Все валидируют:
- distribution match (Spearman / JS-divergence на распределениях популярности);
- stylized facts (Парето, herd, концентрация);
- behavioral elasticity (реакция на интервенции);
- sensitivity к параметрам.

`accuracy@1` на индивидуальном выборе **не является** метрикой валидации LLM-симулятора. Larooij & Törnberg 2025 формулирует: «believability ≠ validity», face validity не означает operational validity.

## Главный валидационный пункт LLM-слоя в защите

**Spearman ρ = 0.4379, p = 8.7 × 10⁻¹⁸** между LLM-предсказанной популярностью талков и реальной Meetup-популярностью на 349 talks (3547 LLM-вызовов).

Артефакты:
- `experiments/results/sim_validation_meetup.json`
- `experiments/results/sim_validation_meetup_grid.json`
- скрипт: `experiments/scripts/validate_simulator_meetup.py`

В тексте формулируется как **face validity LLM-слоя на коллективном распределении**, не как «realism» симулятора. Индивидуальный choice (B2 ≈ random) — ожидаемо в Meetup-домене, согласуется с каноном.

## Защитный нарратив (под новый фрейм)

«Валидация многоуровневая в каноне Sargent + Kleijnen + DMDU + Larooij & Törnberg: (1) внутренняя верификация (toy-case, internal consistency, monotonicity, repeated seeds); (2) sensitivity по параметрам сценария (capacity, compliance) — DOE по Kleijnen; (3) data-side anchor — compliance калиброван на реальных Meetup-RSVPs (3-type 71.7/21.3/7.0); (4) face validity LLM-слоя — distribution match ρ=0.438 на Meetup; (5) external sanity — UMass course allocation (real preferences + real capacities в смежном домене limited-resource allocation, **не конференция**). Operational validity против реальной посещаемости конкретной конференции не выполнена ввиду отсутствия публичных данных и явно зафиксирована в Limitations.»
