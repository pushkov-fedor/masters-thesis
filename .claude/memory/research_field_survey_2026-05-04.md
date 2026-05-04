---
name: Field survey 2026-05-04
description: Литературный обзор предметной области для ВКР под текущий фрейм (сценарный полигон / DSS-стресс-тест программы конференции, два независимых симулятора отклика — параметрический MNL и LLM-агентский)
type: reference
date: 2026-05-04
---

# Обзор предметной области для ВКР

Источник истины по проекту — `/Users/fedor/Study/masters-degree/PROJECT_STATUS.md`. Здесь — расширенный картирующий обзор для главы 1 «Похожие работы», собранный 2026-05-04.

Все ссылки проверены поиском; там, где не получилось подтвердить какой-то один пункт (имена авторов, дата, наличие нужного признака), стоит явная пометка «не подтверждено» или «нужна проверка».

---

## Раздел 1. Карта подходов «помощь организатору при неопределённом спросе на параллельные сессии»

Я разделяю исследовательский ландшафт на три группы по способу обращения с неопределённостью спроса. Деление не строгое, многие работы пересекают границы.

### 1.А. Прогнозные подходы — «предсказать посещаемость и оптимизировать»

Базовая логика: построить модель $\hat{a}_{u,t}$ (вероятность того, что участник $u$ пойдёт на доклад $t$), агрегировать в ожидаемую загрузку зала $\hat{L}_r$, использовать как вход в оптимизатор расписания. Что считается успехом — accuracy attendance prediction, MAE/MAPE по загрузке. Где валидируют — Meetup-RSVP, исторические аналоги конференций, обычно internal-only данные.

Типичные представители:
- Zhang et al. «Who Will Attend? Predicting Event Attendance in Event-Based Social Network», MDM 2015 — semantic / temporal / spatial features на Meetup, predict attendance per user-event pair.
- Nguyen et al. «Predicting individual event attendance with machine learning: a step-forward approach», *Applied Economics* 54(27), 2022 — пошаговая ML-постановка.
- POI-recommendation линия (CTRNext в *ESWA* 2024, MMPAN 2024 и т.п.) — формально прогноз следующего POI, но по структуре ровно та же логика «индивидуальный выбор → агрегат».
- Industrial side: Eventbrite analytics, PredictHQ — prediction-as-a-service для организаторов.

Ограничения как класса:
- Точность индивидуального choice в conference / event domain заведомо низкая (см. наш B2 ≈ random на Meetup); канон LLM-симуляторов (Park, Agent4Rec, RecAgent, SimUSER) явно отказался от этой метрики в пользу distribution-match.
- В нашей задаче нет ground-truth attendance публично, поэтому весь трэк методически закрыт — мы не можем на нём ни обучаться, ни валидироваться.

### 1.Б. Сценарные / робастные подходы — «не предсказывать, а стресс-тестировать»

Базовая логика: вместо точечного прогноза перебрать вероятные миры $\Theta$ и оценить устойчивость политики $\pi$ по всему перебору. То, что делаем мы.

Методический фундамент — DMDU (Decision Making under Deep Uncertainty):
- Lempert, Popper, Bankes. *Shaping the Next One Hundred Years*. RAND, MR-1626-RPC, 2003 — оригинал Robust Decision Making (RDM) и XLRM-фрейма.
- Marchau, Walker, Bloemen, Popper (eds.). *Decision Making under Deep Uncertainty: From Theory to Practice*. Springer, 2019, **open access** через OAPEN — современный sourcebook DMDU; глава Lempert о RDM, главы про DAP, DAPP, Info-Gap, Engineering Options.
- Kwakkel, Pruyt. «Exploratory modeling and analysis: an approach for model-based foresight under deep uncertainty». *Technological Forecasting and Social Change* 80(3), 2013 — формализация EMA.
- Kwakkel. «The Exploratory Modeling Workbench». *Environmental Modelling & Software* 96, 2017 — EMA-workbench, открытый Python-toolkit для exploratory modeling, scenario discovery, multi-objective robust decision making.
- Popper. «Robust decision making and scenario discovery in the absence of formal models». *Futures & Foresight Science*, 2019.
- Webber, Samaras. «A Review of DMDU Applications Using Green Infrastructure for Flood Management». *Earth's Future*, 2022 — обзор 64 публикаций по DMDU; типичный жанр применения, на который мы можем ссылаться методологически.

Что считается успехом в этой группе:
- идентификация сценариев, где политика $\pi$ ломается (scenario discovery, PRIM patient rule induction);
- robustness-метрики (regret, worst-case, satisficing);
- interpretable mapping «при каких комбинациях (X-uncertainties) выбор рычага L-leverage не приводит к нарушению M-measures».

Где валидируют: face validity, internal consistency, sensitivity sweeps, stylized facts; не accuracy против реальной траектории. Это совпадает с нашим каноном (Sargent / Kleijnen / DMDU / JASSS 27/1/11).

### 1.В. Гибрид — «predict-then-optimize» / «forecast-then-optimize»

Базовая логика: тренировать прогнозную модель end-to-end под downstream-метрику оптимизатора (Bertsimas, Kallus и др. — «From Predictive to Prescriptive Analytics», *Management Science* 2020). Forecast-Then-Optimize и Predict-and-Optimize линия в renewable energy scheduling (arXiv:2212.10723), refinery planning *Comput. Chem. Eng.* 2024, hotel staff scheduling *Annals of OR* 2025. В conference-domain прямого применения нет — мешает отсутствие ground-truth.

Условно сюда же относятся подходы CourseMatch / Pseudo-Market with Priorities (course allocation, см. ниже §2): они оптимизируют под предпочтения, но не предсказывают будущую посещаемость.

### Где встаёт наша работа

Наша задача нарративно живёт **в группе 1.Б**, со ссылкой на 1.А как на «класс, который мы сознательно не выбираем из-за отсутствия публичных attendance-данных» и на 1.В как на параллельную линию для domain с available-data (нашей domain недоступна).

---

## Раздел 2. Параметрические / аналитические работы (релевантные нашему MNL-симулятору)

Здесь работы, дающие либо метод выбора участника между параллельными опциями, либо capacity / congestion-aware рекомендатор, либо OR-постановку conference scheduling. Где мог — извлёк (задача / метод / данные / валидация / ограничение / релевантность нам).

### 2.1 Discrete choice models в recommender / event / POI domain

- **Train, Ben-Akiva, McFadden, Rusmevichientong et al. — линия choice modeling.** Многократно цитируемый методический фундамент; для нас релевантно тем, что вся область multinomial-logit assortment optimization подсказывает форму utility и форму choice-noise. Свежее: Aouad, Feldman, Segev, Zhang. «The Click-Based MNL Model». *Management Science*, 2024 — augmentation classical MNL клик-данными перед сравнением utilities. Mitrofanov, Topaloglu, Wang. «Choice Modeling, Assortment Optimization, and Estimation When Customers are Non-Rational: MNL with Non-Parametric Dominance». 2024 (SSRN). Релевантно нам: подсказывает, что включение «non-rational dominance» (наш star_chaser) — ровно та форма расширения MNL, которая обсуждается в OR-сообществе.
- **Han, Pereira, Ben-Akiva, Zegras. «TasteNet-MNL: Modeling Taste Heterogeneity with Flexibility and Interpretability». 2020 (rev 2022, Transportation Research Part B).** Neural-embedded MNL с гетерогенностью предпочтений. Не подтверждено, что есть прямое применение к event domain, но формально описывает то же, что у нас: utility = embedding × interpretable taste vector.
- Arkoudi et al. «Combining discrete choice models and neural networks through embeddings: Formulation, interpretability and performance». *Transportation Research Part B*, 2023 — embedding-based hybrid choice models.
- Ben-Akiva, Aboutaleb, Danaf, Xie. Joint data-driven model selection / specification / estimation of discrete choice models subject to behavioral constraints (MIT ITS, продолжающаяся серия). Точная цитата требует уточнения.

Релевантность нам: формальная база, на которой стоит наш параметрический симулятор. Compliant ≈ standard MNL, star_chaser и curious — heterogeneity / non-rational extensions.

### 2.2 Capacity-aware и congestion-aware recommender systems

- **Mashayekhi, Kang, Lijffijt, De Bie. «ReCon: Reducing Congestion in Job Recommendation using Optimal Transport».** RecSys 2023, ACM (DOI 10.1145/3604915.3608817). Метод: in-processing optimal transport для равномерного распределения вакансий по соискателям, multi-objective objective (Congestion + NDCG). Данные: e-recruitment instance. Метрики: Congestion, Coverage, Gini Index, NDCG, Recall, Hit Rate. Ограничение: optimisation на in-processing уровне, требуется retraining; требует доступа к вместимости/числу вакансий. Релевантность нам: прямой родственник, congestion как первичная метрика (как у нас `mean_overload_excess`).
- **Mashayekhi, Kang, Lijffijt, De Bie. «Scalable Job Recommendation With Lower Congestion Using Optimal Transport».** *IEEE Access* 12, 2024 — масштабируемое расширение ReCon.
- **Mashayekhi, Li, Kang, Lijffijt, De Bie. «A Challenge-based Survey of E-recruitment Recommendation Systems».** *ACM Computing Surveys* 56(10), Article 252, October 2024 (DOI 10.1145/3659942) — обзор семьи задач, частью которой является ReCon. Релевантно для подкладки одного абзаца про congestion-as-fairness в job-domain.
- **Nan Li et al. «FEIR: Quantifying and Reducing Envy and Inferiority for Fair Recommendation of Limited Resources».** *ACM TIST*, 2024 (DOI 10.1145/3643891), arXiv:2311.04542. Группа Joachims / KU Leuven. Метод: post-processing над любым recsys; вводит две меры — envy и inferiority — плюс utility, через дифференцируемое probabilistic-relaxation. Сравнивается напрямую с ReCon. Данные: synthetic + real-world. Ограничение: только post-processing; ground-truth fairness известен только в синтетике. Релевантность нам: формальный язык envy / inferiority, который мы могли бы наложить на нашу `hall_load_gini`.
- **Patro, Biswas, Ganguly, Gummadi, Chakraborty. «FairRec: Two-Sided Fairness for Personalized Recommendations in Two-Sided Platforms».** WWW 2020. Метод: fair allocation indivisible goods, гарантии MMS exposure для производителей и EF1 fairness для потребителей. Релевантность: формальный фрейм «recsys как fair allocation», в нашу сторону частично применим — у нас залы вместо producers.
- **Patro, Biswas, Ganguly, Gummadi, Chakraborty. «Towards Fair Recommendation in Two-Sided Platforms».** *ACM TWEB* 16(2), 2022 — FairRecPlus (improved customer utility при тех же fairness-гарантиях).
- **Capacity-aware fair POI recommendation combining transformer neural networks and resource allocation policy.** *ESWA*, 2023 (DOI: 10.1016/j.eswa.2023.…, S156849462300738X). Транс­формерная сеть + attention-LSTM для personalized POI; над ней — capacity-based allocation с over-demand-cut и under-demand-add policies; гарантии POI exposure ratio и envy-freeness. Данные: 5 real-life POI datasets. Ограничение: нет capacity ground-truth, capacity берётся как параметр. Релевантность нам: прямой жанровый аналог нашего capacity-aware reranker, только для POI и без сценарной развёртки. Авторы по результатам поиска — нужна проверка точного состава имён.
- **Chen et al. «Interpolating Item and User Fairness in Multi-Sided Recommendations».** NeurIPS 2024 — multi-objective fair allocation.
- **Korean follow-ups Bei et al. (2020), Saito & Joachims (KDD 2022) — линия fair ranking as fair division.** Концептуально близка, формально — ranking, не allocation.

Релевантность всему нашему MNL-симулятору: эти работы формализуют ровно то, чего у нас не хватает в одну строку — congestion / capacity как фундаментальную fairness-метрику recsys. Полезно для §1 «постановка задачи» и §2 «метрики».

### 2.3 Calibrated, MMR, DPP — техники reranking и применимость к congestion

- **Steck. «Calibrated Recommendations».** RecSys 2018. Идея: распределение жанров в рекомендованном списке должно совпадать с распределением в истории пользователя. Метрика — KL-дивергенция между двумя распределениями. Сильно связано с тем, что мы делаем «calibrated» (одна из тестируемых политик), только у нас calibration по жанрам track-а, а не по всей истории.
- **Survey: Calibrated Recommendations: Survey and Future Directions.** *ACM TORS* / arXiv:2507.02643, 2025 — последний обзор линии.
- **Abdollahpouri et al. «Calibrated Recommendations as a Minimum-Cost Flow Problem».** WSDM 2023 — alternative formulation, Spotify-co-authored.
- **Spotify Research. «Calibrated Recommendations with Contextual Bandits».** RecSys 2025 / arXiv:2509.05460 — production deployment calibrated reranking на главной странице Spotify, контекстуальные bandit'ы для адаптивной калибровки. Доказывает, что calibrated подход ≠ academic curiosity.
- **MMR (Maximal Marginal Relevance).** Carbonell & Goldstein 1998 — оригинал. Свежее: SMMR — sampling-based MMR, SIGIR 2025 (DOI 10.1145/3726302.3730250) — random-sampling extension MMR с logarithmic speedup. Один из baseline'ов в рекомендательной диверсификации.
- **DPP (Determinantal Point Processes).** Chen, Zhang, Zhou. «Fast greedy MAP inference for determinantal point process». NeurIPS 2018 — greedy MAP-инференс. Свежее: «Diversified recommendations of cultural activities with personalized determinantal point processes» (arXiv:2509.10392).

Ограничение всех этих техник для нашей задачи: сами по себе они не знают про capacity. Их можно скрестить с capacity-mask (как мы делаем в `capacity_aware_mmr`), но в литературе систематически такая комбинация для конференций не описана.

### 2.4 Conference scheduling как OR-задача (родственный жанр)

- **Vangerven, Ficker, Goossens, Passchyn, Spieksma. «Conference scheduling — a personalized approach».** *EJOR*, 2017–2018 (онлайн ранее, версия в журнале S0305048317302013). Three-phase IP: (1) max attendance по preferences, (2) min session-hop / topical overlap, (3) presenter availabilities. Применено к Mathsport 2013, MAPSP 2015 / 2017, ORBEL 2017. Ограничение: используется опрос участников (preference data), который у нас не доступен. Релевантность: эталон работы с реальными preferences в conference-scheduling.
- **Rezaeinia, Góez, Guajardo. «Scheduling conferences using data on attendees' preferences».** *Journal of the Operational Research Society* 75(11), 2024 (DOI 10.1080/01605682.2024.2310722). Three IP-formulations, attendee-based perspective. Two-stage: (1) thematic sessions, (2) talks within sessions. Данные: anonymized preferences trex conferences. Релевантность: ровно тот класс, под который наш scenario-полигон даёт complementary stress-test.
- **Pylyavskyy, Jacko, Kheiri. «A generic approach to conference scheduling with integer programming».** *EJOR* 317(2), 2024 (DOI 10.1016/j.ejor.2024.04.001). Penalty-system для гибридных / онлайн-конференций, два IP-models (exact + extended). Применимо для конференций до нескольких тысяч submissions.
- **Kheiri, Pylyavskyy, Jacko. «Efficient Scheduling of GECCO Conferences using Hyper-heuristic Algorithms».** GECCO Companion 2024 (DOI 10.1145/3638530.3664186) — applied на реальных GECCO instances.
- **Kheiri, Pylyavskyy, Jacko. «Exact and Hyper-heuristic Methods for Solving the Conference Scheduling Problem».** DASA 2024 — обобщение.
- **Kheiri, Pylyavskyy, Jacko. «CoSPLib — A Benchmark Library for Conference Scheduling Problems».** GECCO Companion 2025, p. 131-134 (DOI 10.1145/3712255.3726570). GitHub: ahmedkheiri/CoSPLib. Только cite-only — в инстансах нет attendance / preference / capacity / talk-text данных, поэтому собственно симулятор отклика на эти инстансы не построишь.
- **Bulhões, Correia, Subramanian. «Conference scheduling: A clustering-based approach».** *EJOR* 297(1), 2022 — branch-and-cut / branch-cut-and-price для clustering similar talks; benchmark instances из real-world data.
- **Stidsen, Pisinger, Vigo. «Scheduling EURO-k conferences».** *EJOR* 270(3), 2018, p. 1138-1147 — EURO-2015/2016 scheduling tool, иерархическая постановка с >2000 presentations.
- **Manda, Hahn, Lamm, Vision. «Avoiding "conflicts of interest": a computational approach to scheduling parallel conference tracks and its human evaluation».** *PeerJ Computer Science* 5:e234, 2019. Topic-model + simulated-annealing на ecology conference; ближайший open-source аналог. Артефакты: Zenodo. Очень полезен как методический прецедент «скрытые предпочтения через topic-model по abstract'ам».
- **«A Track-Based Conference Scheduling Problem».** *Mathematics* 10(21), Article 3976, 2022 — track-based variant, simulated annealing + Gurobi baseline. Inspired by GECCO case-study.
- **Bart Vangerven et al.** «Winner Determination in Geometrical Combinatorial Auctions» (SSRN 2777489) — соседняя постановка (комбинаторные аукционы), методическая связь с CourseMatch.

Главное наблюдение по группе: ни одна из этих работ не делает scenario sweep по неопределённости спроса. Они либо считают preferences known (Vangerven, Rezaeinia), либо вообще не считают preferences (Pylyavskyy, CoSPLib, Bulhões, Stidsen, Manda). Это — ровно та ниша, в которой стоим мы.

### 2.5 Course allocation / classroom assignment — methodologically близкий жанр

- **Budish, Cachon, Kessler, Othman. «Course Match: A Large-Scale Implementation of Approximate Competitive Equilibrium from Equal Incomes for Combinatorial Allocation».** *Operations Research* 65(2), 2017, p. 314-336 — A-CEEI mechanism, deployed at Wharton (~1700 students, ~350 courses/sem). Ground-truth от student survey. Raw bids приватны.
- **Budish, Kessler. «Can Agents 'Report Their Types'? An Experiment that Changed the Course Allocation Mechanism at Wharton».** NBER WP 22448, 2017.
- **Kornbluth, Kushnir. «Undergraduate Course Allocation through Competitive Markets».** SSRN 3901146 / arXiv:2412.05691, 2024 (rev 2025). Pseudo-Market with Priorities (PMP) — competitive equilibrium с priorities; approximate stability / efficiency / envy-freeness / strategy-proofness. Empirical evaluation на university data; снижение envy на ~8% (~500 студентов) vs incumbent.
- **Rodríguez, Manlove. «Course Allocation with Credits via Stable Matching».** SAGT 2025 / arXiv:2505.21229 — stable-matching extension для credits.
- **Bobbio, Carvalho. «Capacity Planning in Stable Matching: An Application to School Choice».** ACM EC 2023.
- **Gokhale et al. «Capacity Modification in the Stable Matching Problem».** AAMAS 2024 / arXiv:2402.04645.
- **Sönmez, Ünver. «Course Bidding at Business Schools».** *International Economic Review*, 2010 — Pareto-dominant bidding mechanism, primer для market-design на course-domain.

Этот блок методологически наш ближайший родственник (limited resource + preferences + allocation under capacity), но в conference-domain он не транслируется напрямую: нет формальной валюты и нет процедуры опроса participant'ов.

### 2.6 Event-based social networks (EBSN) — наследники

- **Tong, Meng, She. «On bottleneck-aware arrangement for event-based social networks».** ICDEW 2015. **Tong, She, Meng. «Bottleneck-aware arrangement over event-based social networks: the max-min approach».** *World Wide Web* 19, 2016, p. 1151-1177. Формально определяют BSEA-задачу (NP-hard) — арранжировать events по users с учётом bottleneck. Старо для критерия «не позднее 2018», но цитировать как pre-2018 канон допустимо, потому что это «фундамент» для всей последующей EBSN-литературы.
- Вариации: «Group event recommendation in EBSN», «Multi-feature based event recommendation in EBSN», «Attentive Implicit Relation Embedding for Event Recommendation in EBSN» (*Smart Learning Environments / Engineering Applications of AI*, 2024). Не подтверждено, что в этих работах capacity первичен.

Релевантность: подсказывает, что наша задача формально соседствует с EBSN, но мы внутрь не входим (нет EBSN-структуры user-user friendships).

### 2.7 Calibration источник для наших compliance-долей (Meetup → 71.7 / 21.3 / 7.0)

Прямого канонического источника, из которого все берут именно такую compliance-калибровку, не существует. Мы используем эмпирическое распределение из Meetup-RSVP. Близкие методологически:
- **Liu, He, Tian, Lee, McPherson, Han. «Event-Based Social Networks: Linking the Online and Offline Social Worlds».** SIGKDD 2012 — оригинал Meetup-датасета, использованного позднее десятками работ.
- **Wu Yue (2022). Meetup Dataset.** IEEE DataPort, DOI 10.21227/8dr0-p842 — позднейший public dump.
- **Zhang et al. «Who Will Attend?», MDM 2015** (см. §1.А).
- Линия research-кода (sijopkd, sahanasub, graceh3, ozgekoroglu на GitHub) — иллюстрирует, что Meetup-RSVP — фактически de-facto-стандартный публичный benchmark для individual-event attendance.

Проблема, которую мы формулируем явно: actual attendance в Meetup не публикуется, есть только RSVP-yes-count + флаги attended/noshow на уровне организатора. Это — одно из ограничений domain в Limitations нашего §3.

---

## Раздел 3. LLM-агентские работы (релевантные нашему LLM-слою)

### 3.1 Generative agents — методический фундамент

- **Park, O'Brien, Cai, Morris, Liang, Bernstein. «Generative Agents: Interactive Simulacra of Human Behavior».** UIST 2023 (DOI 10.1145/3586183.3606763), arXiv:2304.03442. The architectural baseline — memory stream + reflection + planning. Validation: interview-based believability check. Релевантность нам: образец архитектуры персона + память + reflection, на которую опирается весь корпус LLM-симуляторов.
- **Park et al. «Generative Agent Simulations of 1,000 People».** arXiv:2411.10109, 2024 (Stanford HAI). 1052 респондента, two-hour interviews → person-specific generative agents. Validation: GSS test-retest, Big Five, behavioral games. 85% accuracy относительно self-test-retest. Это — ближайшая попытка operational validation generative agents «от человека».
- **Park et al. «Social Simulacra».** UIST 2022 — методический прекурсор Park 2023.

### 3.2 LLM-агенты как симулятор пользователей recommender system

- **Wang, Zhang, Yang et al. (RUC-GSAI). «User Behavior Simulation with Large Language Model based Agents».** arXiv:2306.02552 (2023), accepted *ACM TOIS* 2024-2025 (DOI 10.1145/3708985); GitHub: RUC-GSAI/YuLan-Rec. Это RecAgent. Profiling + memory (sensory / short-term / long-term) + action (item-click + chat + post). Поддерживает ~1000 параллельных агентов. Relevance: full-stack LLM-симулятор, на который мы концептуально опираемся (memory module, action module). Ограничение: validation = behavioral elasticity на распределениях, не индивидуальный choice.
- **Zhang, Hou, Xie, Sun, McAuley, Zhao, Lin, Wen. «AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems».** WWW 2024 (DOI 10.1145/3589334.3645537). Both users and items are LLM-agents, collaborative learning; memory updates через mutual reflection. Релевантность: подсказывает альтернативную архитектуру, где «item-как-агент» — мы такого не делаем, talk-as-agent в нашей задаче избыточно.
- **Zhang, Li, Yang, et al. «On Generative Agents in Recommendation» (Agent4Rec).** SIGIR 2024 (DOI 10.1145/3626772.3657844), arXiv:2310.10108; GitHub: LehengTHU/Agent4Rec. 1000 agents initialized from MovieLens-1M, page-by-page recommendation. Validation: filter-bubble emulation, agreement with real-world filter-bubble effects, causal ablations. Релевантность: эталон validation-стиля «agreement at population level».
- **Yang Ziyi, Zhang Zaibin, Zheng Zirui et al. «OASIS: Open Agent Social Interaction Simulations with One Million Agents».** NeurIPS 2024 Workshop OWA / arXiv:2411.11581. 21 actions, X / Reddit replication. Reproduced phenomena: information spreading, group polarization, herd effects. Релевантность: пример «believability через voskresenie стилизованных фактов».
- **Piao, Yan, Zhang, Li, Yan et al. «AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society».** arXiv:2502.08691, 2025; GitHub: tsinghua-fib-lab/AgentSociety. >10k agents, ~5M interactions. Reproduces 4 real-world experiments (polarization, inflammatory message spread, UBI, hurricanes). Псих-социологическая агентская архитектура (emotions / needs / motivations / cognition).
- **Bougie, Watanabe. «SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation».** ACL 2025 Industry Track / arXiv:2504.12722. Self-consistent personas + persona / memory / perception / brain modules. Movies, books, video games. Validation: micro+macro-level alignment. Ограничение: visual-reasoning module ориентирован на UI, не на cognitive correctness.
- **Lei Wang et al. (отдельно) RecAgent updates.** DOI 10.1145/3708985 — финальная TOIS-публикация.
- **PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator.** SIGIR 2025 (DOI 10.1145/3726302.3730238), arXiv:2506.04551. Big Five трейты + psycholinguistic correlates. Amazon reviews. Релевантность: ровно методический прецедент, как у нас compliant / star_chaser / curious — только через Big Five, не через 3-type compliance.
- **AlignUSER.** arXiv:2601.00930, январь 2026. Counterfactual trajectories + world-modeling для LLM-агентов; closer alignment с реальными людьми, чем prior work.
- **SUBER: An RL Environment with Simulated Human Behavior for Recommender Systems.** arXiv:2406.01631, 2024. LLM-as-rater в gym-обёртке; A2C / PPO / TRPO / DQN.
- **iAgent: LLM Agent as a Shield between User and Recommender Systems.** arXiv:2502.14662 — LLM-агент как прослойка, защищающая пользовательские интересы.
- **CSHI: A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems.** WWW 2025.
- **RecUserSim.** arXiv:2507.22897, WWW Companion 2025.
- **«Mirroring Users».** arXiv:2508.18142, 2025.
- **«Agentic Feedback Loop Modeling Improves Recommendation and User Simulation».** arXiv:2410.20027, SIGIR 2025 — agentic feedback loops дают +11.52% vs single-recommendation-agent, +21.12% vs single-user-agent.
- **Mollabagher, Naghizadeh. «The Feedback Loop Between Recommendation Systems and Reactive Users».** arXiv:2504.07105, 2025 — модель reactive users; теоретически близко к нашей compliance-knob.
- **«A Simulation Framework for Studying Systemic Effects of Feedback Loops in Recommender Systems».** arXiv:2510.14857, 2025 — diversity paradox: feedback-loop увеличивает individual diversity, но снижает collective diversity. Прямая поддержка нашего нарратива «cosine без маски ухудшает дисперсию загрузки».
- **«RecoWorld: Building Simulated Environments for Agentic Recommender Systems».** arXiv:2509.10397.
- **AgentRecBench.** arXiv:2505.19623, 2025; NeurIPS 2025 Datasets & Benchmarks Track spotlight. Первый comprehensive benchmark для agentic recommender systems с traditional baselines на трёх сценариях (classic / evolving-interest / cold-start).
- **Survey on LLM-powered Agents for Recommender Systems.** arXiv:2502.10050, 2025 — три парадигмы: recommender-oriented / interaction-oriented / simulation-oriented.

### 3.3 Critical line — believability ≠ validity

Это — методологический хребет, на котором мы сидим. Эта литература защищает наш отказ от accuracy@1.

- **Larooij, Törnberg. «Validation is the central challenge for generative social simulation: a critical review of LLMs in agent-based modeling».** *Artificial Intelligence Review*, Springer, 2025 (DOI 10.1007/s10462-025-11412-6, online 18 Nov 2025). Систематический обзор ~35 исследований generative ABM. Главные выводы: 15/35 полагаются исключительно на subjective believability; «believability ≠ operational validity»; LLM могут усугублять, а не решать проблему validation для ABM. Это — наш самый сильный ссылочный аргумент.
- **Larooij, Törnberg. «Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review of Generative Social Simulations».** arXiv:2504.03274, 2025 — earlier preprint того же ревью.
- **Larooij, Törnberg. «Can We Fix Social Media? Testing Prosocial Interventions using Generative Social Simulation».** arXiv:2508.03385, 2025; GitHub: cssmodels/prosocialinterventions. Тестируют 6 интервенций; находят modest improvements или ухудшение. Прямой пример generative-ABM, использованного как stress-test policy interventions, что концептуально совпадает с нашим жанром.
- **«Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs» / «Prompting is Not All You Need!».** arXiv:2503.20749, 2025. Real-world online customer behavior data (31,865 sessions, 230,965 actions). Главный quantitative finding: prompt-only LLMs (DeepSeek-R1, Llama, Claude) дают только 11.86% accuracy на индивидуальных action-generation; fine-tuning поднимает до 17-34%. Прямое подтверждение нашего тезиса: индивидуальный choice слабый, distribution-match — единственная разумная цель.
- **Tomašević et al. «Towards Operational Validation of LLM-Agent Social Simulations: A Replicated Study of a Voat / Reddit-like Technology Forum».** arXiv:2508.21740, 2025; GitHub: atomashevic/voat-simulation. YSocial framework + Voat MADOC dataset. Reproduced: activity rhythms, heavy-tailed participation, sparse low-clustering networks, core-periphery, topical alignment, toxicity. Это — текущий state-of-the-art operational-validation подход для LLM-ABM.
- **«Validating Generative Agent-Based Models for Logistics and Supply Chain Management Research».** arXiv:2508.20234, 2025. Dual-validation framework: surface-level behavioral equivalence + process-level decision authenticity.
- **«How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation».** WWW Companion 2024 (DOI 10.1145/3589335.3651955). Identifies data leakage in LLM-based simulator; неуправляемый prompt-template; success зависит от conversational history больше, чем от user-simulator output.
- **Chaney, Stewart, Engelhardt. «How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility».** RecSys 2018 (DOI 10.1145/3240323.3240370). Симулятор-«чистая теория»; показывает feedback-loop homogenization. Pre-LLM, но методически релевантно.
- **«Large Language Models Empowered Agent-Based Modeling and Simulation: A Survey and Perspectives».** *Humanities and Social Sciences Communications*, 2024 (DOI 10.1038/s41599-024-03611-3) — first systematic survey LLM-ABM с 4-domain taxonomy.
- **«Large Language Models for Agent-Based Modelling: Current and Possible Uses Across the Modelling Cycle».** arXiv:2507.05723, 2025.
- **Survey on Evaluation of LLM-based Agents.** arXiv:2503.16416, 2025 — taxonomy объективов / процессов оценки.
- **«Evaluation and Benchmarking of LLM Agents: A Survey».** SIGKDD 2025 (DOI 10.1145/3711896.3736570).

### 3.4 Релевантные к нашей задаче, но не conference-domain

- **«Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents».** arXiv:2405.02957, 2024 — LLM-agents в hospital domain.
- **CitySim.** Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation, ResearchGate 393148547.
- **«Simulating Healthcare Scenarios with LLMs as Agents».** NHS England Data Science PhD Internship project, 2024-2025.
- **«MedicalOS: An LLM Agent based Operating System for Digital Healthcare».** arXiv:2509.11507.

Релевантность: показывают, что LLM-ABM-жанр уже расходится по domain'ам; в conference-domain (включая нас) такой работы пока нет.

### 3.5 Прямого поиска по conference / event-domain LLM-агенты

В литературе работы «LLM-агенты для conference scheduling» **не подтверждены**. Это — часть нашего gap'а (см. §6).

---

## Раздел 4. Открытые датасеты

Раздел про данные, которые в принципе могли бы помочь нашему типу задач, и почему мы остаёмся на синтетических сценариях.

### 4.1 Meetup-RSVP / EBSN

- **Liu et al. «Event-Based Social Networks: Linking the Online and Offline Social Worlds».** SIGKDD 2012 — оригинальный Meetup dump. Содержание: события, группы, RSVP-yes-count, теги, локации. Лицензия: research-use, отдельные dump'ы доступны на GitHub (sahanasub, sijopkd, wuyuehit) и Kaggle (prashdash112/meetup-events-data).
- **Wu Yue. «Meetup Dataset».** IEEE DataPort, 2022, DOI 10.21227/8dr0-p842.
- **Kaggle: «Rsvp dataset» (shakirahmadbarbhuiya, 2023).**
- Использовали в наших экспериментах: 349 talks, 3547 LLM-вызовов, ρ=0.4379 popular-correlation.
- **Ограничение:** actual attendance не публикуется, только RSVP-yes-count + флаги attended/noshow на уровне организатора. Capacity не входит в схему.

### 4.2 SocioPatterns Hypertext-2009 / SFHH

- **Cattuto et al. «Hypertext 2009 dynamic contact network».** SocioPatterns. 110 attendees, 2.5 days, 20-second resolution face-to-face. Лицензия: open research use.
- **SFHH conference dataset 2009.**
- **Ограничение:** только face-to-face proximity, нет per-talk attendance; не подходит для нашей задачи напрямую, но даёт пример «conference data в open access».

### 4.3 UMass CICS course allocation

В открытом доступе detailed dataset с per-student preferences и per-course capacity не подтверждён. Поиск на cs.umass.edu / coursicle / cics.umass.edu возвращает только course descriptions и registration policies, не raw preferences. Если в работе нужен external sanity-anchor через course-allocation, придётся сослаться на Kornbluth-Kushnir 2024 (university data, но не публичный) или Budish 2017 (Wharton, summary stats only).

### 4.4 Course allocation бизнес-школ

- **Wharton / CourseMatch (Budish et al. 2017).** ~1700 students × ~350 courses. Raw bids приватны.
- **Wharton experiment dataset (Budish-Kessler NBER WP 22448, 2017).** Не подтверждено публичное распространение.
- **Sönmez-Ünver Michigan Ross School data.** Не подтверждено публичное распространение.

### 4.5 Quaeghebeur Evolution-2014

- **Quaeghebeur. «Evolution 2014» public per-attendee × per-talk attendance dataset.** N=29. По нашей памяти из reference_validation_defense.md — единственный публичный per-attendee × per-talk dataset. Поиск через Google Scholar / Erik Quaeghebeur / Evolution Society не дал прямого подтверждения существования именно такого dataset'а. **Нужна проверка** — рекомендую скачать через ac.erikquaeghebeur.name либо связаться с автором.

### 4.6 CoSPLib

- **Kheiri, Pylyavskyy, Jacko. CoSPLib.** GECCO Companion 2025; GitHub: ahmedkheiri/CoSPLib. Содержит scheduling-instances (tracks, sessions, rooms, submissions, time-slots), но не attendance / preferences / capacity / talk-text. Подходит как cite-only.

### 4.7 OpenReview / EasyChair / CMT

В открытом доступе bidding-данных от reviewer'ов нет. Это часть закрытого peer-review-pipeline'а. CMT / OpenReview документация описывают bidding-process, но не предоставляют public dump.

### 4.8 Industrial recommendation simulators

- **RecSim NG (Google).** OpenAI-Gym wrapped, document / user / choice models.
- **T-RECS (Lucherini, Sun, Winecoff, Narayanan, 2021).** Princeton CITP. Open-source Python-package. Designed for filter-bubble / polarization / misinformation studies. arXiv:2107.08959.
- **Sim4Rec (Volodkevich, Ivanova, Vasilev, Bugaychenko, Savchenko).** ECIR 2025 (Springer). PySpark-based, large-scale industrial. Apache 2.0. GitHub: sb-ai-lab/Sim4Rec.
- **Synthetic Data-Based Simulators for Recommender Systems: A Survey.** arXiv:2206.11338, 2022 — обзор класса.

Релевантность: показывают, что simulation-based evaluation recsys — established methodology, но ни один из этих фреймворков не моделирует именно conference-program-stress-test.

---

## Раздел 5. Валидационный канон (короткий)

Что считается приемлемой валидацией simulation-based DSS в нашей нише — без real A/B и без public attendance.

1. **Sargent (2013) Verification & Validation of Simulation Models, *Journal of Simulation*.** 12 канонических методов V&V: face validity, extreme condition tests, parameter variability, sensitivity, degenerate behaviour, traces, predictive validation, comparison to other models, animation, event validity, internal validity, multistage validation. Sargent 2020 Winter Simulation Conference tutorial — расширенная версия. Operational validity = «accuracy required for the model's intended purpose over the domain of intended applicability».

2. **Robinson (2014) *Simulation: The Practice of Model Development and Use* (2nd ed).** «Near» / «far» abstraction; structure of simulation projects.

3. **Kleijnen (2005). «An overview of the design and analysis of simulation experiments for sensitivity analysis», *EJOR* 164(2).** DOE для sensitivity analysis: classic fractional factorials + modern Latin hypercube / space-filling + group screening для many-factor моделей. Это — формальная база нашего sensitivity sweep по (capacity, compliance) × seed.

4. **JASSS 27/1/11, 2024. «Methods That Support the Validation of Agent-Based Models: An Overview and Discussion».** 9 подходов: docking, empirical validation, sampling, visualization, bootstrapping, causal analysis, inverse generative social science, role-playing. Прямо обсуждает validation **без** реальных данных через consistency-criteria.

5. **DMDU (Lempert 2003 / Marchau 2019).** XLRM-фрейм, scenario-discovery (PRIM, EMA-workbench, Kwakkel 2017), robustness-metrics (regret, satisficing). «How can we choose actions today consistent with long-term interests» — не «what will the future bring».

6. **Larooij & Törnberg (2025).** «Believability ≠ validity»; subjective expert assessment не является доказательством operational validity. Прямо подкрепляет наш отказ от accuracy@1 как метрики validation.

Что считается приемлемым набором валидационных пунктов для нашей работы (синтез):
- **Внутренняя верификация:** toy-cases, internal consistency, monotonicity in capacity и compliance, repeated seeds.
- **Sensitivity:** sweep по (capacity × compliance × seed × policy) с DOE-подходом по Kleijnen.
- **Data-side anchor:** калибровка compliance долей на real Meetup-RSVP (3-type 71.7 / 21.3 / 7.0).
- **Face validity LLM-слоя:** distribution-match popularity ρ=0.438 на Meetup (наш эксперимент).
- **External sanity:** course-allocation вместо conference (Kornbluth-Kushnir 2024 / Budish 2017) — limited-resource allocation, real preferences + real capacities, но **другой domain**, явно так и фиксируется.
- **Limitations:** operational validity против реальной посещаемости конкретной конференции не выполнена ввиду отсутствия публичных данных.

Этот стек — ровно то, что описано в нашем `reference_validation_defense.md`. Литература за последний год ничего из этого не опровергла; Larooij-Törnberg 2025 явно уширили легитимность такого подхода.

---

## Раздел 6. Исследовательский пробел

В обзоре собрано **много** работ по conference scheduling как OR (Vangerven 2018/2024, Pylyavskyy-Jacko-Kheiri 2024, Bulhões 2022, Stidsen-Pisinger-Vigo 2018, Manda 2019), **много** работ по capacity / congestion-aware recsys (ReCon 2023, FEIR 2024, FairRec 2020, capacity-aware POI 2023), **много** работ по LLM-агентам как симулятору пользователей recsys (RecAgent, Agent4Rec, OASIS, AgentSociety, SimUSER, PUB), **много** работ по DSS / DMDU методике (Lempert 2003, Marchau 2019, Kwakkel 2013/2017, Sargent 2013, JASSS 27/1/11) и **много** работ по критике LLM-ABM-validity (Larooij-Törnberg 2025).

Чего в обзоре **нет**:

1. **Сценарного DMDU-стресс-теста, применённого именно к program-design конференции с параллельными сессиями.** Conference-scheduling работают с known preferences (Vangerven, Rezaeinia) или с no preferences (Pylyavskyy, CoSPLib, Bulhões); сценарной развёртки по неопределённости спроса × вместимости × compliance ни одна из них не делает.
2. **Двух независимых симуляторов отклика участника** (параметрический MNL + LLM-агентский), сравниваемых между собой ради фальсификации выводов параметрической модели. В литературе по LLM-симуляторам есть calibration-LLM-vs-real (Park 2024, Tomašević 2025), но не calibration-LLM-vs-parametric-в-той-же-задаче.
3. **Capacity-aware-baseline'ов, специфичных для conference-program (а не job / POI / cultural-activities).** Все capacity / congestion-aware recsys-работы — в job (ReCon, FEIR), в POI (capacity-aware POI ESWA 2023), в e-recruitment (Patro), в cultural activities (DPP 2025). Conference-domain отдельно не разработан.

**Наш вклад в одном предложении.** Мы предлагаем сценарный аналитический полигон / DSS-слой для стресс-теста программы конференции при неопределённом спросе на параллельные сессии; вместо прогноза реальной посещаемости — сравнение политик рекомендаций между собой через `policy × capacity × compliance × seed` матрицу, с двумя независимыми симуляторами отклика (параметрическим MNL и LLM-агентским) для перекрёстной фальсификации выводов и валидацией по канону Sargent / Kleijnen / DMDU / Larooij-Törnberg вместо accuracy против отсутствующих ground-truth данных.

---

## Раздел 7. Что из обзора мы можем встроить в систему

### 7.А. Уже сделано (можно явно упомянуть в §3.1)

- **MNL-структура choice c capacity-penalty.** Соответствует линии Aouad / Mitrofanov 2024 (extensions classical MNL).
- **Capacity-aware reranker (политика capacity_aware / capacity_aware_mmr).** Соответствует ReCon (Mashayekhi 2023) + capacity-aware POI 2023 ESWA.
- **MMR-baseline.** Carbonell-Goldstein 1998 + SMMR SIGIR 2025.
- **Calibrated-baseline.** Steck 2018 + Steck-Survey 2025.
- **DPP-эксперимент.** Chen-Zhang-Zhou NeurIPS 2018.
- **3-type compliance калибровка на Meetup-RSVP.** Линия Liu-He-Tian-Lee-McPherson-Han SIGKDD 2012 + наши собственные 71.7 / 21.3 / 7.0.
- **LLM-симулятор с persona / memory / capacity-policy-вне-агента.** Архитектурно: Park 2023 + RecAgent 2024 + SimUSER 2025 + PUB 2025 (для Big-Five-варианта).
- **Distribution-match validation на Meetup (ρ=0.438).** В каноне Park 2024 (1052 person validation), Tomašević 2025 (Voat operational validation), Larooij-Törnberg 2025 (validation criterion).
- **Sensitivity sweep по сценариям.** В каноне Kleijnen 2005 + Kwakkel 2017 EMA-workbench + Sargent 2013 parameter-variability.

### 7.Б. Лёгкие улучшения (если есть время до 13.05.2026)

1. **Формализовать `hall_load_gini` и `mean_overload_excess` в терминах envy / inferiority / MMS / EF1.** Использовать язык FairRec 2020 + FEIR 2024 — даст красивую формальную обвязку §3.2 без переписывания кода.
2. **Добавить scenario-discovery шаг через PRIM или EMA-workbench (Kwakkel 2017).** На существующих `policy × capacity × compliance × seed` логах. Даст явный ответ «при каких комбинациях параметров политика capacity_aware_mmr ломается» — это standard-практика DMDU и сильно усиливает §4 защиты.
3. **Optimal-transport variant как четвёртая capacity-aware политика.** Ровно ReCon (Mashayekhi 2023) применённый к нашему случаю. Не переоткрытие — реализация ровно того же подхода, проверка совместимости с нашей формулировкой.
4. **Cite-only reference на CoSPLib (Kheiri 2025).** Хотя бы в Limitations: «существующий benchmark conference scheduling в open access; не пригоден для наших exp-целей, потому что нет attendance / preferences / capacity / talk-text».
5. **Расширить compliance-калибровку с 3-type на Big-Five как в PUB (SIGIR 2025).** Не уверен, что окупится; зато даст красивую вторую sensitivity-ось.

### 7.В. Идеи «на потом» (после защиты, в продолжении)

1. **External anchor через course-allocation (Kornbluth-Kushnir 2024).** Прогнать `policy × capacity × compliance × seed` матрицу на course-allocation instances и показать, что качественный pattern перегрузки тот же.
2. **Operational validation через scenario-discovery match с реальной conference (если получим dump).** Текущий `Quaeghebeur Evolution-2014 (N=29)` — единственный публичный per-attendee × per-talk dump, но **нужна проверка**, что он реально существует и доступен.
3. **Reactive-users расширение compliance-knob.** Как в Mollabagher-Naghizadeh 2025 — compliance не статический параметр, а функция от истории рекомендаций. Для DMDU это новая ось неопределённости.
4. **Calibration-LLM-vs-parametric как методический вклад.** Развернуть в отдельный paper для JASSS / *AAMAS* / *RecSys* — «two-simulator falsification framework for DSS without ground-truth».
5. **AgentRecBench-style benchmark для conference-domain.** В обзоре пока такой работы нет; есть AgentRecBench общий, но conference-specific нет.
6. **Scenario-discovery PRIM box на наших логах E2 для нахождения «зоны провала capacity-mask».** Пере­числит явные комбинации параметров, при которых `mean_overload_excess` для capacity-mask ≥ для capacity_aware_mmr — это ровно negative-finding-confirmation, который усилит §4.4.

---

## Хвост (тайминг)

start: 16:18, end: 16:42, elapsed: 24 min (из них поиск и анализ ≈ 18 мин, написание отчёта ≈ 6 мин)
