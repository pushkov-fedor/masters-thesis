# Сводка по LLM-симуляторам, capacity-датасетам и congestion-aware recsys

Дата сборки: 30.04.2026. Сводка для Главы 1 (обзор) и Заключения (направления развития) ВКР по рекомендательной системе для конференций (Mobius/Heisenbug) с учётом ограничений вместимости параллельных сессий.

---

## Задача 1. LLM-симуляторы пользовательского поведения (2023–2026)

### 1.1. Park et al., 2023 — Generative Agents: Interactive Simulacra of Human Behavior

- **Что моделирует.** 25 агентов в Sims-подобной песочнице Smallville; каждый агент планирует день, общается, посещает мероприятия, формирует отношения. Идея, на которой построены все последующие LLM-симуляторы.
- **Архитектура.** LLM + долговременная память (полный лог опыта на естественном языке) + reflection (синтез воспоминаний в обобщения) + planning. Извлечение памяти по recency + importance + relevance.
- **Социальная коммуникация.** Да, ключевая часть работы: парные диалоги, передача информации по графу знакомств, спонтанная самоорганизация (агенты сами устроили вечеринку на День святого Валентина).
- **Валидация.** Human evaluation believability + ablation модулей (memory, reflection, planning); статистических тестов с реальными данными нет.
- **Метрики.** Believability score, частота поддержания отношений, фактическая корректность ответов в интервью.
- **Код.** Открыт: github.com/joonspk-research/generative_agents (MIT-style).
- **Ссылка.** arXiv:2304.03442, UIST '23.

### 1.2. Agent4Rec (Zhang et al., SIGIR 2024 perspective) — On Generative Agents in Recommendation

- **Что моделирует.** 1000 LLM-агентов, инициализированных из MovieLens-1M (вкусы + социальные черты), взаимодействуют с recommender'ом постранично (page-by-page).
- **Архитектура.** Profile + Memory (factual + emotional) + emotion-driven reflection + Action.
- **Действия агента.** Watch, rate, evaluate, exit, interview. То есть классический сигнал «клик/рейтинг/выход», расширенный free-text интервью.
- **Социальная коммуникация.** Отсутствует. Агенты независимы, не обмениваются сообщениями. Это важное отличие от нашей работы и потенциальная точка дифференциации.
- **Валидация.** Сравнение симулятор-распределений с реальными MovieLens (rating distribution, exit behavior, taste alignment); проверка преимуществ персонализации; ablation модулей.
- **Метрики.** Discrepancy между симулированными и реальными распределениями, эффект персонализации, filter bubble.
- **Код.** Открыт: github.com/LehengTHU/Agent4Rec (MIT). Поддерживает Random, Pop, MF, MultVAE, LightGCN; ChatGPT-3.5 в основе.
- **Ссылка.** arXiv:2310.10108.

### 1.3. RecAgent / YuLan-Rec (Wang et al., TOIS 2025) — User Behavior Simulation with LLM-based Agents

- **Что моделирует.** Симуляция пользователей recsys-сервиса Renmin University of China; агент может смотреть фильм, чатиться с другим агентом, постить публично.
- **Архитектура.** Profile + Memory (sensory / short-term / long-term, по аналогии с человеческой когнитивной моделью) + Action. Recommender-модуль выдаёт ленту.
- **Социальная коммуникация.** Да: chat между агентами и broadcasting на «социальную сеть» симулятора. Это ближайший аналог нашему inter-slot chat прототипу.
- **Валидация.** Сравнение с человеческой оценкой (отстаёт от реального человека на 8.7%, обгоняет лучший baseline на 45.8%).
- **Код.** Открыт: github.com/RUC-GSAI/YuLan-Rec.
- **Ссылка.** arXiv:2306.02552, ACM TOIS 2025 (DOI: 10.1145/3708985).

### 1.4. AgentCF (Zhang et al., WWW 2024) — Collaborative Learning with Autonomous Language Agents for Recommender Systems

- **Что моделирует.** Не только пользователи, но и **айтемы** — LLM-агенты с памятью; пользователь и айтем «договариваются», их память итеративно оптимизируется по реальным interaction-логам (collaborative memory-based optimization).
- **Социальная коммуникация.** Между парой user-item, не между пользователями. Имплицитное распространение предпочтений.
- **Валидация.** Стандартные recsys-метрики на CDs and Vinyl, Office Products (Amazon).
- **Код.** Через индекс github.com/tsinghua-fib-lab/LLM-Agent-for-Recommendation-and-Search.
- **Ссылка.** arXiv:2310.09233, DOI: 10.1145/3589334.3645537.

### 1.5. OASIS (Yang et al., ICLR 2025) — Open Agent Social Interaction Simulations with One Million Agents

- **Что моделирует.** До миллиона LLM-агентов в средах, имитирующих X (Twitter) и Reddit. Воспроизводит information spreading, group polarization, herd effects.
- **Действия.** 23 действия, включая создание постов, лайки, дизлайки, репосты, вложенные комментарии, follow/mute, поиск, **создание group chat и messaging**, interview.
- **Социальная коммуникация.** Богатейшая в обзоре — это её ядро. Полноценные посты, комментарии, личные сообщения, групповые чаты.
- **Валидация.** Воспроизведение известных социальных феноменов (polarization, herd) с количественными матчами на реальные паттерны Reddit/X. Полный валидационный пайплайн всё же не закрыт — авторы признают, что поведение зависит от модели, есть герд-эффекты.
- **Код.** Открыт: github.com/camel-ai/oasis (Apache 2.0), пример датасета user_data_36.json (Reddit), также есть oasis-dataset на HuggingFace.
- **Ссылка.** arXiv:2411.11581.

### 1.6. AgentSociety (Piao et al., 2025) — Large-Scale Simulation of LLM-Driven Generative Agents

- **Что моделирует.** ≈10 тыс. агентов и до 5 млн взаимодействий в реалистичной городской среде. Транспорт, инфраструктура, общественные ресурсы. Психологическая модель: эмоции, потребности (Maslow), теория планируемого поведения.
- **Социальная коммуникация.** P2P (личные сообщения), P2G (peer-to-group) и group chat — Message Layer как отдельный архитектурный слой.
- **Сценарии.** Polarization, распространение токсичного контента, эффект UBI, ураганы, urban sustainability.
- **Валидация.** Сопоставление выходов с реальными экспериментальными результатами (по упоминанию авторов). Конкретные метрики по каждому сценарию в репозитории.
- **Код.** github.com/tsinghua-fib-lab/AgentSociety (Apache 2.0, исключение для commercial-папки).
- **Ссылка.** arXiv:2502.08691.

### 1.7. SimUSER (Bougie & Watanabe, 2025) — Simulating User Behavior with LLMs for Recommender System Evaluation

- **Что моделирует.** Believable agent-прокси для оценки recsys; идентификация self-consistent персон из исторических данных, обогащение бэкграундом.
- **Архитектура.** Persona + Memory (episodic + knowledge-graph) + Perception (визуальный reasoning поверх обложек/превью) + Brain.
- **Социальная коммуникация.** Не центральная; фокус — индивидуальное поведение.
- **Валидация.** Сильный пункт: соответствие реальным людям на микро- и макро-уровне; оффлайн A/B-тест помог настроить параметры реальной системы и улучшил онлайн-engagement.
- **Ссылка.** arXiv:2504.12722, ACL Industry Track 2025.

### 1.8. RecoWorld (Liu et al., Meta, 2025) — Simulated Environments for Agentic Recommender Systems

- **Что моделирует.** Blueprint для multi-turn взаимодействия simulated user ↔ agentic recommender. Цель — RL-обучение recsys-агента.
- **Архитектура.** Dual-view: симулятор пользователя (генерирует reflective instructions при намёке на disengagement) + agentic recommender. Поддержка text/multimodal/semantic-ID.
- **Социальная коммуникация.** Не социальная — диалог user-recommender. Поддержка multi-agent для целевых популяций.
- **Валидация.** Парадигма user-instructs, recommender-responds; статья — vision/blueprint, конкретные оффлайн-метрики представлены, но без публичного бенчмарка.
- **Ссылка.** arXiv:2509.10397.

### 1.9. MiroFish (2026) — Swarm-Intelligence Multi-Agent Prediction Engine

- **Что моделирует.** Не научная статья, а инженерный проект (open source + коммерческая обвязка); 1М агентов, knowledge-graph из загружаемых документов, две среды — Twitter-like и Reddit-like, агенты постят, спорят, формируют коалиции, меняют мнение.
- **Социальная коммуникация.** Да, центральная функция; агенты можно «опросить», подключить Report Agent, подсунуть counterfactual-переменную в середине прогона.
- **Валидация.** Открытая проблема и сами авторы это признают. Симуляция «убедительная, но не предсказательно точная»; herd-эффекты сильнее, чем у людей; рекомендуют прототипы <40 агентов <50 циклов.
- **Код.** Open-source: github.com/666ghj/MiroFish (китайский оригинал) и форк github.com/nikmcfly/MiroFish-Offline (Neo4j + Ollama, без облака). Есть статьи в Medium / DEV Community / blocmates как primary source.
- **Статус для ВКР.** Цитировать как практический artefact из индустрии, не как пиаренное research-baseline. На сравнение «свой симулятор vs MiroFish» уйдёт ровно один абзац в Главе 1: масштаб разный, цели разные.

### 1.10. Методологический контекст: критика валидации

#### 1.10.1. Larooij & Törnberg (Amsterdam, 2025) — Do LLMs Solve the Problems of Agent-Based Modeling?

- arXiv:2504.03274. Критический обзор generative ABMs.
- **Главные тезисы.** Валидация в области поставлена слабо: большинство работ опираются только на subjective believability assessments; даже самые строгие подходы не предъявляют **operational validity**. LLM скорее **усугубят** старые проблемы ABM, чем решат: чёрный ящик мешает причинному анализу, нет преемственности с историческими дискуссиями.
- **Что взять.** Различение subjective (Turing-style) vs objective (статистические тесты против реальных данных) валидации — для нашей Главы 1 это must-cite, обосновывает, почему мы делаем гибридный симулятор и метрики как acceptance/diversity/load-balance.

#### 1.10.2. Seshadri et al. (Cohere/UC Irvine, 2026) — Lost in Simulation

- arXiv:2601.17087. Эмпирическая критика на τ-Bench retail tasks с реальными участниками из США, Индии, Кении, Нигерии.
- **Главные тезисы.** LLM-simulated users — ненадёжные прокси: agent success rates варьируются до 9 п.п. при смене user-LLM; систематическая miscalibration (недооценка hard tasks, переоценка средних); fairness gap для AAVE-говорящих усиливается с возрастом.
- **Что взять.** Прямая поддержка нашего тезиса о необходимости двухслойного симулятора (параметрический + LLM) и сравнения нескольких backbone-моделей. Также — обоснование «реальных данных нет, поэтому каждый прогон отчитывается с распределением по seed/LLM».

### 1.11. Сводная таблица социальной коммуникации

| Работа | Социальная коммуникация между агентами | Тип |
| --- | --- | --- |
| Park et al. 2023 | Да, ядро | Диалоги, передача информации по графу |
| Agent4Rec | Нет | Только взаимодействие с recsys |
| RecAgent / YuLan-Rec | Да | Chat + публичные посты |
| AgentCF | Частично | User-item диалог, не user-user |
| OASIS | Да, очень богатая | Посты, комментарии, follow, group chat |
| AgentSociety | Да | P2P + P2G + group chat (Message Layer) |
| SimUSER | Нет | Индивидуальное поведение |
| RecoWorld | Условно | Multi-turn user-recommender, multi-agent опционально |
| MiroFish | Да | Twitter-like + Reddit-like среда |

**Что важно для ВКР.** Наш симулятор с inter-slot chat прототипом находится в нише **между Agent4Rec/SimUSER (нет соцкоммуникации) и OASIS/AgentSociety (всё построено вокруг неё)**. Это правильное позиционирование: capacity-aware recsys + лёгкая социальная динамика для влияния на выбор сессии.

---

## Задача 2. Публичные датасеты с capacity-структурой

### 2.1. GoalZone Fitness Class Bookings (DataCamp)

- **Что внутри.** 1500 записей о бронированиях занятий: booking_id, months_as_member, weight, days_before, day_of_week, time, category, attended.
- **Capacity-ресурс.** Класс (занятие) с зашитой вместимостью **15 или 25**, прямо в задаче. Заявленное business problem — классы переполнены по броням, но по факту приходят меньше; нужно предсказать attendance, чтобы освободить слоты.
- **Размер.** Маленький (1.5K observations).
- **URL.** Kaggle: kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification.
- **Лицензия.** DataCamp Practice License (учебная), де-факто публично доступна.
- **Адаптация к нашей задаче.** Натуральный матч на «N пользователей выбирают между ресурсами с capacity»: пользователи = members, ресурсы = классы (день+время+категория), capacity = 15/25, есть behavioral сигнал (attended). Минусы: нет преференций между несколькими классами одновременно, выбор уже сделан, очень малый размер.

### 2.2. Reviewer-Paper Assignment Gold Standard (Stelmakh et al., 2023)

- **Что внутри.** 477 self-reported expertise scores от 58 исследователей на статьи, которые они читали; ground-truth ranking.
- **Capacity-ресурс.** Reviewer load (typical conference quota — review per reviewer). В исходной публичной версии capacity не зафиксирована, но Reviewer Assignment Problem всегда ставится с per-reviewer load + per-paper minimum (3-4 reviews).
- **Размер.** Очень маленький, но это бенчмарк на качество similarity-метрик, а не end-to-end recsys.
- **URL.** github.com/niharshah/goldstandard-reviewer-paper-match; arXiv:2303.16750.
- **Лицензия.** Открытая (исследовательская).
- **Адаптация.** Косвенная: capacity накладывается алгоритмом (Hungarian / LP с per-reviewer ограничением). Хорош как «второй домен» для робастности.

### 2.3. OutWithFriendz / GEVR (Tang et al., 2019)

- **Что внутри.** 625 пользователей, 500+ групповых событий, mobility traces, voting на места, RSVP, итоговое участие. Группы в основном маленькие (3-6 человек).
- **Capacity-ресурс.** Не явный, но на venue-уровне можно вывести из popularity; модель в статье прямо учитывает «participant scale control».
- **Размер.** Маленький.
- **URL.** arXiv:1710.02609 / arXiv:1903.10512; данные от авторов.
- **Адаптация.** Подходит для group recommendation, но для нашей конференц-задачи слабо: нет параллельных сессий с фиксированной capacity.

### 2.4. Meetup / EBSN Datasets

- **Что внутри.** Несколько публичных снапшотов:
  - **wuyuehit/meetup_dataset** (GitHub): event-group, user-event, user-group relations + теги. На 30.04.2026 репозиторий жив (https://github.com/wuyuehit/meetup_dataset). Альтернатива магистрантскому 404 — если он шёл по ссылке `wuyuehit/Meetup-Recommendation-Dataset`, то правильная — `wuyuehit/meetup_dataset`. **Рекомендую перепроверить вручную.**
  - **IEEE DataPort Meetup Dataset** (Yue Wu, 2022, DOI 10.21227/8dr0-p842) — независимый снапшот.
  - **largenetwork.org/ebsn** — sample-датасет.
  - Dataset-уровни: ≈4M users, ≈2M events, 70K groups, 80K tags для крупных снапшотов; есть локальные снапшоты NYC по 134K событий 2018 г.
- **Capacity-ресурс.** **RSVP yes-count + явно заявленная вместимость события.** В практике EBSN-исследований capacity моделируется (Liu et al. 2017 — reverse random walk + participant scale control); в исходных API-полях Meetup есть `rsvp_limit` для events — нужно проверить, попал ли он в конкретный публичный снапшот.
- **Лицензия.** Mixed: данные изначально через Meetup API (закрытый с 2024), публичные снапшоты — academic research use.
- **Адаптация к нашей задаче.** **Лучшая совместимость**, если capacity действительно есть в данных:
  - События ≈ доклады/сессии,
  - Группы/локации ≈ залы,
  - RSVP ≈ выбор и acceptance@k,
  - rsvp_limit ≈ capacity.
- **Риск.** Старые снапшоты могут не содержать `rsvp_limit`; нужно открыть конкретные CSV/JSON и проверить поле руками. Это перекрывается нашим правилом «verify_data_semantics».

### 2.5. Wharton/Harvard Course Allocation

- **Что внутри.** Wharton — 1700 студентов × до 350 курсов в семестре, преференции и аллокации; Harvard Business School — данные ≈900 студентов с истинными и стратегическими предпочтениями (HBS); UMass Amherst — 1000+ student preferences (Kornbluth & Kushnir, 2024-2025).
- **Capacity-ресурс.** Course max enrollment — ядро задачи. Wharton pseudo-market и Harvard mechanism прямо оптимизируют под per-course capacity.
- **Размер.** От сотен до тысяч studentsxcourses.
- **URL.** Часть данных через arXiv:2412.05691 (Kornbluth-Kushnir), часть — через AEAweb / HBS / Wharton с ограниченным доступом. Чисто публичных дампов на download почти нет.
- **Лицензия.** Зачастую restricted to research-on-request.
- **Адаптация.** Хорошо ложится тематически (студенты ≈ участники, курсы ≈ доклады, capacity = max enrollment, конкуренция за слоты), но **загрузить дампом не выйдет**, придётся идти через авторов.

### 2.6. MOOC datasets — MOOCCube / MOOCCubeX / XuetangX / COCO

- **Что внутри.** XuetangX (Tsinghua) даёт MOOCCube/MOOCCubeX — 700+ курсов, 100K концептов, 8M student behaviors; есть снапшоты до 351M activities, 0.8M learners, 1.6K courses; COCO — отдельный публичный снапшот. URL: moocdata.cn.
- **Capacity-ресурс.** Прямой capacity нет (MOOC по природе нелимитированный), но есть аналог — учебный track / cohort. Слабый матч.
- **Адаптация.** Подходит как «контрольный recsys-домен без capacity», на котором политики Cosine/MMR работают как обычно.

### 2.7. Yelp Open Dataset

- **Capacity-ресурс.** Capacity нигде не зашита: атрибуты бизнеса описывают часы работы, парковку, ambience и т.д. Поле «capacity» отсутствует, поэтому приходится либо априорно проставить (искусственно, как MovieLens), либо исключить.
- **Вердикт.** Не подходит для capacity-aware экспериментов в чистом виде. Использовать только в аппендиксе, если очень нужно «реальные бизнесы».

### 2.8. Recruit Restaurant Visitor Forecasting (Kaggle)

- **Что внутри.** Реальное число визитов (visitors per day) в японских ресторанах + резервирования.
- **Capacity-ресурс.** Не прямой, но reservations на час — proxy для capacity-окна.
- **Адаптация.** Слабая для recsys: задача — forecasting, а не выбор пользователем между N ресурсами.

### 2.9. Рейтинг датасетов от лучшего к худшему для нашей задачи

1. **Meetup / EBSN snapshots** — лучший потенциальный матч (события, RSVP, опционально rsvp_limit). Условие: своими руками открыть JSON и подтвердить наличие capacity-поля.
2. **GoalZone Fitness** — идеальная учебная семантика (capacity 15/25 встроена в задачу), но крошечный размер. Хорош как **второй домен** в Главе 2/3 для робастности.
3. **Wharton/HBS course allocation** — идеальная семантика, но ограниченный доступ. На уровне «упомянуть как мотивацию» в Главе 1 и «направление дальнейшей работы» в Заключении.
4. **Reviewer-Paper Assignment Gold Standard** — отличный методологический фон для capacity-as-load constraint, малый размер. Использовать как **сравнительную методологическую опору** в Главе 1 (см. defense-must-cite).
5. **OutWithFriendz / GEVR** — group recommendation домен; ценность ограничена, упомянуть в обзоре.
6. **MOOC (MOOCCube, COCO)** — без capacity по сути; использовать для «контрольной» проверки методологии.
7. **Yelp / Recruit Restaurant** — без capacity-структуры; не использовать в основной части.

**Итог.** Если получится подтвердить наличие capacity-поля в одном из публичных снапшотов Meetup, можно безболезненно заменить MovieLens на Meetup в основной части и оставить MovieLens как «контрольный» домен. Если нет — компромисс: MovieLens **+** GoalZone (оба с явным capacity-смыслом, GoalZone — реальный, MovieLens — синтетический). Это сильнее методологически, чем только MovieLens.

---

## Задача 3. Свежие congestion-aware / capacity-constrained recsys (2023–2025)

### 3.1. ReCon: Reducing Congestion in Job Recommendation using Optimal Transport (RecSys 2023; расширен в IEEE Access 2024)

- **arXiv.** 2308.09516. **DOI.** 10.1145/3604915.3608817 (RecSys 2023).
- **Авторы.** Mashayekhi, Kang, Lijffijt, De Bie (Ghent).
- **Идея.** Congestion = неравномерное распределение айтемов в выдаче; в job-recommendation это критично (одна вакансия — один наём; рекомендация многим = фрустрация). Решение: optimal transport-компонента поверх ranking-модели, обеспечивающая равномерное распределение вакансий по соискателям; multi-objective.
- **Метрики.** Congestion, Coverage, Gini, NDCG, Recall, Hit Rate. Pareto-оптимально на нескольких гиперпараметрах.
- **Код.** github.com/aida-ugent/ReCon.
- **Статус для ВКР.** **Прямой must-cite в Главе 1**: ровно та же постановка («айтем с ограничением — выдача — congestion»), что и у нас.

### 3.2. Mashayekhi et al., 2024 — Scalable Job Recommendation with Lower Congestion using Optimal Transport (IEEE Access)

- Расширение ReCon на масштабируемый OT.
- **Статус.** Цитировать рядом с ReCon как «эволюция метода».

### 3.3. BankFair: Balancing Accuracy and Fairness under Varying User Traffic (Cao et al., 2024)

- **arXiv.** 2405.16120.
- **Идея.** Provider fairness через **гарантированную минимальную exposure**; bankruptcy-inspired re-ranking. Экспозиция трактуется как «капитал», который перераспределяется между провайдерами при колеблющемся трафике пользователей.
- **Связь с нашей задачей.** Симметрично: «минимум показов на докладе» вместо «максимум показов = capacity». Идея многоэтапной балансировки (long-term provider vs short-term user) полезна.
- **Статус.** Цитировать в Главе 1 как соседнее направление (fairness with hard exposure floor).

### 3.4. User-item fairness tradeoffs in recommendations (Yang & Allcott, 2024)

- **arXiv.** 2412.04466.
- **Идея.** Теоретическая модель fairness с двумя сторонами; явление «misestimated users получают непропорционально плохие рекомендации при item-fairness constraints».
- **Статус.** Цитировать как теоретический фон для Главы 1 (почему capacity ломает persona-recall неравномерно).

### 3.5. Scalable and Provably Fair Exposure Control for Large-Scale Recommender Systems (Sato et al., 2024)

- **arXiv.** 2402.14369.
- **Идея.** Эффективные алгоритмы для ranking с ограничениями на экспозицию item-side; масштабируемость до миллионов айтемов.
- **Статус.** Цитировать как методологическое приближение к нашему capacity-aware reranking.

### 3.6. A Survey of Real-World Recommender Systems (2025)

- **arXiv.** 2509.06002. Промышленный обзор 2020-2024.
- **Польза.** Терминологический фон и список индустриальных constraints; цитировать как контекст в Главе 1.

### 3.7. Course Allocation with Credits via Stable Matching (Rodríguez & Manlove, SAGT 2025)

- **arXiv.** 2505.21229.
- **Идея.** Stable matching студент-курс с credit-ограничениями и lower quotas; полиномиальные алгоритмы и hardness-результаты.
- **Связь.** Чисто алгоритмический angle на capacity-constrained allocation; **полезно для Заключения** как «классическое OR-направление, в которое мы добавляем поведенческий слой через симулятор».

### 3.8. Scheduling Conferences using Data on Attendees' Preferences (Tandfonline 2024)

- **DOI.** 10.1080/01605682.2024.2310722, JORS 75(11).
- **Идея.** Anonymised данные о преференциях посетителей как драйвер scheduling; двухступенчатый pipeline (тематические сессии → доклады внутри).
- **Связь.** **Прямой методологический сосед** для Главы 1 (тот же домен — конференционный scheduling). Из must-cite списка магистранта (вместе с Tandfonline 2012 — hybrid clustering для parallel session scheduling).

### 3.9. Хронологическая шкала ключевых работ для Главы 1

| Год | Работа | Категория |
| --- | --- | --- |
| 2012 | Hybrid clustering for parallel session scheduling (Tandfonline) | Conference scheduling, OR |
| 2023 | Park et al. — Generative Agents | LLM-симулятор, фундамент |
| 2023 | RecAgent v1.0 (arXiv 2306.02552) | LLM-симулятор для recsys |
| 2023 | Stelmakh — Reviewer Assignment Gold Standard | Capacity-constrained matching, бенчмарк |
| 2023 | ReCon (RecSys 2023) | Congestion-aware recsys |
| 2024 | Agent4Rec (SIGIR 2024) | LLM-симулятор для recsys |
| 2024 | AgentCF (WWW 2024) | LLM-симулятор для recsys |
| 2024 | BankFair (arXiv 2405.16120) | Fairness with exposure constraints |
| 2024 | Sato et al. — Provably Fair Exposure (arXiv 2402.14369) | Capacity-aware ranking |
| 2024 | Tandfonline JORS conference scheduling | Conference scheduling, OR |
| 2025 | OASIS (ICLR 2025) | Социальный LLM-симулятор |
| 2025 | AgentSociety (arXiv 2502.08691) | Социальный LLM-симулятор |
| 2025 | SimUSER (ACL Industry 2025) | LLM-симулятор для recsys |
| 2025 | RecoWorld (arXiv 2509.10397) | Agentic recsys environment |
| 2025 | Larooij & Törnberg — критика валидации (arXiv 2504.03274) | Методология |
| 2025 | Course Allocation with Credits (SAGT 2025) | Capacity-constrained matching |
| 2026 | Lost in Simulation (arXiv 2601.17087) | Критика LLM-симуляторов |
| 2026 | MiroFish (Medium / open source) | Индустриальный multi-agent |

---

## Итоговая рекомендация магистранту

### Что взять в Главу 1 как обязательный обзор (must-cite)

- **Park et al. 2023** — фундамент LLM-агентов; одна цитата в начале раздела про симуляторы.
- **Agent4Rec (arXiv:2310.10108) + RecAgent (arXiv:2306.02552) + AgentCF (arXiv:2310.09233)** — три работы для recsys-симуляции; именно их сравнивать с собственным симулятором по таблице (память, личность, социальная коммуникация, capacity-awareness).
- **OASIS (arXiv:2411.11581) + AgentSociety (arXiv:2502.08691)** — социальные крупномасштабные симуляторы; цитировать как «предельный случай», который мы намеренно не воспроизводим (другая задача), но из которого заимствовали идею inter-agent chat.
- **Larooij & Törnberg (arXiv:2504.03274)** — рамка валидации; использовать как обоснование выбора метрик и двухслойной архитектуры.
- **Lost in Simulation (arXiv:2601.17087)** — конкретные эмпирические доказательства того, что один LLM-прокси ненадёжен; обоснование для усреднения по seed/модели.
- **ReCon (arXiv:2308.09516, RecSys 2023)** — прямой методологический предшественник в congestion-aware recsys; обязательная цитата.
- **BankFair (arXiv:2405.16120) + Sato et al. (arXiv:2402.14369)** — соседние ветки exposure/fairness с жёсткими ограничениями; абзац или таблица отличий.
- **Stelmakh — Reviewer Assignment Gold Standard (arXiv:2303.16750)** — пример gold-standard для capacity-constrained matching; уже включён в нашу defense-validation сводку.
- **Tandfonline JORS 2024** — методологический сосед в домене конференционного scheduling; цитировать как «классическая OR-постановка той же задачи» и подчеркнуть, что мы добавляем поведенческий слой.

### Что взять в Заключение (направления дальнейшей работы)

- **OASIS / AgentSociety** — масштабирование на 10K-1M агентов и вынос всей социальной динамики в отдельный слой как полноценное расширение; кандидат на статью после ВКР.
- **RecoWorld** — agentic recsys с RL-обучением политики на симуляторе; прямой следующий шаг после защиты.
- **Course Allocation with Credits via Stable Matching (SAGT 2025)** — формализм для будущей работы с честным алгоритмическим matching, в который пристёгивается наш симулятор как behavioral слой.
- **MiroFish** — упомянуть как индустриальный artefact и контраст: research-симулятор vs production-инструмент.

### Что взять как замену MovieLens

- **Первый выбор:** Meetup (wuyuehit/meetup_dataset или IEEE DataPort 10.21227/8dr0-p842). Перед использованием — открыть JSON/CSV руками и подтвердить наличие `rsvp_limit` или эквивалентного capacity-поля. Если есть — заменить MovieLens на Meetup как основной домен.
- **Второй выбор (или дополнение):** GoalZone (Kaggle). Реальный capacity 15/25, но крошечный размер. Подходит как второй домен для проверки робастности политик.
- **Если оба не дадут capacity:** оставить текущую конфигурацию (MovieLens с искусственным capacity) **+ GoalZone** в качестве дополнения; в тексте честно указать, что это методологический компромисс из-за отсутствия публичного датасета конференций.

### Тактические замечания для текста

- В абзаце про симуляторы важно подчеркнуть **позиционирование**: наш симулятор стоит между «расщеплёнными агентами без социальной коммуникации» (Agent4Rec, SimUSER) и «соцсетевыми крупномасштабными» (OASIS, AgentSociety) — лёгкая социальная динамика (inter-slot chat) поверх capacity-aware recsys.
- В абзаце про congestion важно подчеркнуть, что ReCon работает с soft congestion (равномерность распределения), а наша задача — с **hard capacity** (бинарное переполнение). Это явный gap в литературе и может стать одним из тезисов защиты.
- Для валидации опирать аргумент на двух авторитетных источниках: arXiv:2504.03274 (методологически) и arXiv:2601.17087 (эмпирически).

### Что НЕ выдумывать

- **Cornell course allocation** в открытом виде не нашёл; Wharton/HBS — частично закрытый research-on-request. Если нужно цитировать эту тему, то через arXiv:2412.05691 (Kornbluth-Kushnir, Undergraduate Course Allocation through Competitive Markets).
- **MiroFish как peer-reviewed paper** не существует на 30.04.2026 — это инженерный проект, статьи только в Medium/DEV. Цитировать как «open-source проект» с ссылками на репозитории.
- Для wuyuehit ссылки магистранта (`wuyuehit/Meetup-Recommendation-Dataset`) реальная — `wuyuehit/meetup_dataset`. Перепроверить, нужно ли обновить ссылку в тексте.

---

## Ключевые ссылки одним списком

- arXiv:2304.03442 — Park et al., Generative Agents (UIST '23)
- arXiv:2310.10108 — Agent4Rec (SIGIR '24)
- arXiv:2306.02552 — RecAgent / YuLan-Rec (TOIS '25)
- arXiv:2310.09233 — AgentCF (WWW '24)
- arXiv:2411.11581 — OASIS (ICLR '25)
- arXiv:2502.08691 — AgentSociety (2025)
- arXiv:2504.12722 — SimUSER (ACL Industry '25)
- arXiv:2509.10397 — RecoWorld (Meta, 2025)
- arXiv:2504.03274 — Larooij & Törnberg, Critical Review of Generative ABMs (2025)
- arXiv:2601.17087 — Seshadri et al., Lost in Simulation (2026)
- arXiv:2308.09516 — ReCon, OT for Job Recommendation Congestion (RecSys '23)
- arXiv:2405.16120 — BankFair (2024)
- arXiv:2402.14369 — Provably Fair Exposure Control (2024)
- arXiv:2412.04466 — User-item fairness tradeoffs (2024)
- arXiv:2509.06002 — Survey of Real-World Recommender Systems (2025)
- arXiv:2505.21229 — Course Allocation with Credits via Stable Matching (SAGT '25)
- arXiv:2303.16750 — Stelmakh et al., Reviewer Assignment Gold Standard (2023)
- DOI: 10.1080/01605682.2024.2310722 — Scheduling Conferences with Attendees' Preferences (JORS '24)
- DOI: 10.1080/02664763.2012.760239 — Hybrid Clustering for Parallel Conference Sessions (JAS '12)
- github.com/wuyuehit/meetup_dataset — Meetup dataset snapshot
- IEEE DataPort 10.21227/8dr0-p842 — Meetup Dataset (Yue Wu, 2022)
- github.com/aida-ugent/ReCon — ReCon code
- github.com/joonspk-research/generative_agents — Park 2023 code
- github.com/LehengTHU/Agent4Rec — Agent4Rec code
- github.com/RUC-GSAI/YuLan-Rec — RecAgent code
- github.com/camel-ai/oasis — OASIS code
- github.com/tsinghua-fib-lab/AgentSociety — AgentSociety code
- github.com/666ghj/MiroFish + github.com/nikmcfly/MiroFish-Offline — MiroFish (open-source / fork)
- Kaggle: kaggle.com/datasets/ddosad/datacamps-data-science-associate-certification — GoalZone Fitness
