# Глубокий поиск предшественников для conference recsys с capacity (2026-05-01)

## TL;DR

После прочтения 11 PDF полностью + двух обзоров (Banerjee 2023 на 51 ссылку, Müllner 2025 на 44 ссылки) и серии целевых поисков по сектору 2017–2026 ключевой вывод следующий.

**Ровно нашей задачи (recsys для участников IT-конференции с параллельными залами и hard-capacity на каждый зал, в условиях отсутствия исторических данных, с одновременной коллективной выдачей ранжированных списков всем пользователям) в открытой литературе нет.** Самое близкое — два кластера работ:

1. **Christakopoulou et al. 2017 (CIKM)** + журнальная версия 2017 arXiv: ставят задачу POI/item recsys with capacity constraints (с понятиями user propensity и item capacity), разрабатывают Cap-PMF/Cap-BPR/Cap-GeoMF; используют MovieLens, Foursquare, Gowalla. Формулировка soft (capacity loss как мягкий штраф через логистический сурогат), статическая, на исторических данных, без коллективной согласованности выдач между пользователями. Это **прямой методологический предшественник** ReCon, упомянутый ReCon как «Recommendation with Capacity Constraints» — но ReCon его в Mashayekhi 2023 формально не цитирует, что само по себе пробел.
2. **Wu, Cao, Xu 2020 (FAST, ICSOC) и Wu et al. 2021 (TFROM, SIGIR)**: это уже про *individual-fairness в multi-round* recsys с capacity constraints. FAST вводит метрику Top-N Fairness и стратегию пост-обработки, гарантирующую сходимость дисперсии Top-N Fairness к нулю по раундам. Это самая близкая формулировка к нашей по постановке (capacity → выдача → fairness между пользователями), но домен — ресторан/сервис на Yelp, не конференция, и режим — *многораундовый* (накапливаемая справедливость), а у нас выдача *одноразовая* перед началом конференции.

Кроме них в новейшем сегменте (2024–2026) есть один сильный методологический сосед: **Liu, Gunawan, Wood, Lim 2025 (SCAIR/SCAIRv2, J. Big Data)** — game-theoretic crowd-aware itinerary recommendation для тематических парков с MDP и State Encoding. SCAIR явно формализует Selfish Routing в recsys-постановке, доказывает NP-hardness (редукцией от 0-1 Knapsack), что прямо применимо как академическое подкрепление нашей теоретико-игровой главы.

Конференц-домен напрямую: SARVE (Asabere et al. 2014, IEEE THMS) и предшественники CAMRS (Pham et al. 2012, ASONAM), Conference Navigator (Farzan & Brusilovsky 2008, HT), ROVETS (Asabere & Acakpovi 2019, IJDSST) **не учитывают capacity вообще** — ни soft, ни hard. SARVE сами признают как future limitation: «participants may be recommended good venues through both strong social ties and high similarity… they have to decide which one is more suitable as they cannot be in two venues at the same time». То есть конференц-recsys как поднаправление существует ~12 лет, но capacity-aware разворот в нём остаётся незанятым.

Окончательная рекомендация одной фразой для Главы 1 / Заключения: **«в литературе зафиксированы две независимые ветви — capacity-aware POI/item recsys (Christakopoulou 2017; ReCon 2023; FAST/TFROM 2020–2021) и conference attendee recsys без capacity (CAMRS 2012; Conference Navigator 2008; SARVE 2014; ROVETS 2019); их пересечение — recsys для участников конференций с hard-capacity параллельных залов в одноразовой коллективной выдаче — отсутствует, и именно это пересечение составляет содержательный пробел, замыкаемый настоящей работой»**.

---

## Категория 1: recsys для конференций / параллельных сессий с capacity

### 1.1 SARVE — Socially-Aware Recommendation of Venues and Environments (Asabere et al., IEEE THMS 2014)

- **Ссылка:** doi:10.1109/THMS.2014.2325837 (журнальная версия), arXiv:2008.06310 (препринт). PDF скачан в `experiments/data/external/conference_recsys_research_2026_05/sarve_arxiv.pdf` (12 страниц, прочитан полностью).
- **Domain:** smart academic conference (IT/CS), участники с research interests, презентационные сессии в нескольких залах.
- **Постановка:** «Recommend presentation session venues to conference participants based on Pearson correlation similarity of research interests, social tie strength (контакты по продолжительности и частоте), degree centrality presenters, и контекстных параметров (time, location)».
- **Capacity-aware?** **Нет.** Capacity вообще не модель; конкуренция за места не учитывается. Авторы в Discussion прямо называют это limitation: «participants are recommended good presentation session venues through both strong social ties and high similarities of research interest (tagged) ratings. In such scenarios they have to decide which one is more suitable as they cannot be in two venues at the same time». Это именно та проблема, которую решает наша работа.
- **Данные:** ICWL 2012 simulation. 60 presenters (по 5 контактов), 78 students/Dalian University of Technology, 2 зала (GE Hall, ME Hall). Это синтетика поверх реальной программы; никаких capacity ни в виде поля, ни в виде ограничения нет.
- **Главный результат:** Precision/Recall/F-measure на ICWL 2012 выше, чем у baseline'ов CAMRS (Pham 2012) и Conference Navigator (Farzan & Brusilovsky 2008): SARVE achieves precision 0.096 / recall 0.810 / F 0.172 при Pearson=1.0; B1=0.075/0.759/0.137; B2=0.045/0.698/0.086. На social ties=0.8: precision 0.013/0.0013/0.0011 (B1/B2). Общий вывод — социально-контекстный подход бьёт чисто content-based и социально-навигационный.
- **Применимость к ВКР:** **близко-но-другое (citation в Главу 1)**. Прямо в нашем домене (smart conference recsys), но без capacity. Используем как: (а) подтверждение, что conference recsys как поднаправление существует с 2012 года и активно публикуется в IEEE; (б) точка отстройки — мы добавляем capacity-aware оси, которые SARVE и его наследники игнорируют. В Главе 1, подраздел 1.3.3 или 1.4.1, упомянуть как «предшественник в conference attendee recsys, не учитывающий capacity-ограничений».

### 1.2 Pham et al. 2012 — CAMRS (Context-Aware Mobile Recommendation Services)

- **Ссылка:** Proc. IEEE/ACM ASONAM 2012, pp. 464–471. PDF не скачался напрямую; разобрана через abstract на ResearchGate и через цитирование в SARVE и Banerjee 2023.
- **Domain:** академические конференции (ICWL 2010, EC-TEL 2011), цифровые библиотеки DBLP и CiteSeerX как источник профилей.
- **Постановка:** «Recommend talks and presenters to conference participants based on context (whereabouts at venue, popularity, activity of talks) augmented with academic community context inferred from co-authorship and citation networks via link prediction».
- **Capacity-aware?** **Нет.**
- **Данные:** ICWL 2010, EC-TEL 2011 + DBLP + CiteSeerX.
- **Применимость к ВКР:** **близко-но-другое**. Самая ранняя задокументированная работа в категории «conference talk recommendation», цитируется во всех последующих работах как seminal. Полезна как академический предшественник линии «conference talk recsys» в Главе 1.

### 1.3 Farzan & Brusilovsky 2008 — Conference Navigator (HT '08)

- **Ссылка:** «Where Did the Researchers Go?: Supporting Social Navigation at a Large Academic Conference», Proc. 19th ACM Conference on Hypertext and Hypermedia, 2008.
- **Domain:** E-Learn 2007 conference, real attendees, real bookmarks.
- **Постановка:** «Help researchers attending a large academic conference plan talks they wish to attend through social navigation; encourage participants to add interesting talks to individual schedules and use scheduling information for social navigation support».
- **Capacity-aware?** **Нет.** Эта работа изначально описательная и про user feedback collection, а не про capacity allocation. Однако это **первый публично задокументированный production-уровневый conference attendee recsys** — Conference Navigator 3 продолжает использоваться (CN3 публикации 2010, 2014, 2019).
- **Данные:** E-Learn 2007 — несколько параллельных сессий, large number of papers, реальное использование участниками.
- **Применимость к ВКР:** **citation в Главу 1** как самый ранний production-recsys для конференций. Полезен в обосновании, что задача актуальна и реальная.

### 1.4 ROVETS — Recommendation of Venues and Environments Through Social-awareness (Asabere & Acakpovi 2019)

- **Ссылка:** International Journal of Decision Support System Technology (IJDSST), vol. 11, issue 3, 2019. doi: 10.4018/IJDSST.2019070104.
- **Domain:** smart conference, расширение SARVE.
- **Постановка:** Использует closeness centrality + BFS/DFS для поиска relevant presenters для target attendee, плюс tie strength + degree centrality. Это эволюция SARVE (которое использовало degree centrality) с более сложной графовой алгоритмикой.
- **Capacity-aware?** **Нет.** Та же line of work, что и SARVE; capacity не моделируется.
- **Применимость к ВКР:** **citation один раз** в Главе 1 как индикатор того, что conference recsys как тема продолжает развиваться (2019), но без capacity-aware разворота.

### 1.5 Hornick & Tamayo 2012 — RECONDITUS (IEEE TKDE)

- **Ссылка:** «Extending Recommender Systems for Disjoint User/Item Sets: The Conference Recommendation Problem», IEEE TKDE 24(8): 1478–1490.
- **Domain:** academic conference attendee recommendation на основе past conference session attendance.
- **Постановка:** «Recommend items from a new disjoint set to users; operate on observed user behavior such as past conference session attendance».
- **Capacity-aware?** **Нет.**
- **Применимость к ВКР:** **citation один раз** как ещё один представитель линии «conference recsys без capacity».

---

## Категория 2: EBSN/event recsys с capacity

### 2.1 Bikakis, Kalogeraki, Gunopulos 2019 — Attendance Maximization for Successful Social Event Planning (EDBT)

- **Ссылка:** Proc. 22nd Intl. Conf. on Extending Database Technology (EDBT 2019), arXiv:1811.11593. PDF скачан в `bikakis_attendance_2019.pdf` (12 страниц, прочитан полностью).
- **Domain:** EBSN (Event-based Social Networks: Meetup, Eventbrite, Plancast), фестивали и конференции.
- **Постановка:** Social Event Scheduling (SES). Дано: candidate events E, time intervals T, competing events C, users U. Цель: assign events to time intervals, чтобы максимизировать overall expected attendance. Это **organizer-oriented**, не participant-oriented.
- **Capacity:** Есть **resource budget θ** (organizer's resources, abstraction of staff/budget) и **location constraint** (no two events at same location and time). Это другой тип capacity — capacity ресурсов организатора, а не залов, и не на пользователей. Это и не наша задача.
- **Данные:** Meetup California (16K events, 42K users), Yahoo! Music user ratings (89K albums, 379K users).
- **Главный результат:** Доказана NP-hard аппроксимируемость для SES с любым фактором >1−ε. Три эвристических алгоритма (INC, HOR, HOR-I) — в 3–5 раз быстрее существующих. Algorithms используют incremental updating + interval-based assignment organization.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.3.1** (исследование операций) как пример смежной задачи с NP-hardness через 0-1 Knapsack. Полезен теоретический результат NP-hardness, ту же редукцию мы можем использовать для нашей задачи. **Не пересекаемся**, потому что задача **organizer-side**: assignment events→intervals, а у нас assignment user→event при fixed schedule. Это чёткая отстройка.

### 2.2 Christakopoulou, Kawale, Banerjee 2017 — Recommendation with Capacity Constraints (CIKM 2017)

- **Ссылка:** doi:10.1145/3132847.3133034, CIKM 2017. PDF скачан в `christakopoulou_capacity_2017.pdf` (10 страниц), а также расширенная arXiv-версия `cap_constraints_2017_arxiv.pdf` (20 страниц). Оба прочитаны полностью.
- **Domain:** общий item recsys + POI recsys. Иллюстративные примеры — book lib, route recsys, class recsys, viral content recsys.
- **Постановка:** «Every item j has maximum capacity c_j. Recommend top-N items to each user maximizing rating-prediction accuracy (PMF) или ranking accuracy (BPR), при этом expected usage Σ_i p_i σ(r_ij) ≤ c_j. p_i — user propensity (probability of following recommendations)».
- **Capacity:** **Soft.** Capacity loss формализуется как сурогатная (логистическая, экспоненциальная или hinge) функция от разности cj − E[usage(j)]; добавляется к prediction loss с trade-off α. Constraint никогда не enforced жёстко в каждой выдаче; только penalized в average по training.
- **Данные:** MovieLens 100K (943×1682), MovieLens 1M (6040×3706), Foursquare (2025×2759), Gowalla (7104×8707). Синтетические capacity (actual / binning / uniform / linear / reverse binning) и propensity (actual / median / linear). **Реальных capacity нет — все capacity сконструированы**.
- **Главный результат:** При α=0.2 Cap-PMF снижает Capacity Loss с 11.29 до 1.65 на MovieLens 100K, при росте RMSE с .38 до .71. Cap-GeoMF снижает Capacity Loss с 2.35 до .15 на Foursquare. Trade-off α эффективно регулирует приоритеты; в некоторых случаях Cap-варианты даже улучшают top-N AP.
- **Применимость к ВКР:** **близко-но-другое (must-cite в Главу 1, и в Главу 2 как методологический предшественник)**. Это работа, на которую методологически опирается ReCon (через CAROT 2021 → ReCon 2023, та же оптимально-транспортная линия). Christakopoulou 2017 — **первая работа, формализующая capacity-aware recsys в академическом recsys-сообществе**. От нашей задачи отличается: (а) soft, не hard; (б) static, не одноразовая коллективная выдача; (в) учится на исторических данных; (г) индивидуальная — пользователи независимы; (д) capacity искусственная. Это идеальная точка отстройки в Главе 1: «концепция user propensity и item capacity заимствована и обобщена, но формулировка как одноразовой коллективной выдачи с hard-capacity не имеет аналогов».

### 2.3 Bied et al. 2021 — CAROT: Congestion-Avoiding Job Recommendation with Optimal Transport (FEAST workshop, ECML-PKDD)

- **Ссылка:** hal-03540316. PDF скачан в `bied_carot_2021.pdf` (16 страниц, прочитан полностью).
- **Domain:** job market (Pôle emploi, France).
- **Постановка:** «Reciprocal recommendation in two-sided markets where item j is subject to capacity n_j: only top n_j users selecting it can be served. Coupling Optimal Transport with recommender systems».
- **Capacity:** Soft, через market shares и Sinkhorn-regularised OT.
- **Данные:** JOB dataset Pôle emploi (1.65M job seekers, 477K job ads, 43K matches Feb–Oct 2018), MAR matrimonial benchmark (2475 men+women, 50 clusters).
- **Главный результат:** Trade-off recall@k vs congestion@k. CAROT-XGB g=Id, ε=1: recall@10 = 21.99%, congestion@1 = -0.74. Базовый XGB: 31.40% и -0.62. Surprisingly, decreasing ε в Sinkhorn даёт лучший congestion за счёт более слабого recall — опровергает наивное ожидание.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.3.3 или 1.4.1** как методологический предшественник ReCon. CAROT 2021 → ReCon 2023 — это прямая линия. Не пересекаемся: домен job, не conference; soft, не hard.

### 2.4 Wu, Cao, Xu 2020 — FAST: A Fairness Assured Service Recommendation Strategy Considering Service Capacity Constraint (ICSOC 2020)

- **Ссылка:** doi:10.1007/978-3-030-65310-1_21, arXiv:2012.02292. PDF скачан в `fast_wu_2020.pdf` (17 страниц, прочитан полностью).
- **Domain:** service recsys (рестораны, фитнес, парикмахерские, медицинские услуги, отели). Иллюстрация на Yelp.
- **Постановка:** «Service has capacity constraint c_j. Multi-round recommendation; для каждого пользователя выдаётся top-N. Цель — обеспечить individual fairness across rounds: для каждой пары (user u, service j) Service Fairness Degree F^T_{u,j} = (p^T_{u,j} − p^T_j)/p^T_j → 0, где p^T_{u,j} — частота появления j в top-N выдачах пользователя u, p^T_j — overall market share; одновременно Top-N Fairness F^T_u = Σ_{j∈top-N} F^T_{u,j}, дисперсия среди пользователей D(F^T_u) → 0». **Это самая близкая по постановке к нашей работа, найденная в литературе.**
- **Capacity:** **Hard, но в multi-round формулировке.** В каждом раунде только c_j пользователей получают сервис в top-N (поэтому количество пользователей, у которых j в top-N, ≤ c_j); fairness копится по раундам.
- **Данные:** Yelp (Phoenix: 11252 users, 3774 businesses, 194188 reviews; Toronto: 8867 users, 3505 businesses, 119064 reviews) + 4 синтетических датасета (800 users, 50 services, разная capacity tightness).
- **Главный результат:** Доказаны три теоремы: (1) Σ_u F^T_u = 0 в каждом раунде; (2) D(F^T_u) → 0 при T → ∞; (3) система достигает Individual Level Fair Status (∀u, F^T_u = 0). F-FAST (fixed user set) и D-FAST (dynamic user set) — два варианта pseudo-greedy. Эксперименты: F-FAST теряет 7% recommendation quality vs ILP (точное решение через целочисленное программирование), но даёт на 20% выше quality, чем random, и приводит variance к 0 за 50 раундов; ILP не уравнивает справедливость — variance остаётся высоким. На LF dataset MMS=0 для k<10, MMS=1 для k∈[10,18] — характеристика дискретности capacity.
- **Применимость к ВКР:** **must-cite в Главу 1, подраздел 1.5 (анализ ограничений) и в Главу 2 (методология)**. Это самая близкая к нашей задача в литературе. **Опасности пересечения нет**: их формулировка multi-round (накопительная по T раундам, как Yelp restaurant suggestions for daily visits), наша — одноразовая коллективная выдача в начале конференции. Их fairness — динамическая, наша — мгновенная. Их capacity — capacity сервиса в смысле «сколько пользователей в день»; наша — capacity параллельного зала на 1 час. Их результат «D(F^T_u) → 0» аналитически интересен и его можно цитировать как «существующий многораундовый аналог наших одноразовых fairness-метрик». **Чёткая отстройка одной фразой**: «FAST решает близкую задачу в многораундовом режиме (Yelp-style daily restaurant suggestions); наша задача — одноразовая коллективная выдача программы конференции, в которой fairness не успевает накопиться по раундам и должна быть обеспечена в одном выпуске». Это самая важная находка во всём поиске.

### 2.5 Wu et al. 2021 — TFROM: Two-sided Fairness-Aware Recommendation Model (SIGIR 2021)

- **Ссылка:** arXiv:2104.09024, doi:10.1145/3404835.3462882.
- **Domain:** flight booking (Ctrip), Google Local, Amazon reviews.
- **Постановка:** «Joint optimization of customer fairness (loss in recommendation quality should be evenly distributed) and provider fairness (exposure should be evenly distributed). Post-processing heuristic algorithm с offline и online версиями».
- **Capacity-aware?** **Косвенно** через provider exposure constraints (упоминается в обзоре Müllner 2025 как «two-sided fairness под capacity constraints»).
- **Данные:** Ctrip flight (proprietary), Google Local, Amazon reviews.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.3.3** как продолжение линии two-sided fairness, заложенной FairRec (Patro 2020) → FAST (Wu 2020) → TFROM (Wu 2021). Не пересекаемся: домен flight, не conference; цель — exposure equality, не hard capacity.

### 2.6 Patro, Biswas, Ganguly, Gummadi, Chakraborty 2020 — FairRec: Two-Sided Fairness for Personalized Recommendations in Two-Sided Platforms (WWW 2020)

- **Ссылка:** arXiv:2002.10764, doi:10.1145/3366423.3380196. PDF скачан в `fairrec_patro_2020.pdf` (11 страниц, прочитан полностью).
- **Domain:** двусторонние платформы — Google Local, Last.FM (с расширением на Spotify, Amazon, Netflix).
- **Постановка:** «Recommend k items per customer maximizing customer utility и одновременно гарантировать каждому producer'у минимальную exposure E (≤ MMS = ⌊mk/n⌋). Mapping fair recommendation на constrained fair allocation of indivisible goods. Гарантирует EF1 (envy-free up to one good) для customers и MMS-fairness для большинства producers».
- **Capacity:** **Hard, но это producer-side capacity, а не consumer-side ограничение пропускной способности.** В терминах FairRec, ℓ = ⌊mk/n⌋ копий каждого продукта **создаются** для гарантии минимальной exposure всем продюсерам. Это противоположное направление: вместо «зал не вместит больше N людей», у них «продюсер должен получить минимум E экспозиций».
- **Данные:** Google Local (NYC, 11172 users, 855 producers, 25686 reviews, кастомный relevance score), Last.FM (1892 users, 17632 artists, 92834 plays).
- **Главный результат:** FairRec polynomial-time O(mnk) обеспечивает EF1 + non-zero exposure для всех producers + MMS для как минимум n−k producers. Эксперименты: при k=20 на Google Local-CUSTOM FairRec даёт WAP@10 = 0.153, top-k = 0.152 (т.е. сравнимо), exposure inequality (Gini-style) Z = 0.99 (top-k = 0.92). Mean envy Y = 0.001 (top-k = 0.0), стандартное отклонение customer utilities ≈ 0.1.
- **Применимость к ВКР:** **must-cite в Главу 1, подраздел 1.3.3** как канонический пример two-sided fairness через fair allocation framework. Полезна теоретическая часть (EF1, MMS — формализации, которые мы можем напрямую применить к conference setting). Не пересекаемся в постановке: их «capacity» — producer minimum exposure guarantee, наша — consumer-side hard capacity на зал. **Уже частично упомянуто в Главе 1 как Patro et al. 2020.**

---

## Категория 3: Multi-stakeholder fairness в recsys с congestion

### 3.1 Liu, Gunawan, Wood, Lim 2025 — SCAIR/SCAIRv2: Strategic and Crowd-Aware Itinerary Recommendation (Journal of Big Data)

- **Ссылка:** doi:10.1186/s40537-025-01249-9, J. Big Data 12:201 (Aug 2025). PDF скачан в `scair_itinerary_2025.pdf` (27 страниц, прочитан полностью).
- **Domain:** theme park itinerary recsys (5 Disney parks: Hollywood Studios 13 POIs, Epcot 17, California Adventure 25, Magic Kingdom 27, Disneyland).
- **Постановка:** «Itinerary recommendation как strategic game, social welfare optimization. System acts как central planner, рекомендуя пути каждому agent при прибытии, учитывая текущее состояние и пути уже допущенных agents. Цель — mitigate Selfish Routing (Roughgarden 2005) и Price of Anarchy». **Прямая теоретико-игровая постановка с congestion.**
- **Capacity:** **Hard через queuing-time penalty**. В каждом POI capacity Cap(f_y) ограничивает мгновенный поток; превышение приводит к росту queuing time, что снижает utility. Это soft в техническом смысле (capacity не binding constraint), но hard в эффекте (queuing time бьёт по utility так же, как hard capacity bottleneck).
- **Данные:** Theme park dataset Lim et al. 2017 (655K Flickr geo-tagged photos, **первый публичный датасет с queuing time distributions для Disney POIs**). Расширен авторами через Wikipedia, Google Maps, theme park websites; опубликован на github.com/junhua/SCAIR.
- **Главный результат:** Доказана NP-hardness pathfinding (Theorem 1, редукция от 0-1 Knapsack) и social welfare optimization (Theorem 2). SCAIRv2 — линейная сложность по числу POIs за счёт State Encoding (2D массив time-step × POI). В четырёх parks SCAIR снижает queuing time ratio в 4–6 раз (на DisHolly 0.045 → 0.003, на Epcot 0.076 → 0.016) при одновременном повышении utility.
- **Применимость к ВКР:** **must-cite в Главу 1, подразделы 1.3.2 (теория игр) и 1.4.1 (переранжирование)**. Это **прямой методологический сосед** — same Roughgarden 2005 framework, same NP-hardness через 0-1 Knapsack, same MDP-based crowd-awareness. Theme park ≠ conference, но абстракция одинаковая: пользователи прибывают, central planner раздаёт пути с учётом capacity. Полезно цитировать как: «структурно близкая задача в домене тематических парков с теоретико-игровой формулировкой». **Не пересекаемся фактически**: их режим online/sequential (агенты приходят по одному, recommender отдаёт каждому маршрут с учётом уже выданных рекомендаций), наш — batch/offline (все рекомендации выдаются одновременно перед началом конференции). Их utility — queuing time penalty, наша — admission probability + occupancy violation. Их domain — physical park, наш — конференция с дискретными timeslots.

### 3.2 Stelmakh, Shah, Singh 2021 — PeerReview4All (JMLR)

- **Ссылка:** Stelmakh, Shah, Singh, «PeerReview4All: Fair and Accurate Reviewer Assignment in Peer Review», JMLR 22(163), 2021.
- **Domain:** reviewer assignment в академических конференциях (NeurIPS, ICML, AAAI).
- **Постановка:** «Fair assignment of papers к reviewers; cost-flow algorithm для maximizing review quality of most disadvantaged paper».
- **Capacity-aware?** Да, **hard capacity**: каждому reviewer max load papers, каждому paper exact number of reviews. Constraint enforced жёстко через flow optimization.
- **Данные:** NeurIPS bid data, ICML.
- **Применимость к ВКР:** **уже в Главе 1 как Stelmakh 2023**, подтверждена релевантность. Это в правильной семье «academic conference assignment with hard capacity», но **task другая**: paper-to-reviewer (not user-to-event), и **режим detalierro определён**: assignment, не recommendation; решение бинарное и enforced (не вероятностный отклик). Полезна как «structurally closest hard-capacity assignment problem in academic conference domain».

### 3.3 Mehrotra et al. 2018 (Spotify) — Towards a Fair Marketplace (CIKM 2018)

- **Ссылка:** Mehrotra, McInerney, Bouchard, Lalmas, Diaz, «Towards a Fair Marketplace: Counterfactual Evaluation of the Trade-off between Relevance, Fairness & Satisfaction in Recommendation Systems», CIKM 2018.
- **Domain:** Spotify music streaming.
- **Постановка:** Two-sided fairness между artists (providers) и listeners (consumers); counterfactual evaluation trade-off relevance-fairness-satisfaction.
- **Capacity-aware?** Косвенно через exposure constraints для providers, не для слотов.
- **Применимость к ВКР:** **citation в Главу 1**, уже упомянуто, оставить.

### 3.4 Singh, Joachims 2018; 2019 — Fairness of Exposure in Rankings (KDD/SIGIR)

- **Ссылка:** Singh & Joachims «Fairness of Exposure in Rankings», KDD 2018 (arXiv:1802.07281); «Policy Learning for Fairness in Ranking», NeurIPS 2019.
- **Domain:** общий ranking framework.
- **Постановка:** Stochastic ranking policy с linear constraints на aggregated provider exposure: max π E[Σ π_i(u) r_ui] subject to Σ_u π_i(u) ∈ [c_i^min, c_i^max].
- **Capacity-aware?** Soft, как expected aggregate constraint, не per-list.
- **Применимость к ВКР:** **уже в Главе 1**, оставить.

### 3.5 Patro, Chakraborty, Banerjee, Ganguly 2020 — Towards Safety and Sustainability: Designing Local Recommendations for Post-Pandemic World (RecSys 2020)

- **Ссылка:** doi:10.1145/3383313.3412251.
- **Domain:** local recommendations (Yelp, Google Local) во время COVID.
- **Постановка:** Multi-objective: business sustainability (нижний bound на экспозицию) + safety (верхний bound на mean attendance ≤ social-distancing capacity) + utility. Sformulated как bipartite matching с polynomial-time solution.
- **Capacity-aware?** **Да, hard, soft enforced через bipartite matching constraint.** Именно ограничение «не превышать social-distancing capacity» = hard capacity на flow в каждый ресторан. Это **методологически очень близкая работа** к нашей задаче.
- **Данные:** Yelp, Google Local (NYC).
- **Применимость к ВКР:** **must-cite в Главу 1**. Это, наряду с FAST, наиболее структурно похожая работа: hard-capacity, multi-objective, polynomial-time через bipartite matching. Важная отстройка: их domain — рестораны (стационарные, не ограниченные параллельностью), их режим — online over time, не одновременная коллективная выдача. Их capacity мотивирована COVID-distancing, не залом конференции.

---

## Категория 4: Production-системы конференций

Из проверки Whova, Sched, Eventos, Dryfta, Skedda и других платформ:

- **Whova** имеет «Recommended» tab для networking (similar interests, affiliations, education), но это про **networking-пары людей**, не про сессии. Capacity-aware recsys для сессий нет.
- **Sched** позволяет ставить session caps и tracking RSVPs. Это **capacity tracking для administrators**, не recsys для участников.
- **Dryfta** документирует best practices для управления параллельными сессиями: 15-минутные buffer between sessions, room capacity calculator. Опять же — про event management, не про recsys.
- **Skedda, Vizitor, Centric Events** — про room booking и occupancy tracking; не recsys.

**Вывод по Категории 4:** В индустрии есть отдельные функции (recommendation для networking, capacity tracking для администрирования), но **интегрированной capacity-aware recommender для участников конференции не существует ни в одной коммерческой платформе**, доступной через open web. Никаких блог-постов от разработчиков с описанием recsys-алгоритмов с capacity-awareness не найдено. Это сильное эмпирическое подтверждение пробела — **индустрия его не закрыла**, что усиливает аргументацию для ВКР.

---

## Категория 5: 2024–2026 новейшее

### 5.1 Müllner, Schreuer, Kopeinik, Wieser, Kowald 2025 — Multistakeholder fairness in tourism: what can algorithms learn from tourism management? (Frontiers in Big Data, Sept 2025)

- **Ссылка:** doi:10.3389/fdata.2025.1632766. PDF скачан в `multistakeholder_tourism_2025.pdf` (10 страниц, прочитан полностью).
- **Domain:** tourism (POI, hotel, restaurant), но meta-обзор algorithmic+managerial.
- **Постановка:** Полусистематический обзор 44 публикаций (24 tourism management, 20 computer science) по multistakeholder fairness в туризме. Главный тезис: CS-сообщество фокусируется на quantifiable metrics (popularity bias, exposure, EF1, MMS), а tourism management — на качественных, контекстно-зависимых, мультиизмеримых концепциях fairness (включая environmental health, regional benefits, inclusive decision-making).
- **Главный результат:** Идентифицирован **research gap**: «recommender systems lack sufficient understanding of stakeholder needs, primarily considers fairness through descriptive factors such as measurable discrimination, while heavily relying on few mathematically formalized fairness criteria that fail to capture the multidimensional nature of fairness in tourism». Призывает к interdisciplinary collaboration. Цитируют Wu et al. 2020 (FAST) как «fairness under capacity constraints» в туризме.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.5** как актуальное (2025) подтверждение research gap в multistakeholder capacity-aware recsys. Особенно полезно для аргументации, что наша работа лежит в активном направлении 2024–2025.

### 5.2 Banerjee, Banik, Wörndl 2023 — A review on individual and multistakeholder fairness in tourism recommender systems (Frontiers in Big Data)

- **Ссылка:** doi:10.3389/fdata.2023.1168692. PDF скачан в `banerjee_tourism_review_2023.pdf` (17 страниц, прочитан полностью).
- **Domain:** обзор fairness в TRS.
- **Постановка:** Систематический обзор 66 публикаций (out of 51 cited inline) по fairness в tourism recsys. Категоризация stakeholder'ов (Consumer / Item Provider / Platform / Society) и fairness criteria (C-/I-/P-/S-Fairness).
- **Главный результат:** Подтверждает, что: (а) fairness research в tourism — emerging field с 2014; (б) большинство работ — на Yelp, Airbnb, TripAdvisor, Booking, Google Local; (в) S-Fairness (sustainability, society) — наименее исследованная категория, **только 6% работ покрывают её**; (г) fair conference recsys / fair trip planning — open research direction.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.5**. Это **отличный 2023 source** для motivation of capacity-aware multi-stakeholder recsys. Banerjee et al. 2025 (упомянутая в обзоре Müllner 2025) расширяют до CO2-aware travel recsys.

### 5.3 Banerjee et al. 2025 — Modeling Sustainable City Trips: Integrating CO2e Emissions, Popularity, and Seasonality

- **Ссылка:** Information Technology & Tourism 27, 189–226, 2025 (doi:10.1007/s40558-024-00303-1).
- **Domain:** европейские city trip recommendations.
- **Постановка:** Recommender, который объединяет CO2 emissions + popularity + seasonality с пользовательской ценностью. User study confirms users готовы trade-off utility for sustainability.
- **Capacity-aware?** Косвенно через popularity (overcrowding отдельно).
- **Применимость к ВКР:** **citation один раз** как пример новейшего multi-stakeholder city trip recsys с societal concerns.

### 5.4 Merinov & Ricci 2024 — Positive-sum impact of multistakeholder recommender systems for urban tourism promotion and user utility (RecSys 2024)

- **Ссылка:** doi:10.1145/3640457.3688173, Proc. 18th ACM RecSys 2024.
- **Domain:** urban tourism (Italian village).
- **Постановка:** Симуляция «limited tourist knowledge» + multi-stakeholder utility model; promote less-visited destinations при сохранении user satisfaction.
- **Применимость к ВКР:** **citation один раз** как аналог нашей задачи в туризме.

### 5.5 Khaili, Kofman, Cano, Mende, Hadrian 2024 — Multi-funnel Recommender System for Cold Item Boosting (CEUR 2024)

- **Ссылка:** CEUR-WS Vol. 3886, p. 11–22.
- **Domain:** travel platforms (cold-start novel listings).
- **Постановка:** Multi-funnel architecture для new listed items; диверсификация платформы и долгосрочная partner retention.
- **Применимость к ВКР:** **не цитировать**, не достаточно близко.

### 5.6 Two-sided Competing Matching Recommendation Markets with Quota and Complementary Preferences Constraints (arXiv:2301.10230, 2023)

- **Ссылка:** arXiv:2301.10230.
- **Domain:** общая теория двусторонних matching markets.
- **Постановка:** Bandit learning со стабильным двусторонним matching под quota constraints; алгоритм Multi-agent Multi-type Thompson Sampling (MMTS).
- **Capacity-aware?** Да, **hard через quotas**.
- **Применимость к ВКР:** **citation в Главу 1, подраздел 1.4.2** как пример learning-based hard-capacity matching. Их domain — college admissions / ride-sharing / dating. **Не пересекаемся**: их режим sequential bandit с feedback, наш — offline batch без feedback.

### 5.7 SimUSER (Bougie & Watanabe, ACL Industry 2025) — упомянут в `research-симуляторы-датасеты.md`

Не относится к conference recsys, но проверен — это про believable agent-based user simulation для recsys evaluation в общем. Не пересекается, но рядом.

---

## Дополнительно проверенные и не подходящие источники

- **POI Recommendation Pitfalls (arXiv:2507.13725, 2025)** — общий обзор POI recsys pitfalls, без специфики conference / hard-capacity, скачан, но не углубляюсь, не центрально.
- **Kokkodis & Lappas 2020** — popularity-difference bias на Yelp restaurants, P-Fairness.
- **Halder, Lim, Chan, Zhang 2022 — POI recommendation with queuing time and user interest awareness** (Data Min Knowl Discov) — TLR-M_UI multi-task transformer для next-POI recsys with queuing time prediction. **Близко-но-другое**: queuing time прогнозируется per-POI, не используется как capacity constraint в коллективной выдаче. Domain — Disneyland sequential POI recsys; пересечения с нашей задачей мало. Можно citation один раз в Главе 1 рядом со SCAIR, в подразделе 1.4.

---

## Ответы на главные вопросы

### Существует ли работа, которая делает в точности нашу задачу?

**Нет.** Самые близкие — FAST (Wu, Cao, Xu 2020, ICSOC) для multi-round individual fairness под capacity constraints в сервис-recsys на Yelp; SCAIR (Liu, Gunawan, Wood, Lim 2025, J. Big Data) для game-theoretic crowd-aware itinerary recsys в тематических парках; Patro et al. 2020 (RecSys) для post-pandemic local recommendations с capacity-aware bipartite matching на Yelp/Google Local. Ни одна из них:

- не работает в **conference attendee** домене (FAST — рестораны, SCAIR — парки, Patro — рестораны во время COVID);
- не делает **одноразовую коллективную выдачу** (FAST — multi-round; SCAIR — sequential agent-by-agent; Patro — over time);
- не имеет **жёстких capacity параллельных залов в каждой выдаче** (FAST — soft через market shares, SCAIR — soft через queuing time penalty, Patro — soft через bipartite matching budget);
- не работает в **отсутствии исторических данных** (все три работают на исторических Yelp/park/Google data).

Conference attendee recsys (SARVE, CAMRS, ROVETS, Conference Navigator, RECONDITUS) **полностью игнорируют capacity** — это direct gap, открытый ~12 лет. Capacity-aware recsys (Christakopoulou 2017, ReCon 2023, FAST 2020, TFROM 2021, FairRec 2020) **не идут в conference домен и не делают одноразовую коллективную выдачу**.

**Пересечение этих двух ветвей — тема настоящей работы.**

### Какие работы наиболее близки и должны быть в Главе 1?

См. финальный must-cite список ниже. Краткий список приоритетов:

1. **Christakopoulou et al. 2017 (CIKM)** — первая работа capacity-aware recsys, концепция user propensity и item capacity.
2. **Wu, Cao, Xu 2020 — FAST (ICSOC)** — самая близкая по постановке (capacity + individual fairness), хорошая точка отстройки.
3. **Liu et al. 2025 — SCAIR (J. Big Data)** — game-theoretic crowd-aware с NP-hardness, прямой методологический сосед.
4. **Patro et al. 2020 — Towards Safety and Sustainability (RecSys)** — multi-objective hard-capacity matching, post-COVID local recsys.
5. **SARVE (Asabere et al. 2014, IEEE THMS)** — единственный conference-domain recsys, который явно признаёт capacity-конфликт как future work.
6. **Bikakis et al. 2019 — Attendance Maximization (EDBT)** — NP-hardness через 0-1 Knapsack для smежной EBSN-задачи.
7. **Banerjee, Banik, Wörndl 2023 — TRS fairness review (Frontiers in Big Data)** — актуальный обзор multi-stakeholder fairness, подкрепляет research gap.

Уже в Главе 1: ReCon (Liu/Mashayekhi 2023), FairRec (Patro 2020), Mehrotra 2018, Stelmakh 2023, Singh & Joachims 2018, Park 2023, Agent4Rec, OASIS, JORS 2024 — **все актуальны, оставить**. Дополнить новыми пунктами 1–7 выше.

### Новые датасеты, не проверенные ранее?

- **SCAIR/Theme Park Dataset (Lim et al. 2017, расширенный github.com/junhua/SCAIR)** — 5 Disney parks (DisHolly 13 POIs / Epcot 17 / CalAdv 25 / MagicK 27 / DisneyLand). **Реальные queuing-time distributions**, capacity для каждого POI. Это **самый интересный новый датасет** для cross-domain валидации — 27 POIs (≈ как 27 docs одного трека Mobius), real capacity, real queuing. Применимость к ВКР: **отличный кандидат для дополнительной cross-domain валидации**, рядом с ITC-2007 и MovieLens. Объём данных управляем (655K Flickr photos + 27 POIs), формат — JSON и CSV, активно поддерживается (commits 2024–2025).

- **CareerBuilder Kaggle 2012** — уже разобрано в `research-recon-deep-2026-05.md`, не подходит (job, не conference).

- **VDAB** — ReCon-датасет, под NDA, недоступен.

- **JOB Pôle emploi** — CAROT, под NDA, недоступен.

- **Yelp (Phoenix, Toronto, NYC)** — в FAST. **Не подходит**: рестораны, не conference, multi-round не одноразовая выдача. Уже на сложности по сравнению с Meetup, не лучше.

- **MAR Matrimonial benchmark (Li, Ye, Zhou, Zha 2019)** — 2475 men/women, 50 clusters, 11 ordinal features. Используется в CAROT как public benchmark. **Не подходит**: матримониальный, capacity не зала.

- **Theme park dataset** (новый кандидат) — стоит проверить и подключить.

---

## Финальный must-cite список для Главы 1

1. **Christakopoulou, Kawale, Banerjee 2017 — Recommendation with Capacity Constraints (CIKM 2017)** — must-cite в подразделе 1.3.3 как первая работа, формализующая capacity-aware recsys через концепции user propensity и item capacity. Прямой методологический предшественник нашей работы. Отстраиваемся: они soft / static / individual / на исторических данных; мы hard / одноразовая коллективная выдача / в отсутствии данных.

2. **Wu, Cao, Xu 2020 — FAST: A Fairness Assured Service Recommendation Strategy Considering Service Capacity Constraint (ICSOC 2020)** — must-cite в подразделе 1.5 (анализ ограничений) как **самая близкая по постановке к нашей работе**. Отстраиваемся: их multi-round (накопительная справедливость по T раундам), наша — одноразовая коллективная выдача без накопления; их domain — рестораны/сервисы, наш — конференции с дискретными timeslots. **Эта работа — главная находка поиска и ключевая точка отстройки в Главе 1.**

3. **Liu, Gunawan, Wood, Lim 2025 — SCAIR/SCAIRv2: Strategic and Crowd-Aware Itinerary Recommendation (Journal of Big Data 12:201, 2025)** — must-cite в подразделах 1.3.2 (теория игр) и 1.4.1. Прямой методологический сосед: game-theoretic recsys с congestion, MDP, NP-hardness через 0-1 Knapsack. Отстраиваемся: their domain — physical theme park, online sequential agent admission; наш — конференция, batch одновременная выдача.

4. **Patro, Chakraborty, Banerjee, Ganguly 2020 — Towards Safety and Sustainability: Designing Local Recommendations for Post-Pandemic World (RecSys 2020)** — must-cite в подразделе 1.3.3 как пример hard-capacity multi-objective bipartite matching recsys. Близко методологически (polynomial-time через matching), но domain — рестораны во время COVID, не конференции.

5. **SARVE (Asabere, Xia, Wang, Rodrigues, Basso, Ma 2014) — Improving Smart Conference Participation through Socially-Aware Recommendation (IEEE Trans. Human-Machine Systems)** — must-cite в подразделе 1.3.3 как **единственная работа в conference attendee recsys домене, которая явно признаёт capacity-конфликт как future limitation**. Ключевая цитата: «participants… have to decide which one is more suitable as they cannot be in two venues at the same time». Это прямое подтверждение, что наша работа закрывает зафиксированный в литературе gap.

6. **Bikakis, Kalogeraki, Gunopulos 2019 — Attendance Maximization for Successful Social Event Planning (EDBT 2019)** — must-cite в подразделе 1.3.1 (исследование операций) как смежная NP-hard EBSN-задача с резкой формализацией NP-hardness через 0-1 Knapsack. Отстраиваемся: organizer-side (assign events to intervals), а у нас participant-side (assign users to fixed events).

7. **Banerjee, Banik, Wörndl 2023 — A Review on Individual and Multistakeholder Fairness in Tourism Recommender Systems (Frontiers in Big Data 6:1168692)** + **Müllner, Schreuer, Kopeinik, Wieser, Kowald 2025 — Multistakeholder Fairness in Tourism (Frontiers in Big Data, 2025)** — must-cite в подразделе 1.5 как актуальные (2023, 2025) обзоры, систематически фиксирующие research gap в multi-stakeholder capacity-aware recsys, подкрепляющие положение, что наша задача находится в активном открытом направлении.

**Итого 7 must-cite (плюс уже подключённые ReCon, FairRec, Mehrotra, Singh-Joachims, Stelmakh, Park, Agent4Rec, OASIS, JORS 2024 — все остаются).**

---

## "Опасные" работы (если есть)

**Опасных работ — две, обе в одной семье «multi-round individual fairness with capacity»:**

### Опасная работа 1: FAST (Wu, Cao, Xu 2020)

**Степень опасности:** Высокая. Самая близкая по постановке работа во всём поиске.

**Что у них совпадает с нами:**
- Capacity constraints на сервисы.
- Top-N выдача каждому пользователю.
- Individual fairness across пользователей.
- Полиномиальный жадный алгоритм, теоретически обоснованный (сходимость дисперсии).

**Чем мы отличаемся (must-articulate в Главе 1):**
- **Режим:** их multi-round (T раундов с накоплением справедливости), наш — одноразовая коллективная выдача.
- **Domain:** их сервис-recsys (Yelp restaurants), наш — конференц-recsys.
- **Capacity:** их service per-day (сколько пользователей за день), наш — зал per-timeslot (сколько в одном слоте параллельно).
- **Семантика fairness:** их Top-N Fairness накапливается по раундам (вариация дисперсии → 0); наша — мгновенная согласованность в одном выпуске.
- **Source данных:** они на Yelp historical reviews обучаются и эту-же популяцию на test'е используют; мы — без исторических данных, через симулятор.

**Рекомендованная фраза в Главе 1:** «Близкая по структуре работа Wu, Cao и Xu [2020] (FAST) формализует individual fairness в многораундовой выдаче с ограничениями вместимости сервисов и доказывает сходимость дисперсии справедливости пользователей к нулю с числом раундов. Постановка настоящей работы существенно отличается: вместо многократной выдачи с накоплением справедливости рассматривается одноразовая коллективная выдача без накопительной динамики, что исключает применимость многораундовых гарантий и требует обеспечения справедливости и согласованности в одном выпуске».

### Опасная работа 2: Christakopoulou, Kawale, Banerjee 2017

**Степень опасности:** Средняя. Прямой методологический предшественник, но мягкая formulation.

**Чем мы отличаемся:**
- Capacity у них soft (сурогатная функция penalty в loss), у нас hard в каждой выдаче.
- Они индивидуальные пользователи независимо, мы — коллективная выдача.
- Они на исторических данных (MovieLens, Foursquare, Gowalla) с искусственной capacity, мы — на программе конференции с реальными залами.

**Рекомендованная фраза в Главе 1:** «Концепции пользовательской склонности (user propensity) и вместимости элемента (item capacity), введённые Кристакопулу и соавторами [2017], в настоящей работе используются в обобщённой форме. В отличие от их формулировки с мягким штрафом за превышение вместимости в среднем по обучающей выборке, в работе ограничение вместимости выступает жёстким требованием к каждой одноразовой коллективной выдаче».

### Стелмах 2021/2023 НЕ опасен — task другая (paper-to-reviewer, не attendee-to-event), уже отстроено в существующем тексте Главы 1.

### SCAIR 2025 НЕ опасен — domain (theme park) и режим (online sequential) другие, и формализация через MDP с State Encoding конкретно про single-agent path, не collective list output.

**Итого: две опасные работы (FAST и Christakopoulou), обе требуют явной отстройки в Главе 1 одной фразой каждая.** Третьего «опасного» аналога нет.

---

## Окончательная рекомендация

**Что говорить в Главе 1 / Заключении / на защите про связь с литературой — одной фразой:**

> «В литературе зафиксированы две независимые ветви — capacity-aware recommender systems в общем item/POI-домене (Christakopoulou et al., CIKM 2017; ReCon, RecSys 2023; FAST, ICSOC 2020; FairRec, WWW 2020) и conference attendee recommender systems без учёта вместимости (CAMRS, ASONAM 2012; SARVE, IEEE THMS 2014; ROVETS, IJDSST 2019). Их пересечение — рекомендательная система для участников научно-практической IT-конференции с жёсткими ограничениями вместимости параллельных залов в одноразовой коллективной выдаче, формируемой в отсутствие исторических данных взаимодействий, — в открытой литературе на момент проведения исследования (май 2026) не обнаружено. Самая близкая по структуре работа FAST [Wu, Cao, Xu, ICSOC 2020] формализует индивидуальную справедливость в многораундовом режиме с накоплением; постановка настоящей работы — одноразовый коллективный режим — этим многораундовым гарантиям не сводится. Структурно близкая работа SCAIR [Liu et al., J. Big Data 2025] решает related game-theoretic задачу в домене тематических парков с последовательной выдачей; режим коллективной согласованной программы конференции в этой постановке не рассматривается».

Дополнительная фраза для Заключения (как указание на новизну):

> «Обнаруженный пробел — отсутствие capacity-aware рекомендательной системы для участников конференций с одноразовой коллективной выдачей и жёсткими ограничениями параллельных залов — закрывается совместной формализацией Constrained MDP с congestion-game интерпретацией и гибридным симулятором с двухслойной валидацией. Предложенный в работе подход не пересекается с известными работами и формирует новую точку в пересечении двух установившихся, но прежде не соединявшихся направлений рекомендательных систем».

---

## Скачанные PDF (13 шт., полностью прочитано 11)

```
experiments/data/external/conference_recsys_research_2026_05/
├── christakopoulou_capacity_2017.pdf      (1.3MB, 10p)  ✓ полностью прочитан
├── cap_constraints_2017_arxiv.pdf         (1.2MB, 20p)  ✓ полностью прочитан (расширенная версия)
├── sarve_arxiv.pdf                        (1.3MB, 12p)  ✓ полностью прочитан
├── socially_aware_venue_arxiv.pdf         (1.2MB, 8p)   ✓ полностью прочитан
├── fairrec_patro_2020.pdf                 (1.0MB, 11p)  ✓ полностью прочитан
├── bied_carot_2021.pdf                    (754kB, 16p)  ✓ полностью прочитан
├── bikakis_attendance_2019.pdf            (924kB, 12p)  ✓ полностью прочитан
├── fast_wu_2020.pdf                       (859kB, 17p)  ✓ полностью прочитан
├── scair_itinerary_2025.pdf               (2.0MB, 27p)  ✓ полностью прочитан (только первые 27 страниц)
├── multistakeholder_tourism_2025.pdf      (1.5MB, 10p)  ✓ полностью прочитан
├── banerjee_tourism_review_2023.pdf       (744kB, 17p)  ✓ полностью прочитан
├── poi_recsys_pitfalls_2025.pdf           (890kB)       — не прочитан, не центрально
└── two_sided_matching_2023.pdf            (1.6MB)       — не прочитан, не центрально
```

CAMRS (Pham et al. 2012) попытка скачать с ResearchGate была безуспешна; разобрана через abstract и через цитирование в SARVE/Banerjee 2023 (этого хватает на citation, не на углублённый разбор; работа 2012 года, не прорывная).
