# ReCon (RecSys 2023) — глубокий разбор и оценка применимости (2026-05-01)

## TL;DR

ReCon — это **не симулятор и не conference recsys**, а небольшая (6 страниц, RecSys '23 LBR-style) работа Mashayekhi et al. (Ghent University) по **job recommendation**. Технический вклад — добавить в обучение базовой коллаборативной модели (CNE) дополнительный лосс на основе entropy-regularised optimal transport (Sinkhorn), который штрафует неравномерное распределение пользователей по вакансиям. Реальные данные есть, но это **VDAB** (Flemish employment service, anonymous, под NDA — в репозитории нет) и **CareerBuilder Kaggle 2012** (публичный, в репозитории `Data/career_builder_*` лежит как `(user, item)` пары). Никакого conference dataset, никаких сессий, никакого симулятора отклика нет. Эксперименты статичные: train на 6 днях, test на 3 днях, метрики на top-k без agent-based динамики.

Главный вывод по применимости: **полный пивот на ReCon методологически невозможен и стратегически бессмыслен** — у автора уже есть симулятор с валидацией на Meetup, congestion game, hard-capacity Constrained MDP, 11 политик и 5 кросс-доменных прогонов. ReCon в этой картине — один из baseline'ов, причём из soft-congestion лагеря. Реалистичный сценарий: **минимальный пивот за 1 вечер** — добавить ReCon-style soft-congestion политику (Sinkhorn-OT поверх cosine-скоров) как 12-ю политику и в Главе 1 / Заключении честно сказать, что наша работа обобщает ReCon до hard-capacity случая в условиях отсутствия исторических данных. Этот сценарий ниже расписан в D.3 как «Sprint-add-baseline».

---

## A. Сама статья ReCon

Прочитан полностью PDF arXiv:2308.09516 (RecSys '23 short paper, 6 страниц, скачан в `experiments/data/external/recon_research_2026_05/recon_arxiv.pdf`, постранично распарсен в `recon_arxiv_p{1..6}.txt`). Также частично прочитана журнальная версия 2024 года (IEEE Access, doi:10.1109/ACCESS.2024.3390229) через приложенный к репозиторию `Journal_supplementary_material.pdf` (33 страницы, в основном дополнительные графики).

### A.1 Постановка задачи

Дан набор пользователей U (job seekers) и набор объектов I (вакансий). Базовая модель рекомендаций M обучается на исторических взаимодействиях (apply / no-apply) и выдаёт матрицу скоров P = [p_ui] ∈ (0,1) — вероятностей релевантности пары (u, i).

Цель — обучить модель так, чтобы её топ-k выдачи одновременно были релевантными (NDCG, Recall, Hit Rate) и **избегали congestion** (мера: negative entropy market shares — доля пользователей, у которых данный item попадает в top-k).

Многокритериальный лосс (формула 3 статьи):

```
O_ReCon = O_M + λ · O_C
```

где O_M — стандартная функция потерь рекомендательной модели (binary cross-entropy в их CNE), O_C — кост, отвечающий за congestion, λ — гиперпараметр от 1e-6 до 1e-1.

O_C построен через optimal transport между U и I с равномерными маргиналами w_u = 1/|U|, w_i = 1/|I|. Стоимость переноса c(p_ui) = −ln(p_ui), функция «не-переноса» s(p_ui) = −ln(1 − p_ui) (формула 6). Подставляя в OT-формулу 4:

```
O_C = − Σ_{u,i} [ f_ui · ln(p_ui) + (1 − f_ui) · ln(1 − p_ui) ]
```

— это, по сути, BCE между optimal transport plan F и матрицей скоров P. Транспортный план F находится в линейной программе с матрицей стоимости d_ui = c(p_ui) − s(p_ui) = ln((1 − p_ui) / p_ui).

### A.2 Алгоритм

Чтобы линейка была дифференцируемой по P, используется **Sinkhorn** (Cuturi 2013) — entropy-regularised OT:

```
min_F  Σ f_ui · (d_ui + ε · log f_ui)   s.t. F1 = w_u, F^T 1 = w_i
```

В коде (`source_code/ot_local/ot_pytorch_sinkhorn.py`, `ot_plugin.py`):
- M = log((1 − P) / P)
- a = sinkhorn(M, γ=10.0, maxiters=5..10) — итеративная row/column-нормализация
- sinkhorn_loss = `<a, M>` + Σ −log(1 − P)

Полный лосс модели:

```
total_loss = BCE(P, target) + λ · sinkhorn_loss
```

Sinkhorn-блок дифференцируем (implicit differentiation, реализация из Gould et al.), поэтому всё обучается end-to-end через стандартный PyTorch-Lightning.

Ключевое отличие от пост-процессинга (CAROT, FairRec): congestion-штраф интегрирован **в обучение**, не в реранкинг. По их аргументации это даёт преимущество в инкрементальном дообучении.

В журнальной версии (2024) добавлен `ot_method='batches'` — Sinkhorn только по уникальным user/item id из батча, что даёт scalability на large datasets (career_builder_large).

### A.3 Данные

**Это критическая часть.** Проверено в коде и в самом тексте:

1. **VDAB** (Flemish employment service). Anonymized, **под NDA**, в репозитории нет. Только VDAB сами могут предоставить. Размер (Table 1 статьи): 1693 job seekers, 2931 jobs, 9766 train + 1428 val + 2950 test interactions. Период — последние 10 дней 2018 года.

2. **CareerBuilder** (Kaggle 2012, https://www.kaggle.com/c/job-recommendation). Публичный. **В репозитории лежит** в `source_code/Data/career_builder_small/` (4 файла .rating, общий объём ≈12 МБ; формат: tab-separated `user_id\titem_id` для train/val, `user_id\titem_id\tlabel` для test) и `Data/career_builder_large.zip` (≈70 МБ зипованных, после распаковки ≈350 МБ test_ranking.rating). Размер small: 3876 seekers, 4337 jobs, 24316 train + 1071 val + 4557 test interactions. Опять же 10 дней. В журнальной версии добавлен и VDAB-L, но он по-прежнему недоступен публично.

**Форматы данных тривиальные**: только пары `(user, item)` без any fields beyond id. Никаких текстовых описаний, никакой capacity-информации, никакой темы/тэга. Авторы сами признают (раздел 3.2), что осознанно отказались от content-based features ради чистоты сравнения коллаборативки.

**Никакого conference dataset нет.** Ни в статье, ни в репо, ни в журнальной версии. ReCon существует строго в job-recommendation домене.

### A.4 Метрики

Three desirability + three congestion-related, все на top-k (k=1, 10, 100):

Desirability:
- NDCG@k
- Recall@k
- Hit Rate@k

Congestion-related (формулы из CAROT/Bied et al. 2021):
- **Congestion@k** = Σ MS(i) · log(MS(i)) / log(|I|), где MS(i) — market share, доля пользователей, у которых i в top-k. Нормирована в [−1, 0], optimum −1.
- **Coverage@k** = доля item, попавших хотя бы в один top-k.
- **Gini Index** на market shares.

Реализация — `ot_local/ot_evaluation.py`, OTEvaluation class. Метрики симметризованы: считаются и по item-side, и по user-side.

### A.5 Главные результаты

Из текста статьи и Figure 2 (главный результат):

- ReCon на VDAB и CareerBuilder улучшает Congestion / Coverage / Gini, **не существенно проседая** по Recall / NDCG / Hit Rate. Для некоторых λ — **Pareto-optimal** относительно baseline'ов.
- Baselines: CAROT (Bied et al. 2021, OT post-processing), FairRec (Patro et al. WWW 2020, greedy envy-free + producer MMS), и сама CNE без congestion-loss.
- Конкретных табличных чисел в основном тексте нет — все результаты в виде scatter plots «desirability vs −congestion» с разными λ и hyperparameters baseline'ов. Это методологически слабое место (нет точечных чисел и доверительных интервалов).
- Execution time (Figure 3): ReCon в 5–10 раз медленнее CNE из-за полного Sinkhorn-прохода каждую эпоху, но быстрее FairRec (для k=100 разница порядка). Журнальная версия с `ot_method='batches'` значительно ускоряет.

### A.6 Limitations (по их собственным словам)

В разделе 5 «Conclusion and Future Work» признаются:

1. **Scalability**: на больших датасетах Sinkhorn по всему U×I не выживает. Решение в журнальной версии — батчевый Sinkhorn (что само по себе ослабляет global congestion-guarantee).
2. **Только одна базовая модель** — CNE. Журнальная версия добавляет NN-модель, но всё ещё две.
3. **Нет тестирования с разными типами matching cost / similarity** функций (формула 6 — единственная пара).

Что они **не признают, но критически важно** (мой анализ):

1. **Нет hard capacity**. Congestion в их формулировке — soft (минимизировать энтропию распределения нагрузки). Никакого механизма «эта вакансия заполнена, нельзя больше рекомендовать» нет. Это естественно соответствует job market (где вакансия может закрыться в любой момент, и нет фиксированного дедлайна), но **не соответствует** конференциям с фиксированной capacity сессии и фиксированным расписанием.
2. **Нет агентов/симулятора отклика**. Тест-протокол статичный: даны исторические apply, считаем top-k и метрики. Никаких behavioural-моделей, никакой dynamics, никакого проверки «а что произойдёт после первой выдачи».
3. **Scatter plots вместо чисел.** Нет single-number сравнения «ReCon vs CAROT» с тестом значимости. Только Pareto-фронт на глаз.
4. **Datasets — recruitment**, и CareerBuilder Kaggle-2012 — сильно устарел и часто критикуется в литературе за нерепрезентативность.
5. **Нет cold-start анализа**. Все user/item имеют ≥ 4 interactions (фильтр).

---

## B. Репозиторий

Склонирован в `experiments/data/external/recon_research_2026_05/ReCon/`.

### B.1 Структура

```
ReCon/
  README.md                              ← короткий, 2 BibTeX и note про обновлённые baseline-числа
  RecSys2023_supplementary_material.pdf  ← 1 МБ supplementary к LBR
  Journal_supplementary_material.pdf     ← 4 МБ, 33 страницы, в основном графики
  source_code/
    requirements.txt                     ← pandas, dask, POT, sentence-transformers, pytorch-lightning, wandb
    Data/
      career_builder_small/              ← публичные данные, 4 файла .rating
      career_builder_large.zip           ← 70 МБ зипа CareerBuilder large
    data_handler/                        ← загрузка .rating, negative sampling
      config.py
      data_common.py
      data_utils.py                      ← CustomDataset, ng_sample
    recommendation_method/
      cne/                               ← Conditional Network Embedding (Kang et al. ICLR 2019)
        main.py                          ← CLI с --use_ot, --lambda_p, --sinkhorn_gamma и т.д.
        model.py                         ← IdentityModel(pl.LightningModule)
        config.yml
      common/
        common.py                        ← train loop
        custom_early_stopping.py
    ot_local/
      ot_plugin.py                       ← OTPlugin.get_sinkhorn_loss
      ot_pytorch_sinkhorn.py             ← differentiable Sinkhorn (Gould et al.)
      ot_exact.py                        ← exact OT через emd / fpbm для evaluation
      ot_evaluation.py                   ← Congestion / Coverage / Gini metrics
```

Лицензия — присутствует `LICENSE`, MIT-стиль (Ghent University). Зависимости — стандартный PyTorch-стэк, ничего экзотического. POT (Python Optimal Transport) подключён только в evaluation (exact EMD), а в training используется самописный Sinkhorn в `ot_pytorch_sinkhorn.py`.

### B.2 Данные внутри

**В репо есть только CareerBuilder.** VDAB — приватный, упомянут только в коде через `--dataset vdab_small/vdab_large`, но самих файлов нет.

CareerBuilder small — 3876 seekers × 4337 jobs (см. A.3). Формат — пара `(int, int)` через таб. Никаких признаков, никаких текстов.

### B.3 Адаптируемость к JUG/Mobius

Я внимательно прочитал `data_utils.py`, `model.py`, `main.py`. Чтобы запустить ReCon на JUG-данных, нужно:

1. Сконвертировать JUG-данные в формат `.rating` — пары `(participant_id, talk_id)`. У автора эти пары формально есть как «agent выбрал talk». **Это тривиально (1 час)**.
2. Перенумеровать id в `[0, N)` — этого требует реализация (Embedding слои с фиксированным n_user/n_item).
3. Запустить `python -m recommendation_method.cne.main --dataset jug_mobius --use_ot 1 --lambda_p 1e-3`.

Что **не получится** без серьёзной переделки:
- ReCon использует **исторические apply** для обучения — у JUG их нет (см. постановку: «исторических данных нет», глава 1 черновика). Если в качестве «истории» использовать LLM-агентные траектории, это становится self-fulfilling.
- ReCon оптимизирует soft-congestion в среднем по всему датасету, не по конкретной коллективной выдаче. Hard capacity per slot не выражается.
- ReCon ничего не знает про слоты времени, треки, тематические близости. Это plain U×I.

**Soft vs hard capacity в коде**: явного «hard» режима нет. Capacity — implicit через равномерные маргиналы Sinkhorn (w_i = 1/|I|, т.е. «каждый item должен получить равную долю пользователей»). Чтобы превратить в hard cap, надо либо подменить w_i на нормированный capacity-вектор (тогда итог Sinkhorn — soft hard-capacity, всё ещё average), либо строить per-slot матроидное ограничение поверх scores (это уже выходит из ReCon, ближе к нашему capacity-aware policy).

Вывод: репозиторий технически чист, но **семантически чужой**. Проще написать свою реализацию ReCon-style лосса (10–20 строк PyTorch, Sinkhorn у нас уже не нужен — для inference достаточно одного прохода, для использования как baseline-policy в evaluation ещё проще) и подключить как 12-ю политику.

---

## C. Развитие области 2023–2026

### C.1 Прямое продолжение от тех же авторов

#### C.1.1 ReCon Journal (Mashayekhi et al., IEEE Access 2024, doi:10.1109/ACCESS.2024.3390229)

Расширение RecSys-LBR. Главные отличия:
- **Scalability**: батчевый Sinkhorn на уникальных id из батча, что позволяет работать с CareerBuilder-Large (350 МБ test_ranking).
- **Вторая базовая модель**: NN (нейросетевой коллаборативный фильтр) поверх CNE.
- **Расширенные эксперименты**: 4 датасета (VDAB-S, VDAB-L, CareerBuilder-S, CareerBuilder-L) × 2 модели × 3 значения top-k = 24 фигур (см. journal_supp_pages.txt).
- Выводы те же. Ничего методологически нового.

#### C.1.2 FEIR (Li, Kang, Lijffijt, De Bie, ACM TIST 2024, arXiv:2311.04542)

Та же лаборатория Ghent (Bo Kang, Jefrey Lijffijt, Tijl De Bie — соавторы ReCon). **Прямой методологический критик и преемник ReCon**. Ключевые тезисы (из абстракта):

- ReCon оптимизирует **congestion** (negative entropy market shares), но это **не fairness**. FEIR вводит две более точные меры:
  - **Inferiority** — competitive disadvantage пользователя на его же рекомендациях (новая мера).
  - **Envy** — насколько пользователь предпочёл бы рекомендации другого (классика).
- Все три (utility, envy, inferiority) переведены в дифференцируемые формулы через probabilistic interpretation, и решается multi-objective пост-обработка любой recsys-модели.
- Эксперименты на synthetic + real (e-recruitment + online dating).
- **Прямое сравнение с ReCon**: «improves the trade-offs ... compared to ... the state-of-the-art method for the related problem of congestion alleviation in job recommendation» — то есть FEIR говорит «мы лучше ReCon на правильно поставленных метриках».

Это значит: **сама лаборатория Ghent** через год после ReCon фактически признала, что congestion — слишком грубая мера и предложила более тонкие (envy + inferiority). Можно использовать в защите как «рамка ReCon уже устарела по критериям самих авторов».

### C.2 Расширения / альтернативы за пределами Ghent

OpenAlex отдаёт всего **8 цитирующих работ** для ReCon RecSys '23 (cited_by_count = 8 на 2026-05-01, см. `/tmp/recon_citations.json`). Это вообще немного для recsys-статьи 2.5-летней давности. Перечислю все 8 с оценкой релевантности:

1. **A Challenge-based Survey of E-recruitment Recommendation Systems** (Mashayekhi et al., ACM Computing Surveys 2024, doi:10.1145/3659942) — обзорная работа той же группы, ReCon один из примеров. Полезна как обзорная литература по job recsys, но методологически не расширяет.

2. **A Comprehensive Survey of AI Techniques for Talent Analytics** (Qin et al., Proc. IEEE 2025, doi:10.1109/JPROC.2025.3572744) — крупный обзор HRM/talent analytics. ReCon упомянут в разделе про job-recommendation fairness. Не расширяет.

3. **Scalable Job Recommendation With Lower Congestion Using Optimal Transport** — это и есть journal extension (см. C.1.1).

4. **FEIR** (см. C.1.2).

5. **Improving the Diversity and Fairness in Job Recommendations Using the Stable Matching Algorithm** (Unecha, Moh, Moh, IFIP 2025, doi:10.1007/978-3-031-96228-8_31) — альтернативный подход через stable matching. Абстракт через OpenAlex недоступен (поле пустое). По названию — конкуренция за ту же нишу. Применимо к JUG: stable matching уже есть в нашей литературе как ITC-2007/2019.

6. **Learning Implicit Relations for Collaborative Filtering via Optimal Transport** (Wang et al., IJCNN 2025, doi:10.1109/IJCNN64981.2025.11228433) — OT в коллаборативке для извлечения неявных связей пользователь–объект. Не про congestion. Цитирование скорее формальное — «вот ещё одна работа, использующая OT в recsys».

7. **Dual-Branch Mutual Learning Framework for Human Resources Recommendation** (Ding, Yu, Liu, Int. J. Pattern Recognition 2025, doi:10.1142/S0218001425510255) — BiLSTM + soft attention для матчинга резюме/вакансий. К ReCon отношение слабое — упомянут как часть литературного обзора по job recsys.

8. **ReCareer: Hybrid GNNs for Post-Career-Break Job Recommendation** (Wang et al., LNCS 2026, doi:10.1007/978-981-95-7138-3_30) — GNN-подход, специальный случай (после перерыва в карьере). Не про congestion.

### C.3 Параллельные работы по теме (не цитируют ReCon, но семантически близки)

Расширение области через смежные ключевые слова даёт следующие работы 2024–2026:

- **Congestion and Penalization in Optimal Transport** (Boualem & Mendoza, arXiv:2410.07363, 2024). Math.OC / econ.TH. Заменяет equality constraints на weighted penalization, что естественно моделирует supply/demand mismatch. Применения — образование, здравоохранение Перу. ReCon не цитирует, но методологически совпадает по духу. Полезно процитировать как «рядом стоящая теоретическая работа».
- **EquiPy: Sequential Fairness using Optimal Transport in Python** (arXiv:2503.09866, 2025). Реализация sequential fairness через OT. Не recsys, но та же математическая ниша.
- **Fairness in Social Influence Maximization via Optimal Transport** (arXiv:2406.17736, 2024). OT для outreach diversity в social influence. Параллель к congestion.
- **FairRec / FairRec+** (Patro et al. WWW 2020, arXiv:2002.10764) — один из baseline'ов ReCon. Two-sided fairness, Maximin Share для producers, Envy-Free up to One Good для users. **Это уже у нас в литературе**.
- **CAROT** (Bied et al., FEAST workshop ECML-PKDD 2021, hal-03540316) — прямой предшественник ReCon, two-step OT post-processing. Главное отличие от ReCon — пост-процессинг, а не joint training. У ReCon в Related Work упомянут как закрытый метод-конкурент. Применимо к JUG аналогично.

### C.4 Семантически ближайшие по нашей задаче

Что в литературе 2024–2026 действительно близко именно к conference recsys / hard-capacity / collective ranking:

- **Stelmakh et al., Reviewer Assignment Gold Standard** (arXiv:2303.16750, 2023) — у нас в литературе. Capacity-constrained matching reviewers↔papers, есть real expert labels. Best benchmark для hard capacity.
- **Agent4Rec** (arXiv:2310.10108, SIGIR '24) — у нас в литературе. LLM-симулятор для recsys, без congestion.
- **OASIS** (arXiv:2411.11581, ICLR '25) — у нас в литературе. Соц-сетевой LLM-симулятор, без congestion.
- **Patro/Mehrotra/Stelmakh** — все в нашей defense-validation сводке.

Вывод по C: **поле congestion-aware / capacity-aware recsys осталось узким** (8 цитирований за 2.5 года), и развивается в основном внутри одной лаборатории Ghent (FEIR, Journal version). Это **хорошая новость для нашей работы**: ниша свободна для расширения в hard-capacity formulation.

### C.5 Важная корректировка по arXiv:2504.03274

В наших must-cite и в `reference_validation_defense.md` фигурирует «arXiv:2504.03274». Проверка показывает: это **«Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review of Generative Social Simulations»** (Larooij & Törnberg, 2025) — критический обзор generative ABM. **К ReCon и congestion-aware recsys никакого отношения не имеет.** Возможно, эта статья нужна для другого раздела (валидация LLM-симуляторов), но не для главы про congestion.

---

## D. Применимость к ВКР

### D.1 Что можно переиспользовать

1. **Метрики**:
   - **Congestion = Σ MS(i)·log(MS(i)) / log(|I|)** — мы можем добавить как четвёртую (или пятую) метрику к нашим текущим четырём. По нашей постановке (top-k выдача множеству участников) MS(i) = доля участников с item i в top-k — переносится напрямую. Это даёт **прямую сопоставимость с ReCon**, что полезно в Главе 4 для baseline-разговора.
   - **Coverage@k** и **Gini** — у нас, скорее всего, уже неявно считаются, проверить и явно вынести.
2. **Идею ReCon как политики**:
   - Базовый ranker выдаёт скоры P. Дополнительный Sinkhorn-проход с равномерными w_i превращает их в OT-perturbed скоры, которые можно ранжировать. Это implementable как 12-я политика **в нашей текущей среде** (input — релевантность от cosine/learned, output — adjusted scores → top-k). Не нужно ни VDAB, ни CareerBuilder, ни их CNE-обучения — только обучения нашего симулятора как прежде.
3. **Метрика evaluation Sinkhorn-imbalance**: `compute_exact_imbalance` через POT.emd — для оффлайн-бенчмарка.

### D.2 Что надо адаптировать / переписать с нуля

1. **Sinkhorn-policy для нашей среды**: 30–60 строк PyTorch / NumPy, использует уже существующий векторный input наших политик. Не reusable из их репо как есть (их код заточен под CNE-обучение, а не под inference-time policy).
2. **Hard capacity layer**: ReCon его не имеет; у нас уже есть. Не трогаем.
3. **Train-time лосс**: ReCon обучает CNE с congestion-loss. У нас обучается PPO в рамках Constrained MDP. **Их train-time подход не нужен**, у нас другой алгоритмический режим.
4. **Datasets**: их данные (VDAB, CareerBuilder) семантически чужие. Не переносим.

### D.3 Сценарии пивота

#### D.3.1 Sprint-add-baseline (1 вечер)

**Что делать**:
- Добавить файл `experiments/src/policies/recon_policy.py` ≈80 строк. Принимает на вход cosine-скоры `(N_agents, N_talks)`, делает Sinkhorn-нормализацию с маргиналами `(1/N_agents, 1/N_talks)` (или, более правдоподобно к нашей задаче, с `(K_agent / N_agents, capacity_i / Σ capacity)`), возвращает adjusted scores. Поверх — top-K с уже существующим hard-capacity layer.
- Прогнать на текущих 5 датасетах в нашей среде.
- В Главе 1 поменять формулировку «ReCon [Liu et al., 2023]» на корректную **«ReCon [Mashayekhi et al., 2023]»** — текущая цитата в `глава-1/черновик-главы-1.md:107` и `:193` ошибочна (Liu — это автор EBSN-работ, а ReCon — Mashayekhi/Kang/Lijffijt/De Bie из Ghent).
- В Главе 4 добавить ReCon-policy в таблицу политик и явно прокомментировать: «ReCon — soft-congestion подход; при включении в hard-capacity среду показывает X результат на наших 4 метриках».
- В Главе 1 добавить упоминание FEIR (TIST 2024) как современного критика congestion-меры от тех же авторов.
- В Заключении добавить тезис «мы обобщаем ReCon до hard-capacity случая в условиях отсутствия исторических данных взаимодействий».

**Выигрыш**:
- Закрывает потенциальный вопрос «почему вы не сравнивали с ReCon?» на защите.
- Даёт конкретные числа для сравнения, что престижнее scatter-plot из самой статьи ReCon.
- Усиливает позиционирование «расширение ReCon», что лучше, чем «параллельная работа».
- **Не ломает ничего** в существующих результатах.

**Потери**: ≈4 часа разработки + ≈30 минут перепрогона + ≈1 час правки текста.

#### D.3.2 Soft-mode (2–3 дня)

**Что делать**: добавить в наш симулятор переключатель «hard / soft capacity» (где soft = энтропийный штраф вместо жёсткого ограничения). Прогнать все 11+1 политик в обоих режимах. В Главе 4 — таблица с двумя режимами. В Главе 1 — позиция «ReCon работает в soft-режиме, наша работа в hard-режиме, мы показываем оба».

**Выигрыш**: methodologically honest comparison, защищает от вопроса «а действительно ли hard-capacity лучше для конференций или это артефакт постановки?». Это сильное методологическое заявление.

**Потери**: 1 день кода + 1 день экспериментов + 0.5–1 день переписывания. Итого 2–3 рабочих дня. **На критическом пути до 13.05** это рискованно — теряется буфер на правки текста после обратной связи научрука.

#### D.3.3 Полный пивот (неделя+)

**Что делать**: бросить наш симулятор, перейти на CareerBuilder-данные, запустить наши политики поверх ReCon-обвязки, переписать всё с нуля. **Невыполнимо до 13.05.** Кроме того, теряются:
- собственный симулятор с валидацией (главный технический козырь).
- кросс-доменные эксперименты (5 прогонов).
- LLM-агентный слой.
- congestion game / Constrained MDP формулировка — у ReCon её нет.

Это 3 месяца работы выкидывается ради baseline-набора, который уже есть в открытом виде.

### D.4 Рекомендация

**Делать сценарий D.3.1 Sprint-add-baseline сегодня вечером** (и до 04.05 включительно). Оставить D.3.2 как backup, если после 06.05 будет окно. D.3.3 — отвергнуть полностью.

---

## E. Критика ReCon

### E.1 Из самих Limitations

1. Только одна / две базовые модели — нет универсальности заявленной в Section 2.
2. Нет доверительных интервалов, только scatter plots.
3. Нет анализа cold-start.
4. Soft congestion как замена hard capacity — методологически неточно для job market (где вакансия закрывается одним наймом) и тем более для конференций.

### E.2 Из реакций области

1. **FEIR (TIST 2024) — те же авторы признают, что congestion недостаточен**, и заменяют на envy + inferiority. Это сильнейший аргумент: «методологический преемник от той же лаборатории через год».
2. **Узкое цитирование (8 за 2.5 года)** — поле не подхватило ReCon как основной фреймворк. Это говорит об ограниченной generalizability вне job recsys.
3. **Отсутствие hard-capacity formulation** — ReCon фундаментально работает только в soft-режиме. Никто не показал расширение на hard.
4. **Нет behaviour-моделирования / симулятора** — все эксперименты статические, не показано, что will-happen-after-recommendation. Современный recsys (Agent4Rec, SimUSER) идёт в сторону динамических симуляторов.
5. **CareerBuilder Kaggle-2012 устарел** и не репрезентативен для современного job market.

### E.3 Что мы можем сказать в защиту нашего вклада

Конкретная риторическая канва:

> Работа Mashayekhi et al. (RecSys 2023, IEEE Access 2024) формулирует задачу congestion-aware recommendation как мягкое ограничение на равномерность распределения нагрузки в среднем по выдачам и решается через дополнительный Sinkhorn-loss в обучении базовой коллаборативной модели. Та же лаборатория годом позже (Li et al., FEIR, ACM TIST 2024) фактически признала недостаточность congestion как меры fairness и предложила более тонкие envy + inferiority. Настоящая работа отличается тремя осями. Во-первых, мы рассматриваем hard-capacity случай: ограничения должны выполняться в каждой одноразовой коллективной выдаче, а не в среднем. Во-вторых, мы работаем без исторических данных, на которые опирается обучение ReCon, что закрывает кейсы новых конференций (Mobius/Heisenbug). В-третьих, мы валидируем подход через гибридный симулятор отклика участника с внешней проверкой на реальных Meetup-RSVPs (accuracy@1 = 0,778, JS = 0,063, ρ = +0,84) — этого слоя у ReCon нет. Таким образом, ReCon является нашим baseline'ом в soft-режиме (политика реализована и сравнивается в Главе 4), а наш вклад — обобщение проблемы до hard-capacity / no-history / agent-validated режима.

Эта канва даёт: (а) корректное цитирование, (б) уважительное отношение к ReCon, (в) ясный научный gap, (г) честный методологический контекст с FEIR, (д) опору на наш B1-валидационный результат как главный козырь.

---

## Источники, использованные в этом отчёте

**Прочитано / проверено вручную**:
- arXiv:2308.09516 (PDF, 6 страниц, full text) — `experiments/data/external/recon_research_2026_05/recon_arxiv_p{1..6}.txt`.
- ReCon GitHub: `https://github.com/aida-ugent/ReCon` (склонирован, прочитаны `README.md`, `requirements.txt`, `cne/main.py`, `cne/model.py`, `ot_local/ot_plugin.py`, `ot_local/ot_pytorch_sinkhorn.py`, `ot_local/ot_evaluation.py`, `data_handler/data_utils.py`, `data_handler/config.py`).
- Journal extension supplementary: `Journal_supplementary_material.pdf` (33 страницы, скан содержания).
- Данные: `Data/career_builder_small/*.rating` (формат и размеры подтверждены).
- OpenAlex API: список цитирований ReCon (8 работ), абстракты ключевых: FEIR, Journal extension, Talent Analytics survey, IJCNN OT, DBML, ReCareer.

**Прочитано через WebFetch / WebSearch** (абстракты + summary):
- arXiv:2504.03274 (Larooij & Törnberg) — установлено, что **не относится** к ReCon-области.
- arXiv:2311.04542 (FEIR landing page).
- arXiv:2410.07363 (Congestion and Penalization in OT) — теоретическая параллель.
- CAROT (Bied et al. FEAST 2021) — предшественник.
- FairRec (Patro et al. WWW 2020) — baseline ReCon.

**Не прочитано**:
- VDAB dataset — недоступен, под NDA.
- OpenReview RecSys 2023 reviews — RecSys использует closed reviewing, отзывы не публичны.

**Отдельно проверено**: цитата «ReCon [Liu et al., 2023]» в `глава-1/черновик-главы-1.md:107` и `:193` — **ошибочна**, корректная атрибуция «Mashayekhi, Kang, Lijffijt, De Bie, RecSys 2023». Поправить при сценарии D.3.1.
