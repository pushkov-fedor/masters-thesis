---
theme: neversink
title: Поддержка формирования программы конференции
info: |
  ВКР — магистратура ИТМО, индустриальный трек.
  Автор: Пушков Фёдор Владимирович.
author: Пушков Ф. В.
fonts:
  sans: 'Inter'
  serif: 'IBM Plex Serif'
  mono: 'JetBrains Mono'
  fallbacks: true
mdc: true
colorSchema: light
color: bluegreen-light
layout: cover
---

# Разработка интеллектуальной системы поддержки формирования программы конференции

#### Сценарный стресс-тест программы при отсутствии данных о посещаемости

<div class="mt-12 text-sm opacity-90">

Магистерская выпускная квалификационная работа · Индустриальный трек

**Пушков Фёдор Владимирович** · Университет ИТМО · 2026

</div>

<!--
Меня зовут Фёдор Пушков, тема работы — разработка интеллектуальной системы поддержки формирования программы конференции. Работа выполнена на индустриальном треке.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Проблема

:: content ::

При формировании программы конференции с параллельными сессиями организатор не знает, как фактическая аудитория распределится между залами.

<div class="grid grid-cols-3 gap-6 mt-8">

<div class="p-4 rounded-lg bg-slate-50 border border-slate-200">

**Однократность события**

Программа и аудитория уникальны; прогноз с одной конференции на другую не переносится.

</div>

<div class="p-4 rounded-lg bg-slate-50 border border-slate-200">

**Нет attendance-данных**

Систематический канал сбора фактической посещаемости отсутствует.

</div>

<div class="p-4 rounded-lg bg-slate-50 border border-slate-200">

**Поздний сигнал**

Часть выбора участником докладов происходит в день конференции.

</div>

</div>

<div class="mt-8 text-sm opacity-80">

Результат — перегрузка отдельных залов: ожидаемое число участников доклада превышает вместимость зала, в котором он проходит.

</div>

<!--
Три барьера: однократность, отсутствие данных, поздний выбор. Прогнозную модель построить не на чем.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Класс задачи: глубокая неопределённость

:: content ::

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

#### Не прогноз — сценарный анализ

В литературе по поддержке решений такой класс задач решается **сценарным подходом** (Robust Decision Making, DMDU).

Вместо точечного прогноза — выборка правдоподобных сценариев; для каждой стратегии оценивается её устойчивость на всём множестве.

<div class="mt-4 text-sm opacity-80">

**Каноны:** Lempert, Popper, Bankes (2003); Marchau и соавт. (2019); Kwakkel (2017).

</div>

</div>

<div class="p-6 rounded-xl bg-bluegreen-50 border-2 border-bluegreen-200 text-center self-center">

**Аналогия защиты**

<div class="mt-3 text-base">

Не прогноз погоды, а **стресс-тест здания**.

Не «какой будет ветер», а «при каком ветре что обвалится».

</div>

</div>

</div>

<!--
Главный методический сдвиг: с прогноза на сценарный анализ.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Что нового — пять осей gap-анализа

:: content ::

<div class="text-sm">

| # | Ось | Линии работ в литературе |
|---|---|---|
| 1 | Постановка без attendance-данных | Vangerven 2018, Rezaeinia 2024, Pylyavskyy 2024 — преференции считаются известными |
| 2 | Учёт вместимости как первичный критерий | ReCon 2023, FEIR 2024 — другие домены (вакансии, POI) |
| 3 | Совместное сравнение политик и вариантов программы | Раздельно: scheduling vs recsys |
| 4 | Два независимых механизма отклика | Park 2023, Agent4Rec, OASIS — только LLM |
| 5 | Сценарная робастность как критерий | DMDU-канон — другие домены |

</div>

<div class="mt-6 px-4 py-3 rounded-lg bg-bluegreen-50 border border-bluegreen-200 text-sm">

**Вклад работы:** система поддержки принятия решений на пересечении пяти осей; в открытой литературе работ, закрывающих их одновременно, не найдено.

</div>

<!--
В литературе по каждой оси отдельные работы. Совмещения пяти осей в одной задаче не нашёл.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Архитектура системы

:: content ::

```mermaid {scale: 0.7}
flowchart TB
  P[Программа<br/>конференции]:::data
  A[Синтетическая<br/>аудитория]:::data
  B[Модель поведения<br/>участника]:::model
  Pi[4 политики<br/>рекомендаций]:::model
  Phi[Оператор Φ<br/>локальные перестановки]:::model
  S1[Параметрический<br/>симулятор MNL]:::sim
  S2[LLM-агентский<br/>симулятор]:::sim
  LHS[LHS + CRN<br/>план эксперимента]:::exp
  M[Показатели<br/>загрузки и риска]:::exp
  R[Сравнительный<br/>отчёт]:::exp

  P --> B
  A --> B
  B --> S1
  B --> S2
  Pi --> S1
  Pi --> S2
  Phi --> S1
  Phi --> S2
  LHS --> S1
  LHS --> S2
  S1 --> M
  S2 --> M
  M --> R

  classDef data  fill:#eef2ff,stroke:#6366f1,stroke-width:1px
  classDef model fill:#fef3c7,stroke:#d97706,stroke-width:1px
  classDef sim   fill:#d1fae5,stroke:#059669,stroke-width:2px
  classDef exp   fill:#fce7f3,stroke:#db2777,stroke-width:1px
```

<div class="text-xs opacity-80 mt-2 text-center">

Девять модулей · единственный фактический вход — программа конференции · сердце системы — два независимых симулятора отклика
</div>

<!--
Сердце системы — два независимых симулятора отклика; работают через общий контракт политики.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Модель отклика участника

:: content ::

<div class="text-center my-2">

$$
U(t) = w_{rel}\cdot\mathrm{rel}(u,t) + w_{rec}\cdot\mathbf{1}\{t\in\mathrm{recs}\} + w_{gossip}\cdot\frac{\log(1+n_t)}{\log(1+N)}
$$

</div>

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

#### Распределение выбора

$$
P(t) = \mathrm{softmax}\bigl(U/\tau\bigr)
$$

Симплекс весов: $w_{rel} + w_{rec} + w_{gossip} = 1$.

Параметр стохастичности $\tau$ фиксирован при настройке.

</div>

<div class="p-4 rounded-lg bg-bluegreen-50 border border-bluegreen-200">

**Ключевое решение: capacity вынесена из utility в политику**

- В первой реализации capacity-штраф был в utility — это нарушало граничное свойство «при $w_{rec}\to 0$ политики неразличимы».
- В принятой версии capacity-учёт — **одна из политик** семейства (П3), а не свойство модели поведения.

</div>

</div>

<!--
Три канала, симплекс весов, capacity вынесена в политику — граничная верификация выполняется по построению.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

План эксперимента

:: content ::

<div class="grid grid-cols-2 gap-8 mt-2">

<div>

#### Латинский гиперкуб по 6 осям

1. вместимость залов
2. модель популярности
3. вес $w_{rec}$
4. вес $w_{gossip}$
5. размер аудитории
6. вариант программы (оператор $\Phi$)

Политика — отдельная ось, полный перебор внутри точки.

<div class="text-xs opacity-70 mt-3">

McKay, Beckman, Conover (1979); Kleijnen (2005).

</div>

</div>

<div>

#### Объёмы прогонов

<div class="p-4 rounded-lg bg-slate-50 border border-slate-200 mb-3">

**Параметрический симулятор**

50 точек × 4 политики × 3 seed = **486 evals**

</div>

<div class="p-4 rounded-lg bg-bluegreen-50 border border-bluegreen-200">

**LLM-агентский симулятор**

12 maximin × 4 политики × 1 seed = **48 evals**

44 160 LLM-вызовов

</div>

<div class="mt-3 text-sm">

**Общие случайные числа** внутри точки: все политики работают на одной синтетической аудитории.

</div>

</div>

</div>

<!--
LHS + CRN снимают шум попарного сравнения политик внутри одной точки гиперкуба.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Граничная верификация модели

:: content ::

<div class="grid grid-cols-3 gap-3 text-sm">

<div class="p-3 rounded bg-green-50 border border-green-200">

**EC1 ✓** При множителе вместимости $\geq 3.0$ риск перегрузки $= 0$ для всех политик

</div>

<div class="p-3 rounded bg-green-50 border border-green-200">

**EC2 ✓** Монотонность: при уменьшении вместимости риск не убывает

</div>

<div class="p-3 rounded bg-green-50 border border-green-200">

**EC3 ✓** При $w_{rec}=0$ протоколы прогонов разных политик пословно совпадают

</div>

<div class="p-3 rounded bg-green-50 border border-green-200">

**EC4 ✓** При $w_{rec}=1$ размах между политиками существенно превосходит шум

</div>

<div class="p-3 rounded bg-green-50 border border-green-200">

**+6 расширений ✓** Те же свойства при ненулевом $w_{gossip}$ + монотонность концентрации

</div>

<div class="p-3 rounded bg-bluegreen-100 border-2 border-bluegreen-400 font-semibold text-center flex items-center justify-center">

**ИТОГО:**<br/>**10 / 10 PASS**

</div>

</div>

<div class="mt-6 px-4 py-3 rounded-lg bg-slate-50 border border-slate-200 text-sm">

**Блокирующий фильтр** перед содержательными выводами [Sargent, 2013]. До прохождения четырёх обязательных свойств анализ результатов блокируется. На текущей реализации все свойства выполняются.

</div>

<!--
Блокирующий фильтр. До его прохождения никакие сравнительные выводы не интерпретируются.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Главные численные результаты

:: content ::

<div class="grid grid-cols-2 gap-6">

<div>

#### Попарное сравнение · 50 точек · $\varepsilon = 0.005$

<div class="text-sm">

| Пара | win | ties | loss |
|---|---:|---:|---:|
| no_policy vs cosine | **0.14** | 0.86 | 0.00 |
| no_policy vs capacity_aware | 0.00 | 0.86 | **0.14** |
| **cosine vs capacity_aware** | **0.00** | 0.78 | **0.22** |

</div>

<div class="mt-3 px-3 py-2 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm">

**cosine не выигрывает у capacity_aware** ни на одной из 50 точек — ни строго, ни за $\varepsilon$.

</div>

</div>

<div>

#### Risk-positive подмножество · 13 / 50

<div class="text-sm">

| Точка | вмест. | ауд. | cosine | cap_aware |
|---:|---:|---:|---:|---:|
| 26 | 0.629 | 60 | 0.171 | **0.004** |
| 49 | 0.774 | 100 | 0.545 | **0.356** |
| 35 | 0.963 | 100 | 0.061 | **0.003** |
| 18 | 1.040 | 60 | 0.021 | **0.000** |

</div>

<div class="mt-3 text-sm">

capacity_aware **не уступает** max(no_policy, cosine) на 13 / 13 точек в пределах $\varepsilon$ и **строго снижает риск** на **11 / 13 (85 %)**.

</div>

<div class="mt-2 text-xs opacity-70">

Средняя релевантность между политиками — различие $< 0.002$.

</div>

</div>

</div>

<!--
Центральный результат. cosine не превзойдён ни в одной точке за ε; на risk-positive строгий выигрыш в 11/13.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Сверка двух симуляторов

:: content ::

<div class="grid grid-cols-2 gap-6">

<div>

#### Согласованность на 12 общих точках

<div class="text-sm">

| Показатель | $n$ невырожд. | $\rho$ медиана | top-1 lead |
|---|---:|---:|---:|
| средняя релевантность | 12 | **0.80** | 11 / 12 |
| дисперсия загрузки | 12 | 0.40 | 11 / 12 |
| доля переполнений | 2 | 0.74 | 2 / 2 |
| превышение вместимости | 2 | 0.30 | 2 / 2 |

</div>

<div class="mt-3 text-sm">

Объединённая медиана $\rho = 0.554 \geq 0.5$ — формальный порог пройден.

</div>

</div>

<div>

#### Честная интерпретация

- **Релевантность** — уверенно согласована (12 / 12 точек).
- **Метрики переполнения** опираются на **2 / 12 невырожденных точек**: 74 % LHS-точек структурно безопасны → ранжирование вырождено.
- Это **диагностика**, не сильная валидация.

<div class="mt-3 px-3 py-2 rounded bg-amber-50 border border-amber-200 text-sm">

Канон validation для LLM-симуляторов = distribution-match, не индивидуальная точность. **«Believability ≠ validity»** [Larooij & Törnberg, 2025].

</div>

</div>

</div>

<!--
Симуляторы согласуются по релевантности. По метрикам переполнения — узкая выборка, структурное свойство сценарного анализа.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Ограничения и направления развития

:: content ::

<div class="grid grid-cols-2 gap-6 text-sm">

<div>

#### Ограничения как условия задачи

- Одна основная конференция (Mobius 2025 Autumn); перенос на другие требует проверки.
- Аудитория синтетическая; калибровка на real attendance вне обязательного результата.
- LLM-симулятор — 12 точек, экономичная модель `gpt-5.4-nano`.
- Оператор $\Phi$ — ось эксперимента, не оптимизатор расписания.
- Численные значения — сравнительные внутри модели, не прогноз.

</div>

<div>

#### Направления развития

- Целевой эксперимент по эффекту $\Phi$ при фиксированных остальных осях — переход от диагностики к причинной оценке.
- Расширение LLM-выборки с большей долей risk-positive точек.
- Прогон на программах конференций других форматов.
- Калибровка модели поведения при появлении канала фактических данных.
- A/B-проверка на следующем инстансе конференций сообщества JUG.

</div>

</div>

<!--
Ограничения зафиксированы как условия задачи, не как защитная оговорка.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

Выводы

:: content ::

<div class="grid grid-cols-2 gap-6">

<div>

#### Что разработано

- **Постановка** задачи сценарной оценки риска перегрузки залов при отсутствии attendance-данных.
- **Программная система** из 9 модулей с двумя независимыми симуляторами отклика.
- **План эксперимента** на латинском гиперкубе с общими случайными числами.
- **Сравнительный отчёт** для организатора — карта загрузки, горячие точки, попарные сравнения политик, сценарные характеристики оси $\Phi$.

</div>

<div>

#### Ключевые числа

<div class="p-3 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm mb-2">

**10 / 10** граничных тестов PASS

</div>

<div class="p-3 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm mb-2">

**cosine не выигрывает** у capacity_aware ни на одной из 50 точек за $\varepsilon$

</div>

<div class="p-3 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm mb-2">

На risk-positive: **11 / 13** точек строгого снижения риска

</div>

<div class="p-3 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm">

trade-off риск × релевантность — **7.3 %** комбинаций

</div>

</div>

</div>

<!--
Спасибо за внимание. Готов к вопросам.
-->

---
layout: end
color: bluegreen-light
---

# Спасибо

#### Готов к вопросам

<div class="mt-8 text-sm opacity-80">

Пушков Фёдор Владимирович · Университет ИТМО · 2026

</div>
