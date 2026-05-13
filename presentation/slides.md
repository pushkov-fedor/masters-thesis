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

# Поддержка формирования программы конференции с учётом ограниченной вместимости залов

<div class="mt-12 text-sm opacity-90">

Магистерская выпускная квалификационная работа · Индустриальный трек

Пушков Фёдор Владимирович · Университет ИТМО · 2026

</div>

<!--
Тема представляемой работы — разработка интеллектуальной системы поддержки формирования программы конференции. Защита проходит на индустриальном треке: production-компонент рекомендаций в составе системы — Telegram-бот — развёрнут на конференции Heisenbug; представляемая часть — аналитический слой для организатора.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-4xl font-bold tracking-wide">ПРОБЛЕМА</div>

:: content ::

<div class="grid grid-cols-2 gap-10 mt-4">

<div>

<div class="text-lg font-semibold leading-snug">

Неравномерное распределение участников по залам

</div>

<div class="mt-3 text-sm leading-relaxed">

В параллельных сессиях спрос на доклады заранее неизвестен: часть залов может быть перегружена, часть — недозагружена.

</div>

<div class="mt-5 text-sm leading-relaxed">

**Последствия:**

— переполненные залы;

— ухудшение опыта участника;

— нерациональное использование площадки.

</div>

</div>

<div>

<div class="space-y-4">

<div>
<div class="flex justify-between items-baseline mb-1">
<span class="font-semibold text-sm">Зал A</span>
<span class="text-red-600 text-xs font-medium">120 / 80 — перегрузка</span>
</div>
<div class="relative h-5 bg-slate-100 rounded">
<div class="absolute inset-y-0 left-0 bg-red-400 rounded" style="width: 100%"></div>
<div class="absolute top-0 bottom-0 border-l-2 border-dashed border-slate-700" style="left: 66.7%"></div>
</div>
</div>

<div>
<div class="flex justify-between items-baseline mb-1">
<span class="font-semibold text-sm">Зал B</span>
<span class="text-slate-500 text-xs font-medium">30 / 100 — недозагрузка</span>
</div>
<div class="relative h-5 bg-slate-100 rounded">
<div class="absolute inset-y-0 left-0 bg-slate-400 rounded" style="width: 25%"></div>
<div class="absolute top-0 bottom-0 border-l-2 border-dashed border-slate-700" style="left: 83.3%"></div>
</div>
</div>

<div>
<div class="flex justify-between items-baseline mb-1">
<span class="font-semibold text-sm">Зал C</span>
<span class="text-emerald-700 text-xs font-medium">70 / 70 — норма</span>
</div>
<div class="relative h-5 bg-slate-100 rounded">
<div class="absolute inset-y-0 left-0 bg-emerald-400 rounded" style="width: 58.3%"></div>
<div class="absolute top-0 bottom-0 border-l-2 border-dashed border-slate-700" style="left: 58.3%"></div>
</div>
</div>

</div>

<div class="mt-3 text-xs opacity-60 italic text-right">

пунктир — вместимость зала

</div>

</div>

</div>

<div class="mt-6 px-4 py-3 rounded bg-slate-50">

<div class="text-sm">

**Класс задач:** управление распределением спроса между ограниченными ресурсами без жёсткого назначения.

</div>

<div class="text-xs opacity-70 mt-1">

Примеры: транспортные потоки · сетевой трафик · массовые мероприятия

</div>

</div>

<!--
Главная проблема конференций с параллельными сессиями — неравномерное распределение участников по залам. На этапе формирования программы спрос на отдельные доклады заранее неизвестен, и фактическое распределение оказывается неравномерным: одни залы оказываются переполненными, другие недозагруженными. Условная схема справа иллюстрирует ситуацию: зал A перегружен сверх вместимости, зал B заполнен меньше чем на треть, зал C загружен ровно по проекту. Последствия — переполненные залы, ухудшение опыта участника, нерациональное использование площадки.

Задача относится к более широкому классу: управление распределением спроса между ограниченными ресурсами без жёсткого назначения. К этому классу принадлежат управление транспортными потоками, распределение сетевого трафика, распределение посетителей массовых мероприятий.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">ОТ ПРОГНОЗА К СИМУЛЯЦИОННОЙ ОЦЕНКЕ</div>

:: content ::

<div class="grid grid-cols-12 gap-3 mt-3">

<div class="col-span-5">

<div class="text-xs uppercase tracking-widest opacity-60 mb-1">Прямой путь</div>

<div class="text-xl font-semibold mb-3">Прогноз посещаемости</div>

<div class="text-sm font-medium mb-1">Для него нужны:</div>

<div class="text-sm leading-relaxed">

— фактическая посещаемость;

— профили и интересы участников;

— реакция на рекомендации.

</div>

<div class="mt-4 px-3 py-2 rounded bg-amber-50 border border-amber-200">

<div class="text-xs uppercase tracking-wide opacity-70 mb-1">Heisenbug · production-проверка</div>

<div class="text-sm">

<span class="font-bold">25</span> активаций &nbsp;·&nbsp; <span class="font-bold">7</span> оценок докладов &nbsp;·&nbsp; <span class="font-bold">5</span> оценок программы

</div>

</div>

<div class="mt-3 text-sm font-semibold leading-snug">

Данных недостаточно для калибровки модели поведения.

</div>

</div>

<div class="col-span-2 flex items-center justify-center">

<div class="text-5xl opacity-30 leading-none">→</div>

</div>

<div class="col-span-5">

<div class="text-xs uppercase tracking-widest opacity-60 mb-1">Вместо этого</div>

<div class="text-xl font-semibold mb-3">Симулятор конференции</div>

<div class="text-sm font-medium mb-1">Симулятор позволяет:</div>

<div class="text-sm leading-relaxed">

— задать сценарии аудитории;

— менять вместимость залов и силу рекомендаций;

— прогонять политики в одинаковых условиях;

— сравнивать риск перегрузки и релевантность выбора.

</div>

</div>

</div>

<div class="mt-6 px-4 py-3 rounded bg-bluegreen-50 border border-bluegreen-200 text-sm">

**Цель.** Не предсказать фактическую посещаемость, а сравнить политики рекомендаций в контролируемых сценариях.

</div>

<!--
Естественный путь — построить прогноз посещаемости: заранее оценить, сколько участников придёт на каждый доклад, и оптимизировать программу под этот прогноз.

Но для такого подхода нужны данные: фактическая посещаемость, профили участников и наблюдения о том, как рекомендации влияют на выбор. В рассматриваемой постановке таких данных нет. Production-компонент работы — Telegram-бот на конференции Heisenbug — показал именно эту проблему: за одно событие удалось собрать только 25 активаций, 7 оценок докладов и 5 оценок программы. Этого недостаточно для калибровки модели поведения.

Поэтому работа не ставит задачу точного прогноза посещаемости. Вместо этого строится симулятор конференции — контролируемая среда, в которой можно задавать разные предположения о поведении аудитории и сравнивать политики рекомендаций в одинаковых условиях.

В работе политики намеренно выбраны базовые: без рекомендаций, по релевантности и с учётом вместимости. Они нужны как проверяемые ориентиры. Цель симулятора — не предсказать фактическую посещаемость, а показать, какая политика устойчивее по перегрузке залов и релевантности выбора при разных правдоподобных сценариях.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">АРХИТЕКТУРА СИМУЛЯЦИОННОГО ЭКСПЕРИМЕНТА</div>

:: content ::

<div class="grid grid-cols-12 gap-4 mt-3">

<div class="col-span-4 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">1 · Подготовка</div>

<div class="flex-1 flex flex-col items-center justify-center text-sm space-y-2">

<div class="w-full px-4 py-2 rounded bg-slate-50 border border-slate-200 text-center">Программа конференции</div>

<div class="opacity-40">↓</div>

<div class="w-full px-4 py-3 rounded bg-slate-100 border border-slate-300 text-center font-semibold leading-snug">100 синтетических персон<br/>под эту программу</div>

</div>

</div>

<div class="col-span-1 flex items-center justify-center">

<div class="text-4xl opacity-30 leading-none">→</div>

</div>

<div class="col-span-7">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">2 · Сценарный прогон</div>

<div class="flex flex-col items-center text-sm space-y-2">

<div class="w-full px-4 py-2 rounded bg-bluegreen-50 border border-bluegreen-200 text-center">Сценарий + политика рекомендаций</div>

<div class="opacity-40">↓</div>

<div class="w-full px-5 py-4 rounded-lg bg-bluegreen-100 border-2 border-bluegreen-500">

<div class="text-xs uppercase tracking-widest opacity-70 mb-3 text-center font-semibold">Модель поведения участника</div>

<div class="grid grid-cols-2 gap-3">

<div class="px-3 py-3 rounded bg-white border border-bluegreen-300 text-center">
<div class="font-bold text-base">Параметрическая</div>
<div class="text-xs opacity-70 mt-1">формула выбора</div>
</div>

<div class="px-3 py-3 rounded bg-white border border-bluegreen-300 text-center">
<div class="font-bold text-base">Агентская</div>
<div class="text-xs opacity-70 mt-1">LLM-агент</div>
</div>

</div>

</div>

<div class="opacity-40">↓</div>

<div class="w-full px-4 py-2 rounded bg-bluegreen-50 border border-bluegreen-200 text-center">Метрики и сравнение политик</div>

</div>

</div>

</div>

<div class="mt-5 px-4 py-3 rounded bg-amber-50 border-l-4 border-amber-400 text-sm">

**Один экспериментальный стенд, два независимых способа моделирования выбора.**

</div>

<!--
После перехода от прогноза к симуляционной оценке система строится как сценарный эксперимент. Сначала берётся программа конференции, и под неё генерируется синтетическая аудитория — набор персон, которые могли бы прийти на такую конференцию.

Затем запускается сценарный прогон. В нём задаются вместимость залов, сила влияния рекомендаций и социальный фактор. Политика рекомендаций выдаёт участникам рекомендации, после чего модель поведения участника выбирает доклад в каждом слоте.

Блок выбора реализован двумя независимыми способами: параметрическим, через явную формулу выбора, и агентским, через языковую модель. Это позволяет проверить, сохраняются ли выводы о политиках при разных моделях поведения.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">МОДУЛЬ 1. СИНТЕТИЧЕСКАЯ АУДИТОРИЯ</div>

:: content ::

<div class="mt-3 text-sm leading-relaxed">

Генерируется под программу конференции. Используется как контролируемый пул участников, а не как утверждение о реальной аудитории.

</div>

<div class="grid grid-cols-2 gap-10 mt-6">

<div>

<div class="text-xl font-semibold">100 синтетических персон</div>

<div class="text-xs uppercase tracking-widest opacity-60 mt-4 mb-2">Пример структуры персоны</div>

<div class="px-4 py-3 rounded border border-slate-300 bg-slate-50 text-sm leading-relaxed space-y-1">

<div><span class="font-semibold">роль:</span> iOS Senior Developer</div>

<div><span class="font-semibold">опыт:</span> senior</div>

<div><span class="font-semibold">интересы:</span> архитектура · performance · testing</div>

<div><span class="font-semibold">описание:</span> 5–7 предложений</div>

</div>

</div>

<div>

<div class="text-xl font-semibold">Проверки качества пула</div>

<div class="text-xs uppercase tracking-widest opacity-60 mt-4 mb-2">Условие использования в экспериментах</div>

<div class="grid grid-cols-2 gap-3 text-sm">

<div class="px-3 py-2 rounded border border-slate-200">
<div class="font-semibold">Непротиворечивость</div>
<div class="text-xs opacity-70 mt-1 leading-snug">персоны не содержат внутренних конфликтов</div>
</div>

<div class="px-3 py-2 rounded border border-slate-200">
<div class="font-semibold">Разнообразие</div>
<div class="text-xs opacity-70 mt-1 leading-snug">пул не состоит из одинаковых профилей</div>
</div>

<div class="px-3 py-2 rounded border border-slate-200">
<div class="font-semibold">Покрытие программы</div>
<div class="text-xs opacity-70 mt-1 leading-snug">нет докладов без потенциальной аудитории</div>
</div>

<div class="px-3 py-2 rounded border border-slate-200">
<div class="font-semibold">Правдоподобие</div>
<div class="text-xs opacity-70 mt-1 leading-snug">распределения по ролям и опыту выглядят реалистично</div>
</div>

</div>

</div>

</div>

<!--
Первый модуль — синтетическая аудитория. Она нужна не как утверждение о реальной аудитории конкретной конференции, а как контролируемый пул участников для сценарных экспериментов.

Каждая персона содержит роль, уровень опыта, тематические интересы и текстовое описание. Всего используется 100 синтетических персон, сгенерированных под программу конференции.

Чтобы пул можно было использовать в экспериментах, он проходит проверки качества: внутренняя непротиворечивость персон, разнообразие пула, покрытие программы и правдоподобие распределений по структурным полям.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">МОДУЛЬ 2. ПОЛИТИКИ РЕКОМЕНДАЦИЙ</div>

:: content ::

<div class="mt-3 text-sm leading-relaxed">

Политика определяет, какие доклады показываются участнику в каждом слоте. В работе сравниваются три намеренно базовые политики.

</div>

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-4 rounded border border-slate-200">

<div class="text-xs uppercase tracking-widest opacity-60 mb-1">1 · Контрольная</div>

<div class="text-lg font-semibold mb-2">Без рекомендаций</div>

<div class="text-sm opacity-80 leading-relaxed">

Рекомендации не показываются. Базовая линия: что происходит без рекомендаций.

</div>

</div>

<div class="p-4 rounded border border-slate-200">

<div class="text-xs uppercase tracking-widest opacity-60 mb-1">2 · По релевантности</div>

<div class="text-lg font-semibold mb-2">Топ релевантных</div>

<div class="text-sm opacity-80 leading-relaxed">

Показываются доклады, ближайшие к интересам участника. Стандартный рекомендательный baseline.

</div>

</div>

<div class="p-4 rounded bg-bluegreen-50 border-2 border-bluegreen-400">

<div class="text-xs uppercase tracking-widest opacity-60 mb-1">3 · С учётом вместимости</div>

<div class="text-lg font-semibold mb-2">Релевантность − штраф за загрузку</div>

<div class="text-sm opacity-80 leading-relaxed">

Релевантные доклады штрафуются при высокой загрузке зала. Минимальная модификация под задачу перегрузки.

</div>

</div>

</div>

<div class="mt-6 px-4 py-3 rounded bg-amber-50 border-l-4 border-amber-400 text-sm">

**Проверяется эффект рекомендаций и отдельный эффект учёта вместимости.**

</div>

<!--
Следующий модуль — политики рекомендаций. Они определяют, какие доклады будут показаны участнику в каждом слоте.

Политики в работе намеренно выбраны базовые. Цель состоит не в том, чтобы предложить самый сложный алгоритм рекомендаций, а в том, чтобы проверить саму возможность сценарной оценки таких алгоритмов. Поэтому нужны три интерпретируемые точки сравнения: отсутствие рекомендаций как контроль, рекомендации по релевантности как стандартный baseline и вариант с учётом вместимости как минимальная модификация под задачу перегрузки залов.

Такой набор позволяет отдельно проверить эффект самих рекомендаций и эффект добавления ограничения вместимости.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">МОДУЛЬ 3А. ПАРАМЕТРИЧЕСКАЯ МОДЕЛЬ ВЫБОРА</div>

:: content ::

<div class="mt-3 text-base leading-relaxed">

Для каждого доклада считается полезность; чем она выше, тем выше вероятность выбора.

</div>

<div class="mt-10 flex items-start justify-center gap-3 flex-wrap">

<div class="text-center">
<div class="px-4 py-3 rounded bg-bluegreen-100 border-2 border-bluegreen-500 font-semibold">Полезность доклада</div>
</div>

<div class="text-3xl font-light opacity-50 mt-2">=</div>

<div class="text-center">
<div class="px-3 py-2 rounded bg-slate-50 border border-slate-300 font-semibold">Релевантность</div>
<div class="text-xs opacity-70 mt-1">близость интересам</div>
</div>

<div class="text-2xl font-light opacity-50 mt-2">+</div>

<div class="text-center">
<div class="px-3 py-2 rounded bg-slate-50 border border-slate-300 font-semibold">Рекомендация</div>
<div class="text-xs opacity-70 mt-1">показан политикой</div>
</div>

<div class="text-2xl font-light opacity-50 mt-2">+</div>

<div class="text-center">
<div class="px-3 py-2 rounded bg-slate-50 border border-slate-300 font-semibold">Социальный фактор</div>
<div class="text-xs opacity-70 mt-1">уже выбран другими</div>
</div>

</div>

<div class="mt-12 text-center text-sm">

<div class="flex items-center justify-center gap-3">
<span><em>U</em>(t)</span>
<span class="opacity-50">→</span>
<span>softmax</span>
<span class="opacity-50">→</span>
<span><em>P</em>(t)</span>
<span class="opacity-50">→</span>
<span class="font-semibold">выбранный доклад</span>
</div>

<div class="mt-3 text-xs opacity-60">

$U(t) = w_{rel}\cdot\mathrm{rel} + w_{rec}\cdot\mathrm{rec} + w_{soc}\cdot\mathrm{soc}$

</div>

</div>

<div class="mt-10 text-xs opacity-70 text-center italic">

Веса каналов меняются в сценариях эксперимента.

</div>

<!--
Сами рекомендации не задают посещаемость напрямую. Поэтому нужен третий модуль — модель поведения участника. Она описывает, как участник выбирает доклад из доступных вариантов в каждом слоте.

В параметрическом симуляторе для каждого доклада считается полезность. Она складывается из трёх каналов: тематической релевантности доклада профилю участника, факта показа доклада в рекомендациях и социального фактора — сколько участников уже выбрали этот доклад.

Затем softmax превращает полезности в вероятности выбора. Веса этих каналов варьируются в сценариях эксперимента, чтобы проверить политики при разных предположениях о поведении аудитории.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">МОДУЛЬ 3Б. АГЕНТСКАЯ МОДЕЛЬ ВЫБОРА</div>

:: content ::

<div class="mt-3 text-base leading-relaxed">

Вторая реализация блока выбора. В параметрической модели решение задаётся формулой; здесь его принимает LLM-агент по промпту.

</div>

<div class="mt-6 text-xs uppercase tracking-widest opacity-60 mb-2 text-center">Вход агента в каждом слоте</div>

<div class="grid grid-cols-5 gap-2 text-sm">

<div class="p-3 rounded bg-slate-50 border border-slate-200 text-center">
<div class="font-semibold leading-snug">Профиль участника</div>
</div>

<div class="p-3 rounded bg-slate-50 border border-slate-200 text-center">
<div class="font-semibold leading-snug">История посещённых докладов</div>
</div>

<div class="p-3 rounded bg-slate-50 border border-slate-200 text-center">
<div class="font-semibold leading-snug">Доклады текущего слота</div>
</div>

<div class="p-3 rounded bg-slate-50 border border-slate-200 text-center">
<div class="font-semibold leading-snug">Рекомендация политики</div>
</div>

<div class="p-3 rounded bg-slate-50 border border-slate-200 text-center">
<div class="font-semibold leading-snug">Социальный сигнал</div>
<div class="text-xs opacity-70 mt-1 leading-snug">сколько участников уже выбрали каждый доклад</div>
</div>

</div>

<div class="text-center text-3xl opacity-40 mt-3">↓</div>

<div class="mt-2 mx-auto w-fit px-8 py-3 rounded-lg bg-bluegreen-100 border-2 border-bluegreen-500 text-center">
<div class="font-bold text-lg">LLM-агент</div>
</div>

<div class="text-center text-3xl opacity-40 mt-3">↓</div>

<div class="mt-2 mx-auto w-fit px-6 py-2 rounded bg-bluegreen-50 border border-bluegreen-300 text-center font-semibold">
Выбор доклада или пропуск
</div>

<div class="mt-8 text-sm text-center italic opacity-80 leading-relaxed">

Явной формулы выбора нет: модель сама взвешивает варианты по тем же сигналам, что используются в параметрической модели.

</div>

<!--
Второй вариант модели поведения работает иначе. Каждому участнику соответствует отдельный LLM-агент. В каждом слоте программы агенты принимают решения по очереди.

В промпт передаются профиль участника, история уже посещённых докладов и список докладов в текущем слоте. Дополнительно могут передаваться рекомендация политики и социальный сигнал — сколько участников в этом же слоте уже выбрали каждый из доступных докладов.

На выходе агент возвращает выбор доклада или пропуск слота. В отличие от параметрической модели, здесь нет явной формулы полезности: модель сама взвешивает варианты, но получает сопоставимый набор сигналов.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">ПЛАН ЭКСПЕРИМЕНТА</div>

:: content ::

<div class="mt-3 text-base leading-relaxed">

Политики сравниваются не в одной конфигурации, а в семействе сценариев.

</div>

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">Что варьируем</div>

<div class="text-sm space-y-1.5">
<div>вместимость залов</div>
<div>доверие рекомендациям</div>
<div>социальный фактор</div>
</div>

<div class="mt-auto pt-5">
<div class="text-5xl font-bold leading-none text-bluegreen-700">50</div>
<div class="text-xs opacity-70 mt-2 leading-snug">сценариев, покрывающих пространство трёх параметров</div>
</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col gap-5">

<div class="text-xs uppercase tracking-widest opacity-60">Как запускаем</div>

<div>
<div class="text-sm font-semibold mb-1">Параметрический</div>
<div class="flex items-baseline gap-2">
<div class="text-5xl font-bold leading-none text-bluegreen-700">450</div>
<div class="text-xs opacity-70">запусков</div>
</div>
<div class="text-xs opacity-60 mt-1">50 × 3 повтора × 3 политики</div>
</div>

<div>
<div class="text-sm font-semibold mb-1">LLM</div>
<div class="flex items-baseline gap-2">
<div class="text-5xl font-bold leading-none text-bluegreen-700">12</div>
<div class="text-xs opacity-70">сценариев из 50</div>
</div>
<div class="text-xs opacity-60 mt-1">отбор по уровням риска перегрузки</div>
</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">Как проверяем модель</div>

<div class="text-xs leading-relaxed space-y-2">
<div>избыток вместимости → перегрузки нет</div>
<div>меньше вместимость → риск растёт</div>
<div>рекомендации не влияют → политики совпадают</div>
<div>рекомендации сильно влияют → политики различаются</div>
</div>

</div>

</div>

<div class="mt-6 px-5 py-3 rounded-lg bg-amber-50 border-l-4 border-amber-400 text-center text-sm">

**Внутри одного сценария политики сравниваются на одной и той же аудитории.**

</div>

<!--
Сравнение политик проводится не в одной конфигурации, а в семействе сценариев. В эксперименте меняются три условия: вместимость залов, степень доверия рекомендациям и сила социального фактора. Вместо полной сетки используется 50 точек, равномерно покрывающих это пространство параметров.

Параметрический симулятор прогоняется на всех 50 точках, с тремя повторами и разными случайными зёрнами. LLM-симулятор используется для более дорогой перекрёстной проверки, поэтому запускается на 12 точках из 50. Эти точки отбираются по силе перегрузки, чтобы покрыть разные уровни риска: от безопасных сценариев до сценариев с выраженной перегрузкой.

Перед основными прогонами симулятор проверяется на граничных сценариях — ситуациях, где ожидаемое поведение известно заранее. Если вместимость избыточна, перегрузки быть не должно. Если вместимость уменьшается, риск перегрузки должен расти. Если рекомендации не влияют на выбор, политики должны совпадать. Если влияние рекомендаций высокое, политики должны различаться.

Эти проверки не доказывают реалистичность модели, но позволяют отсеять грубые ошибки в логике симуляции перед интерпретацией результатов.

Внутри одной точки все политики работают на одной и той же аудитории, поэтому различия между ними относятся именно к политикам, а не к случайному шуму.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">МЕТРИКИ ЭКСПЕРИМЕНТА</div>

:: content ::

<div class="mt-3 text-base leading-relaxed">

По каким двум показателям сравниваются политики.

</div>

<div class="grid grid-cols-2 gap-5 mt-6">

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">Главная метрика риска</div>

<div class="text-2xl font-semibold mb-4 leading-snug">Средняя пиковая перегрузка</div>

<div class="text-sm leading-relaxed">

В каждом слоте берётся самый переполненный зал и считается превышение вместимости. Затем значения усредняются по слотам.

</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">Контроль качества выбора</div>

<div class="text-2xl font-semibold mb-4 leading-snug">Релевантность выбора</div>

<div class="text-sm leading-relaxed">

Внутренняя cosine-близость профиля участника и фактически выбранного доклада по эмбеддингам.

</div>

</div>

</div>

<div class="mt-6 px-5 py-3 rounded-lg bg-amber-50 border-l-4 border-amber-400 text-center text-sm">

**Цель сравнения: снизить пиковую перегрузку, не теряя релевантность внутри принятой модели.**

</div>

<!--
Перед результатами фиксируются две метрики сравнения.

Главная метрика — средняя пиковая перегрузка по слотам. В каждом временном слоте берётся самый переполненный зал и считается, насколько он превысил вместимость. Затем эти значения усредняются по слотам. Поэтому значение 0 означает отсутствие перегрузки, а 0.5 — что худший зал слота в среднем был перегружен примерно на 50%.

Вторая метрика — релевантность выбора. Это внутренняя cosine-близость профиля участника и фактически выбранного доклада по эмбеддингам. Она нужна как контроль: проверяется, что снижение перегрузки не достигается ценой полной потери тематического соответствия внутри модели.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">СЦЕНАРНАЯ ОЦЕНКА ВЫЯВЛЯЕТ РЕЖИМЫ ПЕРЕГРУЗКИ</div>

:: content ::

<div class="mt-2 text-sm leading-relaxed opacity-80">

Сравнение политики по релевантности и политики с учётом вместимости на 50 сценариях.

</div>

<div class="grid grid-cols-2 gap-5 mt-4">

<div class="p-5 rounded-lg border border-slate-200">

<div class="text-xl font-semibold">Mobius</div>
<div class="text-xs opacity-60 mb-4">40 докладов · 3 зала</div>

<div class="grid grid-cols-10 gap-1 mb-3 mx-auto w-fit">
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
</div>

<div class="text-sm space-y-1 mb-3">
<div><span class="font-bold">10 / 50</span> &nbsp;сценариев — перегрузка ниже</div>
<div><span class="font-bold">0 / 50</span> &nbsp;сценариев — перегрузка выше</div>
</div>

<div class="px-3 py-2 rounded bg-bluegreen-50 border border-bluegreen-200 text-xs">
Среди сценариев с перегрузкой: <span class="font-semibold">10 / 11 — ниже при учёте вместимости</span>
</div>

</div>

<div class="p-5 rounded-lg border border-slate-200">

<div class="text-xl font-semibold">Demo Day</div>
<div class="text-xs opacity-60 mb-4">210 докладов · 7 залов</div>

<div class="grid grid-cols-10 gap-1 mb-3 mx-auto w-fit">
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
<div class="h-6 w-6 rounded-sm bg-slate-200"></div>
<div class="h-6 w-6 rounded-sm bg-emerald-500"></div>
</div>

<div class="text-sm space-y-1 mb-3">
<div><span class="font-bold">17 / 50</span> &nbsp;сценариев — перегрузка ниже</div>
<div><span class="font-bold">0 / 50</span> &nbsp;сценариев — перегрузка выше</div>
</div>

<div class="px-3 py-2 rounded bg-bluegreen-50 border border-bluegreen-200 text-xs">
Среди сценариев с перегрузкой: <span class="font-semibold">17 / 17 — ниже при учёте вместимости</span>
</div>

</div>

</div>

<div class="mt-4 px-5 py-2 rounded-lg border border-bluegreen-300 bg-bluegreen-50">
<div class="flex items-baseline gap-4 flex-wrap">
<div class="text-xs uppercase tracking-widest opacity-60">Максимальный выигрыш</div>
<div class="text-base"><span class="font-bold">0.516 → 0.365</span> &nbsp;&nbsp; Δ = −0.150 по средней пиковой перегрузке</div>
</div>
</div>

<div class="mt-3 px-4 py-3 rounded-lg bg-amber-50 border-l-4 border-amber-400 text-sm">

**В рискованных сценариях учёт вместимости снижает перегрузку; потери по внутренней релевантности — порядка 10⁻³.**

</div>

<!--
Центральный результат — стенд различает политики именно в сценариях, где возникает риск перегрузки.

На всех 50 сценариях политика с учётом вместимости не проигрывает политике по релевантности по метрике средней пиковой перегрузки. На Mobius она даёт строгий выигрыш в 10 сценариях из 50, на Demo Day — в 17 сценариях из 50.

Более показательно подмножество сценариев, где перегрузка действительно возникает. На Mobius таких точек 11, и в 10 из них политика с учётом вместимости снижает перегрузку. На Demo Day таких точек 17, и во всех 17 она снижает перегрузку.

Максимальный абсолютный выигрыш наблюдается на Demo Day: средняя пиковая перегрузка снижается с 0.516 до 0.365. При этом потери по внутренней метрике релевантности остаются порядка 10⁻³.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">ГРАНИЦЫ ИНТЕРПРЕТАЦИИ</div>

:: content ::

<div class="mt-2 text-sm leading-relaxed opacity-80">

Результат показывает сравнительную устойчивость политик, а не точную посещаемость реального события.

</div>

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xl font-semibold leading-snug mb-3">Не прогноз посещаемости</div>

<div class="text-sm leading-relaxed opacity-80">

Симулятор не предсказывает, сколько людей реально придёт на доклад. Он сравнивает политики при заданных сценариях поведения.

</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xl font-semibold leading-snug mb-3">Синтетическая аудитория</div>

<div class="text-sm leading-relaxed opacity-80">

Пул участников сгенерирован под программу конференции. Он нужен для контролируемого сравнения, а не как модель реальной аудитории.

</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xl font-semibold leading-snug mb-3">Внутренняя релевантность</div>

<div class="text-sm leading-relaxed opacity-80">

Релевантность считается по эмбеддингам профиля и доклада. Это контрольная метрика внутри модели, не пользовательская оценка качества.

</div>

</div>

</div>

<div class="mt-6 px-5 py-3 rounded-lg bg-amber-50 border-l-4 border-amber-400 text-sm">

**Дальше:** калибровка на фактических данных, другие форматы конференций, расширение проверки LLM-симулятора.

</div>

<!--
Важно правильно интерпретировать результат. Работа не утверждает, что симулятор предсказывает фактическую посещаемость конкретной конференции. Входная аудитория синтетическая, а релевантность выбора — внутренняя метрика на эмбеддингах.

Поэтому численные результаты нужно понимать как сравнительную сценарную оценку: при заданных предположениях о поведении аудитории стенд показывает, какие политики устойчивее по риску перегрузки и как меняется релевантность внутри модели.

Дальнейшее развитие — калибровка модели при появлении фактических данных, проверка на конференциях других форматов и расширение проверки LLM-симулятора.
-->

---
layout: top-title
color: bluegreen-light
---

:: title ::

<div class="text-3xl font-bold tracking-wide">ВЫВОДЫ</div>

:: content ::

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">1 · Постановка</div>

<div class="text-sm leading-relaxed">

Сценарная оценка программы конференции без данных о фактической посещаемости.

</div>

</div>

<div class="p-5 rounded-lg border border-slate-200 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">2 · Система</div>

<div class="text-sm leading-relaxed">

Симуляционный стенд: синтетическая аудитория, политики рекомендаций, две модели выбора.

</div>

</div>

<div class="p-5 rounded-lg bg-bluegreen-50 border-2 border-bluegreen-400 flex flex-col">

<div class="text-xs uppercase tracking-widest opacity-60 mb-3">3 · Результат</div>

<div class="text-sm leading-relaxed">

В сценариях с перегрузкой политика с учётом вместимости не хуже baseline и чаще снижает среднюю пиковую перегрузку.

</div>

</div>

</div>

<div class="mt-6 px-5 py-3 rounded-lg bg-amber-50 border-l-4 border-amber-400 text-sm">

**Результат работы — не прогноз посещаемости, а инструмент сравнения политик до мероприятия.**

</div>

<!--
В работе поставлена задача сценарной оценки программы конференции в условиях отсутствия данных о фактической посещаемости.

Разработан симуляционный стенд: синтетическая аудитория, политики рекомендаций и две модели выбора участника — параметрическая и агентская.

Численный результат показывает, что в сценариях с риском перегрузки политика с учётом вместимости не хуже политики по релевантности и чаще снижает среднюю пиковую перегрузку.

Итоговый смысл работы: это не прогноз посещаемости, а инструмент, который помогает организатору заранее сравнивать политики рекомендаций и видеть рискованные режимы программы.
-->

---
layout: end
color: bluegreen-light
---

# Спасибо за внимание

#### Готов ответить на вопросы

<div class="mt-8 text-sm opacity-80">

Пушков Фёдор Владимирович · Университет ИТМО · 2026

</div>
