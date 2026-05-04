# Глубокий поиск датасета с реальной capacity (2026-05-01)

## TL;DR

Найден подходящий датасет — **ITC-2007 (International Timetabling Competition 2007)**, треки 1 (Examination) и 2 (Post Enrollment-based Course Timetabling). Это **реальные данные университетов** (в т. ч. Удине), в которых есть всё нужное и сразу, без искусственного навязывания вместимости:

- идентификаторы студентов (повторяющиеся, по 2–21 экзамена/события на студента в среднем),
- идентификаторы экзаменов/событий и их связи со студентами (фактическая регистрация),
- **реальные вместимости комнат** (от 11 до 1200 посадочных мест, разные у разных комнат),
- сетка тайм-слотов (в Track 1 — 16…80 периодов с конкретными датами и временами; экзамены идут параллельно в одном периоде).

В Track 1 capacity в нескольких экземплярах действительно тесная (`enrollments / room×period_capacity = 0,75…0,86`), что нужно для интересной CMDP-задачи. В Track 2 формат проще (бинарная матрица студент×событие + capacity-вектор), полезен как «синтетический-но-реальный» second dataset.

**Рекомендация:** использовать `exam_comp_set1.exam` (компактный, capacity-tight) как основной кейс в Главе 4 для cross-domain переноса; в дополнение прогнать `comp-2007-2-1.tim` как второй point. Оба уже скачаны в `experiments/data/external/deep_search_2026_05/itc_track1_solver/data/`. Текст в Главе 4 переписать как «cross-domain перенос на ITC-2007 — реальный набор расписаний с врождённой capacity-структурой; MovieLens 1M остаётся как sensitivity на стиль популяции».

---

## Что проверили (по каждому датасету)

### 1. ITC-2007 Track 1 — Examination Timetabling (.exam) — **ПОДХОДИТ**

- **Ссылка (зеркало с данными):** репозиторий solver-чемпиона `tomas-muller/cpsolver-itc2007` на GitHub, папка `data/exam/`. Оригинальный сайт `cs.qub.ac.uk/itc2007/` в момент проверки висел (timeout), поэтому брали зеркало.
- **Скачано:** да, через `git clone --depth 1` в `/Users/fedor/Study/masters-degree/experiments/data/external/deep_search_2026_05/itc_track1_solver/data/exam/` (1,7 МБ, 8 файлов: `exam_comp_set1.exam` … `exam_comp_set8.exam`).
- **Схема (по `head` + парсингу + Java-исходнику solver-а):**
  ```
  [Exams:N]
  duration, stud1_id, stud2_id, ...     # для каждого экзамена 1…N: длительность и список ID студентов, регистрировавшихся
  ...
  [Periods:M]
  date, time, duration, penalty         # для каждого периода: реальная дата + время + длительность + штраф
  ...
  [Rooms:K]
  capacity, penalty                     # для каждой комнаты — реальная вместимость
  ...
  [PeriodHardConstraints]              # ограничения «после/исключение/совпадение» — можно переинтерпретировать как congestion-game constraints
  [RoomHardConstraints]
  [InstitutionalWeightings]            # веса для штрафов
  ```
- **Размер (точно, посчитано Python-парсером — `parse_itc.py`):**

  | Экземпляр | Экзамены | Периоды | Комнаты | Студенты | Enrollments | Среднее экз/студ | Capacity (range) |
  |-----------|---------:|--------:|--------:|---------:|------------:|-----------------:|-----------------:|
  | exam_comp_set1 | 607 | 54 | 7 | 7 883 | 32 380 | 4,1 | 60 … 260 |
  | exam_comp_set2 | 870 | 40 | 49 | 12 484 | 37 379 | 3,0 | 14 … 424 |
  | exam_comp_set3 | 934 | 36 | 48 | 16 365 | 61 150 | 3,7 | 11 … 800 |
  | exam_comp_set4 | 273 | 21 | 1 | 4 421 | 21 740 | 4,9 | 1200 |
  | exam_comp_set5 | 1 018 | 42 | 3 | 8 719 | 34 196 | 3,9 | 500 … 999 |
  | exam_comp_set6 | 242 | 16 | 8 | 7 909 | 18 466 | 2,3 | 80 … 1000 |
  | exam_comp_set7 | 1 096 | 80 | 15 | 13 795 | 45 493 | 3,3 | 35 … 1000 |
  | exam_comp_set8 | 598 | 80 | 8 | 7 718 | 31 374 | 4,1 | 60 … 260 |

- **Capacity-поле:** **да**, секция `[Rooms:N]`, по одной capacity на каждую комнату; capacity варьируется от комнаты к комнате (в `set3`: 11..800 — реалистичные классы / лекционки / большие аудитории).
- **User_id с повторами:** **да**, каждый `student_id` встречается во многих списках экзаменов; среднее 2,3 … 4,9 экзаменов на студента.
- **Параллельные слоты восстановимы:** **да**, в каждом из M периодов одновременно может проходить несколько экзаменов в разных комнатах — именно та структура «зал × тайм-слот» как на JUG; при этом расписание сами решаем (в нашем случае — имитация работы рекомендательной системы, а capacity-binding измеряется как `Σ_e enrollments(e) / Σ_p Σ_r cap(r)`).
- **Применимость:** прямое отображение «студент = участник», «экзамен = доклад», «комната = зал», «период = тайм-слот». Есть факт регистрации (релевантность). Можно либо принять регистрацию как ground-truth релевантности и моделировать только capacity-binding (как в нашей задаче), либо параметризовать релевантность через признаки (есть длительность, дата, время — как минимум есть базис).
- **Capacity-tightness (важная характеристика):**
  - `set1`: total_demand = 32 380, room×period capacity = 43 308 → коэффициент 0,75 — capacity действительно бьёт.
  - `set6`: 18 466 / 32 800 = 0,56 — тоже binding.
  - `set4`: 21 740 / 25 200 = 0,86 — очень tight.
  - В `set2`/`set3`/`set7` много больших аудиторий — capacity почти не bind, для нашей задачи менее интересны.
- **Итог: подходит. Рекомендуется `exam_comp_set1.exam` как основной экземпляр (компактный, tight capacity, реалистичный размер).**

### 2. ITC-2007 Track 2 — Post Enrollment Course Timetabling (.tim) — **ПОДХОДИТ как вторая точка**

- **Ссылка:** тот же репозиторий, папка `data/tim/comp-2007-2-{1..16}.tim`.
- **Скачано:** да (8,4 МБ, 16 файлов).
- **Схема (по Java-парсеру `TimModel.java`):**
  ```
  N_events N_rooms N_features N_students      # одна строка с заголовком
  cap_room_1                                  # вместимости комнат, по одной на строку
  cap_room_2
  ...
  for each student in 0..N_students-1:
      for each event in 0..N_events-1:
          0 или 1                             # хочет ли студент s посетить event e (≈ RSVP)
  for each room: feature_flags                # потом матрицы room-features и event-features
  for each event: feature_flags
  for each event: 45 availability flags       # доступность по 45 слотам (5 дней × 9 слотов)
  for each event-pair: precedence flag
  ```
- **Размер (16 экземпляров):**

  | Экземпляр | События | Комнаты | Студенты | Enrollments | Mean cap |
  |-----------|--------:|--------:|---------:|------------:|---------:|
  | comp-2007-2-1 | 400 | 10 | 500 | 10 510 | 37,7 |
  | comp-2007-2-3 | 200 | 20 | 1 000 | 13 383 | 86,6 |
  | comp-2007-2-11 | 200 | 10 | 1 000 | 13 608 | 84,1 |
  | comp-2007-2-13 | 400 | 20 | 300 | 6 358 | 22,1 |
  | … (всего 16) | | | | | |

- **Capacity-поле:** **да**, по одной строке capacity на каждую комнату.
- **User_id с повторами:** **да**, явная бинарная матрица «студент s ↔ event e», 10..23 events на студента в среднем.
- **Параллельные слоты восстановимы:** **да**, фиксированная сетка `5 дней × 9 слотов = 45 timeslots`, в каждом параллельно идёт несколько событий.
- **Применимость:** даже более прямой формат для recsys — матрица user×item уже бинарно зашифрована как «выбрал/не выбрал».
- **Минус:** нет семантики (только числа без признаков курсов/комнат), сложнее интерпретировать. Поэтому как **второй**, контрольный, point.
- **Итог: подходит как дополнение к Track 1.**

### 3. ITC-2007 Track 3 — Curriculum-based (.ctt) — частично

- **Ссылка:** клонировано из `Docheinstein/itc2007-cct`, папка `datasets/comp01.ctt … comp21.ctt` (276 КБ).
- **Скачано:** да.
- **Схема:** курсы (с числом студентов на курсе, не списком), комнаты с capacity, кьюрриклумы (группы курсов, образующие траекторию), unavailability-constraints. Пример:
  ```
  COURSES: c0001 t000 6 4 130   # course_id teacher #lectures #min_days #students_per_course
  ROOMS:   B 200                # room capacity
  CURRICULA: q000  4 c0001 c0002 c0004 c0005
  ```
- **Capacity-поле:** **да**, секция `ROOMS`.
- **User_id с повторами:** **нет в явном виде** — есть число студентов на курс (агрегат), но не список конкретных студентов. Это менее удобно для recsys-как-recsys.
- **Применимость:** менее прямая. Можно из curricula построить «виртуальные траектории» (студент = curriculum, события = курсы из его curriculum), но это уже наполовину синтетика. Track 1 и Track 2 строго лучше.
- **Итог: не подходит для основной задачи (нет user_id с повторами).**

### 4. EBSN репозиторий `nvk681/EBSN` — **НЕ ПОДОШЁЛ (тупик)**

- **Ссылка:** https://github.com/nvk681/EBSN
- **Скачано:** да.
- **Содержимое:** папка `dataset/` содержит **только notebook `dataset_import.ipynb`**, который инструктирует пользователя самому скачать данные из закрытой Kaggle competition `event-recommendation-engine-challenge`. Никаких CSV/JSON в репо нет.
- **Capacity-поле:** не проверишь, потому что данных нет, но в этой Kaggle competition (схема: `user, event, invited, timestamp, interested, not_interested`) **capacity отсутствует** по описанию.
- **Итог: не подходит.**

### 5. Kaggle Event Recommendation Engine Challenge

- **Ссылка:** https://www.kaggle.com/c/event-recommendation-engine-challenge/data
- **Скачано:** нет, требуется аккаунт Kaggle (у пользователя его нет).
- **Схема (по описанию страницы и обзорам):** `train.csv (user, event, invited, timestamp, interested, not_interested)`, `events.csv (event_id, creator, start_time, lat/lon, bag-of-words)`, `users.csv`, `event_attendees.csv`, `user_friends.csv`. **Поля capacity нет** — это видно из всех публичных описаний схемы.
- **Итог: не подходит (нет capacity и нет логина).**

### 6. UniTime Purdue datasets

- **Ссылка:** https://www.unitime.org/uct_datasets.php
- **Скачано:** нет (есть как минимум 30+ архивов `pu-fal07-*.zip`, скачивание через WebFetch без явных URL не получилось; страница только описывает их).
- **По описанию:** Purdue C8-2007 содержит 9 категорий задач × 2 семестра, формат — расписания университета с capacity комнат и enrolled students per course. **Семантически — близкие сородичи ITC-2007 Track 1**, обычно более крупного масштаба (Purdue — большой кампус). Если нужно подкрепить N=2 ещё одним point — это правильный следующий шаг, но для Главы 4 ITC-2007 (Удине + 16 синт-реальных) и так достаточно.
- **Итог: не проверен глубоко (избыточно при наличии ITC-2007).**

### 7. RecSysDatasets (RUCAIBox)

- **Ссылка:** https://github.com/RUCAIBox/RecSysDatasets
- **Проверено:** да (через WebFetch).
- **Содержит:** Amazon, Yelp, Foursquare, MovieLens, LastFM, Douban (movies/books/music), KDD2010, EndoMondo. **Ни одного event-style датасета с capacity нет.**
- **Итог: не подходит (нет нужного домена).**

### 8. Healthcare appointment datasets (Kaggle, Mendeley)

- **Ссылка:** https://www.kaggle.com/datasets/joniarroba/noshowappointments и аналоги.
- **Скачано:** нет (аналог можно посмотреть, но ясно из описания).
- **Схема:** `PatientID, AppointmentID, ScheduledDay, AppointmentDay, Age, NoShow, …`.
- **Capacity-поле:** **нет** — нет ёмкости провайдера/слота в данных (есть только пациент↔приём).
- **User_id с повторами:** да (PatientID повторяется), но capacity не выгружено.
- **Итог: не подходит.**

### 9. Yelp / TripAdvisor

- **Ссылка:** https://business.yelp.com/data/resources/open-dataset/
- **Проверено по описанию схемы:** `business`, `review`, `checkin`, `tip`, `user`. Есть `business.attributes` с разными полями («WheelchairAccessible», «RestaurantsReservations»), но **поле seating capacity отсутствует**.
- **Итог: не подходит.**

### 10. Hotel-booking datasets, airline reservations

- **Проверено по описанию.** В hotel-booking-demand схема — `lead_time, arrival_date, adults, children, …` — нет user_id с повторами и нет capacity на комнату/гостиницу.
- **Итог: не подходит.**

### 11. Plancast / Last.fm events / Douban events

- **Проверено по описанию и поиску GitHub.** Plancast используется в академических работах (Liu et al., EBSN papers), но публично выгруженного снапшота с capacity не найдено. Last.fm 1K — это listening events, без венюшной capacity. Douban — movies/books/music; «Douban event» в публичном виде не нашлось.
- **Итог: не подходят.**

---

## Команды для воспроизведения

```bash
# 1. Подготовка
mkdir -p /Users/fedor/Study/masters-degree/experiments/data/external/deep_search_2026_05
cd /Users/fedor/Study/masters-degree/experiments/data/external/deep_search_2026_05

# 2. Скачивание ITC-2007 Track 1 + Track 2 (+ ttcomp02, ctt — бонусом)
git clone --depth 1 https://github.com/tomas-muller/cpsolver-itc2007.git itc_track1_solver
ls itc_track1_solver/data/exam/    # 8 .exam файлов, 1.7 MB
ls itc_track1_solver/data/tim/     # 16 .tim файлов, 8.4 MB

# 3. (опционально) Track 3 — curriculum-based
git clone --depth 1 https://github.com/Docheinstein/itc2007-cct.git itc_solver
ls itc_solver/datasets/            # 24 .ctt файла, 276 KB

# 4. Окружение для проверки
uv venv .venv
./.venv/bin/python -m pip install pandas

# 5. Парсинг и проверка (готовый скрипт)
./.venv/bin/python parse_itc.py        # выводит схему всех инстансов
./.venv/bin/python check_parallel.py   # выводит capacity-tightness
```

Скрипт `parse_itc.py` (в той же папке) парсит и Track 1, и Track 2 в pandas DataFrame и готов к подключению как новый адаптер в `experiments/src/data/`.

---

## Если нужно идти ещё глубже (на случай, если Track 1 не убедит научрука)

1. **Purdue C8-2007 (UniTime)** — однотипный, но больший масштаб; реальные данные одного университета. Скачать архивы по ссылкам с `unitime.org/data/pu-fal07-*.zip`. Трудозатраты: 1–2 часа на скачивание и парсинг (формат тоже хорошо документирован).
2. **ITC-2019 (International Timetabling Competition 2019)** — современная версия с десятком университетов разных стран и 60+ инстансами. Сайт https://www.itc2019.org/, инстансы XML-формат. Трудозатраты: ~3–4 часа на парсинг XML и приведение к нашей схеме. Перебор для предзащиты, но мощный аргумент для собственно защиты летом.
3. **Не нужно** скрапить Eventbrite/Sched, обращаться к JUG или строить синтетику на основе IEEE/ACM программ — ITC-2007 это всё закрывает. Эти ходы оставить на «вдруг рецензент попросит ещё».

---

## Конкретная рекомендация одной фразой

Берём `exam_comp_set1.exam` как основной экземпляр для cross-domain секции Главы 4 (реальные данные университета Удине: 7 883 студента × 607 экзаменов × 7 комнат с capacity 60–260 × 54 периода, 32 380 фактических регистраций, capacity-binding ≈ 0,75) — это снимает атаку рецензента, и это можно подключить за один вечер до 08.05.

---

# Расширенный поиск (раунд 2, 2026-05-01)

## TL;DR раунда 2

После повторного поиска с обязательным скачиванием каждого кандидата проверено ещё девять датасетов. Главные выводы:

- **Обнаружен event-domain датасет ближе к JUG, чем ITC-2007.** Это `RSVP-Prediction-Meetup` (репозиторий `ozgekoroglu/RSVP-Prediction-Meetup`) — слепок Meetup.com по Нидерландам за 2007–2016: 6 200 событий, 37 356 пользователей, 1 732 venue, 174 139 RSVPs, **поле `rsvp_limit` есть на 2 211 событиях** (35,7%), 1 028 одновременных слотов, 20 641 пользователь имеет ≥2 событий. Capacity-binding (yes_rsvp / rsvp_limit): mean 0,61, 582 события tight (≥0,8). Это методологически идеальный для ВКР датасет: реальные офлайновые соц-события (а не экзамены), реальные venue с capacity (а не аудитории), реальные RSVP-события (а не регистрации на курс). Это **главный приз раунда 2.**

- **ITC-2019 — масштабное обновление ITC-2007.** Скачано 36 XML-инстансов из `ADDALemos/MPPTimetables` (280 МБ). Реальные данные **десяти университетов** (AGH, BET, IKU, LUMS, MUNI, MARY, NBI, Purdue, TG, WBG, YACH). У `mary-fal18`: 93 комнаты × 5 051 студент × 540 курсов × 21 017 enrollments, capacity 8–100, tightness ≈ 1,13 (binding). У `pu-llr-spr07`: 56 комнат × 27 881 студент × 603 курса × 81 781 enrollment. Современная замена ITC-2007 для cross-domain секции — берём 2-3 инстанса как «панораму» вместо одного экзаменационного.

- **Purdue C8-2007 (UniTime)** — реальные данные Purdue (~30 тыс студентов, 896 курсов, 63 комнаты cap 40–474), формат UniTime XML, есть `<sharing>` с departments и patterns доступности. Третий кандидат для cross-domain.

- **Toronto / Nottingham (классика 1990-х)** — Toronto бенчмарк существует, но в нём **нет capacity комнат** (только списки студентов на экзамене). Nottingham 1995 имеет полные реальные комнаты с capacity (TRENT-HALL=125, SPORT-LGE1=250 и т.д.), 800 экзаменов, 7 896 студентов, 33 997 enrollments, **но это маленький экземпляр и устарелый формат**. Не лучше ITC.

- **Конференционные XML (FOSDEM/JuliaCon/pretalx)** — public schedule данные есть и были скачаны (FOSDEM 2019–2024, JuliaCon 2023). У FOSDEM 2024 даже **есть полная таблица capacity комнат на сайте** (Janson 1415, K.1.105 805, ..., 35 комнат от 40 до 1415 мест). Но в XML-расписании нет per-attendee data → это «scheduled what where», а не recsys-датасет. Использовать как chapter-3 иллюстрацию реалистичных capacity-чисел можно, как полноценный валидационный датасет — нельзя.

- **Healthcare / Cinema / Hotel-booking / Air-cargo / Coursera / Booking.com MDT / Sched.com / Sessionize** — все проверены со скачиванием либо явным probing API. Capacity-поля и/или user_id с повторами и/или параллельных слотов не хватает в каждом случае. Все вердикты см. ниже.

**Меняется ли финальная рекомендация?** Да: **основной датасет — `RSVP-Prediction-Meetup`** (event domain, real venue capacity, real RSVPs); **второй point — `mary-fal18.xml` (ITC-2019)** или `exam_comp_set1.exam` (ITC-2007) для контроля. ITC-2007 теперь — не основной, а sanity-проверка.

## Новые проверенные кандидаты

### 11. ITC-2019 (`MPPTimetables/data/input/ITC-2019/`) — **ПОДХОДИТ, основная замена ITC-2007**

- **Ссылка:** репозиторий `ADDALemos/MPPTimetables` на GitHub (зеркало с XML-инстансами, поскольку `itc2019.org` отдаёт SPA без прямых ссылок).
- **Скачано:** да, `git clone --depth 1 https://github.com/ADDALemos/MPPTimetables.git`, 280 МБ, **36 XML-инстансов** в `data/input/ITC-2019/`.
- **Схема (по парсеру):**
  ```
  <timetable nrDays="7" slotsPerDay="288" nrWeeks="N">
    <rooms>
      <room id="X" capacity="K">...</room>
    </rooms>
    <courses>
      <course id="X">
        <config>
          <subpart>
            <class id="Y" limit="L">
              <room id="X" .../>     # допустимая комната
              <time .../>
            </class>
  ...
    <students>
      <student id="S">
        <course id="X" />            # фактическая регистрация
  ```
- **Размер (девять интересных, посчитано Python-парсером):**

  | Инстанс | Комнаты | Cap min | Cap max | Cap total | Студенты | Курсы | Enrollments | Tightness* |
  |---------|--------:|--------:|--------:|----------:|---------:|------:|------------:|-----------:|
  | agh-fis-spr17 | 80 | 5 | 240 | 3493 | 1641 | 340 | 13 415 | 0,55 |
  | agh-ggis-spr17 | 44 | 15 | 240 | 2301 | 2116 | 272 | 14 762 | **0,92** |
  | mary-fal18 | 93 | 8 | 100 | 2655 | 5051 | 540 | 21 017 | **1,13** |
  | mary-spr17 | 90 | 8 | 198 | 2635 | 3666 | 544 | 10 552 | 0,57 |
  | muni-fi-fal17 | 36 | 8 | 248 | 1403 | 1685 | 188 | 11 101 | **1,13** |
  | muni-fi-spr16 | 35 | 10 | 248 | 1482 | 1543 | 228 | 9 633 | **0,93** |
  | pu-llr-spr07 | 56 | 20 | 474 | 7538 | 27 881 | 603 | 81 781 | **1,55** |
  | bet-fal17 | 62 | 14 | 260 | 2787 | 3018 | 353 | 18 827 | **0,97** |
  | bet-spr18 | 63 | 14 | 260 | 2837 | 2921 | 357 | 19 053 | **0,96** |

  *tightness = enrollments / (cap_total × days), приближение «binding» — везде, где tightness ≥ 0,8, capacity действительно зажимает.
- **Capacity-поле:** **да**, `<room capacity="N">` на каждой комнате; в нескольких инстансах (muni-pdf, pu-proj) ставят 9999 для «unbounded» — таких избегать.
- **User_id с повторами:** **да**, `<student id="...">` с явным списком `<course>` (3–6 курсов на студента в норме).
- **Параллельные слоты:** **да**, формат жёстко фиксирует «день × слот в неделе» с `nrDays`, `slotsPerDay`, `nrWeeks`; в каждом слоте параллельно до десятков занятий в разных комнатах.
- **Применимость:** **прямая замена ITC-2007 с большим разнообразием университетов** (Чехия, Италия, Польша, Новая Зеландия, Турция, США, Дания). Реальный 2010–2019 период, выше шансов на «наш» domain (UNIWE → tech focus).
- **Итог: основной кандидат для cross-domain Главы 4.** Использовать `mary-fal18.xml` (компактный, tight) и `bet-spr18.xml` (compact, tight, bilingual studies) как 2 contrast point.

### 12. Purdue C8-2007 (`pu-fal07-llr.zip` от unitime.org) — **ПОДХОДИТ как третий point**

- **Ссылка:** `https://www.unitime.org/data/pu-fal07-llr.zip` (через wget угаданным URL после анализа `unitime.org/uct_datasets.php`).
- **Скачано:** да, 460 КБ ZIP, разворачивается в `pu-fal07-llr.xml` (6,7 МБ).
- **Схема:** `<timetable initiative="puWestLafayetteTrdtn" term="2007Fal" nrDays="7" slotsPerDay="288">`
  ```
  <rooms> <room id="N" capacity="K" location="x,y"> ... </rooms>
  <classes> <class>...</class> ... </classes>
  <groupConstraints> ... </groupConstraints>
  <students> <student id="S"> ... </students>
  ```
- **Размер:** 63 комнаты (capacity **40 … 474**, mean 126), 896 классов, 30 846 студентов, 282 group-constraint, 7 дней × 288 слотов = 2016 потенциальных временных позиций.
- **Capacity-поле:** **да**, `capacity="..."` на каждой `<room>`.
- **User_id с повторами:** **да**, явный `<student id="...">`.
- **Параллельные слоты:** **да**, минимальная гранулярность 5 минут (slotsPerDay=288 за 24ч).
- **Применимость:** **самый крупный реалистичный конкретный университет** в нашем поиске — 30 тысяч студентов одного семестра. Те же XML-парсеры подойдут.
- **Итог: подходит как «крупный» point. Можно взять как N=3 для робастности в Главе 4.**

### 13. RSVP-Prediction-Meetup (`ozgekoroglu/RSVP-Prediction-Meetup`) — **ПОДХОДИТ И ЛУЧШЕ ВСЕХ ОСТАЛЬНЫХ ПО ДОМЕНУ**

- **Ссылка:** `https://github.com/ozgekoroglu/RSVP-Prediction-Meetup`.
- **Скачано:** да, 62 МБ, в репо лежат `data/{events,users,venues,groups}.json`.
- **Схема (по парсеру):**
  ```json
  events.json: [
    {
      "name": "...",
      "group_id": "...",
      "venue_id": 67082,
      "time": 1310056200000,        # epoch ms
      "duration": null|<minutes>,
      "rsvp_limit": null|<int>,     # capacity!
      "rsvps": [
        {"user_id": 39980, "guests": 0, "when": 1309881442000, "response": "yes"}
      ]
    }
  ]
  users.json: [{user_id, city, country, hometown, memberships}]
  venues.json: [{venue_id, name, lat, lon, city, country}]
  groups.json: [{group_id, ...}]
  ```
- **Размер (точно посчитано):**
  - 6 200 events (1 — 414 RSVPs/event, mean 28,1)
  - 37 356 уникальных users
  - 1 732 venues с координатами и страной/городом
  - 572 групп (тематических meetup-сообществ — IT, language, hobbies)
  - 174 139 user-event пар (RSVPs)
  - **2 211 событий имеют `rsvp_limit`** (35,7%): диапазон 1 — 500
  - временной охват: 2007-03 — 2016-08 (9 лет!)
  - 4 382 distinct event time-slots, **1 028 (23,5%) имеют ≥2 параллельных события** (max 11 параллельных)
- **Capacity-поле:** **да** (`rsvp_limit`), на 35,7% событий. Можно либо ограничиться этим подмножеством (2 211 событий, 92 180 RSVPs), либо для остальных вычислить «факт-capacity» = max(yes_rsvps) как proxy.
- **User_id с повторами:** **да**, 20 641 пользователь имеет ≥2 событий (55%); top user — 364 события за 9 лет.
- **Параллельные слоты:** **да**, 1 028 одновременных тайм-слотов с до 11 событиями параллельно.
- **Применимость:** **семантически близко к JUG**. Real-life event domain: каждый event — собрание людей в физическом venue с capacity. Группы (`group_id`) идеально мапятся на «треки/тематики» JUG. У события есть start/duration. У venue есть имя, координаты, страна. У RSVP — guest-count и timestamp («когда зарегистрировался»).
- **Capacity-tightness (yes_rsvp / rsvp_limit) на подмножестве с лимитом:**
  - mean 0,609, median 0,614 (сравнимо с ITC-2007 0,75)
  - 121 событий «переполнены» (waitlist, ratio ≥ 1.0)
  - 582 событий tight (ratio ≥ 0,8) — основной материал
  - 785 событий loose (ratio < 0,5) — фон
  - **очень похожая на JUG картина**: часть популярных треков переполнена, часть полупустая.
- **Минусы:** 
  1) `rsvp_limit` есть только на 35,7% событий → реальный размер «подходящего» подкорпуса 2 211 events.
  2) Нет category/topic для events явно (только через group → group_tag из соседней работы wuyuehit, можно объединить).
  3) Все venue в Нидерландах — географически узко, но зато локально-сравнимо с JUG (тоже одна страна по запросу).
- **Итог: основной кандидат для cross-domain секции вместо ITC.** Сценарий валидации: «применяем 6+5 политик к 2211 capacity-bound Meetup events; baseline cosine-MMR vs LLM ranker — на параллельных слотах с разной capacity». Это снимает атаку «синтетический capacity» гораздо сильнее, чем экзамены, потому что это event domain.

### 14. Toronto exam timetabling benchmark (`Toronto.zip`) — **НЕ ПОДОШЁЛ (нет capacity комнат)**

- **Ссылка:** `https://people.cs.nott.ac.uk/pszrq/files/Toronto.zip`.
- **Скачано:** да, 643 КБ, 13 университетских инстансов 1983–1993.
- **Схема:** парные файлы `.crs` (ID и количество студентов) и `.stu` (списки studentID на каждом экзамене).
  ```
  car-f-92.crs:  0001 280              # exam_id, num_students
  car-f-92.stu:  0170 ... 0156         # одна строка = один студент, в строке список экзаменов
  ```
- **Размер:** до 30 032 студентов и 2 419 экзаменов в самом большом (`pur-s-93`).
- **Capacity-поле:** **нет** — Toronto benchmark по определению не моделирует комнаты. Стандартное предположение в литературе: «суммарная capacity > всех экзаменов» (т.е. unconstrained). 
- **User_id с повторами:** **да**, явные ID студентов с repeat.
- **Применимость для нашей задачи:** ограничена — без capacity это просто графовая раскраска (graph colouring), не CMDP. Можно использовать только для «чистой релевантности» без capacity-binding.
- **Итог: не подходит для основной задачи.**

### 15. Nottingham 1994/95 exam timetabling (`Nott.zip`) — **частично подходит**

- **Ссылка:** `https://people.cs.nott.ac.uk/pszrq/files/Nott.zip`.
- **Скачано:** да, 172 КБ.
- **Схема:** 4 текстовых файла (students, exams, enrolements, data):
  ```
  exams: AA2016E1 OPERA STUDIES, I               1:30 GM    # exam_id, name, duration, dept
  students: A890186790 R100                                  # 10-char student_id, course_code
  enrolements: A890186790 R13001E1                          # student_id ↔ exam_id
  data: ROOMS\nTRENT-HALL 125\nTRENT-L19 80\n...             # room_name capacity
  ```
- **Размер:** 800 экзаменов, 7 896 студентов, 33 997 enrolments (4,3 экзамена/студент).
- **Capacity-поле:** **да**, секция ROOMS в файле `data` с реальными именами комнат и capacity (TRENT-HALL=125, SPORT-LGE1=250, ART-LECTURE=80, ...). Также есть «room assignments» — какой экзамен в какой комнате (фактическое решение!).
- **User_id с повторами:** **да**, реальные ID типа `A890186790`.
- **Параллельные слоты:** **да**, 2 недели × (Mon-Fri 9:00, 13:30, 16:30 + Sat 9:00) = ~22 слота × несколько комнат.
- **Применимость:** хорошая, но **меньше масштаба** ITC-2007 (Nottingham имеет один университет, ITC — несколько). Семантически — **тот же экзаменационный домен**, что и ITC, поэтому добавлять третьим экзаменационным point не имеет ценности.
- **Итог: годится как 4-й контрольный экзаменационный point, но избыточно при наличии ITC-2007 + ITC-2019.**

### 16. FOSDEM schedule XMLs (2019, 2020, 2022, 2023, 2024) — **НЕ ПОДОШЁЛ для recsys, но полезен как контекст**

- **Ссылка:** `https://archive.fosdem.org/{year}/schedule/xml`.
- **Скачано:** да, 5 файлов 1,6–2,6 МБ каждый.
- **Схема:** Pentabarf XML format
  ```xml
  <schedule>
    <conference>...</conference>
    <day index="1" date="...">
      <room name="Janson">
        <event>
          <start>10:30</start>
          <duration>00:50</duration>
          <title>...</title>
          <persons>
            <person id="123">Speaker name</person>
          </persons>
        </event>
  ```
- **Размер (FOSDEM 2024):** 35 rooms, 874 events, 948 unique persons (only speakers, NOT attendees).
- **Capacity-поле:** **в XML — нет**, но **на сайте** archive.fosdem.org/2024/schedule/rooms/ есть полная таблица: Janson=1415, K.1.105=805, ..., 35 rooms от 40 до 1415 мест (можно scrape WebFetch'ом, что и было сделано).
- **User_id с повторами:** **нет** — есть только speakers (один-два на event), нет attendees.
- **Параллельные слоты:** **да**, в XML очевидно по `<day><room>` иерархии.
- **Применимость:** для **chapter 3** как «иллюстрация реальных capacity» (вот цифры реальной IT-конференции, чтобы аргументировать выбор capacity 100/200/350 в JUG-симуляторе). Как валидационный recsys-датасет — нет, потому что не хватает attendee-уровня.
- **Итог: не основной, но полезен в Главе 3 как «вот FOSDEM 2024: 35 залов от 40 до 1415 мест, 874 talks параллельно — это и есть реальная конференционная capacity».**

### 17. JuliaCon 2023 pretalx XML — **НЕ ПОДОШЁЛ (тот же шаблон что FOSDEM)**

- **Ссылка:** `https://pretalx.com/juliacon2023/schedule/export/schedule.xml`.
- **Скачано:** да, 1 МБ.
- **Размер:** 10 rooms, 354 events, 269 persons (speakers).
- **Capacity-поле:** **нет**, pretalx XML не экспортирует capacity комнат вообще.
- **User_id с повторами:** **нет** (только speakers).
- **Применимость:** none.
- **Итог: не подходит. Pretalx как формат не отдаёт capacity.**

### 18. Air-cargo SQL training dataset (`fortunewalla/air-cargo`) — **НЕ ПОДОШЁЛ (синтетика)**

- **Ссылка:** `https://github.com/fortunewalla/air-cargo`.
- **Скачано:** да.
- **Размер:** 50 customers, 50 passengers_on_flights, 50 routes, 50 ticket_details — это игрушка для SQL-курсов BCG Rise, не реальный авиа-датасет.
- **Capacity-поле:** **нет** (нет seat capacity на aircraft).
- **Итог: не подходит (синтетический туториал).**

### 19. Booking.com Multi-Destination Trips Dataset (`bookingcom/ml-dataset-mdt`) — **НЕ ПОДОШЁЛ (нет hotel-id с capacity)**

- **Ссылка:** `https://github.com/bookingcom/ml-dataset-mdt`.
- **Скачано:** да, ~140 МБ, train_set.csv 1,17M строк.
- **Схема:** `user_id, checkin, checkout, city_id, device_class, affiliate_id, booker_country, hotel_country, utrip_id` — нет hotel_id, нет capacity.
- **Capacity-поле:** **нет**.
- **User_id с повторами:** **есть** (utrip_id со связкой нескольких бронирований).
- **Применимость:** не годится — нет capacity.
- **Итог: не подходит. SIGIR'21 challenge data — не для нашей задачи.**

### 20. `wuyuehit/meetup_dataset` (повторно проверен полным скачиванием) — **НЕ ПОДОШЁЛ (нет capacity)**

- **Ссылка:** `https://github.com/wuyuehit/meetup_dataset`.
- **Скачано:** да, 537 МБ полностью (повторная проверка после первого агента).
- **Размер:** 2,59M event-group пар, и две zip с user-event mapping (~120M строк суммарно), плюс tag-файлы.
- **Capacity-поле:** **нет** в этом релизе. Schema: `user_id, event_id` (две колонки, без metadata).
- **Применимость:** нет, формат «только rsvp-факт без лимита».
- **Итог: подтверждён вердикт первого агента. Не подходит.**

### 21. GoalZone fitness booking (`AnsaBaby/Fitness-Dataset-analysis`) — **НЕ ПОДОШЁЛ (тоже подтверждено)**

- **Ссылка:** `https://github.com/AnsaBaby/Fitness-Dataset-analysis`.
- **Скачано:** да.
- **Схема:** `booking_id, months_as_member, weight, days_before, day_of_week, time, category, attended` (1 500 строк).
- **Capacity-поле:** **нет** (нет class_capacity).
- **User_id с повторами:** **нет** (только booking_id, не member_id).
- **Параллельные слоты:** **нет** (только day_of_week + AM/PM).
- **Итог: не подходит.**

### 22. ozgekoroglu meetup vs `developers-conferences-agenda` — последний — **НЕ ПОДОШЁЛ**

- `scraly/developers-conferences-agenda` — markdown-список IT-конференций с CFP-ссылками, без расписаний и без attendee-данных.
- **Итог: для построения каталога — да, для recsys-датасета — нет.**

### 23. Sched.com / Sessionize APIs — **не отдают capacity без логина**

- Sched.com API — на каждое event свой ключ (`https://event.sched.com/api/session/list?api_key=XXX`); публично закрыто.
- Sessionize API — endpoint `/api/v2/{eventId}/view/Sessions` отдаёт 404 без авторизации; пробовали на `devoxxfr2024`, `jboss-conftest-2024`.
- **Итог: пути нет без личных credentials. Не пробивать.**

### 24. MDPI outpatient clinic dataset (PMC8056789, 6 637 records) — **не скачано (paywall на supplementary)**

- **Ссылка:** `https://www.mdpi.com/2306-5729/8/3/47/s1`.
- **Скачано:** **нет, 403 Access Denied** на supplementary при попытке curl.
- **По описанию (через WebSearch):** `ID, Session, Month, DayOfWeek, AM_PM, Visit.No, Gender, ...`. Нет doctor/clinic capacity явно (только implicit «слотов в half-day»).
- **Итог: не критичен, и не достал. Health-домен дальше не толкаем.**

## Сводка по отвергнутым в раунде 2 (краткая)

| # | Кандидат | Почему не подошёл (одна строка) |
|---|----------|---------------------------------|
| 14 | Toronto | нет capacity комнат, только списки студентов |
| 16 | FOSDEM XML | есть capacity на сайте, но в XML — только schedule, нет attendees |
| 17 | JuliaCon pretalx | pretalx не экспортирует capacity, нет attendees |
| 18 | Air-cargo | синтетика на 50 строк (SQL-туториал) |
| 19 | Booking.com MDT | нет hotel_id и нет capacity |
| 20 | wuyuehit Meetup | гигантская, но capacity-поле отсутствует |
| 21 | GoalZone fitness | нет user_id и нет class_capacity |
| 22 | dev-conf-agenda | markdown-каталог, не данные |
| 23 | Sched.com / Sessionize | требуется per-event ключ |
| 24 | MDPI outpatient | 403, и даже по описанию нет capacity |

## Финальный пересмотренный список

1. **`ozgekoroglu/RSVP-Prediction-Meetup`** — основной cross-domain датасет (event domain, real RSVP, real venue capacity на 2 211 событиях, 9 лет).
2. **ITC-2019 (`mary-fal18.xml` + `bet-spr18.xml`)** — второй point, реальные университеты с tight capacity (>0.9). Лучше ITC-2007 в плане свежести.
3. **`exam_comp_set1.exam` (ITC-2007)** — третий, контрольный point, для совместимости с литературой и поскольку он уже подключён.
4. **Purdue C8-2007 (`pu-fal07-llr.zip`)** — резерв, если рецензент попросит крупный «один кампус» N=4.
5. **FOSDEM 2024 capacity table** — для Главы 3 как иллюстрация реалистичных значений capacity.

## Окончательная рекомендация (после двух раундов)

Финальная архитектура валидации (двухдоменная): 

- **Domain A (event/social)** — `RSVP-Prediction-Meetup`, 2 211 событий с `rsvp_limit`, capacity-tightness среднее 0,61, 1 028 параллельных слотов, mean 1,4 события/слот, 37 356 пользователей. Это **семантически ближе всего к JUG** (Mobius/Heisenbug — это тоже event-RSVP-domain с venue capacity и treck-параллелизмом).
- **Domain B (course/exam)** — `mary-fal18.xml` (ITC-2019, 5 051 студент × 540 курсов × 93 комнаты cap 8–100, tightness 1,13). Это **классическая академическая «параллельные занятия в комнатах с capacity»** структура.
- **MovieLens 1M** оставляем как sensitivity на популяционный шум (как в исходном плане).

Это даёт полную панораму: **event-domain (главное)** + **course-domain (классика)** + **MovieLens (популяция)**. Атаку «капасити синтетическая» снимает раз и навсегда, потому что в Meetup-данных capacity — это `rsvp_limit`, выставленный самими организаторами события (живой реальный сигнал), а в ITC-2019 — это physically-existing university room capacity.

## Команды для воспроизведения раунда 2

```bash
cd /Users/fedor/Study/masters-degree/experiments/data/external/deep_search_2026_05/round2

# 1. ITC-2019 (рекомендуемый основной для course domain)
mkdir -p itc2019 && cd itc2019
git clone --depth 1 https://github.com/ADDALemos/MPPTimetables.git
ls MPPTimetables/data/input/ITC-2019/    # 36 .xml инстансов (280 МБ)

# 2. Purdue C8-2007 (резерв)
cd .. && mkdir -p purdue && cd purdue
curl -sL -o pu-fal07-llr.zip "https://www.unitime.org/data/pu-fal07-llr.zip"
unzip pu-fal07-llr.zip

# 3. Meetup RSVP (основной для event domain)
cd .. && mkdir -p meetup_rsvp && cd meetup_rsvp
git clone --depth 1 https://github.com/ozgekoroglu/RSVP-Prediction-Meetup.git
ls RSVP-Prediction-Meetup/data/    # events.json, users.json, venues.json, groups.json (62 МБ)

# 4. FOSDEM (для иллюстраций в Главе 3)
cd .. && mkdir -p fosdem && cd fosdem
for y in 2019 2020 2022 2023 2024; do
  curl -sL -o "fosdem-${y}.xml" "https://archive.fosdem.org/${y}/schedule/xml"
done

# 5. Парсеры — в /Users/fedor/Study/masters-degree/experiments/data/external/deep_search_2026_05/round2/
#    можно адаптировать parse_itc.py из round 1 на ITC-2019 (тот же XML), и
#    написать новый адаптер для Meetup JSON (tools/data/meetup_adapter.py).
```

## Что дальше

Для подключения в Главу 4 нужно:

1. Написать адаптер `experiments/src/data/meetup_rsvp_adapter.py`, читающий `events.json` и фильтрующий по `rsvp_limit IS NOT NULL`. Эмулировать как N=2211 событий, K_t = `rsvp_limit`, релевантность для пары (user, event) = `+1` если `(user_id, event_id)` ∈ rsvps with response='yes' и `0` иначе. **Время: ~2 часа.**
2. Написать адаптер `experiments/src/data/itc2019_adapter.py` под XML-формат ITC-2019. Мапа `course → set(allowed_rooms)`, `student → set(courses)`, capacity на комнатах. **Время: ~3 часа.**
3. Прогнать 6+5 политик на обоих adapter'ах + старый MovieLens. **Время: ~1 вечер с учётом heartbeat и watchdog.**
4. В Главе 4 написать секцию «Cross-domain validation: event RSVP (Meetup), course timetabling (ITC-2019), синтетический MovieLens».

Все три источника датасетов лежат локально, валидация научного теста снимается.

