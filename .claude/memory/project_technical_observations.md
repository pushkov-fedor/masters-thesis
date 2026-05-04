---
name: Технические наблюдения по проекту
description: Сводка ключевых технических фактов про capacity, метрики, эмбеддинги и B1-leakage; объединяет ранее разрозненные заметки
type: project
originSessionId: 2ab16009-c3da-4bf3-88ce-b864d0acdf4a
---
Сводка устойчивых технических наблюдений по проекту. Все четыре пункта согласованы с финальным фреймом (см. `project_final_thesis_frame.md`) и стоп-листом (`project_thesis_stoplist.md`).

## 1. Capacity на Mobius / Demo Day — параметр сценария, не реальные данные

Вместимости в `experiments/data/conferences/mobius_2025_autumn.json` и `demo_day_2026.json` заданы исследователем по правилу `cap[slot, hall] = ceil(N_population / halls_in_slot[slot])` через `experiments/scripts/patch_capacities.py`. Это **synthetic controlled capacity**, не реальные данные конкретной конференции. В новом фрейме это явный input scenario; в тексте Главы 4 и в Limitations нельзя называть «реальной capacity Mobius».

Reality datasets (Meetup, UMass) — capacity в исходных данных, не патчить.

## 2. Метрика `mean_overload_excess` — фильтр single-hall слотов

В `experiments/src/metrics.py` функция `mean_hall_overload_excess` пропускает слоты с одним залом (`if len(halls_in_slot) < 2: continue`). До правки 02.05 keynote-слоты входили в усреднение, давая всем политикам одинаковую фоновую добавку. Сейчас метрика согласована с `OF_choice (choice_only=True)` и `hall_utilization_variance`. Все три метрики считаются по одинаковому подмножеству слотов с ≥2 параллельными залами.

При цитировании любых старых чисел (из `_legacy_summaries/` или `_legacy_results/`) — учитывать, что они могли быть получены до этой правки.

## 3. Узкий конус косинусов e5

Распределение cosine между нормированными e5-эмбеддингами (user, talk):
- Mobius: min 0.758, max 0.895, std ≈ 0.020.
- Demo Day: min 0.751, max 0.889, std ≈ 0.019.

e5-small (multilingual, query/passage prefixes) на коротких русскоязычных описаниях докладов упаковывает все пары в узкий конус. Прямое следствие: cosine-recsys на этих данных не различает доклады в пределах шума, и utility почти не меняется между политиками. Это методологическое наблюдение, объясняющее «плоскую utility» в результатах E2 и обосновывающее обучаемую preference-модель как функцию релевантности.

При обсуждении utility в новом фрейме: разница между политиками в utility всегда мала, центр сравнения — метрики перегрузки, не utility.

## 4. B1 acc@1 = 0.918 — утечка через memberships, НЕ валидация

В `experiments/scripts/load_meetup_rsvp.py` (см. `_legacy_scripts/`):
- `group_emb[gid] = sum(topic_vec(t) for t in group.topics)`
- `talk_emb = group_emb[talk.group_id]`
- `user_emb = sum(group_emb[gid] for gid in user.memberships)`

В Meetup пользователь физически RSVP'ит только в события собственных групп. `cosine(user_emb, talk_emb)` тривиально высок там, где юзер в группе talk'а; argmax почти всегда даёт talk из самой «весомой» группы юзера. Метрика `accuracy@1 = 0.918` — это в основном замер «насколько часто юзер выбирает talk из своей самой популярной группы», а не качество модели предпочтений.

Этот результат **не идёт** в текст ВКР как валидация. В стоп-листе зафиксировано: «B1 / accuracy@1 = 0.918 как внешнюю валидацию» — нельзя использовать.

**Why:** обнаружено 03.05 при B2-валидации; критика на защите вскроет за час.

**How to apply:** при упоминании B1 в Главе 3 / 4 / приложениях — либо явно проговорить leakage, либо не упоминать. Лучше второе.
