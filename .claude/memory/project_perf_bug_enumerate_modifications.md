---
name: Perf-bug в enumerate_modifications — O(N²) deepcopy
description: Функция `enumerate_modifications` в `experiments/src/program_modification.py` делает deepcopy конференции в каждой итерации внешнего цикла; на крупных программах (Demo Day, 210 talks) забирает 98% wallclock LHS-прогона
type: project
originSessionId: 8088bf28-d328-45d5-97c8-e30324d87cb5
---
В `experiments/src/program_modification.py:_apply_swap` и `enumerate_modifications` используется `copy.deepcopy(conf)` в каждой итерации цикла перебора candidate-pairs. На больших программах это O(N²) deepcopy.

**Обнаружено:** 12 мая 2026 при прогоне Demo Day EN (210 talks × 56 slots, ~21 тыс. кандидатных пар). Из 908 секунд общего wallclock LHS-прогона **886 секунд (98%)** ушло на `enumerate_modifications` для `program_variant > 0`. Сама симуляция П1–П3 заняла всего 22 секунды.

**На Mobius (40 talks × 16 slots) эффект незаметный** — несколько секунд из общего минутного прогона.

**Why:** функция вызывается один раз на каждую LHS-точку с `program_variant > 0`. На Mobius это 5 точек × ~150 кандидатных пар = ~750 deepcopy. На Demo Day это 5 точек × ~21000 пар = ~105000 deepcopy с большой Conference. Корректность не нарушена.

**How to apply (в финальной версии работы после защиты):**

1. **Оптимизировать `_apply_swap`** — mutate-in-place + revert после проверки конфликта, либо хранить только diff (`SwapDescriptor` уже есть, можно использовать lazy evaluation).
2. **Альтернатива** — кешировать список валидных пар на уровне Conference, чтобы `enumerate_modifications` не пересчитывал заново для каждой LHS-точки с одной и той же базовой программой.
3. **Тривиальная оптимизация** — `has_speaker_conflict` можно проверять до `_apply_swap`, не делая deepcopy для заведомо невалидных пар.

**Не критично для защиты 13.05:**
- В упрощённом перегоне Mobius (план #35) `program_variant = 0` фиксируется → `enumerate_modifications` не вызывается вообще, perf-bug не активен.
- В Demo Day прогон уже сделан, повторять не нужно.
- В тексте ВКР конкретные числа wallclock не зафиксированы как защищаемая характеристика.

**Связь с другим известным багом:** см. `project_phi_hall_conflict_bug.md` — отдельная семантическая проблема в том же коде (свап сохраняет hall, не проверяет конфликт залов). Оба бага исправить в одной итерации после защиты.
