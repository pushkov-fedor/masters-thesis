---
name: Модель генерации EN-персон Mobius
description: EN-пул 100 персон под Mobius (50 spike + 50 part2) сгенерирован через Claude Opus 4.7 — не Sonnet, как ошибочно зафиксировано в spike-аудите
type: project
originSessionId: 8088bf28-d328-45d5-97c8-e30324d87cb5
---
EN-пул синтетических персон под Mobius `experiments/data/personas/personas_mobius_en.json` (100 персон, использован в EN-перегоне 12.05.2026) сгенерирован через **Claude Opus 4.7** — оба прохода: первые 50 в spike и догенерация 50 part2.

**Why:** в `docs/spikes/spike_relevance_function_audit.md` §II и §V модель ошибочно указана как «Sonnet». Это устаревшая запись, оставшаяся от черновика. Источник истины — подтверждение автора 2026-05-12.

**Как отличается от старого RU-пула:** `personas_100.json` (RU, не используется в основном эксперименте после пивота) сгенерирован через скрипт `experiments/scripts/generate_personas.py` с дефолтным `anthropic/claude-haiku-4.5`. Это другой пул, не путать.

**How to apply:**
- В тексте речи, слайдах, ВКР писать «через Claude Opus 4.7».
- При правке spike-аудита в финальной версии работы (после 13.05) — исправить «Sonnet» на «Opus 4.7» в §II.4, §V.
- Если subagent в будущей сессии говорит «Sonnet сгенерировал…» — не верить slepо, сверять с этим фактом.
