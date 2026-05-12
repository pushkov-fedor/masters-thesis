## ⚠ ГЛАВНОЕ ПРАВИЛО

**Когда пользователь говорит «ознакомься с проектом» (или эквивалент — «погрузись», «вспомни проект», «get up to speed»):** сначала прочитать `/Users/fedor/Study/masters-degree/PROJECT_OVERVIEW.md` — это сжатый снимок всего проекта (модель, код, результаты, текст ВКР, открытые вопросы предзащиты), специально подготовленный как точка входа. Только потом при необходимости углубляться в `PROJECT_STATUS.md` / `PROJECT_DESIGN.md` / `PIVOT_IMPLEMENTATION_PLAN.md` и код.

Перед любой содержательной работой по ВКР `PROJECT_STATUS.md` остаётся авторитетным документом по тезису, фрейму, валидации и стоп-листу. Эта memory — рабочий кэш Claude Code, синхронизирован с репо `<repo>/.claude/memory/`.

## Активный фрейм

- **Точка входа в проект — [`PROJECT_OVERVIEW.md`](/Users/fedor/Study/masters-degree/PROJECT_OVERVIEW.md) в корне репо** (актуальный блок «Update 2026-05-12 — EN pipeline pivot» в начале).
- [EN-pipeline rerun 2026-05-12](project_en_rerun_2026-05-12.md) — перегон на BGE-large-en + ABTT-1; новые числа Q + V + cross-validation; Q-O7 пройден сильнее (median ρ 0.769 vs 0.554)
- [Финальный фрейм ВКР](project_final_thesis_frame.md) — сценарный аналитический полигон / DSS для стресс-теста программы конференции
- [Стоп-лист тезисов](project_thesis_stoplist.md) — что больше НЕ защищается в работе

## Контекст автора и формальности

- [Магистратура и ВКР](user_masters_context.md)
- [Антиплагиат](project_antiplagiat_check.md) — до 08.05.2026
- [Предзащита](project_predzashchita.md) — 13.05.2026
- [Тема ВКР](project_thesis_topic.md)
- [Кодовая база jug-rec-sys](project_jug_recsys_codebase.md)

## Технические наблюдения и литература

- [Технические наблюдения](project_technical_observations.md) — capacity, mean_overload_excess, узкий конус e5, B1-leakage
- [Модель генерации EN-персон](project_persona_generation_model.md) — EN-пул Mobius 100 персон сделан через Claude Opus 4.7, не Sonnet (запись в spike-аудите устарела)
- [Повторный поиск датасетов 12.05.2026](project_dataset_search_2026-05-12.md) — исчерпывающий поиск перед предзащитой, ничего не нашёл по структурной причине; список проверенных доменов и Q6 в speech.md
- [Оператор Φ деприкейтнут 12.05.2026](project_phi_operator_deprecated_2026-05-12.md) — на параметрическом симуляторе эффект слабый, на защите не выносится; ось program_variant можно убирать в будущих LHS-перегонах
- [Hall-conflict bug в program_modification.py](project_phi_hall_conflict_bug.md) — основной оператор Φ сохраняет hall и не проверяет конфликт залов; исправить в финальной версии после защиты
- [Литература и канон валидации](reference_validation_defense.md) — must-cite, канон Sargent / Kleijnen / DMDU / Larooij & Törnberg, distribution-match Meetup ρ=0.438
- [Обзор предметной области 2026-05-04](research_field_survey_2026-05-04.md) — карта подходов, capacity-aware recsys, LLM-агенты, gap-анализ, плюс EMA-workbench / PRIM как лёгкое усиление

## Поведенческие правила

- [Не ставить пакеты глобально без разрешения](feedback_no_global_installs.md)
- [Не злоупотреблять англицизмами](feedback_no_anglicisms.md)
- [Академический стиль](feedback_academic_style.md)
- [Без первого лица в академическом тексте](feedback_no_first_person.md) — нет «наш / мой / мы / я» в тексте ВКР, слайдах, заметках докладчика
- [Самосовершенствование](feedback_self_improvement.md)
- [uv для Python](feedback_python_uv.md)
- [Проверять семантику данных](feedback_verify_data_semantics.md)
- [Не переоценивать время на эксперименты](feedback_no_time_estimates.md)
- [Watchdog для долгих прогонов](feedback_watchdog_long_runs.md)
- [Видимый прогресс и self-check](feedback_long_runs_visible_progress.md)
- [Видимый ETA через tqdm](feedback_show_eta.md)
- [Не отодвигать задачи за защиту](feedback_no_post_defense_excuses.md)
- [Скептический режим](feedback_skeptical_honest.md)
- [Не быть да-человеком](feedback_no_yes_man.md)
- [НЕ СРАТЬ ФАЙЛАМИ](feedback_no_backup_files.md)
- [PM-режим в момент паники](feedback_pm_mode_not_writer.md)
- [Валидировать параметры важных прогонов](feedback_validate_run_params.md)
- [Проактивно предлагать реалистичные параметры](feedback_propose_realistic_params.md)
- [Без скобочных пояснений в лейблах графиков](feedback_no_parenthetical_label_hints.md)
- [На consistency-проверке открывать все \input{}](feedback_check_all_input_files.md)
- [Slidev — пустые строки внутри HTML-блоков](feedback_slidev_html_blank_lines.md)
