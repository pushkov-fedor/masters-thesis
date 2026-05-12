# THESIS_CORRECTIONS_AFTER_DEFENSE

Дата фиксации: 2026-05-12
Целевая итерация: финальная версия ВКР после предзащиты 13.05.2026

> Структурированный чек-лист правок текста ВКР под перегон основного эксперимента 12.05.2026 (BGE-large-en + ABTT-1, 100 EN-персон, обновлённый Q + V + cross-validation). Текст после антиплагиата 08.05.2026 (PDF `thesis/ПушковФВ_ВКР.pdf`) использует числа RU-прогона; правки откладываются на финальную версию. Этот документ — запасной мозг, чтобы при возврате к тексту ничего не забыть.
>
> Базовая источниковая ссылка для всех правок: `experiments/results/report_mobius_2025_autumn_en_full.md` + блок «Update 2026-05-12» в `PROJECT_OVERVIEW.md`.

---

## 1. Глава 3 — устройство системы и методология

### 1.1. Функция релевантности

- [ ] Добавить раздел «Аудит и выбор функции релевантности». В первой версии работы использовалась косинусная близость через `intfloat/multilingual-e5-small`. По итогам аудита (`docs/spikes/spike_relevance_function_audit.md`) функция была заменена на `BAAI/bge-large-en-v1.5` с пост-обработкой ABTT-1.
- [ ] Описать формулу ABTT-1: центрирование (вычитание среднего по пулу) + ортогональная проекция к топ-1 PCA-направлению (Mu, Bhat, Viswanath, ICLR 2018). Применяется одним SVD на vstack(personas, talks), чтобы оба пула жили в одном postprocessed-пространстве.
- [ ] Зафиксировать численное обоснование выбора: разрыв между релевантным и нерелевантным докладом +0.316 (vs +0.068 у e5); Spearman с LLM-judge 0.327 (vs 0.240); Vendi Score пула после ABTT-1 = 63 эффективных distinct из 100 (vs 4 на raw e5).
- [ ] В формуле полезности участника (gap-блок) пометить, что `rel(u, t) = cos(ABTT-1(emb_u), ABTT-1(emb_t))`, где эмбеддинги предварительно нормированы.

### 1.2. Английский pipeline

- [ ] Зафиксировать, что основная программа конференции и пул синтетических персон переведены на английский для паритета каналов с LLM-агентским симулятором (`mobius_2025_autumn_en.json`, `personas_mobius_en.json`).
- [ ] Описать пул 100 синтетических EN-персон: 50 из spike + 50 догенерация через subagent Claude Sonnet с сохранением distribution (8/30/44/18 jr/mid/sr/lead; 18/44/27/11 startup/midsize/large/enterprise; 30 уникальных ролей; все 8 категорий preferred_topics Mobius). Internal-consistency check через LLM-judge: 50/50 consistent для новой половины.

### 1.3. LLM-агентский симулятор

- [ ] Зафиксировать bilingual templates LLMAgent: параметр `language` (`ru`/`en`), переключение на EN-промпты для паритета каналов на EN-пайплайне.
- [ ] Описать конкретные тексты EN-промптов (system + user template) либо отослать к `experiments/src/llm_agent.py`.

### 1.4. Скрипты и реализация

- [ ] Зафиксировать новые скрипты:
  - `experiments/scripts/embed_bge_abtt.py` — генерация эмбеддингов BGE + ABTT
  - `experiments/scripts/diagnose_mobius_personas_en_100.py` — диагностика пула 100
  - `experiments/scripts/smoke_ec_mobius_en.py` — smoke EC на реальной программе

---

## 2. Глава 4 — результаты

### 2.1. Per-policy distribution (раздел про средние по политикам)

- [ ] Обновить таблицу overload mean / median / p75 на новые числа:

  | Policy | overload mean | overload median | utility mean (центрировано) |
  |---|---:|---:|---:|
  | no_policy | 0.0399 | 0.0000 | −0.0055 |
  | cosine | 0.0405 | 0.0000 | +0.0011 |
  | capacity_aware | **0.0345** | 0.0000 | +0.0006 |

- [ ] Заметка про сдвиг utility-шкалы: на EN после ABTT cos в [−0.343, 0.732], среднее центрировано около нуля (vs RU где e5 cos был в [0.79, 0.93]). Это сдвиг шкалы, не интерпретации.

### 2.2. Pairwise win-rates (центральный численный результат)

- [ ] Обновить таблицу pairwise full-50 (`mean_overload_excess`, ε=0.005):

  | Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps |
  |---|---:|---:|---:|---:|---:|
  | no_policy vs cosine | 0.10 | 0.02 | 0.94 | 0.08 | 0.04 |
  | no_policy vs capacity_aware | 0.00 | 0.00 | 0.84 | 0.20 | 0.16 |
  | cosine vs capacity_aware | 0.00 | 0.00 | 0.86 | 0.20 | 0.14 |

- [ ] Переформулировать утверждение про центральный тезис: «cosine не выигрывает у capacity_aware ни на одной из 50 LHS-точек ни строго (0/50), ни за ε (0/50); обратное — capacity_aware строго лучше в 20%, лучше за ε в 14%».

### 2.3. Risk-positive subset

- [ ] Обновить численность: **11 / 50 (22%)** risk-positive (вместо 13/50 = 26% в RU).
- [ ] Обновить долю строгого снижения: **8 / 11 (73%)** (вместо 11/13 = 85% в RU).
- [ ] Обновить таблицу top reductions с новыми числами для LHS = 49, 26, 35, 18, 3, 0 (см. report §6).
- [ ] Три critical infeasible LHS (3, 35, 49) остались те же — это структурное свойство сетки.

### 2.4. Sensitivity по `capacity_multiplier` и `w_gossip`

- [ ] Обновить таблицы bucket × policy под новые числа (см. report §7).
- [ ] Нелинейный пик `w_gossip` в среднем bucket [0.25, 0.5) сохранился, `capacity_aware` лучший во всех трёх bucket'ах.

### 2.5. Program variant (Φ) — diagnostic only

- [ ] Обновить таблицу per-PV agg (см. report §8). PV=4 mean overload ≈ 0.171 (vs 0.176 в RU), sign-test p≈0.84 — гипотеза не отвергается.

### 2.6. Trade-off risk × utility

- [ ] Обновить: trade-off markers на full-50 = **3 / 150 (2.0%)** (vs 11/150 = 7.3% в RU). Trade-off на EN-пайплайне ещё реже — расширенный конус релевантности уменьшает дифференциал utility, оставляя политики различимыми только по риску.

### 2.7. Volatile points (стабильность)

- [ ] Обновить: **72 entries** (vs 24 в RU). Комментарий: повышение объясняется тем, что после ABTT softmax менее концентрирован, увеличивается стохастичность выбора при равной utility. Все volatile entries — режимы редких событий перегрузки, не влияют на центральные выводы.

### 2.8. Cross-validation (V) — главное усиление

- [ ] Обновить таблицу cross-validation:

  | Метрика | n_LHS_in_ρ | EN median ρ | passed |
  |---|---:|---:|---|
  | `hall_utilization_variance` | 12 | **0.80** | **PASS** (RU: 0.40 FAIL) |
  | `mean_user_utility` | 12 | 0.67 | PASS |
  | `overflow_rate_slothall` | 2 | 0.41 | FAIL (узкая выборка) |
  | `mean_overload_excess` | 2 | 0.15 | FAIL (узкая выборка) |

- [ ] **Overall median ρ = 0.769** (vs 0.554 в RU) — главный текстовый абзац: «паритет каналов между параметриком и LLM на едином EN-пайплайне даёт более согласованную картину, чем смесь языков RU-прогона. Метрика `hall_utilization_variance` перешла из формального проваленного состояния в пройденное на полной выборке 12 из 12 LHS — это методический результат, не просто арифметическое улучшение ρ».
- [ ] Top-1 match non-degenerate: overload 2/2 (100%), hall_var 9/12 (75%) — добавить в текст.

### 2.9. Стоимость LLM-прогона

- [ ] Обновить: V cost = **$10.22** (vs $11.55 в RU, тот же gpt-5.4-nano); cap $20, под бюджетом 2×. Q LLMRanker cost = $0.29 (gpt-4o-mini, тёплый кэш в V → 100% hit, $0 в V).
- [ ] Wallclock V: 2 ч 23 мин (vs 1 ч 46 мин в RU — `parallel-lhs=2` вместо 4 из соображений безопасности на macOS asyncio + httpx 0.27, см. инциденты 08.05).

---

## 3. Аннотация

- [ ] Обновить конкретные числа: 486 evals параметрика, 48 evals V, 44 160 LLMAgent-вызовов, median ρ = 0.77, capacity_aware строго лучше cosine в 20% / за ε в 14% / risk-positive 11/50 / на нём 8/11 (73%) snip снижает.
- [ ] Упомянуть BGE-large-en + ABTT-1 как функцию релевантности.

---

## 4. Введение и заключение

- [ ] Упомянуть паритет каналов между двумя симуляторами как методический вклад (а не только реализационный).
- [ ] Зафиксировать `hall_utilization_variance` FAIL → PASS как ключевое наблюдение по согласованности.

---

## 5. Список литературы

- [ ] Добавить: Mu, Bhat, Viswanath (ICLR 2018) — «All-but-the-Top: Simple and Effective Postprocessing for Word Representations». Источник ABTT.
- [ ] Добавить: Xiao et al. — описание `BAAI/bge-large-en-v1.5` (если есть формальная цитата; иначе — HuggingFace model card как secondary ref).
- [ ] Уже есть: Friedman & Dieng (2022) — Vendi Score. Проверить, что цитируется в §диагностики пула.

---

## 6. Приложения

- [ ] Опционально: вставить `spike_relevance_function_audit.md` как приложение C (методика выбора функции релевантности).
- [ ] Опционально: распределения по структурным полям пула 100 (`experiments/results/en/...` или вывод `diagnose_mobius_personas_en_100.py`).

---

## 7. Технические ссылки в тексте

- [ ] `experiments/src/llm_agent.py` — bilingual templates (RU/EN), параметр `language`.
- [ ] `experiments/scripts/embed_bge_abtt.py` — генерация эмбеддингов.
- [ ] `experiments/scripts/run_llm_lhs_subset.py` — флаг `--language`.
- [ ] `experiments/results/report_mobius_2025_autumn_en_full.md` — финальный отчёт перегона.
- [ ] `experiments/results/en/` — все analysis_*.json + V + cross-validation.

---

## 8. Что НЕ менять

- Раздел 16 PROJECT_OVERVIEW «Что НЕ входит в защиту» — стоп-лист тезисов сохраняется (B1, cross-domain Spearman, CoSPLib как attendance, MMR как победитель, jug-rec-sys как защищаемый результат). Эти тезисы не возвращаются.
- Архитектурные решения 1–7 (§14 PROJECT_OVERVIEW) не меняются: capacity вне utility, двусимуляторная архитектура с паритетом каналов, LHS + CRN, симплекс на весах, gossip per-talk счётчик, Φ как ось не оптимизатор, EC как блокирующий фильтр.
- Постановка задачи (§3 PROJECT_OVERVIEW) — не меняется.

---

## 9. Чек-лист самоконтроля перед финальной сдачей

- [ ] Все таблицы в гл. 4 пересчитаны на EN-числа.
- [ ] Аннотация согласована с гл. 4 по конкретным числам.
- [ ] §16 (что НЕ входит в защиту) сохранён без расширения.
- [ ] PDF собран, оригинальность ≥ 80% (повторный антиплагиат для финальной версии).
- [ ] Ссылки на новые артефакты живые (проверить относительные пути).
- [ ] Список литературы дополнен (Mu et al. 2018 как минимум).
