---
name: Cross-validation на стратифицированной выборке (2026-05-12/13)
description: Cross-val Q-V на 12 стратифицированных по силе перегрузки точках Mobius simplified. Overall median ρ=0.5 (PASS на пороге, заметно слабее основного EN-прогона 0.77). В основную речь защиты 13.05 не выносится — держим для Q&A.
type: project
originSessionId: 2010cee2-1a6b-460e-a9b1-a452d64b57fc
---
**Что.** Перекрёстная проверка двух симуляторов (параметрик ↔ LLM) на стратифицированной по силе перегрузки выборке точек Mobius simplified.

**Когда.** Q-прогон 12.05.2026, V-прогон стартовал 21:26 12.05, завершён 01:10 13.05. Cross-val выполнен в ~02:00 13.05.

**Зачем.** В основном EN-прогоне maximin-выборка отбирала точки равномерно по параметрам — в результате 10 из 12 точек попадали в безопасную зону (overload=0 у всех политик), и cross-validation по метрикам перегрузки опирался всего на 2 невырожденные точки. Стратификация по силе перегрузки была разработана как замена maximin для исследования согласованности симуляторов на сценариях разной интенсивности риска.

**Параметры simplified:**
- 3 оси LHS: `capacity_multiplier ∈ [0.5, 1.5]` (узкий стрессовый диапазон), `w_rec`, `w_gossip` с симплексом.
- Зафиксированы: `audience_size=100`, `popularity_source=cosine_only`, `program_variant=0`.
- 50 LHS точек × 3 политики (без llm_ranker) × 3 replicate = 450 evals параметрика.
- Stratified subset: 12 точек разнесены по 4 корзинам `mean_overload_excess` (3 safe / 3 light / 3 moderate / 3 severe).
- V на 12 stratified × 3 политики × 1 seed = 36 evals; wallclock 3:43, cost $13.23, 57 600 LLM calls.

**Результаты cross-validation:**
- `mean_user_utility`: median ρ=1.000 на 12/12 LHS, top-1 match 10/12 — сильное согласование.
- `hall_utilization_variance`: median ρ=0.500 на 12/12, top-1 match 11/12 — умеренное.
- `mean_overload_excess`: median ρ=0.500 на 8/12 (4 точки degenerate, параметрик не разделяет политики), top-1 match 6/8 на невырожденных. Главное достижение стратификации — охват overload-метрик вырос с 2/12 (maximin) до 8/12.
- `overflow_rate_slothall`: median ρ=0.000 на 8/12 — **FAIL** по этой метрике. LLM и параметрик ставят политики в разных порядках по «частоте слотов с переполнением», без видимой системы.
- **Overall median ρ = 0.500** — PASS точно на пороге (vs основной EN 0.77).

**Интерпретация (для финальной версии текста ВКР):**
- Главное содержательное направление подтверждается: на 6 из 8 невырожденных точек оба симулятора согласны, что cap_aware на первом месте по силе переполнения.
- Численное согласие на сложных точках слабее основного прогона. Возможная причина: стратификация выбирает точки с тонкими различиями между политиками, на которых LLM шумнее формулы.
- По частоте переполнения симуляторы расходятся. Гипотеза: метрика дискретная (порог «слот переполнен» — да/нет), и LLM-шум вокруг порога ломает корреляцию.

**На защите 13.05.2026 — не выносится в основную речь.** В Q&A — на случай вопроса «проверяли ли на сценариях с реальной перегрузкой» — ответ: «да, на стратифицированной выборке главный численный паттерн подтверждается (cap_aware top-1 в 6 из 8 невырожденных точек), общая медианная согласованность 0.5 на пороге PASS, численное согласие на сложных точках слабее основного из-за тонких различий между политиками».

**Артефакты:**
- Q simplified: `experiments/results/lhs_parametric_simplified_2026-05-12_mobius_2025_autumn_en.{json,csv,md}`
- Stratified subset: `experiments/results/lhs_parametric_simplified_2026-05-12_mobius_2025_autumn_en_stratified_subset.json` (`selected_lhs_row_ids = [2, 33, 34, 0, 6, 12, 17, 31, 39, 1, 14, 36]`)
- V: `experiments/results/llm_agents_simplified_stratified_2026-05-12.{json,csv,md,partial.jsonl}`
- Cross-val: `experiments/results/en_simplified/analysis_cross_validation_simplified_2026-05-12.{json,md}`
- Helper (ranking_vectors): `experiments/results/en_simplified/analysis_simplified_diagnostic.json`
- Plot: `experiments/results/en_simplified/plots/cross_validation_rho_per_metric.png`

**Where to apply at thesis correction.** В разделе «Эксперимент / Перекрёстная проверка двух симуляторов» главы 4 ВКР упомянуть оба прогона: основной EN (12 maximin × 4 политики, overall ρ=0.77) как защитный, и стратифицированный (12 stratified × 3 политики, overall ρ=0.5) как диагностический по широкому спектру сценариев риска. Подчеркнуть: расхождение между прогонами — методический результат (стратификация выявляет точки, где LLM шумнее).
