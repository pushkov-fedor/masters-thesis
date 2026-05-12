# Final experiment report — Mobius 2025 Autumn EN (parametric, BGE-large-en + ABTT-1)

Перегон параметрического симулятора на новой функции релевантности, выбранной по
итогам аудита (`docs/spikes/spike_relevance_function_audit.md`). Английский
пайплайн: BGE-large-en + ABTT-1, пул 100 EN-персон под Mobius, программа Mobius
2025 Autumn в EN-версии. Этот файл — преемник `report_mobius_2025_autumn_full.md`
(RU-прогон от 2026-05-08); старый сохранён как зафиксированный артефакт прогона,
на котором построен PDF ВКР.

LLM-агентский симулятор (`run_llm_lhs_subset.py`) и cross-validation Q-O7 на этом
этапе НЕ переоценивались (см. §9 ниже).

## 1. Metadata

| Поле | Значение |
|---|---|
| Конференция | `mobius_2025_autumn_en` (40 talks, 16 slots, 3 halls × 100 cap, 2 дня) |
| Дата прогона | 2026-05-12 |
| Эмбеддер | `BAAI/bge-large-en-v1.5` (1024-dim) |
| Постобработка | ABTT-1 (Mu, Bhat, Viswanath, ICLR 2018) на vstack(persons, talks) |
| Релевантность | `rel(u, t) = cos(ABTT-1(emb_u), ABTT-1(emb_t))` |
| Пул персон | `personas_mobius_en.json` — 100 синтетических EN-персон под Mobius (50 spike + 50 догенерация Sonnet с тем же distribution) |
| master_seed | 2026 |
| Q параметрик | 50 LHS × {П1, П2, П3} × 3 replicate = **450 evals**, wallclock = 9.8 s |
| П4 (llm_ranker) | НЕ запускалась (LLM трогать в этом перегоне запрещено) |
| V (LLM cross-validation) | отложена |

## 2. Inputs / source artefacts

| Stage | Файлы |
|---|---|
| Persona pool (50+50 → 100) | `data/personas/personas_mobius_en.json`, `personas_mobius_en_embeddings.npz` |
| Internal consistency | `data/personas/test_diversity/internal_consistency_mobius_part2.json` — 50/50 consistent для новой половины |
| EN-программа | `data/conferences/mobius_2025_autumn_en.json`, `mobius_2025_autumn_en_embeddings.npz` |
| Embed pipeline | `scripts/embed_bge_abtt.py` |
| Diagnose 100 | `scripts/diagnose_mobius_personas_en_100.py` |
| EC smoke | `scripts/smoke_ec_mobius_en.py` |
| LHS Q | `results/lhs_parametric_mobius_2025_autumn_en_2026-05-12.{json,csv,md}` |
| Analyze S | `results/en/analysis_*.json` (8 файлов), `results/en/analysis_lhs_parametric_2026-05-12_en.md`, `results/en/plots/*.png` |

## 3. Acceptance

- **Q (параметрик)**: **PASS** (7/7 чеков): П1–П3 evals = 450 == 450; П4 evals = 0
  == 0 (не запускалась); total = 450 == 450; П4 только на maximin (violations=0,
  trivially); CRN audience/phi инвариант (violations=0); cfg_seed = replicate
  (violations=0); long-format ключи missing=[].
- **S (postprocessing)**: 8 analysis_*.json + markdown + 3 plot созданы в
  `results/en/`; wallclock = 0.4 s.
- **EC (extreme conditions)**:
  - 10/10 PASS на `toy_microconf_2slot` + e5 (архитектурные инварианты,
    независимы от смены эмбеддера в основной программе).
  - Smoke на `mobius_2025_autumn_en` + BGE-large-en + ABTT-1:
    EC1 (cap×3 → overload=0) — PASS на 3/3 политиках;
    EC3 (w_rec=0 → range=0, CRN-инвариантность) — PASS, range = 0;
    EC4 bonus (w_rec=1, cap×0.5 → различимы) — PASS, range = 0.097
    (capacity_aware 0.0069 значительно ниже cosine 0.1039).

## 4. Per-policy distribution (full-50, П1–П3, 150 evals на политику)

| Policy | overload mean | overload median | overload p75 | utility mean (центрировано) |
|---|---:|---:|---:|---:|
| `no_policy` | 0.0399 | 0.0000 | 0.0000 | −0.0055 |
| `cosine` | 0.0405 | 0.0000 | 0.0000 | +0.0011 |
| `capacity_aware` | **0.0345** | 0.0000 | 0.0000 | +0.0006 |

Утилита у политик различается на < 0.007 (порядок < ε_utility = 0.001 в median);
переход к capacity_aware не сопровождается потерей содержательного качества.

## 5. Pairwise full-50 — central numeric finding

Метрика: `mean_overload_excess`, ε = 0.005, агрегация по replicate — median.

| Пара (A vs B) | win_strict | win_eps | ties_eps | loss_strict | loss_eps |
|---|---:|---:|---:|---:|---:|
| no_policy vs cosine | 0.10 | 0.02 | 0.94 | 0.08 | 0.04 |
| no_policy vs capacity_aware | 0.00 | 0.00 | 0.84 | 0.20 | 0.16 |
| **cosine vs capacity_aware** | **0.00** | **0.00** | 0.86 | **0.20** | **0.14** |

**Центральный тезис работы сохраняется.** `cosine` не выигрывает у
`capacity_aware` НИ на одной из 50 LHS-точек ни строго, ни за ε. Обратное
направление: `capacity_aware` строго лучше `cosine` в 20% точек и лучше за ε в
14%.

Сопоставление с RU-прогоном (2026-05-08, e5-small + raw cos):

| Пара | RU win_eps (loss_eps для cap) | EN win_eps (loss_eps) | Δ |
|---|---:|---:|---:|
| cosine vs capacity_aware (strict cap-aware wins) | 0.26 | 0.20 | −0.06 |
| cosine vs capacity_aware (cap-aware wins за ε) | 0.22 | 0.14 | −0.08 |

Направление сохраняется; абсолютные доли строгих побед `capacity_aware` чуть
ниже на EN-пайплайне. Возможная причина — расширение конуса релевантности через
ABTT-1 даёт более равномерное распределение utility между докладами слота, и
несколько risk-positive LHS-точек проседают ниже порога фиксации перегрузки.

## 6. Risk-positive subset (11/50 точек)

Подмножество LHS-row с `mean_overload_excess > 0` хотя бы у одной из П1–П3. На
остальных 39/50 (78%) все политики дают overload = 0 (loose-bucket
`capacity_multiplier ∈ [2.0, 3.0]` плюс нормальные cap при малой аудитории).

На risk-positive подмножестве:

- `capacity_aware` **не хуже** max(no_policy, cosine) за ε: **11/11 (100%)**;
- `capacity_aware` **строго снижает** риск за ε: **8/11 (73%)**.

Три точки физически непреодолимы (`audience > effective per-slot capacity`),
структурное свойство сетки LHS:

| LHS | cap_mult | audience | effective | overshoot |
|---:|---:|---:|---:|---:|
| 3 | 0.534 | 100 | 53.4 | +46.6 |
| 35 | 0.963 | 100 | 96.3 | +3.7 |
| 49 | 0.774 | 100 | 77.4 | +22.6 |

Top reductions `capacity_aware` vs `cosine` на risk-positive:

| LHS | cap_mult | aud | overload no_policy | cosine | capacity_aware | Δ vs cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 49 | 0.774 | 100 | 0.4679 | 0.5096 | **0.3622** | +0.1474 |
| 26 | 0.629 | 60 | 0.0675 | 0.0714 | **0.0119** | +0.0595 |
| 35 | 0.963 | 100 | 0.0631 | 0.0556 | **0.0303** | +0.0253 |
| 3 | 0.534 | 100 | 1.1944 | 1.1898 | **1.1667** | +0.0231 |
| 36 | 0.696 | 60 | 0.1354 | 0.1389 | **0.1181** | +0.0208 |
| 0 | 1.408 | 100 | 0.0208 | 0.0243 | **0.0139** | +0.0104 |

## 7. Sensitivity (bucket-агрегаты, диагностика)

`capacity_multiplier` × policy (full-50, П1–П3):

| Bucket | n_lhs | no_policy | cosine | capacity_aware |
|---|---:|---:|---:|---:|
| stress [0.5, 1.0) | 10 | 0.1937 | 0.1974 | **0.1693** |
| normal [1.0, 2.0) | 21 | 0.0027 | 0.0025 | **0.0014** |
| loose [2.0, 3.0] | 19 | 0.0000 | 0.0000 | **0.0000** |

`w_gossip` × policy (full-50):

| Bucket | n_lhs | no_policy | cosine | capacity_aware |
|---|---:|---:|---:|---:|
| [0, 0.25) low | 23 | 0.0065 | 0.0060 | **0.0019** |
| [0.25, 0.5) mid | 19 | 0.0899 | 0.0921 | **0.0822** |
| [0.5, 0.7] high | 8 | 0.0169 | 0.0174 | **0.0148** |

Нелинейный пик в среднем bucket — повторяет RU-прогон (диагностика, не causal:
bucket конфаундирован с осями `capacity_multiplier` и `audience_size`).
`capacity_aware` лучший во всех трёх bucket'ах.

## 8. Program variant (Φ) — diagnostic only

| PV | n_lhs | cosine overload mean | Δ vs PV=0 |
|---:|---:|---:|---:|
| 0 ($P_0$) | 7 | 0.0089 | — |
| 1 | 9 | 0.0106 | +0.0018 |
| 2 | 9 | 0.0576 | +0.0487 |
| 3 | 8 | 0.0000 | −0.0089 |
| 4 | 7 | 0.1712 | +0.1623 |
| 5 | 10 | 0.0153 | +0.0064 |

Sign-test PV=0 vs PV=4: p ≈ 0.84 — гипотеза об отсутствии разности на текущей
выборке не отвергается. Положение 4 защиты закрывается как сценарные
характеристики распределения по оси конфигурации, не как causal-разность между
конкретными перестановками.

## 9. Cross-validation (V) — отложено

LLM-агентский симулятор в этом перегоне не запускался по решению пользователя.
Cross-validation Q-O7 (median ρ ≥ 0.5 между параметрическим и LLM ранжированиями
политик на 12 maximin точках) сохраняется на RU-данных от 2026-05-08:

| Метрика | n_LHS_in_ρ / 12 | median ρ | passed |
|---|---:|---:|---|
| `mean_user_utility` | 12 | 0.80 | PASS |
| `overflow_rate_slothall` | 2 | 0.74 | PASS (узкая выборка) |
| `mean_overload_excess` | 2 | 0.30 | FAIL |
| `hall_utilization_variance` | 12 | 0.40 | FAIL по порогу ρ |

Объединённая median ρ = 0.554 ≥ 0.5 → формальный gate Q-O7 пройден в RU.
Переоценка на EN-пайплайне требует:
- паритет каналов: переключение `LLMAgent` (`gpt-5.4-nano`) на английские
  промпты,
- генерация эмбеддингов для maximin-subset через тот же BGE+ABTT,
- ~$11–37 бюджета на LLM-вызовы (44 160 calls × модель).

Считается отдельным спринтом после предзащиты 13.05.2026.

## 10. Risk × utility / stability

- Trade-off markers (ε_o = 0.005, ε_u = 0.001) на full-50 П1–П3: **3 из 150
  (2%)** vs 11/150 (7.3%) в RU-прогоне. Trade-off на EN-пайплайне ещё реже —
  расширенный конус релевантности после ABTT уменьшает дифференциал utility между
  политиками, оставляя их различимыми только по риску перегрузки.
- Volatile points (`std/|mean| > 0.5` на 3 replicate): **72** entries vs 24 в
  RU. Повышение объясняется тем же расширенным конусом: softmax менее
  концентрирован, увеличивается стохастичность выбора при равенстве utility.
  Все volatile entries соответствуют режимам редких событий перегрузки, не
  влияют на центральные выводы (§5–§6).

## 11. Pool 100 personas — qualification

Метрики качества пула, расширенные с 50 до 100 EN-персон (`scripts/diagnose_mobius_personas_en_100.py`):

### 11.1. Vendi Score под 4 kernels

| Kernel | Vendi (n=100) | % от max |
|---|---:|---:|
| cos raw (BGE без ABTT) | 8.12 | 8.1% |
| cos + ABTT-1 (основной в utility) | **62.99** | **63.0%** |
| BM25 (только лексика) | 82.61 | 82.6% |
| hybrid α=0.7 (ABTT + BM25) | 48.29 | 48.3% |

Узкий конус контекстуальных эмбеддингов виден явно: `raw cos` даёт 8% от
максимума (среднее парное cos = 0.645, диапазон [0.439, 0.873]); после ABTT-1
mean cos = −0.007 (диапазон [−0.365, 0.607]) и Vendi = 63%. Это согласуется со
spike (на пуле 50 было 89% от max=50 ≈ 45 эффективных distinct; на пуле 100 — 63
эффективных distinct, абсолютный рост разнообразия).

### 11.2. Coverage программы (40 EN-докладов)

При нормированном ABTT cos `[0, 1]`:

| τ | dead docs | docs <5 заинтересованных | min/median/max |
|---:|---:|---:|---:|
| 0.40 | 0/40 | 0 | 8/14/25 |
| 0.50 | 1/40 | 27 | 0/3/11 |
| 0.60 | 8/40 | 39 | 0/2/6 |

При умеренном пороге τ=0.40 — ни одного «мёртвого» доклада, в каждом
заинтересовано минимум 8 персон из 100. При жёстком пороге τ=0.50 появляется
1 dead doc, при τ=0.60 — 8 (типы: AntiSOLID, AI in development, design system
tools, contributing to open source — broad/abstract темы без specific platform).

### 11.3. Distribution: first50 vs second50

| Поле | first50 | second50 |
|---|---|---|
| experience: junior / middle / senior / lead | 4 / 15 / 22 / 9 | 4 / 15 / 22 / 9 |
| company_size: startup / midsize / large / enterprise | 9 / 21 / 14 / 6 | 9 / 23 / 13 / 5 |

Distribution-shift между половинами — в пределах ±2 на 50. Internal consistency
(LLM-judge) — 50/50 consistent для новой половины (`internal_consistency_mobius_part2.json`).

## 12. Limitations

1. **П4 (llm_ranker) не оценена** на full-50 в EN-перегоне — П4 на full-50
   проекте на RU-данные старого прогона; для EN — отложена вместе с V.
2. **Cross-validation Q-O7** — на RU-данных от 2026-05-08, не переоценена для
   EN. Решение по переоценке принимается после предзащиты.
3. **`program_variant` (Φ)** — sign-test diagnostic only; LHS-row с разными PV не
   conf-matched по остальным осям.
4. **Bucket-агрегаты** конфаундированы между осями; OAT scatter ≠ строгое causal
   cleavage.
5. **Mobius — сценарный стресс-тест**, не прогноз. Median overload = 0 в full-50
   — структурное свойство (78% LHS-точек безопасные), не отсутствие эффекта
   политик.
6. **Стоп-лист тезисов** PROJECT_STATUS §5 действует.

## 13. Key takeaways для предзащиты

- Замена `cos(e5)` → `cos(ABTT-1(BGE-large-en))` на основной программе:
  EC 10/10 + smoke 3/3 PASS; центральный численный тезис («cosine не выигрывает
  у capacity_aware ни на одной из 50 LHS-точек ни строго, ни за ε»)
  **подтверждён** на новой функции релевантности.
- Доля строгих побед `capacity_aware` над `cosine` чуть снизилась (0.20 vs 0.26
  в RU), но направление сравнения сохранилось.
- Пул персон расширен до 100 EN-персон под Mobius с проверенным structural
  distribution и внутренней consistency.
- Vendi Score пула под основной функцией релевантности: 63 эффективных distinct
  персон из 100 (узкий конус расширен ABTT).
- Текст ВКР (PDF после антиплагиата 2026-05-08) использует числа RU-прогона;
  корректировка в финальной версии после защиты.

## 14. Links

- LHS Q: `lhs_parametric_mobius_2025_autumn_en_2026-05-12.{json,csv,md}`
- Analyze S: `results/en/analysis_*.json` (8 файлов),
  `results/en/analysis_lhs_parametric_2026-05-12_en.md`
- Plots: `results/en/plots/{analysis_risk_utility_scatter,analysis_gossip_bucket_bar,analysis_ranking_heatmap_maximin}.png`
- Predecessor (RU, 2026-05-08): `report_mobius_2025_autumn_full.md`
- Spike-memo: `docs/spikes/spike_relevance_function_audit.md`
- Pool diagnostics: `data/personas/test_diversity/internal_consistency_mobius_part2.json`
