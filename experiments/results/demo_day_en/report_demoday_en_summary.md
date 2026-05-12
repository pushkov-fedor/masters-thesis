# Demo Day 2026 EN — параметрический Q (второй инстанс)

Дата прогона: 2026-05-12. Цель — воспроизвести центральный численный тезис Mobius EN на втором инстансе конференции (Demo Day ITMO 2026, IT-конференция широкого профиля). BGE-large-en + ABTT-1 + 150 EN-персон + 50 LHS × 3 replicate × {П1, П2, П3} = 450 evals.

## Данные

- Конференция: 210 talks, 7 halls, 56 slots, 2 дня (2026-01-22 / 23).
- Персон: 150 EN, distribution exp jr/mid/sr/lead = 12/45/66/27 (8/30/44/18%); company smallish-startup/midsize/large/enterprise = 27/63/42/18 (18/42/28/12%); 47 distinct roles.
- Эмбеддинги: BGE-large-en + ABTT-1 (1024-dim), fame.json скопирован из RU-версии (id совпадают).

## Acceptance персон (4 проверки)

- **Internal consistency (LLM-judge claude-haiku-4.5):** 148/150 (98.7%). Цель ≥ 95% — **PASS**.
- **Vendi Score (cos+ABTT):** 54.25 из 150 (36.2%). Raw без ABTT = 6.24 (узкий конус BGE); BM25 = 99.62 (66.4%). Парные ABTT cos: mean -0.001, max 0.785; дублей > 0.95: 0.
- **Coverage программы (cos+ABTT τ=0.50, 0.60):** dead docs 0/210 при τ=0.5; 1/210 при τ=0.6; mean interested per talk = 33.0 (τ=0.5).
- **EC smoke:** EC1 PASS (cap×3 → overload=0), EC3 PASS (w_rec=0 range=0), EC4 PASS (cap×0.5 различимы).

## Центральный численный тезис

Pairwise `cosine` vs `capacity_aware` по `mean_overload_excess` на full_50 (фракции от 50 valid пар):

- **strict wins cosine:** 0%
- **strict wins capacity_aware:** 34% (loss_strict для cosine)
- **ε-equivalent (ε=0.005):** 86%
- **eps wins cosine (cosine лучше за ε):** 0%
- **eps wins capacity_aware (cap_aware лучше за ε):** 14%

Per-policy mean(median) overload по full_50:
- `no_policy`: mean=0.0486, median=0.0000, max=1.1966
- `cosine`: mean=0.0484, median=0.0000, max=1.1937
- `capacity_aware`: mean=0.0399, median=0.0000, max=1.1568

## Сравнение с Mobius EN

| Метрика | Mobius EN | Demo Day EN |
|---|---:|---:|
| n LHS-точек | 50 | 50 |
| cosine strict wins | 0% | 0% |
| capacity_aware strict wins | 20% | 34% |
| eps cosine wins | 0% | 0% |
| eps cap_aware wins | 14% | 14% |
| ε-equivalent | 86% | 86% |

**Тезис о том, что `cosine` не выигрывает у `capacity_aware`:**
- Demo Day: **подтверждён**. 0 strict побед cosine, 0 ε-побед cosine.

## Бюджеты

- Wallclock LHS-прогон: 907.9s (15.1 мин); из них prep (enumerate_modifications deepcopy для program_variant>0) — 886s, sim P1-P3 — 22.3s.
- LLM-вызовы: translation 210 talks → $0.23 (claude-haiku-4.5, 38с), persona generation 150 → $0.21 (claude-haiku-4.5, 58с), consistency audit 150 → $0.08 (claude-haiku-4.5, 25с). Total ≈ $0.52.

## Артефакты

- `data/conferences/demo_day_2026_en.json` — переведённая программа.
- `data/conferences/demo_day_2026_en_embeddings.npz` — BGE+ABTT эмбеддинги talks.
- `data/conferences/demo_day_2026_en_fame.json` — fame scores (skipped из RU-копии).
- `data/personas/personas_demoday_en.json` — 150 EN-персон.
- `data/personas/personas_demoday_en_embeddings.npz` — BGE+ABTT эмбеддинги personas.
- `data/personas/test_diversity/internal_consistency_demoday.json` — LLM-judge consistency audit.
- `data/personas/test_diversity/diagnose_demoday_en.json` — Vendi / coverage / distributions.
- `results/lhs_parametric_demo_day_2026_en_2026-05-12.{json,csv,md}` — параметрический Q-прогон.
- `results/demo_day_en/analysis_*.json` — постобработка.
- `results/demo_day_en/analysis_lhs_parametric_2026-05-12_demoday_en.md` — markdown отчёт постобработки.
