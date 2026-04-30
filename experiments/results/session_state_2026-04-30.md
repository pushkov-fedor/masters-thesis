# Состояние проекта на 2026-04-30

Снимок для возврата в работу после паузы. Описывает: что готово, что нашли исследовательские агенты, какие развилки остаются, с чего начать в следующей сессии.

## Что готово в коде

| Компонент | Состояние | Где |
|---|---|---|
| 11 базовых политик (Random, Cosine, MMR, Capacity-aware, Capacity-aware MMR, DPP, Calibrated, Sequential, GNN, Constrained-PPO, LLM-ranker) | работает | `src/policies/` |
| 12-я политика — Constrained-PPO-v2 (multi-agent batch episode) | обучена 200K шагов, intergrated | `src/policies/ppo_v2_policy.py`, `scripts/train_ppo_v2.py` |
| Параметрический симулятор + multinomial logit choice | работает | `src/simulator.py` |
| LLM-агентный симулятор v2 (память, личность, усталость, social граф, рефлексия) | работает | `src/agents/agent_simulator_v2.py` |
| Multi-signal fame для Mobius и Demo Day | работает | `scripts/build_fame_score.py`, `data/conferences/*_fame.json` |
| Поддержка `user_compliance` и `w_fame` в SimConfig | работает | `src/simulator.py` |
| Sweep по compliance ∈ {0.3, 0.5, 0.7, 0.9, 1.0} | пройден, числа в `results/compliance_sweep.json` | `scripts/run_compliance_sweep.py` |
| Permutation test для H3 (Manski reflection) | сделан, social effect — артефакт | `scripts/test_hypotheses.py` |
| Cross-conference robustness (Mobius vs Demo Day) | ρ=0.99 интра-Mobius, ρ=0.68 кросс, p=0.042 | `scripts/cross_conference.py` |
| Inter-slot chat (MiroFish-вдохновение) | модуль реализован, НЕ интегрирован, НЕ smoke-tested | `src/agents/inter_slot_chat.py` |

## Что готово в текстах ВКР

- Введение, Глава 1, Глава 2 свёрстаны в LaTeX-шаблон (коммит 6490702)
- Главы 1-2 ещё не подавались — можно переписывать

## Главный научный результат (на текущий момент)

В среде с реалистичными переполнениями (fame + неполная compliance) простая Capacity-aware политика даёт в 5-11× меньше overflow, чем 10 более сложных методов (Cosine, MMR, DPP, Sequential, GNN, Calibrated, Constrained-PPO, Constrained-PPO-v2, LLM-ranker). Граница применимости — compliance < 0.5: при низкой compliance recsys теряет преимущество, потому что звёздные доклады собирают толпу независимо от подсказок.

| compliance | Capacity-aware OF | Cosine OF | Соотношение |
|---|---|---|---|
| 0.3 | 0.093 | 0.287 | 3× |
| 0.5 | 0.065 | 0.306 | 4.7× |
| 0.7 | 0.028 | 0.315 | 11× |
| 0.9 | 0.000 | 0.306 | ∞ |
| 1.0 | 0.000 | 0.315 | ∞ |

Multi-agent PPO v2: в **training среде** OF=0.0 (постановка congestion game корректна — обучаемая политика МОЖЕТ научиться не переполнять). В **полной симуляции** при compliance=0.7 даёт OF=0.32. Это содержательный sim-to-real gap.

## Главное признанное ограничение

Релевантность = cosine между эмбеддингами персона⊕доклад, либо обученная HistGB-модель на 12K LLM-оценок (Pearson r=0.79 на val). Это **не реальное удовлетворение пользователей**, а семантический proxy. Реальных данных о посещаемости JUG нет и не будет.

## Что нашли исследовательские агенты в эту сессию

### Поиск 1 — реальные датасеты по IT-конференциям

**Результат: публичных датасетов посещаемости IT-конференций не существует.**

Топ-работы по теме:
- **Tandfonline 2024** (Algorithms for IT-conference scheduling) — частный опрос JUG-подобного типа, данные не выложены
- **Vangerven 2022** — synthetic
- **Chakrabarti 2022** — synthetic
- **Vincent 2024** (RecSys workshop) — кейс с приватной статистикой

Вывод: сообщество conference scheduling работает на синтетике. Это **наша защита, а не наша слабость**.

### Поиск 2 — близкие домены с capacity-constraint

| Датасет | Размер | Близость к нашей задаче |
|---|---|---|
| **Meetup Dataset** (GitHub wuyuehit/Meetup-Recommendation-Dataset) | 4M пользователей, 2M событий, RSVPs | **Лучший cross-domain аналог.** Группы по интересам, capacity per event, реальные attendance |
| **Reviewer Assignment Gold Standard** (Stelmakh 2023) | ~600 reviewer-paper пар | **Близкий формальный аналог.** Reviewer-paper assignment с capacity per reviewer |
| Yelp Open Dataset | 6.9M reviews, 150K businesses | Рестораны как залы, посещения как RSVPs |
| Citi Bike NYC | публичные API | Реальные capacity по станциям |
| GoalZone Fitness | 1500 записей | Реальная capacity 15 или 25 мест |
| Google/Alibaba cluster traces | TB-scale | Строгие capacity-constraint, но другой домен |

### Поиск 3 — методологическая защита (главное открытие)

**arXiv:2504.03274 "Validation is the central challenge in LLM-based agent simulations"**:
- **15 из 35 топ-работ** по LLM-agent-simulators используют только subjective believability validation
- **22 из 35** используют subjective believability как primary метод

Это означает: моя методология — мейнстрим в поле, а не недостаток.

**arXiv:2601.17087 "Lost in Simulation"** — критика и рекомендации по валидации agent-based симуляторов.

Защитные стратегии, которые работают за 1-3 дня:
1. Ablation studies — **уже есть** (compliance sweep, w_fame sweep, capacity sweep)
2. Sensitivity analysis — **уже есть**
3. **Stylized facts replication** — не делал
4. **Mini human study** (10-20 человек на Google Forms) — не делал
5. **Cross-domain validation** на Meetup Dataset — не делал

## Развилка следующих шагов

### Вариант A — объективная валидация (3 дня, $0 LLM)
- День 1: `experiments/scripts/stylized_facts.py` — проверить, воспроизводит ли симулятор Pareto-attendance, time-of-day-эффект, track-affinity (по существующим логам)
- День 2: портировать 11 политик на Meetup Dataset, сравнить ranking (если ρ > 0.5 — выводы устойчивы кросс-доменно)
- День 3: Google Forms на 10-15 знакомых-разработчиков, face validity проверка

Защита: «применили 4 типа валидации (subjective + sensitivity + stylized facts + cross-domain), что выше медианы по полю».

### Вариант B — переписать Главы 1-2 под новую защитную позицию (2 дня)
- Внести цитирование arXiv:2504.03274, arXiv:2601.17087, Tandfonline 2024
- Позиционировать работу как multi-faceted validation framework
- Без варианта A — слабее, потому что нечем подкрепить «multi-faceted»

### Вариант C — полный MiroFish-style симулятор (3-5 дней, $50-100)
- Графовая база знаний агентов
- Агент-к-агенту посты с темпоральной диффузией
- Изучение https://github.com/666ghj/MiroFish

Высокий риск (может не сойтись или дать невнятные числа). Прирост к защите неочевиден на фоне варианта A.

## Рекомендация

A → B → C в указанном порядке. A даёт объективные числа за 3 дня без денег. B превращает работу в защищаемый текст с опорой на свежую (2024-2025) литературу. C опционально, если останется время после антиплагиата 08.05.

## Точки возобновления

1. Открыть этот файл и `experiments/results/preliminary_findings_v5.md`
2. Принять решение по варианту (A/B/C)
3. Если A → начать с `experiments/scripts/stylized_facts.py` (Pareto-attendance из существующих логов)
4. Если B → открыть `thesis/chapter1.tex`, `thesis/chapter2.tex`
5. Если C → читать https://github.com/666ghj/MiroFish, проектировать `agent_simulator_v3.py`

## Календарь

- **08.05.2026** — антиплагиат (нужен PDF на aitalents@itmo.ru, оригинальность ≥ 80%)
- **13.05.2026** — предзащита (7 мин + Q&A)
- **Лето 2026** — основная защита

Между сегодня (30.04) и 08.05 — 8 дней. Из них реалистично 5-6 рабочих. Вариант A+B укладывается в этот бюджет.
