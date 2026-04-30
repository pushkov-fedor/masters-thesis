#!/bin/bash
# Финальная сборка результатов после завершения state-aware прогона:
# 1. Сохранить state-aware (200 пользователей) в отдельный файл
# 2. Перезапустить основной 900-user прогон для восстановления results.json
# 3. Сгенерировать графики
set -e

cd "$(dirname "$0")/.."

# 1. backup state-aware results
if [ -f results/results.json ]; then
    cp results/results.json results/results_state_aware_200.json
    cp results/summary.md results/summary_state_aware_200.md
    echo "Backed up state-aware results to results_state_aware_200.{json,md}"
fi

# 2. re-run main 900-user experiment (cache full, ~6s)
.venv/bin/python scripts/run_experiments.py --personas personas_x3 --with-llm --llm-budget 0.5

# 3. regen plots
.venv/bin/python scripts/make_plots.py
