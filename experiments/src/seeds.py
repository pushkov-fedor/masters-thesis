"""CRN-контракт для LHS-прогонов (этап P).

См. `docs/spikes/spike_experiment_protocol.md` §8 (CRN strategy) и
Accepted decision блок. Контракт:

- `audience_seed` зависит ТОЛЬКО от `lhs_row_id`;
- `phi_seed` зависит ТОЛЬКО от `lhs_row_id`;
- `cfg_seed` зависит ТОЛЬКО от `replicate`;
- между политиками внутри одной (lhs_row_id, replicate) пары — одинаковая
  аудитория и одинаковый program_variant-эффект.

Это формальная реализация common random numbers (CRN) на уровне LHS-точки:
все 4 политики получают идентичные unit-cube координаты эксперимента,
варьируется только `cfg.seed`, который изолирован через `choice_rng`/`policy_rng`
в `_process_one_slot`.
"""
from __future__ import annotations

from typing import Dict


def derive_seeds(lhs_row_id: int, replicate: int) -> Dict[str, int]:
    """Производит CRN-сиды для LHS-точки и реплики.

    Параметры
    ---------
    lhs_row_id : int >= 0
        Индекс LHS-точки в плане эксперимента.
    replicate : int >= 1
        Номер seed-реплики (1, 2, 3 в основной матрице PROJECT_DESIGN §11).

    Возвращает
    ----------
    dict с тремя ключами:
        ``audience_seed`` — RNG-seed для отбора подмножества аудитории.
            Фиксирован по `lhs_row_id` ⇒ одинаковая аудитория во всех
            политиках и seed-репликах внутри LHS-точки.
        ``phi_seed`` — RNG-seed для оператора Φ (выбор `program_variant`).
            Фиксирован по `lhs_row_id` ⇒ одинаковый program_variant-эффект
            во всех политиках и seed-репликах внутри LHS-точки.
        ``cfg_seed`` — `cfg.seed` для `_process_one_slot`. Равен
            `replicate` ⇒ варьируется между репликами; изолирован через
            `choice_rng`/`policy_rng` в ядре, не сдвигает audience и phi.
    """
    if lhs_row_id < 0:
        raise ValueError(f"lhs_row_id must be >= 0, got {lhs_row_id}")
    if replicate < 1:
        raise ValueError(f"replicate must be >= 1, got {replicate}")
    return {
        "audience_seed": lhs_row_id * 1_000_003,
        "phi_seed":      lhs_row_id * 1_000_003 + 17,
        "cfg_seed":      replicate,
    }
