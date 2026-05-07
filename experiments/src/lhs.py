"""LHS-генератор и maximin-subset для этапа P (PIVOT_IMPLEMENTATION_PLAN r5).

Реализация строго по принятому memo O (`docs/spikes/spike_experiment_protocol.md`),
с учётом 5 технических уточнений Accepted decision 2026-05-07.

Шесть осей:
    1. capacity_multiplier ∈ [0.5, 3.0]                continuous
    2. popularity_source ∈ {cosine_only, fame_only, mixed}  categorical
    3. w_rec ∈ [0, 0.7]                                 continuous (симплекс)
    4. w_gossip ∈ [0, 0.7]                              continuous (симплекс)
    5. audience_size ∈ {30, 60, 100}                    discrete
    6. program_variant ∈ {0, 1, ..., 5}                 discrete

Симплексная нормировка: w_rel + w_rec + w_gossip = 1 (Accepted Q-J4).
Точки с w_rec + w_gossip > 1 отбрасываются rejection sampling'ом.

После rejection — repair дискретных осей при дисбалансе.

Также:
    maximin_subset(rows, k=12, force_program_variant_zero=True) — отбор k
    LHS-точек по greedy maximin distance в unit-cube; принудительное
    включение хотя бы одной точки с program_variant=0 (Q-O4 accepted).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import qmc


# ---------- Константы каталога осей ----------

LHS_AXES = (
    "capacity_multiplier",
    "popularity_source",
    "w_rec",
    "w_gossip",
    "audience_size",
    "program_variant",
)

POPULARITY_SOURCES = ("cosine_only", "fame_only", "mixed")
AUDIENCE_SIZES = (30, 60, 100)
PROGRAM_VARIANT_LEVELS = (0, 1, 2, 3, 4, 5)

# Минимальное покрытие дискретных осей после симплекс-фильтра
# (Accepted decision уточнение 4).
DEFAULT_MIN_PER_LEVEL: Dict[str, int] = {
    "program_variant": 5,    # 50/6 ≈ 8 ожидаемо
    "audience_size":   12,   # 50/3 ≈ 17 ожидаемо
    "popularity_source": 12,
}


# ---------- Маппинг unit-cube → реальные значения ----------

def _map_unit_to_row(u: np.ndarray) -> Optional[Dict]:
    """Маппинг одной unit-cube точки [0,1)^6 в реальные значения осей.

    Возвращает None, если точка нарушает симплекс w_rec + w_gossip ≤ 1
    (rejection sampling).
    """
    cap_mult = 0.5 + float(u[0]) * 2.5            # [0.5, 3.0]
    pop_src = POPULARITY_SOURCES[min(2, int(u[1] * 3))]
    w_rec = float(u[2]) * 0.7                     # [0, 0.7)
    w_gossip = float(u[3]) * 0.7                  # [0, 0.7)
    if w_rec + w_gossip > 1.0:
        return None
    aud_size = AUDIENCE_SIZES[min(2, int(u[4] * 3))]
    prog_idx = PROGRAM_VARIANT_LEVELS[min(5, int(u[5] * 6))]
    w_rel = max(0.0, 1.0 - w_rec - w_gossip)
    return {
        "u_raw": [float(x) for x in u],
        "capacity_multiplier": cap_mult,
        "popularity_source": pop_src,
        "w_rec": w_rec,
        "w_gossip": w_gossip,
        "w_rel": w_rel,
        "audience_size": aud_size,
        "program_variant": prog_idx,
    }


# ---------- Контроль баланса дискретных осей ----------

def _check_balance(rows: List[Dict]) -> Dict[str, Dict]:
    """Возвращает counter покрытия дискретных осей."""
    return {
        "program_variant": {
            lv: sum(1 for r in rows if r["program_variant"] == lv)
            for lv in PROGRAM_VARIANT_LEVELS
        },
        "audience_size": {
            lv: sum(1 for r in rows if r["audience_size"] == lv)
            for lv in AUDIENCE_SIZES
        },
        "popularity_source": {
            lv: sum(1 for r in rows if r["popularity_source"] == lv)
            for lv in POPULARITY_SOURCES
        },
    }


def _is_balanced(
    counts: Dict[str, Dict],
    min_per_level: Dict[str, int],
    n_points: int,
) -> bool:
    """Проверяет покрытие дискретных осей.

    Если для какой-то оси баланс физически недостижим
    (`n_levels × min_per_level[axis] > n_points`), эта ось пропускается
    как не-проверяемая — иначе repair обречён на бесконечный цикл (например,
    n=5 точек и 6 уровней `program_variant` при min=1 — невозможно покрыть
    все 6 одновременно). Этот случай возникает при использовании
    `generate_lhs` с малым n для smoke / unit-тестов.
    """
    for axis in min_per_level:
        n_levels = len(counts[axis])
        if n_levels * min_per_level[axis] > n_points:
            # Не проверяемое условие — баланс физически недостижим
            continue
        for v in counts[axis].values():
            if v < min_per_level[axis]:
                return False
    return True


def _force_level_row(row: Dict, axis: str, level) -> Dict:
    """Возвращает копию row с принудительно установленным уровнем по axis.

    Помечает row флагом `repaired_axis`/`repaired_level` для прозрачности
    в LHS-метаданных.
    """
    new_row = dict(row)
    new_row[axis] = level
    new_row["repaired_axis"] = axis
    new_row["repaired_level"] = level
    return new_row


# ---------- Главная функция: generate_lhs ----------

def generate_lhs(
    n_points: int = 50,
    master_seed: int = 2026,
    block_size: int = 64,
    max_blocks: int = 200,
    max_repair_attempts: int = 100,
    min_per_level: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Возвращает ровно n_points валидных LHS-rows.

    Уточнение 3 Accepted decision: rejection sampling блоками `block_size`
    (каждый — `LatinHypercube(d=6, scramble=True, optimization='random-cd')`
    с sub-seed от master_rng), пока не наберётся `n_points` валидных строк
    по симплексу `w_rec + w_gossip ≤ 1`.

    Уточнение 4: после набора — repair дискретных осей если хотя бы один
    уровень имеет cnt < `min_per_level[axis]`. Repair = заменить случайную
    over-represented точку на forced-level точку. До `max_repair_attempts`.

    При неудаче — `ValueError` с диагностикой.

    Параметры
    ---------
    n_points : int
        Целевое число LHS-точек (50 в основной матрице).
    master_seed : int
        Главный seed; sub-seeds блоков берутся от него детерминированно.
    block_size : int
        Размер блока rejection sampling.
    max_blocks : int
        Hard cap на число блоков rejection (защита от inf-loop).
    max_repair_attempts : int
        Hard cap на число попыток repair дискретных осей.
    min_per_level : dict, optional
        Override для DEFAULT_MIN_PER_LEVEL. Используется для smoke с малым
        n_points (где DEFAULT не достижим): передаётся ослабленный порог,
        например `{"program_variant": 1, "audience_size": 1,
        "popularity_source": 1}`.
    """
    if min_per_level is None:
        min_per_level = dict(DEFAULT_MIN_PER_LEVEL)

    master_rng = np.random.default_rng(master_seed)
    rows: List[Dict] = []
    block_idx = 0
    while len(rows) < n_points:
        block_seed = int(master_rng.integers(0, 2**31 - 1))
        sampler = qmc.LatinHypercube(
            d=6, scramble=True, optimization='random-cd',
            rng=np.random.default_rng(block_seed),
        )
        raw = sampler.random(block_size)
        for u in raw:
            mapped = _map_unit_to_row(u)
            if mapped is not None:
                rows.append(mapped)
                if len(rows) >= n_points:
                    break
        block_idx += 1
        if block_idx > max_blocks:
            raise ValueError(
                f"generate_lhs: после {block_idx} блоков набрано "
                f"только {len(rows)}/{n_points} валидных по симплексу точек; "
                f"проверьте ограничение w_rec + w_gossip ≤ 1"
            )

    rows = rows[:n_points]

    # Repair: пока баланс дискретных осей не достигнут, заменяем
    # случайные over-represented точки на forced-level. Skip оси, для которых
    # баланс физически недостижим (n_levels × min > n_points) — иначе repair
    # зацикливается, см. _is_balanced.
    for attempt in range(max_repair_attempts):
        counts = _check_balance(rows)
        if _is_balanced(counts, min_per_level, n_points=n_points):
            break
        replaced = False
        for axis in ("program_variant", "audience_size", "popularity_source"):
            level_counts = counts[axis]
            n_levels = len(level_counts)
            # Skip оси, для которых баланс физически недостижим
            if n_levels * min_per_level[axis] > n_points:
                continue
            for level, cnt in level_counts.items():
                if cnt < min_per_level[axis]:
                    over_axis = max(level_counts, key=level_counts.get)
                    # Сначала пытаемся заменить ещё-не-repaired точку
                    candidates = [
                        i for i, r in enumerate(rows)
                        if r[axis] == over_axis and not r.get("repaired_axis")
                    ]
                    if not candidates:
                        candidates = [
                            i for i, r in enumerate(rows)
                            if r[axis] == over_axis
                        ]
                    if not candidates:
                        continue
                    replace_idx = int(master_rng.choice(candidates))
                    rows[replace_idx] = _force_level_row(
                        rows[replace_idx], axis, level
                    )
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            counts = _check_balance(rows)
            raise ValueError(
                f"generate_lhs: дисбаланс не устранён за {attempt + 1} "
                f"попыток repair. Counts: {counts}; "
                f"min_per_level: {min_per_level}"
            )
    else:
        counts = _check_balance(rows)
        raise ValueError(
            f"generate_lhs: дисбаланс не устранён за {max_repair_attempts} "
            f"попыток repair. Counts: {counts}; "
            f"min_per_level: {min_per_level}"
        )

    # Проставить lhs_row_id после repair (стабильная индексация)
    for i, row in enumerate(rows):
        row["lhs_row_id"] = i
    return rows


# ---------- maximin subset selection ----------

def maximin_subset(
    rows: List[Dict],
    k: int = 12,
    force_program_variant_zero: bool = True,
) -> List[int]:
    """Greedy maximin distance отбор k LHS-точек.

    Координаты — `u_raw` (нормализованные [0,1]^6 unit-cube). Выбирает k
    точек, максимизируя минимальную попарную дистанцию в unit-cube
    (PROJECT_DESIGN §11 «крайние и центральные значения каждой оси»).

    Параметры
    ---------
    rows : list of dict
        Выход `generate_lhs`. Должен содержать `u_raw` каждой строки.
    k : int
        Размер subset (12 для основной матрицы PROJECT_DESIGN §11).
    force_program_variant_zero : bool
        Если True (default), хотя бы одна точка с program_variant=0
        включается принудительно (control-точка для cross-validation).
        Если в `rows` нет ни одной точки с program_variant=0 и флаг True —
        ValueError.

    Возвращает
    ----------
    Список из k уникальных индексов в `rows` (порядок — порядок выбора).
    """
    n = len(rows)
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k must be <= len(rows)={n}, got {k}")

    # Извлекаем u_raw как numpy-массив
    if "u_raw" not in rows[0]:
        raise ValueError("rows must contain 'u_raw' key (output of generate_lhs)")
    coords = np.array([r["u_raw"] for r in rows], dtype=np.float64)

    selected: List[int] = []
    if force_program_variant_zero:
        zero_indices = [i for i, r in enumerate(rows) if r["program_variant"] == 0]
        if not zero_indices:
            raise ValueError(
                "force_program_variant_zero=True, но в rows нет ни одной "
                "точки с program_variant=0; включите её или отключите флаг"
            )
        selected.append(zero_indices[0])

    while len(selected) < k:
        if not selected:
            # force_pv0=False и rows непуст: берём точку 0 как seed
            selected.append(0)
            continue
        sel_coords = coords[selected]
        # min distance от каждой точки до уже выбранных
        diffs = coords[:, None, :] - sel_coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        # Маска: уже выбранные → -inf, чтобы не выбрать снова
        for i in selected:
            min_dists[i] = -np.inf
        best_i = int(np.argmax(min_dists))
        selected.append(best_i)

    return selected
