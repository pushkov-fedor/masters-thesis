"""Тесты оператора локальных модификаций программы Φ (этап N).

Источник истины: `docs/spikes/spike_program_modification.md` §11
(acceptance этапа N) + accepted Q-M1 — Q-M8 от 2026-05-07.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.program_modification import (
    SwapDescriptor,
    _apply_swap,
    _enumerate_all_pairs,
    enumerate_modifications,
    has_speaker_conflict,
)
from src.simulator import Conference


# ---------- Фикстуры данных ----------

@pytest.fixture(scope="session")
def toy_conflict_conf(experiments_root: Path) -> Conference:
    """toy_speaker_conflict.json — 2 timeslot × 2 hall × 4 talks с тем,
    что одна пара swap создаёт конфликт спикера, другая — нет.
    """
    return Conference.load(
        experiments_root / "data/conferences/toy_speaker_conflict.json",
        experiments_root / "data/conferences/toy_speaker_conflict_embeddings.npz",
    )


@pytest.fixture(scope="session")
def toy_microconf_conf(experiments_root: Path) -> Conference:
    """toy_microconf.json — 1 timeslot × 2 halls × 2 talks. Φ не должна
    возвращать никаких модификаций (нет другого слота для swap)."""
    return Conference.load(
        experiments_root / "data/conferences/toy_microconf.json",
        experiments_root / "data/conferences/toy_microconf_embeddings.npz",
    )


@pytest.fixture(scope="session")
def mobius_conf(experiments_root: Path) -> Conference:
    """mobius_2025_autumn.json — реальная программа JUG с заполненным
    полем `speakers` (string через запятую)."""
    return Conference.load(
        experiments_root / "data/conferences/mobius_2025_autumn.json",
        experiments_root / "data/conferences/mobius_2025_autumn_embeddings.npz",
    )


# ---------- Базовые инварианты Φ на toy_microconf_2slot ----------

def test_phi_preserves_talk_set(toy_2slot_conf):
    """L.N invariant: множество talk_id в P_k идентично P_0."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_2slot_conf, k_max=5, rng=rng)
    assert mods, "должна быть хотя бы одна валидная модификация на toy_2slot"
    base_ids = set(toy_2slot_conf.talks.keys())
    for modified, _ in mods:
        assert set(modified.talks.keys()) == base_ids, (
            "Φ изменила состав talks"
        )


def test_phi_changes_slot_id_for_exactly_two_talks(toy_2slot_conf):
    """L.N invariant: ровно 2 талка имеют изменённый slot_id относительно P_0."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_2slot_conf, k_max=5, rng=rng)
    base_slot_ids = {tid: t.slot_id for tid, t in toy_2slot_conf.talks.items()}
    for modified, desc in mods:
        diffs = {
            tid for tid, t in modified.talks.items()
            if t.slot_id != base_slot_ids[tid]
        }
        assert diffs == {desc.t1, desc.t2}, (
            f"ожидался ровно swap (t1, t2)={desc.t1, desc.t2}, "
            f"но изменены {diffs}"
        )


def test_phi_preserves_slot_sizes(toy_2slot_conf):
    """L.N invariant: размеры слотов не меняются (swap — это перестановка
    1↔1, не вставка/удаление)."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_2slot_conf, k_max=5, rng=rng)
    base_sizes = sorted(len(s.talk_ids) for s in toy_2slot_conf.slots)
    for modified, _ in mods:
        sizes = sorted(len(s.talk_ids) for s in modified.slots)
        assert sizes == base_sizes, (
            f"размеры слотов изменились: было {base_sizes}, стало {sizes}"
        )


def test_phi_preserves_halls_per_talk(toy_2slot_conf):
    """L.N invariant: hall у каждого талка сохраняется (swap двигает только
    slot_id; hall не трогаем — Q-M1 семантика)."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_2slot_conf, k_max=5, rng=rng)
    base_halls = {tid: t.hall for tid, t in toy_2slot_conf.talks.items()}
    for modified, _ in mods:
        for tid, t in modified.talks.items():
            assert t.hall == base_halls[tid], (
                f"hall изменился у {tid}: было {base_halls[tid]}, стало {t.hall}"
            )


def test_phi_returned_modifications_have_no_speaker_conflict(
    toy_conflict_conf,
):
    """L.N invariant: ни одна возвращённая Φ модификация не содержит
    speaker-конфликта (hard validation)."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_conflict_conf, k_max=10, rng=rng)
    assert mods, "на toy_speaker_conflict должна быть хотя бы одна валидная пара"
    for modified, _ in mods:
        assert not has_speaker_conflict(modified), (
            "Φ вернула модификацию со speaker-конфликтом"
        )


# ---------- TC5: hard-validation speaker-конфликта ----------

def test_tc5_phi_rejects_speaker_conflict_swap(toy_conflict_conf):
    """**TC5** (PIVOT_IMPLEMENTATION_PLAN §10.1, accepted Q-M8): swap,
    создающий speaker-конфликт, отбрасывается оператором Φ.

    На toy_speaker_conflict есть 4 candidate-pair swap'ов между двумя
    timeslot'ами одного дня; 2 из них создают конфликт (Alice оказывается
    в обоих талках одного слота), 2 — не создают. Φ должна вернуть только
    2 валидные.
    """
    # Sanity-check: исходная программа без конфликтов.
    assert not has_speaker_conflict(toy_conflict_conf), (
        "fixture toy_speaker_conflict сам по себе не должен иметь конфликтов"
    )

    all_pairs = _enumerate_all_pairs(toy_conflict_conf)
    assert len(all_pairs) == 4, (
        f"ожидалось 4 candidate-pair (2×2 swap между двумя слотами одного дня), "
        f"получили {len(all_pairs)}"
    )

    # Через Φ при k_max=10 должны прийти только валидные.
    rng = np.random.default_rng(0)
    mods = enumerate_modifications(toy_conflict_conf, k_max=10, rng=rng)
    assert len(mods) == 2, (
        f"ожидалось 2 валидные модификации (2 из 4 swap не дают конфликта), "
        f"получили {len(mods)}"
    )

    # И ни одной с конфликтом.
    for modified, _ in mods:
        assert not has_speaker_conflict(modified)

    # Прямая проверка: swap (t_s0_h1 ↔ t_s1_h2) и (t_s0_h2 ↔ t_s1_h1) ДОЛЖНЫ
    # создать конфликт (Alice в одном слоте дважды).
    desc_conflict_a = SwapDescriptor("slot_00", "slot_01", "t_s0_h1", "t_s1_h2")
    desc_conflict_b = SwapDescriptor("slot_00", "slot_01", "t_s0_h2", "t_s1_h1")
    for desc in (desc_conflict_a, desc_conflict_b):
        modified = _apply_swap(toy_conflict_conf, desc)
        assert has_speaker_conflict(modified), (
            f"swap {desc} ДОЛЖЕН создавать конфликт, но Φ его не обнаружила"
        )


# ---------- Edge cases ----------

def test_phi_returns_empty_for_single_slot_program(toy_microconf_conf):
    """Q-M4 / §6 V1: на одно-слотной программе нет «другого временного слота»
    одного дня → Φ возвращает пустой список."""
    rng = np.random.default_rng(42)
    mods = enumerate_modifications(toy_microconf_conf, k_max=5, rng=rng)
    assert mods == []


def test_phi_returns_empty_when_k_max_zero(toy_2slot_conf):
    """k_max=0 → пустой список (control-точка `program_variant=0` клеится
    вызывающим, не Φ)."""
    rng = np.random.default_rng(42)
    assert enumerate_modifications(toy_2slot_conf, k_max=0, rng=rng) == []


def test_phi_returns_empty_when_k_max_negative(toy_2slot_conf):
    """k_max<0 → пустой список (sanity)."""
    rng = np.random.default_rng(42)
    assert enumerate_modifications(toy_2slot_conf, k_max=-3, rng=rng) == []


# ---------- Determinism ----------

def test_phi_deterministic_under_fixed_rng(toy_2slot_conf):
    """L.N invariant: при одинаковом rng-seed выдача Φ идентична."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    mods_a = enumerate_modifications(toy_2slot_conf, k_max=2, rng=rng_a)
    mods_b = enumerate_modifications(toy_2slot_conf, k_max=2, rng=rng_b)
    assert len(mods_a) == len(mods_b)
    for (_, desc_a), (_, desc_b) in zip(mods_a, mods_b):
        assert desc_a == desc_b


def test_phi_different_rng_may_yield_different_subsamples(toy_2slot_conf):
    """Sanity: при k_max < |valid_pool| разные rng-seed могут дать разные
    подмножества (это и есть смысл V5 random sampling).

    Не требуем строгого неравенства — на маленькой выборке возможны
    совпадения; требуем хотя бы возможности разности (не падает с двумя
    seed'ами)."""
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    mods_a = enumerate_modifications(toy_2slot_conf, k_max=1, rng=rng_a)
    mods_b = enumerate_modifications(toy_2slot_conf, k_max=1, rng=rng_b)
    assert len(mods_a) == 1 and len(mods_b) == 1


# ---------- Реальное чтение speakers из JSON ----------

def test_speakers_are_parsed_from_mobius_json(mobius_conf):
    """Q-M7: после расширения `Talk.speakers` поле `speakers` (comma-separated
    string) в Mobius JSON реально попадает в `Talk.speakers` как список."""
    n_talks_with_speakers = sum(
        1 for t in mobius_conf.talks.values() if t.speakers
    )
    assert n_talks_with_speakers >= 30, (
        f"ожидалось ≥ 30 mobius-талков с распарсенными спикерами, "
        f"получено {n_talks_with_speakers} из {len(mobius_conf.talks)}"
    )
    # Хоть один талк с несколькими спикерами (ко-докладчики).
    multi_speaker = [
        t for t in mobius_conf.talks.values() if len(t.speakers) >= 2
    ]
    assert multi_speaker, "ожидался хотя бы один талк с несколькими спикерами"
    # Никаких пустых строк или неотстрипанных пробелов.
    for t in mobius_conf.talks.values():
        for sp in t.speakers:
            assert sp == sp.strip()
            assert sp != ""


def test_speakers_are_empty_for_toy_microconf(toy_microconf_conf):
    """Q-M2: на конференции без поля `speakers` `Talk.speakers = []` (no-op
    validation; не «нет конфликтов», а отсутствие данных)."""
    for t in toy_microconf_conf.talks.values():
        assert t.speakers == []


def test_speakers_are_empty_for_toy_2slot(toy_2slot_conf):
    """Q-M2: toy_microconf_2slot тоже без speakers — validation no-op."""
    for t in toy_2slot_conf.talks.values():
        assert t.speakers == []


# ---------- has_speaker_conflict ----------

def test_has_speaker_conflict_false_when_no_data(toy_2slot_conf):
    """has_speaker_conflict тривиально False, если speakers=[] везде —
    это no-op-валидация, не доказательство отсутствия конфликта."""
    assert not has_speaker_conflict(toy_2slot_conf)


def test_has_speaker_conflict_true_when_synthetic_collision(
    toy_conflict_conf,
):
    """has_speaker_conflict обнаруживает синтетически созданный конфликт."""
    # Применяем conflict-creating swap руками
    desc = SwapDescriptor("slot_00", "slot_01", "t_s0_h1", "t_s1_h2")
    modified = _apply_swap(toy_conflict_conf, desc)
    assert has_speaker_conflict(modified)
