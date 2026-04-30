"""Multi-signal fame score для каждого доклада.

Идея: вместо ручного выписывания «известных спикеров», вычислить fame
как функцию структурных признаков самой программы. Это методологически
чище и воспроизводимо на любой конференции.

Сигналы:
1. Длина abstract (нормализованная) — звёзды пишут детальнее.
2. Количество соавторов — ПК-доклады с 3+ спикерами заведомо звёздные.
3. Категория-хайп — LLM, AI, Architecture получают boost.
4. Слот-формат — single-talk слоты (keynote) фиксированно звёздные.
5. Компания — крупные компании дают boost.

Output: JSON {talk_id: fame ∈ [0, 1]} для каждой конференции.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Набор «хайповых» категорий (русско-английский смешанный, как в данных)
HYPE_KEYWORDS = {
    "llm", "ai", "ml", "agent", "neural", "transformer", "gpt",
    "architecture", "архитектур", "архитектура", "архитект",
    "perform", "производительн", "оптимиз", "performance",
    "on-device", "ondevice", "inference",
    "ux", "ui", "design",
    "biohack", "биохакинг",
    "trend", "тренд", "state of",
}

# Крупные компании (примерные паттерны для Mobius/Demo Day)
LARGE_COMPANIES = {
    "JUG Ru Group", "JUG", "Делимобиль", "Газпромбанк", "Альфа-Банк",
    "Альфа", "Сбер", "Яндекс", "Yandex", "ITMO", "ИТМО",
    "ВТБ", "Тинькофф", "T-Bank", "Tinkoff",
    "X5", "Wildberries", "Ozon", "Авито", "Avito",
    "Kaspersky", "Касперский", "Лаборатория Касперского",
}


def normalize(values: list, min_v=None, max_v=None) -> list:
    """Min-max нормализация в [0, 1]."""
    if not values:
        return []
    if min_v is None:
        min_v = min(values)
    if max_v is None:
        max_v = max(values)
    if max_v == min_v:
        return [0.5] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def score_abstract_length(text: str) -> float:
    """Длина abstract в словах. Звёзды пишут детальнее."""
    if not text:
        return 0.0
    words = len(text.split())
    return min(1.0, words / 100.0)  # 100 слов = max


def score_coauthor_count(speakers: str) -> float:
    """3+ спикеров — обычно ПК-доклад (звёздный); 1-2 — обычный."""
    if not speakers:
        return 0.0
    parts = re.split(r"[,;]", speakers)
    parts = [p.strip() for p in parts if p.strip()]
    n = len(parts)
    if n >= 3:
        return 0.8  # ПК-формат
    elif n == 1:
        return 0.5  # обычный single-speaker
    else:
        return 0.4  # парный


def score_category_hype(category: str) -> float:
    """Хайповые категории получают boost."""
    if not category:
        return 0.0
    cat_lower = category.lower()
    matches = sum(1 for kw in HYPE_KEYWORDS if kw in cat_lower)
    return min(1.0, matches / 2.0)  # 2+ matches = max


def score_slot_position(slot_id: str, slot_index: int, total_slots: int,
                        is_single_talk: bool) -> float:
    """Single-talk слоты (keynote) фиксированно звёздные.
    Первый/последний слот дня тоже boost.
    """
    if is_single_talk:
        return 1.0
    # Первый или последний слот тоже немного boost
    if slot_index == 0 or slot_index == total_slots - 1:
        return 0.5
    return 0.0


def score_company(companies: str) -> float:
    """Крупные компании дают boost."""
    if not companies:
        return 0.0
    for company in LARGE_COMPANIES:
        if company.lower() in companies.lower():
            return 0.7
    return 0.0


def compute_fame(prog_path: Path) -> dict:
    with open(prog_path, encoding="utf-8") as f:
        prog = json.load(f)

    talks = prog["talks"]
    slots = prog["slots"]

    # Группируем доклады по slot_id
    slot_talk_counts = Counter(t["slot_id"] for t in talks)

    # Слот-индексы
    slot_id_to_idx = {s["id"]: i for i, s in enumerate(slots)}
    total_slots = len(slots)

    fame_scores = {}
    detailed = {}

    for t in talks:
        sid = t["slot_id"]
        slot_idx = slot_id_to_idx.get(sid, 0)
        is_single = slot_talk_counts[sid] == 1

        scores = {
            "abstract_length": score_abstract_length(t.get("abstract", "")),
            "coauthor_count": score_coauthor_count(t.get("speakers", "")),
            "category_hype": score_category_hype(t.get("category", "")),
            "slot_position": score_slot_position(sid, slot_idx, total_slots, is_single),
            "company": score_company(t.get("companies", "") or ""),
        }

        # Взвешенная сумма
        weights = {
            "abstract_length": 0.20,
            "coauthor_count": 0.15,
            "category_hype": 0.20,
            "slot_position": 0.30,  # сильнее всего — keynote — заведомо звезда
            "company": 0.15,
        }
        fame = sum(scores[k] * weights[k] for k in scores)
        fame_scores[t["id"]] = round(fame, 3)
        detailed[t["id"]] = {**scores, "final": round(fame, 3)}

    return fame_scores, detailed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conferences", nargs="+",
                   default=["mobius_2025_autumn", "demo_day_2026"])
    args = p.parse_args()

    for conf in args.conferences:
        prog_path = ROOT / "data" / "conferences" / f"{conf}.json"
        if not prog_path.exists():
            print(f"SKIP: {prog_path} not found")
            continue

        fame_scores, detailed = compute_fame(prog_path)

        out_path = ROOT / "data" / "conferences" / f"{conf}_fame.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "fame": fame_scores,
                "detailed": detailed,
                "weights": {
                    "abstract_length": 0.20,
                    "coauthor_count": 0.15,
                    "category_hype": 0.20,
                    "slot_position": 0.30,
                    "company": 0.15,
                },
            }, f, ensure_ascii=False, indent=2)

        # Статистика
        values = list(fame_scores.values())
        n_stars = sum(1 for v in values if v > 0.6)
        n_medium = sum(1 for v in values if 0.3 < v <= 0.6)
        n_normal = sum(1 for v in values if v <= 0.3)
        print(f"\n=== {conf} ===")
        print(f"  Total talks: {len(values)}")
        print(f"  Stars (fame > 0.6):    {n_stars}  ({n_stars/len(values)*100:.0f}%)")
        print(f"  Medium (0.3 < fame ≤ 0.6): {n_medium} ({n_medium/len(values)*100:.0f}%)")
        print(f"  Normal (fame ≤ 0.3):   {n_normal} ({n_normal/len(values)*100:.0f}%)")
        print(f"  Mean: {sum(values)/len(values):.3f}, Max: {max(values):.3f}, Min: {min(values):.3f}")

        # Топ-5 fame
        print(f"  Top 5 fame talks:")
        sorted_fame = sorted(fame_scores.items(), key=lambda x: -x[1])
        with open(prog_path, encoding="utf-8") as f:
            prog = json.load(f)
        title_by_id = {t["id"]: t["title"] for t in prog["talks"]}
        for tid, score in sorted_fame[:5]:
            print(f"    {score:.3f}: {title_by_id[tid][:70]}")
        print(f"  WROTE: {out_path}")


if __name__ == "__main__":
    main()
