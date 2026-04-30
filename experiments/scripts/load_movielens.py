"""Cross-domain валидация на MovieLens 1M:
адаптация (movies, users, ratings) → (talks, users, slots).

Маппинг:
- top-N фильмов по числу ratings → talks (N=160)
- жанры → halls (~18, основной жанр первый из списка)
- 8 виртуальных слотов, по 20 фильмов в каждом (стратифицированно по жанру)
- capacity per hall: подобрано так, чтобы при 1000 user был режим border (~10-15% переполнения)
- эмбеддинги фильмов: one-hot векторы жанров (18-dim)
- эмбеддинги пользователей: усреднение one-hot жанров их top-K оценённых фильмов
- LearnedPreferenceFn пропускаем (он обучен на конференциях); используем cosine.

Цель: тот же набор политик прогнать на этой "конференции" и сравнить ranking
с Mobius/Demo Day (Spearman ρ).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ML_DIR = ROOT / "data" / "external" / "movielens" / "ml-1m"


GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_movies():
    movies = {}
    with open(ML_DIR / "movies.dat", encoding="latin-1") as f:
        for line in f:
            mid, title, genres_str = line.strip().split("::")
            genres = genres_str.split("|")
            movies[int(mid)] = {"title": title, "genres": genres}
    return movies


def load_ratings(min_rating: int = 4):
    """Возвращает {user_id: {movie_id: rating}} только для rating >= min."""
    ratings = {}
    with open(ML_DIR / "ratings.dat") as f:
        for line in f:
            uid, mid, rating, _ts = line.strip().split("::")
            r = int(rating)
            if r < min_rating:
                continue
            ratings.setdefault(int(uid), {})[int(mid)] = r
    return ratings


def build_movie_emb(movie):
    v = np.zeros(len(GENRES), dtype=np.float32)
    for g in movie["genres"]:
        if g in GENRES:
            v[GENRES.index(g)] = 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def build_user_emb(user_ratings, movies, weight_by_rating=True):
    v = np.zeros(len(GENRES), dtype=np.float32)
    for mid, r in user_ratings.items():
        if mid not in movies:
            continue
        m_emb = build_movie_emb(movies[mid])
        v += m_emb * (r if weight_by_rating else 1.0)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def main():
    movies = load_movies()
    ratings = load_ratings(min_rating=4)

    movie_popularity = {mid: 0 for mid in movies}
    for ur in ratings.values():
        for mid in ur:
            movie_popularity[mid] += 1

    # топ-N по популярности с rating>=4, у которых известный жанр
    sorted_movies = sorted(
        ((p, mid) for mid, p in movie_popularity.items()
         if any(g in GENRES for g in movies[mid]["genres"]) and p > 0),
        reverse=True,
    )
    N_TALKS = 160
    selected_mids = [mid for _, mid in sorted_movies[:N_TALKS]]
    print(f"Selected {len(selected_mids)} top movies (by #ratings>=4)")

    # 8 слотов, 20 фильмов в каждом — стратифицированный split по основному жанру
    rng = np.random.default_rng(42)
    rng.shuffle(selected_mids)
    n_slots = 8
    per_slot = N_TALKS // n_slots

    talks = {}
    halls = {}
    slots = []
    hall_id_for_genre = {g: i + 1 for i, g in enumerate(GENRES)}
    talk_embeddings = []
    talk_ids = []

    for slot_idx in range(n_slots):
        slot_mids = selected_mids[slot_idx * per_slot:(slot_idx + 1) * per_slot]
        slot_id = f"s{slot_idx+1:02d}"
        slot_talk_ids = []
        for mid in slot_mids:
            m = movies[mid]
            primary_genre = next((g for g in m["genres"] if g in GENRES), "Drama")
            hall_idx = hall_id_for_genre[primary_genre]
            tid = f"m{mid:04d}_{slot_idx}"
            talks[tid] = {
                "id": tid,
                "title": m["title"][:80],
                "hall": hall_idx,
                "slot_id": slot_id,
                "category": primary_genre,
                "abstract": "/".join(m["genres"]),
                "fame": 0.0,
            }
            slot_talk_ids.append(tid)
            talk_ids.append(tid)
            talk_embeddings.append(build_movie_emb(m))
        slots.append({
            "id": slot_id,
            "datetime": f"2026-01-{slot_idx+1:02d}T10:00:00",
            "talk_ids": slot_talk_ids,
        })

    # Capacity per hall — подобрано так, чтобы при 1000 user был режим border
    # На каждый слот в среднем 20/18 ≈ 1.1 talk на жанр. Если capacity = 50,
    # тогда 1000 пользователей × 1/18 = ~55 на жанр в слоте → лёгкое переполнение
    for g, hid in hall_id_for_genre.items():
        halls[hid] = {"id": hid, "capacity": 60}

    # Сборка users из ratings: 1000 случайных пользователей с >= 20 рейтингов >= 4
    users_meta = []
    user_embeddings = []
    qualifying = [uid for uid, ur in ratings.items() if len(ur) >= 20]
    print(f"Users with >=20 high ratings: {len(qualifying)}")
    rng2 = np.random.default_rng(7)
    selected_uids = rng2.choice(qualifying, size=min(1000, len(qualifying)), replace=False)

    for uid in selected_uids:
        u_emb = build_user_emb(ratings[int(uid)], movies)
        users_meta.append({
            "id": f"u_{int(uid):05d}",
            "background": "MovieLens user",
        })
        user_embeddings.append(u_emb)

    # Save
    out_dir = ROOT / "data" / "conferences"
    conf_path = out_dir / "movielens_cross.json"
    emb_path = out_dir / "movielens_cross_embeddings.npz"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": "MovieLens 1M Cross-Domain",
            "talks": list(talks.values()),
            "halls": list(halls.values()),
            "slots": slots,
        }, f, ensure_ascii=False, indent=2)
    np.savez(emb_path,
             ids=np.array(talk_ids),
             embeddings=np.array(talk_embeddings, dtype=np.float32))
    print(f"WROTE: {conf_path}")
    print(f"WROTE: {emb_path}")

    persona_path = ROOT / "data" / "personas" / "movielens_users.json"
    persona_emb_path = ROOT / "data" / "personas" / "movielens_users_embeddings.npz"
    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump(users_meta, f, ensure_ascii=False, indent=2)
    np.savez(persona_emb_path,
             ids=np.array([u["id"] for u in users_meta]),
             embeddings=np.array(user_embeddings, dtype=np.float32))
    print(f"WROTE: {persona_path}")
    print(f"WROTE: {persona_emb_path}")
    print(f"\nSummary: {len(talks)} talks, {len(halls)} halls, {len(slots)} slots, {len(users_meta)} users")
    print(f"Talks per slot: {len(talks)//len(slots)}, capacity per hall: 60 (total {60*len(halls)}/slot)")


if __name__ == "__main__":
    main()
