"""Подготовка Scholar Inbox dataset для обучения preference-модели.

1. Из rated_papers.csv выбираем top-N самых рейтингованных arxiv-папер.
2. Через arxiv.org/api/query (urllib, без зависимостей) скачиваем title+abstract.
3. Эмбеддим e5-multilingual-small (пассажный режим).
4. Строим user-эмбеддинги как centroid эмбеддингов положительно оценённых папер.
5. Сохраняем в формат, совместимый с experiments/data/conferences/* и /personas/*:
   - data/conferences/scholar_inbox_papers.json (talks-style)
   - data/conferences/scholar_inbox_papers_embeddings.npz
   - data/personas/scholar_inbox_users.json
   - data/personas/scholar_inbox_users_embeddings.npz

Запуск:
    python scripts/prepare_scholar_inbox.py --top-papers 5000 --min-ratings 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_BASE = "http://export.arxiv.org/api/query"
SI_DIR = ROOT / "data" / "external" / "preference_dataset_search_2026_05" / "scholar_inbox_datasets" / "data"


def fetch_arxiv_batch(arxiv_ids, max_retries=3):
    """Скачивает title+abstract батчем до 100 ID за раз через arxiv.org/api/query."""
    id_str = ",".join(arxiv_ids)
    params = {"id_list": id_str, "max_results": str(len(arxiv_ids))}
    url = ARXIV_BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ITMO-thesis-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  ERR на batch {arxiv_ids[:3]}...: {e}", flush=True)
                return {}
            time.sleep(3 * (attempt + 1))
    out = {}
    try:
        root = ET.fromstring(data)
        for entry in root.findall(f"{ATOM}entry"):
            id_el = entry.find(f"{ATOM}id")
            title_el = entry.find(f"{ATOM}title")
            summary_el = entry.find(f"{ATOM}summary")
            if id_el is None or title_el is None:
                continue
            # arxiv id из URL "http://arxiv.org/abs/2103.15595v2"
            arxiv_url = id_el.text or ""
            base = arxiv_url.split("/abs/")[-1]
            base = base.split("v")[0] if "v" in base else base
            out[base] = {
                "title": (title_el.text or "").strip().replace("\n", " "),
                "abstract": (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else "",
            }
    except ET.ParseError as e:
        print(f"  XML parse error: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-papers", type=int, default=5000,
                    help="Сколько папер брать. При --sampling top — самые рейтингованные. "
                         "При --sampling random — случайные из тех, что прошли фильтр min-paper-ratings.")
    ap.add_argument("--min-ratings", type=int, default=5,
                    help="Минимум ratings на user для включения в обучающий набор")
    ap.add_argument("--min-paper-ratings", type=int, default=3,
                    help="(только --sampling random) Минимум ratings на paper для включения в pool — "
                         "выкидываем мусор из 1-2 случайных оценок")
    ap.add_argument("--sampling", choices=["top", "random"], default="top",
                    help="top — top-N по числу ratings (popularity-bias); "
                         "random — случайный sample из papers с ≥min-paper-ratings (natural distribution).")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed для random sampling")
    ap.add_argument("--batch-size", type=int, default=80,
                    help="ID на один arxiv API запрос (max 100)")
    ap.add_argument("--rate-sleep", type=float, default=3.0,
                    help="Пауза между запросами (arxiv просит 3 сек)")
    args = ap.parse_args()

    print("Loading rated_papers.csv...")
    rp = pd.read_csv(SI_DIR / "rated_papers.csv")
    print(f"  {len(rp)} ratings, {rp.user_id.nunique()} users, {rp.arxiv_id.nunique()} papers")

    paper_counts = rp.groupby("arxiv_id").size().sort_values(ascending=False)

    if args.sampling == "top":
        top_papers = paper_counts.head(args.top_papers).index.tolist()
        print(f"\n[sampling=top] selected top-{args.top_papers} papers, "
              f"ratings range {paper_counts.head(args.top_papers).min()} ... "
              f"{paper_counts.head(args.top_papers).max()}")
    else:  # random
        eligible = paper_counts[paper_counts >= args.min_paper_ratings].index.tolist()
        rng_p = np.random.default_rng(args.seed)
        n_take = min(args.top_papers, len(eligible))
        idx = rng_p.choice(len(eligible), size=n_take, replace=False)
        top_papers = [eligible[i] for i in idx]
        sub_counts = paper_counts.loc[top_papers]
        print(f"\n[sampling=random, seed={args.seed}, min-paper-ratings={args.min_paper_ratings}] "
              f"selected {len(top_papers)} of {len(eligible)} eligible papers")
        print(f"  ratings/paper: min={sub_counts.min()} median={int(sub_counts.median())} "
              f"max={sub_counts.max()} mean={sub_counts.mean():.1f}")

    rp_sub = rp[rp.arxiv_id.isin(set(top_papers))]
    n_pos = (rp_sub["rating"] == 1).sum()
    print(f"  raw rating distribution before user-filter: "
          f"+1={n_pos} ({n_pos/len(rp_sub)*100:.1f}%), "
          f"-1={(rp_sub['rating']==-1).sum()} ({(rp_sub['rating']==-1).sum()/len(rp_sub)*100:.1f}%)")

    user_counts = rp_sub.groupby("user_id").size()
    keep_users = user_counts[user_counts >= args.min_ratings].index
    rp_sub = rp_sub[rp_sub.user_id.isin(keep_users)]
    n_pos = (rp_sub["rating"] == 1).sum()
    print(f"After filter (>= {args.min_ratings} ratings/user): "
          f"{len(rp_sub)} ratings, {rp_sub.user_id.nunique()} users, {rp_sub.arxiv_id.nunique()} papers, "
          f"pos={n_pos/len(rp_sub)*100:.1f}%")

    # === Скачивание abstracts ===
    cache_path = ROOT / "data" / "external" / "scholar_inbox_abstracts.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"\nFound abstracts cache with {len(cache)} entries")

    todo = [aid for aid in top_papers if aid not in cache]
    print(f"To fetch: {len(todo)} new abstracts via arxiv API")
    if todo:
        t0 = time.time()
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            res = fetch_arxiv_batch(batch)
            cache.update(res)
            elapsed = time.time() - t0
            remaining = len(todo) - i - len(batch)
            print(f"  [{i + len(batch)}/{len(todo)}] +{len(res)} abstracts, "
                  f"elapsed {elapsed:.0f}s, eta {elapsed / max(1, i + len(batch)) * remaining:.0f}s",
                  flush=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
            if i + args.batch_size < len(todo):
                time.sleep(args.rate_sleep)
    print(f"Total abstracts in cache: {len(cache)}")

    # Оставляем только papers с реально скачанными abstracts
    valid_papers = [aid for aid in top_papers if aid in cache and cache[aid].get("abstract")]
    print(f"Papers with valid abstracts: {len(valid_papers)} / {len(top_papers)}")

    rp_sub = rp_sub[rp_sub.arxiv_id.isin(valid_papers)]
    user_counts = rp_sub.groupby("user_id").size()
    keep_users = user_counts[user_counts >= args.min_ratings].index
    rp_sub = rp_sub[rp_sub.user_id.isin(keep_users)]
    print(f"Final: {len(rp_sub)} ratings, {rp_sub.user_id.nunique()} users, "
          f"{rp_sub.arxiv_id.nunique()} papers")

    # === Эмбеддим папер через e5 ===
    print("\nEmbedding papers via e5...")
    from src.embedder import embed_texts
    paper_ids = sorted(rp_sub.arxiv_id.unique())
    paper_texts = []
    for aid in paper_ids:
        meta = cache.get(aid, {})
        text = (meta.get("title", "") + ". " + meta.get("abstract", "")).strip()
        paper_texts.append(text or "Unknown paper")
    paper_embs = embed_texts(paper_texts, kind="passage")
    print(f"  paper embeddings: {paper_embs.shape}")

    # === User_emb через centroid +1 (fallback baseline для legacy use), быстрый groupby ===
    print("\nBuilding user embeddings as centroid of liked papers (groupby, fast)...")
    paper_idx = {aid: i for i, aid in enumerate(paper_ids)}
    user_ids = sorted(rp_sub.user_id.unique())
    user_id_to_pos = {uid: i for i, uid in enumerate(user_ids)}
    user_embs = np.zeros((len(user_ids), paper_embs.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(42)
    grouped = rp_sub.groupby("user_id")
    for uid, g in grouped:
        pos = user_id_to_pos[uid]
        liked = g[g.rating == 1].arxiv_id.tolist()
        if not liked:
            liked = g.arxiv_id.tolist()
        idxs = [paper_idx[aid] for aid in liked if aid in paper_idx]
        if not idxs:
            user_embs[pos] = rng.standard_normal(paper_embs.shape[1]).astype(np.float32)
            continue
        c = paper_embs[idxs].mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        user_embs[pos] = c.astype(np.float32)
    print(f"  user embeddings: {user_embs.shape}")

    # === Сохраняем user_history (positives + negatives) и paper_popularity ===
    # Это нужно trainer'у для LOO user_emb и popularity feature.
    print("\nBuilding user_history.json + paper_popularity.json...")
    user_history = {}
    for uid, g in grouped:
        user_history[f"si_{uid}"] = {
            "positives": g[g.rating == 1].arxiv_id.tolist(),
            "negatives": g[g.rating == -1].arxiv_id.tolist(),
        }
    history_path = ROOT / "data" / "scholar_inbox_user_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(user_history, f, ensure_ascii=False)
    print(f"  WROTE: {history_path}")

    # popularity = число +1 ratings на paper В обучающей выборке (rp_sub, после фильтров)
    pop_pos = rp_sub[rp_sub.rating == 1].groupby("arxiv_id").size().to_dict()
    pop_neg = rp_sub[rp_sub.rating == -1].groupby("arxiv_id").size().to_dict()
    paper_popularity = {aid: {"pos": int(pop_pos.get(aid, 0)),
                              "neg": int(pop_neg.get(aid, 0))}
                        for aid in paper_ids}
    pop_path = ROOT / "data" / "scholar_inbox_paper_popularity.json"
    with open(pop_path, "w", encoding="utf-8") as f:
        json.dump(paper_popularity, f, ensure_ascii=False)
    print(f"  WROTE: {pop_path}")

    # === Сохраняем в формат проекта ===
    talks = []
    for aid in paper_ids:
        meta = cache[aid]
        talks.append({
            "id": aid,
            "title": meta["title"][:200],
            "speakers": "",
            "hall": 1,
            "date": "2025-01-01",
            "start_time": "10:00:00",
            "end_time": "11:00:00",
            "category": "Research",
            "abstract": meta["abstract"][:1500],
            "slot_id": "slot_00",
        })
    conf_blob = {
        "conf_id": "scholar_inbox_papers",
        "name": "Scholar Inbox",
        "date": "2025-01-01",
        "talks": talks,
        "halls": [{"id": 1, "capacity": 100000}],
        "slots": [{"id": "slot_00", "datetime": "2025-01-01 10:00:00"}],
    }
    conf_path = ROOT / "data" / "conferences" / "scholar_inbox_papers.json"
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump(conf_blob, f, ensure_ascii=False)
    np.savez(ROOT / "data" / "conferences" / "scholar_inbox_papers_embeddings.npz",
             ids=np.array(paper_ids), embeddings=paper_embs)
    print(f"\nWROTE: {conf_path}")
    print(f"WROTE: scholar_inbox_papers_embeddings.npz")

    users = [{"id": f"si_{uid}", "background": f"Scholar Inbox user {uid}"} for uid in user_ids]
    users_path = ROOT / "data" / "personas" / "scholar_inbox_users.json"
    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False)
    np.savez(ROOT / "data" / "personas" / "scholar_inbox_users_embeddings.npz",
             ids=np.array([f"si_{uid}" for uid in user_ids]),
             embeddings=user_embs)
    print(f"WROTE: {users_path}")
    print(f"WROTE: scholar_inbox_users_embeddings.npz")

    # === Сохраняем pairs для trainer ===
    pairs = []
    for r in rp_sub.itertuples():
        pairs.append({
            "persona_id": f"si_{r.user_id}",
            "talk_id": r.arxiv_id,
            "score": 1.0 if r.rating == 1 else 0.0,
        })
    pairs_path = ROOT / "data" / "preferences_matrix_scholar_inbox.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"WROTE: {pairs_path} ({len(pairs)} pairs)")
    print(f"\n=== Done. Next step: train_preference_model.py "
          f"--source llm_ratings --prefs preferences_matrix_scholar_inbox "
          f"--personas-npz scholar_inbox_users --talks-conference scholar_inbox_papers "
          f"--out preference_model_scholar_inbox ===")


if __name__ == "__main__":
    main()
