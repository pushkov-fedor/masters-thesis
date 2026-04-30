"""Эксплоративный анализ: насколько cosine(profile, talk) соответствует человеческим оценкам.

Использует 10 эталонных профилей и 390 оценок из mobius_topics_users.
Считает корреляцию (Pearson, Spearman) и AUC для трёх порогов.

Это НЕ ground-truth для эксперимента — просто разведка для решения утром.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROG_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn.json"
EMB_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz"
USERS_PATH = ROOT / "data" / "conferences" / "mobius_2025_autumn_users.json"


def main():
    with open(PROG_PATH, encoding="utf-8") as f:
        prog = json.load(f)
    npz = np.load(EMB_PATH, allow_pickle=False)
    talk_ids = list(npz["ids"])
    talk_emb = {tid: npz["embeddings"][i] for i, tid in enumerate(talk_ids)}

    with open(USERS_PATH, encoding="utf-8") as f:
        ref = json.load(f)

    print(f"Reference users: {len(ref['users'])}")
    print(f"Score records: {len(ref['scores'])}")

    print("\nLoading embedder...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    user_texts = [u["profile"] for u in ref["users"]]
    user_emb_arr = model.encode(user_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
    user_emb = {u["id"]: user_emb_arr[i] for i, u in enumerate(ref["users"])}

    cosines = []
    scores = []
    for s in ref["scores"]:
        u = user_emb.get(s["user_id"])
        t = talk_emb.get(s["topic_id"])
        if u is None or t is None:
            continue
        cosines.append(float(np.dot(u, t)))
        scores.append(float(s["score"]))
    cosines = np.array(cosines)
    scores = np.array(scores)

    print(f"\nMatched score records: {len(scores)}")
    print(f"Cosine: mean={cosines.mean():.3f}, std={cosines.std():.3f}, "
          f"min={cosines.min():.3f}, max={cosines.max():.3f}")
    print(f"Score:  mean={scores.mean():.3f}, std={scores.std():.3f}, "
          f"min={scores.min():.0f}, max={scores.max():.0f}")
    print(f"\nДистрибуция оценок:")
    for v in sorted(set(scores)):
        print(f"  {v}: {(scores == v).sum()} records")

    pearson_r, pearson_p = pearsonr(cosines, scores)
    spearman_r, spearman_p = spearmanr(cosines, scores)
    print(f"\nPearson  r={pearson_r:.4f} (p={pearson_p:.4g})")
    print(f"Spearman r={spearman_r:.4f} (p={spearman_p:.4g})")

    # Бинаризация для AUC: «релевантно» = score >= порог
    print("\nAUC (cosine как предсказатель «релевантно»):")
    for threshold in [1, 2, 3]:
        y = (scores >= threshold).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  threshold={threshold}: пропущено (нет двух классов)")
            continue
        auc = roc_auc_score(y, cosines)
        print(f"  threshold={threshold}: AUC={auc:.4f} "
              f"(positives: {y.sum()}/{len(y)} = {y.mean()*100:.1f}%)")

    # Per-user корреляция
    print("\nКорреляция Spearman внутри каждого пользователя (n=39 талков):")
    by_user = {}
    for s in ref["scores"]:
        d = by_user.setdefault(s["user_id"], {"c": [], "s": []})
        u = user_emb.get(s["user_id"])
        t = talk_emb.get(s["topic_id"])
        if u is None or t is None:
            continue
        d["c"].append(float(np.dot(u, t)))
        d["s"].append(float(s["score"]))
    for uid, d in by_user.items():
        if len(d["c"]) < 5:
            continue
        sr, _ = spearmanr(d["c"], d["s"])
        print(f"  {uid[:8]}…: ρ={sr:+.4f} (n={len(d['c'])})")


if __name__ == "__main__":
    main()
