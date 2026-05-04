"""Honest preference-model training on Scholar Inbox with LOO + popularity + Hadamard.

Архитектура (по совету recsys-эксперта):
- user_emb строится **leave-one-out** для каждой пары: исключаем target paper
  из его positives → centroid(остатка) → нормализация. Без leakage.
- features: cos(u, p) + log_pop(p) + Hadamard u⊙p (поэлементное произведение).
- target: binary {0, 1}.
- model: HistGradientBoostingClassifier (бинарный, native AUC).
- baseline: LOO cosine на тех же тестах.

Group split по user_id; на test — те же LOO features (test users не видены при обучении).

Запуск:
    python scripts/train_scholar_inbox_loo.py --out preference_model_si_loo

Опционально:
    --max-train-pairs 200000  — подвыборка для скорости.
    --hadamard / --no-hadamard — отключить u⊙p (для ablation).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def loo_user_emb(positives_idx, exclude_idx, paper_embs):
    """LOO: centroid(positives \ {exclude_idx}), нормализован.

    positives_idx: list[int] — индексы positively-rated papers user'а.
    exclude_idx: int — индекс target paper, который надо исключить.
    paper_embs: (N, d) array.

    Returns: (d,) emb или zeros если положительных < 2.
    """
    rest = [i for i in positives_idx if i != exclude_idx]
    if len(rest) < 1:
        return np.zeros(paper_embs.shape[1], dtype=np.float32)
    c = paper_embs[rest].mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def build_features_loo(user_emb, paper_emb, log_pop, use_hadamard=True):
    cos = float(np.dot(user_emb, paper_emb))
    parts = [[cos, log_pop]]
    if use_hadamard:
        parts.append(user_emb * paper_emb)
    return np.concatenate([np.array(parts[0], dtype=np.float32),
                           *parts[1:]] if use_hadamard
                          else [np.array(parts[0], dtype=np.float32)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="preference_model_si_loo")
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--hadamard", dest="hadamard", action="store_true", default=True)
    ap.add_argument("--no-hadamard", dest="hadamard", action="store_false")
    ap.add_argument("--gbm-iter", type=int, default=400)
    args = ap.parse_args()

    print("Loading data...", flush=True)
    p_npz = np.load(ROOT / "data" / "personas" / "scholar_inbox_users_embeddings.npz")
    t_npz = np.load(ROOT / "data" / "conferences" / "scholar_inbox_papers_embeddings.npz")
    paper_ids_arr = t_npz["ids"]
    paper_embs = t_npz["embeddings"].astype(np.float32)
    paper_idx = {str(aid): i for i, aid in enumerate(paper_ids_arr.tolist())}
    print(f"  papers: {paper_embs.shape}", flush=True)

    with open(ROOT / "data" / "scholar_inbox_user_history.json") as f:
        user_history = json.load(f)
    print(f"  user_history: {len(user_history)} users", flush=True)

    with open(ROOT / "data" / "scholar_inbox_paper_popularity.json") as f:
        pop_meta = json.load(f)
    log_pop = {aid: float(np.log1p(meta["pos"] + meta["neg"]))
               for aid, meta in pop_meta.items()}
    print(f"  popularity loaded for {len(log_pop)} papers", flush=True)

    with open(ROOT / "data" / "preferences_matrix_scholar_inbox.json") as f:
        prefs = json.load(f)
    print(f"  prefs: {len(prefs)}", flush=True)

    # Pre-cache user positives indices.
    user_pos_idx = {}
    for uid, hist in user_history.items():
        idxs = [paper_idx[aid] for aid in hist["positives"] if aid in paper_idx]
        user_pos_idx[uid] = idxs
    print(f"  cached user positives: {len(user_pos_idx)}", flush=True)

    # Group split by user.
    user_ids = sorted(set(r["persona_id"] for r in prefs))
    rng = np.random.default_rng(42)
    user_ids_arr = np.array(user_ids)
    rng.shuffle(user_ids_arr)
    n_test = int(0.2 * len(user_ids_arr))
    n_val = int(0.1 * len(user_ids_arr))
    test_users = set(user_ids_arr[:n_test].tolist())
    val_users = set(user_ids_arr[n_test:n_test + n_val].tolist())
    train_users = set(user_ids_arr[n_test + n_val:].tolist())
    print(f"  groups: train={len(train_users)}, val={len(val_users)}, test={len(test_users)}",
          flush=True)

    train_pairs = [r for r in prefs if r["persona_id"] in train_users
                   and r["talk_id"] in paper_idx
                   and r["persona_id"] in user_pos_idx]
    val_pairs = [r for r in prefs if r["persona_id"] in val_users
                 and r["talk_id"] in paper_idx
                 and r["persona_id"] in user_pos_idx]
    test_pairs = [r for r in prefs if r["persona_id"] in test_users
                  and r["talk_id"] in paper_idx
                  and r["persona_id"] in user_pos_idx]
    print(f"  pairs: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}",
          flush=True)

    # Subsample train for speed (LOO is per-pair → expensive).
    if args.max_train_pairs > 0 and len(train_pairs) > args.max_train_pairs:
        idx = rng.choice(len(train_pairs), args.max_train_pairs, replace=False)
        train_pairs = [train_pairs[i] for i in idx]
        print(f"  subsampled train_pairs to {len(train_pairs)}", flush=True)

    # Build features
    feat_dim = 2 + (paper_embs.shape[1] if args.hadamard else 0)
    print(f"  feature dim = {feat_dim} (cos + log_pop + {'Hadamard ' + str(paper_embs.shape[1]) if args.hadamard else 'no Hadamard'})", flush=True)

    def build_split(pairs, label="?"):
        n = len(pairs)
        X = np.zeros((n, feat_dim), dtype=np.float32)
        y = np.zeros(n, dtype=np.int8)
        cos_only = np.zeros(n, dtype=np.float32)
        valid_mask = np.zeros(n, dtype=bool)
        t0 = time.time()
        for k, r in enumerate(pairs):
            tid = r["talk_id"]
            uid = r["persona_id"]
            t_idx = paper_idx[tid]
            pos_idxs = user_pos_idx.get(uid, [])
            ue = loo_user_emb(pos_idxs, t_idx, paper_embs)
            if np.all(ue == 0):
                continue  # skip if no other positives
            pe = paper_embs[t_idx]
            lp = log_pop.get(tid, 0.0)
            cos = float(np.dot(ue, pe))
            X[k, 0] = cos
            X[k, 1] = lp
            if args.hadamard:
                X[k, 2:] = ue * pe
            y[k] = int(r["score"])
            cos_only[k] = cos
            valid_mask[k] = True
            if (k + 1) % 50000 == 0:
                print(f"    {label}: {k+1}/{n} in {time.time()-t0:.0f}s", flush=True)
        return X[valid_mask], y[valid_mask], cos_only[valid_mask]

    print("Building train features...", flush=True)
    X_tr, y_tr, cos_tr = build_split(train_pairs, "train")
    print(f"  train: {X_tr.shape}  pos_rate={y_tr.mean():.3f}", flush=True)
    print("Building val features...", flush=True)
    X_v, y_v, cos_v = build_split(val_pairs, "val")
    print(f"  val: {X_v.shape}  pos_rate={y_v.mean():.3f}", flush=True)
    print("Building test features...", flush=True)
    X_te, y_te, cos_te = build_split(test_pairs, "test")
    print(f"  test: {X_te.shape}  pos_rate={y_te.mean():.3f}", flush=True)

    # Cosine baseline metrics (LOO honest)
    print("\n=== Honest cosine LOO baseline ===")
    for name, c, y in [("train", cos_tr, y_tr), ("val", cos_v, y_v), ("test", cos_te, y_te)]:
        try:
            auc = roc_auc_score(y, c)
        except Exception:
            auc = float("nan")
        pr, _ = pearsonr(c, y)
        sp, _ = spearmanr(c, y)
        print(f"  {name}: AUC={auc:.4f}  Pearson={pr:+.4f}  Spearman={sp:+.4f}")

    # GBM Classifier
    print(f"\nTraining HistGradientBoostingClassifier (max_iter={args.gbm_iter})...", flush=True)
    model = HistGradientBoostingClassifier(
        max_iter=args.gbm_iter, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    )
    model.fit(X_tr, y_tr)
    print(f"  iterations: {model.n_iter_}", flush=True)

    print("\n=== Model results ===")
    metrics = {"hadamard": args.hadamard, "gbm_iter": args.gbm_iter,
               "max_train_pairs": args.max_train_pairs}
    for name, X, y, c in [("train", X_tr, y_tr, cos_tr),
                          ("val", X_v, y_v, cos_v),
                          ("test", X_te, y_te, cos_te)]:
        prob = model.predict_proba(X)[:, 1]
        auc_m = roc_auc_score(y, prob)
        auc_c = roc_auc_score(y, c)
        pr_m, _ = pearsonr(prob, y.astype(np.float32))
        pr_c, _ = pearsonr(c, y.astype(np.float32))
        metrics[name] = {
            "n": int(len(y)), "pos_rate": float(y.mean()),
            "model_auc": float(auc_m), "cosine_auc": float(auc_c),
            "model_pearson": float(pr_m), "cosine_pearson": float(pr_c),
            "lift_auc_pp": float((auc_m - auc_c) * 100),
            "lift_pearson_pp": float((pr_m - pr_c) * 100),
        }
        print(f"  {name}: model AUC={auc_m:.4f} (cos={auc_c:.4f}, lift={(auc_m-auc_c)*100:+.2f} п.п.)  "
              f"Pearson={pr_m:+.4f} (cos={pr_c:+.4f})")

    out_path = ROOT / "data" / "models" / f"{args.out}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feat_dim": feat_dim,
                     "hadamard": args.hadamard}, f)
    print(f"\nWROTE: {out_path}")
    metrics_path = ROOT / "results" / f"preference_model_metrics_{args.out}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"WROTE: {metrics_path}")


if __name__ == "__main__":
    main()
