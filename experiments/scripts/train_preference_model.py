"""Стадия 3: обучение модели интереса f(persona_emb, talk_emb) -> preference.

По умолчанию обучает Mobius-модель из preferences_matrix.json. Через CLI
параметризован для Demo Day и других конференций.

Группировка train/val/test по persona_id (group split) — никакого leakage.

Сохраняет:
- data/models/<model_out>.pkl — обученная модель
- results/preference_model_metrics_<conference>.json — метрики
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]


def build_features(persona_emb, talk_emb):
    """769-мерный вектор признаков."""
    cosine = float(np.dot(persona_emb, talk_emb))
    diff = persona_emb - talk_emb
    feat = np.concatenate([
        persona_emb,
        talk_emb,
        [cosine],
    ])
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", default="preferences_matrix",
                    help="имя файла в data/ без .json (preferences_matrix или preferences_matrix_demoday)")
    ap.add_argument("--personas-npz", default="personas",
                    help="имя файла в data/personas/ без _embeddings.npz")
    ap.add_argument("--talks-conference", default="mobius_2025_autumn",
                    help="имя конференции для talks_embeddings.npz")
    ap.add_argument("--out", default="preference_model",
                    help="имя выходной модели в data/models/ без .pkl")
    args = ap.parse_args()

    pref_path = ROOT / "data" / f"{args.prefs}.json"
    personas_npz = ROOT / "data" / "personas" / f"{args.personas_npz}_embeddings.npz"
    talks_npz = ROOT / "data" / "conferences" / f"{args.talks_conference}_embeddings.npz"
    model_out = ROOT / "data" / "models" / f"{args.out}.pkl"
    metrics_out = ROOT / "results" / f"preference_model_metrics_{args.out}.json"
    plot_out = ROOT / "results" / "plots" / f"20_preference_calibration_{args.out}.png"

    with open(pref_path, encoding="utf-8") as f:
        prefs = json.load(f)
    print(f"Loaded {len(prefs)} preference pairs from {pref_path}")
    print(f"Personas: {personas_npz}")
    print(f"Talks: {talks_npz}")
    print(f"Output: {model_out}")

    p_npz = np.load(personas_npz, allow_pickle=False)
    t_npz = np.load(talks_npz, allow_pickle=False)
    p_emb = {pid: p_npz["embeddings"][i] for i, pid in enumerate(p_npz["ids"])}
    t_emb = {tid: t_npz["embeddings"][i] for i, tid in enumerate(t_npz["ids"])}

    X = []
    y = []
    cosines = []
    persona_ids = []
    for r in prefs:
        pid, tid = r["persona_id"], r["talk_id"]
        if pid not in p_emb or tid not in t_emb:
            continue
        X.append(build_features(p_emb[pid], t_emb[tid]))
        y.append(r["score"])
        cosines.append(float(np.dot(p_emb[pid], t_emb[tid])))
        persona_ids.append(pid)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    cosines = np.array(cosines, dtype=np.float32)
    persona_ids = np.array(persona_ids)
    print(f"Feature matrix: {X.shape}, target: {y.shape}")
    print(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")
    n_unique_personas = len(np.unique(persona_ids))
    print(f"Unique personas: {n_unique_personas}")

    # Group split по persona_id: персоны не пересекаются между train/val/test.
    # Это устраняет leakage — модель тестируется на не виденных персонах,
    # а не на новых парах виденных персон с другими докладами.
    indices = np.arange(len(X))
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_full_idx, test_idx = next(gss_test.split(indices, groups=persona_ids))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss_val.split(
        indices[train_full_idx], groups=persona_ids[train_full_idx]
    ))
    train_idx = train_full_idx[train_idx]
    val_idx = train_full_idx[val_idx]
    print(f"Group splits (by persona_id): train={len(train_idx)} ({len(np.unique(persona_ids[train_idx]))} personas), "
          f"val={len(val_idx)} ({len(np.unique(persona_ids[val_idx]))} personas), "
          f"test={len(test_idx)} ({len(np.unique(persona_ids[test_idx]))} personas)")
    # Sanity check: пересечений быть не должно
    train_set = set(persona_ids[train_idx])
    val_set = set(persona_ids[val_idx])
    test_set = set(persona_ids[test_idx])
    assert not (train_set & test_set), "leak: train ∩ test"
    assert not (train_set & val_set), "leak: train ∩ val"
    assert not (val_set & test_set), "leak: val ∩ test"

    print("\nTraining HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X[train_idx], y[train_idx])
    print(f"Training done. Iterations: {model.n_iter_}")

    # Predictions
    y_val_pred = np.clip(model.predict(X[val_idx]), 0, 1)
    y_test_pred = np.clip(model.predict(X[test_idx]), 0, 1)
    y_train_pred = np.clip(model.predict(X[train_idx]), 0, 1)

    metrics = {}
    for split, idx, pred in [
        ("train", train_idx, y_train_pred),
        ("val", val_idx, y_val_pred),
        ("test", test_idx, y_test_pred),
    ]:
        target = y[idx]
        cos = cosines[idx]
        pearson_r, _ = pearsonr(pred, target)
        spearman_r, _ = spearmanr(pred, target)
        mse = mean_squared_error(target, pred)
        mae = mean_absolute_error(target, pred)
        # baseline: cosine как предсказатель
        baseline_pearson, _ = pearsonr(cos, target)
        baseline_spearman, _ = spearmanr(cos, target)
        metrics[split] = {
            "n": len(idx),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "mse": float(mse),
            "mae": float(mae),
            "baseline_cosine_pearson": float(baseline_pearson),
            "baseline_cosine_spearman": float(baseline_spearman),
        }
        print(f"\n{split.upper()} (n={len(idx)}):")
        print(f"  Model — Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}")
        print(f"          MSE={mse:.4f}, MAE={mae:.4f}")
        print(f"  Baseline cosine — Pearson r={baseline_pearson:.4f}, Spearman ρ={baseline_spearman:.4f}")

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {metrics_out}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"WROTE: {model_out} ({model_out.stat().st_size // 1024}KB)")

    # Plot calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y[val_idx], y_val_pred, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    ax.set_xlabel("Target (LLM-rated preference)")
    ax.set_ylabel("Model prediction")
    ax.set_title(f"Validation calibration\nSpearman ρ={metrics['val']['spearman_r']:.3f}, "
                 f"Pearson r={metrics['val']['pearson_r']:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(cosines[val_idx], y[val_idx], alpha=0.3, s=10, color="orange")
    ax.set_xlabel("Cosine(persona, talk)")
    ax.set_ylabel("Target (LLM-rated preference)")
    ax.set_title(f"Cosine baseline on val\nPearson r={metrics['val']['baseline_cosine_pearson']:.3f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"WROTE: {plot_out}")


if __name__ == "__main__":
    main()
