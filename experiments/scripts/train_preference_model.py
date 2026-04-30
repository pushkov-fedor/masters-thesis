"""Стадия 3: обучение параметрической модели выбора f(persona_emb, talk_emb) -> preference.

Признаки: persona_emb (384) ⊕ talk_emb (384) ⊕ cosine + |diff|. Итого 769 dim.
Цель: preference ∈ [0, 1] из preferences_matrix.json (12000 пар, gpt-4o-mini ground truth).
Модель: sklearn HistGradientBoostingRegressor (без libomp, нативно).

Сохраняем:
- data/models/preference_model.pkl — обученная модель
- results/preference_model_metrics.json — Spearman ρ, Pearson r, MSE на val
- results/preference_calibration.png — scatter prediction vs target на val
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
PREF_PATH = ROOT / "data" / "preferences_matrix.json"
PERSONAS_NPZ = ROOT / "data" / "personas" / "personas_embeddings.npz"
TALKS_NPZ = ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz"
MODEL_OUT = ROOT / "data" / "models" / "preference_model.pkl"
METRICS_OUT = ROOT / "results" / "preference_model_metrics.json"
PLOT_OUT = ROOT / "results" / "plots" / "20_preference_calibration.png"


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
    with open(PREF_PATH, encoding="utf-8") as f:
        prefs = json.load(f)
    print(f"Loaded {len(prefs)} preference pairs")

    p_npz = np.load(PERSONAS_NPZ, allow_pickle=False)
    t_npz = np.load(TALKS_NPZ, allow_pickle=False)
    p_emb = {pid: p_npz["embeddings"][i] for i, pid in enumerate(p_npz["ids"])}
    t_emb = {tid: t_npz["embeddings"][i] for i, tid in enumerate(t_npz["ids"])}

    X = []
    y = []
    cosines = []
    keys = []
    for r in prefs:
        pid, tid = r["persona_id"], r["talk_id"]
        if pid not in p_emb or tid not in t_emb:
            continue
        X.append(build_features(p_emb[pid], t_emb[tid]))
        y.append(r["score"])
        cosines.append(float(np.dot(p_emb[pid], t_emb[tid])))
        keys.append((pid, tid))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    cosines = np.array(cosines, dtype=np.float32)
    print(f"Feature matrix: {X.shape}, target: {y.shape}")
    print(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")

    # Разделяем по парам (8000/2000/2000 примерно)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)
    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

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

    METRICS_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {METRICS_OUT}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"WROTE: {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024}KB)")

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
    fig.savefig(PLOT_OUT, dpi=150)
    plt.close(fig)
    print(f"WROTE: {PLOT_OUT}")


if __name__ == "__main__":
    main()
