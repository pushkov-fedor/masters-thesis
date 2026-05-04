"""Transfer test: модель Scholar Inbox (LOO+popularity+Hadamard, 386-dim) → Mobius.

Цель: переносится ли learned-on-real-arxiv-ratings модель на конференц-домен JUG.

Входные данные:
- Модель: data/models/preference_model_si_loo.pkl (HistGB-Classifier).
- Mobius personas: e5-эмбеддинги русских описаний участников (как proxy user_emb,
  без LOO history — у синтетических personas её нет).
- Mobius talks: e5-эмбеддинги описаний докладов.
- Mobius popularity: fame_score (наша эвристика структурной популярности).
  Преобразуем в pseudo-pos-count: round(fame * 100) → log1p(...) для совместимости
  со шкалой Scholar Inbox (log1p(pos+neg) где pos+neg ~ 2-200).
- Ground-truth: continuous LLM-разметка (gpt-5.4-mini, behavioral prompt).

Метрики:
- Pearson/Spearman model_predictions vs ground-truth, сравнение с cosine baseline
  на тех же парах.

Запуск:
    python scripts/transfer_test_scholar_inbox_mobius.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("=== Transfer test: Scholar Inbox → Mobius ===\n")

    # 1) Load model bundle
    model_path = ROOT / "data" / "models" / "preference_model_si_loo.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feat_dim = bundle["feat_dim"]
    has_hadamard = bundle["hadamard"]
    print(f"Loaded model: {model_path}")
    print(f"  feat_dim={feat_dim}, hadamard={has_hadamard}")

    # 2) Load Mobius embeddings + fame
    p_npz = np.load(ROOT / "data" / "personas" / "personas_embeddings.npz")
    t_npz = np.load(ROOT / "data" / "conferences" / "mobius_2025_autumn_embeddings.npz")
    p_emb = {str(pid): p_npz["embeddings"][i].astype(np.float32)
             for i, pid in enumerate(p_npz["ids"].tolist())}
    t_emb = {str(tid): t_npz["embeddings"][i].astype(np.float32)
             for i, tid in enumerate(t_npz["ids"].tolist())}
    print(f"Mobius personas embeddings: {len(p_emb)}, dim={p_npz['embeddings'].shape[1]}")
    print(f"Mobius talks embeddings: {len(t_emb)}")

    with open(ROOT / "data" / "conferences" / "mobius_2025_autumn_fame.json") as f:
        fame_blob = json.load(f)
    fame = fame_blob.get("fame", {})
    print(f"Mobius fame: {len(fame)} entries, range "
          f"{min(fame.values()):.3f}..{max(fame.values()):.3f}")

    # 3) Load LLM ground-truth
    with open(ROOT / "data" / "preferences_matrix_continuous.json") as f:
        prefs = json.load(f)
    print(f"Mobius LLM ground-truth: {len(prefs)} pairs")

    # 4) Build features per pair, predict
    preds = []
    cosines = []
    targets = []
    skipped = 0
    for r in prefs:
        pid, tid = r["persona_id"], r["talk_id"]
        if pid not in p_emb or tid not in t_emb:
            skipped += 1
            continue
        ue = p_emb[pid]
        pe = t_emb[tid]
        cos = float(np.dot(ue, pe))
        # Pseudo-popularity: fame_score * 100 ~ pos+neg в Scholar Inbox шкале (median ~3).
        f = fame.get(tid, 0.0)
        log_pop = float(np.log1p(f * 100))
        feat = np.zeros(feat_dim, dtype=np.float32)
        feat[0] = cos
        feat[1] = log_pop
        if has_hadamard:
            feat[2:] = ue * pe
        preds.append(model.predict_proba(feat.reshape(1, -1))[0, 1])
        cosines.append(cos)
        targets.append(r["score"])

    preds = np.array(preds)
    cosines = np.array(cosines)
    targets = np.array(targets)
    print(f"\nEvaluated: {len(targets)} pairs, skipped {skipped}")
    print(f"Target  : mean={targets.mean():.3f}  std={targets.std():.3f}")
    print(f"Predict : mean={preds.mean():.3f}  std={preds.std():.3f}")
    print(f"Cosine  : mean={cosines.mean():.3f}  std={cosines.std():.3f}")

    # 5) Метрики
    pearson_model, _ = pearsonr(preds, targets)
    spearman_model, _ = spearmanr(preds, targets)
    pearson_cos, _ = pearsonr(cosines, targets)
    spearman_cos, _ = spearmanr(cosines, targets)

    print("\n=== Результаты transfer ===")
    print(f"Scholar-Inbox-trained model on Mobius:")
    print(f"  Pearson  = {pearson_model:+.4f}")
    print(f"  Spearman = {spearman_model:+.4f}")
    print(f"\nCosine baseline on Mobius (e5):")
    print(f"  Pearson  = {pearson_cos:+.4f}")
    print(f"  Spearman = {spearman_cos:+.4f}")
    print(f"\nLift learned over cosine:")
    print(f"  Pearson  {(pearson_model - pearson_cos) * 100:+.2f} п.п.")
    print(f"  Spearman {(spearman_model - spearman_cos) * 100:+.2f} п.п.")

    out = {
        "n_pairs": int(len(targets)),
        "skipped": int(skipped),
        "scholar_inbox_model": {
            "pearson": float(pearson_model), "spearman": float(spearman_model),
        },
        "cosine_baseline": {
            "pearson": float(pearson_cos), "spearman": float(spearman_cos),
        },
        "lift_pp": {
            "pearson": float((pearson_model - pearson_cos) * 100),
            "spearman": float((spearman_model - spearman_cos) * 100),
        },
        "target_stats": {"mean": float(targets.mean()), "std": float(targets.std())},
        "pred_stats": {"mean": float(preds.mean()), "std": float(preds.std())},
        "cos_stats": {"mean": float(cosines.mean()), "std": float(cosines.std())},
    }
    out_path = ROOT / "results" / "transfer_scholar_inbox_to_mobius.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {out_path}")

    print("\n=== Интерпретация ===")
    if pearson_model >= pearson_cos + 0.05:
        print("✓ Transfer РАБОТАЕТ: модель Scholar Inbox даёт больший Pearson, чем cosine.")
    elif abs(pearson_model - pearson_cos) < 0.05:
        print("○ Transfer ЧАСТИЧНЫЙ: модель ≈ cosine на Mobius.")
    else:
        print("✗ Transfer НЕ РАБОТАЕТ: модель хуже cosine.")


if __name__ == "__main__":
    main()
