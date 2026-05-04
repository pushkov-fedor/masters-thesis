"""Стадия 3: обучение модели интереса f(persona_emb, talk_emb) -> preference.

Источники разметки:
- llm_ratings (default): preferences_matrix*.json с континуальными LLM-оценками.
- meetup_rsvps: реальные RSVP yes/no в Meetup-датасете. Положительные пары —
  фактический yes; отрицательные — другие events того же слота, не выбранные
  пользователем. Параллельная capacity-структура сохраняется.

Группировка train/val/test по persona_id (group split) — никакого leakage.

Сохраняет:
- data/models/<model_out>.pkl — обученная модель
- results/preference_model_metrics_<out>.json — метрики
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

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


def load_meetup_rsvp_pairs(rng):
    """Строит (user, event, label) пары из реальных Meetup RSVPs.

    label=1 — пользователь RSVP'нул yes на event в слоте с ≥2 параллельными
    events. label=0 — другие events того же слота, на которые user не RSVPил.

    Дополнительно возвращает slot_lookup для slot-aware accuracy@1 на test.

    Returns: prefs (list of dict с persona_id/talk_id/score), slot_lookup
    (dict: user_id -> list of (slot_id, chosen_talk_id, [parallel talks]))
    """
    raw_path = ROOT / "data" / "conferences" / "meetup_rsvp_raw_choices.json"
    conf_path = ROOT / "data" / "conferences" / "meetup_rsvp.json"

    raw = json.load(open(raw_path, encoding="utf-8"))
    conf = json.load(open(conf_path, encoding="utf-8"))

    # slot_id -> list of talk_ids (все параллельные events слота)
    slot_to_talks = defaultdict(list)
    for t in conf["talks"]:
        slot_to_talks[t["slot_id"]].append(t["id"])

    # slot_id -> {talk_id: set(user_ids who RSVP'd yes)}
    slot_real = defaultdict(lambda: defaultdict(set))
    for r in raw:
        for uid in r["yes_user_ids"]:
            slot_real[r["slot_id"]][r["talk_id"]].add(uid)

    prefs = []
    slot_lookup = defaultdict(list)  # user_id -> list of (slot, chosen, parallels)

    n_slots_used = 0
    n_pos = 0
    n_neg = 0
    for sid, talk_to_users in slot_real.items():
        parallel_talks = slot_to_talks.get(sid, [])
        if len(parallel_talks) < 2:
            continue
        n_slots_used += 1

        # Все user'ы, у которых есть yes на хоть один event этого слота
        users_with_yes = set()
        for uids in talk_to_users.values():
            users_with_yes.update(uids)

        for uid in users_with_yes:
            user_key = f"mu_{uid}"
            chosen_talks = [tid for tid, uids in talk_to_users.items() if uid in uids]
            if not chosen_talks:
                continue
            # Если user RSVP'нул на несколько events слота — каждый учим как positive
            for chosen in chosen_talks:
                prefs.append({
                    "persona_id": user_key,
                    "talk_id": chosen,
                    "score": 1.0,
                })
                n_pos += 1
            # Negatives: остальные события этого слота, на которые не yes
            negatives = [tid for tid in parallel_talks if tid not in chosen_talks]
            # Сэмплируем все негативы (датасет небольшой)
            for neg in negatives:
                prefs.append({
                    "persona_id": user_key,
                    "talk_id": neg,
                    "score": 0.0,
                })
                n_neg += 1
            # Сохраняем для slot-accuracy@1
            slot_lookup[user_key].append({
                "slot_id": sid,
                "chosen": chosen_talks,
                "parallel": parallel_talks,
            })

    print(f"Meetup RSVP source: {n_slots_used} slots used, "
          f"{n_pos} positives, {n_neg} negatives, ratio={n_neg/max(1,n_pos):.2f}")
    return prefs, slot_lookup


def slot_accuracy_at_1(scorer, slot_lookup, p_emb, t_emb, idx_personas):
    """Per-(user,slot) accuracy@1: для каждого user-слота, в котором user
    RSVPил yes на ≥1 event, считаем — argmax scorer-а по параллельным events
    совпадает ли с реальным выбором (yes-event).

    `scorer` — функция (user_emb, talk_emb_array) -> array of scores.

    Возвращает: dict с n, hits, accuracy, random_baseline.
    """
    test_users = set(idx_personas.tolist())
    hits = 0
    total = 0
    rb_sum = 0.0
    for user_key, slots in slot_lookup.items():
        if user_key not in test_users:
            continue
        if user_key not in p_emb:
            continue
        ue = p_emb[user_key]
        for s in slots:
            parallel = [tid for tid in s["parallel"] if tid in t_emb]
            if len(parallel) < 2:
                continue
            chosen_set = set(s["chosen"])
            t_arr = np.stack([t_emb[tid] for tid in parallel])
            scores = scorer(ue, t_arr)
            argmax_tid = parallel[int(np.argmax(scores))]
            if argmax_tid in chosen_set:
                hits += 1
            total += 1
            rb_sum += len(chosen_set) / len(parallel)
    if total == 0:
        return {"n": 0, "accuracy": float("nan"),
                "random_baseline": float("nan")}
    return {
        "n": total,
        "hits": hits,
        "accuracy": hits / total,
        "random_baseline": rb_sum / total,
    }


def make_model_scorer(model):
    def _f(ue, t_arr):
        feats = np.array([build_features(ue, t) for t in t_arr])
        return model.predict(feats)
    return _f


def cosine_scorer(ue, t_arr):
    return t_arr @ ue


class TorchSklearnWrapper:
    """Имитирует sklearn-API: .predict(X) -> 1d array. Module-level для pickle."""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(X).to(self.device)
            preds = []
            bs = 4096
            for i in range(0, len(t), bs):
                p = self.net(t[i:i + bs]).squeeze(-1).cpu().numpy()
                preds.append(p)
            return np.concatenate(preds)


def train_mlp_torch(prefs, p_emb, t_emb, train_personas, val_personas,
                    epochs=8, batch_size=512, lr=1e-3, device=None):
    """Обучает MLP батчами, без хранения полной матрицы X в памяти.

    Возвращает обёрнутую модель с .predict(X) для совместимости со sklearn-API.
    """
    import torch
    import torch.nn as nn
    if device is None:
        device = "cpu"
    train_set = set(train_personas.tolist())
    val_set = set(val_personas.tolist())
    train_pairs = [r for r in prefs
                   if r["persona_id"] in train_set
                   and r["persona_id"] in p_emb and r["talk_id"] in t_emb]
    val_pairs = [r for r in prefs
                 if r["persona_id"] in val_set
                 and r["persona_id"] in p_emb and r["talk_id"] in t_emb]
    print(f"  MLP train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    sample_feat = build_features(p_emb[train_pairs[0]["persona_id"]],
                                 t_emb[train_pairs[0]["talk_id"]])
    in_dim = len(sample_feat)
    print(f"  Input dim: {in_dim}")

    net = nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rng = np.random.default_rng(42)

    def batch_iter(pairs, shuffle=True):
        idxs = np.arange(len(pairs))
        if shuffle:
            rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            b = idxs[i:i + batch_size]
            xs = np.stack([build_features(p_emb[pairs[j]["persona_id"]],
                                          t_emb[pairs[j]["talk_id"]]) for j in b])
            ys = np.array([pairs[j]["score"] for j in b], dtype=np.float32)
            yield torch.from_numpy(xs).float().to(device), torch.from_numpy(ys).to(device)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in batch_iter(train_pairs, shuffle=True):
            opt.zero_grad()
            pred = net(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        net.eval()
        with torch.no_grad():
            v_loss = 0.0
            v_n = 0
            for xb, yb in batch_iter(val_pairs, shuffle=False):
                pred = net(xb).squeeze(-1)
                v_loss += loss_fn(pred, yb).item()
                v_n += 1
            v_loss /= max(1, v_n)
        print(f"  Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}  val_loss={v_loss:.4f}")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    return TorchSklearnWrapper(net, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["llm_ratings", "meetup_rsvps"],
                    default="llm_ratings",
                    help="llm_ratings — preferences_matrix*.json (LLM-разметка); "
                         "meetup_rsvps — реальные RSVP yes/no из Meetup.")
    ap.add_argument("--prefs", default="preferences_matrix",
                    help="(только source=llm_ratings) имя файла в data/ без .json")
    ap.add_argument("--personas-npz", default="personas",
                    help="имя файла в data/personas/ без _embeddings.npz "
                         "(для meetup_rsvps — meetup_users)")
    ap.add_argument("--talks-conference", default="mobius_2025_autumn",
                    help="имя конференции для talks_embeddings.npz "
                         "(для meetup_rsvps — meetup_rsvp)")
    ap.add_argument("--out", default="preference_model",
                    help="имя выходной модели в data/models/ без .pkl")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Если >0, случайная подвыборка из prefs до этого размера. Защита от OOM на больших датасетах.")
    ap.add_argument("--backend", choices=["histgb", "mlp_torch"], default="histgb",
                    help="histgb — HistGradientBoostingRegressor (быстро, но строит full X); "
                         "mlp_torch — батчевый MLP в PyTorch (для больших датасетов >150K пар).")
    ap.add_argument("--mlp-epochs", type=int, default=8)
    ap.add_argument("--mlp-batch", type=int, default=512)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    args = ap.parse_args()

    personas_npz = ROOT / "data" / "personas" / f"{args.personas_npz}_embeddings.npz"
    talks_npz = ROOT / "data" / "conferences" / f"{args.talks_conference}_embeddings.npz"
    model_out = ROOT / "data" / "models" / f"{args.out}.pkl"
    metrics_out = ROOT / "results" / f"preference_model_metrics_{args.out}.json"
    plot_out = ROOT / "results" / "plots" / f"20_preference_calibration_{args.out}.png"

    rng = np.random.default_rng(42)
    slot_lookup = None
    if args.source == "llm_ratings":
        pref_path = ROOT / "data" / f"{args.prefs}.json"
        with open(pref_path, encoding="utf-8") as f:
            prefs = json.load(f)
        print(f"Loaded {len(prefs)} preference pairs from {pref_path}")
    else:  # meetup_rsvps
        prefs, slot_lookup = load_meetup_rsvp_pairs(rng)
        print(f"Built {len(prefs)} (user, event, label) pairs from Meetup RSVPs")

    if args.max_pairs > 0 and len(prefs) > args.max_pairs:
        idx = rng.choice(len(prefs), args.max_pairs, replace=False)
        prefs = [prefs[i] for i in idx]
        print(f"Subsampled to {len(prefs)} pairs (--max-pairs={args.max_pairs})")

    print(f"Personas: {personas_npz}")
    print(f"Talks: {talks_npz}")
    print(f"Output: {model_out}")

    print("Loading personas npz...", flush=True)
    p_npz = np.load(personas_npz, allow_pickle=False)
    p_ids_list = p_npz["ids"].tolist()
    p_embs_arr = p_npz["embeddings"]
    print(f"  personas npz: {len(p_ids_list)} ids, embeddings shape {p_embs_arr.shape}", flush=True)
    print("Loading talks npz...", flush=True)
    t_npz = np.load(talks_npz, allow_pickle=False)
    t_ids_list = t_npz["ids"].tolist()
    t_embs_arr = t_npz["embeddings"]
    print(f"  talks npz: {len(t_ids_list)} ids, embeddings shape {t_embs_arr.shape}", flush=True)

    print("Building p_emb dict...", flush=True)
    p_emb = {}
    for i, pid in enumerate(p_ids_list):
        p_emb[str(pid)] = p_embs_arr[i]
    print(f"  p_emb: {len(p_emb)} keys", flush=True)
    print("Building t_emb dict...", flush=True)
    t_emb = {}
    for i, tid in enumerate(t_ids_list):
        t_emb[str(tid)] = t_embs_arr[i]
    print(f"  t_emb: {len(t_emb)} keys", flush=True)

    # Лёгкая проходка: собираем только y, cosines, persona_ids (без X).
    # X строим только если backend=histgb (полная матрица в памяти).
    y_list = []
    cosines_list = []
    persona_ids_list = []
    valid_prefs = []
    for r in prefs:
        pid, tid = r["persona_id"], r["talk_id"]
        if pid not in p_emb or tid not in t_emb:
            continue
        valid_prefs.append(r)
        y_list.append(r["score"])
        cosines_list.append(float(np.dot(p_emb[pid], t_emb[tid])))
        persona_ids_list.append(pid)

    y = np.array(y_list, dtype=np.float32)
    cosines = np.array(cosines_list, dtype=np.float32)
    persona_ids = np.array(persona_ids_list)
    prefs = valid_prefs
    del y_list, cosines_list, persona_ids_list, valid_prefs
    print(f"Valid pairs: {len(prefs)} / {len(y)}")

    if args.backend == "histgb":
        print("Building feature matrix (histgb backend)...")
        X = np.empty((len(prefs), 769), dtype=np.float32)
        for i, r in enumerate(prefs):
            X[i] = build_features(p_emb[r["persona_id"]], t_emb[r["talk_id"]])
        print(f"Feature matrix: {X.shape}, target: {y.shape}")
    else:
        X = None
        print(f"Skip building full X (mlp_torch builds features per batch). target: {y.shape}")
    print(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")
    n_unique_personas = len(np.unique(persona_ids))
    print(f"Unique personas: {n_unique_personas}")

    # Group split по persona_id: персоны не пересекаются между train/val/test.
    # Это устраняет leakage — модель тестируется на не виденных персонах,
    # а не на новых парах виденных персон с другими докладами.
    indices = np.arange(len(y))
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

    if args.backend == "mlp_torch":
        print(f"\nTraining MLP via PyTorch (batched, batch_size={args.mlp_batch})...")
        train_personas = persona_ids[train_idx]
        val_personas = persona_ids[val_idx]
        model = train_mlp_torch(prefs, p_emb, t_emb, train_personas, val_personas,
                                epochs=args.mlp_epochs, batch_size=args.mlp_batch,
                                lr=args.mlp_lr)
    else:
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
    def predict_idx(idx):
        if X is not None:
            return np.clip(model.predict(X[idx]), 0, 1)
        # mlp_torch: собираем фичи батчами
        out = np.empty(len(idx), dtype=np.float32)
        bs = 4096
        for i in range(0, len(idx), bs):
            sub = idx[i:i + bs]
            feats = np.stack([build_features(p_emb[prefs[j]["persona_id"]],
                                             t_emb[prefs[j]["talk_id"]]) for j in sub])
            out[i:i + len(sub)] = np.clip(model.predict(feats), 0, 1)
        return out

    y_val_pred = predict_idx(val_idx)
    y_test_pred = predict_idx(test_idx)
    y_train_pred = predict_idx(train_idx)

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

    # Дополнительно: AUC и slot-aware accuracy@1, если у нас бинарные метки
    is_binary = bool(set(np.unique(y).tolist()) <= {0.0, 1.0})
    if is_binary:
        from sklearn.metrics import roc_auc_score
        for split, idx, pred in [
            ("train", train_idx, y_train_pred),
            ("val", val_idx, y_val_pred),
            ("test", test_idx, y_test_pred),
        ]:
            target = y[idx]
            if len(np.unique(target)) == 2:
                metrics[split]["auc"] = float(roc_auc_score(target, pred))
        # slot-aware accuracy@1 на test (только для meetup_rsvps)
        if slot_lookup is not None:
            test_personas = persona_ids[test_idx]
            slot_acc = slot_accuracy_at_1(make_model_scorer(model), slot_lookup,
                                          p_emb, t_emb, test_personas)
            slot_acc_cos = slot_accuracy_at_1(cosine_scorer, slot_lookup,
                                              p_emb, t_emb, test_personas)
            metrics["test_slot_accuracy_at_1"] = slot_acc
            metrics["test_slot_accuracy_at_1_cosine_baseline"] = slot_acc_cos
            print(f"\nTEST slot accuracy@1 (real-RSVP grounded):")
            print(f"  Learned model: n={slot_acc['n']}  hits={slot_acc.get('hits','-')}  "
                  f"accuracy={slot_acc['accuracy']:.4f}")
            print(f"  Cosine baseline: hits={slot_acc_cos.get('hits','-')}  "
                  f"accuracy={slot_acc_cos['accuracy']:.4f}")
            print(f"  Random baseline: {slot_acc['random_baseline']:.4f}")
            print(f"  → Learned model lift over cosine: "
                  f"{(slot_acc['accuracy']-slot_acc_cos['accuracy'])*100:+.2f} п.п.")

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
