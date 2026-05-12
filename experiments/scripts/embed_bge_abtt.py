"""Эмбеддинги через BGE-large-en + ABTT-1 для основного эксперимента (EN-pipeline).

Аудит функции релевантности (docs/spikes/spike_relevance_function_audit.md) выбрал
победителем `BAAI/bge-large-en-v1.5 + ABTT-1` (gap +0.316, Spearman 0.327, Vendi 89%).

Скрипт:
1. Загружает программу конференции и пул персон.
2. Эмбеддит тексты через BGE-large-en (1024-dim) с правильными префиксами:
   - personas (background) — query-prefix
     "Represent this sentence for searching relevant passages: ";
   - talks (title [category] abstract) — без префикса.
3. Применяет ABTT-1 (Mu, Bhat, Viswanath 2018) к vstack(personas, talks):
   вычитание среднего + ортогональная проекция к топ-1 PCA направлению,
   единым SVD по всему пулу — чтобы оба пула жили в одном postprocessed-пространстве.
4. Финальная L2-нормировка → cos(a,b) = dot(a,b).
5. Сохраняет два .npz файла (ids + 1024-dim float32 normalized vectors).

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/embed_bge_abtt.py \\
        --conference mobius_2025_autumn_en \\
        --personas personas_mobius_en
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def all_but_the_top(vecs: np.ndarray, n_components: int = 1) -> np.ndarray:
    """ABTT (Mu, Bhat, Viswanath, ICLR 2018):

    1) центрирование (вычитание среднего по пулу),
    2) ортогональная проекция к top-n_components PCA-направлениям.

    Топ-PCA направления у контекстуальных эмбеддингов кодируют доменный фон
    (общее у всех векторов), а не семантику; их удаление расширяет узкий конус.
    """
    mu = vecs.mean(axis=0, keepdims=True)
    centered = vecs - mu
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    top = Vt[:n_components]
    return centered - centered @ top.T @ top


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conference", required=True,
                    help="имя файла в data/conferences/ без .json")
    ap.add_argument("--personas", required=True,
                    help="имя файла в data/personas/ без .json")
    ap.add_argument("--n-abtt", type=int, default=1,
                    help="сколько PCA-направлений удалять (default 1)")
    args = ap.parse_args()

    conf_json = ROOT / "data" / "conferences" / f"{args.conference}.json"
    pers_json = ROOT / "data" / "personas" / f"{args.personas}.json"
    conf_out = ROOT / "data" / "conferences" / f"{args.conference}_embeddings.npz"
    pers_out = ROOT / "data" / "personas" / f"{args.personas}_embeddings.npz"

    assert conf_json.exists(), f"missing {conf_json}"
    assert pers_json.exists(), f"missing {pers_json}"

    with open(conf_json, encoding="utf-8") as f:
        prog = json.load(f)
    with open(pers_json, encoding="utf-8") as f:
        personas = json.load(f)

    talks = prog["talks"]
    t_ids = [t["id"] for t in talks]
    t_texts = [make_talk_text(t) for t in talks]

    p_ids = [p["id"] for p in personas]
    p_texts = [p.get("background") or p.get("profile") or "" for p in personas]

    print(f"Conference: {args.conference}  ({len(talks)} talks)")
    print(f"Personas:   {args.personas}     ({len(personas)} personas)")
    print(f"Model:      {MODEL_NAME}")
    print(f"ABTT n:     {args.n_abtt}")
    print()

    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"  loaded in {time.time() - t0:.1f}s")
    print()

    print(f"Encoding {len(p_texts)} personas (query prefix)...")
    t0 = time.time()
    p_emb = model.encode(
        [QUERY_PROMPT + t for t in p_texts],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    ).astype(np.float32)
    print(f"  shape={p_emb.shape}  time={time.time() - t0:.1f}s")
    print()

    print(f"Encoding {len(t_texts)} talks (no prefix)...")
    t0 = time.time()
    t_emb = model.encode(
        t_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    ).astype(np.float32)
    print(f"  shape={t_emb.shape}  time={time.time() - t0:.1f}s")
    print()

    print(f"Applying ABTT-{args.n_abtt} on vstack(personas, talks) ...")
    t0 = time.time()
    all_emb = np.vstack([p_emb, t_emb])
    all_proc = all_but_the_top(all_emb, n_components=args.n_abtt)
    p_proc = l2_normalize(all_proc[: len(personas)]).astype(np.float32)
    t_proc = l2_normalize(all_proc[len(personas):]).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  p_proc.shape={p_proc.shape}, t_proc.shape={t_proc.shape}")
    print()

    # Sanity: после ABTT+normalize пары всё ещё в [-1, 1]
    cos_pt = p_proc @ t_proc.T
    print(f"Sanity: cos(persona, talk) range "
          f"[{cos_pt.min():.3f}, {cos_pt.max():.3f}], "
          f"mean={cos_pt.mean():.3f}")
    cos_pp_iu = np.triu_indices(len(personas), k=1)
    cos_pp = (p_proc @ p_proc.T)[cos_pp_iu]
    print(f"        cos(persona,persona) [{cos_pp.min():.3f}, {cos_pp.max():.3f}], "
          f"mean={cos_pp.mean():.3f}")
    print()

    # `embeddings` = ABTT-обработанные нормализованные векторы (используются
    # Conference.load и основным симулятором). `embeddings_raw` = до-ABTT
    # нормированные векторы, сохранены для диагностических метрик.
    np.savez(conf_out,
             ids=np.array(t_ids),
             embeddings=t_proc,
             embeddings_raw=t_emb)
    print(f"WROTE: {conf_out}  (ABTT shape={t_proc.shape}, raw shape={t_emb.shape})")
    np.savez(pers_out,
             ids=np.array(p_ids),
             embeddings=p_proc,
             embeddings_raw=p_emb)
    print(f"WROTE: {pers_out}  (ABTT shape={p_proc.shape}, raw shape={p_emb.shape})")


if __name__ == "__main__":
    main()
