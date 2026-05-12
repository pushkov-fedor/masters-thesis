"""Корреляция всех функций релевантности с LLM-judge эталоном.

Считает Spearman/Kendall корреляцию между каждым из 7 методов оценки релевантности
(cos, ABTT-1, BM25, гибрид α=0.5/0.7/0.9, reranker) и эталоном LLM-judge на 2500 парах.

Использование:
    cd experiments && source .venv/bin/activate
    python scripts/score_vs_judge.py --lang ru
    python scripts/score_vs_judge.py --lang en
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "data" / "personas" / "test_diversity"

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.doc_lens = np.array([len(d) for d in docs])
        self.avgdl = self.doc_lens.mean() if len(docs) > 0 else 0.0
        self.doc_freqs = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score_matrix(self, queries):
        n_q, n_d = len(queries), len(self.docs)
        out = np.zeros((n_q, n_d))
        for i, q in enumerate(queries):
            for j in range(n_d):
                freqs = self.doc_freqs[j]
                dl = self.doc_lens[j]
                s = 0.0
                for term in q:
                    if term in self.idf and freqs.get(term, 0) > 0:
                        f = freqs[term]
                        s += self.idf[term] * f * (self.k1 + 1) / (
                            f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                        )
                out[i, j] = s
        return out


def all_but_the_top(vecs, n_components=1):
    mu = vecs.mean(axis=0, keepdims=True)
    c = vecs - mu
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    top = Vt[:n_components]
    return c - c @ top.T @ top


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def make_talk_text(t):
    parts = [t["title"]]
    if t.get("category") and t["category"] != "Other":
        parts.append(f"[{t['category']}]")
    if t.get("abstract"):
        parts.append(t["abstract"])
    return " — ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ru", "en"], required=True)
    ap.add_argument("--personas", default=None, help="path to personas JSON (overrides default)")
    ap.add_argument("--talks", default=None, help="path to talks JSON (overrides default)")
    ap.add_argument("--judge", default=None, help="path to judge JSON (overrides default)")
    ap.add_argument("--label", default="", help="label for output filename suffix")
    args = ap.parse_args()
    lang = args.lang

    if lang == "ru":
        personas_file = Path(args.personas) if args.personas else TEST_DIR / "personas_ext.json"
        talks_file = Path(args.talks) if args.talks else TEST_DIR / "talks_ext.json"
        judge_file = Path(args.judge) if args.judge else TEST_DIR / "judge_ru.json"
        embedder_name = "intfloat/multilingual-e5-small"
        is_e5 = True
    else:
        personas_file = Path(args.personas) if args.personas else TEST_DIR / "personas_ext_en.json"
        talks_file = Path(args.talks) if args.talks else TEST_DIR / "talks_ext_en.json"
        judge_file = Path(args.judge) if args.judge else TEST_DIR / "judge_en.json"
        embedder_name = "BAAI/bge-large-en-v1.5"
        is_e5 = False

    with open(personas_file) as f:
        personas = json.load(f)
    with open(talks_file) as f:
        talks = json.load(f)
    with open(judge_file) as f:
        judge = json.load(f)

    print(f"Lang: {lang}")
    print(f"Personas: {len(personas)}, Talks: {len(talks)}, Judge pairs: {len(judge)}")

    # Build judge matrix indexed by (persona_idx, talk_idx)
    p_idx = {p["id"]: i for i, p in enumerate(personas)}
    t_idx = {t["id"]: i for i, t in enumerate(talks)}
    judge_matrix = np.zeros((len(personas), len(talks)))
    for entry in judge:
        if entry["persona_id"] in p_idx and entry["talk_id"] in t_idx:
            judge_matrix[p_idx[entry["persona_id"]], t_idx[entry["talk_id"]]] = entry["score"]

    # Embedding
    p_texts = [p["background"] for p in personas]
    t_texts = [make_talk_text(t) for t in talks]

    if is_e5:
        from src.embedder import embed_texts
        print(f"Embedding {len(p_texts)} personas (e5-small, query)...")
        p_emb = embed_texts(p_texts, kind="query")
        print(f"Embedding {len(t_texts)} talks (e5-small, passage)...")
        t_emb = embed_texts(t_texts, kind="passage")
    else:
        from sentence_transformers import SentenceTransformer
        QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
        embedder = SentenceTransformer(embedder_name, device="cpu")
        print(f"Embedding {len(p_texts)} personas (BGE-large-en)...")
        p_emb = embedder.encode([QUERY_PROMPT + t for t in p_texts],
                                normalize_embeddings=True, show_progress_bar=False)
        print(f"Embedding {len(t_texts)} talks (BGE-large-en)...")
        t_emb = embedder.encode(t_texts, normalize_embeddings=True, show_progress_bar=False)

    # Cos baseline
    pe, te = normalize(p_emb), normalize(t_emb)
    sim_cos = pe @ te.T

    # ABTT-1
    all_emb = np.vstack([p_emb, t_emb])
    proc = all_but_the_top(all_emb, n_components=1)
    pe_a, te_a = normalize(proc[:len(personas)]), normalize(proc[len(personas):])
    sim_abtt = pe_a @ te_a.T
    sim_abtt_norm = (sim_abtt - sim_abtt.min()) / (sim_abtt.max() - sim_abtt.min() + 1e-12)

    # BM25
    p_tokens = [tokenize(p["background"]) for p in personas]
    t_tokens = [tokenize(make_talk_text(t).replace("[", " ").replace("]", " ")) for t in talks]
    bm25 = BM25(t_tokens)
    bm25_raw = bm25.score_matrix(p_tokens)
    bm25_norm = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-12)

    # Hybrids
    sim_hybrid_05 = 0.5 * sim_abtt_norm + 0.5 * bm25_norm
    sim_hybrid_07 = 0.7 * sim_abtt_norm + 0.3 * bm25_norm
    sim_hybrid_09 = 0.9 * sim_abtt_norm + 0.1 * bm25_norm

    # Reranker
    print("Loading cross-encoder reranker...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(p_texts[i], t_texts[j]) for i in range(len(personas)) for j in range(len(talks))]
    print(f"Scoring {len(pairs)} pairs with reranker (this is the slow part)...")
    rerank_scores = reranker.predict(pairs, show_progress_bar=False)
    rerank_matrix = np.array(rerank_scores).reshape(len(personas), len(talks))

    # Flatten everything to 1D arrays of length n_pairs
    methods = {
        "cos": sim_cos.flatten(),
        "ABTT-1": sim_abtt.flatten(),
        "BM25": bm25_norm.flatten(),
        "hybrid α=0.5": sim_hybrid_05.flatten(),
        "hybrid α=0.7": sim_hybrid_07.flatten(),
        "hybrid α=0.9": sim_hybrid_09.flatten(),
        "reranker": rerank_matrix.flatten(),
    }
    judge_flat = judge_matrix.flatten()

    print()
    print("=" * 78)
    print(f"КОРРЕЛЯЦИЯ С LLM-JUDGE ({lang.upper()}, 2500 пар)")
    print("=" * 78)
    print(f"  {'method':<18} {'Spearman':>10} {'Kendall':>10}")
    print("-" * 42)
    rows = []
    for name, scores in methods.items():
        rho, _ = spearmanr(scores, judge_flat)
        tau, _ = kendalltau(scores, judge_flat)
        rows.append((name, rho, tau))
        print(f"  {name:<18} {rho:>10.4f} {tau:>10.4f}")
    print()

    # Save matrices for further analysis
    suffix = f"_{args.label}" if args.label else ""
    out_path = TEST_DIR / f"all_scores_{lang}{suffix}.npz"
    np.savez(out_path,
             cos=sim_cos, abtt=sim_abtt, bm25=bm25_norm,
             hybrid_05=sim_hybrid_05, hybrid_07=sim_hybrid_07, hybrid_09=sim_hybrid_09,
             reranker=rerank_matrix, judge=judge_matrix)
    print(f"Saved all matrices to {out_path}")


if __name__ == "__main__":
    main()
