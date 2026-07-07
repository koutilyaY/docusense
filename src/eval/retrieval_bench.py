"""
Retrieval benchmark on CUAD: baseline vs hybrid vs reranker vs better embeddings.

This extends src/eval/retrieval_eval.py. That script established the honest
baseline -- general-purpose MiniLM dense retrieval on real CUAD contracts,
hit-rate@5 = 0.37, @10 = 0.46 over 458 clause questions. This script keeps that
exact evaluation (same contracts, same chunking, same gold-span matching) and
adds real retrieval engineering on top, measuring each variant on the same gold
set so the comparison is apples-to-apples:

  baseline   : dense retrieval, all-MiniLM-L6-v2                (the existing floor)
  +hybrid    : BM25 (sparse) + MiniLM dense, fused with RRF
  +reranker  : hybrid top-N, reranked by a cross-encoder
  +bge       : dense retrieval with BAAI/bge-small-en-v1.5 (stronger embedder)
  +bge_hybrid: BM25 + BGE dense, RRF
  +bge_rerank: BGE hybrid top-N, reranked by the cross-encoder

Optionally (--clause-chunking) it also re-runs the winning stack under a
clause-aware splitter to see whether splitting on clause boundaries helps.

Everything is retrieval/rerank only -- no LLM, no Ollama, no API key. The gold
labels are real CUAD expert clause spans; a variant "hits" at k if any of its
top-k chunks contains the gold span text. Same honest floor, same seed, so any
machine reproduces the same table.

Usage:
    python src/eval/retrieval_bench.py --contracts 50
    python src/eval/retrieval_bench.py --contracts 50 --clause-chunking
    python src/eval/retrieval_bench.py --contracts 20 --variants baseline hybrid reranker
"""
import argparse
import datetime
import json
import os
import random
import sys

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss is required: pip install faiss-cpu")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("rank_bm25 is required: pip install rank-bm25")

from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SPLIT_FILES = {
    "test": os.path.join(ROOT, "data", "cuad", "test.json"),
    "CUADv1": os.path.join(ROOT, "data", "cuad", "CUADv1.json"),
}

# Same chunking the baseline (and the RAG pipeline) uses.
BASE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "],
)

# Clause-aware variant: prefer breaking on numbered-section / clause markers and
# sentence ends before falling back to raw characters. Same size budget so the
# only thing that changes is *where* the cuts land.
CLAUSE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=[
        "\n\n",
        "\n",
        ";",
        ". ",
        " ",
    ],
    keep_separator=True,
)

BASE_EMB = "all-MiniLM-L6-v2"
BGE_EMB = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BGE retrieval quality depends on a query instruction prefix; the model card
# specifies this exact string for retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def norm(s: str) -> str:
    """Collapse whitespace so span-in-chunk matching survives re-wrapping."""
    return " ".join(s.split()).lower()


def load_cuad(split: str):
    path = SPLIT_FILES[split]
    if not os.path.exists(path):
        sys.exit(f"CUAD file not found: {path}\nRun: python data/cuad/download_cuad.py")
    with open(path) as f:
        return json.load(f)["data"]


def tokenize(text: str):
    """Cheap word tokenizer for BM25 -- lowercase alphanumeric runs."""
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def rrf_fuse(rank_lists, k_rrf=60):
    """Reciprocal Rank Fusion over several ranked id lists.

    Each input is an ordered list of chunk ids (best first). Returns chunk ids
    ordered by summed 1/(k_rrf + rank). Standard RRF; k_rrf=60 is the common
    default from the original Cormack et al. paper.
    """
    scores = {}
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


class ContractCase:
    """One contract: its chunks, precomputed representations, and gold questions.

    Built once per contract and reused across every variant so the comparison is
    apples-to-apples (identical chunks, identical gold set).
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.norm_chunks = [norm(c) for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c) for c in chunks])
        self.dense = {}   # model_name -> faiss index over chunk embeddings
        self.questions = []
        self.gold_norms = []

    def hit(self, ordered_ids, k, gold):
        for cid in ordered_ids[:k]:
            if cid >= 0 and gold in self.norm_chunks[cid]:
                return True
        return False


def build_cases(contracts, splitter, max_ctx_chars):
    """Chunk every contract and collect its answerable gold clause questions."""
    cases = []
    for c in contracts:
        para = c["paragraphs"][0]
        context = para["context"]
        if max_ctx_chars and len(context) > max_ctx_chars:
            context = context[:max_ctx_chars]
        chunks = splitter.split_text(context)
        if not chunks:
            continue
        case = ContractCase(chunks)
        ncontext = norm(context)
        for qa in para["qas"]:
            if not qa["answers"]:
                continue
            ans = qa["answers"][0]["text"].strip()
            if len(ans) < 3:
                continue
            g = norm(ans)
            if g not in ncontext:
                continue  # span sits past the ctx cap; skip to stay honest
            case.questions.append(qa["question"])
            case.gold_norms.append(g)
        if case.questions:
            cases.append(case)
    return cases


def embed_chunks(cases, model, model_name, is_bge=False):
    """Attach a FAISS index of chunk embeddings (for `model_name`) to each case."""
    for case in cases:
        emb = model.encode(case.chunks, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        case.dense[model_name] = idx


def dense_ranks(case, model, model_name, questions, is_bge, topn):
    """Top-`topn` chunk ids per question from the dense index for `model_name`."""
    qs = [BGE_QUERY_PREFIX + q for q in questions] if is_bge else questions
    q_emb = model.encode(qs, normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype="float32")
    n = min(topn, len(case.chunks))
    _, ids = case.dense[model_name].search(q_emb, n)
    return [[int(i) for i in row if i >= 0] for row in ids]


def bm25_ranks(case, questions, topn):
    """Top-`topn` chunk ids per question from BM25."""
    out = []
    n = min(topn, len(case.chunks))
    for q in questions:
        scores = case.bm25.get_scores(tokenize(q))
        order = np.argsort(scores)[::-1][:n]
        out.append([int(i) for i in order])
    return out


def score_variant(cases, ordered_by_case, ks):
    """Given per-case, per-question ordered chunk-id lists, compute hit-rate@k."""
    hits = {k: 0 for k in ks}
    total = 0
    for case, ordered_lists in zip(cases, ordered_by_case):
        for gold, ordered in zip(case.gold_norms, ordered_lists):
            total += 1
            for k in ks:
                if case.hit(ordered, k, gold):
                    hits[k] += 1
    return {k: hits[k] / total for k in ks}, total


def run(cases, ks, variants, topn, rerank_pool, seed):
    """Compute every requested variant on the shared set of cases."""
    results = {}
    max_k = max(ks)

    # --- shared cheap representations ---------------------------------------
    bm25_by_case = None
    minilm_dense_by_case = None
    bge_dense_by_case = None

    need_bm25 = any(v in variants for v in ("hybrid", "reranker", "bge_hybrid", "bge_rerank"))
    need_minilm = any(v in variants for v in ("baseline", "hybrid", "reranker"))
    need_bge = any(v in variants for v in ("bge", "bge_hybrid", "bge_rerank"))
    need_rerank = any(v in variants for v in ("reranker", "bge_rerank"))

    if need_bm25:
        print("  computing BM25 rankings ...")
        bm25_by_case = [bm25_ranks(c, c.questions, topn) for c in cases]

    if need_minilm:
        print(f"  loading dense embedder '{BASE_EMB}' ...")
        m = SentenceTransformer(BASE_EMB)
        embed_chunks(cases, m, BASE_EMB)
        print("  computing MiniLM dense rankings ...")
        minilm_dense_by_case = [dense_ranks(c, m, BASE_EMB, c.questions, False, topn) for c in cases]

    if need_bge:
        print(f"  loading dense embedder '{BGE_EMB}' ...")
        mb = SentenceTransformer(BGE_EMB)
        embed_chunks(cases, mb, BGE_EMB)
        print("  computing BGE dense rankings ...")
        bge_dense_by_case = [dense_ranks(c, mb, BGE_EMB, c.questions, True, topn) for c in cases]

    reranker = None
    if need_rerank:
        from sentence_transformers import CrossEncoder
        print(f"  loading cross-encoder reranker '{CROSS_ENCODER}' ...")
        reranker = CrossEncoder(CROSS_ENCODER)

    def hybrid_by_case(dense_by_case):
        out = []
        for c_i, case in enumerate(cases):
            per_q = []
            for q_i in range(len(case.questions)):
                fused = rrf_fuse([dense_by_case[c_i][q_i], bm25_by_case[c_i][q_i]])
                per_q.append(fused)
            out.append(per_q)
        return out

    def rerank_by_case(hybrid_lists):
        """Rerank the top `rerank_pool` fused candidates with the cross-encoder."""
        out = []
        for case, per_q in zip(cases, hybrid_lists):
            case_out = []
            for q, cand in zip(case.questions, per_q):
                pool = cand[:rerank_pool]
                if not pool:
                    case_out.append(cand)
                    continue
                pairs = [(q, case.chunks[cid]) for cid in pool]
                scores = reranker.predict(pairs, show_progress_bar=False)
                order = np.argsort(scores)[::-1]
                reranked = [pool[i] for i in order]
                # append any fused candidates beyond the reranked pool so @10
                # still has depth if pool < 10
                tail = [cid for cid in cand if cid not in set(pool)]
                case_out.append(reranked + tail)
            out.append(case_out)
        return out

    # --- assemble each variant ----------------------------------------------
    if "baseline" in variants:
        results["baseline"], total = score_variant(cases, minilm_dense_by_case, ks)[0], None

    minilm_hybrid = hybrid_by_case(minilm_dense_by_case) if need_bm25 and minilm_dense_by_case else None
    if "hybrid" in variants:
        results["hybrid"] = score_variant(cases, minilm_hybrid, ks)[0]

    if "reranker" in variants:
        reranked = rerank_by_case(minilm_hybrid)
        results["reranker"] = score_variant(cases, reranked, ks)[0]

    if "bge" in variants:
        results["bge"] = score_variant(cases, bge_dense_by_case, ks)[0]

    bge_hybrid = hybrid_by_case(bge_dense_by_case) if need_bge and bm25_by_case else None
    if "bge_hybrid" in variants:
        results["bge_hybrid"] = score_variant(cases, bge_hybrid, ks)[0]

    if "bge_rerank" in variants:
        reranked = rerank_by_case(bge_hybrid)
        results["bge_rerank"] = score_variant(cases, reranked, ks)[0]

    return results


VARIANT_LABELS = {
    "baseline": "baseline (MiniLM dense)",
    "hybrid": "+hybrid (BM25 + MiniLM, RRF)",
    "reranker": "+reranker (hybrid + cross-encoder)",
    "bge": "+BGE embeddings (dense)",
    "bge_hybrid": "+BGE hybrid (BM25 + BGE, RRF)",
    "bge_rerank": "+BGE hybrid + reranker",
}

ALL_VARIANTS = list(VARIANT_LABELS.keys())


def print_table(results, ks, total, contracts_used):
    print("\n" + "=" * 74)
    print("CUAD RETRIEVAL BENCHMARK  (real contracts, expert gold spans, no LLM)")
    print("=" * 74)
    print(f"Contracts scored:  {contracts_used}")
    print(f"Clause questions:  {total}")
    print(f"Chunking:          512 chars / 64 overlap")
    print("-" * 74)
    header = "variant".ljust(36) + "".join(f"@{k}".rjust(9) for k in ks)
    print(header)
    print("-" * 74)
    base = results.get("baseline")
    for v in ALL_VARIANTS:
        if v not in results:
            continue
        row = VARIANT_LABELS[v].ljust(36)
        row += "".join(f"{results[v][k]:.3f}".rjust(9) for k in ks)
        print(row)
    print("=" * 74)
    if base:
        print("Deltas vs baseline (hit-rate@k, absolute):")
        for v in ALL_VARIANTS:
            if v not in results or v == "baseline":
                continue
            deltas = "  ".join(f"@{k} {results[v][k]-base[k]:+.3f}" for k in ks)
            print(f"  {VARIANT_LABELS[v]:<36} {deltas}")
    print("hit-rate@k = fraction of gold clause spans surfaced in the top-k chunks.")


def save_chart(results, ks, out_path, contracts_used, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v for v in ALL_VARIANTS if v in results]
    x = np.arange(len(ks))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#9aa0a6", "#4c8bf5", "#1a73e8", "#f0a020", "#e37400", "#137333"]
    for i, v in enumerate(variants):
        vals = [results[v][k] for k in ks]
        bars = ax.bar(x + i * width, vals, width, label=VARIANT_LABELS[v],
                      color=colors[i % len(colors)])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([f"hit-rate@{k}" for k in ks])
    ax.set_ylabel("hit-rate (retriever recall of gold clause span)")
    ax.set_ylim(0, min(1.0, max(max(results[v].values()) for v in variants) + 0.12))
    ax.set_title(
        f"CUAD retrieval: baseline vs hybrid vs reranker vs stronger embeddings\n"
        f"{contracts_used} contracts, {total} gold clause questions, no LLM",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=int, default=50)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--split", choices=list(SPLIT_FILES), default="test")
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS)
    ap.add_argument("--topn", type=int, default=50,
                    help="candidate depth for hybrid/dense before rerank")
    ap.add_argument("--rerank-pool", type=int, default=30,
                    help="how many fused candidates the cross-encoder reranks")
    ap.add_argument("--max-ctx-chars", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clause-chunking", action="store_true",
                    help="also re-run variants under a clause-aware splitter")
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "retrieval_bench_results.json"))
    ap.add_argument("--chart", default=os.path.join(ROOT, "assets", "retrieval_comparison.png"))
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_cuad(args.split)
    print(f"Loaded {len(data)} real CUAD contracts from {args.split}.json")
    sample = data if args.contracts >= len(data) else random.sample(data, args.contracts)
    print(f"Sampling {len(sample)} contracts (seed {args.seed}).")

    print("\nBuilding cases with the standard 512/64 chunker ...")
    cases = build_cases(sample, BASE_SPLITTER, args.max_ctx_chars)
    total = sum(len(c.questions) for c in cases)
    print(f"  {len(cases)} contracts contributed {total} answerable clause questions.")

    ks = sorted(args.ks)
    print(f"\nRunning variants: {', '.join(args.variants)}")
    results = run(cases, ks, args.variants, args.topn, args.rerank_pool, args.seed)
    print_table(results, ks, total, len(cases))

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "CUAD (Contract Understanding Atticus Dataset)",
        "split": args.split,
        "contracts_scored": len(cases),
        "clause_questions_scored": total,
        "seed": args.seed,
        "chunking": {"chunk_size": 512, "chunk_overlap": 64, "clause_aware": False},
        "topn": args.topn,
        "rerank_pool": args.rerank_pool,
        "embedders": {"baseline": BASE_EMB, "stronger": BGE_EMB},
        "reranker": CROSS_ENCODER,
        "llm_involved": False,
        "metric": "hit-rate@k (retriever recall of the gold clause span)",
        "variant_labels": VARIANT_LABELS,
        "results": {v: {str(k): round(results[v][k], 4) for k in ks} for v in results},
    }

    # --- optional clause-aware chunking pass --------------------------------
    if args.clause_chunking:
        print("\n" + "#" * 74)
        print("CLAUSE-AWARE CHUNKING PASS (same size budget, clause-biased splits)")
        print("#" * 74)
        cases_c = build_cases(sample, CLAUSE_SPLITTER, args.max_ctx_chars)
        total_c = sum(len(c.questions) for c in cases_c)
        print(f"  {len(cases_c)} contracts, {total_c} clause questions.")
        # Run the same variants so the whole table is comparable.
        results_c = run(cases_c, ks, args.variants, args.topn, args.rerank_pool, args.seed)
        print_table(results_c, ks, total_c, len(cases_c))
        payload["clause_aware_pass"] = {
            "contracts_scored": len(cases_c),
            "clause_questions_scored": total_c,
            "results": {v: {str(k): round(results_c[v][k], 4) for k in ks} for v in results_c},
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results: {args.out}")

    if len(results) > 1:
        save_chart(results, ks, args.chart, len(cases), total)


if __name__ == "__main__":
    main()
