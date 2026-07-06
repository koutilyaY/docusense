"""
Real retrieval evaluation on the CUAD dataset.

What this measures
------------------
DocuSense's RAG pipeline answers questions by first retrieving the top-k most
relevant chunks from a contract corpus, then feeding them to a local LLM. The
retrieval step is the part that decides whether the model even *sees* the right
clause. This script measures that step directly, against real gold labels, with
no LLM in the loop.

Method:
  1. Take a sample of real CUAD contracts (test split).
  2. Chunk each contract with the same splitter DocuSense uses (512 chars,
     64 overlap) and embed the chunks with a sentence-transformers model into
     a FAISS index -- one index per contract.
  3. For every answerable CUAD clause question (one that carries an expert gold
     answer span), embed the question, retrieve the top-k chunks, and check
     whether any retrieved chunk actually contains the gold span text.
  4. Report hit-rate @ k (a.k.a. recall@k for single-span retrieval): the
     fraction of clause questions whose gold span was surfaced in the top-k.

Why this is honest and verifiable:
  - The contracts are real (CUAD / The Atticus Project).
  - The labels are real (expert-annotated clause spans).
  - No LLM is involved, so the number is fully reproducible on any machine with
     the same embedding model -- no Ollama, no API keys, no generation step.

This replaces the earlier self-reported RAGAS figures (which required a running
LLM and used a partly synthetic corpus) with a single, checkable retrieval
metric. It is deliberately a lower-level, more modest number: it tells you how
often the retriever puts the right clause in front of the model, nothing more.

Usage:
    python src/eval/retrieval_eval.py                 # default: 30 contracts, k in {1,3,5,10}
    python src/eval/retrieval_eval.py --contracts 50 --ks 1 3 5 10
    python src/eval/retrieval_eval.py --split CUADv1  # use full 510-contract file
"""
import argparse
import json
import os
import random
import sys
import datetime

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss is required: pip install faiss-cpu")

from sentence_transformers import SentenceTransformer

# Reuse the exact chunking DocuSense's pipeline uses, so the eval reflects the
# real retriever rather than an idealised one. Prefer the lightweight standalone
# splitters package (no full langchain needed) and fall back to langchain.
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

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "],
)


def load_cuad(split: str):
    path = SPLIT_FILES[split]
    if not os.path.exists(path):
        sys.exit(
            f"CUAD file not found: {path}\n"
            "Run: python data/cuad/download_cuad.py"
        )
    with open(path) as f:
        return json.load(f)["data"]


def norm(s: str) -> str:
    """Collapse whitespace so span-in-chunk matching is robust to re-wrapping."""
    return " ".join(s.split()).lower()


def build_contract_index(context: str, model: SentenceTransformer):
    """Chunk one contract and build a FAISS index over its chunks."""
    chunks = SPLITTER.split_text(context)
    if not chunks:
        return None, None
    emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # cosine sim via normalized IP
    index.add(emb)
    return index, chunks


def evaluate(contracts, model, ks, max_ctx_chars):
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0
    contracts_used = 0

    norm_chunks_cache = None
    for c in contracts:
        para = c["paragraphs"][0]
        context = para["context"]
        if max_ctx_chars and len(context) > max_ctx_chars:
            # Some CUAD contracts are very long; cap for a faster, still-real
            # sample. Gold spans past the cap are dropped from scoring too so
            # the metric stays honest (we never count an unreachable span).
            context = context[:max_ctx_chars]

        index, chunks = build_contract_index(context, model)
        if index is None:
            continue
        norm_chunks = [norm(ch) for ch in chunks]

        # Collect answerable questions whose gold span falls inside the context.
        questions, gold_norms = [], []
        for qa in para["qas"]:
            if not qa["answers"]:
                continue
            ans = qa["answers"][0]["text"].strip()
            if len(ans) < 3:
                continue
            if norm(ans) not in norm(context):
                continue  # span sits past the ctx cap; skip to stay honest
            questions.append(qa["question"])
            gold_norms.append(norm(ans))

        if not questions:
            continue
        contracts_used += 1

        q_emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")
        _, ids = index.search(q_emb, min(max_k, len(chunks)))

        for row, gold in enumerate(gold_norms):
            total += 1
            retrieved = ids[row]
            for k in ks:
                topk = retrieved[:k]
                if any(gold in norm_chunks[i] for i in topk if i >= 0):
                    hits[k] += 1

    return hits, total, contracts_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=int, default=30,
                    help="number of contracts to sample (default 30)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10],
                    help="k values for hit-rate@k")
    ap.add_argument("--split", choices=list(SPLIT_FILES), default="test")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="sentence-transformers embedding model")
    ap.add_argument("--max-ctx-chars", type=int, default=40000,
                    help="cap per-contract context length for speed (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "retrieval_results.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_cuad(args.split)
    print(f"Loaded {len(data)} real CUAD contracts from {args.split}.json")

    sample = data if args.contracts >= len(data) else random.sample(data, args.contracts)
    print(f"Sampling {len(sample)} contracts. Loading embedding model '{args.model}' ...")
    model = SentenceTransformer(args.model)

    print("Building per-contract FAISS indexes and scoring gold clause spans ...")
    hits, total, contracts_used = evaluate(sample, model, sorted(args.ks), args.max_ctx_chars)

    if total == 0:
        sys.exit("No scorable clause questions found -- check the CUAD download.")

    print("\n" + "=" * 56)
    print("REAL RETRIEVAL EVALUATION ON CUAD (no LLM involved)")
    print("=" * 56)
    print(f"Dataset:            CUAD ({args.split}.json) -- real contracts, expert gold spans")
    print(f"Contracts scored:   {contracts_used}")
    print(f"Clause questions:   {total} (answerable, gold span inside scored context)")
    print(f"Embedding model:    {args.model}")
    print(f"Chunking:           512 chars / 64 overlap (same as the RAG pipeline)")
    print("-" * 56)
    rates = {}
    for k in sorted(args.ks):
        rate = hits[k] / total
        rates[str(k)] = round(rate, 4)
        print(f"hit-rate@{k:<3}       {rate:.4f}   ({hits[k]}/{total})")
    print("=" * 56)
    print("hit-rate@k = fraction of gold clause spans surfaced in the top-k")
    print("retrieved chunks. This is the retriever's recall; it does not")
    print("measure the LLM's final answer.")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "CUAD (Contract Understanding Atticus Dataset)",
        "dataset_note": "real commercial contracts, expert-annotated clause spans",
        "split": args.split,
        "contracts_scored": contracts_used,
        "clause_questions_scored": total,
        "embedding_model": args.model,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "metric": "hit-rate@k (retriever recall of the gold clause span)",
        "llm_involved": False,
        "hit_rate_at_k": rates,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
