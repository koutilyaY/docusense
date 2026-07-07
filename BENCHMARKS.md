# DocuSense — Evaluation and Numbers

This file records the actual measured numbers for DocuSense and is careful about
which are real and which are directional. Where a number is self-reported or
depends on a local LLM run, it says so.

---

## 1. Retrieval evaluation on CUAD (real data, real labels, no LLM)

This is the primary, verifiable benchmark. It measures how often DocuSense's
retriever surfaces the correct clause, evaluated against real expert
annotations, with no language model involved.

- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — 510 real
  commercial contracts annotated by legal experts across 41 clause categories,
  released by The Atticus Project. We use the `test.json` split (102 contracts).
- **What is scored:** every answerable CUAD clause question (one that carries an
  expert gold answer span). A question counts as a "hit" at k if any of the
  top-k retrieved chunks actually contains the gold span text.
- **Retriever under test:** the same 512-char / 64-overlap chunking the RAG
  pipeline uses, embedded with `sentence-transformers/all-MiniLM-L6-v2` into a
  per-contract FAISS index.
- **Metric:** hit-rate@k (the retriever's recall of the gold clause span).

### 1a. The baseline (the honest floor)

Result from `src/eval/retrieval_eval.py --contracts 50` (sample: 50 contracts,
458 answerable clause questions, seed 42), general-purpose
`all-MiniLM-L6-v2` dense retrieval:

| Metric | Score |
|---|---|
| hit-rate@1 | 0.17 |
| hit-rate@3 | 0.31 |
| hit-rate@5 | 0.37 |
| hit-rate@10 | 0.46 |

Logged in `notebooks/retrieval_results.json`. With a general-purpose MiniLM
embedding, exact-span matching, and no legal-domain tuning, the retriever puts
the right clause in the top-5 about 37% of the time. That is a real,
reproducible floor, not a marketing figure — same model + seed gives the same
result on any machine, no Ollama or API key required.

### 1b. Retrieval engineering: what moved the number, measured

`src/eval/retrieval_bench.py` takes that exact evaluation (same 50 contracts,
same chunking, same 458 gold spans) and layers real retrieval techniques on top,
scoring each on the *identical* gold set so the comparison is apples-to-apples.
Still no LLM: dense retrieval, BM25, and a cross-encoder reranker only.

Variants:

- **baseline** — MiniLM dense (the floor above).
- **+hybrid** — BM25 (sparse, `rank_bm25`) fused with MiniLM dense via
  Reciprocal Rank Fusion (RRF, k=60).
- **+reranker** — hybrid top-30, reranked by
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **+BGE** — swap MiniLM for `BAAI/bge-small-en-v1.5` (a stronger general
  sentence-transformer) as the dense embedder.
- **+BGE hybrid** — BM25 fused with BGE dense via RRF.
- **+BGE hybrid + reranker** — the BGE hybrid candidates, cross-encoder reranked.

Result (`python src/eval/retrieval_bench.py --contracts 50`, seed 42, 458 gold
clause questions), best in each column in **bold**:

| Variant | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| baseline (MiniLM dense) | 0.170 | 0.306 | 0.369 | 0.459 |
| +hybrid (BM25 + MiniLM, RRF) | 0.214 | 0.314 | 0.376 | 0.467 |
| +reranker (hybrid + cross-encoder) | 0.197 | 0.308 | 0.380 | 0.439 |
| +BGE embeddings (dense) | 0.227 | **0.330** | **0.389** | 0.463 |
| +BGE hybrid (BM25 + BGE, RRF) | **0.227** | 0.321 | 0.376 | **0.483** |
| +BGE hybrid + reranker | 0.197 | 0.310 | 0.376 | 0.443 |

Deltas vs baseline (absolute hit-rate):

| Variant | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| +hybrid | +0.044 | +0.009 | +0.007 | +0.009 |
| +reranker | +0.026 | +0.002 | +0.011 | **−0.020** |
| +BGE | **+0.057** | +0.024 | +0.020 | +0.004 |
| +BGE hybrid | +0.057 | +0.015 | +0.007 | +0.024 |
| +BGE hybrid + reranker | +0.026 | +0.004 | +0.007 | −0.015 |

Logged in `notebooks/retrieval_bench_results.json`; chart in
`assets/retrieval_comparison.png`.

**What actually moved the needle, blunt version:**

- **Swapping the embedding model (MiniLM → BGE-small) was the single biggest
  lever, and it was cheap.** hit-rate@1 went from 0.170 to 0.227 — a **+0.057
  absolute, ~34% relative** jump — and @5 from 0.369 to 0.389. Same chunks, same
  eval, just a better general-purpose embedder. This is the headline.
- **Hybrid (BM25 + dense, RRF) reliably helps precision at the top**, adding
  +0.044 @1 on MiniLM. Sparse lexical matching catches clause questions whose
  wording overlaps the contract text, which dense embeddings alone miss.
- **The best single stack is BGE + hybrid**: it ties for the best @1 (0.227) and
  gives the best deep recall @10 (0.483, +0.024 over baseline).
- **The cross-encoder reranker did *not* help here — an honest negative
  result.** It nudges @5 by a hair but actually *lowers* @10 (−0.020 on MiniLM,
  −0.015 on BGE). Two reasons: (1) each contract is scored against its own
  chunks, so the fused candidate pool is small — there is little for a reranker
  to re-sort; and (2) `ms-marco-MiniLM` is a general web-passage reranker, not
  legal-tuned, so it does not model contract-clause relevance better than the
  fused ranking already does. Reranking is often the big win in RAG; on this
  per-contract, single-span setup it is not. We keep it in the table because
  showing the technique that *didn't* help is the honest thing to do.

Bottom line: the retriever improved from **hit-rate@1 0.170 → 0.227 (+34%
relative)** and **@10 0.459 → 0.483**, driven by a stronger embedding model plus
hybrid sparse+dense fusion — not by the reranker. Everything is reproducible
with `pip install -r requirements-bench.txt` and one script, no LLM.

### 1c. Clause-aware chunking (measured; did not help)

`src/eval/retrieval_bench.py --clause-chunking` re-runs every variant with a
clause-biased splitter (same 512/64 size budget, but breaking preferentially on
`;`, sentence ends, and newlines before raw characters). The idea was that
cutting on clause boundaries would keep gold spans intact inside a single chunk.
It did not help — it came out flat to slightly worse (e.g. baseline @5 0.349 vs
0.369 with the standard splitter). CUAD gold spans are often long and cross the
512-char budget regardless of *where* the cut lands, so nudging the boundaries
does not change how often a span sits wholly inside a retrieved chunk. Kept in
the results JSON (`clause_aware_pass`) as a measured negative, not dropped.

---

## 2. RAGAS scores (directional, LLM-based, partly synthetic — kept for context)

The earlier RAGAS run in `notebooks/ragas_results.json` measured the *generation*
side of the pipeline. It required a running `llama3.2` and used a corpus that was
mostly synthetic at the time, so it is not independently reproducible and is
kept only as a directional signal.

| RAGAS metric | Score | Caveat |
|---|---|---|
| Faithfulness | 0.68 | single local run |
| Answer relevancy | 0.73 | 20 hand-written questions |
| Context precision | 0.90 | partly synthetic corpus |

These are self-reported and should not be read as an accuracy guarantee.

---

## 3. What is real vs synthetic

| Item | Status |
|---|---|
| CUAD contracts used in the retrieval eval | **Real** (Atticus Project, expert-labeled) |
| Retrieval hit-rate@k numbers | **Real**, reproducible, no LLM |
| CUAD gold clause spans | **Real** expert annotations |
| Old `cuad_contracts.jsonl` template corpus | **Synthetic** (150 generated templates) — retained only for the demo RAG index; can be replaced with real CUAD via `src/eval/cuad_to_jsonl.py` |
| SEC EDGAR filings under `data/raw/contracts/edgar/` | Real filings, used only in the demo corpus |
| RAGAS faithfulness / relevancy / precision | Self-reported, LLM-dependent, directional |

---

## 4. Latency (informal, single machine)

Rough local timings on an Apple Silicon laptop, not a controlled benchmark:

| Step | Approx time |
|---|---|
| FAISS retrieval (k=5) | ~10 ms |
| `llama3.2` answer generation | 15–20 s |

The generation step is the bottleneck and is the price of running a local LLM
instead of a hosted API.
