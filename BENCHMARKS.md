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

Result from `src/eval/retrieval_eval.py --contracts 50` (sample: 50 contracts,
458 answerable clause questions, seed 42):

| Metric | Score |
|---|---|
| hit-rate@1 | 0.17 |
| hit-rate@3 | 0.31 |
| hit-rate@5 | 0.37 |
| hit-rate@10 | 0.46 |

Logged in `notebooks/retrieval_results.json`.

**Read this honestly.** These numbers are modest. With a general-purpose MiniLM
embedding, exact-span matching, and no legal-domain fine-tuning, the retriever
puts the right clause in the top-5 about 37% of the time and in the top-10 about
46% of the time. That is a real, reproducible floor, not a marketing figure. It
is the honest replacement for the "95.7%" headline that appeared in earlier
versions of this repo, which was not backed by any committed artifact.

The number is reproducible: same embedding model + same seed gives the same
result on any machine, no Ollama or API key required.

Ways this could be improved (not yet done): a legal-domain embedding model,
sentence-window or clause-aware chunking, hybrid BM25 + dense retrieval, or
reranking. Those are listed as future work rather than claimed.

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
