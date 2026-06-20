# DocuSense — Local-First Contract Intelligence

> Privacy-preserving contract Q&A over legal documents. Ask natural-language questions, get **cited** answers grounded in the source contracts, plus automated **risk flags** on uploaded clauses. Runs **100% on your machine** — no API keys, no data leaving the laptop, zero per-query cost.

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-00D084?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2%20%2B%20nomic--embed--text-000000?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-cpu-005BBB?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009485?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## TL;DR — Run it locally

DocuSense runs as three local processes: **Ollama** (model server), a **FastAPI** RAG backend, and a **Streamlit** UI. You need [Ollama](https://ollama.ai) installed and a prebuilt FAISS index at `data/faiss_index/` (already committed in this repo).

```bash
# 0. One-time: pull the two local models (~2.3 GB total)
ollama pull llama3.2          # LLM for answers + risk classification
ollama pull nomic-embed-text  # embedding model for retrieval

# 1. Python env + deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then open **three terminal tabs** from the repo root (all with `venv` activated):

```bash
# Tab 1 — model server
ollama serve

# Tab 2 — RAG API  (http://localhost:8000, docs at /docs)
uvicorn src.api.main:app --port 8000

# Tab 3 — UI  (http://localhost:8501)
streamlit run src/ui/app.py
```

Open <http://localhost:8501>. Use the **Ask the contracts** panel to query the indexed corpus, and the **Risk Scanner** panel to upload a `.txt` contract (try `tests/test_contract.txt`) and get clause-level risk flags.

Config lives in `.env` (currently a single flag, `USE_LOCAL=true`).

> **Rebuilding the index:** the FAISS index ships prebuilt. To regenerate it from your own contracts, run the ingestion + chunking scripts under `src/ingestion/` and `src/rag/` (see [Repo layout](#repo-layout)).

---

## The problem

Legal, procurement, and compliance teams review large volumes of contracts, where a single missed indemnification clause, liability cap, or auto-renewal trap carries real cost. Commercial contract-AI tooling is typically SaaS, which means **sending privileged documents — NDAs, IP terms, financials — to a third-party cloud**. For sensitive material that is often a non-starter.

DocuSense takes the opposite stance: **everything runs locally**. The LLM (`llama3.2`) and the embedding model (`nomic-embed-text`) are served by Ollama on your machine, retrieval is an on-disk FAISS index, and no document text ever crosses the network. That removes the privacy objection, the per-query API bill, and rate limits in one move.

What you get:

- **Cited Q&A.** Ask a question, retrieve the top-k relevant chunks, and the LLM answers *only* from that context — instructed to cite the source filename and to say "Not found in provided documents" rather than hallucinate.
- **Risk flagging.** Upload a contract; each substantial clause is classified into one of 8 risk categories with a `low/medium/high` severity and a one-line rationale.

---

## Architecture

```
                          ┌──────────────────────────────────────┐
  Raw contracts           │  INGESTION (offline, batch)          │
  (.txt / EDGAR / CUAD) ─▶│  parser.py → pipeline.py (PySpark →   │
                          │  Delta Lake)                         │
                          └──────────────────┬───────────────────┘
                                             │ clean documents
                                             ▼
                          ┌──────────────────────────────────────┐
                          │  INDEXING (offline)                  │
                          │  chunker.py: 512-tok chunks /         │
                          │  64 overlap → nomic-embed-text →      │
                          │  FAISS index  (data/faiss_index/)    │
                          └──────────────────┬───────────────────┘
                                             │
            ┌────────────────────────────────┼────────────────────────────────┐
            ▼ (query path)                    │                  ▼ (risk path)
  ┌───────────────────────────┐               │     ┌──────────────────────────────┐
  │ FastAPI  POST /query      │               │     │ Streamlit upload → risk_agent │
  │ chain.py: FAISS retrieve  │               │     │ scan_document(): split clauses│
  │ (k=5) → llama3.2          │               │     │ → classify_risk() per clause  │
  │ RetrievalQA (cite-only    │               │     │ (llama3.2, JSON out)          │
  │ prompt)                   │               │     └──────────────┬───────────────┘
  └────────────┬──────────────┘               │                    │
               │ answer + source filenames    │                    │ risk flags
               ▼                               │                    ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │  Streamlit UI — cited answers  +  HIGH/MEDIUM/LOW risk flags      │
        └──────────────────────────────────────────────────────────────────┘
```

**Two independent inference paths**, both backed by the same local `llama3.2`:

1. **RAG Q&A** — `src/api/main.py` exposes `POST /query`; `src/rag/chain.py` builds a LangChain `RetrievalQA` chain over the FAISS retriever (`k=5`) with a strict citation prompt and returns `{answer, sources}`.
2. **Risk classification** — `src/agents/risk_agent.py` splits an uploaded document into clauses and classifies each via a JSON-constrained LLM prompt. The Streamlit UI calls it directly.

> **On "agents":** DocuSense ships **one agent** — the risk classifier (`risk_agent.py`). It is a single-LLM, prompt-driven classifier called per clause; it is **not** a multi-agent graph and there is no LangGraph orchestration in the codebase today. (`langgraph` is listed as a dependency but is not currently wired into either path.) The README describes only what is actually in the flow.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| LLM | **Ollama + llama3.2** | Local, private, free per query; powers both Q&A and risk classification |
| Embeddings | **Ollama + nomic-embed-text** | Local embedding model (~274 MB); no embedding API calls |
| Vector store | **FAISS (faiss-cpu)** | On-disk, fast in-process retrieval, no external service |
| RAG orchestration | **LangChain 0.2** (`RetrievalQA`) | Retriever + cite-only prompt + source-document return |
| API | **FastAPI 0.111 + Uvicorn** | `POST /query`, `GET /health`, auto OpenAPI docs at `/docs` |
| UI | **Streamlit 1.35** | Two-panel app: contract Q&A and risk scanner |
| Ingestion | **PySpark 3.5 + Delta Lake 3.2** | Batch parse/clean of the contract corpus into Delta tables |
| Evaluation | **RAGAS 0.1.9** | Faithfulness / answer-relevancy / context-precision scoring |

---

## Privacy-first by design

- **No network egress of document text.** Inference (LLM + embeddings) and retrieval are entirely local via Ollama and FAISS.
- **No API keys, no cloud account, no per-query cost.** After the one-time model pull, it runs offline.
- **Deterministic answers.** The LLM runs at `temperature=0` for reproducible, auditable output.
- **Grounded, refusal-capable prompt.** The Q&A prompt forbids answering outside the retrieved context and returns "Not found in provided documents" when the corpus doesn't support an answer.

---

## Evaluation

RAG quality is measured with [RAGAS](https://github.com/explodinggradients/ragas). The script is `notebooks/ragas_eval.py`; the most recent run is recorded in `notebooks/ragas_results.json`.

> **Scope of these numbers (read this).** The figures below come from **one** RAGAS run over **20 hand-written Q&A pairs** against a **200-contract corpus (150 synthetic + 50 SEC EDGAR filings)** using `llama3.2` + `nomic-embed-text`, 512-token chunks / 64 overlap. They reflect a small, partly synthetic sample and a single local run — they are **not independently reproducible without re-running the eval on your own corpus and models**, and they are **not** a generalized accuracy guarantee. Treat them as a directional, self-reported benchmark.

| RAGAS metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.68** | Are answer claims supported by the retrieved context? |
| Answer relevancy | **0.73** | Does the answer address the question? |
| Context precision | **0.90** | Is the retrieved context on-topic / well-ranked? |

*(Source: `notebooks/ragas_results.json`, run dated 2026-04-16, 20 questions.)*

> Earlier versions of this README cited higher RAG scores and a "95.7% extraction accuracy" headline. Those figures were **not reproducible from any committed artifact** and have been removed in favor of the actual logged RAGAS output above. Re-run `notebooks/ragas_eval.py` to regenerate these numbers for your setup.

---

## API reference

```bash
# Ask a question (answers are grounded in the FAISS-indexed corpus)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the termination terms?"}'
```

```json
{
  "answer": "Either party may terminate upon 30 days written notice ... (Source: <filename>)",
  "sources": ["Software License Agreement 1", "Software License Agreement 7"]
}
```

```bash
curl http://localhost:8000/health     # {"status": "ok"}
```

Interactive OpenAPI docs: <http://localhost:8000/docs>.

---

## Risk taxonomy

`risk_agent.py` classifies each clause into one of these 8 categories (or `other`), with a `low/medium/high` severity and a one-line rationale. The Streamlit scanner surfaces only `medium`/`high` flags.

`indemnification` · `auto_renewal` · `penalty_clause` · `ip_ownership` · `arbitration` · `limitation_of_liability` · `termination_for_convenience` · `data_privacy`

---

## Repo layout

```
docusense/
├── data/
│   ├── raw/contracts/          # source contracts (synthetic + EDGAR/CUAD)
│   ├── delta/                  # Delta Lake tables from ingestion
│   └── faiss_index/            # prebuilt vector index (index.faiss / index.pkl)
├── src/
│   ├── ingestion/
│   │   ├── parser.py           # text extraction + metadata
│   │   ├── pipeline.py         # PySpark → Delta Lake ETL
│   │   └── load_all.py         # corpus loader
│   ├── rag/
│   │   ├── chunker.py          # chunk + embed (nomic) → FAISS; load_vectorstore()
│   │   ├── chain.py            # RetrievalQA chain (k=5, cite-only prompt)
│   │   └── rebuild_index.py    # rebuild the FAISS index
│   ├── agents/
│   │   └── risk_agent.py       # single risk-classification agent (8 categories)
│   ├── api/
│   │   └── main.py             # FastAPI: POST /query, GET /health
│   └── ui/
│       └── app.py              # Streamlit two-panel app (Q&A + risk scanner)
├── notebooks/
│   ├── ragas_eval.py           # RAGAS evaluation script
│   └── ragas_results.json      # logged eval output
├── tests/test_contract.txt     # sample contract for the risk scanner
├── requirements.txt
└── README.md
```

---

## Design notes

- **Local LLM over a hosted API.** Privileged contract text never leaves the machine; the trade-off is local inference latency on `llama3.2` instead of a hosted frontier model.
- **FAISS over a managed vector DB.** For a single-tenant, hundreds-of-contracts corpus, an in-process index is simpler and has no network hop or monthly cost.
- **512-token chunks, 64 overlap.** Large enough to keep a clause intact with surrounding context; overlap avoids splitting clauses across chunk boundaries.
- **Delta Lake for ingestion.** ACID writes and schema enforcement for a corpus that gets re-ingested as it changes.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Built by **Koutilya Yenumula** — [GitHub](https://github.com/koutilyaY) · [LinkedIn](https://www.linkedin.com/in/koutilya-yenumula)
