# DocuSense — Contract Intelligence Platform

> A production-grade, multi-agent AI system that ingests legal contracts, extracts structured risk insights, and answers natural-language questions — grounded in cited source documents. Runs 100% locally with zero API cost.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange)](https://spark.apache.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.5-purple)](https://langchain-ai.github.io/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-black)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.2-blue)](https://delta.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## The problem

Legal and procurement teams manually review thousands of contracts every year. A single missed indemnification clause or auto-renewal trap can cost a company millions. Existing tools are expensive, cloud-dependent, and opaque.

DocuSense solves this with a local-first, privacy-preserving AI pipeline that turns dense contract PDFs into structured, queryable knowledge — surfacing risks, answering questions with citations, and flagging dangerous clauses automatically.

---

## Live demo

![DocuSense Demo](assets/demo.gif)

**Risk scanner in action:**
- Upload a contract → AI scans every clause → returns HIGH/MEDIUM/LOW risk flags with rationale
- Ask "What is the termination clause?" → cited answer with source document in under 30 seconds
- Runs fully offline on your machine — no data leaves your laptop

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DocuSense Pipeline                   │
│                                                          │
│  Raw Contracts (.txt/.pdf/.docx)                         │
│        │                                                  │
│        ▼                                                  │
│  ┌─────────────┐    PySpark     ┌──────────────┐         │
│  │   Parser    │ ────────────► │  Delta Lake   │         │
│  │ (ingestion) │               │  (storage)    │         │
│  └─────────────┘               └──────────────┘         │
│        │                              │                   │
│        ▼                              ▼                   │
│  ┌─────────────┐               ┌──────────────┐          │
│  │   Chunker   │               │  Risk Agent  │          │
│  │  512 tokens │               │  (LangGraph) │          │
│  │  64 overlap │               └──────────────┘          │
│  └─────────────┘                      │                   │
│        │                              ▼                   │
│        ▼                       ┌──────────────┐          │
│  ┌─────────────┐               │  Risk Flags  │          │
│  │   FAISS     │               │  HIGH/MED/LOW│          │
│  │  Vector DB  │               └──────────────┘          │
│  └─────────────┘                                         │
│        │                                                  │
│        ▼                                                  │
│  ┌─────────────┐    FastAPI     ┌──────────────┐         │
│  │  RAG Chain  │ ────────────► │  REST API    │          │
│  │  Llama 3.2  │               │  /query      │          │
│  └─────────────┘               └──────────────┘          │
│                                       │                   │
│                                       ▼                   │
│                                ┌──────────────┐          │
│                                │  Streamlit   │          │
│                                │  Dashboard   │          │
│                                └──────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Ingestion | PySpark 3.5, Delta Lake 3.2 | Batch contract parsing, ACID storage |
| Embeddings | Ollama + nomic-embed-text | Local vector embeddings, zero cost |
| Vector store | FAISS (local) | Semantic similarity search |
| LLM | Ollama + Llama 3.2 | Answer generation, risk classification |
| Orchestration | LangChain 0.2, LangGraph 0.1.5 | RAG chain, multi-agent pipeline |
| API | FastAPI 0.111, Uvicorn | REST endpoint with Pydantic validation |
| UI | Streamlit 1.35 | Interactive contract dashboard |
| Workflow | Apache Airflow (optional) | Scheduled ingestion DAGs |
| Experiment tracking | MLflow | Latency, chunk counts, eval scores |
| Containerization | Docker, docker-compose | Reproducible deployment |

---

## Key design decisions

**Why local LLMs (Ollama) over OpenAI?**
Legal documents contain sensitive information. Running Llama 3.2 locally means zero data leaves the machine — a non-negotiable requirement in legal and compliance contexts. It also eliminates API costs and rate limits entirely.

**Why FAISS over Pinecone?**
For a single-tenant deployment processing hundreds of contracts, FAISS delivers sub-10ms retrieval with no network overhead, no monthly cost, and no vendor lock-in. Pinecone becomes relevant only at multi-tenant or millions-of-vectors scale.

**Why chunk size 512 with 64-token overlap?**
After benchmarking 256/512/1024 token chunks on contract retrieval, 512 hit the best balance: large enough to contain a full clause with context, small enough to stay within the LLM's effective attention range. The 64-token overlap prevents clause splitting at boundaries.

**Why Delta Lake over raw Parquet?**
Contracts are updated and re-ingested frequently. Delta Lake gives ACID transactions, schema enforcement, and time-travel — critical when you need to audit which version of a contract was analyzed and when.

---

## Project structure

```
docusense/
├── data/
│   ├── raw/contracts/          # Source contracts (.txt, .pdf, .docx)
│   ├── delta/documents/        # Delta Lake tables
│   └── faiss_index/            # Persisted vector store
├── src/
│   ├── ingestion/
│   │   ├── parser.py           # Document parser (text extraction + metadata)
│   │   └── pipeline.py         # PySpark → Delta Lake pipeline
│   ├── rag/
│   │   ├── chunker.py          # Semantic chunking + FAISS embedding
│   │   └── chain.py            # RetrievalQA chain with citation enforcement
│   ├── agents/
│   │   ├── risk_agent.py       # Clause risk classifier (8 risk types)
│   │   └── graph.py            # LangGraph multi-agent orchestration
│   ├── api/
│   │   └── main.py             # FastAPI /query and /health endpoints
│   └── ui/
│       └── app.py              # Streamlit dashboard
├── notebooks/
│   └── ragas_eval.py           # RAG evaluation with RAGAS metrics
├── tests/
│   └── test_contract.txt       # Sample contract for testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### Prerequisites

- Mac (Apple Silicon or Intel) or Linux
- Python 3.11
- Java 17 (for PySpark)
- Docker Desktop
- [Ollama](https://ollama.ai) installed

### 1. Clone and set up environment

```bash
git clone https://github.com/koutilyaY/docusense.git
cd docusense
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull local models (free, one-time)

```bash
ollama pull nomic-embed-text   # embedding model (274MB)
ollama pull llama3.2            # LLM (2GB)
ollama serve                    # keep running in a terminal tab
```

### 3. Ingest contracts and build vector store

```bash
# Generate sample contracts
cd data/raw/contracts && python generate_contracts.py && cd ../../..

# Run PySpark ingestion → Delta Lake
python src/ingestion/pipeline.py

# Chunk and embed → FAISS index
python src/rag/chunker.py
```

### 4. Start the API

```bash
uvicorn src.api.main:app --port 8000
```

### 5. Start the UI

```bash
streamlit run src/ui/app.py
```

Open `http://localhost:8501` — upload `tests/test_contract.txt` and start asking questions.

---

## API reference

### `POST /query`

Ask a natural-language question across all ingested contracts.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause?"}'
```

**Response:**
```json
{
  "answer": "Either party may terminate this Agreement upon thirty (30) days written notice. Source document: 2. TERM AND TERMINATION",
  "sources": [
    "Software License Agreement 1",
    "Software License Agreement 7",
    "Software License Agreement 13"
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Risk taxonomy

The risk agent classifies clauses across 8 risk types:

| Risk type | Severity range | Example trigger |
|---|---|---|
| `indemnification` | Medium–High | "indemnify, defend, and hold harmless" |
| `auto_renewal` | Medium | "automatically renew unless 60 days notice" |
| `penalty_clause` | High | "liquidated damages of $500,000 per breach" |
| `ip_ownership` | Medium–High | "all work product owned exclusively by client" |
| `arbitration` | Medium | "binding arbitration, waiving right to jury trial" |
| `limitation_of_liability` | Medium | "liability capped at fees paid in prior 3 months" |
| `termination_for_convenience` | Low–Medium | "terminate for any reason upon 14 days notice" |
| `data_privacy` | Medium–High | "comply with GDPR and CCPA" |

---

## Evaluation results

Evaluated on 20 ground-truth Q&A pairs using [RAGAS](https://github.com/explodinggradients/ragas):

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | 0.87 | > 0.85 ✅ |
| Answer relevancy | 0.83 | > 0.80 ✅ |
| Context precision | 0.79 | > 0.75 ✅ |

**Retrieval performance:**
- Average retrieval latency: ~180ms (FAISS, local)
- Answer generation latency: ~18s (Llama 3.2, Apple M-series)
- Chunk count: 450 chunks from 150 contracts
- Index size: ~12MB on disk

---

## Docker deployment

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

---

## Roadmap

- [ ] PDF and DOCX ingestion via Unstructured.io
- [ ] Contract version diffing (clause-level change detection)
- [ ] RAGAS automated eval CI/CD pipeline
- [ ] Pinecone integration for multi-tenant vector storage
- [ ] Airflow DAG for scheduled ingestion
- [ ] Contract summary report generation (PDF export)
- [ ] Fine-tuned risk classifier on CUAD dataset labels

---

## What I learned building this

The hardest part wasn't the LLM — it was the data layer. Getting PySpark + Delta Lake to play nicely with local Python paths, managing chunking strategy trade-offs, and enforcing citation grounding in the LLM prompt required more iteration than expected.

The key insight: **RAG quality is 80% retrieval quality, 20% LLM quality.** The chunking strategy, overlap size, and retrieval k-value matter far more than which LLM you use. I benchmarked chunk sizes from 256 to 1024 tokens and documented the precision@5 scores in `notebooks/ragas_eval.py`.

---

## About

Built by [Koutilya Yenumula](https://linkedin.com/in/koutilya716-yenumula-b675911b1) — Data Engineer with 3 years of experience at Visa and Cognizant, currently completing an M.S. in Computer Science at University of South Florida (May 2026). AWS Certified Data Engineer – Associate.

- GitHub: [github.com/koutilyaY](https://github.com/koutilyaY)
- LinkedIn: [linkedin.com/in/koutilya716-yenumula-b675911b1](https://linkedin.com/in/koutilya716-yenumula-b675911b1)

---

## License

MIT — see [LICENSE](LICENSE) for details.
