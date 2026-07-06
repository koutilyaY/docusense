# DocuSense Architecture & Design

This document details the system architecture, data flow, component interactions, and key design decisions.

---

## System Architecture

### High-Level Flow
[Raw Contracts]
↓
[Parser] ─→ Extract text, metadata, source path
↓
[Delta Lake] ─→ ACID storage, versioning, audit trail
↓
[Chunker] ─→ 512-token chunks, 64-token overlap, semantic boundaries
↓
[Embeddings] ─→ nomic-embed-text (local, 274MB)
↓
[FAISS Index] ─→ Vector store, sub-10ms retrieval
↓
[LangGraph Agent] ─→ Risk classification (8 categories)
↓
[FastAPI] ─→ REST endpoints (/query, /health, /docs)
↓
[Streamlit UI] ─→ Interactive dashboard, risk flagging
---

## Design Rationale

### Why Delta Lake?
- ACID transactions
- Schema enforcement
- Time-travel auditing
- Works with PySpark directly

### Why FAISS?
- Sub-10ms local retrieval
- No vendor lock-in
- Fully deterministic
- No monthly cost

### Why Local LLMs?
- **Privacy**: Zero data leaves the machine
- **Cost**: $0 after one-time download
- **Latency**: No network roundtrips
- **Compliance**: No third-party agreements

---

## Performance Characteristics

| Component | Latency |
|-----------|---------|
| FAISS retrieval (k=3) | 8–12ms |
| Prompt construction | 2–5ms |
| Llama 3.2 generation (150 tokens) | 15–20s |
| **Total end-to-end** | **15–20s** |

---

## Deployment

DocuSense runs as three local processes — `ollama serve`, the FastAPI backend,
and the Streamlit UI — started by hand from the repo root. There is no container
image; the project is deliberately local-first and single-machine. See
`HOW_TO_RUN.md` for the exact commands.

---

## Future Improvements

- [ ] Improve retrieval recall on CUAD: legal-domain embedding model, clause-aware chunking, hybrid BM25 + dense retrieval, reranking
- [ ] End-to-end answer eval against CUAD gold spans once retrieval is stronger
- [ ] PDF/DOCX ingestion
- [ ] Fine-tuned risk classifier
- [ ] Multi-document comparison
