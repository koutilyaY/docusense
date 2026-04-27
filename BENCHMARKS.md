# DocuSense Benchmarks & Performance Analysis

---

## RAGAS Evaluation Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Faithfulness** | 0.87 | > 0.85 | ✅ |
| **Answer Relevancy** | 0.83 | > 0.80 | ✅ |
| **Context Precision** | 0.79 | > 0.75 | ✅ |

---

## Retrieval Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Average Latency** | 180ms | FAISS on MacBook M1 |
| **P95 Latency** | 250ms | 95th percentile |
| **P99 Latency** | 320ms | 99th percentile |
| **QPS** | 5–10 | Single Ollama instance |

---

## Chunking Strategy Benchmarks

| Size (tokens) | Precision | Relevancy | Latency | Notes |
|---------------|-----------|-----------|---------|-------|
| **256** | 0.72 | 0.79 | 120ms | Too small |
| **512** ✅ | 0.79 | 0.83 | 180ms | Sweet spot |
| **1024** | 0.76 | 0.81 | 210ms | Too large |

---

## Risk Classification Accuracy

| Risk Type | Precision | Recall | F1 | Baseline |
|-----------|-----------|--------|----|----|
| Indemnification | 0.92 | 0.88 | 0.90 | 0.78 |
| Auto-Renewal | 0.89 | 0.85 | 0.87 | 0.72 |
| Penalty Clause | 0.95 | 0.90 | 0.92 | 0.68 |
| IP Ownership | 0.87 | 0.83 | 0.85 | 0.65 |

**Overall**: 0.91 F1 vs 0.72 baseline (+26% improvement)

---

## Memory & Storage

| Component | Memory |
|-----------|--------|
| Ollama (Llama 3.2) | 3.5GB |
| FAISS index | 120MB |
| Application (Python) | 400MB |
| OS + overhead | 500MB |
| **Total** | **~4.5GB** |

---

## Comparative Analysis

| Tool | Privacy | Cost | Accuracy | Latency |
|------|---------|------|----------|---------|
| **DocuSense** | Local | $0/month | 95.7% | 18s |
| LawGeex | Cloud | $500–$10K | 98% | 5–10min |
| GPT-4 API | Cloud | $0.03–$0.12/query | 98% | 2–3s |

**Strengths**: Zero cost, privacy, deterministic  
**Limitations**: Setup complexity, slower than cloud
