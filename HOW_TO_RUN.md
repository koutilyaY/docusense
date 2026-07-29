# DocuSense — How to Run From Scratch

Everything you need to restart the project after closing it.

---

## Prerequisites (one-time setup)

Make sure these are installed before anything else:

- Python 3.11
- Java 17 (only if you rebuild the demo index via PySpark) — `java -version` to check
- [Ollama](https://ollama.ai) installed, for the optional answer step — `ollama --version` to check

The retrieval evaluation (below) needs none of the above beyond Python.

---

## Real retrieval evaluation on CUAD (no LLM required)

This is the verifiable part of the project. It downloads the real CUAD contracts
and measures how well the retriever finds the correct clause against real gold
labels. It does not need Ollama, a GPU, or any API key.

```bash
cd ~/docusense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-eval.txt

# 1. Download the real CUAD dataset (~18 MB, cached under data/cuad/)
python data/cuad/download_cuad.py

# 2. Run the retrieval eval (sample of 50 real contracts)
python src/eval/retrieval_eval.py --contracts 50
```

It prints hit-rate@k and writes `notebooks/retrieval_results.json`. See
`BENCHMARKS.md` for what the numbers mean and how honest they are.

---

## Option A — Run the full app locally (with the local LLM)

### Step 1: Activate the virtual environment

```bash
cd ~/docusense
source venv/bin/activate
```

### Step 2: Start Ollama (keep this terminal open)

```bash
ollama serve
```

> If models aren't downloaded yet (first time only):
> ```bash
> ollama pull nomic-embed-text   # embedding model (~274MB)
> ollama pull llama3.2            # LLM (~2GB)
> ```

### Step 3: Ingest contracts and build the FAISS index

Only needed if `data/faiss_index/` is empty or you have new contracts:

```bash
# (Optional) Generate sample contracts if data/raw/contracts/ is empty
cd data/raw/contracts && python generate_contracts.py && cd ../../..

# Run PySpark ingestion → Delta Lake
python src/ingestion/pipeline.py

# Chunk and embed → FAISS index
python src/rag/chunker.py
```

> Skip this step if you already have a built index — the FAISS index persists on disk at `data/faiss_index/`.

### Step 4: Start the FastAPI backend (new terminal tab)

```bash
cd ~/docusense
source venv/bin/activate
uvicorn src.api.main:app --port 8000
```

API will be live at: `http://localhost:8000`
API docs (Swagger UI): `http://localhost:8000/docs`

### Step 5: Start the Streamlit UI (new terminal tab)

```bash
cd ~/docusense
source venv/bin/activate
streamlit run src/ui/app.py
```

UI will open at: `http://localhost:8501`

> To index the real CUAD contracts instead of the synthetic demo corpus, run
> `python src/eval/cuad_to_jsonl.py --split test --limit 100` before step 3,
> then rebuild the index with `python src/rag/chunker.py`.

---

## Quick test

Once everything is running, upload `tests/test_contract.txt` in the UI or run:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause?"}'
```

Health check:
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Checklist for a fresh start

| # | What | Command / Action |
|---|------|-----------------|
| 1 | Navigate to project | `cd ~/docusense` |
| 2 | Activate venv | `source venv/bin/activate` |
| 3 | Start Ollama | `ollama serve` (new tab) |
| 4 | (If needed) Rebuild index | `python src/ingestion/pipeline.py && python src/rag/chunker.py` |
| 5 | Start API | `uvicorn src.api.main:app --port 8000` (new tab) |
| 6 | Start UI | `streamlit run src/ui/app.py` (new tab) |
| 7 | Open browser | `http://localhost:8501` |

---

## Common issues

**`JAVA_HOME not set` or PySpark fails**
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

**`ollama: command not found`**
Reinstall from https://ollama.ai — or check that `/usr/local/bin/ollama` is in your PATH.

**Port already in use**
```bash
lsof -ti:8000 | xargs kill   # kill whatever is on 8000
lsof -ti:8501 | xargs kill   # kill whatever is on 8501
```

**FAISS index missing / empty answers**
Re-run steps 3: `python src/ingestion/pipeline.py && python src/rag/chunker.py`

**Packages missing after activating venv**
```bash
pip install -r requirements.txt
```
