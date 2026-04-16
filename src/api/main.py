import sys, os
sys.path.insert(0, os.path.abspath("src/rag"))
from chain import build_qa_chain

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DocuSense API", version="1.0")
print("Loading QA chain...")
qa_chain = build_qa_chain()
print("Ready.")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = qa_chain.invoke({"query": req.question})
    sources = list(set([d.metadata["filename"] for d in result["source_documents"]]))
    return QueryResponse(answer=result["result"], sources=sources)

@app.get("/health")
def health():
    return {"status": "ok"}
