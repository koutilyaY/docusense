import sys
sys.path.insert(0, "src/rag")
sys.path.insert(0, "src/ingestion")
from chain import build_qa_chain

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
import json, datetime

print("Loading QA chain...")
chain = build_qa_chain()

# Wire RAGAS to use Ollama instead of OpenAI
ollama_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2", temperature=0))
ollama_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

for metric in [faithfulness, answer_relevancy, context_precision]:
    metric.llm = ollama_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ollama_embeddings

test_questions = [
    "What is the termination clause?",
    "What are the indemnification obligations?",
    "What happens upon auto-renewal?",
    "What is the limitation of liability?",
    "Who owns the intellectual property?",
    "What are the payment terms?",
    "What constitutes a breach of contract?",
    "What are the confidentiality obligations?",
    "What is the arbitration process?",
    "How many days notice is required for termination?",
    "What data privacy laws must be complied with?",
    "What are the penalty clauses?",
    "Can the agreement be assigned to a third party?",
    "What are the service provider obligations?",
    "What is the term of the agreement?",
    "What are the licensor restrictions?",
    "What triggers the indemnification obligation?",
    "What are the renewal terms?",
    "What happens to work product after termination?",
    "What is the governing law?",
]

data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

print(f"Running {len(test_questions)} queries...\n")
for i, q in enumerate(test_questions):
    print(f"  [{i+1}/{len(test_questions)}] {q}")
    result = chain.invoke({"query": q})
    data["question"].append(q)
    data["answer"].append(result["result"])
    data["contexts"].append([d.page_content for d in result["source_documents"]])
    data["ground_truth"].append("")

print("\nRunning RAGAS evaluation (this takes ~10 mins locally)...")
dataset = Dataset.from_dict(data)
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ollama_llm,
    embeddings=ollama_embeddings,
)

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(f"Faithfulness:      {scores['faithfulness']:.4f}  (target: >0.85)")
print(f"Answer Relevancy:  {scores['answer_relevancy']:.4f}  (target: >0.80)")
print(f"Context Precision: {scores['context_precision']:.4f}  (target: >0.75)")
print("="*50)

results = {
    "timestamp": datetime.datetime.now().isoformat(),
    "num_questions": len(test_questions),
    "corpus": "200 contracts (150 synthetic + 50 SEC EDGAR)",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "embedding_model": "nomic-embed-text",
    "llm": "llama3.2",
    "scores": {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
    }
}

with open("notebooks/ragas_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to notebooks/ragas_results.json")
