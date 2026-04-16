import sys, os
sys.path.insert(0, os.path.abspath("src/ingestion"))
from parser import load_contracts

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "]
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def chunk_and_embed(jsonl_path: str, save_path: str = "data/faiss_index"):
    docs = load_contracts(jsonl_path)
    all_chunks, all_metadata = [], []

    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

    print(f"Created {len(all_chunks)} chunks from {len(docs)} docs")
    print("Embedding... this takes 2-3 mins locally...")
    vectorstore = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadata)
    vectorstore.save_local(save_path)
    print(f"Done. FAISS index saved to {save_path}")
    return vectorstore

def load_vectorstore(path: str = "data/faiss_index"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    chunk_and_embed("data/raw/contracts/cuad_contracts.jsonl")
