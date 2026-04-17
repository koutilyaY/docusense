import sys
sys.path.insert(0, "src/ingestion")
sys.path.insert(0, "src/rag")
from load_all import load_all_contracts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = load_all_contracts()
all_chunks, all_metadata = [], []

for doc in docs:
    chunks = splitter.split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "source": doc["source"],
            "chunk_index": i,
        })

print(f"Embedding {len(all_chunks)} chunks... (~5 mins)")
vectorstore = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadata)
vectorstore.save_local("data/faiss_index")
print(f"Done. Index saved with {len(all_chunks)} chunks from {len(docs)} contracts.")
