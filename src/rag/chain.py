import sys, os
sys.path.insert(0, os.path.abspath("src/rag"))
from chunker import load_vectorstore

from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a legal document analyst. Answer ONLY using the context below.
Always cite the source document filename in your answer.
If the answer is not in the context, say "Not found in provided documents."

Context:
{context}

Question: {question}

Answer (with citation):"""

def build_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="llama3.2", temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

if __name__ == "__main__":
    print("Loading chain...")
    chain = build_qa_chain()
    questions = [
        "What is the termination clause?",
        "What are the indemnification obligations?",
        "What is the governing law?"
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = chain.invoke({"query": q})
        print(f"A: {result['result']}")
        sources = list(set([d.metadata["filename"] for d in result["source_documents"]]))
        print(f"Sources: {sources}")
