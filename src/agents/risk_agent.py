import sys, os, json
sys.path.insert(0, os.path.abspath("src/rag"))

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

RISK_TYPES = [
    "indemnification", "auto_renewal", "penalty_clause",
    "ip_ownership", "arbitration", "limitation_of_liability",
    "termination_for_convenience", "data_privacy"
]

RISK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a contract risk analyst. Given a contract clause, identify:
1. risk_type: one of {risk_types} or "other"
2. severity: low | medium | high
3. rationale: one sentence explanation

Respond ONLY with valid JSON like this:
{{"risk_type": "indemnification", "severity": "high", "rationale": "Broad indemnification with no cap on liability."}}

No other text. JSON only."""),
    ("human", "Clause: {clause}")
])

llm = ChatOllama(model="llama3.2", temperature=0)
chain = RISK_PROMPT | llm

def classify_risk(clause: str) -> dict:
    try:
        result = chain.invoke({"clause": clause, "risk_types": RISK_TYPES})
        text = result.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        return {"risk_type": "unknown", "severity": "low", "rationale": f"Parse error: {e}"}

def scan_document(doc_text: str, doc_id: str) -> list[dict]:
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if len(p.strip()) > 80]
    flags = []
    print(f"Scanning {len(paragraphs)} paragraphs...")
    for i, para in enumerate(paragraphs[:20]):
        risk = classify_risk(para)
        if risk.get("severity") in ["medium", "high"]:
            flags.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "clause": para[:300],
                **risk
            })
    return flags

if __name__ == "__main__":
    sample_clause = "Licensee shall indemnify, defend, and hold harmless Licensor from any and all claims, damages, and expenses arising out of Licensee's use of the Software. This indemnification shall survive termination."
    print("Testing single clause...")
    result = classify_risk(sample_clause)
    print(json.dumps(result, indent=2))

    print("\nScanning full document...")
    sample_doc = open("data/raw/contracts/cuad_contracts.jsonl").readline()
    import json as j
    record = j.loads(sample_doc)
    flags = scan_document(record["context"], "test_001")
    print(f"\nFound {len(flags)} risk flags:")
    for f in flags:
        print(f"  [{f['severity'].upper()}] {f['risk_type']}: {f['rationale']}")
