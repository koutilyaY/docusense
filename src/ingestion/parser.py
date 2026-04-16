import json, hashlib

def parse_text_record(record: dict) -> dict:
    text = record.get("context", "")
    doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
    return {
        "doc_id": doc_id,
        "filename": record.get("title", "unknown"),
        "file_type": ".txt",
        "page_count": max(1, len(text) // 3000),
        "text": text,
        "char_count": len(text),
    }

def load_contracts(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(parse_text_record(rec))
    return records

if __name__ == "__main__":
    records = load_contracts("data/raw/contracts/cuad_contracts.jsonl")
    print(f"Loaded {len(records)} contracts")
    print(json.dumps({k: v for k, v in records[0].items() if k != "text"}, indent=2))
    print(f"Text preview: {records[0]['text'][:200]}")
