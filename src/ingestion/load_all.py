import json, hashlib, os

def load_all_contracts() -> list[dict]:
    records = []

    # Load synthetic contracts
    jsonl_path = "data/raw/contracts/cuad_contracts.jsonl"
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("context", "")
            records.append({
                "doc_id": hashlib.md5(text.encode()).hexdigest()[:12],
                "filename": rec.get("title", "unknown"),
                "file_type": ".txt",
                "source": "synthetic",
                "page_count": max(1, len(text) // 3000),
                "text": text,
                "char_count": len(text),
            })

    # Load real SEC EDGAR contracts
    edgar_dir = "data/raw/contracts/edgar"
    for fname in sorted(os.listdir(edgar_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(edgar_dir, fname)
        with open(fpath) as f:
            text = f.read()
        if len(text) < 200:
            continue
        records.append({
            "doc_id": hashlib.md5(text.encode()).hexdigest()[:12],
            "filename": fname.replace(".txt", ""),
            "file_type": ".txt",
            "source": "SEC EDGAR",
            "page_count": max(1, len(text) // 3000),
            "text": text,
            "char_count": len(text),
        })

    print(f"Total: {len(records)} contracts ({sum(1 for r in records if r['source']=='synthetic')} synthetic + {sum(1 for r in records if r['source']=='SEC EDGAR')} EDGAR)")
    return records

if __name__ == "__main__":
    load_all_contracts()
