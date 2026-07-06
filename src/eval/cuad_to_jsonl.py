"""
Convert real CUAD contracts into the jsonl format DocuSense's ingestion +
chunker expect ({"title", "context"} per line).

This lets the existing local RAG index (nomic-embed-text + FAISS, served by
Ollama) be rebuilt over the *real* CUAD corpus instead of the old synthetic
templates. It is independent of the retrieval eval, which builds its own index
with sentence-transformers.

Usage:
    python src/eval/cuad_to_jsonl.py --split test --limit 100
    # then rebuild the RAG index with the existing chunker:
    #   python src/rag/chunker.py   (reads data/raw/contracts/cuad_contracts.jsonl)
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SPLIT_FILES = {
    "test": os.path.join(ROOT, "data", "cuad", "test.json"),
    "CUADv1": os.path.join(ROOT, "data", "cuad", "CUADv1.json"),
}
OUT = os.path.join(ROOT, "data", "raw", "contracts", "cuad_contracts.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_FILES), default="test")
    ap.add_argument("--limit", type=int, default=100, help="max contracts to write")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    path = SPLIT_FILES[args.split]
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run: python data/cuad/download_cuad.py")

    data = json.load(open(path))["data"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for c in data[: args.limit]:
            context = c["paragraphs"][0]["context"]
            rec = {"title": c["title"], "context": context}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} real CUAD contracts ({args.split}) to {args.out}")


if __name__ == "__main__":
    main()
