"""
Download the real CUAD dataset (Contract Understanding Atticus Dataset).

CUAD is 510 real commercial contracts annotated by legal experts across 41
clause categories, released by The Atticus Project under CC BY 4.0. Each
contract comes as a SQuAD-style record: one context (the full contract text)
and 41 questions (one per clause category), where an answerable question
carries a gold answer span (the exact clause text plus its character offset).

We pull the official data.zip from the Atticus GitHub repo and keep the
extracted JSON under data/cuad/ (gitignored). This is the raw material for
the retrieval evaluation in src/eval/retrieval_eval.py.

Source: https://github.com/TheAtticusProject/cuad  (file: data.zip)
Paper:  Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal
        Contract Review", NeurIPS 2021.

Usage:
    python data/cuad/download_cuad.py
"""
import os
import zipfile
import urllib.request

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "data.zip")
EXPECTED = ["CUADv1.json", "test.json", "train_separate_questions.json"]


def already_present() -> bool:
    return all(os.path.exists(os.path.join(HERE, f)) for f in EXPECTED)


def main():
    if already_present():
        print("CUAD already extracted under data/cuad/ — nothing to do.")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading CUAD data.zip (~18 MB) from {CUAD_URL} ...")
        urllib.request.urlretrieve(CUAD_URL, ZIP_PATH)
        print(f"Saved {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1e6:.1f} MB)")
    else:
        print(f"Using existing {ZIP_PATH}")

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(HERE)

    for f in EXPECTED:
        p = os.path.join(HERE, f)
        status = f"{os.path.getsize(p)/1e6:.1f} MB" if os.path.exists(p) else "MISSING"
        print(f"  {f}: {status}")
    print("Done. Real CUAD contracts are ready under data/cuad/.")


if __name__ == "__main__":
    main()
