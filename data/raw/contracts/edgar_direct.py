import urllib.request, json, os, time

os.makedirs("edgar", exist_ok=True)
headers = {"User-Agent": "DocuSense koutilya@usf.edu"}

def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# Step 1: search for 8-K filings with contract language
search = "https://efts.sec.gov/LATEST/search-index?q=%22indemnification%22+%22termination%22&forms=8-K&dateRange=custom&startdt=2023-01-01&enddt=2024-06-01&hits.hits.total.value=true"

print("Searching EDGAR...")
data = json.loads(fetch(search))
hits = data.get("hits", {}).get("hits", [])
print(f"Got {len(hits)} results")

saved = 0
for hit in hits:
    if saved >= 50:
        break
    try:
        src = hit.get("_source", {})
        entity = src.get("entity_name", f"entity_{saved}").replace(" ", "_").replace("/","_")[:35]
        accession = hit.get("_id", "")          # e.g. 0001234567-24-000001
        cik = str(hit.get("_routing", ""))       # numeric CIK

        # Build the EDGAR viewer URL for the filing document
        acc_nodash = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{accession}.txt"

        text = fetch(doc_url, timeout=25)

        if len(text) > 500:
            fname = f"edgar/edgar_{saved+1:02d}_{entity}.txt"
            with open(fname, "w") as f:
                f.write(f"COMPANY: {src.get('entity_name', 'Unknown')}\n")
                f.write(f"FILED: {src.get('file_date', 'Unknown')}\n")
                f.write(f"SOURCE: SEC EDGAR\n\n")
                f.write(text[:10000])
            saved += 1
            print(f"  [{saved}/50] {src.get('entity_name', 'unknown')}")
            time.sleep(0.4)
        else:
            print(f"  skip (too short): {entity}")

    except Exception as e:
        print(f"  skip: {e}")
        continue

print(f"\nDone. Saved {saved} real SEC contracts to edgar/")
