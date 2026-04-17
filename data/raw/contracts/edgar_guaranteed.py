import urllib.request, json, os, time

os.makedirs("data/raw/contracts/edgar", exist_ok=True)
headers = {"User-Agent": "DocuSense koutilya@usf.edu"}

def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# Use EDGAR full text search — returns actual document text snippets
search_url = "https://efts.sec.gov/LATEST/search-index?q=%22indemnification%22+%22termination+clause%22&forms=8-K&dateRange=custom&startdt=2023-01-01&enddt=2024-12-01"

print("Fetching from EDGAR full-text search...")
data = json.loads(fetch(search_url))
hits = data.get("hits", {}).get("hits", [])
print(f"Found {len(hits)} filings")

saved = 0
for hit in hits:
    if saved >= 50:
        break
    try:
        src = hit.get("_source", {})
        entity = src.get("entity_name", f"co_{saved}").replace(" ", "_").replace("/","_").replace(",","")[:35]
        accession = hit.get("_id", "")
        cik = str(hit.get("_routing", ""))

        # Correct EDGAR URL: use unpadded CIK + nodash accession
        acc_nodash = accession.replace("-", "")
        cik_unpadded = str(int(cik)) if cik.isdigit() else cik

        # Try the primary document index first
        index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_unpadded}&type=8-K&dateb=&owner=include&count=1&output=atom"
        
        # Direct approach: fetch the filing index page
        filing_index = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{acc_nodash}/{accession}-index.htm"
        
        index_text = fetch(filing_index, timeout=15)
        
        # Find .htm or .txt document links in the index
        import re
        links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.(?:htm|txt))"', index_text)
        
        doc_text = ""
        for link in links[:3]:
            try:
                full_url = "https://www.sec.gov" + link
                content = fetch(full_url, timeout=15)
                # Strip HTML tags
                clean = re.sub(r'<[^>]+>', ' ', content)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if len(clean) > 1000 and any(w in clean.lower() for w in ["agreement", "termination", "indemnif"]):
                    doc_text = clean[:10000]
                    break
            except:
                continue

        if len(doc_text) > 500:
            fname = f"data/raw/contracts/edgar/edgar_{saved+1:02d}_{entity}.txt"
            with open(fname, "w") as f:
                f.write(f"COMPANY: {src.get('entity_name', 'Unknown')}\n")
                f.write(f"FILED: {src.get('file_date', 'Unknown')}\n")
                f.write(f"ACCESSION: {accession}\n")
                f.write(f"SOURCE: SEC EDGAR\n\n")
                f.write(doc_text)
            saved += 1
            print(f"  [{saved}/50] {src.get('entity_name', 'unknown')}")
        else:
            print(f"  skip (no contract text): {entity}")

        time.sleep(0.5)

    except Exception as e:
        print(f"  skip: {type(e).__name__}: {str(e)[:60]}")
        continue

print(f"\nDone. Saved {saved} SEC contracts")
