import urllib.request, json, os, time

os.makedirs("edgar", exist_ok=True)

# SEC EDGAR full-text search — material definitive agreements (real contracts)
search_url = "https://efts.sec.gov/LATEST/search-index?q=%22material+definitive+agreement%22&forms=8-K&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31&hits.hits._source=period_of_report,entity_name,file_date"

headers = {"User-Agent": "DocuSense koutilya@usf.edu"}

print("Fetching SEC EDGAR index...")
req = urllib.request.Request(search_url, headers=headers)
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read())

hits = data.get("hits", {}).get("hits", [])
print(f"Found {len(hits)} filings")

downloaded = 0
for hit in hits[:80]:
    if downloaded >= 50:
        break
    try:
        src = hit.get("_source", {})
        entity = src.get("entity_name", "unknown").replace(" ", "_").replace("/", "_")[:40]
        accession = hit.get("_id", "").replace("-", "")
        cik = hit.get("_source", {}).get("cik", hit.get("_routing", ""))

        # Fetch the filing index
        index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=1&search_text="
        
        # Use the direct document URL pattern
        doc_url = f"https://efts.sec.gov/LATEST/search-index?q=%22exhibit+10%22&forms=8-K&dateRange=custom&startdt=2024-01-01&enddt=2024-06-30"
        
        # Get actual exhibit text
        txt_url = f"https://www.sec.gov/Archives/edgar/{accession[:10]}/{accession}.txt" if accession else None
        if not txt_url:
            continue

        req2 = urllib.request.Request(txt_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=15) as r2:
            content = r2.read().decode("utf-8", errors="ignore")

        if len(content) > 500:
            fname = f"edgar/sec_{entity}_{downloaded+1}.txt"
            with open(fname, "w") as f:
                f.write(f"ENTITY: {src.get('entity_name', 'Unknown')}\n")
                f.write(f"DATE: {src.get('file_date', 'Unknown')}\n\n")
                f.write(content[:8000])
            downloaded += 1
            print(f"  [{downloaded}/50] {src.get('entity_name', 'unknown')}")
            time.sleep(0.5)
    except Exception as e:
        continue

print(f"\nDownloaded {downloaded} SEC filings to data/raw/contracts/edgar/")
