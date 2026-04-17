import urllib.request, json, os, time, re

os.makedirs("data/raw/contracts/edgar", exist_ok=True)
headers = {"User-Agent": "DocuSense koutilya@usf.edu"}

def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# Different search queries to get more variety
queries = [
    "https://efts.sec.gov/LATEST/search-index?q=%22governing+law%22+%22indemnify%22&forms=8-K&dateRange=custom&startdt=2024-01-01&enddt=2024-12-01",
    "https://efts.sec.gov/LATEST/search-index?q=%22limitation+of+liability%22+%22termination%22&forms=8-K&dateRange=custom&startdt=2024-01-01&enddt=2024-12-01",
    "https://efts.sec.gov/LATEST/search-index?q=%22intellectual+property%22+%22confidential%22+%22agreement%22&forms=8-K&dateRange=custom&startdt=2023-06-01&enddt=2024-06-01",
]

existing = len(os.listdir("data/raw/contracts/edgar"))
saved = 0

for query_url in queries:
    if saved + existing >= 50:
        break
    try:
        data = json.loads(fetch(query_url))
        hits = data.get("hits", {}).get("hits", [])
        print(f"Query returned {len(hits)} hits")

        for hit in hits:
            if saved + existing >= 50:
                break
            try:
                src = hit.get("_source", {})
                entity = src.get("entity_name", f"co_{saved}").replace(" ","_").replace("/","_").replace(",","")[:35]
                accession = hit.get("_id", "")
                cik = str(hit.get("_routing", ""))
                acc_nodash = accession.replace("-", "")
                cik_unpadded = str(int(cik)) if cik.isdigit() else cik

                filing_index = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{acc_nodash}/{accession}-index.htm"
                index_text = fetch(filing_index, timeout=15)
                links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.(?:htm|txt))"', index_text)

                doc_text = ""
                for link in links[:3]:
                    try:
                        content = fetch("https://www.sec.gov" + link, timeout=15)
                        clean = re.sub(r'<[^>]+>', ' ', content)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        if len(clean) > 1000 and any(w in clean.lower() for w in ["agreement","termination","indemnif","liability"]):
                            doc_text = clean[:10000]
                            break
                    except:
                        continue

                if len(doc_text) > 500:
                    fname = f"data/raw/contracts/edgar/edgar_{existing+saved+1:02d}_{entity}.txt"
                    with open(fname, "w") as f:
                        f.write(f"COMPANY: {src.get('entity_name','Unknown')}\n")
                        f.write(f"FILED: {src.get('file_date','Unknown')}\n")
                        f.write(f"SOURCE: SEC EDGAR\n\n")
                        f.write(doc_text)
                    saved += 1
                    print(f"  [{existing+saved}/50] {src.get('entity_name','unknown')}")
                time.sleep(0.5)
            except Exception as e:
                continue
    except Exception as e:
        print(f"Query failed: {e}")
        continue

print(f"\nAdded {saved} more contracts. Total EDGAR: {existing+saved}")
