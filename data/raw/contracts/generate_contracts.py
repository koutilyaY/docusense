import json, random, os

templates = [
    {
        "title": "Software License Agreement {n}",
        "context": """SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 1, 202{n}, between TechCorp Inc. ("Licensor") and Client Company {n} ("Licensee").

1. GRANT OF LICENSE
Licensor hereby grants Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business purposes.

2. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and continue for a period of two (2) years. Either party may terminate this Agreement upon thirty (30) days written notice.

3. INDEMNIFICATION
Licensee shall indemnify, defend, and hold harmless Licensor from any claims arising out of Licensee's use of the Software. This indemnification obligation shall survive termination of this Agreement.

4. LIMITATION OF LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES. LICENSOR'S TOTAL LIABILITY SHALL NOT EXCEED THE FEES PAID IN THE PRECEDING THREE MONTHS.

5. AUTO-RENEWAL
This Agreement shall automatically renew for successive one-year terms unless either party provides sixty (60) days written notice of non-renewal prior to expiration.

6. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

7. INTELLECTUAL PROPERTY
All intellectual property rights in the Software remain exclusively with Licensor. Licensee acquires no ownership rights whatsoever.

8. ARBITRATION
Any dispute arising out of this Agreement shall be resolved by binding arbitration in accordance with the rules of the American Arbitration Association."""
    },
    {
        "title": "Non-Disclosure Agreement {n}",
        "context": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is made between Alpha Corp {n} ("Disclosing Party") and Beta Solutions {n} ("Receiving Party").

1. CONFIDENTIAL INFORMATION
Receiving Party agrees to hold all Confidential Information in strict confidence and not disclose it to any third party without prior written consent.

2. TERM
This NDA shall remain in effect for a period of three (3) years from the Effective Date.

3. PENALTIES
Breach of this Agreement may result in irreparable harm. Disclosing Party shall be entitled to seek injunctive relief and liquidated damages of $500,000 per breach.

4. RETURN OF INFORMATION
Upon termination, Receiving Party shall immediately return or destroy all Confidential Information and certify such destruction in writing.

5. GOVERNING LAW
This Agreement is governed by the laws of California."""
    },
    {
        "title": "Service Agreement {n}",
        "context": """PROFESSIONAL SERVICES AGREEMENT

This Agreement is between ServicePro {n} ("Service Provider") and Enterprise Client {n} ("Client").

1. SERVICES
Service Provider shall perform the services described in each Statement of Work executed by the parties.

2. PAYMENT TERMS
Client shall pay invoices within thirty (30) days. Late payments shall accrue interest at 1.5% per month.

3. INTELLECTUAL PROPERTY OWNERSHIP
All work product created under this Agreement shall be owned exclusively by Client upon full payment of all fees.

4. TERMINATION FOR CONVENIENCE
Client may terminate this Agreement for any reason upon fourteen (14) days written notice. Service Provider shall be paid for all work completed through termination date.

5. DATA PRIVACY
Service Provider shall comply with all applicable data protection laws including GDPR and CCPA when processing Client data.

6. LIMITATION OF LIABILITY
Service Provider's liability shall not exceed the total fees paid in the six months preceding the claim."""
    }
]

os.makedirs(".", exist_ok=True)
records = []
for i in range(150):
    tmpl = templates[i % len(templates)]
    records.append({
        "title": tmpl["title"].replace("{n}", str(i+1)),
        "context": tmpl["context"].replace("{n}", str(i+1))
    })

with open("cuad_contracts.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Generated {len(records)} synthetic contracts")
