import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src/agents"))
sys.path.insert(0, os.path.abspath("src/rag"))

import streamlit as st
import requests
from risk_agent import scan_document

st.set_page_config(page_title="DocuSense", page_icon="", layout="wide")
st.title("DocuSense — Contract Intelligence Platform")
st.caption("Powered by Llama 3.2 + FAISS | Runs 100% locally, zero cost")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Risk Scanner")
    uploaded = st.file_uploader("Upload a contract (.txt)", type=["txt"])
    
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"Loaded: {uploaded.name} ({len(text):,} characters)")
        
        if st.button("Scan for risks", type="primary"):
            with st.spinner("Analyzing clauses with AI..."):
                flags = scan_document(text, uploaded.name)
            
            if flags:
                st.metric("Risk flags found", len(flags))
                for f in flags:
                    if f["severity"] == "high":
                        st.error(f"HIGH — {f['risk_type'].replace('_',' ').title()}: {f['rationale']}")
                    elif f["severity"] == "medium":
                        st.warning(f"MEDIUM — {f['risk_type'].replace('_',' ').title()}: {f['rationale']}")
                    else:
                        st.info(f"LOW — {f['risk_type'].replace('_',' ').title()}: {f['rationale']}")
                    st.caption(f['clause'][:200] + "...")
                    st.divider()
            else:
                st.success("No significant risks detected.")

with col2:
    st.subheader("Ask the contracts")
    question = st.text_input("e.g. What is the termination clause?")
    
    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching contracts..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/query",
                    json={"question": question},
                    timeout=60
                )
                if resp.ok:
                    data = resp.json()
                    st.write(data["answer"])
                    st.caption("Sources: " + ", ".join(data["sources"]))
                else:
                    st.error("API error. Make sure the FastAPI server is running.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: uvicorn src.api.main:app --port 8000")
