import streamlit as st
import os
from rag_backend import load_index, build_index, load_pdfs, answer_query

st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("📄 Chat with your PDFs")

# -----------------------------
# Build index (only once)
# -----------------------------
if not os.path.exists("faiss_ivf.index"):
    st.warning("⚠️ Building index (first time only, may take a few minutes)...")

    with st.spinner("Processing PDFs..."):
        docs = load_pdfs()
        build_index(docs)

    st.success("✅ Index ready!")
    st.rerun()

# -----------------------------
# Load index
# -----------------------------
index, docs = load_index()

# -----------------------------
# Chat memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask something about your PDFs...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        answer, sources = answer_query(query, index, docs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# -----------------------------
# Display chat
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} (chunk {s['chunk_id']})")