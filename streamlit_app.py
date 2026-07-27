"""Streamlit demo of the RAG Document Q&A system.

A self-contained, deployable version of the pipeline:
  upload a document -> chunk -> embed -> hybrid retrieve -> rerank -> generate.

Differences from the local app (and why):
  - LLM: Groq (hosted, free) instead of Ollama (local). The local app keeps
    Ollama for privacy; this public demo uses Groq so it can run on Streamlit
    Cloud, which can't host local models. Same pipeline, swappable backend.
  - Embeddings: a small local sentence-transformers model (runs on Streamlit
    Cloud's limited RAM) instead of Ollama's nomic-embed.
  - Vector store: built in-memory from the uploaded document, so the demo is
    self-contained ("try it with your own file") and ships no database.

The retrieval logic — structure-aware chunking, hybrid (vector + BM25) recall,
cross-encoder reranking — mirrors the production pipeline.
"""

import os
import re

import streamlit as st

# ---- Config ----
GROQ_MODEL = "llama-3.1-8b-instant"          # free, fast
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 8
FINAL_K = 3
SECTION_RE = r"(?=\n\d+\.\s+[A-Z])"
MAX_SECTION_CHARS = 1000

st.set_page_config(page_title="RAG Document Q&A", page_icon="\U0001F4C4", layout="wide")


# ---- Cached heavy resources (load once per session) ----
@st.cache_resource
def load_models():
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return embeddings, reranker


def get_groq_llm():
    """Groq client. Key from env var (local) or Streamlit secrets (cloud) — never hardcoded."""
    from langchain_groq import ChatGroq
    # env var first (local dev); st.secrets only if a secrets file exists (cloud)
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = ""
    if not api_key:
        return None
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key)


# ---- Pipeline pieces ----
def structure_aware_split(text):
    """Split on section headers; fall back to size for long sections."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    fallback = RecursiveCharacterTextSplitter(chunk_size=MAX_SECTION_CHARS, chunk_overlap=100)
    chunks = []
    for section in re.split(SECTION_RE, text):
        section = section.strip()
        if not section:
            continue
        if len(section) <= MAX_SECTION_CHARS:
            chunks.append(section)
        else:
            chunks.extend(fallback.split_text(section))
    return chunks


def build_store(text, embeddings):
    """In-memory vector store + BM25 from the uploaded document."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    chunks = structure_aware_split(text)
    docs = [Document(page_content=c, metadata={"chunk": i}) for i, c in enumerate(chunks)]

    vector = FAISS.from_documents(docs, embeddings)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = RETRIEVE_K
    return vector, bm25, len(chunks)


def hybrid_retrieve(query, vector, bm25):
    """Vector + BM25 union, deduped. High recall."""
    v = vector.similarity_search(query, k=RETRIEVE_K)
    b = bm25.invoke(query)
    seen, merged = set(), []
    for d in v + b:
        if d.page_content not in seen:
            seen.add(d.page_content)
            merged.append(d)
    return merged


def rerank(query, docs, reranker, top_k=FINAL_K):
    """Cross-encoder precision stage."""
    if not docs:
        return []
    scores = reranker.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


RAG_PROMPT = """You are a precise assistant answering questions about a document.
Answer ONLY using the context below. Do not use outside knowledge.
If the answer is not in the context, say exactly: "I don't know — that isn't covered in the document."
Cite the source number in brackets after each fact, like [1] or [2].

CONTEXT:
{context}

QUESTION: {question}
"""


def answer(question, docs, llm):
    context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = llm.invoke(prompt)
    return resp.content


# ---- UI ----
st.title("\U0001F4C4 RAG Document Q&A")
st.caption(
    "Upload a document, ask a question, get a grounded answer with citations — "
    "or an honest \u201cI don't know\u201d if it's not in the document. "
    "Pipeline: structure-aware chunking \u2192 hybrid retrieval (vector + BM25) \u2192 "
    "cross-encoder reranking \u2192 grounded generation."
)

llm = get_groq_llm()
if llm is None:
    st.error("No GROQ_API_KEY configured. Add it in the app's Secrets to enable answers.")
    st.stop()

with st.spinner("Loading models (first run downloads them)…"):
    embeddings, reranker = load_models()

# sample document so the demo works with zero setup
SAMPLE = """1. Definitions
"Services" means the data analytics and machine learning services provided by Provider.
2. Term and Termination
The initial term is twelve (12) months. Either party may terminate for material breach if uncured for fifteen (15) days after written notice.
3. Fees and Payment
Client shall pay Provider a monthly fee of $25,000, due within thirty (30) days of receipt of invoice. Late payments accrue interest at 1.5% per month.
4. Confidentiality
Each party agrees to protect the other's Confidential Information for five (5) years.
5. Governing Law
This Agreement is governed by the laws of the State of Delaware.
"""

col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload a .txt document", type=["txt"])
    text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else st.text_area(
        "…or paste / edit document text", value=SAMPLE, height=260
    )

with col2:
    question = st.text_input("Your question", value="How much does the service cost?")
    ask = st.button("Ask", type="primary")

if ask and text and question:
    with st.spinner("Retrieving, reranking, generating…"):
        vector, bm25, n_chunks = build_store(text, embeddings)
        candidates = hybrid_retrieve(question, vector, bm25)
        top = rerank(question, candidates, reranker)
        result = answer(question, top, llm)

    st.subheader("Answer")
    st.write(result)

    with st.expander(f"Retrieved context ({len(top)} chunks reranked from {len(candidates)} candidates, {n_chunks} total)"):
        for i, d in enumerate(top, 1):
            st.markdown(f"**[{i}]** {d.page_content}")

st.divider()
st.caption(
    "Local version uses Ollama (private, on-device). This public demo uses Groq so it "
    "can run on Streamlit Cloud. Same retrieval pipeline. "
    "Code: github.com/robiulrafi/rag-document-qa-api"
)
