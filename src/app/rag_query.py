"""RAG with reranking: retrieve broadly, rerank with a cross-encoder, keep the best.

Only READS the vector store (ingest.py writes it).

Retrieval is a two-stage funnel:
  1. HYBRID (vector + BM25) casts a WIDE net — high recall, retrieve ~10 candidates.
     Vector alone missed exact-term answers (e.g. "$25,000 monthly fee"); BM25 catches them.
  2. RERANK with a cross-encoder scores each (query, chunk) pair jointly and keeps the
     top few — high precision. This is what drops the 4-of-5 noise chunks the retriever
     returns. Measured context precision was ~0.18 (≈1 relevant chunk per 5) before reranking.

Why two stages: a bi-encoder (embeddings) encodes query and chunk SEPARATELY and compares
by cosine — fast enough to search the whole corpus, but imprecise. A cross-encoder reads
(query, chunk) TOGETHER — far more accurate, but too slow to run over every chunk. So you
retrieve broadly with the fast method, then rerank the small candidate set with the slow,
accurate one. Standard production pattern.
"""

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVE_K = 10     # cast a wide net (per retriever) before reranking
FINAL_K = 3         # keep this many after reranking


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise assistant answering questions about a document.\n\n"
            "Answer ONLY using the context below. Do not use outside knowledge.\n"
            "If the answer is not in the context, say exactly: "
            "\"I don't know — that isn't covered in the document.\"\n"
            "Cite the source number in brackets after each fact, like [1] or [2].\n"
            "Every factual claim must have a citation.\n\n"
            "CONTEXT:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation history and a follow-up question, rewrite the "
            "follow-up as a standalone question. Do not answer it. Return only the "
            "rewritten question.",
        ),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ]
)


# --------------------------------------------------------------------------
# Store, models, retrievers — built once at import
# --------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
vector_retriever = store.as_retriever(search_kwargs={"k": RETRIEVE_K})

# Rebuild BM25 in-memory from chunks already in Chroma (no PDF re-parse).
_raw = store.get()
_docs = [
    Document(page_content=t, metadata=m or {})
    for t, m in zip(_raw["documents"], _raw["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(_docs)
bm25_retriever.k = RETRIEVE_K

# Cross-encoder reranker — loaded once (downloads on first run, then cached).
reranker = CrossEncoder(RERANK_MODEL)

llm = ChatOllama(model=CHAT_MODEL, temperature=0)
rag_chain = RAG_PROMPT | llm | StrOutputParser()
rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Retrieval: hybrid recall  ->  cross-encoder precision
# --------------------------------------------------------------------------
def _hybrid_candidates(query: str) -> list[Document]:
    """Wide net: union vector + BM25, dedupe by content. High recall."""
    hits = vector_retriever.invoke(query) + bm25_retriever.invoke(query)
    seen, merged = set(), []
    for doc in hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged


def rerank(query: str, docs: list[Document], top_k: int = FINAL_K) -> list[Document]:
    """Score each (query, chunk) pair with the cross-encoder; keep the top_k."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def retrieve(query: str) -> list[Document]:
    """Full two-stage retrieval: hybrid recall, then cross-encoder precision."""
    candidates = _hybrid_candidates(query)   # ~up to 2*RETRIEVE_K, deduped
    return rerank(query, candidates)         # trimmed to FINAL_K best


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (page {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def rewrite_query(question: str, history: list[str]) -> str:
    if not history:
        return question
    return rewrite_chain.invoke({"history": "\n".join(history), "question": question})


def answer_question(question: str, history: list[str] | None = None):
    """Retrieve (hybrid + rerank) -> augment -> generate.

    Retrieval uses the REWRITTEN query; generation answers the ORIGINAL question.
    Returns (answer, docs) so the caller can cite sources.
    """
    search_query = rewrite_query(question, history or [])
    docs = retrieve(search_query)
    context = format_context(docs)
    answer = rag_chain.invoke({"context": context, "question": question})
    return answer, docs


if __name__ == "__main__":
    for q in ["How much does the service cost?",
              "What is the employee vacation policy?"]:
        ans, srcs = answer_question(q)
        print(f"\n{'='*60}\nQ: {q}\nA: {ans}\n")
        print(f"Sources ({len(srcs)} after rerank):")
        for i, d in enumerate(srcs, 1):
            print(f"  [{i}] page {d.metadata.get('page')}: {d.page_content[:65]}...")