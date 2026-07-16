"""RAG: hybrid retrieval over an ingested document, then a grounded answer.

This module only READS the vector store. Ingestion (writing) lives in
ingest.py and runs separately — mixing the two is what causes duplicate
chunks, because Chroma.from_documents() appends on every call.

Retrieval is hybrid: a dense vector search (semantic) merged with BM25
(exact keyword). The BM25 index is rebuilt in memory at startup from the
documents already stored in Chroma, so there is still a single source of
truth and no second index to keep in sync.
"""

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
TOP_K = 3          # per retriever
MAX_CONTEXT = 5    # cap after merging, to bound prompt size


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
            "follow-up as a standalone question that makes sense without the history. "
            "Do not answer it. Return only the rewritten question.",
        ),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ]
)


# --------------------------------------------------------------------------
# Store + models — built once at import, reused across requests
# --------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Chroma(...) READS an existing store. Never use from_documents() here.
store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
vector_retriever = store.as_retriever(search_kwargs={"k": TOP_K})


def _load_chunks_from_store() -> list[Document]:
    """Read every stored chunk back out of Chroma, as Documents.

    Chroma persists the original text alongside the vectors, so BM25 can be
    rebuilt from the store itself — no need to re-parse the source PDF, and
    no second index file to keep in sync.
    """
    raw = store.get()  # {'ids': [...], 'documents': [...], 'metadatas': [...]}
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


# BM25 is an in-memory index — unlike Chroma it does not persist, so it is
# rebuilt on each startup. Cheap at this scale; at large scale you would use
# a store that does both keyword and vector search (e.g. OpenSearch).
_chunks = _load_chunks_from_store()
bm25_retriever = BM25Retriever.from_documents(_chunks) if _chunks else None
if bm25_retriever:
    bm25_retriever.k = TOP_K

llm = ChatOllama(model=CHAT_MODEL, temperature=0)

rag_chain = RAG_PROMPT | llm | StrOutputParser()
rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def hybrid_search(query: str) -> list[Document]:
    """Merge dense (semantic) and sparse (keyword) retrieval.

    Vector search matches meaning — "cost" finds a clause that says "fee".
    BM25 matches exact tokens — party names, case numbers, dollar figures —
    which dense embeddings represent weakly. Running both covers each one's
    blind spot.
    """
    hits: list[Document] = vector_retriever.invoke(query)
    if bm25_retriever:
        hits = hits + bm25_retriever.invoke(query)

    seen, merged = set(), []
    for doc in hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged[:MAX_CONTEXT]


def format_context(docs: list[Document]) -> str:
    """Number each chunk so the model can cite it by index."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (page {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def rewrite_query(question: str, history: list[str]) -> str:
    """Turn a context-dependent follow-up into a standalone search query.

    "When is it due?" + history  ->  "When is the $25,000 monthly fee due?"
    Skipped when there is no history, to avoid a wasted LLM call.
    """
    if not history:
        return question
    return rewrite_chain.invoke(
        {"history": "\n".join(history), "question": question}
    )


def answer_question(question: str, history: list[str] | None = None):
    """Retrieve -> augment -> generate.

    Retrieval uses the REWRITTEN query (so follow-ups find the right chunks),
    but the model answers the ORIGINAL question the user actually asked.

    Returns (answer, retrieved_docs) so the caller can cite sources.
    """
    search_query = rewrite_query(question, history or [])
    docs = hybrid_search(search_query)                      # 1. RETRIEVE
    context = format_context(docs)                          # 2. AUGMENT
    answer = rag_chain.invoke(                              # 3. GENERATE
        {"context": context, "question": question}
    )
    return answer, docs


if __name__ == "__main__":
    print(f"Loaded {len(_chunks)} chunks from the store "
          f"(BM25: {'on' if bm25_retriever else 'off'})")

    tests = [
        ("How much does the service cost?", []),            # semantic win
        ("Northwind Trading", []),                          # exact-term / BM25 win
        ("What is the employee vacation policy?", []),      # not in the doc
        (
            "When is it due?",                              # follow-up
            ["Q: How much does the service cost?",
             "A: The monthly fee is $25,000."],
        ),
    ]
    for q, hist in tests:
        ans, srcs = answer_question(q, hist)
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        if hist:
            print(f"   (rewritten -> {rewrite_query(q, hist)!r})")
        print(f"A: {ans}\n")
        print("Sources:")
        for i, d in enumerate(srcs, 1):
            print(f"  [{i}] page {d.metadata.get('page')}: {d.page_content[:70]}...")