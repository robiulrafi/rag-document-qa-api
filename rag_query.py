"""RAG: retrieve relevant chunks from the vector store, then answer from them.

This module only READS the vector store. Ingestion (writing) lives in
ingest.py and runs separately — mixing the two is what causes duplicate
chunks, because Chroma.from_documents() appends on every call.
"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
TOP_K = 3


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

# NOTE: Chroma(...) READS an existing store. Never use from_documents() here.
store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": TOP_K})

# temperature=0: faithful, reproducible answers — not creativity.
llm = ChatOllama(model=CHAT_MODEL, temperature=0)

rag_chain = RAG_PROMPT | llm | StrOutputParser()
rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------
def format_context(docs) -> str:
    """Number each chunk so the model can cite it by index."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (page {page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def rewrite_query(question: str, history: list[str]) -> str:
    """Turn a context-dependent follow-up into a standalone search query.

    "When is it due?" + history  ->  "When is the $25,000 monthly fee due?"
    Skipped entirely when there is no history, to avoid a wasted LLM call.
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
    search_query = rewrite_query(question, history or [])   # rewrite for retrieval
    docs = retriever.invoke(search_query)                   # 1. RETRIEVE
    context = format_context(docs)                          # 2. AUGMENT
    answer = rag_chain.invoke(                              # 3. GENERATE
        {"context": context, "question": question}
    )
    return answer, docs


if __name__ == "__main__":
    tests = [
        ("How much does the service cost?", []),
        ("What is the employee vacation policy?", []),          # not in the doc
        (
            "When is it due?",                                   # follow-up
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