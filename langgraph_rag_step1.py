"""Self-correcting RAG — Step 1: State + the retrieve node.

We build ONE node and run it, to confirm the plumbing works before adding
grade / rewrite / generate. Everything here calls YOUR existing code.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# --- your existing pipeline pieces ---
# (import from your module; adjust the import path to match your project)
# from src.app.rag_query import retriever, rewrite_query, format_context
#
# For this standalone demo we import the retriever the same way your file does:
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})


# 1. STATE — everything that flows through the graph
class State(TypedDict):
    question: str        # the original question (never changes)
    query: str           # the search query (will be rewritten later)
    documents: list      # chunks from the last retrieval
    attempts: int        # loop guard for later
    answer: str          # filled in by generate, later


# 2. THE retrieve NODE — reads `query`, writes `documents`
def retrieve(state: State) -> dict:
    query = state["query"]
    docs = retriever.invoke(query)          # <-- your real retrieval
    print(f"[retrieve] query={query!r} -> {len(docs)} chunks")
    return {"documents": docs}              # only return what changed


# 3. BUILD a one-node graph
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", END)
graph = builder.compile()


# 4. RUN it
if __name__ == "__main__":
    initial = {
        "question": "How much does the service cost?",
        "query":    "How much does the service cost?",   # same as question on first pass
        "documents": [],
        "attempts": 0,
        "answer": "",
    }
    result = graph.invoke(initial)

    print(f"\nquestion in state : {result['question']!r}")
    print(f"documents in state: {len(result['documents'])} chunks")
    for i, d in enumerate(result["documents"], 1):
        print(f"  [{i}] page {d.metadata.get('page')}: {d.page_content[:70]}...")
