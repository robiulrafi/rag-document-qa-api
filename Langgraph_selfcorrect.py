"""Self-correcting RAG as a LangGraph graph.

Integrates everything: structure-aware chunks (from ingest.py), hybrid retrieval
+ cross-encoder reranking, an LLM grader, query rewriting, and grounded generation.

The graph adds ONE capability a straight chain can't express: if the retrieved
context is judged irrelevant, it REWRITES the query and retries — a loop with a
conditional branch.

    START -> retrieve -> grade --+--> generate -> END      (context good enough)
               ^                 |
               |                 +--> rewrite --> (back to retrieve)   (too weak, retry)
               |                 |
               +-----------------+--> generate -> END      (out of attempts: answer anyway,
                                                            grounding will likely refuse)

Reuses the functions already built and measured in src.app.rag_query — the graph
is control flow ON TOP of the existing pipeline, not a reimplementation.
"""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# --- reuse the real pipeline pieces (hybrid + rerank live in retrieve()) ---
from src.app.rag_query import retrieve as hybrid_rerank_retrieve
from src.app.rag_query import format_context, rag_chain, rewrite_chain

MAX_ATTEMPTS = 2          # loop guard: how many rewrites before giving up
RELEVANCE_THRESHOLD = 1   # need at least this many relevant chunks to proceed

# A grader model — separate/stronger than the generator, same lesson as the eval harness.
grader_llm = ChatOllama(model="llama3.1:8b", temperature=0)
GRADE_PROMPT = ChatPromptTemplate.from_template(
    "Is this CHUNK relevant to answering the QUESTION?\n"
    "Answer with one word: YES or NO.\n\n"
    "QUESTION: {question}\n\nCHUNK: {chunk}"
)
grade_chain = GRADE_PROMPT | grader_llm | StrOutputParser()

# For rewriting when retrieval fails (different from the follow-up rewrite).
REWRITE_RETRY_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite this search query using different keywords that might match the document.\n"
    "Output ONLY the rewritten query itself — no preamble, no quotes, no explanation, "
    "no line breaks. Just the query text.\n\nQUERY: {query}"
)
retry_rewrite_chain = REWRITE_RETRY_PROMPT | grader_llm | StrOutputParser()


def _yes(text: str) -> bool:
    first = text.strip().upper().lstrip("*- ").split()
    return bool(first) and first[0].startswith("YES")


# --------------------------------------------------------------------------
# STATE
# --------------------------------------------------------------------------
class State(TypedDict):
    question: str          # original question — answered against this, never changes
    query: str             # search query — rewritten on each retry
    documents: list        # reranked chunks from the last retrieval
    relevant_count: int    # how many passed grading
    attempts: int          # rewrites so far (loop guard)
    answer: str


# --------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------
def retrieve(state: State) -> dict:
    docs = hybrid_rerank_retrieve(state["query"])   # hybrid + cross-encoder rerank
    print(f"[retrieve] attempt {state['attempts']+1}: {len(docs)} chunks for {state['query']!r}")
    return {"documents": docs}


def grade(state: State) -> dict:
    q = state["question"]
    kept = sum(_yes(grade_chain.invoke({"question": q, "chunk": d.page_content}))
               for d in state["documents"])
    print(f"[grade] {kept}/{len(state['documents'])} chunks relevant")
    return {"relevant_count": kept}


def rewrite(state: State) -> dict:
    raw = retry_rewrite_chain.invoke({"query": state["query"]}).strip()
    # small models sometimes add preamble or quotes; take the last non-empty
    # line and strip quotes so a chatty model can't poison the next retrieval.
    lines = [ln.strip().strip('"').strip() for ln in raw.split("\n") if ln.strip()]
    new_q = lines[-1] if lines else state["query"]
    print(f"[rewrite] {state['query']!r} -> {new_q!r}")
    return {"query": new_q, "attempts": state["attempts"] + 1}

def generate(state: State) -> dict:
    context = format_context(state["documents"])
    answer = rag_chain.invoke({"context": context, "question": state["question"]})
    print(f"[generate] answered ({len(state['documents'])} chunks in context)")
    return {"answer": answer}


# --------------------------------------------------------------------------
# THE ROUTER — the conditional edge out of grade
# --------------------------------------------------------------------------
def route_after_grade(state: State) -> str:
    if state["relevant_count"] >= RELEVANCE_THRESHOLD:
        return "good"                     # enough relevant context -> answer
    if state["attempts"] >= MAX_ATTEMPTS:
        return "give_up"                  # out of retries -> answer anyway (likely refuses)
    return "retry"                        # weak context, retries left -> rewrite


# --------------------------------------------------------------------------
# BUILD
# --------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {"good": "generate", "retry": "rewrite", "give_up": "generate"},
)
builder.add_edge("rewrite", "retrieve")     # the cycle
builder.add_edge("generate", END)

graph = builder.compile()


def answer_question_selfcorrecting(question: str):
    """Public entry point mirroring answer_question, but using the graph."""
    final = graph.invoke({
        "question": question, "query": question,
        "documents": [], "relevant_count": 0, "attempts": 0, "answer": "",
    })
    return final["answer"], final["documents"]


if __name__ == "__main__":
    for q in ["How much does the service cost?",
              "What is the employee vacation policy?"]:   # should retry, then refuse
        print(f"\n{'='*62}\nQ: {q}")
        ans, docs = answer_question_selfcorrecting(q)
        print(f"\nA: {ans}")