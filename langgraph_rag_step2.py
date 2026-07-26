"""Self-correcting RAG — Step 2: add the grade node.

Now: retrieve -> grade.  Grade asks the LLM, per chunk, "is this relevant to
the question?" and records how many passed. No routing yet — that's Step 3.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- your existing pieces ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="llama3.2", temperature=0)

# --- grading prompt: forces a single YES/NO first token ---
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You judge whether a document chunk is relevant to a question.\n"
     "Answer with exactly one word: YES or NO. No explanation."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}\n\nRelevant?"),
])
grader_llm = ChatOllama(model="llama3.1:8b", temperature=0)   # separate, stronger
grade_chain = GRADE_PROMPT | grader_llm | StrOutputParser()


def _is_yes(text: str) -> bool:
    """Read only the FIRST token, so 'NO, but...' is never read as YES."""
    return text.strip().upper().startswith("YES")


# --- STATE ---
class State(TypedDict):
    question: str
    query: str
    documents: list
    relevant_count: int      # NEW: how many chunks passed grading
    attempts: int
    answer: str


# --- NODE 1: retrieve ---
def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["query"])
    print(f"[retrieve] {len(docs)} chunks for query={state['query']!r}")
    return {"documents": docs}


# --- NODE 2: grade ---
def grade(state: State) -> dict:
    question = state["question"]
    kept = 0
    for i, doc in enumerate(state["documents"], 1):
        verdict = grade_chain.invoke({"question": question, "chunk": doc.page_content})
        ok = _is_yes(verdict)
        kept += ok
        print(f"[grade] chunk {i}: {verdict.strip()[:12]!r} -> {'KEEP' if ok else 'drop'}")
    print(f"[grade] {kept}/{len(state['documents'])} chunks relevant")
    return {"relevant_count": kept}


# --- BUILD: retrieve -> grade -> END (routing comes in Step 3) ---
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_edge("grade", END)
graph = builder.compile()


if __name__ == "__main__":
    initial = {
        "question": "How much does the service cost?",
        "query": "How much does the service cost?",
        "documents": [], "relevant_count": 0, "attempts": 0, "answer": "",
    }
    result = graph.invoke(initial)
    print(f"\nFINAL: {result['relevant_count']}/{len(result['documents'])} chunks graded relevant")
