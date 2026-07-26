"""LangGraph Day 1 — the primitives, with no LLM involved.

A graph has three things:
  STATE  — a dict that flows through the graph; each node reads it and returns updates
  NODES  — plain functions: take state, return a dict of changes
  EDGES  — wiring that says which node runs next (fixed, or conditional)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. STATE — declare what flows through the graph
class State(TypedDict):
    number: int
    log: list[str]


# 2. NODES — each takes state, returns ONLY the keys it changes
def double(state: State) -> dict:
    n = state["number"] * 2
    return {"number": n, "log": state["log"] + [f"doubled -> {n}"]}

def add_ten(state: State) -> dict:
    n = state["number"] + 10
    return {"number": n, "log": state["log"] + [f"added 10 -> {n}"]}


# 3. BUILD the graph: register nodes, then wire edges
builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("add_ten", add_ten)

builder.add_edge(START, "double")      # entry point
builder.add_edge("double", "add_ten")  # fixed edge: always goes here next
builder.add_edge("add_ten", END)       # exit

graph = builder.compile()

# 4. RUN
result = graph.invoke({"number": 5, "log": []})
print("=== linear graph ===")
print("final number:", result["number"])
for line in result["log"]:
    print("  ", line)
