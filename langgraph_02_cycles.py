"""The thing a chain CANNOT do: a loop with a conditional branch.

This is structurally identical to self-correcting RAG:
    retrieve -> grade -> (good? generate : rewrite and retry)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: int
    attempts: int
    log: list[str]


def attempt(state: State) -> dict:
    """Stand-in for 'retrieve'."""
    a = state["attempts"] + 1
    v = state["value"] + 4          # gets closer to the target each try
    return {"value": v, "attempts": a,
            "log": state["log"] + [f"attempt {a}: value now {v}"]}


def finish(state: State) -> dict:
    """Stand-in for 'generate'."""
    return {"log": state["log"] + [f"SUCCESS at value {state['value']}"]}


def give_up(state: State) -> dict:
    return {"log": state["log"] + [f"gave up after {state['attempts']} attempts"]}


# --- THE CONDITIONAL EDGE: a function that returns the NAME of the next node ---
def grade(state: State) -> str:
    """Stand-in for 'are the retrieved chunks relevant?'"""
    if state["value"] >= 10:
        return "good"
    if state["attempts"] >= 3:        # loop guard — essential, or you cycle forever
        return "exhausted"
    return "retry"


builder = StateGraph(State)
builder.add_node("attempt", attempt)
builder.add_node("finish", finish)
builder.add_node("give_up", give_up)

builder.add_edge(START, "attempt")
builder.add_conditional_edges(
    "attempt",              # after this node runs...
    grade,                  # ...call this to decide where to go
    {                       # ...and map its return value to a node
        "good": "finish",
        "retry": "attempt",     # <-- THE CYCLE: back to itself
        "exhausted": "give_up",
    },
)
builder.add_edge("finish", END)
builder.add_edge("give_up", END)

graph = builder.compile()

print("=== succeeds on the 3rd loop ===")
r = graph.invoke({"value": 0, "attempts": 0, "log": []})
for l in r["log"]: print("  ", l)

print("\n=== hits the loop guard ===")
r = graph.invoke({"value": -20, "attempts": 0, "log": []})
for l in r["log"]: print("  ", l)
