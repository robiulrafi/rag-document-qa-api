"""LLM-as-judge evaluation for the RAG pipeline.

RAGAS was the intended tool, but it pins to LangChain internals that have since
moved — installing it broke the working stack, and downgrading langchain-community
to satisfy it broke langchain-ollama. So the metrics are implemented directly.

Two design decisions came out of measurement, not theory:

1. Never ask the judge for a score.
   A first version asked for "a number between 0.0 and 1.0" and got 0.5 on 14 of
   18 scores — including a refusal that makes no factual claims and should have
   been 1.0. The judge was hedging, not evaluating. Small models cannot produce
   calibrated continuous scores. The fix is to decompose into atomic units, ask a
   BINARY question about each, and compute the score arithmetically. That is what
   RAGAS does internally, and it is why it tolerates weaker judges.

2. The judge must be a different, stronger model than the one under test.
   With llama3.2 (the generation model) as judge, relevancy scored 0.50 and
   context precision 0.13 — while the answers were citing retrieved chunks and
   answering correctly. A self-contradiction: if a chunk yields a faithful answer,
   it is relevant. Swapping to llama3.1:8b moved relevancy 0.50 -> 1.00 and made
   claim extraction meaningful (1 claim per answer -> 3-5). The judge was wrong,
   not the system.

   The lesson: a small model can do near-extractive judgments (does this context
   support this claim?) but not abstract relevance judgments. Validate the judge
   before trusting the metric — an unvalidated harness will confidently tell you
   to fix things that aren't broken.

Current results (6-question golden set, llama3.1:8b judge):
    faithfulness      0.96   reliable
    answer relevancy  1.00   reliable
    context precision 0.21   directional — hybrid retrieval favours recall over
                             precision; ~4 of 5 retrieved chunks are noise.
                             This is the number reranking should move.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

from src.app.rag_query import answer_question

VERBOSE = True

# A separate, larger model for judging. The generation model (llama3.2) is fine
# at extractive work but cannot make abstract relevance judgments — it scored
# 0/5 chunks relevant on a question its own answer had correctly cited.
judge = ChatOllama(model="llama3.1:8b", temperature=0)


# --------------------------------------------------------------------------
# Judge prompts — all binary or extractive. None ask for a number.
# --------------------------------------------------------------------------
CLAIMS_PROMPT = ChatPromptTemplate.from_template(
    "List each distinct factual claim in the ANSWER, one per line.\n"
    "No numbering, no commentary, no preamble.\n"
    "If the ANSWER makes no factual claims, output exactly: NONE\n\n"
    "ANSWER: {answer}"
)

SUPPORTED_PROMPT = ChatPromptTemplate.from_template(
    "Does the CONTEXT support this CLAIM?\n"
    "Answer with one word: YES or NO.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM: {claim}"
)

ADDRESSES_PROMPT = ChatPromptTemplate.from_template(
    "Does the ANSWER address the QUESTION?\n"
    "An honest refusal counts as addressing it.\n"
    "Answer with one word: YES or NO.\n\n"
    "QUESTION: {question}\n\nANSWER: {answer}"
)

CHUNK_RELEVANT_PROMPT = ChatPromptTemplate.from_template(
    "Is this CHUNK relevant to answering the QUESTION?\n"
    "Answer with one word: YES or NO.\n\n"
    "QUESTION: {question}\n\nCHUNK: {chunk}"
)

claims_chain = CLAIMS_PROMPT | judge | StrOutputParser()
supported_chain = SUPPORTED_PROMPT | judge | StrOutputParser()
addresses_chain = ADDRESSES_PROMPT | judge | StrOutputParser()
chunk_chain = CHUNK_RELEVANT_PROMPT | judge | StrOutputParser()


def _yes(text: str) -> bool:
    """Binary verdict. Checks the first token so 'NO, because...' isn't read as YES."""
    first = text.strip().upper().lstrip("*- ").split()
    return bool(first) and first[0].startswith("YES")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's factual claims that the context supports.

    This is the hallucination metric. An answer with no claims (a refusal)
    scores 1.0 — there is nothing unsupported in it.
    """
    raw = claims_chain.invoke({"answer": answer})
    claims = [
        c.strip("*- ").strip()
        for c in raw.split("\n")
        if c.strip() and "NONE" not in c.upper()
    ]
    if not claims:
        if VERBOSE:
            print("      claims: none -> 1.00 (nothing to hallucinate)")
        return 1.0

    hits = sum(
        _yes(supported_chain.invoke({"context": context, "claim": c})) for c in claims
    )
    if VERBOSE:
        print(f"      claims: {hits}/{len(claims)} supported")
    return hits / len(claims)


def relevancy(question: str, answer: str) -> float:
    """Binary: does the answer address the question at all."""
    verdict = _yes(addresses_chain.invoke({"question": question, "answer": answer}))
    if VERBOSE:
        print(f"      addresses question: {'YES' if verdict else 'NO'}")
    return 1.0 if verdict else 0.0


def context_precision(question: str, docs) -> float:
    """Fraction of retrieved chunks that are actually relevant.

    Scored per chunk, not as one 'how many are relevant' question — the model
    is far better at one binary judgment than at estimating a proportion.
    """
    if not docs:
        return 0.0
    hits = sum(
        _yes(chunk_chain.invoke({"question": question, "chunk": d.page_content}))
        for d in docs
    )
    if VERBOSE:
        print(f"      chunks relevant: {hits}/{len(docs)}")
    return hits / len(docs)


GOLDEN = [
    "How much does the service cost?",
    "When is payment due?",
    "Which state's law governs the agreement?",
    "How quickly must a data breach be reported?",
    "What is the maximum liability?",
    "What is the employee vacation policy?",   # not in the doc — should refuse
]


if __name__ == "__main__":
    totals = {"faith": 0.0, "rel": 0.0, "prec": 0.0}

    for q in GOLDEN:
        answer, docs = answer_question(q)
        context = "\n\n".join(d.page_content for d in docs)

        print(f"\n{'=' * 62}")
        print(f"Q: {q}")
        print(f"A: {answer[:90]}...")

        faith = faithfulness(answer, context)
        rel = relevancy(q, answer)
        prec = context_precision(q, docs)

        totals["faith"] += faith
        totals["rel"] += rel
        totals["prec"] += prec

        print(f"   faithfulness {faith:.2f} | relevancy {rel:.2f} | precision {prec:.2f}")

    n = len(GOLDEN)
    print(f"\n{'=' * 62}")
    print("AVERAGES")
    print(f"  faithfulness:      {totals['faith']/n:.2f}   (claims supported by context)")
    print(f"  answer relevancy:  {totals['rel']/n:.2f}   (answer addresses question)")
    print(f"  context precision: {totals['prec']/n:.2f}   (retrieved chunks relevant)")
    print("\nJudge: llama3.1:8b, separate from the generation model (llama3.2).")
    print("Faithfulness and relevancy are reliable. Context precision is still noisy —")
    print("one question scored 0/5 relevant while correctly citing a retrieved chunk.")
    print("Treat precision as directional, not exact.")