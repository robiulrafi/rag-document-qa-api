# RAG Document Q&A API

A production-oriented Retrieval-Augmented Generation system that answers questions
about a document with **grounded, cited answers** — and refuses when the answer
isn't there. Built to handle the failure modes that separate a demo from something
you'd trust: hallucination, missing citations, and unmeasured retrieval quality.

Runs fully locally (Ollama + ChromaDB) — no API keys, no data leaving the machine,
which matters for the confidential-document use cases this targets.

**Stack:** Python · FastAPI · LangChain (LCEL) · LangGraph · Ollama (llama3.2) ·
ChromaDB · nomic-embed-text · BM25 · cross-encoder reranking · pytest

---

## What it does

- **Ingests** a PDF, chunks it on section boundaries, embeds it, stores vectors in ChromaDB
- **Retrieves** with a two-stage funnel: hybrid (vector + BM25) recall, then cross-encoder reranking for precision
- **Generates** answers grounded strictly in retrieved context, with inline citations
- **Refuses** honestly ("I don't know — that isn't covered in the document") when the answer isn't present
- **Handles multi-turn** — rewrites follow-up questions into standalone queries before retrieval
- **Self-corrects** (optional agentic path) — grades retrieved context and rewrites the query to retry when it's too weak
- **Evaluates itself** with an LLM-as-judge harness measuring faithfulness, relevancy, and context precision

---

## Architecture

```
INGESTION (offline, writes the store)          QUERY (online, reads the store)
─────────────────────────────────────          ──────────────────────────────────────
PDF                                             question
 └─ structure-aware chunking                     └─ (rewrite follow-up if multi-turn)
     └─ embed (nomic-embed-text)                     └─ HYBRID retrieve (vector + BM25)  ← recall
         └─ ChromaDB                                     └─ cross-encoder RERANK (top 3) ← precision
                                                             └─ grounded generation + citations
                                                                 └─ answer  |  or honest refusal
```

Ingestion and query are deliberately separate modules. Ingestion *writes*; query
*reads*. Mixing them causes duplicate chunks, because `Chroma.from_documents()`
appends on every call.

---

## Retrieval quality — investigation and results

The interesting part of this project isn't that it works — it's the measured
investigation into *why* it initially didn't, and what fixed it.

### The investigation

Initial context precision was **0.21**, and an answerable question — "how much does
the service cost?" — failed to surface the fee at all. Rather than guess at a fix,
I traced it:

1. **Measured the failure** — the cost query never returned the fee chunk.
2. **Ruled out query phrasing** — rewording returned the same wrong chunks, so query rewriting wouldn't have helped.
3. **Ruled out retrieval method** — BM25 keyword search *also* missed it, so hybrid alone wouldn't have fixed it.
4. **Found the root cause** — the "$25,000 monthly fee" sentence was trapped in a chunk *dominated by termination text*. Character-based splitting (`chunk_size=250`) had cut across the Section 2 → Section 3 boundary, gluing the termination tail to the fees head. No retriever can rank a mixed-topic chunk well for a single-topic question.

### The fixes (three distinct problems, three distinct fixes)

**Structure-aware chunking** — split on section headers (`\n(?=\d+\.\s+[A-Z])`) so each
numbered clause is its own chunk. The fee clause became a clean, fee-only chunk that
ranks #1 for "how much does it cost." *This fixed the answers:* faithfulness rose to
**0.89**, and previously-broken questions now resolve correctly.

**Hybrid retrieval** — vector search unioned with BM25, so exact terms (party names,
dollar amounts, statute numbers) that dense embeddings represent weakly are still
caught. This improves *recall robustness* — it does **not** raise precision, because
it doesn't change how many chunks are retrieved (confirmed: precision stayed 0.18).

**Cross-encoder reranking** — a two-stage funnel: hybrid retrieves ~10 candidates
(recall), then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each (query, chunk)
pair jointly and keeps the top 3 (precision). Unlike embeddings, which encode query
and chunk separately, a cross-encoder reads them together — more accurate, but too
slow to run over the whole corpus, so it reranks a small candidate set. **This moved
context precision from 0.18 to 0.33.**

### An honest measurement note

An early "improved" precision of 0.33 turned out to be inflated: `Chroma.from_documents()`
appends, and re-ingesting without clearing the store left duplicate and stale chunks.
On a clean single-ingest store the baseline was **0.18** — the true starting point.
The production fix is deterministic chunk IDs so re-ingesting *upserts* instead of
appending.

### Results (6-question golden set, llama3.1:8b judge)

| Metric | Value | Notes |
|---|---|---|
| Faithfulness | 0.89 | claims supported by retrieved context |
| Answer relevancy | 1.00 | answer addresses the question |
| Context precision | 0.33 | after reranking (0.18 before); directional |

**Precision progression:** 0.18 (clean baseline, hybrid only) → **0.33** (with reranking).
The remaining ceiling is the corpus itself: this 10-section contract has ~1 answer
chunk per question, so retrieving 3 caps precision near 0.33 for single-answer
questions. Reranking's benefit scales with corpus size — filtering 3-of-10 here, but
3-of-hundreds in production, where the gain is far larger.

---

## Evaluation harness

`evaluate_rag.py` implements LLM-as-judge metrics directly (RAGAS pins to LangChain
internals that have since moved). Two design decisions came out of measurement:

**Never ask the judge for a score.** A first version asked for "a number between 0.0
and 1.0" and got 0.5 on 14 of 18 scores — including a refusal that should have been
1.0. Small models can't produce calibrated continuous scores. The fix: decompose the
answer into atomic claims and ask a *binary* YES/NO per claim, then compute the metric
arithmetically. That's what RAGAS does internally.

**The judge must be a different, stronger model than the one under test.** With the
generation model (llama3.2) as judge, relevancy scored 0.50 while answers were
correctly citing chunks — a self-contradiction. Swapping to llama3.1:8b fixed it.
A small model can do near-extractive judgments (does this context support this claim?)
but not abstract relevance judgments. *Validate the judge before trusting the metric —
an unvalidated harness will confidently tell you to fix things that aren't broken.*

---

## Agentic RAG (LangGraph) — optional self-correcting path

`langgraph_selfcorrect.py` implements a self-correcting retrieval loop as a LangGraph
state machine. It adds one capability a straight chain can't express: a **conditional
branch and a cycle**.

```
START → retrieve → grade ──good────→ generate → END
          ↑          │
          │          ├──retry───→ rewrite ──┐
          │          │                      │  (cycle)
          │          └──give_up─→ generate  │
          └──────────────────────────────────┘
```

- **grade** judges each retrieved chunk for relevance (llama3.1:8b — the "don't let a model grade its own homework" lesson from the eval harness)
- If context is too weak and retries remain, **rewrite** reformulates the query and loops back to retrieve
- A loop guard (`MAX_ATTEMPTS`) prevents infinite cycling — an unanswerable question retries twice, then gives up and produces an honest refusal rather than an error

**Design decisions worth noting:**
- *Grade before generate* — fail fast at the cheap step; don't spend an ~18s generation on noise
- *give_up routes to generate* — so an out-of-retries path still produces a grounded refusal, not a crash
- *Defensive output parsing* — the rewrite node strips model preamble (small models add "Here is a rewritten version:..."), so a chatty model can't poison the next retrieval

**Honest scope:** this is the *agentic capability*, not a metrics win. The investigation
above showed query rewriting does not fix this project's core failure (chunking did),
so the self-correcting loop's value is graceful handling of unanswerable questions and
the agentic architecture itself — not improved precision. The API uses the straight
path by default (lower latency); the self-correcting path is available for hard queries.

---

## Known limitations → next steps

- **Wrong citation index (occasional)** — the small model sometimes cites correct information under the wrong bracket number; needs a citation-correctness check.
- **Naive hybrid merge** — concatenation, not rank fusion. The proper fix is Reciprocal Rank Fusion (score each doc by sum of 1/(k+rank) across retrievers), which works on ranks rather than raw scores — cosine similarity and BM25 scores aren't on comparable scales.
- **Non-idempotent ingestion** — deterministic chunk IDs would make re-ingestion upsert instead of append.
- **Section regex** fits numbered clauses; production needs format-agnostic structure detection.
- **Scalability** (discussion): millions of docs → managed vector store; many users → the ~18s generation is the bottleneck, needs batching/GPU/streaming; access control → metadata filtering at retrieval; cost → a semantic cache (e.g. Redis) so repeated queries skip generation.

---

## Running it

```bash
# 1. install
pip install -r requirements.txt

# 2. pull models (Ollama must be running)
ollama pull llama3.2
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. ingest a document (clear the store first — ingestion appends)
rm -rf chroma_db
python ingest.py

# 4. query from the command line
python -m src.app.rag_query

# 5. or run the self-correcting graph
python -m src.app.langgraph_selfcorrect

# 6. run the evaluation harness
python evaluate_rag.py

# 7. serve the API
uvicorn src.app.main:app --reload

# 8. run tests
pytest
```

---

## Project layout

```
ingest.py                     structure-aware chunking + embedding (writes the store)
src/app/rag_query.py          hybrid retrieval + reranking + grounded generation (reads)
src/app/langgraph_selfcorrect.py   self-correcting agentic RAG graph
src/app/main.py               FastAPI endpoint with error handling + structured logging
evaluate_rag.py               LLM-as-judge evaluation harness
tests/                        pytest suite (LLM mocked)
```