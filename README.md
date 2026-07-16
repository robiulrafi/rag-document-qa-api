# RAG Document Q&A API

A retrieval-augmented question-answering API built on a **locally-hosted LLM** — no API keys, no per-token cost, no data leaving the machine.

Point it at a PDF and ask questions. It retrieves the relevant passages, answers **only** from them, cites the source of every claim, and says *"I don't know"* when the answer isn't in the document. That grounding is the point: a system that confidently invents answers is worse than useless for contracts, policies, or case law.

It mirrors the architecture of a production retrieval-augmented generation system I built at enterprise scale, rebuilt here on open models and public data so the pipeline, prompt orchestration, and evaluation approach can be inspected end to end.

> **Note:** This repository uses public datasets and open-source models only. It contains no proprietary data, internal systems, or confidential information from any employer.

---

## Status

| Milestone | State |
|---|---|
| LangChain (LCEL) chain on local Ollama | ✅ Done |
| FastAPI service — `/ask`, `/stream`, `/health` | ✅ Done |
| Structured output with schema validation | ✅ Done |
| Test suite (mocked LLM, runs offline) | ✅ Done |
| PDF ingestion → chunking → embeddings → ChromaDB | ✅ Done |
| Grounded RAG `/query` endpoint with refusal | ✅ Done |
| Numbered source citations | ✅ Done |
| History-aware retrieval (multi-turn follow-ups) | ✅ Done |
| Hybrid retrieval (BM25 + vector) | ✅ Done |
| RAGAS evaluation harness | 🔜 Next |
| Reranking + semantic chunking | 🔜 Planned |
| Docker + CI/CD + live deployment | 🔜 Planned |

---

## Architecture

**Ingestion** — runs once, offline, when documents change:

```
   document.pdf
        │
        ▼
   PyPDFLoader ──► RecursiveCharacterTextSplitter ──► chunks
                                                        │
                                                        ▼
                                        nomic-embed-text  (768-dim vectors)
                                                        │
                                                        ▼
                                        ChromaDB  (persisted to disk)
```

**Query** — runs per request:

```
   question (+ conversation history)
        │
        ▼
   rewrite follow-up into a standalone query
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
   embed query    BM25 keyword        │   1. RETRIEVE
        │              │              │
        ▼              ▼              │
   ChromaDB        BM25 index         │
   (cosine sim)    (exact terms)      │
        │              │              │
        └──────┬───────┘              │
               ▼                      │
        merge + dedupe ───────────────┘
               │
               ▼
   numbered context: "[1] (page 0) ..."      2. AUGMENT
               │
               ▼
   grounding prompt: "answer ONLY from context,
   cite [n], say I don't know if absent"
               │
               ▼
   llama3.2 (Ollama, local)                  3. GENERATE
               │
               ▼
   grounded answer + numbered citations
```

The LLM never fetches anything. Retrieval happens in application code; the model receives the chunks as plain text in its prompt. That separation is what **R**etrieve → **A**ugment → **G**enerate actually means.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe; reports the loaded model |
| `POST` | `/query` | **RAG** — grounded answer from the ingested document, with citations |
| `POST` | `/ask` | Ungrounded answer from the model's own knowledge |
| `POST` | `/stream` | Streams tokens as they generate (Server-Sent Events) |
| `POST` | `/ask/structured` | Schema-validated JSON (`summary`, `confidence`) |

`/ask` and `/query` answer the same question two ways — one from the model's parametric memory, one grounded in your documents. Useful for demonstrating what grounding actually buys you.

Interactive docs at `/docs` when running.

---

## Quickstart

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
# 1. Pull the models (~2.3 GB total, runs on 8 GB RAM)
ollama pull llama3.2            # generation
ollama pull nomic-embed-text    # embeddings

# 2. Clone and enter
git clone https://github.com/robiulrafi/rag-document-qa-api.git
cd rag-document-qa-api

# 3. Isolated environment
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # macOS / Linux

# 4. Install
pip install -r requirements.txt

# 5. Ingest a document — run ONCE per document
python -m src.app.ingest

# 6. Run
uvicorn src.app.main:app --reload
```

Open <http://localhost:8000/docs>.

No configuration, no API key. Inference runs against your local Ollama instance, and every setting in `src/app/config.py` has a working default. Override one with an environment variable:

```bash
OLLAMA_MODEL=qwen3:8b uvicorn src.app.main:app --reload
```

---

## Usage

```bash
# Grounded question — answered from the document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How much does the service cost?"}'
```

```json
{
  "answer": "The monthly service fee is $25,000 [3].",
  "sources": [
    {"id": 1, "page": 0, "excerpt": "1. Definitions..."},
    {"id": 2, "page": 0, "excerpt": "payments accrue interest at 1.5%..."},
    {"id": 3, "page": 0, "excerpt": "3. Fees and Payment. Client shall pay..."}
  ]
}
```

```bash
# Question the document does NOT answer — it refuses rather than inventing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the employee vacation policy?"}'
# → "I don't know — that isn't covered in the document."
```

```bash
# Follow-up question — history lets it resolve "it"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "question": "When is it due?",
        "history": ["Q: How much does the service cost?",
                    "A: The monthly fee is $25,000."]
      }'
# → "The monthly fee of $25,000 is due within thirty (30) days of invoice [1]."
```

---

## Tests

The suite mocks the LLM, so it runs offline and in CI without an Ollama process.

```bash
pytest -v
```

---

## Design notes

**Why grounding is the whole product.** An LLM predicts likely tokens; it has no notion of truth. Asked about something it doesn't know, it produces fluent, plausible, wrong answers. The system prompt instructs the model to answer only from retrieved context and to say *"I don't know"* otherwise. Tested directly: asked for a vacation policy in a services contract that has none, retrieval still returned three chunks — vector search always returns nearest neighbours — and the model correctly refused. That refusal is the behaviour that makes the system trustworthy.

**Why citations are numbered, not just listed.** Chunks are injected as `[1] (page 0) ...` and the model cites `[3]` inline. Each claim maps to a specific chunk ID in the response, so a UI can link every statement back to its source text. "Here are some documents I looked at" is not traceability; "this claim came from this passage" is.

**Why hybrid retrieval.** Pure vector search matches meaning — asking about *cost* correctly finds a clause that says *fee*. But it's weak on exact tokens: party names, case numbers, dollar figures. BM25 keyword matching catches those. Running both and merging gets each one's strength. Verified with the contrast: *"How much does the service cost?"* is won by the vector side, *"Northwind Trading"* by BM25.

**Why the follow-up rewrite.** *"When is it due?"* has no standalone meaning — its embedding retrieves noise. Before retrieving, the query is rewritten against conversation history into *"When is the $25,000 monthly fee due?"*. Retrieval uses the rewritten query; the model still answers the original. Costs one extra LLM call, skipped when there's no history.

**Why ingestion is a separate module.** `Chroma.from_documents()` **appends** — it is not idempotent. Running ingestion inside the query path silently duplicated every chunk on each call, polluting retrieval until the store held ~10 copies and returned the same chunk three times. `ingest.py` writes; `rag_query.py` only reads. Handling document *updates* properly needs deterministic chunk IDs (hash of source + page + index) so re-ingestion upserts rather than appends.

**Why `temperature=0`.** For grounded Q&A the goal is faithful, reproducible answers, not creativity.

**Why streaming.** A single-shot response makes the user wait for the whole generation — measured at ~18s locally for a cold model. Streaming over SSE emits each token as produced, so time-to-first-token drops to milliseconds. Perceived latency, not total latency, is what users judge.

**Why local inference.** Running `llama3.2` through Ollama removes per-token cost and keeps documents on the machine — which matters in any domain where sending text to a third-party API isn't an option. The trade-off is throughput and peak quality versus a frontier model.

**Known limitation: chunk boundaries.** Fixed-size character splitting cuts across section boundaries — one retrieved chunk contains the tail of the termination clause glued to the start of the fees clause. Retrieval quality is the ceiling on answer quality, so the next improvements are upstream: structure-aware chunking (split on section headings), then reranking a broad candidate set down to the few genuinely relevant chunks.

---

## Stack

`Python` · `FastAPI` · `LangChain (LCEL)` · `Ollama` · `ChromaDB` · `nomic-embed-text` · `BM25` · `Pydantic` · `pytest`

---

## Roadmap

1. RAGAS evaluation — faithfulness and answer relevancy tracked as regression tests
2. Structure-aware chunking and cross-encoder reranking
3. Metadata filtering for access control (retrieve only what a user may see)
4. Containerization, CI/CD, and a live deployment

---

## License

MIT
