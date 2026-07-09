# RAG Document Q&A API

A question-answering API built on a **locally-hosted LLM** — no API keys, no per-token cost, no data leaving the machine.

The service currently exposes a LangChain (LCEL) chain over a local `llama3.2` model through FastAPI, with token streaming and schema-validated structured output. Retrieval is the next milestone: PDF ingestion, embeddings, a vector store, and a grounded `/query` endpoint that answers only from source documents.

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
| PDF ingestion → chunking → embeddings → ChromaDB | 🔜 Next |
| Grounded RAG `/query` endpoint | 🔜 Next |
| RAGAS evaluation harness | 🔜 Planned |
| Docker + CI/CD + live deployment | 🔜 Planned |

---

## Architecture

**Today**

```
                  ┌────────────────────────┐
   HTTP request   │      FastAPI app       │
  ──────────────► │  /ask  /stream  /health│
                  └───────────┬────────────┘
                              │
                   LCEL chain │  prompt | llm | parser
                              ▼
                  ┌────────────────────────┐
                  │   Ollama (localhost)   │
                  │      llama3.2          │
                  └────────────────────────┘
```

**Next milestone** — retrieval slots in ahead of the model, so answers are grounded in source documents rather than the model's parametric memory:

```
   PDF ──► chunk ──► embed ──► ChromaDB
                                   │
   question ──► embed ──► similarity search
                                   │
                          retrieved context
                                   ▼
                     prompt | llm | parser  ──► grounded answer
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe; reports the loaded model |
| `POST` | `/ask` | Returns the complete answer |
| `POST` | `/stream` | Streams tokens as they generate (Server-Sent Events) |
| `POST` | `/ask/structured` | Returns schema-validated JSON (`summary`, `confidence`) |

Interactive docs are served at `/docs` when the app is running.

---

## Quickstart

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
# 1. Pull a model (2 GB, runs comfortably on 8 GB RAM)
ollama pull llama3.2

# 2. Clone and enter the project
git clone https://github.com/robiulrafi/rag-document-qa-api.git
cd rag-document-qa-api

# 3. Create an isolated environment
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run — no configuration needed
uvicorn src.app.main:app --reload
```

Open <http://localhost:8000/docs>.

There is no configuration step and no API key to obtain. Inference runs against your local Ollama instance, and every setting in `src/app/config.py` has a working default. To override one (for example, to try a larger model), set an environment variable:

```bash
OLLAMA_MODEL=qwen3:8b uvicorn src.app.main:app --reload
```

---

## Usage

```bash
# Single-shot answer
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'

# Streaming answer — tokens arrive as they are produced
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain embeddings simply"}' --no-buffer

# Schema-validated answer
curl -X POST http://localhost:8000/ask/structured \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
# → {"summary": "...", "confidence": 0.9}
```

---

## Tests

The suite mocks the LLM, so it runs offline and in CI without an Ollama process.

```bash
pytest -v
```

---

## Design notes

**Why streaming.** A single-shot response makes the user wait for the whole generation. Streaming over SSE emits each token as it is produced, so time-to-first-token drops from seconds to milliseconds. Perceived latency, not total latency, is what users judge.

**Why local inference.** Running `llama3.2` through Ollama removes per-token cost and keeps documents on the machine — which matters for any domain where sending text to a third-party API is not an option. The trade-off is throughput and peak quality versus a frontier model.

**Why `with_structured_output` over an output parser.** Enforcing the schema at the model layer is more reliable than generating free text and repairing it afterwards with a fixing parser. Fewer moving parts, fewer failure modes.

**Why a chain factory.** `build_qa_chain()` constructs the chain rather than exposing a module-level singleton, which keeps the LLM swappable in tests and makes the dependency explicit.

**Why there is no `.env` file.** Local inference means there are no API keys, so there is nothing to keep out of version control. Settings are read from environment variables with working defaults instead — the app runs with zero configuration on a developer machine, and a hosting platform can still override `PORT` at runtime by injecting it directly. A `.env` file would add a setup step and a dependency in exchange for nothing.

---

## Stack

`Python` · `FastAPI` · `LangChain (LCEL)` · `Ollama` · `Pydantic` · `pytest`

---

## Roadmap

1. PDF ingestion pipeline — load, chunk, embed, persist to ChromaDB
2. Grounded `/query` endpoint that answers only from retrieved context
3. RAGAS evaluation — faithfulness and answer relevancy tracked over time
4. Containerization, CI/CD, and a live deployment

---

## License

MIT
