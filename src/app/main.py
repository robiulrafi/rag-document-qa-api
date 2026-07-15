"""FastAPI application exposing a local LLM over HTTP.

Endpoints
---------
GET  /health           liveness + which model is loaded
POST /ask              single-shot answer (from the model's own knowledge)
POST /stream           token-by-token answer over Server-Sent Events
POST /ask/structured   validated JSON answer (summary + confidence)
POST /query            RAG: grounded answer from the ingested document

Run locally:
    uvicorn src.app.main:app --reload
"""

import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chains import Answer, build_qa_chain, build_structured_chain
from .config import settings
from .rag_query import answer_question

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Chains are built once at import time — they are stateless and reusable.
qa_chain = build_qa_chain()
structured_chain = build_structured_chain()


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question")


class AskResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    model: str


class QueryRequest(BaseModel):
    """A question to answer using only the ingested document."""

    question: str = Field(min_length=1, description="Question about the document")


class Source(BaseModel):
    """One retrieved chunk, returned so the answer is traceable."""

    page: int | str
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Deployment platforms poll this."""
    return HealthResponse(status="ok", model=settings.OLLAMA_MODEL)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer from the model's own knowledge — no retrieval, no grounding."""
    answer = await qa_chain.ainvoke({"question": req.question})
    return AskResponse(answer=answer)


@app.post("/stream")
async def stream(req: AskRequest) -> StreamingResponse:
    """Stream tokens as they are generated, using Server-Sent Events."""

    async def token_generator():
        async for chunk in qa_chain.astream({"question": req.question}):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/ask/structured", response_model=Answer)
async def ask_structured(req: AskRequest) -> Answer:
    """Return a schema-validated answer (summary + confidence score)."""
    return await structured_chain.ainvoke({"question": req.question})


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """RAG endpoint: retrieve relevant chunks, then answer *only* from them.

    Unlike /ask, this is grounded — if the answer is not in the document, the
    model is instructed to say so rather than guess.
    """
    start = time.perf_counter()
    answer, docs = answer_question(req.question)
    latency_ms = round((time.perf_counter() - start) * 1000)

    # Structured log: one JSON line per query, ready for evaluation later.
    print(
        json.dumps(
            {
                "event": "query",
                "question": req.question,
                "latency_ms": latency_ms,
                "chunks_retrieved": len(docs),
                "pages": [d.metadata.get("page") for d in docs],
            }
        )
    )

    return QueryResponse(
        answer=answer,
        sources=[
            Source(
                page=d.metadata.get("page", "?"),
                excerpt=d.page_content[:150],
            )
            for d in docs
        ],
    )