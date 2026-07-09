"""FastAPI application exposing a local LLM over HTTP.

Endpoints
---------
GET  /health      liveness + which model is loaded
POST /ask         single-shot answer
POST /stream      token-by-token answer over Server-Sent Events
POST /ask/structured   validated JSON answer (summary + confidence)

Run locally:
    uvicorn src.app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chains import Answer, build_qa_chain, build_structured_chain
from .config import settings

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


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Deployment platforms poll this."""
    return HealthResponse(status="ok", model=settings.OLLAMA_MODEL)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Return the full answer once the model has finished generating."""
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
