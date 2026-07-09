"""API tests.

The LangChain chain is replaced with a mock, so these run fast, offline, and in
CI — no Ollama process required.

Note: the chain is a Pydantic model, which rejects new attributes being set on
it. So we patch the module-level `qa_chain` name to point at a mock, rather
than trying to patch a method onto the real chain object.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_ask_returns_answer():
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="RAG retrieves, then generates.")

    with patch("src.app.main.qa_chain", mock_chain):
        response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "RAG retrieves, then generates."
    mock_chain.ainvoke.assert_awaited_once_with({"question": "What is RAG?"})


def test_ask_rejects_empty_question():
    """Pydantic's min_length=1 rejects this before it reaches the LLM."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_missing_field():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_stream_emits_sse_events():
    async def fake_astream(_inputs):
        for token in ["Hello", " world"]:
            yield token

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("src.app.main.qa_chain", mock_chain):
        response = client.post("/stream", json={"question": "Hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: Hello" in response.text
    assert "data:  world" in response.text
    assert "data: [DONE]" in response.text
