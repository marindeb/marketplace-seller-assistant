"""
Unit tests for the FastAPI endpoints.

Covers:
1. Health check (`/health`)
2. Query endpoint (`/query`) in both RAG and Agent modes.

Mocks are used to avoid calling Ollama or Chroma during tests.
"""

import pytest

import app.main as main


# --- Setup ---
@pytest.fixture(scope="module")
def client():
    """
    Create a FastAPI test client for API endpoint testing.
    """
    from fastapi.testclient import TestClient

    return TestClient(main.app)


# --- Tests ---
def test_health_endpoint(client):
    """
    Ensure the health endpoint returns status 200 and correct message.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rag(monkeypatch, client):
    """
    Test `/query` endpoint in RAG mode.
    Ensures it returns the mocked RAG response and status 200.
    """

    import logging

    logger = logging.getLogger(__name__)

    def mock_query(question: str):
        logger.info("⚠️ MOCK RAG USED ⚠️")
        return {
            "result": "Mocked RAG answer",
            "query": "Mocked RAG query",
            "source_documents": ["Mocked RAG document"],
        }

    monkeypatch.setattr("app.rag_pipeline.query", mock_query)

    response = client.post(
        "/query",
        json={"question": "Which products have high return rates?", "mode": "rag"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rag"
    assert "result" in data["answer"]


def test_query_agent(monkeypatch, client):
    """
    Test `/query` endpoint in Agent mode.
    Ensures it returns the mocked Agent response and status 200.
    """

    def mock_ask_agent(question: str):  # noqa: ARG001
        return "Mocked Agent answer"

    monkeypatch.setattr("app.agent.ask_agent", mock_ask_agent)

    response = client.post(
        "/query",
        json={"question": "Which categories perform best?", "mode": "agent"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "agent"
    assert data["answer"] == "Mocked Agent answer"
