from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_integration_rag_mode():
    """Full integration test: API -> RAG pipeline."""
    payload = {"question": "Which products have high return rates?", "mode": "rag"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "mode" in data and data["mode"] == "rag"
    assert "question" in data
    assert isinstance(data["answer"], dict)


def test_integration_agent_mode():
    """Full integration test: API -> Pandas agent."""
    payload = {
        "question": "What is the average return rate by category?",
        "mode": "agent",
    }
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "mode" in data and data["mode"] == "agent"
    assert "question" in data
    assert isinstance(data["answer"], str)
