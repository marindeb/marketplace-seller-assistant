"""
Tests: Intent classification logic (router.classify_intent)

We mock the Ollama LLM so tests do not depend on local models.
"""

from unittest.mock import patch

from app.agents.router import classify_intent


@patch("app.agents.router.Ollama")
def test_classify_policy(mock_llm):
    mock_llm.return_value = lambda prompt: "policy"
    assert classify_intent("Is selling knives allowed?") == "policy"


@patch("app.agents.router.Ollama")
def test_classify_recommendation(mock_llm):
    mock_llm.return_value = lambda prompt: "recommendation"
    assert (
        classify_intent("How can I improve my conversion?") == "recommendation"
    )


@patch("app.agents.router.Ollama")
def test_classify_analytics(mock_llm):
    mock_llm.return_value = lambda prompt: "analytics"
    assert (
        classify_intent("Which product has the highest return rate?")
        == "analytics"
    )


@patch("app.agents.router.Ollama")
def test_classify_refusal(mock_llm):
    mock_llm.return_value = lambda prompt: "refusal"
    assert classify_intent("What is the GDP of France?") == "refusal"
