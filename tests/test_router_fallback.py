"""
Tests: Router fallback logic

- Unexpected label from LLM → refusal
- Empty question → refusal (via intent logic)
"""

from unittest.mock import patch

from app.agents.router import classify_intent, route


@patch("app.agents.router.Ollama")
def test_classify_unexpected_label_fallback(mock_llm):
    mock_llm.return_value = lambda prompt: "???"
    assert classify_intent("weird question") == "refusal"


@patch("app.agents.router.classify_intent", return_value="refusal")
def test_route_refusal(mock_intent):
    out = route("any")
    assert out["intent"] == "refusal"
    assert "sorry" in out["answer"].lower()
