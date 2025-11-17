"""
Tests: Router dispatch to correct agent.run()

We mock:
- classify_intent() to force routing
- each Agent.run() to verify correct dispatch
"""

from unittest.mock import patch

from app.agents.router import route


@patch("app.agents.router.PolicyAgent.run")
@patch("app.agents.router.classify_intent", return_value="policy")
def test_router_policy(mock_intent, mock_run):
    mock_run.return_value = {"intent": "policy", "answer": "ok"}
    out = route("dummy question")
    assert out["intent"] == "policy"
    mock_run.assert_called_once()


@patch("app.agents.router.AnalyticsAgent.run")
@patch("app.agents.router.classify_intent", return_value="analytics")
def test_router_analytics(mock_intent, mock_run):
    mock_run.return_value = {"intent": "analytics", "answer": "ok"}
    out = route("dummy question")
    assert out["intent"] == "analytics"
    mock_run.assert_called_once()


@patch("app.agents.router.RecommendationAgent.run")
@patch("app.agents.router.classify_intent", return_value="recommendation")
def test_router_recommendation(mock_intent, mock_run):
    mock_run.return_value = {"intent": "recommendation", "answer": "ok"}
    out = route("dummy question")
    assert out["intent"] == "recommendation"
    mock_run.assert_called_once()
