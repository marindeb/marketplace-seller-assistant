"""
Module: router.py
-----------------
Intent classifier and routing logic for Marketplace X agents.

Responsible for:
- Classifying a user question into one of:
  {policy, recommendation, analytics, refusal}
- Dispatching the question to the correct agent
- Returning a unified response schema
"""

from __future__ import annotations

from typing import Any, Dict

import logging
from langchain_community.llms import Ollama

from app import constants
from app.agents.analytics_agent import AnalyticsAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.reco_agent import RecommendationAgent
from app.agents.refusal_agent import RefusalAgent


logger = logging.getLogger(__name__)


def classify_intent(question: str) -> str:
    """Classify a question into a routing intent.

    Args:
        question: User question.

    Returns:
        One of: "policy", "recommendation", "analytics", "refusal".
    """
    llm = Ollama(model=constants.LLM_MODEL_ROUTER, temperature=0.0)

    prompt = (
        "Classify the following user query into exactly one category:\n"
        "- policy: rules, compliance, prohibited items, penalties.\n"
        "- recommendation: growth, conversion, SEO, listing improvements.\n"
        "- analytics: metrics, performance, trends, comparisons.\n"
        "- refusal: requests outside Marketplace X scope.\n\n"
        "Respond with one word: policy, recommendation, analytics, or refusal.\n\n"
        f"Query: {question}"
    )

    result = llm(prompt).strip().lower()

    if result not in {"policy", "recommendation", "analytics", "refusal"}:
        logger.warning(f"Router LLM returned unexpected label: {result}")
        return "refusal"

    return result


# Agent registry --------------------------------------------------------------

AGENTS = {
    "policy": PolicyAgent(),
    "recommendation": RecommendationAgent(),
    "analytics": AnalyticsAgent(),
    "refusal": RefusalAgent(),
}


# Router ----------------------------------------------------------------------


def route(question: str, seller_id: str | None = None) -> Dict[str, Any]:
    """Route a question to the appropriate Marketplace X agent.

    Args:
        question:
            The user question in natural language.
        seller_id:
            Optional seller identifier for analytics/personalization.

    Returns:
        Unified agent response dict:
        {
            "intent": str,
            "answer": str,
            "citations": list[str],
            "confidence": float,
            "sources": list[dict]
        }
    """
    intent = classify_intent(question)
    agent = AGENTS.get(intent, RefusalAgent())

    logger.info(f"Routing intent={intent} for question={question}")

    return agent.run(question=question, seller_id=seller_id)
