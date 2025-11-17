"""
Module: refusal_agent.py
------------------------
Deterministic refusal agent for unsupported queries.
"""

from __future__ import annotations

from typing import Any, Dict


class RefusalAgent:
    """Simple fallback agent that always refuses."""

    def run(
        self, question: str, seller_id: str | None = None
    ) -> Dict[str, Any]:
        """Return a polite refusal message.

        Args:
            question: User question.
            seller_id: Unused.

        Returns:
            A unified refusal response.
        """
        return {
            "intent": "refusal",
            "answer": (
                "I’m sorry, but I don’t have enough information to answer this "
                "question on Marketplace X."
            ),
            "citations": [],
            "confidence": 0.0,
            "sources": [],
        }
