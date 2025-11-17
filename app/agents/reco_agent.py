"""
Module: reco_agent.py
---------------------
Wrapper around the recommendation RAG chain.
"""

from __future__ import annotations

from typing import Any, Dict

from app.rag import chains


class RecommendationAgent:
    """Growth and listing optimization agent."""

    def run(
        self, question: str, seller_id: str | None = None
    ) -> Dict[str, Any]:
        """Execute the recommendation RAG pipeline.

        Args:
            question: Seller question.
            seller_id: Unused for now. Will support hybrid RAG + analytics later.

        Returns:
            RAG result as a dict.
        """
        return {
            "intent": "recommendation",
            **chains.run_recommendation_rag(question),
        }
