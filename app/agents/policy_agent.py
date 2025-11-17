"""
Module: policy_agent.py
-----------------------
Wrapper around the strict policy RAG chain.
"""

from __future__ import annotations

from typing import Any, Dict

from app.rag import chains


class PolicyAgent:
    """Strict policy/compliance agent."""

    def run(
        self, question: str, seller_id: str | None = None
    ) -> Dict[str, Any]:
        """Execute the policy RAG pipeline.

        Args:
            question: User question.
            seller_id: Unused for policy questions.

        Returns:
            RAG result as a dict.
        """
        return {
            "intent": "policy",
            **chains.run_policy_rag(question),
        }
