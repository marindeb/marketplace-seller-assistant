"""
Module: chains.py
-----------------
Define RAG chains for Marketplace X:
- Policy/Compliance chain (strict)
- Recommendation chain (reasoning allowed but grounded)
- Refusal logic if retrieval confidence is low
"""

from __future__ import annotations

from typing import Any, Dict, List

import logging
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.llms import Ollama

from app import constants
from app.rag.index import get_doc_retriever


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _format_citation(doc: Document) -> str:
    """Format unified citation style `[doc_id > section > chunk_id]`.

    Args:
        doc: Retrieved LangChain Document.

    Returns:
        A string citation.
    """
    meta = doc.metadata
    return f"[{meta['doc_id']} > {meta['section']} > {meta['chunk_id']}]"


def _extract_citations(docs: List[Document]) -> List[str]:
    """Extract formatted citations from retrieved documents."""
    return [_format_citation(d) for d in docs]


def _retrieval_confidence(docs: List[Document]) -> float:
    """Compute a simple confidence score for retrieved docs.

    Heuristic:
    - If no docs: 0.0
    - If all docs have very short text (<40 chars): suspicious
    - Later we can extend with cosine similarity checks.

    Args:
        docs: Retrieved chunks.

    Returns:
        Confidence score in [0, 1].
    """
    if not docs:
        return 0.0

    avg_len = sum(len(d.page_content) for d in docs) / len(docs)

    if avg_len < 40:
        return 0.2

    return 1.0


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


def get_policy_chain() -> RetrievalQA:
    """Create a strict policy/compliance chain using RetrievalQA.

    Behavior:
    - Must cite at least one retrieved document
    - Must refuse if retrieval confidence is low
    - No speculation allowed

    Returns:
        A configured RetrievalQA chain.
    """
    retriever = get_doc_retriever(k=4)
    llm = Ollama(
        model=constants.LLM_MODEL_RAG,
        temperature=constants.TEMPERATURE_RAG,
    )

    prompt_template = (
        "You are the Marketplace X Policy Assistant.\n"
        "Your answer MUST be strictly based on the retrieved documentation.\n"
        "If the documentation does not support the answer, you MUST refuse with:\n"
        '"I’m sorry, but I don’t have enough information to answer this question based on Marketplace X documentation.".\n'
        "Always include citations using the format: [doc_id > section > chunk_id].\n"
        "Do not invent or speculate.\n"
        "\n"
        "Question:\n{question}\n\n"
        "Relevant documentation:\n{context}\n\n"
        "Answer:"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return chain


def get_recommendation_chain() -> RetrievalQA:
    """Create a recommendation RAG chain with reasoning allowed.

    Behavior:
    - Can synthesize recommendations
    - But claims MUST be grounded and cited
    - Must refuse if retrieval confidence is too low

    Returns:
        A configured RetrievalQA chain.
    """
    retriever = get_doc_retriever(k=4)
    llm = Ollama(
        model=constants.LLM_MODEL_RAG,
        temperature=constants.TEMPERATURE_RAG,
    )

    prompt_template = (
        "You are the Marketplace X Growth & Listing Assistant.\n"
        "You can provide recommendations and reasoning, but all factual claims must be grounded "
        "in the retrieved documentation. Always include citations.\n"
        "If not enough context is available, refuse politely.\n\n"
        "Question:\n{question}\n\n"
        "Relevant documentation:\n{context}\n\n"
        "Answer:"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return chain


# ---------------------------------------------------------------------------
# RAG Pipeline Wrappers
# ---------------------------------------------------------------------------


def run_policy_rag(question: str) -> Dict[str, Any]:
    """Run the full policy RAG pipeline and apply refusal logic.

    Args:
        question: User question.

    Returns:
        A dictionary with keys:
        - "answer"
        - "citations"
        - "sources"
        - "confidence"
    """
    chain = get_policy_chain()
    raw = chain({"query": question})

    docs = raw.get("source_documents", [])
    confidence = _retrieval_confidence(docs)

    if confidence < 0.5:
        return {
            "answer": (
                "I’m sorry, but I don’t have enough information to answer "
                "this question based on Marketplace X documentation."
            ),
            "citations": [],
            "sources": [],
            "confidence": confidence,
        }

    citations = _extract_citations(docs)

    return {
        "answer": raw["result"],
        "citations": citations,
        "sources": [d.metadata for d in docs],
        "confidence": confidence,
    }


def run_recommendation_rag(question: str) -> Dict[str, Any]:
    """Run the recommendation RAG pipeline.

    Args:
        question: User question.

    Returns:
        A dictionary with keys:
        - "answer"
        - "citations"
        - "sources"
        - "confidence"
    """
    chain = get_recommendation_chain()
    raw = chain({"query": question})

    docs = raw.get("source_documents", [])
    confidence = _retrieval_confidence(docs)

    if confidence < 0.5:
        return {
            "answer": (
                "I’m sorry, but I don’t have enough information to provide "
                "a grounded recommendation based on Marketplace X documentation."
            ),
            "citations": [],
            "sources": [],
            "confidence": confidence,
        }

    citations = _extract_citations(docs)

    return {
        "answer": raw["result"],
        "citations": citations,
        "sources": [d.metadata for d in docs],
        "confidence": confidence,
    }
