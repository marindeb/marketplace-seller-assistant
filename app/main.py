"""
Module: main.py
----------------
Exposes FastAPI interface for querying RAG and agent pipelines.
Handles request validation and response normalization.
"""

from typing import Any, Dict

import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import agent, config, rag_pipeline


config.setup_logging()

logger = logging.getLogger(__name__)
logger.info("Application started")

app = FastAPI()


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint for the FastAPI application.

    This route allows monitoring tools or users to verify that the API
    is running and responsive.

    Returns:
        dict: JSON object indicating the service status.
        Example: {"status": "ok"}
    """
    return {"status": "ok"}


class QueryRequest(BaseModel):
    """
    Request schema for the `/query` endpoint.

    Attributes:
        question (str): The user’s natural language question.
        mode (Optional[str]): The processing mode ("rag" or "agent").
            If not provided, it is inferred automatically from the question content.
    """

    question: str
    mode: str | None = None  # optional now


@app.post("/query")
def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
    """
    Handle analytical or knowledge-based user queries.

    The endpoint automatically selects between two modes:
    - **RAG mode** for retrieval-augmented generation (context-based responses).
    - **Agent mode** for analytical reasoning over structured data.

    If no mode is specified, the function infers it based on analytical keywords.

    Args:
        request (QueryRequest): Request payload containing the user's question
            and optionally the processing mode.

    Returns:
        dict: JSON response with the selected mode, original question,
        and generated answer. Example:
        {
            "mode": "rag",
            "question": "Which products have the highest return rate?",
            "answer": {...}
        }

    Raises:
        HTTPException: If an invalid mode is provided or a runtime error occurs.
    """
    question = request.question.strip()
    mode = request.mode

    # --- Automatic mode selection ---
    if not mode:
        analytical_keywords = [
            "average",
            "rate",
            "compare",
            "trend",
            "correlation",
            "return",
            "sales",
        ]
        mode = (
            "agent"
            if any(k in question.lower() for k in analytical_keywords)
            else "rag"
        )

    IS_TEST = os.getenv("APP_ENV") == "test"
    try:
        if mode == "rag":
            answer = rag_pipeline.query(question)
        elif mode == "agent":
            answer = agent.ask_agent(question)
        else:
            if not IS_TEST:
                raise HTTPException(status_code=400, detail="Invalid mode")
            else:
                raise ValueError(
                    "Invalid mode"
                )  # simple exception for unit tests
        return {"mode": mode, "question": question, "answer": answer}

    except Exception as e:
        if not IS_TEST:
            logger.info(e)
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise e
