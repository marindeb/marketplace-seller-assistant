"""
Tests: rag/chains.py (policy/recommendation RAG orchestration)
"""

from unittest.mock import MagicMock, patch

from app.rag import chains


@patch("app.rag.chains.get_retriever")
@patch("app.rag.chains.Ollama")
def test_run_policy_rag(mock_llm, mock_retriever):
    mock_retriever.return_value = MagicMock()
    mock_llm.return_value = lambda prompt: "Mock answer with [1]"

    with patch("app.rag.chains.RetrievalQA.from_chain_type") as mock_chain:
        mock_chain.return_value.__call__.return_value = {
            "result": "Mock answer",
            "source_documents": [
                {"metadata": {"source": "doc1"}, "page_content": "txt"}
            ],
        }

        out = chains.run_policy_rag("Are knives allowed?")
        assert out["answer"] == "Mock answer"
        assert len(out["citations"]) == 1


@patch("app.rag.chains.get_retriever")
@patch("app.rag.chains.Ollama")
def test_run_recommendation_rag(mock_llm, mock_retriever):
    mock_retriever.return_value = MagicMock()
    mock_llm.return_value = lambda prompt: "Reco answer"

    with patch("app.rag.chains.RetrievalQA.from_chain_type") as mock_chain:
        mock_chain.return_value.__call__.return_value = {
            "result": "Reco answer",
            "source_documents": [
                {"metadata": {"source": "docX"}, "page_content": "abc"}
            ],
        }

        out = chains.run_recommendation_rag("How to improve listing?")
        assert out["answer"] == "Reco answer"
        assert "docX" in out["citations"][0]
