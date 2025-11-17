"""
Tests: rag/docs_index.py (index building + retriever wrapper)
"""

from unittest.mock import MagicMock, patch

from app.rag import docs_index


@patch("app.rag.docs_index.HuggingFaceEmbeddings")
@patch("app.rag.docs_index.Chroma")
def test_build_index(mock_chroma, mock_embed, tmp_path, monkeypatch):
    monkeypatch.setattr(docs_index, "CHROMA_DIR", str(tmp_path / "idx"))

    mock_chroma.from_documents.return_value = MagicMock()

    docs = [{"title": "T", "content": "C"}]
    idx = docs_index.build_index(docs, force_rebuild=True)

    mock_chroma.from_documents.assert_called_once()
    assert idx is not None


@patch("app.rag.docs_index.Chroma")
def test_get_retriever(mock_chroma, tmp_path, monkeypatch):
    mock_instance = MagicMock()
    mock_chroma.return_value = mock_instance

    monkeypatch.setattr(docs_index, "CHROMA_DIR", str(tmp_path))
    retriever = docs_index.get_retriever()

    assert retriever is not None
    mock_chroma.assert_called_once()
