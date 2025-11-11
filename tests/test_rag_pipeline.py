import shutil

import os
import pytest

from app import constants, rag_pipeline


@pytest.fixture(scope="function")
def clean_index(tmp_path):  # noqa: ARG001
    """
    Ensure a clean chroma index directory before each test.
    """
    index_path = "data/chroma_index"
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
    yield index_path
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


def test_build_vectorstore_creates_index(monkeypatch, tmp_path):
    """
    Test that running the RAG pipeline creates the Chroma index directory.
    """
    temp_dir = tmp_path / "chroma_index"
    monkeypatch.setattr(constants, "CHROMA_DIR", str(temp_dir))
    rag_pipeline.build_vectorstore(force_rebuild=True)
    assert temp_dir.exists(), "Chroma index directory was not created"
    assert len(os.listdir(temp_dir)) > 0, "Chroma index directory is empty"


def test_load_documents_returns_strings():
    """
    Ensure that `rag_pipeline.load_documents` returns a list of textual items.
    This test validates the preprocessing phase before embedding.
    """
    docs = rag_pipeline.load_documents()

    assert isinstance(docs, list), "Expected a list of documents."
    assert all(isinstance(d, str) for d in docs), "All documents should be strings."
    assert len(docs) > 0, "Documents list should not be empty."
