"""
Module: index.py
----------------
Build and load the Chroma vector index for Marketplace X documentation.

This module:
- Converts chunks from loader.py into LangChain Documents
- Embeds them using HuggingFace sentence-transformers
- Persists them locally with ChromaDB
- Exposes a retriever for downstream RAG chains
"""

from __future__ import annotations

from typing import Dict, List

import logging
import os
from langchain.schema import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from app import constants


logger = logging.getLogger(__name__)


def _get_index_dir() -> str:
    """Return the directory where the Chroma index is stored."""
    return constants.CHROMA_DOCS_DIR


def _convert_chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Convert chunk dictionaries into LangChain Document objects.

    Args:
        chunks (List[Dict]): Output from loader.load_and_chunk_docs().

    Returns:
        List[Document]: Documents ready for vectorstore ingestion.
    """
    docs: List[Document] = []

    for chunk in chunks:
        metadata = {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "subsection": chunk.get("subsection"),
            "path": chunk["metadata"].get("path"),
            "start_line": chunk["metadata"].get("start_line"),
            "end_line": chunk["metadata"].get("end_line"),
        }

        docs.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return docs


def build_doc_index(chunks: List[Dict], force: bool = False) -> Chroma:
    """Build a Chroma index for Marketplace X documentation.

    If the index already exists and force=False, it is reused.

    Args:
        chunks (List[Dict]): Chunk data from loader.
        force (bool): Force rebuild even if the index exists.

    Returns:
        Chroma: A persistent Chroma vector store.
    """
    index_dir = _get_index_dir()

    if os.path.exists(index_dir) and not force:
        logger.info(f"Using existing documentation index at {index_dir}.")
        return Chroma(
            persist_directory=index_dir,
            embedding_function=HuggingFaceEmbeddings(
                model_name=constants.EMBEDDING_MODEL
            ),
        )

    logger.info("Building documentation index...")

    docs = _convert_chunks_to_documents(chunks)
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDING_MODEL)

    os.makedirs(index_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=index_dir,
    )

    vectorstore.persist()
    logger.info("Documentation index built and persisted.")

    return vectorstore


def load_doc_index() -> Chroma:
    """Load the existing Chroma documentation index.

    Returns:
        Chroma: Loaded vector store.

    Raises:
        FileNotFoundError: If the index does not exist.
    """
    index_dir = _get_index_dir()

    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"Documentation index not found at {index_dir}. "
            "Build it first with build_doc_index()."
        )

    return Chroma(
        persist_directory=index_dir,
        embedding_function=HuggingFaceEmbeddings(
            model_name=constants.EMBEDDING_MODEL
        ),
    )


def get_doc_retriever(k: int = 3):
    """Return a retriever for Marketplace X documentation.

    Args:
        k (int): Number of top results to return.

    Returns:
        BaseRetriever: Configured retriever ready for RAG chains.
    """
    vectorstore = load_doc_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})
