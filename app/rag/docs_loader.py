"""
Module: loader.py
-----------------
Load and chunk Marketplace X documentation into structured, retrieval-friendly
chunks for the RAG index.

This module:
- Reads all markdown files from the seller documentation directory.
- Detects sections and subsections based on headings.
- Splits content into character-based chunks.
- Produces a standardized schema ready for vector indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import constants


logger = logging.getLogger(__name__)


def _get_docs_dir() -> Path:
    """Return the absolute path to the marketplace documentation directory."""
    return Path(constants.SELLER_DOCS_DIR).resolve()


def load_raw_docs() -> List[Dict]:
    """Load raw markdown documents from the seller docs directory.

    Returns:
        List[Dict]: A list of dictionaries, each representing a single document.
            [
                {
                    "doc_id": "overview",
                    "path": "/abs/path/01_overview.md",
                    "content": "Full markdown text..."
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the docs directory does not exist.
        RuntimeError: If no markdown files are found.
    """
    docs_dir = _get_docs_dir()
    if not docs_dir.exists():
        raise FileNotFoundError(f"Seller docs directory not found: {docs_dir}")

    raw_docs: List[Dict] = []

    for path in sorted(docs_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")

        stem = path.stem  # ex: "01_overview"
        parts = stem.split("_", 1)
        doc_id = parts[1] if len(parts) == 2 else stem

        raw_docs.append(
            {
                "doc_id": doc_id,
                "path": str(path),
                "content": content,
            }
        )

    if not raw_docs:
        raise RuntimeError(f"No markdown documents found in {docs_dir}")

    return raw_docs


def extract_sections(markdown_text: str) -> List[Dict]:
    """Extract structured sections from markdown text.

    Identifies:
        - Level-2 headings (## ) → sections
        - Level-3 headings (### ) → subsections

    If no sections are present, the entire document becomes a single section
    titled "General".

    Args:
        markdown_text (str): The raw markdown content.

    Returns:
        List[Dict]: A list of parsed section blocks.
            [
                {
                    "section": "Delivery Standards",
                    "subsection": "Late Shipments",
                    "text": "...",
                    "start_line": 42,
                    "end_line": 78,
                },
                ...
            ]
    """
    lines = markdown_text.splitlines()

    sections: List[Dict] = []
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    buffer: List[str] = []
    start_line: Optional[int] = None

    def flush(end_line: int) -> None:
        """Store the current buffered section/subsection."""
        nonlocal buffer, current_section, current_subsection, start_line
        if current_section and buffer:
            sections.append(
                {
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": "\n".join(buffer).strip(),
                    "start_line": start_line or 1,
                    "end_line": end_line,
                }
            )
        buffer = []
        start_line = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("## "):
            flush(line_no - 1)
            current_section = stripped[3:].strip()
            current_subsection = None
            start_line = line_no + 1
            continue

        if stripped.startswith("### "):
            flush(line_no - 1)
            current_subsection = stripped[4:].strip()
            start_line = line_no + 1
            continue

        if stripped.startswith("# "):
            continue  # Ignore top-level doc title

        buffer.append(line)

    flush(len(lines))

    if not sections:
        sections.append(
            {
                "section": "General",
                "subsection": None,
                "text": markdown_text.strip(),
                "start_line": 1,
                "end_line": len(lines),
            }
        )

    return sections


def chunk_section(
    section: Dict,
    doc_id: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """Split a section/subsection into smaller text chunks.

    Args:
        section (Dict): A section block returned by `extract_sections`.
        doc_id (str): Parent document identifier.
        chunk_size (int): Maximum chunk size in characters.
        chunk_overlap (int): Character overlap between chunks.

    Returns:
        List[Dict]: A list of chunks with metadata, except `chunk_id`
                    which will be added later in `chunk_docs`.
    """
    text = section["text"]
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)

    chunks: List[Dict] = []
    for chunk_text in raw_chunks:
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": None,  # filled later
                "text": chunk_text.strip(),
                "section": section["section"],
                "subsection": section.get("subsection"),
                "metadata": {
                    "path": None,  # filled later
                    "start_line": section["start_line"],
                    "end_line": section["end_line"],
                },
            }
        )
    return chunks


def chunk_docs(raw_docs: List[Dict]) -> List[Dict]:
    """Chunk all raw documents into retrieval-friendly chunks.

    Pipeline:
        1. Extract sections via `extract_sections`.
        2. Chunk each section via `chunk_section`.
        3. Assign global `chunk_id`s in the format "{doc_id}_{001}".

    Args:
        raw_docs (List[Dict]): Output of `load_raw_docs`.

    Returns:
        List[Dict]: A flattened list of all chunks.
    """
    all_chunks: List[Dict] = []

    for raw_doc in raw_docs:
        doc_id = raw_doc["doc_id"]
        path = raw_doc["path"]

        sections = extract_sections(raw_doc["content"])

        doc_chunks: List[Dict] = []
        for section in sections:
            section_chunks = chunk_section(section, doc_id=doc_id)
            for chunk in section_chunks:
                chunk["metadata"]["path"] = path
            doc_chunks.extend(section_chunks)

        for idx, chunk in enumerate(doc_chunks, start=1):
            chunk["chunk_id"] = f"{doc_id}_{idx:03d}"

        all_chunks.extend(doc_chunks)

    if not all_chunks:
        raise RuntimeError("No chunks generated from seller documentation.")

    return all_chunks


def load_and_chunk_docs() -> List[Dict]:
    """Full loading and chunking pipeline.

    Returns:
        List[Dict]: Chunk documents ready for indexing.
    """
    raw_docs = load_raw_docs()
    return chunk_docs(raw_docs)
