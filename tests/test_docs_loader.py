"""
Tests: rag/docs_loader.py (Markdown loading + section splitting)
"""

from app.rag import docs_loader


def test_extract_sections_basic():
    text = (
        "# Title\nIntro text.\n## Section A\nContent A\n## Section B\nContent B"
    )
    sections = docs_loader.extract_sections(text)

    assert len(sections) == 3
    assert sections[0]["title"] == "Title"
    assert "Intro" in sections[0]["content"]
    assert sections[1]["title"] == "Section A"
    assert "Content A" in sections[1]["content"]


def test_load_markdown_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Test\nHello world")

    out = docs_loader.load_markdown_file(str(f))
    assert "# Test" in out
    assert "Hello" in out


def test_load_all_docs(tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\nxx")
    (d / "b.md").write_text("# B\nyy")

    monkeypatch.setattr(docs_loader, "DOCS_DIR", str(d))
    docs = docs_loader.load_all_docs()

    assert len(docs) == 2
    assert any(doc["title"] == "A" for doc in docs)
