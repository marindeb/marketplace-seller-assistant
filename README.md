# Mistral E-commerce Intelligence Agent

A production-ready RAG + Agent system powered by Mistral 7B (via Ollama) and ChromaDB.

This project simulates an internal e-commerce assistant capable of:
- Answering product-related questions grounded in internal docs (RAG)
- Running analytics on operational data (returns, delivery performance…) using a Pandas agent
- Exposing an API (FastAPI) + optional UI (Streamlit)

---

## Architecture Overview

| Component | Purpose |
|---|---|
RAG Pipeline | Semantic product knowledge grounded in internal docs
Pandas Agent | Data analysis + Python execution sandbox
FastAPI | Unified interface for frontends / automation
Streamlit UI | Optional demo interface
ChromaDB | Vector store for embeddings
Ollama (Mistral) | Local LLM inference

Pipeline:

```
User Query → Router → RAG OR Pandas Agent → Mistral → Response
```

---

## Tech Stack

- Python 3.10.13
- Mistral 7B via Ollama
- LangChain
- ChromaDB
- FastAPI + Streamlit
- Pytest
- Ruff + Pre-commit

---

## Project Structure

```
mistral-ecommerce-agent/
├── app/
│   ├── agent.py
│   ├── rag_pipeline.py
│   ├── main.py
│   └── constants.py
├── data/
├── tests/
├── ui/
└── requirements*.txt
```

---

## Setup

### Python Environment

```bash
pyenv install 3.10.13
pyenv virtualenv 3.10.13 mistral-agent-env
pyenv activate mistral-agent-env

pip install --upgrade pip
pip install -r requirements.txt
```

### Dev Tools

```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## Run Ollama

```bash
ollama serve &
ollama pull mistral
```

---

## Run System

### Build RAG DB
```bash
python -m app.rag_pipeline
```

### API
```bash
uvicorn app.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

### Streamlit
```bash
streamlit run ui/app.py
```

---

## Use Pandas Agent

```python
from app.agent import ask_agent
ask_agent("Which categories have the highest return rate?")
```

---

## Tests
```bash
pytest -v
```

---

## Reset Vector DB

Force rebuild:
```bash
python -m app.rag_pipeline --rebuild
```

Manually clear:
```bash
rm -rf data/chroma_index
```

---

## Roadmap

- Add citations + confidence scores
- Sandbox execution fully
- CI/CD + Docker Compose
- Incremental RAG index updates
- Multi-model support

---

## License

MIT
