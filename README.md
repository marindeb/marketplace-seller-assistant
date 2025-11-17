# Marketplace Seller Assistant

A production-grade AI assistant for **marketplace sellers**, built to demonstrate advanced applied AI engineering.
It helps vendors understand platform policies, improve listings, analyze performance, and make data-driven decisions.
All answers are **grounded, cited**, and the system **refuses** when information is missing.

## Capabilities
- Policy guidance (eligibility, prohibited items, penalties)
- Growth and listing recommendations
- Seller analytics via a Pandas agent
- Strict grounded answers with citations
- Intent routing across multiple specialized agents
- Deterministic refusal behavior when documentation is insufficient

## Architecture Overview

```
User Query
   ↓
Intent Classifier (Mistral, Ollama)
   ↓
Router
   → Policy RAG Agent
   → Recommendation RAG Agent
   → Analytics Agent (Pandas)
   → Refusal Agent
   ↓
Grounded LLM Response (citations + guardrails)
```

### Core Components
- Python 3.10
- FastAPI API layer
- ChromaDB vector store (HuggingFace embeddings)
- LangChain RetrievalQA orchestration
- Ollama (Mistral) for routing and generation
- Pandas analytics agent for structured reasoning
- Extensible multi-agent design

## RAG System

- Markdown documentation automatically loaded and chunked
- Vector index built with Sentence-Transformer embeddings
- Policy and Recommendation RAG chains with strict grounding rules
- Confidence scoring and fallback refusal logic
- Future-ready for reranking and long-form retrieval

## Data
(Currently being regenerated for the new architecture)

```
data/
  docs/                   # Marketplace X seller documentation
  synthetic/              # Generated product, order, and return datasets
```

Planned tabular datasets:
- products.csv (catalog)
- orders.csv (order history)
- returns.csv (returns & disputes)
- seller_metrics.csv (aggregated KPIs)
- scenarios.json (end-to-end test scenarios)

## Evaluation Metrics
| Phase | Metric | Purpose |
|-------|--------|---------|
| Retrieval | Recall@k | Measures context quality |
| Grounding | Faithfulness | Ensures answers use retrieved content |
| Response | Citation compliance | Ensures all claims are sourced |
| Behavior | Refusal accuracy | Prevent unsafe or unsupported answers |
| Overall | LLM-judge | Helpfulness and correctness evaluation |

A **golden set** (50–100 curated Q/A) will benchmark the whole pipeline.

## Roadmap
- [x] Modular RAG architecture
- [x] Multi-agent router (policy / recommendation / analytics / refusal)
- [x] Full test suite for router & R
