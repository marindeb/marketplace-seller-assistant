# Marketplace Seller Assistant

An assistant designed for **marketplace sellers**.
It helps vendors understand platform policies, optimize catalog and pricing, improve conversion, and avoid penalties.
All answers are **grounded and cited**. The model **refuses** to answer when uncertain.

## Capabilities
- Answers about marketplace policies (allowed products, returns, penalties)
- Provides catalog and conversion insights based on synthetic datasets
- Suggests operational improvements (price, logistics, ads)
- Responds with citations and confidence
- Refuses to answer if not supported by documentation

## Architecture

User query
→ Intent classifier (LLM)
→ Router (policy / product / logistics / unknown)
→ RAG or data lookup or rule module
→ LLM response (citations + guardrails)

### Components
- Python 3.10
- FastAPI
- ChromaDB or FAISS for vector storage
- LangChain for RAG orchestration
- Pandas for data lookup & reasoning
- LoRA fine-tuning (style + refusal behavior)
- Evaluation with Ragas + LLM judge

## Data
data/
docs/ # Public marketplace docs
synthetic/
products.csv
sales.csv
benchmarks.csv

pgsql
Copier le code

## Evaluation Metrics
| Phase | Metric | Purpose |
|---|---|---|
| Retrieval | Recall@k | Context relevance |
| Grounding | Faithfulness | Consistency with retrieved data |
| Response | Citation compliance | Cited sources in output |
| Behavior | Refusal accuracy | Refuse when uncertain |
| UX | Helpfulness | LLM-judge pairwise comparison |

The **golden set** (~100 curated Q&A pairs) covers topics such as product eligibility, delays, conversion levers, and penalties.

## Roadmap
- [x] RAG baseline
- [ ] Golden set creation
- [ ] Citations + refusal rules
- [ ] Rerank + evaluation pipeline
- [ ] Synthetic dataset generation
- [ ] LoRA fine-tuning (marketplace style)
- [ ] Simple monitoring (latency, hallucination rate)

## Run
```bash
uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
```

Important
This project uses synthetic and public data only.
It does not use any proprietary or confidential marketplace data.

yaml
Copier le code

---
