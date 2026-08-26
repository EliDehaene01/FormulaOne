# Pit Wall

The application. Pick a past F1 session and get charts, an AI-written recap, a
data-grounded chat assistant, and a qualifying-pace prediction model.

- **Setup, prerequisites, and how to run everything:** see the [repository README](../README.md#getting-started).
- **How the system is built and why:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **PyTorch model notes (results, limitations, tuning ideas):** see [`backend/models/README.md`](backend/models/README.md).
- **Latest evaluation and experiment results:** [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md), [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

## Layout

| Path | What |
|---|---|
| `backend/data/` | OpenF1 client, SQLite cache, scheduled refresh |
| `backend/models/` | Feature engineering, PyTorch model, training, Optuna tuning, permutation importance, Captum attribution |
| `backend/llm/` | Azure chat client + tool-calling loop, tool definitions, race-summary generator |
| `backend/rag/` | Knowledge-base chunking, embeddings, Chroma store |
| `backend/knowledge_base/` | Hand-written F1 reference notes (glossary, tyre strategy, circuit guides, regulations) |
| `backend/api/` | FastAPI app |
| `backend/tests/` | Test suite (runs against the local cache, no live network / Azure) |
| `frontend/` | React + Vite + TypeScript single-page app |
| `evals/` | LLM evaluation suite (`azure-ai-evaluation`) |
| `docs/` | Architecture write-up and auto-generated result logs |

## Built in phases

| Phase | What | Where |
|---|---|---|
| 1 | Data layer — OpenF1 client + SQLite cache | `backend/data/` |
| 2 | PyTorch qualifying lap-time predictor | `backend/models/` |
| 3 | LLM tool-calling chat layer (Azure AI Foundry) | `backend/llm/tools.py`, `client.py` |
| 4 | LLM race-summary generator | `backend/llm/summary.py` |
| 5 | React frontend | `frontend/src/` |

Later additions: RAG knowledge base (`backend/rag/`), model hyperparameter
tuning and explainability (`backend/models/tune.py`, `importance.py`,
`explain.py`), and the evaluation suite (`evals/`).
