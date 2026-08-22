# Pit Wall — F1 Race Analytics Assistant

Pick a real past Formula 1 race and get:
- **Charts** — lap time deltas, position changes over the race, and tire strategy, built from real [OpenF1](https://api.openf1.org) session data.
- **An AI-generated summary** — a headline + narrative recap written by an LLM, grounded strictly in the race's actual race-control messages, lap times, and pit stops (never invented).
- **Chat** — ask "Pit Wall" (the in-app assistant) about the selected race — tire strategy, pit stops, who gained positions — answered by an LLM that calls real data-lookup tools rather than guessing.
- A first-pass **PyTorch model** that predicts qualifying lap times (accessible via chat, or backtestable directly) — see the honest results/limitations below.

A portfolio project built in 5 phases: data ingestion → PyTorch model → LLM tool-calling → LLM race summaries → this frontend.

## Prerequisites

- **Python 3.10+** and **Node.js 20+**
- An **Azure AI Foundry** project with a chat-capable model deployed (e.g. `gpt-4.1-mini`) — needed for the summary/chat features. Charts and the race selector work without one.
- **Windows only:** if `import torch` fails with an "Application Control policy" error, that's a known Windows DLL-blocking issue unrelated to this project — see [Known limitations](#known-limitations) below. Everything except the PyTorch model itself works fine without resolving it.

## Quick start

### 1. Backend setup

```
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

### 2. Configure the LLM connection

```
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your real Azure AI Foundry endpoint, API key, and deployment name. Use the **resource root** as the endpoint (e.g. `https://<resource>.services.ai.azure.com`), not the `/api/projects/...` path from the Foundry portal's URL bar — see `backend/llm/client.py` for why.

### 3. Get race data

Data is already cached in `backend/storage/cache.sqlite` if you're continuing this project. To (re)fetch it from scratch:

```
.venv\Scripts\python.exe -m backend.data.ingest --years 2024 2025 2026
```

This takes a while (OpenF1 is polite-rate-limited) — see `backend/data/ingest.py`'s docstring for options like `--limit` for a quick smoke test.

### 4. Start the backend API

```
.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload --port 8000
```

Leave this running. Visit `http://localhost:8000/docs` to confirm it's up.

### 5. Start the frontend

In a second terminal:

```
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — pick a race from the dropdown and explore the Charts / Summary / Chat tabs.

### (Optional) Train the qualifying predictor

The PyTorch model needs `torch`, which doesn't import natively on this Windows dev machine (see below) — run it under WSL instead:

```
wsl -d Ubuntu
cd /mnt/c/Users/elide/projects/f1-race-analytics
python3 -m venv ~/.venvs/f1-race-analytics
~/.venvs/f1-race-analytics/bin/pip install -r backend/requirements.txt
~/.venvs/f1-race-analytics/bin/python -m backend.models.train
```

This saves `backend/storage/qualifying_model.pt`, which the chat tool `predict_qualifying_pace` (and thus the Chat tab) uses automatically once it exists.

## Known limitations

- **Windows + PyTorch:** a Windows Application Control policy blocks `torch`'s native DLL on this dev machine. Training and the `predict_qualifying_pace` chat tool need to run under WSL as a result; everything else (data ingestion, charts, summaries, non-prediction chat) runs fine on plain Windows Python.
- **The qualifying predictor doesn't beat its own baseline yet** — an honest first-pass result on a small dataset (~800 training rows). Full writeup, bugs found/fixed, and tuning ideas: `backend/models/README.md`.
- **Chart data caveats:** a handful of tire stints in the raw OpenF1 data have missing lap ranges (likely early-race incidents) — handled by falling back to a zero-length stint rather than fabricating a number (see `backend/llm/tools.py`'s `get_tire_strategy`).

## How it's built

Each phase's implementation notes (design decisions, what was verified, bugs found along the way):

| Phase | What | Details |
|---|---|---|
| 1 | Data layer — OpenF1 client + SQLite cache | `backend/data/` |
| 2 | PyTorch qualifying lap time predictor | `backend/models/README.md` |
| 3 | LLM tool-calling chat layer (Azure AI Foundry) | `backend/llm/tools.py`, `client.py` |
| 4 | LLM race summary generator | `backend/llm/summary.py` |
| 5 | React + Vite + Tailwind/shadcn frontend | `frontend/src/` |

Backend: Python, FastAPI, SQLite, pandas, PyTorch, Azure AI Foundry (via the `openai` SDK). Frontend: React, Vite, TypeScript, Tailwind CSS v4, shadcn/ui, Recharts, TanStack Query. No database beyond the local SQLite cache — this is a single-user local/portfolio project, not a deployed multi-tenant service.

### Running tests

```
.venv\Scripts\python.exe -m backend.tests.test_cache
.venv\Scripts\python.exe -m backend.tests.test_openf1_client
.venv\Scripts\python.exe -m backend.tests.test_features
.venv\Scripts\python.exe -m backend.tests.test_tools
.venv\Scripts\python.exe -m backend.tests.test_summary
```

`test_model.py` needs `torch` — run it under WSL the same way as training. None of the tests call the real Azure API (that's verified manually, since it costs real tokens) or fetch from OpenF1 live (they run against the already-cached data).
