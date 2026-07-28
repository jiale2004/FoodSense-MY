# Local Development Setup

This guide walks through creating a Python virtual environment, starting the
FastAPI backend with uvicorn, and running the primary React (Vite) frontend with
npm. Commands assume macOS or Linux from the repository root
(`FoodSense-MY/`).

## Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| Python 3.10+ | Backend, YOLO, training scripts | `python3 --version` |
| Node.js 18+ and npm | React frontend in `frontend/` | `node --version` · `npm --version` |
| Promoted weights | Inference | `data/weights/best.pt` must exist |

LLM API keys are optional. Without them the advisory falls back to a local
template. See [LLM advisory (OpenAI / Gemini)](#4-llm-advisory-openai--gemini)
below for key setup and pricing notes.

## 1. Create and activate `.venv`

From the repository root:

```bash
cd FoodSense-MY

# Create the virtual environment (once)
python3 -m venv .venv

# Activate it (every new terminal)
source .venv/bin/activate

# Confirm you are inside the venv
which python
# → .../FoodSense-MY/.venv/bin/python
```

Install backend dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

Copy environment config (once):

```bash
cp .env.example .env
# Optional: edit .env and add OPENAI_API_KEY or GEMINI_API_KEY
```

Deactivate later with `deactivate`. Reactivate in a new shell with
`source .venv/bin/activate` before running uvicorn or training scripts.

### Windows (PowerShell)

```powershell
cd FoodSense-MY
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
copy .env.example .env
```

### Linux HPC (CUDA training only)

Use a separate venv and the CUDA 12.4 pin — see [`README.md`](../README.md)
Setup (Linux HPC) and [`requirements-hpc.txt`](../requirements-hpc.txt). Do not
mix that environment with day-to-day local uvicorn on macOS.

## 2. Start the backend (uvicorn)

Keep the venv activated. From the **repository root**:

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

`--app-dir backend` puts the `app` package on the import path. Equivalently:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

When startup finishes you should see logs that the model and knowledge base
loaded. Leave this terminal running.

### Smoke checks

In another terminal (venv not required for curl):

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/classes
```

Or open:

- Legacy static UI (fallback): [http://localhost:8000](http://localhost:8000)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Healthy `/api/health` includes `"status":"ok"` and `"model_loaded":true`.

## 3. Start the React frontend (npm)

The primary local UI is the Vite + React app in [`frontend/`](../frontend/).
Uvicorn still serves a legacy vanilla page at port 8000 if you need a no-npm
fallback.

**Terminal A** — uvicorn (section 2) must already be running on port 8000.

**Terminal B**:

```bash
cd FoodSense-MY/frontend
npm install          # once, or after package.json changes
npm run dev          # http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173). Use the bottom-right
**Ask FoodSense** chat anytime (no photo required), or drop an image and click
**Analyze**. Chat calls `POST /api/chat` with the knowledge base always attached
and optional last-scan context. The Vite proxy forwards `/api` and `/uploads` to
`http://127.0.0.1:8000`, so you do not need to change CORS for local testing.

Point at a different backend if needed:

```bash
VITE_API_TARGET=http://127.0.0.1:8000 npm run dev
```

## Two-terminal checklist

| Terminal | Commands |
|----------|----------|
| A — backend | `source .venv/bin/activate` → `uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000` |
| B — React UI | `cd frontend` → `npm install` (first time) → `npm run dev` |

| URL | What |
|-----|------|
| http://localhost:5173 | Primary React (Vite) frontend |
| http://localhost:8000 | FastAPI + legacy static UI fallback |
| http://localhost:8000/api/health | Backend health JSON |

## 4. LLM advisory (OpenAI / Gemini)

Detection and nutrition lookup do **not** need an LLM. The key only improves
the natural-language advisory returned by `/api/predict`. Without a key, the
app uses a local template fallback (still correct numbers from
`knowledge_base.json`).

### OpenAI

1. Create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Edit `.env` at the repo root:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

3. Restart uvicorn. Check `GET /api/health` → `"llm_configured": true`.

### Gemini

1. Create a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Edit `.env`:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-lite-latest
```

3. Restart uvicorn and re-check `/api/health`.

Do not commit `.env`. Prefer OpenAI `gpt-4o-mini` or Gemini
`gemini-flash-lite-latest` / `gemini-3.5-flash-lite`. Many new keys get 404
on `gemini-2.5-*` models even when they still appear in `list_models`.

Official pricing pages (rates change; verify before budgeting):

- OpenAI: [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing)
- Gemini: [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Common issues

| Symptom | Fix |
|---------|-----|
| `uvicorn: command not found` | Activate `.venv` (`source .venv/bin/activate`) and confirm `pip install -r requirements.txt` succeeded |
| `ModuleNotFoundError: No module named 'app'` | Run from repo root with `--app-dir backend`, or `cd backend` first |
| Model not loaded / missing weights | Ensure `data/weights/best.pt` exists; check `MODEL_WEIGHTS_PATH` in `.env` |
| Frontend analyze fails / network error | Start uvicorn on port 8000 before `npm run dev` |
| Port 8000 or 5173 already in use | Stop the other process, or change `--port` / Vite `server.port` |
| `npm: command not found` | Install Node.js 18+ from [nodejs.org](https://nodejs.org/) or your package manager |

## Related docs

- [`README.md`](../README.md) — full project overview, env vars, HPC setup
- [`docs/architecture.md`](architecture.md) — module layout and request flow
- [`docs/handoff.md`](handoff.md) — current model status and next steps
