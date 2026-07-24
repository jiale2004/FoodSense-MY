# Local Development Setup

This guide walks through creating a Python virtual environment, starting the
FastAPI backend with uvicorn, and running the optional React test frontend with
npm. Commands assume macOS or Linux from the repository root
(`FoodSense-MY/`).

## Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| Python 3.10+ | Backend, YOLO, training scripts | `python3 --version` |
| Node.js 18+ and npm | React test UI in `frontend/` | `node --version` · `npm --version` |
| Promoted weights | Inference | `data/weights/best.pt` must exist |

LLM API keys are optional. Without them the advisory falls back to a local
template.

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

- Bundled static UI: [http://localhost:8000](http://localhost:8000)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Healthy `/api/health` includes `"status":"ok"` and `"model_loaded":true`.

## 3. Start the React test frontend (npm)

The Vite app in [`frontend/`](../frontend/) is optional. Use it to exercise
image upload against the live API. The production static UI is already served
by uvicorn at port 8000.

**Terminal A** — uvicorn (section 2) must already be running on port 8000.

**Terminal B**:

```bash
cd FoodSense-MY/frontend
npm install          # once, or after package.json changes
npm run dev          # http://localhost:5173
```

Open [http://localhost:5173](http://localhost:5173). Drop an image and click
**Analyze**. The Vite proxy forwards `/api` and `/uploads` to
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
| http://localhost:8000 | FastAPI + bundled static frontend |
| http://localhost:5173 | Vite React upload-test UI |
| http://localhost:8000/api/health | Backend health JSON |

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
