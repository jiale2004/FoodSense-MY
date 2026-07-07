# FoodSense-MY

Malaysian Food Object Detection and Nutritional Advisory System. Upload a food image, detect Malaysian dishes with YOLOv11n, look up verified nutrition data from a local JSON knowledge base, and receive a user-friendly advisory via OpenAI or Gemini.

## Requirements

- Python 3.10+
- macOS or Windows

## Setup

```bash
# Clone and enter the project
cd FoodSense-MY

# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai` or `gemini` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `MODEL_WEIGHTS_PATH` | Path to YOLO weights | `data/weights/best.pt` |
| `KNOWLEDGE_BASE_PATH` | Path to nutrition JSON | `data/knowledge_base.json` |
| `CONFIDENCE_THRESHOLD` | Detection confidence cutoff | `0.25` |
| `MAX_UPLOAD_SIZE_MB` | Max upload size in MB | `10` |
| `API_KEY_ENABLED` | Require X-API-Key header | `false` |
| `API_KEY` | API key when protection enabled | — |

## Model Weights

Place your trained YOLOv11n weights at `data/weights/best.pt`. Until custom weights are available, the app falls back to Ultralytics pretrained `yolo11n.pt` (COCO classes).

After training:

```bash
cp runs/detect/train/weights/best.pt data/weights/best.pt
```

## Training Pipeline

```bash
# 1. Convert PASCAL VOC annotations to YOLO format
python training_scripts/convert_voc_to_yolo.py --help

# 2. Scrape supplemental images (optional)
python training_scripts/scrape_images.py --help

# 3. Prepare train/val/test split and data.yaml
python training_scripts/prepare_dataset.py --help

# 4. Train YOLOv11n
yolo detect train model=yolo11n.pt data=data/data.yaml epochs=100 imgsz=640
```

## Project Structure

```
app/
├── main.py                 # FastAPI entry point
├── api/routes.py           # API endpoints
├── core/config.py          # Settings
├── core/security.py        # Upload validation
├── services/               # Vision, data, LLM services
├── models/schemas.py       # Pydantic models
└── static/                 # Frontend assets
data/
├── knowledge_base.json     # Nutrition data
├── weights/                # YOLO weights
└── raw_images/             # Training images
training_scripts/           # Dataset preparation
docs/                       # Documentation
```

## Disclaimer

This system provides nutritional information for educational purposes only. It is not medical advice. Consult a healthcare professional for dietary guidance.
