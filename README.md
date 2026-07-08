# FoodSense-MY

Malaysian Food Object Detection and Nutritional Advisory System. Upload a food image, detect 6 Malaysian dishes with PyTorch-accelerated YOLOv11n (MPS on Apple Silicon), look up verified nutrition data from a local JSON knowledge base, and receive a user-friendly advisory via OpenAI or Gemini.

## Target Classes

Nasi Lemak, Roti Canai, Char Kuey Teow, Chicken Rice, Laksa, Mee Goreng

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended for MPS acceleration) or Windows

## Setup (macOS)

```bash
cd FoodSense-MY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
| `CONFIDENCE_THRESHOLD` | NMS confidence cutoff | `0.5` |
| `IOU_THRESHOLD` | NMS IoU threshold | `0.45` |
| `DEVICE` | Compute device (`auto`, `mps`, `cpu`) | `auto` |
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
# 1. Convert PASCAL VOC annotations to YOLO format (Pandas)
python training_scripts/convert_voc_to_yolo.py --voc-dir ANNOTATIONS --images-dir IMAGES

# 2. Scrape supplemental images with icrawler (optional)
python training_scripts/scrape_images.py --output-dir data/raw_images --max-images 50

# 3. Prepare train/val/test split and data.yaml
python training_scripts/prepare_dataset.py --source-dir data/yolo_dataset --output-dir data/dataset

# 4. Tune hyperparameters with Optuna (Mixup/Mosaic for similar classes)
python training_scripts/tune_yolo.py --data data/dataset/data.yaml --n-trials 20

# 5. Train YOLOv11n with best params
yolo detect train model=yolo11n.pt data=data/dataset/data.yaml epochs=100 imgsz=640
```

## Project Structure

```
app/
├── main.py                 # FastAPI entry point
├── api/routes.py           # API endpoints (with DI)
├── core/config.py          # Settings + TARGET_CLASSES
├── core/security.py        # Upload validation
├── services/
│   ├── vision_service.py   # VisionProcessor (OpenCV + YOLOv11n + NMS)
│   ├── data_service.py     # KnowledgeRetriever (JSON lookup)
│   └── llm_service.py      # AdvisoryGenerator (LLM formatting layer)
├── models/schemas.py       # Pydantic models
└── static/                 # Frontend assets
data/
├── knowledge_base.json     # 6-class nutrition data
├── weights/                # YOLO weights
└── raw_images/             # Training images
training_scripts/
├── utils.py                # ReproducibilityManager
├── tune_yolo.py            # Optuna hyperparameter tuning
├── convert_voc_to_yolo.py  # VOC → YOLO via Pandas
├── scrape_images.py        # icrawler image scraper
└── prepare_dataset.py      # Dataset splits + data.yaml
docs/                       # Architecture documentation
```

## Disclaimer

This system provides nutritional information for educational purposes only. It is not medical advice. Consult a healthcare professional for dietary guidance.
