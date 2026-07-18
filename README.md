# FoodSense-MY

Malaysian Food Object Detection and Nutritional Advisory System. Upload a food image, detect 6 Malaysian dishes with PyTorch-accelerated YOLOv11n (MPS on Apple Silicon), look up verified nutrition data from a local JSON knowledge base, and receive a user-friendly advisory via OpenAI or Gemini.

## Target Classes

Nasi Lemak, Roti Canai, Char Kuey Teow, Chicken Rice, Laksa, Mee Goreng

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended for MPS acceleration), Windows, or Linux HPC with NVIDIA GPU

## Setup (macOS)

```bash
cd FoodSense-MY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Setup (Linux HPC / NVIDIA GPU)

Do not install the newest CUDA PyTorch wheel blindly. On nodes whose driver
reports CUDA 12.8 (driver API `12080`), a too-new wheel fails with:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
```

Use the CUDA 12.4 pin in [`requirements-hpc.txt`](requirements-hpc.txt):

```bash
cd FoodSense-MY
python3 -m venv .venv-hpc
source .venv-hpc/bin/activate
pip install -U pip
pip install -r requirements-hpc.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

See comments in [`requirements.txt`](requirements.txt) for the manual
`cu124` index-url alternative. Train with `device=0`, not `device=mps`.

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

Place only an approved YOLO detector at `data/weights/best.pt`. Until approved custom weights are available, the app falls back to Ultralytics pretrained `yolo11n.pt` (COCO classes).

After training and accepted holdout evaluation:

```bash
cp runs/detect/<approved-run>/weights/best.pt data/weights/best.pt
```

## Training Pipeline

The canonical unsplit staging dataset is `data/dataset3/`. After the completed
Phase A CVAT audit it contains 5,299 usable images, including 838 annotated
images with 855 boxes. Another 4,461 images still require annotation.

Do not run `training_scripts/prepare_dataset.py` against dataset3. That legacy
utility performs a file-level random split and does not preserve the manifest's
near-duplicate `leakage_group` values. Phase B generated the annotated-only,
group-aware `data/dataset3-baseline/` split with 587 train, 167 validation, and
84 candidate test images.

To create the split in a new workspace or versioned output directory:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-baseline \
  --seed 42 \
  --materialize hardlink
```

Revalidate the frozen split at any time with:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-baseline \
  --validate-only
```

The 84 test candidates in
`data/dataset3-baseline/test-review-queue.jsonl` still require manual review
before the test holdout is considered frozen. Pilot training commands:

```bash
# macOS Apple Silicon
yolo detect train \
  model=yolo11n.pt \
  data=data/dataset3-baseline/data.yaml \
  epochs=100 \
  imgsz=640 \
  seed=42 \
  device=mps

# Linux HPC NVIDIA GPU (after pip install -r requirements-hpc.txt)
yolo detect train \
  model=yolo11n.pt \
  data=data/dataset3-baseline/data.yaml \
  epochs=100 \
  imgsz=640 \
  seed=42 \
  device=0 \
  project=runs/detect \
  name=dataset3_pilot_v1
```

The Phase C run `runs/detect/dataset3_pilot_v1/` completed successfully. Its best
epoch reached mAP50 0.925 and mAP50–95 0.689 on validation. It is approved for
CVAT proposals, not production deployment; do not copy it to
`data/weights/best.pt` before the 84-image test candidate set is reviewed,
frozen, and evaluated. See
[`docs/experiments/dataset3_pilot_v1.md`](docs/experiments/dataset3_pilot_v1.md)
for the per-class assessment and limitations.

Before training on HPC, fix the absolute `path:` in
`data/dataset3-baseline/data.yaml` to the cluster path, and transfer the
baseline with hardlinks dereferenced (`rsync -aL`) because baseline images are
hardlinked into `data/dataset3/`.

See [`docs/handoff.md`](docs/handoff.md) for current counts and next-step gates,
[`docs/architecture.md`](docs/architecture.md) for the complete data flow, and
[`docs/bounding-box-policy.md`](docs/bounding-box-policy.md) for CVAT rules.

## Scraping and Image Curation

New downloads are appended to `data/manifests/downloads.jsonl`. UC mode records
the source URL and exact query; icrawler engines record the query and engine but
cannot always recover the original image URL.

```bash
# Scrape candidates. Raw images are not treated as verified labels.
python training_scripts/scrape_images.py \
  --engine uc \
  --classes char_kuey_teow chicken_rice \
  --max-images 1100 \
  --google-fallback

# Pilot technical validation, global deduplication, and semantic filtering.
python training_scripts/curate_images.py \
  --input-dir data/scraped_raw \
  --output-dir data/curation \
  --run-id two-class-pilot-new \
  --limit-per-class 100
```

Each curation run writes `curation.jsonl` and `summary.json` under
`data/curation/runs/<UTC timestamp>/`. Review the configured thresholds and
prompts in `training_scripts/configs/curation.yaml` after the pilot. Use
`--skip-semantic` to run only technical validation and deduplication, or
`--materialize none` to create manifests without image views.

After manually moving every pilot image from `review/` into `accepted/` or
`rejected/`, use that completed run to calibrate the full pass. This creates
hard-linked, run-scoped accepted/manual-review/rejected/duplicate views and
never modifies the source image folders:

```bash
python training_scripts/curate_images.py \
  --input-dir data/scraped_raw \
  --output-dir data/curation \
  --run-id two-class-full-new \
  --decisions-from data/curation/runs/two-class-pilot-new \
  --target-precision 0.98
```

Calibrated mode preserves the pilot decisions, auto-rejects only technical
failures, and routes semantic uncertainty into `manual_review/`. The saved
`calibration.json` contains out-of-fold precision, recall, and coverage for
each class. Manually sort only `manual_review/`, then audit a sample of
`accepted/` before training.

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
├── dataset3/               # Canonical unsplit staging dataset
├── cvat/pilot-300/         # CVAT input, exports, reports, revisions
├── dataset3-baseline/      # Generated group-safe 70/20/10 pilot split
└── weights/                # Approved custom weights
training_scripts/
├── utils.py                # ReproducibilityManager
├── tune_yolo.py            # Optuna hyperparameter tuning
├── convert_voc_to_yolo.py  # VOC → YOLO via Pandas
├── scrape_images.py        # icrawler image scraper
├── build_dataset3.py       # Source consolidation + leakage groups
├── prepare_cvat_pilot.py   # Deterministic CVAT batch packaging
├── import_cvat_annotations.py # Validated first merge and revisions
├── split_dataset3.py       # Deterministic leakage-safe baseline split
└── prepare_dataset.py      # Legacy splitter; not safe for dataset3
docs/                       # Handoff, architecture, annotation policy
```

## Disclaimer

This system provides nutritional information for educational purposes only. It is not medical advice. Consult a healthcare professional for dietary guidance.
