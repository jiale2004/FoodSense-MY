# FoodSense-MY — Project Handoff

**Last updated:** 18 July 2026  
**Repository:** [FoodSense-MY](https://github.com/jiale2004/FoodSense-MY)  
**Purpose:** Malaysian food object detection and nutritional advisory system for 6 target dish classes.

---

## 1. Executive Summary

FoodSense-MY is a FastAPI web app that:

1. Accepts a food photo upload
2. Detects Malaysian dishes with **YOLOv11n** (PyTorch, MPS on Apple Silicon)
3. Looks up verified nutrition from a local JSON knowledge base
4. Returns a user-friendly advisory via OpenAI or Gemini (formatting layer only)

The **application scaffold is complete and runnable**. The **custom detection model is not trained yet** — inference currently falls back to Ultralytics pretrained `yolo11n.pt` (COCO classes). Raw web scraping for **Char Kuey Teow** and **Chicken Rice** is complete, and a calibrated curation run has reduced the remaining work to a single manual-review queue. Bounding-box annotation and custom YOLO training remain outstanding.

---

## 2. Target Classes (6)

| Class key | Display name | Images available (mapped/raw) | Status |
|-----------|--------------|---------------------------|--------|
| `nasi_lemak` | Nasi Lemak | ~1,310 | Ready |
| `roti_canai` | Roti Canai | ~1,000 | Ready |
| `laksa` | Laksa | ~1,103 | Ready |
| `mee_goreng` | Mee Goreng | ~1,106 | Ready |
| `char_kuey_teow` | Char Kuey Teow | 1,490 raw scraped | Curation in progress |
| `chicken_rice` | Chicken Rice | 1,500 raw scraped | Curation in progress |

All six classes have entries in [`data/knowledge_base.json`](../data/knowledge_base.json).

---

## 3. What's Built

### Backend (complete)

| Component | Class / file | Notes |
|-----------|--------------|-------|
| Entry point | [`app/main.py`](../app/main.py) | FastAPI + lifespan, static mount |
| Routes | [`app/api/routes.py`](../app/api/routes.py) | DI for services; `/health`, `/classes`, `/predict` |
| Vision | `VisionProcessor` — [`app/services/vision_service.py`](../app/services/vision_service.py) | OpenCV preprocess, YOLOv11n, NMS `conf=0.5` / `iou=0.45`, MPS/CPU |
| Data | `KnowledgeRetriever` — [`app/services/data_service.py`](../app/services/data_service.py) | JSON lookup, normalized class names |
| LLM | `AdvisoryGenerator` — [`app/services/llm_service.py`](../app/services/llm_service.py) | OpenAI/Gemini; strict formatting-only prompt; `DISCLAIMER` in response |
| Config | [`app/core/config.py`](../app/core/config.py) | `TARGET_CLASSES`, device, NMS thresholds |
| Security | [`app/core/security.py`](../app/core/security.py) | Upload validation, optional API key |

### Frontend (complete)

- [`app/static/`](../app/static/) — drag-and-drop upload, detection/nutrition cards, advisory panel, disclaimer footer

### Training pipeline (scripts ready, training not run)

| Script | Purpose |
|--------|---------|
| [`training_scripts/convert_voc_to_yolo.py`](../training_scripts/convert_voc_to_yolo.py) | PASCAL VOC → YOLO via Pandas (`VocToYoloConverter`) |
| [`training_scripts/prepare_dataset.py`](../training_scripts/prepare_dataset.py) | Train/val/test split + `data.yaml` (`DatasetPreparer`) |
| [`training_scripts/tune_yolo.py`](../training_scripts/tune_yolo.py) | Optuna hyperparameter tuning (`YoloHyperparameterTuner`) |
| [`training_scripts/scrape_images.py`](../training_scripts/scrape_images.py) | Image scraping — Google/Bing icrawler + SeleniumBase UC |
| [`training_scripts/google_crawler.py`](../training_scripts/google_crawler.py) | Hardened Google icrawler wrapper |
| [`training_scripts/uc_crawler.py`](../training_scripts/uc_crawler.py) | SeleniumBase Undetected Chrome scraper |
| [`training_scripts/curate_images.py`](../training_scripts/curate_images.py) | Curation CLI: validation, deduplication, semantic scoring, calibration, materialization |
| [`training_scripts/curation.py`](../training_scripts/curation.py) | Curation records, manifests, dHash/BK-tree deduplication, OpenCLIP scoring, pilot calibration |
| [`training_scripts/configs/curation.yaml`](../training_scripts/configs/curation.yaml) | Class prompts and technical/semantic thresholds |
| [`training_scripts/utils.py`](../training_scripts/utils.py) | `ReproducibilityManager` (fixed seeds) |

### Web scraping and calibrated curation (implemented)

The scraping/curation pipeline is non-destructive:

```text
Google/Bing/UC scrape
    → data/scraped_raw/<class>/
    → data/manifests/downloads.jsonl
    → technical validation
    → exact SHA-256 + perceptual dHash deduplication
    → OpenCLIP pilot scoring
    → manual pilot decisions
    → class-specific centroid calibration
    → accepted / manual_review / rejected / duplicate
```

Key behavior:

- Future downloads append provenance records including class, query, engine, hash, dimensions, and source URL when UC can recover it.
- Source images are never deleted or modified by curation.
- Exact duplicates are detected globally before technical routing; near-duplicates use dHash with a BK-tree index.
- Technical checks cover decoding, file size, dimensions, pixel count, aspect ratio, and blur warnings.
- A completed pilot's folder moves are authoritative manual decisions and are preserved in later runs.
- Calibrated mode never semantically auto-rejects a valid new image. It auto-accepts only above a cross-validated precision threshold and sends all other semantic uncertainty to `manual_review/`.
- Run outputs are hard links by default, so status views do not duplicate image storage.

The completed 200-image pilot is:

```text
data/curation/runs/two-class-pilot/
```

Pilot manual decisions:

| Class | Accepted | Rejected | Unresolved |
|-------|---------:|---------:|-----------:|
| Char Kuey Teow | 42 | 58 | 0 |
| Chicken Rice | 84 | 16 | 0 |

The authoritative full run is:

```text
data/curation/runs/two-class-full-v2/
```

| Class | Accepted | Manual review | Rejected | Duplicate | Total |
|-------|---------:|--------------:|---------:|----------:|------:|
| Char Kuey Teow | 285 | 894 | 283 | 28 | 1,490 |
| Chicken Rice | 527 | 722 | 220 | 31 | 1,500 |
| **Total** | **812** | **1,616** | **503** | **59** | **2,990** |

All 2,990 manifest records have corresponding materialized files. The run preserved all 200 pilot decisions and introduced zero semantic auto-rejections for new images.

Calibration metrics are out-of-fold estimates from the manually sorted pilot:

| Class | Auto-accept precision | Recall | Coverage |
|-------|----------------------:|-------:|---------:|
| Char Kuey Teow | 100% | 47.6% | 20% |
| Chicken Rice | 98.2% | 65.5% | 56% |

To reproduce the calibrated routing in a new immutable run:

```bash
python training_scripts/curate_images.py \
  --input-dir data/scraped_raw \
  --output-dir data/curation \
  --run-id two-class-full-v3 \
  --decisions-from data/curation/runs/two-class-pilot \
  --target-precision 0.98
```

Each run writes `curation.jsonl`, `summary.json`, and the routed class folders. Calibrated
runs also write `calibration.json` and `manual_decisions.jsonl`. Moving files between routed
folders does not rewrite `curation.jsonl`; that folder state is imported as manual decisions
when the run is later supplied through `--decisions-from`.

Only `two-class-full-v2/manual_review/` requires complete manual sorting. Audit a sample of `accepted/`; `rejected/` contains technical failures and preserved manual rejections, while `duplicate/` requires no review.

Seven regression tests cover provenance, corrupt images, technical duplicates, semantic routing, calibrated routing, pilot-decision preservation, and calibration thresholds:

```bash
python -m unittest discover -s tests -v
```

### Documentation

- [`README.md`](../README.md) — setup and usage
- [`docs/architecture.md`](architecture.md) — system design, env vars, training workflow

---

## 4. Data Inventory (local, gitignored)

Data lives under `data/` (not committed). Approximate counts:

### `data/dataset1/` — 1,591 images (11 classes, ~100–470 each)

Includes target classes: Nasi Lemak (110), Laksa (103), Mee Goreng (104).  
Also contains non-target classes (Rice, Burger, Pizza, etc.) — filter during merge.

### `data/dataset2/` — 11,000 images (11 classes, 1,000 each)

Includes target classes: Nasi Lemak, Roti Canai, Laksa, Mee Goreng.  
Non-target: Satay, Popiah, Mixed Rice, Kaya Toast, Hamburger, Fried Rice, Fish and Chips.

### `data/dataset3/` — 202 legacy scraped images

| Folder | Count | Notes |
|--------|-------|-------|
| `nasi_lemak` | 200 | Extra scraped images |
| `mee_goreng` | 2 | Just started |
| `char_kuey_teow` | 0 | New scrape is in `data/scraped_raw/` |
| `chicken_rice` | 0 | New scrape is in `data/scraped_raw/` |
| `roti_canai`, `laksa` | 0 | — |

### `data/scraped_raw/` — 2,990 raw candidates

| Folder | Count | Notes |
|--------|------:|-------|
| `char_kuey_teow` | 1,490 | UC/Google/Bing candidate images |
| `chicken_rice` | 1,500 | UC/Google/Bing candidate images |

Download provenance is appended to `data/manifests/downloads.jsonl`.

### `data/curation/runs/` — run-scoped curated views

- `two-class-pilot/` — manually completed 200-image calibration pilot
- `two-class-full-v2/` — authoritative calibrated full run
- Earlier `phase*`, `calibrated-verification*`, and `two-class-full/` directories are verification/superseded runs and must not be used for the final dataset.

**Format:** The source datasets and curated views are still **classification folders** (images only, no bounding boxes). Real object-detection boxes should be created and reviewed in CVAT before YOLO training. Full-image boxes are possible as a bootstrap shortcut but are not equivalent to object-detection annotation.

### Not yet created

- `data/yolo_dataset/` — merged YOLO images + labels
- `data/dataset/data.yaml` — train/val/test split for Ultralytics
- `data/weights/best.pt` — trained custom weights

---

## 5. Model & Inference Status

| Item | Status |
|------|--------|
| Custom YOLO weights (`data/weights/best.pt`) | **Missing** |
| Dev fallback (`yolo11n.pt`) | Present in project root (gitignored) |
| MPS acceleration | Supported; auto-selected on Apple Silicon |
| End-to-end `/api/predict` | Works with fallback model (COCO classes, not Malaysian dishes) |

Until custom weights are trained and deployed, detections will **not** reliably identify Malaysian food classes.

---

## 6. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Model loaded, device, KB entries, LLM status |
| `GET` | `/api/classes` | List of 6 known dish classes from knowledge base |
| `POST` | `/api/predict` | Upload image → detections, nutrition, advisory, disclaimer |
| `GET` | `/` | Static frontend |

### `PredictResponse` fields

`detections`, `nutrition`, `advisory_text`, `disclaimer`, `image_url`, `processing_ms`

---

## 7. Environment Setup

```bash
cd FoodSense-MY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY or GEMINI_API_KEY (optional — template fallback works without)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Key env vars: `CONFIDENCE_THRESHOLD=0.5`, `IOU_THRESHOLD=0.45`, `DEVICE=auto`, `LLM_PROVIDER=openai|gemini`.

---

## 8. Git History (recent)

```
a0d42fe feat: add SeleniumBase UC engine for large-scale image scraping
5c0731d fix: harden Google image scraper and prevent icrawler parser crash
b2a3235 feat: upgrade to PyTorch/MPS YOLOv11 with OOP services and Optuna tuning
d80ad7e Update .gitignore
89533ad feat: scaffold FoodSense-MY app with YOLOv11n pipeline
```

**Current working tree:** The scraping/curation implementation, tests, requirements, README, and this handoff are uncommitted. `data/` run outputs are intentionally gitignored. There are also pre-existing workspace changes (`data/raw_images/.gitkeep` deleted) that should be reviewed separately before committing.

---

## 9. Architecture (high level)

```
User upload → VisionProcessor (OpenCV + YOLOv11n + NMS)
           → KnowledgeRetriever (knowledge_base.json)
           → AdvisoryGenerator (LLM formatting / template fallback)
           → JSON response + disclaimer
```

See [`docs/architecture.md`](architecture.md) for diagrams and module table.

---

## 10. Design Constraints (do not break)

1. **Strict OOP** — logic in service classes with docstrings and type hints
2. **LLM is formatting-only** — never invent calories, macros, or medical advice
3. **NMS explicit** — `conf=0.5`, `iou=0.45` (configurable via `.env`)
4. **6 classes only** — `TARGET_CLASSES` in config is canonical
5. **YOLOv11n only** — Ultralytics nano variant
6. **Disclaimer always returned** — server-side `DISCLAIMER` constant, independent of LLM

---

## 11. Recommended Next Steps

### Priority 1 — Complete the single manual-review queue

Manually move every image from:

```text
data/curation/runs/two-class-full-v2/manual_review/<class>/
```

into the same run's matching `accepted/<class>/` or `rejected/<class>/` folder. Do not use or edit the superseded `two-class-full/` run. After sorting, preserve the decisions in a manifest before regenerating another run.

### Priority 2 — Create and review bounding boxes in CVAT

1. Build CVAT tasks from the accepted curated images plus accepted images for the other four classes.
2. Use exactly the canonical six class keys and class order from `app/core/config.py`.
3. Use an open-vocabulary detector for the first proposal pass, then manually correct boxes.
4. Label every visible target dish, not only the folder/search class.
5. Export **Ultralytics YOLO Detection** format.
6. Validate image/label pairing, class IDs, normalized coordinates, and box bounds before splitting.

CVAT automation is planned but is not implemented in this repository yet.

### Priority 3 — Prepare leakage-safe train/val/test splits

Do not split near-duplicates across datasets. Group by SHA-256/perceptual hash and source provenance, then stratify by object instances. The existing `prepare_dataset.py` must be updated to clean or version its output directory so stale files cannot appear in multiple splits.

### Priority 4 — Train and deploy model

```bash
python training_scripts/tune_yolo.py --data data/dataset/data.yaml --n-trials 20
yolo detect train model=yolo11n.pt data=data/dataset/data.yaml epochs=100 imgsz=640
cp runs/detect/train/weights/best.pt data/weights/best.pt
```

### Priority 5 — Validate end-to-end

Upload Malaysian food photos via UI; confirm detections match expected classes and nutrition lookup works.

---

## 12. Known Gaps & Risks

| Gap | Impact |
|-----|--------|
| 1,616 curated candidates still require manual review | Final accepted two-class dataset is incomplete |
| CVAT task/export automation is not implemented | Bounding-box workflow is still manual |
| No merge/export validator for CVAT → final YOLO layout | Blocks a reproducible training dataset |
| No custom weights | Production detections use COCO, not Malaysian food |
| Classification-only data (no real boxes yet) | Cannot train a reliable object detector until CVAT annotation is complete |
| Existing split script retains stale output | Reruns with changed seed/source can leak images across splits |
| Scraper fragility | Google/UC scraping may break or hit CAPTCHAs |

---

## 13. Project Structure

```
FoodSense-MY/
├── app/
│   ├── main.py
│   ├── api/routes.py
│   ├── core/config.py, security.py
│   ├── services/vision_service.py, data_service.py, llm_service.py
│   ├── models/schemas.py
│   └── static/                    # Frontend
├── data/                          # gitignored
│   ├── dataset1/, dataset2/       # Kaggle/GitHub classification data
│   ├── dataset3/                  # Legacy scraped images
│   ├── scraped_raw/               # 2,990 two-class raw candidates
│   ├── manifests/                 # Download provenance JSONL
│   ├── curation/runs/             # Pilot and calibrated run outputs
│   ├── knowledge_base.json        # 6-class nutrition JSON
│   ├── raw_images/, weights/
├── training_scripts/
│   ├── scrape_images.py, uc_crawler.py, google_crawler.py
│   ├── curate_images.py, curation.py
│   ├── configs/curation.yaml
│   ├── convert_voc_to_yolo.py, prepare_dataset.py, tune_yolo.py
│   └── utils.py
├── tests/
│   └── test_curation.py           # 7 curation/scraping regression tests
├── docs/
│   ├── architecture.md
│   └── handoff.md                 # this file
├── requirements.txt
├── .env.example
└── README.md
```

---

## 14. Contacts & Conventions

- **Commit style:** Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`)
- **Cursor rules:** Local only (`.cursor/` gitignored); agent suggests commit messages on job completion
- **Large files:** `data/*`, `datasets/`, `*.pt`, scraped images — all gitignored

---

*For detailed system design, see [architecture.md](architecture.md). For setup instructions, see [README.md](../README.md).*
