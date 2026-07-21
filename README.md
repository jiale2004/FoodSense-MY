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

The canonical unsplit staging dataset is `data/dataset3/`. After assisted
batch 007 and ingest `ingest_mee_goreng_full` it contains 5,428 usable images,
including 3,277 annotated images with 3,464 boxes. Another 2,151 images still
require annotation; 130 reviewed non-target images are quarantined. Mee Goreng
now has 922 usable images (178 annotated / 744 missing), including 251 newly
curated web images.

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

Before dataset3 changed, the frozen split could be revalidated against its
source snapshot with:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-baseline \
  --validate-only
```

After the Phase D assisted-batch merges, that command is expected to report a source
manifest mismatch because the baseline is an immutable pre-Phase-D snapshot.
Do not overwrite it; create a new versioned split after the annotation cycle or
when an interim retraining checkpoint is needed.

The immutable Phase B queue still records the original 84 candidates as
pending, but their no-proposal manual audit is now complete and applied to
Dataset3. The package is generated with:

```bash
python training_scripts/prepare_test_holdout_review.py
```

The current package is at `data/cvat/test-holdout-review-v1/`. CVAT project
`425516` now contains task `2441672`, job `4262178`, with all 84 images from
`images.zip`. `current-annotations.zip` was imported as **Ultralytics YOLO
Detection**. Human review retained 82 images with 84 boxes, rejected two
non-target frames, adjusted boxes on 52 other images, and corrected one
`mee_goreng` image to `char_kuey_teow`. The three completed pilot and
assisted-batch tasks were deleted from CVAT only after their local archives
passed integrity checks, leaving two hosted task slots free for rotation.

The following completed commands are retained as provenance; do not rerun the
same revision ID:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/test-holdout-review-v1 \
  --archive data/cvat/test-holdout-review-v1/cvat-reviewed-export.zip \
  --revision-id test-holdout-v1 \
  --task-id 2441672 \
  --job-id 4262178

# Only after checking the report:
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/test-holdout-review-v1 \
  --archive data/cvat/test-holdout-review-v1/cvat-reviewed-export.zip \
  --revision-id test-holdout-v1 \
  --task-id 2441672 \
  --job-id 4262178 \
  --apply
```

Revision `test-holdout-v1` updated Dataset3 with a recoverable pre-apply backup
while leaving `data/dataset3-baseline/` immutable. The locked incremental split
is now materialized at `data/dataset3-interim-v2/`:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-interim-v2 \
  --base-split-manifest data/dataset3-baseline/split-manifest.jsonl \
  --locked-test-selection data/cvat/test-holdout-review-v1/selection.jsonl \
  --incremental-train-fraction 0.8 \
  --seed 42 \
  --materialize hardlink
```

It contains 1,067 train images, 267 validation images, and exactly the 82
accepted reviewed holdout images in test. All 754 surviving baseline
train/validation assignments are preserved; the 580 newly annotated images are
assigned only to train or validation. The split has zero cross-split leakage
groups and its split-manifest SHA-256 is
`8e9db98b57dd53f01778afe8d1b66bc4e07975f639356c9758226102eecd90dd`.

Validate it locally with:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-interim-v2 \
  --validate-only
```

The earlier pilot training commands are retained as provenance:

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
`data/weights/best.pt` before final annotation and a single accepted holdout
evaluation. See
[`docs/experiments/dataset3_pilot_v1.md`](docs/experiments/dataset3_pilot_v1.md)
for the per-class assessment and limitations.

Phase D batch 001 has been prepared locally with model proposals:

```bash
python training_scripts/prepare_cvat_assisted_batch.py \
  --dataset-dir data/dataset3 \
  --split-manifest data/dataset3-baseline/split-manifest.jsonl \
  --output-dir data/cvat/assisted-batch-001 \
  --model runs/detect/dataset3_pilot_v1/weights/best.pt \
  --seed 43 \
  --confidence 0.20 \
  --iou 0.50 \
  --device cpu
```

This creates `images.zip` for a new task in the existing CVAT project and
`preannotations.zip` for an Ultralytics YOLO annotation import. Batch 001 has
300 images and 338 proposal boxes; it excludes every current candidate-test
leakage group. Upload the images first, import the pre-annotations, then review
every frame. A missing proposal is not an automatic rejection.

Batch 001 was completed in CVAT task `2439970`, job `4260450`; the hosted task
was deleted after its local archive passed integrity checks. All 338 proposals
were imported successfully. The reviewed export was validated and applied: 293
images received 309 boxes, 7 non-target images were quarantined, and recoverable
pre-merge metadata was retained.

Phase D batch 002 was prepared at `data/cvat/assisted-batch-002/` using seed 44
and the same class quotas and inference thresholds. It contains 300 new leakage
groups, has zero overlap with the 83 candidate-test groups or 600 prior CVAT
groups, and provides 326 proposal boxes on 296 images. Its image and
pre-annotation archives pass ZIP integrity checks. CVAT task `2441400`, job
`4261934`, contained all 300 images and all 326 proposals imported as
Ultralytics YOLO Detection rectangles; the hosted task was deleted after local
archival. Human review accepted 304 boxes on 287 images and
quarantined 13 non-target frames. The reviewed export passed dry validation and
was applied with a recoverable pre-merge backup.

Phase D batch 003 was prepared at `data/cvat/assisted-batch-003/` using seed 45
and `runs/detect/dataset3_interim_v2/weights/best.pt`. It contains 300 new
leakage groups, has zero overlap with the 81 locked test groups or 951 prior
CVAT groups, and provides 342 proposal boxes on 298 images. CVAT task
`2441914`, job `4262530`, reviewed the batch: human review accepted 304 boxes
on 289 images and quarantined 11 non-target frames. The reviewed export passed
dry validation and was applied with a recoverable pre-merge backup.

Phase D batch 004 was prepared at `data/cvat/assisted-batch-004/` using seed 46
and the same interim v2 checkpoint. It contains 500 new leakage groups, has
zero overlap with the 81 locked test groups or 1,251 prior selection groups,
and provides 589 proposal boxes on 494 images. CVAT task `2442189`, job
`4262800`, reviewed the batch: human review accepted 497 boxes on 469 images
and quarantined 31 non-target frames. The reviewed export passed dry validation
and was applied with a recoverable pre-merge backup.

Phase D batch 005 was prepared at `data/cvat/assisted-batch-005/` using seed 47
and `runs/detect/dataset3_interim_v3/weights/best.pt`. It contains 300 new
leakage groups, has zero overlap with the 81 locked test groups or 1,751 prior
selection groups, and provides 358 proposal boxes on 299 images. CVAT task
`2442437`, job `4263052`, reviewed the batch: human review accepted 304 boxes
on 290 images and quarantined 10 non-target frames, including 35
`mee_goreng` → `char_kuey_teow` corrections. Accepted Mee Goreng boxes were only
4, so later batches should deliberately raise Mee Goreng quotas. The reviewed
export passed dry validation and was applied with a recoverable pre-merge
backup.

Phase D batch 006 was prepared at `data/cvat/assisted-batch-006/` using seed 48
with raised Mee Goreng / Roti Canai / Laksa quotas and zero Char Kuey Teow
selections. It contains 361 new leakage groups, has zero overlap with the 81
locked test groups or 2,051 prior selection groups, and provides 441 proposal
boxes on all 361 images. CVAT task `2442499`, job `4263273`, reviewed the batch:
human review accepted 365 boxes on 334 images and quarantined 27 non-target
frames, including 81 `mee_goreng` → `char_kuey_teow` corrections. One
multi-class frame whose source `mee_goreng` class was absent from the reviewed
boxes was resolved with the new `import_cvat_annotations.py`
`--primary-class-override` flag. In total 93 of the 100 source Mee Goreng frames
were mislabeled or empty, leaving only 4 accepted Mee Goreng boxes, so genuine
Mee Goreng images must now be recruited by web scraping. The reviewed export
passed dry validation and was applied with a recoverable pre-merge backup.

Phase D batch 007 was prepared at `data/cvat/assisted-batch-007/` using seed 49
and `runs/detect/dataset3_interim_v3/weights/best.pt`. It contains 500 new
leakage groups, has zero overlap with the 81 locked test groups or 2,361 prior
selection groups, and provides 617 proposal boxes on 497 images. CVAT task
`2443011`, job `4263873`, reviewed the batch: human review accepted 528 boxes
on 479 images and quarantined 21 non-target frames, including 50
`mee_goreng` → `char_kuey_teow` corrections. Accepted Mee Goreng boxes were only
5. The reviewed export passed dry validation and was applied with a recoverable
pre-merge backup. Web scraping for genuine Mee Goreng followed: 600 candidates
were curated as `mee-goreng-full` (251 accepted) and incrementally ingested into
Dataset3 via `training_scripts/ingest_curated_images.py`.

Phase D batch 008 was prepared at `data/cvat/assisted-batch-008/` using seed 50
with Mee Goreng–heavy quotas (200/100/80/80/40/0). It contains 500 new leakage
groups, has zero overlap with the 81 locked test groups or 2,912 prior
selection groups, and provides 608 proposal boxes on 496 images (48 Mee Goreng
boxes). Of the 200 Mee Goreng slots, 64 are from the curated ingest. Upload
`images.zip` first and import `preannotations.zip` as **Ultralytics YOLO
Detection**.

The reserved holdout package contains 84 images, 86 existing human boxes, and
83 leakage groups. It includes all six object classes and no model-generated
annotations. `selection.jsonl` makes the later correction import revision-safe,
and package creation fails if the baseline queue, current images, or labels
have drifted.

Two HPC interim runs are complete. `runs/detect/dataset3_interim_v2/` selected
epoch 75 with mAP50 0.938 and mAP50–95 0.747. `runs/detect/dataset3_interim_v3/`
trained from the v2 checkpoint on the expanded
`data/dataset3-interim-v3/` split (1,674 / 418 / 82) and selected epoch 1 with
mAP50 0.932 and mAP50–95 0.761; its Mee Goreng recall (0.513) is the current
weakness, and the epoch-1 peak indicates the fine-tuning learning rate should be
lowered next retrain. Both checkpoints are accepted for assisted-batch proposals
only, not production deployment, and the locked test set remains untouched. See
[`docs/experiments/dataset3_interim_v2.md`](docs/experiments/dataset3_interim_v2.md)
and
[`docs/experiments/dataset3_interim_v3.md`](docs/experiments/dataset3_interim_v3.md)
for hashes, per-class validation, and error analysis.

See [`docs/handoff.md`](docs/handoff.md) for current counts and next-step gates,
[`docs/architecture.md`](docs/architecture.md) for the complete data flow, and
[`docs/bounding-box-policy.md`](docs/bounding-box-policy.md) for CVAT rules.
Group members should follow the end-to-end
[`docs/cvat-collaborator-guide.md`](docs/cvat-collaborator-guide.md) for ZIP
upload, annotation import, review, export, and reviewer handoff.

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
├── cvat/assisted-batch-001/ # Phase D images, proposals, and review metadata
├── cvat/assisted-batch-002/ # Reviewed export, merge report, and recovery metadata
├── cvat/assisted-batch-003/ # Interim-v2 proposals, reviewed export, and merge report
├── cvat/assisted-batch-004/ # 500-image reviewed export and merge report
├── cvat/assisted-batch-005/ # Interim-v3 reviewed export and merge report
├── cvat/assisted-batch-006/ # Mee Goreng-priority reviewed export and merge report
├── cvat/assisted-batch-007/ # 500-image reviewed export and merge report
├── cvat/assisted-batch-008/ # Mee Goreng–heavy proposal package; review pending
├── dataset3-baseline/      # Generated group-safe 70/20/10 pilot split
├── dataset3-interim-v2/    # Locked reviewed holdout + expanded train/validation
├── dataset3-interim-v3/    # Interim-v3 locked split (1,674/418/82)
└── weights/                # Approved custom weights
training_scripts/
├── utils.py                # ReproducibilityManager
├── tune_yolo.py            # Optuna hyperparameter tuning
├── convert_voc_to_yolo.py  # VOC → YOLO via Pandas
├── scrape_images.py        # icrawler / UC image scraper
├── ingest_curated_images.py # Incremental curated ingest into dataset3
├── build_dataset3.py       # Source consolidation + leakage groups
├── prepare_cvat_pilot.py   # Deterministic CVAT batch packaging
├── prepare_cvat_assisted_batch.py # Leakage-safe selection + YOLO proposals
├── import_cvat_annotations.py # Validated first merge and revisions
├── split_dataset3.py       # Fresh or locked-incremental leakage-safe split
└── prepare_dataset.py      # Legacy splitter; not safe for dataset3
docs/                       # Handoff, architecture, annotation policy
```

## Disclaimer

This system provides nutritional information for educational purposes only. It is not medical advice. Consult a healthcare professional for dietary guidance.
