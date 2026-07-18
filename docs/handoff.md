# FoodSense-MY — Project Handoff

**Last updated:** 18 July 2026  
**Repository:** [FoodSense-MY](https://github.com/jiale2004/FoodSense-MY)  
**Purpose:** Six-class Malaysian food object detection and nutritional advisory.

## 1. Current State

The FastAPI application and its static frontend are runnable. The first custom six-class YOLO11n pilot has completed, but it is retained as an experiment and has not been promoted to the application. The current inference fallback is still Ultralytics `yolo11n.pt`, whose COCO classes are not suitable for the target dishes.

Image acquisition and consolidation are paused. The canonical staging dataset is now `data/dataset3/`. It combines approved content from dataset1, dataset2, the manually curated two-class web run, and two Roboflow imports. Exact duplicates were collapsed, near-duplicate leakage groups were recorded, and the initial CVAT pilot plus its Phase A priority audit have been completed and merged.

Current dataset3 totals:

- 5,299 usable images
- 838 annotated images
- 855 bounding boxes
- 4,461 images still missing bounding-box annotations
- 8 CVAT pilot images rejected because none of the six target classes was present
- 5,253 leakage groups; 52 groups contain more than one image
- no custom `data/weights/best.pt` yet
- Phase C pilot checkpoint available at `runs/detect/dataset3_pilot_v1/weights/best.pt`

Phase B has also materialized the 838 annotated images as the validated
`data/dataset3-baseline/` training view. Its 84 test candidates are not yet a
frozen holdout because their final manual review is pending.

Do not treat an empty YOLO label as an accepted background image. Images confirmed to contain none of the six classes are quarantined under `data/dataset3/rejected/` and marked `rejected` in the manifest.

## 2. Canonical Classes and Annotation Coverage

Class IDs are fixed and must not be reordered:

| ID | Class key | Usable images | Annotated images | Boxes | Missing annotations |
|---:|-----------|--------------:|-----------------:|------:|--------------------:|
| 0 | `nasi_lemak` | 996 | 50 | 51 | 946 |
| 1 | `roti_canai` | 995 | 46 | 55 | 949 |
| 2 | `char_kuey_teow` | 493 | 152 | 153 | 341 |
| 3 | `chicken_rice` | 709 | 313 | 316 | 396 |
| 4 | `laksa` | 1,100 | 150 | 151 | 950 |
| 5 | `mee_goreng` | 1,006 | 127 | 129 | 879 |
| **Total** | — | **5,299** | **838** | **855** | **4,461** |

The class folder records an image's primary/source class. The YOLO label content is authoritative and may contain more than one class when multiple target dishes are visible.

## 3. Dataset3 Assembly

The reproducible builder is [`training_scripts/build_dataset3.py`](../training_scripts/build_dataset3.py). It materializes one canonical copy per SHA-256 digest, assigns a dHash-based leakage group, preserves available YOLO labels, and writes a full provenance manifest.

```bash
python training_scripts/build_dataset3.py \
  --project-root . \
  --output-dir data/dataset3 \
  --materialize hardlink \
  --dhash-distance 6
```

Important assembly decisions:

- `data/dataset1/6_Nasi_Goreng/` is excluded. It was renamed from a misleading Nasi Lemak folder and is not a target source.
- `data/dataset2/6_Nasi_Lemak_2/918.jpg` is an MP4 file with a `.jpg` suffix and is excluded.
- Two exact dataset2 cross-folder conflicts were resolved by content hash: one belongs to `chicken_rice`, the other to `roti_canai`.
- One duplicated Mee Goreng annotation pair was resolved with the union bounding box.
- 75 exact duplicate source occurrences were collapsed during assembly.
- The initial assembly produced 5,293 usable images in 5,253 leakage groups. Phase A restored six valid Laksa images from the pilot quarantine, producing the current 5,299 usable images. Any train/validation/test splitter must keep each group in exactly one split.

Dataset3 is a staging dataset, not a finished 70/20/10 training split:

```text
data/dataset3/
├── nasi_lemak/{images,labels}/
├── roti_canai/{images,labels}/
├── char_kuey_teow/{images,labels}/
├── chicken_rice/{images,labels}/
├── laksa/{images,labels}/
├── mee_goreng/{images,labels}/
├── rejected/cvat_pilot_300/<source-class>/images/
├── manifest.jsonl
├── summary.json
└── README.md
```

`manifest.jsonl` is the source of truth for file paths, SHA-256, dHash, leakage group, source provenance, annotation status, and annotation counts. `summary.json` is its generated aggregate.

## 4. CVAT Pilot Completed

CVAT Online objects:

- Project: `FoodSense-MY dataset3` — project ID `425516`
- Task: `dataset3 bounding-box pilot 300` — task ID `2438268`
- Job ID: `4258646`, frames `0–299`

[`training_scripts/prepare_cvat_pilot.py`](../training_scripts/prepare_cvat_pilot.py) selected 50 missing-label images per source class with seed 42. All 300 selected images came from distinct leakage groups. It created `data/cvat/pilot-300/images.zip` and its selection metadata.

```bash
python training_scripts/prepare_cvat_pilot.py \
  --dataset-dir data/dataset3 \
  --output-dir data/cvat/pilot-300 \
  --per-class 50 \
  --seed 42
```

Initial manual annotation outcome before the Phase A audit:

- 286 images labelled
- 299 new boxes
- 14 images rejected: 4 sourced as Roti Canai, 1 as Chicken Rice, and 9 as Laksa
- one image contains two target dishes: Char Kuey Teow and Mee Goreng
- all exported rows passed class-ID, coordinate, image-mapping, and bounds validation

Pilot boxes by the annotator's actual class:

| Class | New pilot boxes |
|-------|----------------:|
| `nasi_lemak` | 51 |
| `roti_canai` | 55 |
| `char_kuey_teow` | 76 |
| `chicken_rice` | 50 |
| `laksa` | 41 |
| `mee_goreng` | 26 |

The initial pilot also exposed meaningful source-class errors:

- 33 source `mee_goreng` images were labelled `char_kuey_teow`
- 8 source `char_kuey_teow` images were labelled `mee_goreng`
- 1 source `laksa` image was labelled `mee_goreng`

The merge was applied on 18 July 2026 with [`training_scripts/import_cvat_annotations.py`](../training_scripts/import_cvat_annotations.py):

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/pilot-300 \
  --archive data/cvat/pilot-300/cvat-export.zip \
  --task-id 2438268 \
  --job-id 4258646 \
  --apply
```

The importer validates the complete archive before changing dataset3. The original manifest, summary, and dataset README were backed up to `data/cvat/pilot-300/pre-merge/`. Merge evidence is retained in:

```text
data/cvat/pilot-300/
├── images.zip
├── selection.jsonl
├── summary.json
├── cvat-export.zip
├── merge-report.json
├── rejected.jsonl
├── pre-merge/
├── cvat-audited-export-v2.zip
└── revisions/phase-a-v2/
```

### Phase A audit and correction

Phase A was executed on 18 July 2026. The audit covered 63 priority frames: all 42 initial source-to-label transitions, all 14 initially rejected frames, the multi-class frame, and one ordinary example per class.

Ten high-confidence corrections were applied in CVAT:

- six valid Laksa images were restored from the rejected quarantine;
- one Mee Goreng label was corrected to Char Kuey Teow;
- the initial Laksa-to-Mee Goreng transition was corrected back to Laksa Johor;
- one second Char Kuey Teow serving box was added;
- one second Mee Goreng serving box was added.

The audited export contains 292 labelled images and 307 boxes, with eight frames remaining rejected. Five additional export differences were row-order-only and were intentionally ignored by the semantic comparison.

The corrected export was validated and applied with the revision-safe importer:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/pilot-300 \
  --archive data/cvat/pilot-300/cvat-audited-export-v2.zip \
  --revision-id phase-a-v2 \
  --task-id 2438268 \
  --job-id 4258646

# Run only after inspecting the dry-run report above.
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/pilot-300 \
  --archive data/cvat/pilot-300/cvat-audited-export-v2.zip \
  --revision-id phase-a-v2 \
  --task-id 2438268 \
  --job-id 4258646 \
  --apply
```

`data/cvat/pilot-300/revisions/phase-a-v2/` contains the audited export, revision report, pre-apply dataset metadata, pilot reports, and every replaced label. The original first-merge export and report remain unchanged.

## 5. Bounding-Box Policy

The full policy is [`docs/bounding-box-policy.md`](bounding-box-policy.md). Its key rules are:

- annotate every visible instance of the six target dishes, regardless of the image's source folder;
- use a tight rectangle around the complete edible serving, including integral components of the dish;
- exclude unrelated tableware, hands, people, packaging, and surrounding table where practical;
- keep partially visible or occluded dishes when the class is still confidently identifiable;
- use separate boxes for distinct servings, and separate class boxes when multiple target dishes are present;
- do not force a source class onto the image;
- mark an image as rejected when no target class is present instead of accepting it with an empty label.

## 6. Reproducible Dataset Utilities

| Script | Responsibility |
|--------|----------------|
| [`import_roboflow_subset.py`](../training_scripts/import_roboflow_subset.py) | Import a selected Roboflow class, validate labels, preserve source metadata, and deduplicate |
| [`build_dataset3.py`](../training_scripts/build_dataset3.py) | Consolidate all approved sources into the canonical six-class staging dataset |
| [`prepare_cvat_pilot.py`](../training_scripts/prepare_cvat_pilot.py) | Select a deterministic class-balanced missing-label batch and package it for CVAT |
| [`import_cvat_annotations.py`](../training_scripts/import_cvat_annotations.py) | Validate first-time or replacement CVAT exports, merge or revise labels, retain recoverable backups, and quarantine rejected frames |
| [`split_dataset3.py`](../training_scripts/split_dataset3.py) | Build and validate a deterministic annotated-only split while preserving leakage groups and immutable label snapshots |
| [`scrape_images.py`](../training_scripts/scrape_images.py) | Acquire Google/Bing/UC image candidates with provenance |
| [`curate_images.py`](../training_scripts/curate_images.py) | Validate, deduplicate, score, calibrate, and route scraped candidates |
| [`convert_voc_to_yolo.py`](../training_scripts/convert_voc_to_yolo.py) | Convert reviewed PASCAL VOC boxes to YOLO labels |
| [`prepare_dataset.py`](../training_scripts/prepare_dataset.py) | Legacy random split utility; do not use for dataset3 without leakage-aware changes |
| [`tune_yolo.py`](../training_scripts/tune_yolo.py) | Run Optuna hyperparameter trials for YOLO |

## 7. Application and Model Status

| Area | Status |
|------|--------|
| FastAPI app, routes, services | Implemented |
| Static upload/result UI | Implemented |
| Six-class nutrition knowledge base | Implemented |
| OpenAI/Gemini advisory formatting | Implemented, optional |
| Apple Silicon MPS inference | Supported |
| Canonical dataset3 assembly | Implemented |
| CVAT batch preparation/import | Implemented and pilot-tested |
| Group-aware annotated-only splitter | Implemented and validated |
| Custom six-class YOLO model | Phase C pilot trained and evaluated; assisted-labelling use only |
| Production `data/weights/best.pt` | Missing |

Until custom weights are approved and copied to `data/weights/best.pt`, `/api/predict` is only a scaffold test and does not reliably identify the six Malaysian dishes.

## 8. Verification Completed

After the Phase A audited revision, Phase B split, and Phase C pilot evaluation:

- `manifest.jsonl` contains 5,307 records: 5,299 usable plus 8 rejected
- status totals are 838 annotated, 4,461 missing, and 8 rejected
- all 855 YOLO rows use canonical class IDs and valid normalized coordinates
- every usable manifest path and annotation file is consistent
- the baseline contains 587 train, 167 validation, and 84 candidate test images
- Phase B validation reports zero cross-split leakage groups and zero missing pairs
- every object class is represented in validation and test
- the reproducible split-manifest hash is `f87d6f4ab07e463ddca111c4add9c5a6236acf4d08e0c0500b8802f1e45e7d1e`
- the best Phase C checkpoint is epoch 85 with precision 0.928, recall 0.867, mAP50 0.925, and mAP50–95 0.689
- the checkpoint's six IDs match the canonical class order
- the pilot artifacts and evaluation are recorded in [`docs/experiments/dataset3_pilot_v1.md`](experiments/dataset3_pilot_v1.md)
- all ten repository regression tests pass, including revision and split-integrity coverage
- `git diff --check` passes

Run the current test suite with:

```bash
python -m unittest discover -s tests -v
```

## 9. Recommended Next Steps

### Phase A — Pilot audit: completed

The 63-frame priority audit, CVAT corrections, audited export, revision-safe import, manifest reconciliation, and regression validation are complete. Phase B has generated 84 proposed test images; every one must still be manually verified before the holdout is frozen.

**Gate result:** passed for the audited priority set. Two ambiguous/low-quality noodle examples remain intentionally unchanged rather than being guessed.

### Phase B — Leakage-safe baseline split: completed

Phase B was executed on 18 July 2026 with:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-baseline \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42 \
  --materialize hardlink
```

| Split | Images | Leakage groups | Boxes |
|-------|-------:|---------------:|------:|
| Train | 587 | 584 | 598 |
| Validation | 167 | 167 | 171 |
| Test candidate | 84 | 83 | 86 |

The output includes `data.yaml`, `split-manifest.jsonl`, `summary.json`, and `test-review-queue.jsonl`. Images are hard-linked to avoid duplicate storage; labels are copied so later dataset3 corrections cannot alter this baseline. The output is immutable by default and the CLI refuses to overwrite it.

Revalidation command:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-baseline \
  --validate-only
```

**Gate result:** passed. There are zero cross-split leakage groups, zero missing image/label pairs, all six object classes occur in validation and test, and an independent regeneration produced the same split-manifest hash.

### Phase C — Train and evaluate a pilot baseline: completed

Start with YOLO11n at 640 pixels rather than tuning immediately.

macOS Apple Silicon:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=data/dataset3-baseline/data.yaml \
  epochs=100 \
  imgsz=640 \
  seed=42 \
  device=mps
```

Linux HPC NVIDIA GPU — install with [`requirements-hpc.txt`](../requirements-hpc.txt) so PyTorch uses CUDA 12.4 wheels compatible with driver CUDA 12.8 (API `12080`). A newer CUDA wheel fails with `NVIDIA driver on your system is too old (found version 12080)`. Fix `data.yaml` `path:` to the cluster absolute path, transfer `data/dataset3-baseline/` with `rsync -aL` (images are hardlinks into dataset3), then train with `device=0`:

```bash
pip install -r requirements-hpc.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

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

Training completed on 18 July 2026. The best checkpoint was selected at epoch 85 with precision 0.928, recall 0.867, mAP50 0.925, and mAP50–95 0.689. Training used Ultralytics `8.4.100`; the full configuration, artifact hashes, per-class revalidation, confusion-matrix findings, and limitations are recorded in [`docs/experiments/dataset3_pilot_v1.md`](experiments/dataset3_pilot_v1.md).

The model is accepted as a CVAT assisted-labelling baseline, not as a production application model. Char Kuey Teow is the weakest class, the validation set has only 10 Nasi Lemak and 11 Roti Canai objects, and the test candidates remain unreviewed. Keep the checkpoint in its versioned run directory and do not replace the application's `best.pt` until a reviewed holdout evaluation is accepted.

**Gate result:** passed for pipeline and assisted-labelling use. Production promotion remains pending.

### Phase D — Use the baseline for assisted labelling

1. Connect the approved baseline to CVAT auto-annotation or generate prediction labels for import.
2. Process the remaining 4,461 images in batches of 300–500.
3. Prioritize underrepresented classes and low-confidence/high-disagreement images.
4. Require human correction; model predictions are proposals, not ground truth.
5. Export each batch, run a dry validation import, then apply it and archive the export/report.
6. Quarantine empty/non-target images and preserve every relabel transition.

**Gate per batch:** valid YOLO export, reviewed class mappings, rejected images quarantined, manifest totals reconciled, and regression tests passing.

### Phase E — Final split, training, and deployment

1. Freeze the final annotated manifest and preserve the untouched test groups.
2. Rebuild the leakage-safe split and train larger model variants only if YOLO11n is capacity-limited.
3. Tune confidence and IoU thresholds on validation data, never the test set.
4. Run final test-set evaluation and save metrics, confusion matrix, and representative failure cases.
5. Copy the accepted weights to `data/weights/best.pt`.
6. Restart FastAPI and test `/api/health`, `/api/classes`, and `/api/predict` end to end.

## 10. Immediate Recommended Action

Use the completed Phase C checkpoint to propose boxes in CVAT, with human correction required. Prioritize Char Kuey Teow, Nasi Lemak, and Roti Canai, plus noodle images on which Char Kuey Teow and Mee Goreng disagree. In parallel, manually verify the 84 entries in `data/dataset3-baseline/test-review-queue.jsonl`. If any test annotation changes, correct dataset3/CVAT and regenerate a new versioned split; do not edit the immutable baseline directly. Freeze the accepted test groups before running the final holdout evaluation exactly once.

## 11. Environment and Runtime

Local app / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Linux HPC GPU training:

```bash
python3 -m venv .venv-hpc
source .venv-hpc/bin/activate
pip install -U pip
pip install -r requirements-hpc.txt
```

`requirements-hpc.txt` pulls torch/torchvision from the CUDA 12.4 wheel index, then installs [`requirements.txt`](../requirements.txt). This avoids the driver mismatch seen when a newer CUDA build is installed on nodes with NVIDIA driver API `12080` (CUDA 12.8). Comments in `requirements.txt` document the same constraint.

Important environment variables include `MODEL_WEIGHTS_PATH`, `KNOWLEDGE_BASE_PATH`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `DEVICE`, `LLM_PROVIDER`, and the optional OpenAI/Gemini credentials.

## 12. Working Tree Note

At this handoff, the Phase A and Phase B implementation plus HPC install notes are represented by modifications to:

- `README.md`
- `docs/architecture.md`
- `docs/bounding-box-policy.md`
- `docs/experiments/dataset3_pilot_v1.md`
- `docs/handoff.md`
- `requirements.txt`
- `requirements-hpc.txt`
- `training_scripts/import_cvat_annotations.py`
- `training_scripts/split_dataset3.py`
- `tests/test_cvat_revision.py`
- `tests/test_dataset3_split.py`

The repository-wide documentation rule is recorded in `AGENTS.md`. Dataset outputs under `data/` are intentionally gitignored. Review and commit code/documentation deliberately; do not assume the gitignored data can be recreated without the local source datasets and archived CVAT exports.
