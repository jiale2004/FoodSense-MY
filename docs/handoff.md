# FoodSense-MY — Project Handoff

**Last updated:** 18 July 2026  
**Repository:** [FoodSense-MY](https://github.com/jiale2004/FoodSense-MY)  
**Purpose:** Six-class Malaysian food object detection and nutritional advisory.

## 1. Current State

The FastAPI application and its static frontend are runnable, but the custom six-class detector has not yet been trained. The current inference fallback is Ultralytics `yolo11n.pt`, whose COCO classes are not suitable for the target dishes.

Image acquisition and consolidation are paused. The canonical staging dataset is now `data/dataset3/`. It combines approved content from dataset1, dataset2, the manually curated two-class web run, and two Roboflow imports. Exact duplicates were collapsed, near-duplicate leakage groups were recorded, and an initial CVAT pilot has been completed and merged.

Current dataset3 totals:

- 5,293 usable images
- 832 annotated images
- 847 bounding boxes
- 4,461 images still missing bounding-box annotations
- 14 CVAT pilot images rejected because none of the six target classes was present
- 5,253 leakage groups; 52 groups contain more than one image
- no custom `data/weights/best.pt` yet

Do not treat an empty YOLO label as an accepted background image. Images confirmed to contain none of the six classes are quarantined under `data/dataset3/rejected/` and marked `rejected` in the manifest.

## 2. Canonical Classes and Annotation Coverage

Class IDs are fixed and must not be reordered:

| ID | Class key | Usable images | Annotated images | Boxes | Missing annotations |
|---:|-----------|--------------:|-----------------:|------:|--------------------:|
| 0 | `nasi_lemak` | 996 | 50 | 51 | 946 |
| 1 | `roti_canai` | 995 | 46 | 55 | 949 |
| 2 | `char_kuey_teow` | 492 | 151 | 151 | 341 |
| 3 | `chicken_rice` | 709 | 313 | 316 | 396 |
| 4 | `laksa` | 1,093 | 143 | 144 | 950 |
| 5 | `mee_goreng` | 1,008 | 129 | 130 | 879 |
| **Total** | — | **5,293** | **832** | **847** | **4,461** |

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
- The resulting 5,293 images form 5,253 leakage groups. Any train/validation/test splitter must keep each group in exactly one split.

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

Manual annotation outcome:

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

The pilot also exposed meaningful source-class errors:

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
└── pre-merge/
```

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
| [`import_cvat_annotations.py`](../training_scripts/import_cvat_annotations.py) | Validate a CVAT Ultralytics-YOLO export, merge labels, and quarantine empty/rejected frames |
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
| Group-aware annotated-only splitter | Not implemented |
| Custom six-class YOLO model | Not trained |
| Production `data/weights/best.pt` | Missing |

Until custom weights are approved and copied to `data/weights/best.pt`, `/api/predict` is only a scaffold test and does not reliably identify the six Malaysian dishes.

## 8. Verification Completed

After the pilot merge:

- `manifest.jsonl` contains 5,307 records: 5,293 usable plus 14 rejected
- status totals are 832 annotated, 4,461 missing, and 14 rejected
- all 847 YOLO rows use canonical class IDs and valid normalized coordinates
- every usable manifest path and annotation file is consistent
- all seven repository regression tests pass
- `git diff --check` passes

Run the current test suite with:

```bash
python -m unittest discover -s tests -v
```

## 9. Recommended Next Steps

### Phase A — Freeze and audit the pilot

1. Review a stratified sample of at least 60 pilot images in CVAT, with extra attention to Char Kuey Teow versus Mee Goreng and boxes containing multiple dishes.
2. Check box tightness, integral side dishes, occluded instances, and whether all target objects in each reviewed frame were labelled.
3. Correct any policy inconsistencies in CVAT and re-export/re-import the same task.
4. Freeze a manually verified test subset before assisted labelling so auto-labelled images cannot leak into final evaluation.

**Gate:** zero class-ID errors, no known missing target boxes, and consistent application of the bounding-box policy.

### Phase B — Build a leakage-safe baseline split

Implement a dataset3-specific splitter that:

- includes only `annotation_status == "annotated"` images;
- groups by `leakage_group` before assigning splits;
- uses a deterministic seed and approximately 70/20/10 train/validation/test ratios;
- balances class/object coverage as far as group constraints allow;
- writes immutable split manifests, YOLO folders, and `data.yaml`;
- fails if a leakage group crosses splits, a label is missing, or a class is absent from validation/test.

Do not use the current `prepare_dataset.py` unchanged: it performs a random file split, does not understand the dataset3 class layout or manifest status, and does not protect leakage groups.

**Gate:** 0 cross-split leakage groups, 0 missing label pairs, all six classes represented, and repeatable split hashes.

### Phase C — Train and evaluate a pilot baseline

Start with YOLO11n at 640 pixels rather than tuning immediately:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=data/dataset3-baseline/data.yaml \
  epochs=100 \
  imgsz=640 \
  seed=42 \
  device=mps
```

Evaluate per-class precision, recall, AP50, AP50–95, confusion matrix, and a qualitative error gallery. Overall mAP alone will hide the current imbalance: Nasi Lemak and Roti Canai have only 50 and 46 annotated images, while Chicken Rice has 313.

Save the first approved experiment under a versioned filename such as `data/weights/foodsense_dataset3_pilot_v1.pt`. Do not replace the application's `best.pt` until evaluation is accepted.

**Gate:** inference works for all six canonical IDs, no pipeline mapping errors, and baseline results are recorded for comparison.

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

The next implementation should be the annotated-only, leakage-group-aware splitter, followed by one baseline training run on the 832 currently annotated images. That baseline is not expected to be production quality; its purpose is to validate class mappings, expose annotation-policy problems, and accelerate the remaining CVAT work.

## 11. Environment and Runtime

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Important environment variables include `MODEL_WEIGHTS_PATH`, `KNOWLEDGE_BASE_PATH`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `DEVICE`, `LLM_PROVIDER`, and the optional OpenAI/Gemini credentials.

## 12. Working Tree Note

At this handoff, `.env.example` is modified and the four dataset utilities listed below are untracked:

- `training_scripts/build_dataset3.py`
- `training_scripts/import_cvat_annotations.py`
- `training_scripts/import_roboflow_subset.py`
- `training_scripts/prepare_cvat_pilot.py`

Dataset outputs under `data/` are intentionally gitignored. Review and commit code/documentation deliberately; do not assume the gitignored data can be recreated without the local source datasets and archived CVAT exports.
