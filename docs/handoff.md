# FoodSense-MY — Project Handoff

**Last updated:** 20 July 2026
**Repository:** [FoodSense-MY](https://github.com/jiale2004/FoodSense-MY)  
**Purpose:** Six-class Malaysian food object detection and nutritional advisory.

## 1. Current State

The FastAPI application and its static frontend are runnable. The pilot and
expanded interim six-class YOLO11n runs have completed, but neither has been
promoted to the application. The current inference fallback is still
Ultralytics `yolo11n.pt`, whose COCO classes are not suitable for the target
dishes.

Image acquisition and consolidation are paused. The canonical staging dataset is now `data/dataset3/`. It combines approved content from dataset1, dataset2, the manually curated two-class web run, and two Roboflow imports. Exact duplicates were collapsed, near-duplicate leakage groups were recorded, and the initial CVAT pilot plus its Phase A priority audit have been completed and merged.

Current dataset3 totals:

- 5,266 usable images
- 1,705 annotated images
- 1,770 bounding boxes
- 3,561 images still missing bounding-box annotations
- 41 rejected images: 8 from the original pilot, 7 from batch 001, 13 from batch 002, 2 from the holdout audit, and 11 from batch 003
- 5,212 usable leakage groups; 52 groups contain more than one usable image
- no custom `data/weights/best.pt` yet
- Phase C pilot checkpoint available at `runs/detect/dataset3_pilot_v1/weights/best.pt`
- interim proposal checkpoint available at `runs/detect/dataset3_interim_v2/weights/best.pt`

Phase B materialized the then-current 838 annotated images as the validated
`data/dataset3-baseline/` training view. Its 84 test candidates have now been
manually reviewed without model proposals, and revision `test-holdout-v1` has
been applied to Dataset3. This immutable baseline intentionally retains the
pre-review labels and pending queue. The new immutable
`data/dataset3-interim-v2/` view now preserves its surviving train/validation
assignments and locks the 82 accepted holdout images into test.

`data/cvat/test-holdout-review-v1/` contains the deterministic manual-review
package for those 84 candidates: 84 images, 86 existing human boxes, 83 leakage
groups, all six object classes, and no model proposals. The package reconciles
exactly with the baseline queue and the unchanged candidate labels in current
Dataset3. CVAT task `2441672`, completed job `4262178`, reviewed all 84 frames.
The applied outcome is 82 accepted images with 84 boxes, two quarantined
non-target images, 52 box-only adjustments, and one `mee_goreng` →
`char_kuey_teow` primary-class correction. The three earlier hosted tasks were deleted only
after their local archives passed integrity checks, leaving two of the free
account's three task slots available for rotating later batches.

Phase D is in progress. `data/cvat/assisted-batch-001/` contains a validated
300-image CVAT package and pilot-model pre-annotations. Former CVAT task
`2439970` and completed job `4260450` contained all 300 images; the hosted task
has been deleted after archival. Human review reduced the 338 proposals
to 309 accepted boxes on 293 images and rejected 7 non-target frames. The
reviewed export passed dry validation and was applied with a recoverable
pre-merge backup. Dataset3 now includes the batch 001 annotations.

`data/cvat/assisted-batch-002/` is also reviewed, validated, and applied. It has
300 new images, 326 proposed boxes on 296 images, zero candidate-test overlap,
and zero overlap with the 600 prior CVAT selection groups. Former CVAT task
`2441400`, completed job `4261934`, contained the complete batch and was deleted
after archival. Human review accepted 304
boxes on 287 images and rejected 13 non-target frames. The guarded import
created a recoverable pre-merge backup and updated dataset3.

`data/cvat/assisted-batch-003/` was prepared from the interim v2 checkpoint
(seed 45), reviewed in CVAT task `2441914`, completed job `4262530`, and
applied on 20 July 2026. It selected 300 new leakage groups with zero overlap
against the 81 locked test groups and 951 prior CVAT groups, proposed 342 boxes
on 298 images, and human review accepted 304 boxes on 289 images while
quarantining 11 non-target frames. The guarded import created a recoverable
pre-merge backup and updated dataset3.

Do not treat an empty YOLO label as an accepted background image. Images confirmed to contain none of the six classes are quarantined under `data/dataset3/rejected/` and marked `rejected` in the manifest.

Group members should follow
[`cvat-collaborator-guide.md`](cvat-collaborator-guide.md) for task creation,
ZIP and annotation upload, review, export, and reviewer handoff. Annotation
decisions remain governed by
[`bounding-box-policy.md`](bounding-box-policy.md).

## 2. Canonical Classes and Annotation Coverage

Class IDs are fixed and must not be reordered:

| ID | Class key | Usable images | Annotated images | Boxes | Missing annotations |
|---:|-----------|--------------:|-----------------:|------:|--------------------:|
| 0 | `nasi_lemak` | 994 | 228 | 242 | 766 |
| 1 | `roti_canai` | 986 | 217 | 247 | 769 |
| 2 | `char_kuey_teow` | 565 | 404 | 412 | 161 |
| 3 | `chicken_rice` | 712 | 436 | 442 | 276 |
| 4 | `laksa` | 1,094 | 264 | 269 | 830 |
| 5 | `mee_goreng` | 915 | 156 | 158 | 759 |
| **Total** | — | **5,266** | **1,705** | **1,770** | **3,561** |

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
- The initial assembly produced 5,293 usable images in 5,253 leakage groups. Phase A restored six valid Laksa images, reaching 5,299 usable images. Phase D batches 001–003 quarantined 31 reviewed non-target frames; the holdout audit quarantined another two. The current state is 5,266 usable images in 5,212 usable groups. Any train/validation/test splitter must keep each group in exactly one split.

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

Archived CVAT identifiers (the hosted task was deleted on 20 July 2026 after
local archive verification):

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
| [`prepare_cvat_assisted_batch.py`](../training_scripts/prepare_cvat_assisted_batch.py) | Select leakage-safe missing records, exclude test/prior-batch groups, run the pilot detector, and package CVAT images plus YOLO proposals |
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
| CVAT batch preparation/import | Implemented; assisted batches 001–003 reviewed, validated, and applied |
| Group-aware annotated-only splitter | Implemented and validated |
| Custom six-class YOLO model | Interim v2 trained and validated; approved for assisted-batch proposals only |
| Production `data/weights/best.pt` | Missing |

Until custom weights are approved and copied to `data/weights/best.pt`, `/api/predict` is only a scaffold test and does not reliably identify the six Malaysian dishes.

## 8. Verification Completed

After the Phase A audited revision, Phase B split, Phase C pilot evaluation,
Phase D batch 001–003 merges, locked interim split, and interim HPC run:

- `manifest.jsonl` contains 5,307 records: 5,266 usable plus 41 rejected
- status totals are 1,705 annotated, 3,561 missing, and 41 rejected
- all 1,770 YOLO rows use canonical class IDs and valid normalized coordinates
- every usable manifest path and annotation file is consistent
- the baseline contains 587 train, 167 validation, and 84 candidate test images
- Phase B validation reports zero cross-split leakage groups and zero missing pairs
- every object class is represented in validation and test
- the reproducible split-manifest hash is `f87d6f4ab07e463ddca111c4add9c5a6236acf4d08e0c0500b8802f1e45e7d1e`
- the best Phase C checkpoint is epoch 85 with precision 0.928, recall 0.867, mAP50 0.925, and mAP50–95 0.689
- the checkpoint's six IDs match the canonical class order
- the pilot artifacts and evaluation are recorded in [`docs/experiments/dataset3_pilot_v1.md`](experiments/dataset3_pilot_v1.md)
- assisted batch 001 contains 300 images from 300 unique leakage groups with zero candidate-test overlap
- its 0.20-confidence proposals contained 338 boxes on 299 images; human review accepted 309 boxes on 293 images and rejected 7 frames
- batch 001 recorded 26 `mee_goreng` → `char_kuey_teow` and 4 reverse primary-class corrections
- assisted batch 002 contains 300 images from 300 new leakage groups, with zero test/prior-selection overlap and 326 proposals on 296 images
- former CVAT task `2441400`, completed job `4261934`, reviewed 326 proposals into 304 accepted boxes on 287 images and 13 rejects before archival and hosted deletion
- batch 002 retained two multi-class images and recorded 36 primary-class corrections, including 28 `mee_goreng` → `char_kuey_teow`
- assisted batch 003 contains 300 images from 300 new leakage groups, with zero overlap against 81 locked test groups and 951 prior CVAT groups, and 342 proposals on 298 images
- CVAT task `2441914`, completed job `4262530`, reviewed those proposals into 304 accepted boxes on 289 images and 11 rejects
- batch 003 retained one multi-class image and recorded 30 primary-class corrections, including 26 `mee_goreng` → `char_kuey_teow`
- holdout task `2441672`, completed job `4262178`, reviewed all 84 frames into 82 accepted images with 84 boxes and 2 rejected non-target images
- holdout revision `test-holdout-v1` changed 55 images: 52 box-only adjustments, 2 rejections, and 1 `mee_goreng` → `char_kuey_teow` correction
- `dataset3-interim-v2` contains 1,067 train, 267 validation, and 82 reviewed test images with zero cross-split leakage
- all 754 surviving baseline train/validation images keep their original split and all 580 new annotations remain outside test
- the interim split-manifest SHA-256 is `8e9db98b57dd53f01778afe8d1b66bc4e07975f639356c9758226102eecd90dd`
- interim training stopped normally at epoch 95 after selecting epoch 75 by mAP50–95 fitness
- interim best metrics are precision 0.912, recall 0.868, mAP50 0.938, and mAP50–95 0.747
- interim checkpoint SHA-256 is `0babcfc246b9af4c003277a4f50bc33c79f4a32b34c4473ee0aa0d36329f3705`
- the locked test split was not evaluated
- all seventeen repository regression tests pass, including locked incremental assignment, assisted selection, holdout packaging, README reconciliation, batch-path safety, revision, and split-integrity coverage
- `git diff --check` passes

Run the current test suite with:

```bash
python -m unittest discover -s tests -v
```

## 9. Recommended Next Steps

### Phase A — Pilot audit: completed

The 63-frame priority audit, CVAT corrections, audited export, revision-safe import, manifest reconciliation, and regression validation are complete. Phase B generated 84 proposed test images; their later no-proposal holdout audit is complete and applied, and the 82 accepted images are now locked in `dataset3-interim-v2`.

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

After the Phase D assisted-batch merges, source-hash validation against current
dataset3 is expected to fail because `dataset3-baseline` is an immutable
pre-Phase-D snapshot. Its own split manifest remains unchanged at the hash
above. Do not overwrite it; use a new output directory for a later split.

### Test-holdout manual verification: reviewed and applied

[`training_scripts/prepare_test_holdout_review.py`](../training_scripts/prepare_test_holdout_review.py)
reconciles the test split, pending queue, and current Dataset3 manifest before
creating a no-proposal CVAT package:

```bash
python training_scripts/prepare_test_holdout_review.py
```

The current output is:

```text
data/cvat/test-holdout-review-v1/
├── images.zip
├── current-annotations.zip
├── selection.jsonl
├── review-queue.jsonl
├── cvat-task.json
├── cvat-reviewed-export.zip
├── rejected.jsonl
├── revisions/test-holdout-v1/
│   ├── cvat-export.zip
│   ├── report.json
│   └── pre-apply/
└── summary.json
```

Validated content:

| Metric | Count |
|--------|------:|
| Images | 84 |
| Existing human boxes | 86 |
| Leakage groups | 83 |
| Nasi Lemak boxes | 5 |
| Roti Canai boxes | 6 |
| Char Kuey Teow boxes | 15 |
| Chicken Rice boxes | 32 |
| Laksa boxes | 15 |
| Mee Goreng boxes | 13 |

Artifact SHA-256 values:

- `images.zip`: `b9135f9d73cb7bffcb1ee3ba94ab2c811a0908d5aea73ef82e82e6f8d1f3c82d`
- `current-annotations.zip`: `e2b14ce3c62fabc401e3dd712bf6bb65bf7411c170cb548fe58a34c94c6ea566`
- `selection.jsonl`: `867de00db16c989c4ac41ac57001d7f769bfc7174935e2a273d8e6fc42ea2b5f`
- `review-queue.jsonl`: `fbaab330673d9c801653cf6bf71105c699c830274116a8588247accb4dd3c8af`
- `cvat-task.json`: `196ed0b79834ca0b6ed038485f4af8ef9a00888504657eff0875ebac2bc5d52a`
- `summary.json`: `461e5688f78b88102e1885bcc614c58513d218270036b4a559dd79be64acae79`
- `cvat-reviewed-export.zip`: `76d1f5f3215f2f4cb35e9d81d8a669fde516d1e301834c63e18301118dd6e89e`
- `revisions/test-holdout-v1/report.json`: `88ff7a112d68d356b1b6f3c508fb227fdb88ba671eb4a38c8ac9a5fcf8b2c658`

The packager validates image hashes, label hashes, YOLO rows, box counts,
queue membership, pending status, and leakage-group identity. It refuses an
existing output directory. CVAT project `425516` task `2441672`, job `4262178`,
was created from `images.zip`, and `current-annotations.zip` was imported as
**Ultralytics YOLO Detection**. CVAT's initial job statistics matched the
package exactly: 84 frames and 86 rectangles with per-class totals 5, 6, 15,
32, 15, and 13 in canonical ID order. Human review completed all frames without
model proposals and produced 82 labelled images with 84 boxes. Two frames were
confirmed non-target, 52 other frames received box-only adjustments, one
`mee_goreng` image became `char_kuey_teow`, and one export differed only by row
order.

The reviewed export passed a dry-run semantic comparison and the same command
was then run with `--apply`:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/test-holdout-review-v1 \
  --archive data/cvat/test-holdout-review-v1/cvat-reviewed-export.zip \
  --revision-id test-holdout-v1 \
  --task-id 2441672 \
  --job-id 4262178
```

The unique revision ID created recoverable pre-apply metadata and label backups
before updating Dataset3. The immutable Phase B baseline was not edited.

### Locked incremental split: completed

The splitter now has a backward-compatible locked incremental mode. It requires
the immutable Phase B split manifest and reviewed selection together, rejects a
partial or drifted selection, excludes current rejected records, preserves
surviving base train/validation groups, forces accepted reviewed groups into
test, and assigns only new groups between train and validation.

The immutable output was generated with:

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

| Split | Images | Leakage groups | Boxes |
|-------|-------:|---------------:|------:|
| Train | 1,067 | 1,064 | 1,106 |
| Validation | 267 | 267 | 276 |
| Reviewed test | 82 | 81 | 84 |

Validation reports zero missing pairs, zero cross-split leakage, all six object
classes in validation and test, 82 accepted test queue records, and no pending
test records. Assignment reconciliation confirms 754 preserved baseline
train/validation images, 580 new train/validation-only images, and exact test-ID
equality with the accepted holdout.

Artifact SHA-256 values:

- `split-manifest.jsonl`: `8e9db98b57dd53f01778afe8d1b66bc4e07975f639356c9758226102eecd90dd`
- `test-review-queue.jsonl`: `6f42bdc23ba014063b466bc66bf88eb4ffbe1e3f2aa7858d3cd524338ba1929e`
- `summary.json`: `a1e46088a9617813f6216051af6594575ef7ce0fd3c2280a12be0e5c03c582a3`

Revalidate without modification using:

```bash
python training_scripts/split_dataset3.py \
  --dataset-dir data/dataset3 \
  --output-dir data/dataset3-interim-v2 \
  --validate-only
```

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

The completed pilot used Linux HPC NVIDIA GPU training with
[`requirements-hpc.txt`](../requirements-hpc.txt), CUDA 12.4 wheels compatible
with driver CUDA 12.8 (API `12080`), the baseline transferred using `rsync
-aL`, and `device=0`:

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

The model is accepted as a CVAT assisted-labelling baseline, not as a production application model. Char Kuey Teow is the weakest class, and the validation set has only 10 Nasi Lemak and 11 Roti Canai objects. The accepted holdout groups are now locked in `dataset3-interim-v2` but remain unevaluated. Keep the checkpoint in its versioned run directory and do not replace the application's `best.pt` until annotation freeze and an accepted final holdout evaluation.

**Gate result:** passed for pipeline and assisted-labelling use. Production promotion remains pending.

### Interim HPC retraining: completed

`runs/detect/dataset3_interim_v2/` was trained from the Phase C checkpoint on
the expanded locked split with Ultralytics `8.4.100`, image size 640, batch 16,
seed 42, deterministic mode, and `resume: false`. Epoch 75 achieved the best
mAP50–95 fitness; early stopping ended the run normally at epoch 95 after 20
epochs without improvement.

| Run | Precision | Recall | mAP50 | mAP50–95 |
|-----|----------:|-------:|------:|----------:|
| Phase C pilot best | 0.928 | 0.867 | 0.925 | 0.689 |
| Interim v2 best | 0.912 | 0.868 | 0.938 | 0.747 |

The expanded validation view is not identical to the pilot view, so the
comparison is directional. A validation-only local pass found mAP50–95 of
0.830 Nasi Lemak, 0.694 Roti Canai, 0.704 Char Kuey Teow, 0.664 Chicken Rice,
0.834 Laksa, and 0.766 Mee Goreng. Noodle-class confusion remains the primary
semantic error, while Chicken Rice's high mAP50 but lower mAP50–95 points to
box-localization error. The locked test split was not accessed.

The checkpoint is approved for human-reviewed batch-003 proposals. It remains
ineligible for `data/weights/best.pt`. Full configuration, hashes, per-class
metrics, and confusion findings are recorded in
[`docs/experiments/dataset3_interim_v2.md`](experiments/dataset3_interim_v2.md).

### Phase D — Use the baseline for assisted labelling: batches 001–003 applied

1. Connect the approved baseline to CVAT auto-annotation or generate prediction labels for import.
2. Process the remaining 3,561 images in batches of 300–500.
3. Prioritize underrepresented classes and low-confidence/high-disagreement images.
4. Require human correction; model predictions are proposals, not ground truth.
5. Export each batch, run a dry validation import, then apply it and archive the export/report.
6. Quarantine empty/non-target images and preserve every relabel transition.

**Gate per batch:** valid YOLO export, reviewed class mappings, rejected images quarantined, manifest totals reconciled, and regression tests passing.

Batch 001 was generated on 19 July 2026 using seed 43, image size 640,
confidence 0.20, and IoU 0.50. Selection intentionally prioritizes the weakest
or underrepresented classes:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 60 | 70 |
| Roti Canai | 60 | 69 |
| Char Kuey Teow | 60 | 80 |
| Chicken Rice | 40 | 49 |
| Laksa | 40 | 48 |
| Mee Goreng | 40 | 22 |
| **Total** | **300** | **338** |

The source class and proposal class are deliberately recorded separately.
`predictions.jsonl` retains confidence and review reasons, while the CVAT
archive contains five-column YOLO rows without treating confidence as ground
truth. Structural dry-run validation and the guarded apply both succeeded.

Reviewed outcome:

| Metric | Count |
|--------|------:|
| Labelled images | 293 |
| Rejected non-target images | 7 |
| Accepted boxes | 309 |
| `mee_goreng` → `char_kuey_teow` | 26 |
| `char_kuey_teow` → `mee_goreng` | 4 |
| Multi-class images | 1 |

Accepted object boxes are 63 Nasi Lemak, 68 Roti Canai, 82 Char Kuey Teow,
43 Chicken Rice, 39 Laksa, and 14 Mee Goreng. Important artifacts:

```text
data/cvat/assisted-batch-001/
├── images.zip          # upload when creating the CVAT task
├── preannotations.zip  # import as Ultralytics YOLO Detection annotations
├── cvat-reviewed-export.zip # human-reviewed CVAT export
├── cvat-export.zip     # importer-archived applied export
├── selection.jsonl     # immutable source/provenance records
├── predictions.jsonl   # confidences and review-priority reasons
├── rejected.jsonl      # seven quarantined-frame records
├── merge-report.json   # applied outcome and post-merge counts
├── pre-merge/          # recoverable dataset3 metadata backup
└── summary.json        # configuration and proposal counts
```

Archive SHA-256 values:

- `images.zip`: `d546984fe8d54df0108ac3c4ddcb90835a4a057e21ac8583bb73ba2db22d9e75`
- `preannotations.zip`: `f26afa11a9f83b2065713d60b204c5fe503568685cb0660adc94f937ad1056b9`
- `cvat-reviewed-export.zip`: `808152b054ccc3b921c9dc07aa1d991d88d2a7549dcff0a678f39291ce6e4aa6`
- `merge-report.json`: `16f7cdbddc19f9deffe48cbe74f915a0e97b526815aaeed609096209b41decf4`

The following commands were executed in order. They are retained as provenance,
not as a command to rerun against the now-annotated selection:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/assisted-batch-001 \
  --archive data/cvat/assisted-batch-001/cvat-reviewed-export.zip \
  --batch-id cvat_assisted_batch_001 \
  --task-id 2439970 \
  --job-id 4260450

# Only after inspecting the dry-run report:
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/assisted-batch-001 \
  --archive data/cvat/assisted-batch-001/cvat-reviewed-export.zip \
  --batch-id cvat_assisted_batch_001 \
  --task-id 2439970 \
  --job-id 4260450 \
  --apply
```

Any later correction to batch 001 must use a new `--revision-id`; do not run a
second first-merge apply.

Batch 002 was generated on 20 July 2026 with seed 44 and otherwise identical
selection/inference settings:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 60 | 72 |
| Roti Canai | 60 | 63 |
| Char Kuey Teow | 60 | 89 |
| Chicken Rice | 40 | 46 |
| Laksa | 40 | 39 |
| Mee Goreng | 40 | 17 |
| **Total** | **300** | **326** |

It had 296 images with proposals, four without proposals, and 231 high-priority
frames. The 300 selections use 300 unique leakage groups, with zero overlap
against the 83 candidate-test groups and 600 prior CVAT groups. Both input
archives pass ZIP integrity checks.

Reviewed outcome:

| Metric | Count |
|--------|------:|
| Labelled images | 287 |
| Rejected non-target images | 13 |
| Accepted boxes | 304 |
| Primary-class corrections | 36 |
| `mee_goreng` → `char_kuey_teow` | 28 |
| Multi-class images | 2 |

Accepted object boxes are 65 Nasi Lemak, 60 Roti Canai, 89 Char Kuey Teow,
43 Chicken Rice, 38 Laksa, and 9 Mee Goreng. Current artifacts are:

```text
data/cvat/assisted-batch-002/
├── images.zip
├── preannotations.zip
├── cvat-reviewed-export.zip
├── cvat-export.zip
├── selection.jsonl
├── predictions.jsonl
├── rejected.jsonl
├── merge-report.json
├── pre-merge/
└── summary.json
```

SHA-256 values:

- `images.zip`: `66b79cfdcea5143f81e04fa373b8f5cd834ccdfd1b4808394e0342526b93a8d7`
- `preannotations.zip`: `bd4be995a97942deb6a1540843ed37b3a68c55193cd6f6b803e80283d5905e30`
- `selection.jsonl`: `d544ee37b0d57e230d713d3cb91c33ba9da6e248633daa9566acee40fe26dd70`
- `predictions.jsonl`: `914d53e5be2471fb25a95675a2c6167f0731cda2c2e19797aece655dfb47835d`
- `summary.json`: `1147c65466fc55da2acc2fead2423582073bc68939389ce6dff97a087e6839a5`
- `cvat-reviewed-export.zip`: `05c841a65aac8477d7cefadcf516718989fbc37b4b95eac0be1cb2faedff75fa`
- `merge-report.json`: `25c6867a09afc5b45c15828a099f7f7f9fcdcaa4aaf015210a91fd433c114566`

Former CVAT task `2441400` and completed job `4261934` contained all 300 images.
The hosted task was deleted on 20 July 2026 after local archive validation. The
reviewed export passed structural and guarded dry validation before the apply.
The importer archived the exact export, retained the pre-merge manifest,
summary, and README, moved the 13 rejects recoverably, and reconciled every
label and manifest path. Do not rerun the first-merge apply; later corrections
must use a unique `--revision-id`.

Batch 003 was generated on 20 July 2026 with seed 45 from
`runs/detect/dataset3_interim_v2/weights/best.pt` and otherwise identical
selection/inference settings:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 60 | 68 |
| Roti Canai | 60 | 71 |
| Char Kuey Teow | 60 | 88 |
| Chicken Rice | 40 | 42 |
| Laksa | 40 | 48 |
| Mee Goreng | 40 | 25 |
| **Total** | **300** | **342** |

It had 298 images with proposals, two without proposals, and 213 high-priority
frames. The 300 selections use 300 unique leakage groups, with zero overlap
against the 81 locked test groups and 951 prior CVAT groups. Both input
archives pass ZIP integrity checks.

Reviewed outcome:

| Metric | Count |
|--------|------:|
| Labelled images | 289 |
| Rejected non-target images | 11 |
| Accepted boxes | 304 |
| Primary-class corrections | 30 |
| `mee_goreng` → `char_kuey_teow` | 26 |
| Multi-class images | 1 |

Accepted object boxes are 62 Nasi Lemak, 63 Roti Canai, 88 Char Kuey Teow,
41 Chicken Rice, 41 Laksa, and 9 Mee Goreng. Current artifacts are:

```text
data/cvat/assisted-batch-003/
├── images.zip
├── preannotations.zip
├── cvat-reviewed-export.zip
├── cvat-export.zip
├── selection.jsonl
├── predictions.jsonl
├── rejected.jsonl
├── merge-report.json
├── pre-merge/
└── summary.json
```

SHA-256 values:

- `images.zip`: `8405d32b0173e989bb3f2d4ab66bafa6aa914a6740f619f248b6a49a94645e18`
- `preannotations.zip`: `efbf170a263a7eac82ba40cf1e8dd0d13d93055540623d8ac044c48e1fd0f713`
- `selection.jsonl`: `4faf9eff7e709bc5564915675002ac398bbc0c0afa1c4844b12782df5200b3d9`
- `predictions.jsonl`: `544479fdb38ff17c51f7a61f87331b4a95bce1b7c639c9378cb143a899a4959d`
- `summary.json`: `f90a40786064c8018c9238e387148b0eaed579c54af757da88ab61254051936b`
- `cvat-reviewed-export.zip`: `5df6061b5d34beb971aacb05d53dad5b564372e25c7ada6461885b0ea2ec9c2b`
- `merge-report.json`: `84142c259803ed3222e1e67d0ca8be7ee3e597658e32a5220a9ad4f33e8f4cd6`

CVAT task `2441914` and completed job `4262530` contained all 300 images. The
reviewed export passed structural and guarded dry validation before the apply.
The importer archived the exact export, retained the pre-merge manifest,
summary, and README, moved the 11 rejects recoverably, and reconciled every
label and manifest path. Do not rerun the first-merge apply; later corrections
must use a unique `--revision-id`.

### Phase E — Final split, training, and deployment

1. Freeze the final annotated manifest and preserve the untouched test groups.
2. Rebuild the leakage-safe split and train larger model variants only if YOLO11n is capacity-limited.
3. Tune confidence and IoU thresholds on validation data, never the test set.
4. Run final test-set evaluation and save metrics, confusion matrix, and representative failure cases.
5. Copy the accepted weights to `data/weights/best.pt`.
6. Restart FastAPI and test `/api/health`, `/api/classes`, and `/api/predict` end to end.

## 10. Immediate Recommended Action

Prepare assisted batch 004 from the accepted interim checkpoint, continuing the
seed sequence at 46 and using the locked interim manifest for test-group
exclusion:

```bash
python training_scripts/prepare_cvat_assisted_batch.py \
  --dataset-dir data/dataset3 \
  --split-manifest data/dataset3-interim-v2/split-manifest.jsonl \
  --output-dir data/cvat/assisted-batch-004 \
  --model runs/detect/dataset3_interim_v2/weights/best.pt \
  --seed 46 \
  --confidence 0.20 \
  --iou 0.50 \
  --device mps
```

Use `--device cpu` if the local Ultralytics build cannot execute MPS inference.
Before CVAT upload, verify 300 new unique leakage groups, zero overlap with the
81 locked test groups or prior selections, ZIP integrity, canonical class IDs,
and proposal counts. Human reviewers must inspect every frame, with extra
attention to Char Kuey Teow/Mee Goreng disagreements and Chicken Rice box
tightness. After the local batch-003 archive passes integrity checks, the hosted
CVAT task `2441914` may be deleted to free a task slot.

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
pip install "ultralytics==8.4.100"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements-hpc.txt` pulls torch/torchvision from the CUDA 12.4 wheel index, then installs [`requirements.txt`](../requirements.txt). This avoids the driver mismatch seen when a newer CUDA build is installed on nodes with NVIDIA driver API `12080` (CUDA 12.8). Comments in `requirements.txt` document the same constraint.

Important environment variables include `MODEL_WEIGHTS_PATH`, `KNOWLEDGE_BASE_PATH`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `DEVICE`, `LLM_PROVIDER`, and the optional OpenAI/Gemini credentials.

## 12. Working Tree Note

The returned `runs/detect/dataset3_interim_v2/` HPC artifacts are tracked in
the repository. This validation pass adds or updates:

- `.gitignore`
- `README.md`
- `docs/architecture.md`
- `docs/experiments/dataset3_interim_v2.md`
- `docs/handoff.md`

The validation-only diagnostic directory
`runs/detect/dataset3_interim_v2_local_val/` is intentionally ignored. The
repository-wide documentation rule is recorded in `AGENTS.md`. Dataset outputs
under `data/` are intentionally gitignored. Review and commit documentation
deliberately; do not assume gitignored data can be recreated without the local
source datasets and archived CVAT exports.
