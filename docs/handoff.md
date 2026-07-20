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

- 5,198 usable images
- 2,798 annotated images
- 2,936 bounding boxes
- 2,400 images still missing bounding-box annotations
- 109 rejected images: 8 from the original pilot, 7 from batch 001, 13 from batch 002, 2 from the holdout audit, 11 from batch 003, 31 from batch 004, 10 from batch 005, and 27 from batch 006
- 5,144 usable leakage groups; 52 groups contain more than one usable image
- no custom `data/weights/best.pt` yet
- Phase C pilot checkpoint available at `runs/detect/dataset3_pilot_v1/weights/best.pt`
- interim v2 checkpoint available at `runs/detect/dataset3_interim_v2/weights/best.pt`
- interim v3 checkpoint available at `runs/detect/dataset3_interim_v3/weights/best.pt` (current assisted-labelling model)

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

`data/cvat/assisted-batch-004/` was reviewed in CVAT task `2442189`, completed
job `4262800`, and applied on 20 July 2026. Seed 46 selected 500 new leakage
groups with zero overlap against the 81 locked test groups and 1,251 prior CVAT
groups. The interim v2 checkpoint proposed 589 boxes on 494 images; human review
accepted 497 boxes on 469 images, quarantined 31 non-target frames, retained
three multi-class images, and recorded 61 primary-class corrections including
51 `mee_goreng` → `char_kuey_teow`. The guarded import created a recoverable
pre-merge backup and updated dataset3. Batches 003 and 004 together add enough
new labels for the next locked incremental split and interim HPC retrain.

The interim v3 retrain is complete. `data/dataset3-interim-v3/` (1,674 train /
418 validation / 82 locked test) was materialized from the interim v2 base
split, and `runs/detect/dataset3_interim_v3/` trained from the interim v2
checkpoint. Its best epoch reached validation mAP50 0.932 and mAP50–95 0.761;
Mee Goreng recall is the current weakness at 0.513. The checkpoint is accepted
for assisted-batch-005 proposals only, not production. Full metrics are in
[`experiments/dataset3_interim_v3.md`](experiments/dataset3_interim_v3.md).

`data/cvat/assisted-batch-005/` was reviewed in CVAT task `2442437`, completed
job `4263052`, and applied on 20 July 2026. Seed 47 selected 300 new leakage
groups with zero overlap against the 81 locked test groups and 1,751 prior CVAT
groups. The interim v3 checkpoint proposed 358 boxes on 299 images; human review
accepted 304 boxes on 290 images, quarantined 10 non-target frames, and recorded
43 primary-class corrections including 35 `mee_goreng` → `char_kuey_teow`.
Accepted Mee Goreng boxes remain only 4, so genuine Mee Goreng recruitment must
be prioritized in later batches. The guarded import created a recoverable
pre-merge backup and updated dataset3.

`data/cvat/assisted-batch-006/` was reviewed in CVAT task `2442499`, completed
job `4263273`, and applied on 20 July 2026. Seed 48 selected 361 new leakage
groups with zero overlap against the 81 locked test groups or 2,051 prior CVAT
selection groups, using raised Mee Goreng / Roti Canai / Laksa quotas and zero
Char Kuey Teow. The interim v3 checkpoint proposed 441 boxes on all 361 images;
human review accepted 365 boxes on 334 images, quarantined 27 non-target frames
(12 `laksa`, 11 `mee_goreng`, 4 `roti_canai`), and recorded 86 primary-class
corrections — overwhelmingly 81 `mee_goreng` → `char_kuey_teow`, plus 2
`mee_goreng` → `nasi_lemak`, 1 `mee_goreng` → `laksa`, 1 `chicken_rice` →
`nasi_lemak`, and 1 `laksa` → `char_kuey_teow`. One multi-class frame whose
source `mee_goreng` class was entirely absent from the reviewed boxes was
resolved with the new `--primary-class-override` importer flag, assigning
`char_kuey_teow` (the larger box). Combined with the 11 `mee_goreng` rejections,
93 of the 100 source Mee Goreng frames in this batch were mislabeled or empty,
leaving only 4 accepted Mee Goreng boxes. The guarded import created a
recoverable pre-merge backup and updated dataset3.

`data/cvat/assisted-batch-007/` is prepared for review. Seed 49 selected 500 new
leakage groups with zero overlap against the 81 locked test groups or 2,361 prior
CVAT selection groups (pilot + batches 001–006). Quotas favor remaining volume
classes (Laksa 130, Nasi Lemak 120, Roti Canai 120, Chicken Rice 70, Mee Goreng
60, Char Kuey Teow 0). The interim v3 checkpoint proposed 617 boxes on 497 of
500 images (only five Mee Goreng boxes). Web scraping is deferred until after
this remaining-folder review cycle. This package has not been imported into CVAT
or merged into Dataset3.

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
| 0 | `nasi_lemak` | 994 | 448 | 473 | 546 |
| 1 | `roti_canai` | 974 | 445 | 506 | 529 |
| 2 | `char_kuey_teow` | 726 | 725 | 742 | 1 |
| 3 | `chicken_rice` | 703 | 575 | 593 | 128 |
| 4 | `laksa` | 1,075 | 432 | 446 | 643 |
| 5 | `mee_goreng` | 726 | 173 | 176 | 553 |
| **Total** | — | **5,198** | **2,798** | **2,936** | **2,400** |

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
- The initial assembly produced 5,293 usable images in 5,253 leakage groups. Phase A restored six valid Laksa images, reaching 5,299 usable images. Phase D batches 001–006 quarantined 99 reviewed non-target frames; the holdout audit quarantined another two. The current state is 5,198 usable images in 5,144 usable groups. Any train/validation/test splitter must keep each group in exactly one split.

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
| CVAT batch preparation/import | Implemented; assisted batches 001–005 reviewed, validated, and applied |
| Group-aware annotated-only splitter | Implemented and validated |
| Custom six-class YOLO model | Interim v3 trained and validated; approved for assisted-batch proposals only |
| Production `data/weights/best.pt` | Missing |

Until custom weights are approved and copied to `data/weights/best.pt`, `/api/predict` is only a scaffold test and does not reliably identify the six Malaysian dishes.

## 8. Verification Completed

After the Phase A audited revision, Phase B split, Phase C pilot evaluation,
Phase D batch 001–006 merges, locked interim split, and interim HPC run:

- `manifest.jsonl` contains 5,307 records: 5,198 usable plus 109 rejected
- status totals are 2,798 annotated, 2,400 missing, and 109 rejected
- all 2,936 YOLO rows use canonical class IDs and valid normalized coordinates
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
- assisted batch 004 contains 500 images from 500 new leakage groups, with zero overlap against 81 locked test groups and 1,251 prior selection groups, and 589 proposals on 494 images
- CVAT task `2442189`, completed job `4262800`, reviewed those proposals into 497 accepted boxes on 469 images and 31 rejects
- batch 004 retained three multi-class images and recorded 61 primary-class corrections, including 51 `mee_goreng` → `char_kuey_teow`
- holdout task `2441672`, completed job `4262178`, reviewed all 84 frames into 82 accepted images with 84 boxes and 2 rejected non-target images
- holdout revision `test-holdout-v1` changed 55 images: 52 box-only adjustments, 2 rejections, and 1 `mee_goreng` → `char_kuey_teow` correction
- `dataset3-interim-v2` contains 1,067 train, 267 validation, and 82 reviewed test images with zero cross-split leakage
- all 754 surviving baseline train/validation images keep their original split and all 580 new annotations remain outside test
- the interim split-manifest SHA-256 is `8e9db98b57dd53f01778afe8d1b66bc4e07975f639356c9758226102eecd90dd`
- `dataset3-interim-v3` contains 1,674 train, 418 validation, and 82 reviewed test images with zero cross-split leakage
- all 1,334 surviving interim-v2 train/validation images keep their original split and all 758 new annotations remain outside test
- the interim-v3 split-manifest SHA-256 is `74560af5a330fde8005ae953552a17a2bb84abc419c76ac0a3418ddae0574091`
- interim v2 training stopped normally at epoch 95 after selecting epoch 75 by mAP50–95 fitness; its best metrics are precision 0.912, recall 0.868, mAP50 0.938, and mAP50–95 0.747
- interim v2 checkpoint SHA-256 is `0babcfc246b9af4c003277a4f50bc33c79f4a32b34c4473ee0aa0d36329f3705`
- interim v3 training stopped normally at epoch 21 after selecting epoch 1 by mAP50–95 fitness; its best metrics are precision 0.891, recall 0.859, mAP50 0.932, and mAP50–95 0.761
- interim v3 checkpoint SHA-256 is `5d6dde4b927c53dd1697ffbb759046608f492d47f2f7349452b7e01b3c5b8080`
- assisted batch 005 contains 300 images from 300 new leakage groups, with zero overlap against 81 locked test groups and 1,751 prior selection groups, and 358 proposals on 299 images
- CVAT task `2442437`, completed job `4263052`, reviewed those proposals into 304 accepted boxes on 290 images and 10 rejects
- batch 005 recorded 43 primary-class corrections, including 35 `mee_goreng` → `char_kuey_teow`, and retained only 4 accepted Mee Goreng boxes
- assisted batch 006 contains 361 images from 361 new leakage groups, with zero overlap against 81 locked test groups and 2,051 prior selection groups, and 441 proposals on 361 images
- CVAT task `2442499`, completed job `4263273`, reviewed those proposals into 365 accepted boxes on 334 images and 27 rejects
- batch 006 recorded 86 primary-class corrections, including 81 `mee_goreng` → `char_kuey_teow`, and used one `--primary-class-override` (`char_kuey_teow`) for a source `mee_goreng` frame whose original class was absent from the reviewed boxes; 93 of 100 source Mee Goreng frames were mislabeled or empty, retaining only 4 accepted Mee Goreng boxes
- assisted batch 007 contains 500 images from 500 new leakage groups, with zero overlap against 81 locked test groups and 2,361 prior selection groups, and 617 proposals on 497 images
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

### Interim HPC retraining v3: completed

`runs/detect/dataset3_interim_v3/` was trained from the interim v2 checkpoint on
`data/dataset3-interim-v3/` (1,674 train / 418 validation / 82 locked test),
image size 640, batch 16, seed 42, deterministic mode, and `resume: false`.
Because it started from an already-converged checkpoint on a larger validation
view, epoch 1 achieved the best mAP50–95 fitness; early stopping ended the run
normally at epoch 21.

| Run | Precision | Recall | mAP50 | mAP50–95 |
|-----|----------:|-------:|------:|----------:|
| Interim v2 best | 0.912 | 0.868 | 0.938 | 0.747 |
| Interim v3 best | 0.891 | 0.859 | 0.932 | 0.761 |

A validation-only local pass found per-class mAP50–95 of 0.810 Nasi Lemak,
0.728 Roti Canai, 0.783 Char Kuey Teow, 0.690 Chicken Rice, 0.822 Laksa, and
0.731 Mee Goreng. Mee Goreng recall dropped to 0.513 and is the current
weakness; it is also the most under-annotated class. The best epoch landing at
epoch 1 indicates the default `lr0=0.01` is too high for fine-tuning from a
converged checkpoint; lower it next retrain. The locked test split was not
accessed.

The checkpoint is approved for human-reviewed batch-005 proposals and remains
ineligible for `data/weights/best.pt`. Full details are in
[`docs/experiments/dataset3_interim_v3.md`](experiments/dataset3_interim_v3.md).

### Phase D — Use the baseline for assisted labelling: batches 001–006 applied

1. Connect the approved baseline to CVAT auto-annotation or generate prediction labels for import.
2. Process the remaining 2,400 images in batches of 300–500.
3. Prioritize underrepresented classes (especially genuine Mee Goreng) and low-confidence/high-disagreement images.
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

Batch 004 was generated on 20 July 2026 with seed 46 from
`runs/detect/dataset3_interim_v2/weights/best.pt`, confidence 0.20, IoU 0.50,
and CPU inference:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 100 | 123 |
| Roti Canai | 100 | 117 |
| Char Kuey Teow | 100 | 153 |
| Chicken Rice | 67 | 78 |
| Laksa | 67 | 86 |
| Mee Goreng | 66 | 32 |
| **Total** | **500** | **589** |

It has 494 images with proposals, six without proposals, and 390 high-priority
frames. The 500 selections use 500 unique leakage groups, with zero overlap
against the 81 locked test groups and 1,251 prior CVAT selection groups. Both
input archives pass ZIP integrity checks.

Reviewed outcome:

| Metric | Count |
|--------|------:|
| Labelled images | 469 |
| Rejected non-target images | 31 |
| Accepted boxes | 497 |
| Primary-class corrections | 61 |
| `mee_goreng` → `char_kuey_teow` | 51 |
| Multi-class images | 3 |

Accepted object boxes are 100 Nasi Lemak, 104 Roti Canai, 149 Char Kuey Teow,
68 Chicken Rice, 66 Laksa, and 10 Mee Goreng. Current artifacts are:

```text
data/cvat/assisted-batch-004/
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

- `images.zip`: `2b7b1bd4f21f402b0e181f7df42b35ffe6750893d6ce42ad1c39338f25ae00ef`
- `preannotations.zip`: `397b5eabb7578e204b50648c794efb940d3d52aab3d94bb66a4af2ccd16bd41a`
- `selection.jsonl`: `01cd4c4167c0fd678192dda182005b2f48d523a56a2133d6f0e3baf10348413e`
- `predictions.jsonl`: `749909f00c4692275de6da0a925dc1b6ea438b92b67e86b9bd863539c2e6ddd9`
- `summary.json`: `f1a5f36f14d2d8452a60c961a9c33ba5e725ba94d88e59f07d46ec7df12831a8`
- `cvat-reviewed-export.zip`: `306ef725a76302af8d64be9048d11d7b3062fff976dd9cd51152cf4c5383bd64`
- `merge-report.json`: `be98f680f7e3b1b3b4e16fa80d64c3265a6911f61240149a355655888557a368`

CVAT task `2442189` and completed job `4262800` contained all 500 images. The
reviewed export passed structural and guarded dry validation before the apply.
The importer archived the exact export, retained the pre-merge manifest,
summary, and README, moved the 31 rejects recoverably, and reconciled every
label and manifest path. Do not rerun the first-merge apply; later corrections
must use a unique `--revision-id`.

Batch 005 was generated on 20 July 2026 with seed 47 from
`runs/detect/dataset3_interim_v3/weights/best.pt`, confidence 0.20, IoU 0.50,
and MPS inference. It selects 300 missing-label images from 300 distinct leakage
groups using the same 60/60/60/40/40/40 source-class quotas:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 60 | 86 |
| Roti Canai | 60 | 95 |
| Char Kuey Teow | 60 | 94 |
| Chicken Rice | 40 | 39 |
| Laksa | 40 | 40 |
| Mee Goreng | 40 | 4 |
| **Total** | **300** | **358** |

It has 299 images with proposals, one without a proposal, and 239 high-priority
frames. The 300 selections use 300 unique leakage groups, with zero overlap
against the 81 locked test groups and 1,751 prior CVAT selection groups. Both
input archives pass ZIP integrity checks.

Reviewed outcome:

| Metric | Count |
|--------|------:|
| Labelled images | 290 |
| Rejected non-target images | 10 |
| Accepted boxes | 304 |
| Primary-class corrections | 43 |
| `mee_goreng` → `char_kuey_teow` | 35 |
| Multi-class images | 0 |

Accepted object boxes are 62 Nasi Lemak, 69 Roti Canai, 92 Char Kuey Teow,
36 Chicken Rice, 41 Laksa, and 4 Mee Goreng. Current artifacts are:

```text
data/cvat/assisted-batch-005/
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

- `images.zip`: `84b58abad29a6c1ac06649e1f24bc96e5ec52805aba866f65ce89c8b94d58de7`
- `preannotations.zip`: `425accb6d1e9e773b2c613a04ec48c407fc45d0d705ed7d0c46f88b7b9180b23`
- `selection.jsonl`: `f7bf954509ac7ba32e75054aabca128755d253dde3bfecd7123eb74fd52e86cc`
- `predictions.jsonl`: `fe75e973347b2e627cef224dcf28ece2c318c504a174c323bb6cba252277c71e`
- `summary.json`: `17c67a741b0c93ed3c48b0b6d614b1acef432799668c2e104e51bde28e815523`
- `cvat-reviewed-export.zip`: `1e9940052a5d52cf5b7f5c874bd7a5fe3162498751d4bfed0a50da348a7b73f1`
- `merge-report.json`: `87040f32db6ef0d3df9f2f7e3db3a8ac0ca68c9bc255019b49375bfcc777cc4a`

CVAT task `2442437` and completed job `4263052` contained all 300 images. The
reviewed export passed structural and guarded dry validation before the apply.
The importer archived the exact export, retained the pre-merge manifest,
summary, and README, moved the 10 rejects recoverably, and reconciled every
label and manifest path. Do not rerun the first-merge apply; later corrections
must use a unique `--revision-id`. Char Kuey Teow now has only 1 missing
annotation left in Dataset3; Mee Goreng remains severely under-annotated
relative to demand (169 annotated / 653 missing).

Batch 006 was generated on 20 July 2026 with seed 48 from
`runs/detect/dataset3_interim_v3/weights/best.pt`, confidence 0.20, IoU 0.50,
and MPS inference. Quotas were deliberately rebalanced away from Char Kuey Teow:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 60 | 95 |
| Roti Canai | 80 | 117 |
| Char Kuey Teow | 0 | 84 |
| Chicken Rice | 41 | 54 |
| Laksa | 80 | 86 |
| Mee Goreng | 100 | 5 |
| **Total** | **361** | **441** |

Char Kuey Teow proposal boxes still appear because the interim v3 model predicts
that class on other source folders. It has 361 images with proposals, zero
without proposals, and 272 high-priority frames. The 361 selections use 361
unique leakage groups, with zero overlap against the 81 locked test groups and
2,051 prior CVAT selection groups. CVAT task `2442499` and completed job
`4263273` contained all 361 images. The reviewed export passed structural and
guarded dry validation before the apply. Human review accepted 365 boxes on 334
images and quarantined 27 non-target frames (12 `laksa`, 11 `mee_goreng`, 4
`roti_canai`). The importer surfaced one multi-class frame whose source
`mee_goreng` class was entirely absent from the reviewed boxes; it was resolved
with `--primary-class-override <sha>=char_kuey_teow` (the larger of the two
boxes). Do not rerun the first-merge apply; later corrections must use a unique
`--revision-id`. Current artifacts are:

```text
data/cvat/assisted-batch-006/
├── images.zip
├── preannotations.zip
├── selection.jsonl
├── predictions.jsonl
├── summary.json
├── cvat-reviewed-export.zip
└── merge-report.json
```

SHA-256 values:

- `images.zip`: `df2d2161242a6cc906cae91d5d29e2b47c221f26556a450d557383e72c32ef69`
- `preannotations.zip`: `7cd0b8b424f826d56018e6aba3ebf9cb1ffafa35d8a0f0725ff4d01b0cdf2dee`
- `selection.jsonl`: `2f31025361a114cb0448994a6e62d3c37b7bebf3ae22e44b15f0e06b8ab93901`
- `predictions.jsonl`: `7e43939c3b1a5e65138e7287ce59106bf4f35fb1ec1cb57a37a8e91fad76afed`
- `summary.json`: `aa9ad2a0b09e876fbd205a8c28b90a1e6c39d017c3a225d607b6e6d621e57de4`
- `cvat-reviewed-export.zip`: `b7ffe0f860266848f11aae3eb1e8f59b0e6b38569361b00ed87230d956bd1dd4`
- `merge-report.json`: `11c806755761a84998a4f28373b2d3c3f2e00509ccd8eaf2696e1fcd6e5d5e31`

Batch 007 was generated on 20 July 2026 with seed 49 from
`runs/detect/dataset3_interim_v3/weights/best.pt`, confidence 0.20, IoU 0.50,
and MPS inference:

| Source class | Images | Proposal boxes of class |
|--------------|-------:|------------------------:|
| Nasi Lemak | 120 | 159 |
| Roti Canai | 120 | 182 |
| Char Kuey Teow | 0 | 46 |
| Chicken Rice | 70 | 87 |
| Laksa | 130 | 138 |
| Mee Goreng | 60 | 5 |
| **Total** | **500** | **617** |

It has 497 images with proposals, 3 without proposals, and 330 high-priority
frames. The 500 selections use 500 unique leakage groups, with zero overlap
against the 81 locked test groups and 2,361 prior CVAT selection groups. Both
input archives pass ZIP integrity checks. Human review and guarded import are
pending. Current artifacts are:

```text
data/cvat/assisted-batch-007/
├── images.zip
├── preannotations.zip
├── selection.jsonl
├── predictions.jsonl
└── summary.json
```

SHA-256 values:

- `images.zip`: `c878f8d7881afbc2e49fdfe24df973df139c064efbdb95ad667fa89e0b3f6cac`
- `preannotations.zip`: `cbb20e9730487ec75aed031bca7eb85b9006ad8a59e810712a9e0823d03b9ad5`
- `selection.jsonl`: `66a0280be91ff4cb736395287a48be251ec2ea3dff95a9e1f23f2ae47ed6e3f0`
- `predictions.jsonl`: `65ca1edcfd78549881d0c794e27bb70db1b913dc6ae1eb5a10e789f8611ea704`
- `summary.json`: `89933c0e55c76a9ca03c114d8c8ec64fc1be5a6551ccc9b60c3fc0a219c66723`

### Phase E — Final split, training, and deployment

1. Freeze the final annotated manifest and preserve the untouched test groups.
2. Rebuild the leakage-safe split and train larger model variants only if YOLO11n is capacity-limited.
3. Tune confidence and IoU thresholds on validation data, never the test set.
4. Run final test-set evaluation and save metrics, confusion matrix, and representative failure cases.
5. Copy the accepted weights to `data/weights/best.pt`.
6. Restart FastAPI and test `/api/health`, `/api/classes`, and `/api/predict` end to end.

## 10. Immediate Recommended Action

Review assisted batch 007 in CVAT, then import it. The package
`data/cvat/assisted-batch-007/` holds 500 interim-v3 proposal images (617 boxes
on 497 images, zero test/prior overlap). Create a task from `images.zip`, import
`preannotations.zip` as **Ultralytics YOLO Detection**, and review all 500
frames. Expect many source-folder corrections on Mee Goreng frames (only five
Mee Goreng boxes were proposed). Export to
`data/cvat/assisted-batch-007/cvat-reviewed-export.zip`, then run the guarded
importer:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/assisted-batch-007 \
  --archive data/cvat/assisted-batch-007/cvat-reviewed-export.zip \
  --batch-id cvat_assisted_batch_007 \
  --task-id <TASK_ID> \
  --job-id <JOB_ID>

# Only after inspecting the dry-run report:
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/assisted-batch-007 \
  --archive data/cvat/assisted-batch-007/cvat-reviewed-export.zip \
  --batch-id cvat_assisted_batch_007 \
  --task-id <TASK_ID> \
  --job-id <JOB_ID> \
  --apply
```

Batch 007 was prepared with:

```bash
python training_scripts/prepare_cvat_assisted_batch.py \
  --dataset-dir data/dataset3 \
  --split-manifest data/dataset3-interim-v3/split-manifest.jsonl \
  --output-dir data/cvat/assisted-batch-007 \
  --model runs/detect/dataset3_interim_v3/weights/best.pt \
  --seed 49 \
  --confidence 0.20 \
  --iou 0.50 \
  --device mps \
  --class-count laksa=130 \
  --class-count nasi_lemak=120 \
  --class-count roti_canai=120 \
  --class-count chicken_rice=70 \
  --class-count mee_goreng=60 \
  --class-count char_kuey_teow=0
```

Web scraping for genuine Mee Goreng is deferred until after this remaining-folder
review. After batch 007 (and optionally one more), decide whether to scrape, then
rebuild `dataset3-interim-v4` and retrain with `lr0=0.002`. Optionally delete
hosted CVAT tasks `2442437` and `2442499` after local archive checks.

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

The returned `runs/detect/dataset3_interim_v2/` and
`runs/detect/dataset3_interim_v3/` HPC artifacts are tracked in the repository.
This pass adds or updates:

- `training_scripts/import_cvat_annotations.py` (adds `--primary-class-override`)
- `README.md`
- `docs/architecture.md`
- `docs/handoff.md`

The validation-only diagnostic directories
`runs/detect/dataset3_interim_v2_local_val/` and
`runs/detect/dataset3_interim_v3_local_val/` are intentionally ignored. The
repository-wide documentation rule is recorded in `AGENTS.md`. Dataset outputs
under `data/` (including `data/cvat/assisted-batch-007/` and
`data/dataset3-interim-v3/`) are intentionally gitignored. Review and commit
documentation deliberately; do not assume gitignored data can be recreated
without the local source datasets and archived CVAT exports.
