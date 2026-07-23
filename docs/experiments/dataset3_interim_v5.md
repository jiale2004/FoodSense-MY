# Dataset3 YOLO11n Interim — `dataset3_interim_v5`

**Split materialized:** 22 July 2026
**Training completed:** 22 July 2026
**Thresholds calibrated:** 23 July 2026 (conf 0.47 / NMS-IoU 0.45)
**Locked-test evaluated:** 23 July 2026 (mAP50 0.926, mAP50–95 0.678)
**Assessment:** production-approved detector; promoted to `data/weights/best.pt`

## Split configuration and artifacts

- source staging: `data/dataset3/` (5,246 annotated images / 5,579 boxes)
- output: `data/dataset3-interim-v5/`
- base split: `data/dataset3-interim-v4/split-manifest.jsonl`
  (SHA-256 `c03a554400fe8298a5cca2fe707bbaf32825b80179d6e5c0b11c4d18c3a12700`
  after the Ultralytics JPEG identity migration; see below)
- locked test selection: `data/cvat/test-holdout-review-v1/accepted-selection.jsonl`
  (SHA-256 `2094c4915d282ef9580ca2d1cd5da3464bd2f530d3a50674e4e444aba536108a`)
- algorithm: `dataset3-group-stratified-locked-v2`
- incremental train fraction: 0.8
- seed: 42
- materialization: hardlink images / copy labels
- split: 4,131 train, 1,033 validation, 82 locked reviewed test images
- boxes: 4,391 train, 1,104 validation, 84 test
- cross-split leakage groups: 0
- split-manifest SHA-256: `3f4e2fc58133e18c8a11ae41b39a66c78594a1168534b11e53c1ebb1d2108eb6`
- summary SHA-256: `719c1645a282b8c8bb1c357b4500a21fd6ccc29488879d09ed64cc8371c96992`
- data.yaml SHA-256: `99513c654f587a7bafd4b60166491fff5933ff065b5aa2c959fc13b025ae26b0`

All 4,124 interim-v4 train/validation assignments are preserved (0 moves).
The 1,040 newly annotated images from batch 010 were assigned only to
train/validation. The locked test set is identical by SHA-256 to interim-v4.

### Train/val/test primary-class image counts

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| `nasi_lemak` | 779 | 192 | 5 |
| `roti_canai` | 755 | 186 | 5 |
| `char_kuey_teow` | 908 | 237 | 15 |
| `chicken_rice` | 533 | 133 | 31 |
| `laksa` | 811 | 203 | 15 |
| `mee_goreng` | 345 | 82 | 11 |
| **Total** | **4,131** | **1,033** | **82** |

Mee Goreng object boxes in the split: 355 train / 90 val / 11 test (456 total
across object counts in summary; primary-class table above is by folder).

## JPEG identity repair (pre-split)

Local interim-v4 validation rewrote one corrupt JPEG in place through hardlinks
(`char_kuey_teow` image formerly
`06a2d25cd701ce79fe30f928735b286ae704c81319185e866d5d5e0333fe2b20`). The
restored bytes hash to
`8e0f9c364c79f2752087cc95206dbdb7e4c33b459fc663ced4e0c5cc006c42a7`. Dataset3
manifest identity and the interim-v4 split entry were migrated to that digest
before materializing interim-v5. Backup:
`data/dataset3/_repair_ultralytics_jpeg_rewrite/`.

## Configuration and training artifacts

- initialization: `runs/detect/dataset3_interim_v4/weights/best.pt`
- dataset: `data/dataset3-interim-v5/data.yaml`
- image size: 640
- batch size: 16
- requested epochs: 100
- early-stopping patience: 20
- completed epochs: 21
- `lr0`: 0.002; `resume: false`
- seed: 42, deterministic mode enabled
- accelerator: HPC NVIDIA GPU (`device=0`)
- run directory: `runs/detect/dataset3_interim_v5/`
- best checkpoint SHA-256: `3b84619b715d1f2b0c7c10f8094f799b84972b195a207b8c9c1912c270c5b892`
- last checkpoint SHA-256: `dbf088b92a10ea801862b8e60278ba25dd85307ad451e186c569371999efb920`
- results CSV SHA-256: `ccb7f04adee637ef6b8c180b7f1f5edcdcd94ec5181f28851c6ebc28e4d1b68e`
- args SHA-256: `051f4191e2f5e07d23447c9bb9c11c2ffd060c3df09373a5a114f5d2391a1ff1`
- HPC confusion matrix SHA-256: `6feb69027e88835d8e226dce0f2e0bf9e978f7685fab04a80f197e7829f60cea`
- HPC normalized confusion matrix SHA-256: `982d5ef429ec2d26366d758a339803d2461af4a1c6bcbb4b941c2252793daeac`

## Training result

Best mAP50–95 (0.7927) was reached at epoch 1, so patience stopped training
normally after epoch 21. Validation mAP50 peaked at epoch 17 (0.9569). This is
the best interim run to date on every headline metric.

| Checkpoint/epoch | Precision | Recall | mAP50 | mAP50–95 |
|------------------|----------:|-------:|------:|----------:|
| Best, epoch 1 | 0.894 | 0.899 | 0.945 | 0.793 |
| Last, epoch 21 | 0.876 | 0.868 | 0.935 | 0.771 |
| Peak mAP50, epoch 17 | 0.907 | 0.920 | 0.957 | 0.790 |

Relative to interim v4 (embedded best mAP50–95 0.783), interim v5 improves to
0.793 with recall rising from 0.848 to 0.899. Comparisons are directional
because the interim v5 validation set is larger (1,033 vs 825 images), though
every surviving interim v4 train/validation assignment was preserved.

## Validation-only local review

The best checkpoint was revalidated locally on the 1,033-image validation split
using Ultralytics `8.4.90`, PyTorch `2.12.1`, and CPU. This pass did not access
the locked test split. The embedded HPC metrics above remain authoritative; the
local pass is used for per-class diagnosis.

| Class | Validation instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|---------------------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 203 | 0.915 | 0.936 | 0.973 | 0.882 |
| Roti Canai | 215 | 0.881 | 0.865 | 0.916 | 0.730 |
| Char Kuey Teow | 239 | 0.871 | 0.937 | 0.949 | 0.822 |
| Chicken Rice | 144 | 0.909 | 0.924 | 0.953 | 0.704 |
| Laksa | 213 | 0.922 | 0.948 | 0.982 | 0.852 |
| Mee Goreng | 90 | 0.866 | 0.778 | 0.893 | 0.772 |
| **All classes** | **1,104** | **0.894** | **0.898** | **0.944** | **0.794** |

## Error review

- Mee Goreng continues to improve: local recall rose from 0.717 (interim v4) to
  0.778, and mAP50–95 from 0.746 to 0.772. It remains the weakest class.
- The HPC normalized confusion matrix still shows Char Kuey Teow ↔ Mee Goreng
  as the main inter-class confusion (~21% of true Char Kuey Teow predicted as
  Mee Goreng at that operating point) and a large Roti Canai background
  false-positive share (~28%).
- Chicken Rice keeps strong mAP50 (0.953) but the lowest mAP50–95 (0.704),
  a localization/box-tightness issue rather than recognition.
- Laksa and Nasi Lemak are strongest overall.

Local diagnostic hashes:

- confusion matrix: `8263d26276a6daac43f39c39d768c06d5af984e1378559dc04b65bda29b38980`
- normalized confusion matrix: `1515b30e91a3ef8c95d645413d5198929643d429523d1f18fec01976dd3d056e`

## Threshold calibration (validation split only)

Confidence and NMS-IoU thresholds were calibrated on the 1,033-image validation
split with `training_scripts/calibrate_thresholds.py` (deterministic; the locked
test split is refused by the script). Predictions were collected at a 0.001
confidence floor, matched to ground truth at evaluation IoU 0.5, and a confidence
grid (0.05–0.95, step 0.01) was swept across NMS-IoU candidates {0.45, 0.6, 0.7}.
The selection criterion is macro-averaged F1.

Recommended operating point: **confidence 0.47, NMS-IoU 0.45** (macro-F1 0.891,
micro P 0.904 / R 0.896 / F1 0.900). The macro-F1 surface is flat near the
optimum (0.891 at 0.47 vs 0.888 at the previous default 0.50), and NMS-IoU has
negligible effect (macro-F1 0.891 / 0.890 / 0.888 at 0.45 / 0.6 / 0.7), as
expected for predominantly single-item frames.

Per-class metrics at the recommended global threshold:

| Class | Precision | Recall | F1 | TP | FP | FN |
|-------|----------:|-------:|---:|---:|---:|---:|
| Nasi Lemak | 0.905 | 0.936 | 0.920 | 190 | 20 | 13 |
| Roti Canai | 0.905 | 0.847 | 0.875 | 182 | 19 | 33 |
| Char Kuey Teow | 0.877 | 0.929 | 0.902 | 222 | 31 | 17 |
| Chicken Rice | 0.929 | 0.903 | 0.915 | 130 | 10 | 14 |
| Laksa | 0.930 | 0.930 | 0.930 | 198 | 15 | 15 |
| Mee Goreng | 0.870 | 0.744 | 0.802 | 67 | 10 | 23 |

Per-class independent F1-optimal confidences (diagnostic only; the app uses one
global threshold): Nasi Lemak 0.57, Roti Canai 0.34, Char Kuey Teow 0.44,
Chicken Rice 0.47, Laksa 0.60, Mee Goreng 0.36. The spread confirms Mee Goreng
and Roti Canai favor lower thresholds (recall-limited), while Laksa and Nasi
Lemak tolerate higher thresholds; a single 0.47 cutoff is a reasonable
compromise until per-class thresholds are supported.

Applied to configuration: `CONFIDENCE_THRESHOLD=0.47`, `IOU_THRESHOLD=0.45`
(`backend/app/core/config.py`, `.env.example`, local `.env`).

- calibration report: `runs/detect/dataset3_interim_v5_calibration/calibration.json`
- report SHA-256: `5b76a7becc4ea4664e8f34a52af53e4a98e862c36e4e2fad8caa61e2e866786c`

## Single locked-test evaluation (one-shot, final)

The locked test split was evaluated exactly once on 23 July 2026 with the frozen
thresholds (`conf=0.47`, `iou=0.45`), imgsz 640, CPU, Ultralytics `8.4.90`,
PyTorch `2.12.1`. The test split is byte-identical (by SHA-256) to interim v4 and
was never used for model selection or threshold tuning. This is the unbiased
generalization estimate; it must not be reused for tuning.

Overall: precision 0.930, recall 0.930, mAP50 0.926, mAP50–95 0.678
(82 images, 84 instances).

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|-------:|----------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 5 | 5 | 0.828 | 0.967 | 0.962 | 0.691 |
| Roti Canai | 5 | 6 | 1.000 | 0.833 | 0.835 | 0.603 |
| Char Kuey Teow | 15 | 15 | 0.926 | 1.000 | 0.991 | 0.849 |
| Chicken Rice | 31 | 32 | 0.968 | 0.938 | 0.928 | 0.528 |
| Laksa | 15 | 15 | 1.000 | 0.933 | 0.935 | 0.728 |
| Mee Goreng | 11 | 11 | 0.858 | 0.909 | 0.905 | 0.669 |

The test split is small (84 instances; 5–6 for Nasi Lemak/Roti Canai), so
per-class numbers are noisy. Test mAP50 (0.926) tracks validation (0.944), and
Mee Goreng recall holds at 0.909. Chicken Rice again shows the lowest mAP50–95
(0.528) — a box-tightness/localization issue consistent with the validation
diagnosis rather than a recognition failure. The normalized confusion matrix
shows a residual Char Kuey Teow → Mee Goreng leak (~9%) and small background
false positives, in line with training.

Artifacts and provenance:

- run directory: `runs/detect/dataset3_interim_v5_test/`
- metrics summary: `runs/detect/dataset3_interim_v5_test/test-metrics.json`
- confusion matrix: `0a8dc2d95774a18afc17d60dde64a2644e829e843cf1422b26deafbe7b6b727f`
- normalized confusion matrix: `194788ce48861ba34af984b01bc0e7001d58b14378c8a3ce3c97d1977df1ebbb`
- PR curve: `288366856882fcd1e456bf341eb4224978809bc2722ab64fe0aa2a28e04ee485`
- F1 curve: `4f767586297fac3b84c0568bf0c8c51a52f368ac14cfb7bc6f69e0366f7b0ac1`

## Promotion and smoke test

The accepted checkpoint was copied to `data/weights/best.pt` (SHA-256
`3b84619b715d1f2b0c7c10f8094f799b84972b195a207b8c9c1912c270c5b892`, byte-identical
to the source). The FastAPI app was restarted and smoke-tested:

- `GET /api/health` → `model_loaded: true`, custom weights path, device `mps`,
  knowledge base loaded (6 entries).
- `GET /api/classes` → all six classes.
- `POST /api/predict` on a locked-test image → `chicken_rice` (0.848),
  advisory generated (template fallback; no LLM key configured).

Interim v5 is now the production-approved detector.

## Follow-ups

The epoch-1 fitness peak persists across interim runs; if a future run is
attempted, consider a shorter warmup or `cos_lr`, or evaluate `yolo11s` for the
noodle classes. Chicken Rice box-tightness (low mAP50–95) is the main quality
target for the next iteration. Per-class inference thresholds are not yet
supported; the calibration diagnostics suggest they would help Mee Goreng and
Roti Canai recall if added.
