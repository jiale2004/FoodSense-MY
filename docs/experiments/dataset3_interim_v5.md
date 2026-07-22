# Dataset3 YOLO11n Interim — `dataset3_interim_v5`

**Split materialized:** 22 July 2026
**Training completed:** 22 July 2026
**Assessment:** strongest interim run to date; candidate for final evaluation, not yet production-approved

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

## Decision and next gate

Interim v5 is the strongest interim checkpoint and the recommended candidate for
Phase E finalization. Next: calibrate confidence/IoU on the validation split
only, then run the single locked-test evaluation. Only after that gate copy the
accepted weights to `data/weights/best.pt` and restart the app.

Do not evaluate `split=test` before calibration is fixed. The epoch-1 fitness
peak persists; if a future run is attempted, consider a shorter warmup or
`cos_lr`, or evaluate `yolo11s` for the noodle classes.
