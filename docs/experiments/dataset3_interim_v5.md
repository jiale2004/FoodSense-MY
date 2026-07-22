# Dataset3 YOLO11n Interim — `dataset3_interim_v5`

**Split materialized:** 22 July 2026  
**Training status:** pending HPC retrain (`lr0=0.002` from interim v4)  
**Assessment:** not started; not approved for production inference

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

## Planned training configuration

```bash
yolo detect train \
  model=.../runs/detect/dataset3_interim_v4/weights/best.pt \
  data=.../data/dataset3-interim-v5/data.yaml \
  epochs=100 \
  patience=20 \
  imgsz=640 \
  batch=16 \
  seed=42 \
  deterministic=True \
  lr0=0.002 \
  device=0 \
  workers=8 \
  project=.../runs/detect \
  name=dataset3_interim_v5
```

Rewrite `data.yaml` `path:` for the cluster. Transfer hardlinks with
`rsync -aL` unless the full `data/dataset3/` tree is available on the cluster.

## Gate

Do not evaluate `split=test`. Do not copy weights to `data/weights/best.pt`
until validation-only model selection and the single locked-test evaluation are
complete.
