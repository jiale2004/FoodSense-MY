# Dataset3 YOLO11n Interim — `dataset3_interim_v4`

**Split materialized:** 22 July 2026  
**Training status:** pending HPC retrain (lower `lr0=0.002`)  
**Assessment:** not started; not approved for production inference

## Split configuration and artifacts

- source staging: `data/dataset3/` (4,206 annotated images / 4,440 boxes)
- output: `data/dataset3-interim-v4/`
- base split: `data/dataset3-interim-v3/split-manifest.jsonl`
  (SHA-256 `74560af5a330fde8005ae953552a17a2bb84abc419c76ac0a3418ddae0574091`)
- locked test selection: `data/cvat/test-holdout-review-v1/accepted-selection.jsonl`
  (SHA-256 `2094c4915d282ef9580ca2d1cd5da3464bd2f530d3a50674e4e444aba536108a`)
- algorithm: `dataset3-group-stratified-locked-v2`
- incremental train fraction: 0.8
- seed: 42
- materialization: hardlink images / copy labels
- split: 3,299 train, 825 validation, 82 locked reviewed test images
- boxes: 3,486 train, 870 validation, 84 test
- cross-split leakage groups: 0
- split-manifest SHA-256: `367e09f986babc091e495d52ec1622d1cf578988db34dd8df5b0ccb5986d57d1`
- summary SHA-256: `de2d7ce94c48829c00edc7cd549db9affa8a04c23c89b5926fb27e56b6140c59`
- data.yaml SHA-256: `ef706aeb8242644ff39e760530222f908d808ce543a776f7516c87d438b60d1f`

All 2,174 interim-v3 images keep their original split assignment (0 moves).
All 2,092 surviving interim-v3 train/validation images remain outside test.
The 2,032 newly annotated images (post interim-v3) were assigned only to
train/validation. The locked test set is identical by SHA-256 to interim-v3.

### Train/val/test primary-class image counts

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| `nasi_lemak` | 580 | 143 | 5 |
| `roti_canai` | 575 | 142 | 5 |
| `char_kuey_teow` | 798 | 203 | 15 |
| `chicken_rice` | 533 | 132 | 31 |
| `laksa` | 581 | 146 | 15 |
| `mee_goreng` | 232 | 59 | 11 |
| **Total** | **3,299** | **825** | **82** |

Mee Goreng object boxes in the split: 237 train / 60 val / 11 test (308 total).

## Planned training configuration

```bash
yolo detect train \
  model=.../runs/detect/dataset3_interim_v3/weights/best.pt \
  data=.../data/dataset3-interim-v4/data.yaml \
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
  name=dataset3_interim_v4
```

Initialization must be the interim v3 best checkpoint
(`runs/detect/dataset3_interim_v3/weights/best.pt`, SHA-256
`5d6dde4b927c53dd1697ffbb759046608f492d47f2f7349452b7e01b3c5b8080`), not a
fresh `yolo11n.pt`. Use `resume: false` with that checkpoint as `model=`.
Lower `lr0=0.002` addresses the interim-v3 epoch-1 peak under default
`lr0=0.01`.

Rewrite `data.yaml` `path:` for the cluster, or regenerate the split on HPC.
Images are hardlinks into `data/dataset3/`, so transfers must dereference
(`rsync -aL`) unless the full staging tree is available on the cluster.

## Gate

Do not evaluate `split=test`. Do not copy weights to `data/weights/best.pt`
until annotation freeze and validation-only model selection are complete.
After training, record HPC metrics, local validation-only per-class diagnosis,
and whether Mee Goreng recall improved versus interim v3 (0.513).
