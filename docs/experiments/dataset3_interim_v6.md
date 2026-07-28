# Dataset3 Interim v6 — HPC retrain plan

**Status:** planned / queued for HPC (25 July 2026)
**Split:** reuse locked `data/dataset3-interim-v5/` (no new split)
**Production baseline to beat:** interim v5 validation mAP50–95 0.793
**Assessment:** not yet trained; do not promote until validation clearly beats v5

Interim v6 addresses the persistent epoch-1 fitness peak on fine-tunes from
converged checkpoints, Chicken Rice box-tightness (lowest locked-test
mAP50–95), and Char Kuey Teow ↔ Mee Goreng confusion. Two complementary HPC
runs share the same split, seed, and **batch size 16**.

## Shared configuration

| Param | Value |
|-------|--------|
| `data` | `data/dataset3-interim-v5/data.yaml` |
| `imgsz` | `640` |
| `batch` | `16` |
| `epochs` | `100` |
| `patience` | `20` |
| `seed` | `42` |
| `deterministic` | `true` |
| `device` | `0` (HPC NVIDIA CUDA) |
| `resume` | `false` |
| `project` | `runs/detect` |

Selection and early stopping use the **validation** split only. Do not run
`split=test` until a candidate is accepted on validation and thresholds are
re-calibrated on validation.

## Run A — schedule / fine-tune fix (`yolo11n`)

**Hypothesis:** Lower LR, shorter warmup, and cosine decay let training improve
past epoch 1 and tighten boxes (especially Chicken Rice).

| Param | Interim v5 | Run A |
|-------|------------|-------|
| `model` | v4 `best.pt` | `runs/detect/dataset3_interim_v5/weights/best.pt` |
| `lr0` | `0.002` | `0.0005` |
| `lrf` | `0.01` | `0.01` |
| `warmup_epochs` | `3.0` | `1.0` |
| `warmup_bias_lr` | `0.1` | `0.01` |
| `cos_lr` | `false` | `true` |
| `close_mosaic` | `10` | `15` |
| `box` | `7.5` | `10.0` |
| `batch` | `16` | `16` |
| `name` | `dataset3_interim_v5` | `dataset3_interim_v6_n_cos` |

```bash
yolo detect train \
  model=runs/detect/dataset3_interim_v5/weights/best.pt \
  data=data/dataset3-interim-v5/data.yaml \
  epochs=100 \
  patience=20 \
  imgsz=640 \
  batch=16 \
  seed=42 \
  deterministic=true \
  device=0 \
  resume=false \
  lr0=0.0005 \
  lrf=0.01 \
  warmup_epochs=1.0 \
  warmup_bias_lr=0.01 \
  cos_lr=true \
  close_mosaic=15 \
  box=10.0 \
  project=runs/detect \
  name=dataset3_interim_v6_n_cos
```

## Run B — capacity (`yolo11s`)

**Hypothesis:** Nano is capacity-limited on noodle-class separation; YOLO11s
from COCO pretrained weights may reduce Char Kuey Teow ↔ Mee Goreng confusion.

| Param | Choice | Notes |
|-------|--------|--------|
| `model` | `yolo11s.pt` | Fresh small backbone; do **not** load nano `best.pt` |
| `lr0` | `0.01` | Fresh train from pretrained, not fine-tune |
| `warmup_epochs` | `3.0` | Default fresh-train warmup |
| `cos_lr` | `false` | Match pilot / early interim defaults |
| `mosaic` | `1.0` | Default |
| `mixup` | `0.1` | Mild extra diversity for noodle classes |
| `close_mosaic` | `10` | Default |
| `batch` | `16` | Same as all prior interim HPC runs |
| `name` | `dataset3_interim_v6_s` | |

```bash
yolo detect train \
  model=yolo11s.pt \
  data=data/dataset3-interim-v5/data.yaml \
  epochs=100 \
  patience=20 \
  imgsz=640 \
  batch=16 \
  seed=42 \
  deterministic=true \
  device=0 \
  resume=false \
  lr0=0.01 \
  lrf=0.01 \
  warmup_epochs=3.0 \
  cos_lr=false \
  mosaic=1.0 \
  mixup=0.1 \
  close_mosaic=10 \
  project=runs/detect \
  name=dataset3_interim_v6_s
```

## Acceptance gate (after both runs finish)

1. Compare validation mAP50–95 against interim v5 (`0.793`).
2. Check per-class: Chicken Rice mAP50–95, Mee Goreng recall, Char Kuey Teow ↔
   Mee Goreng confusion.
3. If a run wins on validation, re-run
   `training_scripts/calibrate_thresholds.py` on validation only.
4. Run the locked test split **once** with the new thresholds.
5. Promote to `data/weights/best.pt` only if locked-test metrics justify replacing
   interim v5.

If Run A still peaks at epoch 1, next lever is interim v7 (YOLO11n only):
`lr0=0.0002` with `freeze=10`, plus a localization-focused nano fine-tune.
See [`dataset3_interim_v7.md`](dataset3_interim_v7.md). Do not start another
`lr0=0.002` fine-tune from the same checkpoint.

## Artifacts (to fill after training)

| Run | Directory | Best ckpt SHA-256 | Results CSV SHA-256 |
|-----|-----------|-------------------|---------------------|
| A `dataset3_interim_v6_n_cos` | `runs/detect/dataset3_interim_v6_n_cos/` | _pending_ | _pending_ |
| B `dataset3_interim_v6_s` | `runs/detect/dataset3_interim_v6_s/` | _pending_ | _pending_ |
