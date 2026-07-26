# Dataset3 Interim v7 — HPC retrain plan (YOLO11n only)

**Status:** planned (25 July 2026); run after interim v6 results are in
**Model family:** YOLO11 **nano** only — no `yolo11s` / larger variants
**Split:** reuse locked `data/dataset3-interim-v5/` (no new split)
**Production baseline to beat:** interim v5 validation mAP50–95 0.793
  (or the better of v5 / v6 nano if v6 Run A wins on validation)
**Assessment:** not yet trained; do not promote until validation clearly beats
the current best nano checkpoint

Interim v7 stays on `yolo11n` and explores the levers left after v6 Run A
(cosine fine-tune): backbone freeze with a still-lower LR, and a
localization-focused fine-tune for Chicken Rice box-tightness. Both runs use
**batch size 16**.

## Init checkpoint

Prefer the strongest **nano** checkpoint available at train time:

1. `runs/detect/dataset3_interim_v6_n_cos/weights/best.pt` if its validation
   mAP50–95 beats interim v5
2. otherwise `runs/detect/dataset3_interim_v5/weights/best.pt`

Do **not** initialize from `dataset3_interim_v6_s` (YOLO11s). Commands below use
the v5 path as the default; swap the `model=` path if v6 nano wins.

## Shared configuration

| Param | Value |
|-------|--------|
| `model` family | `yolo11n` fine-tune from a nano `best.pt` |
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

## Run A — freeze + ultra-low LR (`yolo11n`)

**Hypothesis:** Freezing the early backbone and using a very low LR stops the
epoch-1 fitness peak seen on v3–v5 (and possibly v6 cosine) by updating only
the detection head / late stages.

| Param | v6 Run A (`n_cos`) | v7 Run A |
|-------|--------------------|----------|
| `model` | v5 `best.pt` | best nano at train time (default v5) |
| `lr0` | `0.0005` | `0.0002` |
| `warmup_epochs` | `1.0` | `0.5` |
| `warmup_bias_lr` | `0.01` | `0.01` |
| `cos_lr` | `true` | `true` |
| `freeze` | _(none)_ | `10` |
| `close_mosaic` | `15` | `15` |
| `box` | `10.0` | `10.0` |
| `batch` | `16` | `16` |
| `name` | `dataset3_interim_v6_n_cos` | `dataset3_interim_v7_n_freeze` |

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
  freeze=10 \
  lr0=0.0002 \
  lrf=0.01 \
  warmup_epochs=0.5 \
  warmup_bias_lr=0.01 \
  cos_lr=true \
  close_mosaic=15 \
  box=10.0 \
  project=runs/detect \
  name=dataset3_interim_v7_n_freeze
```

## Run B — localization focus (`yolo11n`)

**Hypothesis:** Stronger box loss, multi-scale training, and a longer
post-mosaic phase improve Chicken Rice mAP50–95 without leaving the nano
family.

| Param | Choice | Why |
|-------|--------|-----|
| `model` | best nano at train time (default v5) | Stay on `yolo11n` |
| `lr0` | `0.0005` | Same soft fine-tune scale as v6 cosine |
| `warmup_epochs` | `1.0` | Short warmup |
| `cos_lr` | `true` | Smooth decay |
| `multi_scale` | `0.5` | Scale jitter for plate / crop variation |
| `box` | `12.0` | Emphasize localization vs classification |
| `close_mosaic` | `20` | Longer clean-box phase |
| `scale` | `0.9` | Stronger geometric scale aug |
| `degrees` | `5.0` | Mild rotation (plates are not upright-locked) |
| `batch` | `16` | Match prior interim HPC runs |
| `name` | `dataset3_interim_v7_n_box` | |

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
  multi_scale=0.5 \
  box=12.0 \
  close_mosaic=20 \
  scale=0.9 \
  degrees=5.0 \
  project=runs/detect \
  name=dataset3_interim_v7_n_box
```

## Acceptance gate (after both runs finish)

1. Compare validation mAP50–95 against the best prior **nano** checkpoint
   (v5 `0.793`, or v6 `n_cos` if better).
2. Check per-class: Chicken Rice mAP50–95, Mee Goreng recall, Char Kuey Teow ↔
   Mee Goreng confusion.
3. If a run wins on validation, re-run
   `training_scripts/calibrate_thresholds.py` on validation only.
4. Run the locked test split **once** with the new thresholds.
5. Promote to `data/weights/best.pt` only if locked-test metrics justify replacing
   the current production nano weights.

If both v7 runs still peak at epoch 1 with no val gain, stop nano fine-tune
sweeps and treat data / inference-threshold work as the next lever (not another
`lr0` tweak from the same v5 init).

## Artifacts (to fill after training)

| Run | Directory | Best ckpt SHA-256 | Results CSV SHA-256 |
|-----|-----------|-------------------|---------------------|
| A `dataset3_interim_v7_n_freeze` | `runs/detect/dataset3_interim_v7_n_freeze/` | _pending_ | _pending_ |
| B `dataset3_interim_v7_n_box` | `runs/detect/dataset3_interim_v7_n_box/` | _pending_ | _pending_ |
