# Dataset3 Interim v8 — HPC retrain plan (YOLO11n only)

**Status:** planned (26 July 2026); run after interim v7 freeze review
**Model family:** YOLO11 **nano** only — no `yolo11s` / larger variants
**Split:** reuse locked `data/dataset3-interim-v5/` (no new split)
**Nano baseline to beat:** interim v7 freeze validation mAP50–95 **0.820**
  (and keep Mee Goreng recall ≥ interim v5 **0.778**)
**Assessment:** not yet trained; do not promote until validation clearly improves
nano quality without sacrificing Mee Goreng

Interim v7 freeze (`dataset3_interim_v7_n_freeze`) proved that `freeze=10` +
`lr0=0.0002` + `cos_lr` escapes the epoch-1 fitness peak and lifts aggregate
mAP50–95 to 0.820, but local Mee Goreng recall fell to **0.722** (v5 was
0.778; v6_s was 0.878). Chicken Rice mAP50–95 remains weak (~0.735). Interim
v8 stays on `yolo11n` and targets those two gaps. Both runs use **batch size
16**.

## Prior nano / small results (validation)

| Checkpoint | mAP50–95 | Mee Goreng recall | Notes |
|------------|----------:|------------------:|-------|
| v5 `best.pt` (prod nano) | 0.793 | 0.778 | Locked-test done |
| v7_n_freeze (ep 30) | **0.820** | 0.722 | Epoch-1 peak fixed; MG regressed |
| v6_s (yolo11s) | 0.826 | 0.878 | Strongest overall; **not** used as init |

## HPC root

All paths below are under:

`/home/user/22059034/FoodSense-MY/`

## Init checkpoint

Default init for both v8 runs:

`/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v7_n_freeze/weights/best.pt`

Do **not** initialize from `dataset3_interim_v6_s` (architecture mismatch). If
v7 Run B (`n_box`) finishes first and beats freeze on validation **without**
worsening Mee Goreng further, swap `model=` to that checkpoint instead.

## Shared configuration

| Param | Value |
|-------|--------|
| `model` family | `yolo11n` fine-tune from v7 freeze `best.pt` |
| `data` | `/home/user/22059034/FoodSense-MY/data/dataset3-interim-v5/data.yaml` |
| `imgsz` | `640` |
| `batch` | `16` |
| `epochs` | `100` |
| `patience` | `20` |
| `seed` | `42` |
| `deterministic` | `true` |
| `device` | `0` (HPC NVIDIA CUDA) |
| `resume` | `false` |
| `project` | `/home/user/22059034/FoodSense-MY/runs/detect` |

Selection and early stopping use the **validation** split only. Do not run
`split=test` until a candidate is accepted on validation and thresholds are
re-calibrated on validation.

## Run A — Mee Goreng recovery (`yolo11n`)

**Hypothesis:** A lighter freeze and higher classification loss let late backbone
/ head layers re-separate Char Kuey Teow ↔ Mee Goreng while keeping the soft
schedule that avoided epoch-1 collapse.

| Param | v7 freeze | v8 Run A |
|-------|-----------|----------|
| `model` | v5 `best.pt` | v7 freeze `best.pt` |
| `freeze` | `10` | `5` |
| `lr0` | `0.0002` | `0.0003` |
| `warmup_epochs` | `0.5` | `0.5` |
| `cos_lr` | `true` | `true` |
| `cls` | `0.5` | `1.0` |
| `mixup` | `0.0` | `0.15` |
| `close_mosaic` | `15` | `15` |
| `box` | `10.0` | `10.0` |
| `batch` | `16` | `16` |
| `name` | `dataset3_interim_v7_n_freeze` | `dataset3_interim_v8_n_mg` |

```bash
yolo detect train \
  model=/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v7_n_freeze/weights/best.pt \
  data=/home/user/22059034/FoodSense-MY/data/dataset3-interim-v5/data.yaml \
  epochs=100 \
  patience=20 \
  imgsz=640 \
  batch=16 \
  seed=42 \
  deterministic=true \
  device=0 \
  resume=false \
  freeze=5 \
  lr0=0.0003 \
  lrf=0.01 \
  warmup_epochs=0.5 \
  warmup_bias_lr=0.01 \
  cos_lr=true \
  cls=1.0 \
  mixup=0.15 \
  close_mosaic=15 \
  box=10.0 \
  project=/home/user/22059034/FoodSense-MY/runs/detect \
  name=dataset3_interim_v8_n_mg
```

## Run B — localization focus (`yolo11n`)

**Hypothesis:** From the stronger v7 freeze init, multi-scale + higher box loss
tightens Chicken Rice boxes without a hard `freeze=10` that starved noodle
features. This supersedes the still-unrun v7 `n_box` plan (same idea, better
init, lighter freeze).

| Param | Choice | Why |
|-------|--------|-----|
| `model` | v7 freeze `best.pt` | Best nano mAP init |
| `freeze` | `5` | Allow mid-backbone adaptation |
| `lr0` | `0.0003` | Soft fine-tune |
| `warmup_epochs` | `0.5` | Short warmup |
| `cos_lr` | `true` | Smooth decay |
| `multi_scale` | `0.5` | Plate / crop scale jitter |
| `box` | `12.0` | Emphasize localization |
| `close_mosaic` | `20` | Longer clean-box phase |
| `scale` | `0.9` | Stronger geometric scale aug |
| `degrees` | `5.0` | Mild rotation |
| `cls` | `0.5` | Keep default class weight (Run A owns MG) |
| `batch` | `16` | Match prior interim HPC runs |
| `name` | `dataset3_interim_v8_n_box` | |

```bash
yolo detect train \
  model=/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v7_n_freeze/weights/best.pt \
  data=/home/user/22059034/FoodSense-MY/data/dataset3-interim-v5/data.yaml \
  epochs=100 \
  patience=20 \
  imgsz=640 \
  batch=16 \
  seed=42 \
  deterministic=true \
  device=0 \
  resume=false \
  freeze=5 \
  lr0=0.0003 \
  lrf=0.01 \
  warmup_epochs=0.5 \
  warmup_bias_lr=0.01 \
  cos_lr=true \
  multi_scale=0.5 \
  box=12.0 \
  close_mosaic=20 \
  scale=0.9 \
  degrees=5.0 \
  project=/home/user/22059034/FoodSense-MY/runs/detect \
  name=dataset3_interim_v8_n_box
```

## Acceptance gate (after both runs finish)

Promote a nano candidate only if **all** hold on validation:

1. mAP50–95 ≥ v7 freeze (`0.820`), or within ~0.005 with a clear MG recall win.
2. Mee Goreng recall ≥ v5 (`0.778`); prefer ≥ 0.80.
3. Chicken Rice mAP50–95 improves vs v7 freeze local (~0.735) for Run B, or at
   least does not regress for Run A.
4. Char Kuey Teow ↔ Mee Goreng confusion does not worsen vs v7 freeze (~22% MG→CKT).

Then:

1. Re-run `training_scripts/calibrate_thresholds.py` on validation only.
2. Run the locked test split **once** with the new thresholds.
3. Promote to `data/weights/best.pt` only if locked-test metrics justify replacing
   interim v5 (and justify staying on nano vs deploying v6_s).

If neither run recovers Mee Goreng recall, stop nano schedule sweeps and prefer
**v6_s** for promotion (calibrate + locked test), treating further nano work as
optional size-constrained experiments only.

## Artifacts (to fill after training)

| Run | Directory | Best ckpt SHA-256 | Results CSV SHA-256 |
|-----|-----------|-------------------|---------------------|
| A `dataset3_interim_v8_n_mg` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_mg/` | _pending_ | _pending_ |
| B `dataset3_interim_v8_n_box` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_box/` | _pending_ | _pending_ |
