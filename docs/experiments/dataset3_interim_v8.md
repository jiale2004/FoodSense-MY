# Dataset3 Interim v8 — HPC retrain plan (YOLO11n only)

**Status:** Run A completed (26 July 2026); Run B in progress on HPC
**Model family:** YOLO11 **nano** only — no `yolo11s` / larger variants
**Split:** reuse locked `data/dataset3-interim-v5/` (no new split)
**Nano baseline to beat:** interim v7 freeze validation mAP50–95 **0.820**
  (and keep Mee Goreng recall ≥ interim v5 **0.778**)
**Assessment:** Run A passes the nano acceptance gate on validation (mAP50–95
0.830, Mee Goreng recall 0.822). Hold promotion until Run B finishes, then
compare; calibrate + locked-test only the winning nano candidate.

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
| v7_n_freeze (ep 30) | 0.820 | 0.722 | Epoch-1 peak fixed; MG regressed |
| v6_s (yolo11s) | 0.826 | 0.878 | Strongest small; **not** used as init |
| **v8_n_mg (ep 73)** | **0.830** | **0.822** | Best nano so far; Run A complete |

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

### Run A result (completed)

Config matched the plan (`freeze=5`, `lr0=0.0003`, `cls=1.0`, `mixup=0.15`,
`cos_lr=true`, HPC absolute paths). Best fitness at **epoch 73**; early-stop
ended normally at epoch 93 after patience 20.

| Checkpoint | Precision | Recall | mAP50 | mAP50–95 |
|------------|----------:|-------:|------:|----------:|
| Best, epoch 73 (HPC) | 0.936 | 0.926 | 0.959 | **0.830** |
| Last, epoch 93 (HPC) | 0.935 | 0.923 | 0.958 | 0.829 |
| Local val (MPS, best.pt) | 0.937 | 0.924 | 0.959 | 0.831 |

Local per-class (validation only; test not accessed):

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|----------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 203 | 0.973 | 0.946 | 0.976 | 0.896 |
| Roti Canai | 215 | 0.927 | 0.892 | 0.936 | 0.772 |
| Char Kuey Teow | 239 | 0.908 | 0.955 | 0.967 | 0.869 |
| Chicken Rice | 144 | 0.951 | 0.947 | 0.970 | 0.751 |
| Laksa | 213 | 0.963 | 0.981 | 0.992 | 0.876 |
| Mee Goreng | 90 | 0.899 | **0.822** | 0.916 | 0.823 |
| **All** | **1,104** | **0.937** | **0.924** | **0.959** | **0.831** |

Gate check vs plan: mAP50–95 ≥ 0.820 ✓; Mee Goreng recall ≥ 0.778 ✓ (and ≥
0.80 ✓). MG→CKT on the HPC normalized matrix fell from ~22% (v7 freeze) to
~12%. Chicken Rice mAP50–95 0.751 is a small lift vs v7 (~0.735); Run B still
targets localization.

best.pt SHA-256:
`f0cda9e12125326f24d61bab789e6e09118855a8fd56cb8b0a96e4eec95ee412`

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
| A `dataset3_interim_v8_n_mg` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_mg/` | `f0cda9e12125326f24d61bab789e6e09118855a8fd56cb8b0a96e4eec95ee412` | `27fe60d04e382b316dcbae87914d8cb6e2f59dc8ef54786cbf6bdabf87e5995f` |
| B `dataset3_interim_v8_n_box` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_box/` | _pending (running)_ | _pending_ |
