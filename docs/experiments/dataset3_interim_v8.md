# Dataset3 Interim v8 — HPC retrain plan (YOLO11n only)

**Status:** production-approved (28 July 2026)
**Model family:** YOLO11 **nano** only — no `yolo11s` / larger variants
**Split:** reuse locked `data/dataset3-interim-v5/` (no new split)
**Nano baseline beaten:** interim v7 freeze validation mAP50–95 **0.820**;
  Mee Goreng recall recovered above v5 (**0.822** ≥ **0.778**)
**Assessment:** **Run A (`v8_n_mg`) is the production detector.** Calibrated on
validation (conf **0.5** / NMS-IoU **0.45**), evaluated once on the locked test
set (mAP50 0.886, mAP50–95 0.673), and promoted to `data/weights/best.pt`.
Locked-test recall/mAP50 are slightly behind interim v5; validation and MG
recall are stronger. Monitor chicken_rice test recall.

Results visualization (4-page PDF):
[`docs/logs/dataset3_interim_v8_results.pdf`](../logs/dataset3_interim_v8_results.pdf)
— headline metrics, v8_n_mg training curves, per-class locked-test/validation
charts, and validation-only threshold calibration.

Interim v7 freeze (`dataset3_interim_v7_n_freeze`) proved that `freeze=10` +
`lr0=0.0002` + `cos_lr` escapes the epoch-1 fitness peak and lifts aggregate
mAP50–95 to 0.820, but local Mee Goreng recall fell to **0.722** (v5 was
0.778; v6_s was 0.878). Chicken Rice mAP50–95 remains weak (~0.735). Interim
v8 stays on `yolo11n` and targets those two gaps. Both runs use **batch size
16**.

## Prior nano / small results (validation)

| Checkpoint | mAP50–95 | Mee Goreng recall | Notes |
|------------|----------:|------------------:|-------|
| v5 `best.pt` (prior prod) | 0.793 | 0.778 | Locked-test done; superseded |
| v7_n_freeze (ep 30) | 0.820 | 0.722 | Epoch-1 peak fixed; MG regressed |
| v6_s (yolo11s) | 0.826 | 0.878 | Strongest small; **not** used as init |
| **v8_n_mg (ep 73)** | **0.830** | **0.822** | **Production nano** |
| v8_n_box (ep 94) | 0.828 | 0.807 | Localization goal not met vs Run A |

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

### Run B result (completed)

Config matched the plan (`freeze=5`, `lr0=0.0003`, `multi_scale=0.5`,
`box=12.0`, `close_mosaic=20`, `scale=0.9`, `degrees=5.0`). Best fitness at
**epoch 94**; run completed all 100 epochs (no early stop).

| Checkpoint | Precision | Recall | mAP50 | mAP50–95 |
|------------|----------:|-------:|------:|----------:|
| Best, epoch 94 (HPC) | 0.923 | 0.912 | 0.960 | **0.828** |
| Last, epoch 100 (HPC) | 0.935 | 0.901 | 0.959 | 0.825 |
| Local val (MPS, best.pt) | 0.925 | 0.907 | 0.959 | 0.827 |

Local per-class (validation only; test not accessed):

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|----------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 203 | 0.966 | 0.926 | 0.981 | 0.902 |
| Roti Canai | 215 | 0.902 | 0.884 | 0.939 | 0.772 |
| Char Kuey Teow | 239 | 0.891 | 0.958 | 0.961 | 0.869 |
| Chicken Rice | 144 | 0.916 | 0.910 | 0.960 | **0.749** |
| Laksa | 213 | 0.963 | 0.958 | 0.989 | 0.877 |
| Mee Goreng | 90 | 0.912 | **0.807** | 0.922 | 0.793 |
| **All** | **1,104** | **0.925** | **0.907** | **0.959** | **0.827** |

Gate: mAP50–95 ≥ 0.820 ✓; Mee Goreng recall ≥ 0.778 ✓. Chicken Rice
mAP50–95 **0.749** is essentially flat vs Run A (0.751) and only a small lift
vs v7 (~0.735) — the localization hypothesis did not beat Run A. MG→CKT on the
HPC normalized matrix is ~14% (worse than Run A’s ~12%).

best.pt SHA-256:
`131ad0d7d209f747e48b4f6a16fa9e3c1a652f3967d468bea78437a2ce43f501`

## Decision: nano winner = Run A → production

| Run | mAP50–95 | MG recall | Chicken Rice mAP50–95 | Winner? |
|-----|----------:|----------:|----------------------:|---------|
| A `v8_n_mg` | **0.830** | **0.822** | 0.751 | **Yes → promoted** |
| B `v8_n_box` | 0.828 | 0.807 | 0.749 | No |

## Threshold calibration (validation only)

Ran `training_scripts/calibrate_thresholds.py` on the 1,033-image validation
split (refuses `split=test`). Recommended operating point:

| Setting | Value |
|---------|------:|
| `CONFIDENCE_THRESHOLD` | **0.5** |
| `IOU_THRESHOLD` (NMS) | **0.45** |
| macro-F1 | 0.932 |
| micro P / R / F1 | 0.946 / 0.932 / 0.939 |

Per-class at the global threshold (eval IoU 0.5):

| Class | P | R | F1 | TP | FP | FN |
|-------|--:|--:|---:|---:|---:|---:|
| nasi_lemak | 0.970 | 0.946 | 0.958 | 192 | 6 | 11 |
| roti_canai | 0.938 | 0.916 | 0.927 | 197 | 13 | 18 |
| char_kuey_teow | 0.915 | 0.946 | 0.930 | 226 | 21 | 13 |
| chicken_rice | 0.971 | 0.917 | 0.943 | 132 | 4 | 12 |
| laksa | 0.963 | 0.977 | 0.970 | 208 | 8 | 5 |
| mee_goreng | 0.914 | 0.822 | 0.866 | 74 | 7 | 16 |

Report: `runs/detect/dataset3_interim_v8_n_mg_calibration/calibration.json`
(SHA-256 `c91e8317f22413a370ac9abd70663e88d49b0d0a99e45b4e5431c2ee289d227d`).

Applied to `backend/app/core/config.py`, `.env.example`, and local `.env`.

## Locked-test evaluation (one-shot)

`conf=0.5 iou=0.45` on the 82-image / 84-instance locked test split
(Ultralytics 8.4.90, MPS). Do not reuse for tuning.

| | Precision | Recall | mAP50 | mAP50–95 |
|--|----------:|-------:|------:|----------:|
| **v8_n_mg test** | **0.940** | **0.868** | **0.886** | **0.673** |
| v5 test (prior) | 0.930 | 0.930 | 0.926 | 0.678 |

Per-class locked test:

| Class | Images | Instances | P | R | mAP50 | mAP50–95 |
|-------|-------:|----------:|--:|--:|------:|----------:|
| nasi_lemak | 5 | 5 | 1.000 | 0.800 | 0.795 | 0.676 |
| roti_canai | 5 | 6 | 0.702 | 0.833 | 0.835 | 0.599 |
| char_kuey_teow | 15 | 15 | 0.938 | 1.000 | 0.974 | 0.836 |
| chicken_rice | 31 | 32 | 1.000 | 0.733 | 0.873 | 0.518 |
| laksa | 15 | 15 | 1.000 | 0.933 | 0.935 | 0.733 |
| mee_goreng | 11 | 11 | 1.000 | 0.909 | 0.905 | 0.677 |

Artifacts: `runs/detect/dataset3_interim_v8_n_mg_test/` including
`test-metrics.json` (SHA-256
`107831cf1bc5426ce60152248e8ebb25cdbb51bb09c25316e26d0293cd4d71f8`).

Locked-test tradeoff vs v5: slightly lower recall and mAP50 (chicken_rice
recall 0.733 is the main gap); Mee Goreng test recall matches v5 (0.909).
Promotion proceeds on stronger validation / MG recovery and nano size.

## Promotion

Copied `runs/detect/dataset3_interim_v8_n_mg/weights/best.pt` →
`data/weights/best.pt` (SHA-256
`f0cda9e12125326f24d61bab789e6e09118855a8fd56cb8b0a96e4eec95ee412`,
byte-identical). Restart uvicorn to load the new weights and thresholds.

## Artifacts

| Run | Directory | Best ckpt SHA-256 | Results CSV SHA-256 |
|-----|-----------|-------------------|---------------------|
| A `dataset3_interim_v8_n_mg` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_mg/` | `f0cda9e12125326f24d61bab789e6e09118855a8fd56cb8b0a96e4eec95ee412` | `27fe60d04e382b316dcbae87914d8cb6e2f59dc8ef54786cbf6bdabf87e5995f` |
| B `dataset3_interim_v8_n_box` | `/home/user/22059034/FoodSense-MY/runs/detect/dataset3_interim_v8_n_box/` | `131ad0d7d209f747e48b4f6a16fa9e3c1a652f3967d468bea78437a2ce43f501` | `4444fd122cb135832956818dd1f01d351d1f08fe67887fe7ed01302289994898` |
| Calibration | `runs/detect/dataset3_interim_v8_n_mg_calibration/` | — | report SHA above |
| Locked test | `runs/detect/dataset3_interim_v8_n_mg_test/` | — | `test-metrics.json` SHA above |
