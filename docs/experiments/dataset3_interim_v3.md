# Dataset3 YOLO11n Interim — `dataset3_interim_v3`

**Training completed:** 20 July 2026  
**Assessment:** accepted for assisted-batch-005 proposals; not approved for production inference

## Configuration and artifacts

- initialization: `runs/detect/dataset3_interim_v2/weights/best.pt`
- dataset: `data/dataset3-interim-v3/data.yaml`
- split: 1,674 train, 418 validation, 82 locked reviewed test images
- image size: 640
- batch size: 16
- requested epochs: 100
- early-stopping patience: 20
- completed epochs: 21
- seed: 42, deterministic mode enabled
- accelerator: HPC NVIDIA GPU (`device=0`)
- run directory: `runs/detect/dataset3_interim_v3/`
- best checkpoint SHA-256: `5d6dde4b927c53dd1697ffbb759046608f492d47f2f7349452b7e01b3c5b8080`
- last checkpoint SHA-256: `65ad18d6f75c43e9654045f06de4b2c987cfc5e065dce85d78ec1cd2439ccd1d`
- results CSV SHA-256: `ee7875d1024e25f2e8cc262e2c6b32af4ea53534aa807fa1627dd087e9343248`
- args SHA-256: `77476bff408c477d4dcc9e00953172b199c6c8eaf4ae3e3138661d7d7d59fee0`

The run correctly used `resume: false` with the interim v2 checkpoint as
initialization. The checkpoint embeds the canonical class IDs: `nasi_lemak`,
`roti_canai`, `char_kuey_teow`, `chicken_rice`, `laksa`, and `mee_goreng` at
IDs 0–5.

## Training result

Ultralytics uses mAP50–95 as detection fitness. Because training started from a
strong interim v2 checkpoint on a larger, shifted validation view, the best
mAP50–95 (0.76078) was reached at epoch 1. No later epoch improved it, so the
configured 20-epoch patience stopped training normally after epoch 21. The run
was not interrupted.

| Checkpoint/epoch | Precision | Recall | mAP50 | mAP50–95 |
|------------------|----------:|-------:|------:|----------:|
| Best, epoch 1 | 0.891 | 0.859 | 0.932 | 0.761 |
| Last, epoch 21 | 0.887 | 0.882 | 0.927 | 0.737 |

Relative to the interim v2 embedded validation result, mAP50–95 increased from
0.747 to 0.761 while mAP50 is comparable. These comparisons are directional
because the interim v3 validation set is larger (418 vs 267 images), although
every surviving interim v2 train/validation assignment was preserved.

The best epoch landing at epoch 1 indicates the default fine-tuning learning
rate (`lr0=0.01`) is too high when continuing from an already-converged
checkpoint. A lower `lr0` (for example 0.002) with a short warmup is the
recommended change for the next interim run.

## Validation-only local review

The best checkpoint was revalidated locally on the 418-image validation split
using Ultralytics `8.4.87`, PyTorch `2.12.1`, and CPU. This pass did not access
the locked test split. Version differences cause a small aggregate difference,
so the embedded HPC metrics above remain authoritative; the local pass is used
for per-class diagnosis.

| Class | Validation instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|---------------------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 66 | 0.868 | 0.924 | 0.938 | 0.810 |
| Roti Canai | 68 | 0.767 | 0.941 | 0.892 | 0.728 |
| Char Kuey Teow | 109 | 0.884 | 0.907 | 0.941 | 0.783 |
| Chicken Rice | 97 | 0.931 | 0.928 | 0.949 | 0.690 |
| Laksa | 64 | 0.968 | 0.952 | 0.982 | 0.822 |
| Mee Goreng | 33 | 0.944 | 0.513 | 0.899 | 0.731 |
| **All classes** | **437** | **0.894** | **0.861** | **0.933** | **0.761** |

## Error review

- Mee Goreng is now the dominant weakness: recall fell to 0.513, and the
  normalized confusion matrix shows 33% of true Mee Goreng predicted as Char
  Kuey Teow. Precision stays high (0.944), so the model is conservative on Mee
  Goreng rather than over-predicting it.
- Mee Goreng also remains the most under-annotated class (165 annotated images,
  167 boxes) versus Char Kuey Teow (551 / 561), so the gap is primarily a
  data-balance and label-consistency problem.
- Roti Canai has the lowest precision (0.767); the confusion matrix shows the
  largest background false-positive count for Roti Canai.
- Chicken Rice again has strong mAP50 (0.949) but the lowest mAP50–95 (0.690),
  pointing to box tightness/localization rather than recognition.
- Laksa is strongest (mAP50 0.982, mAP50–95 0.822).

Local diagnostic hashes:

- confusion matrix: `81bbafe2e9c7479bd4c41a491b1c5254b03a7139ff886ee5358bf0196df113ca`
- normalized confusion matrix: `d93805c3524cac85258e4504c8a40eb57a057031fa18fa6ff426ff8d41008aef`

## Decision and next gate

The run passes the interim assisted-labelling gate. Batch 005 was prepared from
`runs/detect/dataset3_interim_v3/weights/best.pt` for human-reviewed proposals.
Continue the assisted-labelling loop while requiring human correction on every
frame, prioritizing Mee Goreng and Roti Canai recruitment and Char Kuey
Teow/Mee Goreng disagreements.

Do not copy this checkpoint to `data/weights/best.pt` and do not run
`split=test`. After batch 005, Dataset3 still has 2,761 missing annotations, so
the locked holdout remains untouched until the final annotation freeze and
validation-only model selection are complete. For the next interim retrain,
lower `lr0` and consider `yolo11s` if the small model is capacity-limited on the
noodle classes. Prioritize genuine Mee Goreng recruitment in later assisted
batches.
