# Dataset3 YOLO11n Interim — `dataset3_interim_v2`

**Training completed:** 20 July 2026  
**Assessment:** accepted for assisted-batch-003 proposals; not approved for production inference

## Configuration and artifacts

- initialization: `runs/detect/dataset3_pilot_v1/weights/best.pt`
- dataset: `data/dataset3-interim-v2/data.yaml`
- split: 1,067 train, 267 validation, 82 locked reviewed test images
- image size: 640
- batch size: 16
- requested epochs: 100
- early-stopping patience: 20
- completed epochs: 95
- seed: 42, deterministic mode enabled
- accelerator: HPC NVIDIA GPU (`device=0`)
- training Ultralytics version: `8.4.100`
- run directory: `runs/detect/dataset3_interim_v2/`
- best checkpoint SHA-256: `0babcfc246b9af4c003277a4f50bc33c79f4a32b34c4473ee0aa0d36329f3705`
- last checkpoint SHA-256: `ac2e38555979d8f9749156450218c24f7ca4325ea00ec1f4bf52d72603d30dee`
- results CSV SHA-256: `547ff3c928f725f5a140bd2744c0db6b8303df83323d9fd10698516f7ea44ca3`
- args SHA-256: `0e158f2128de39b8efebce7d8dd83ce9cb26a383781871a880a7e8205dc9f226`

The run correctly used `resume: false` with the prior checkpoint as
initialization. The checkpoint embeds the canonical class IDs: `nasi_lemak`,
`roti_canai`, `char_kuey_teow`, `chicken_rice`, `laksa`, and `mee_goreng` at
IDs 0–5.

## Training result

Ultralytics `8.4.100` uses mAP50–95 as detection fitness. The best checkpoint
is epoch 75, where mAP50–95 reached 0.74679. No later epoch improved that value,
so the configured 20-epoch patience stopped training normally after epoch 95.
The run was not interrupted.

| Checkpoint/epoch | Precision | Recall | mAP50 | mAP50–95 |
|------------------|----------:|-------:|------:|----------:|
| Best, epoch 75 | 0.912 | 0.868 | 0.938 | 0.747 |
| Last, epoch 95 | 0.914 | 0.900 | 0.947 | 0.737 |

Compared with the Phase C checkpoint's authoritative validation result, mAP50
increased from 0.925 to 0.938 and mAP50–95 from 0.689 to 0.747. Precision is
lower by 0.016 and recall is effectively unchanged. These comparisons are
directional because the interim validation set is larger, although every
surviving Phase B validation assignment was preserved.

Training loss continued to decline after epoch 75 while validation box and DFL
loss later rose. Selecting epoch 75 therefore avoids the mild localization
overfit visible near the end of the run.

## Validation-only local review

The best checkpoint was revalidated locally on the 267-image validation split
using Ultralytics `8.4.87`, PyTorch `2.12.1`, and CPU. This pass did not access
the locked test split. Version differences cause a small aggregate difference,
so the embedded HPC metrics above remain authoritative; the local pass is used
for per-class diagnosis.

| Class | Validation instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|---------------------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 34 | 0.967 | 0.850 | 0.957 | 0.830 |
| Roti Canai | 35 | 0.848 | 0.886 | 0.897 | 0.694 |
| Char Kuey Teow | 62 | 0.875 | 0.789 | 0.905 | 0.704 |
| Chicken Rice | 75 | 0.990 | 0.893 | 0.978 | 0.664 |
| Laksa | 42 | 0.919 | 1.000 | 0.993 | 0.834 |
| Mee Goreng | 28 | 0.916 | 0.786 | 0.921 | 0.766 |
| **All classes** | **276** | **0.919** | **0.867** | **0.942** | **0.749** |

Relative to the earlier local pilot validation, every class has higher
mAP50–95. Char Kuey Teow increased from 0.572 to 0.704 while its validation
support doubled from 31 to 62 objects. Nasi Lemak and Roti Canai support rose
from 10 and 11 objects to 34 and 35, making their interim estimates less
fragile.

## Error review

- Char Kuey Teow and Mee Goreng remain the dominant semantic confusion: six
  Char Kuey Teow instances were predicted as Mee Goreng and four Mee Goreng
  instances as Char Kuey Teow in the default confusion matrix.
- Char Kuey Teow and Mee Goreng recall remain below 0.80.
- Chicken Rice has excellent mAP50 but the lowest mAP50–95, pointing to box
  tightness/localization rather than basic class recognition as its main issue.
- Laksa is strongest, with 1.000 recall and 0.834 mAP50–95.
- Of 36 background false-positive detections in the matrix, 12 were Char Kuey
  Teow. This is the distribution of background errors, not a class false-positive
  rate.
- Side-by-side validation galleries expose several visually borderline noodle
  examples, including possible Char Kuey Teow/Mee Goreng annotation
  inconsistencies. They should be routed into a targeted audit rather than
  silently relabelled from model output.

Local diagnostic hashes:

- confusion matrix: `cb46b2884a368e4e1d39431d36db6cded6a00048cc4877e51417ad02f59bfb7e`
- normalized confusion matrix: `fe7c0d6f776225d2d9302e79baad82e2e4ba5b57126086fb4c35e7831dbd3823`

## Decision and next gate

The run passes the interim assisted-labelling gate. Batch 003 was prepared from
`runs/detect/dataset3_interim_v2/weights/best.pt`, reviewed, and applied.
Continue with later assisted batches while requiring human correction on every
frame. Prioritize noodle disagreements and Chicken Rice box tightness during
review.

Do not copy this checkpoint to `data/weights/best.pt` and do not run
`split=test`. After batch 003, Dataset3 still has 3,561 missing annotations, so
the locked holdout remains untouched until the final annotation freeze and
validation-only model selection are complete.
