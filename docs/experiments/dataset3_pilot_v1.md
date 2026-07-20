# Dataset3 YOLO11n Pilot — `dataset3_pilot_v1`

**Training completed:** 18 July 2026  
**Assessment:** suitable as an assisted-labelling baseline; not approved for production inference

## Configuration and artifacts

- initialization: Ultralytics `yolo11n.pt`
- dataset: `data/dataset3-baseline/data.yaml`
- split: 587 train, 167 validation, 84 unreviewed test candidates
- image size: 640
- batch size: 16
- epochs: 100
- seed: 42, deterministic mode enabled
- accelerator: HPC NVIDIA GPU (`device=0`)
- training Ultralytics version recorded in the checkpoint: `8.4.100`
- run directory: `runs/detect/dataset3_pilot_v1/`
- best checkpoint SHA-256: `42b87d4e47bf30c47936e1797ca0b6b1c3e8b264029e087387745319882dc5f8`
- results CSV SHA-256: `813f613536601ef23b5cfaefa2f78a0d0a3f5be080b9171e7c2154aff3947fd8`

The checkpoint class map was verified against the canonical contract: IDs 0–5 are `nasi_lemak`, `roti_canai`, `char_kuey_teow`, `chicken_rice`, `laksa`, and `mee_goreng`.

## Training result

`best.pt` corresponds to epoch 85, selected by validation fitness.

| Checkpoint/epoch | Precision | Recall | mAP50 | mAP50–95 |
|------------------|----------:|-------:|------:|----------:|
| Best, epoch 85 | 0.928 | 0.867 | 0.925 | 0.689 |
| Final, epoch 100 | 0.838 | 0.891 | 0.897 | 0.655 |

Training loss continued to decrease, but validation box and distribution focal losses rose late in training. The best checkpoint correctly avoids the mild overfitting visible after approximately epoch 80–85.

## Per-class validation review

The repository checkpoint was revalidated locally on the 167-image validation split to obtain per-class metrics. This pass used Ultralytics `8.4.90`, whereas training used `8.4.100`; therefore its small aggregate difference from the HPC result is expected and these values are directional. The HPC checkpoint metrics above remain authoritative.

| Class | Validation instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|---------------------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 10 | 0.972 | 0.900 | 0.895 | 0.760 |
| Roti Canai | 11 | 0.941 | 0.909 | 0.905 | 0.603 |
| Char Kuey Teow | 31 | 0.818 | 0.742 | 0.818 | 0.572 |
| Chicken Rice | 63 | 1.000 | 0.979 | 0.995 | 0.650 |
| Laksa | 30 | 0.965 | 0.915 | 0.988 | 0.787 |
| Mee Goreng | 26 | 0.841 | 0.808 | 0.929 | 0.731 |
| **All classes** | **171** | **0.923** | **0.875** | **0.922** | **0.684** |

Nasi Lemak and Roti Canai have only 10 and 11 validation objects, so their class metrics have high uncertainty. They must not be interpreted as stable production estimates.

## Qualitative and confusion-matrix findings

- Chicken Rice and Laksa are the strongest validation classes.
- Char Kuey Teow is the weakest class and has several low-confidence detections or Chicken Rice/Mee Goreng confusions.
- Of the validation detections assigned to background false positives, 44% were predicted as Char Kuey Teow. This is a distribution of background errors, not a 44% false-positive rate.
- The raw matrix includes two Roti Canai objects predicted as Mee Goreng and two Laksa objects predicted as Mee Goreng.
- The galleries expose a few missed packets, visually borderline noodle dishes, and likely annotation-policy inconsistencies that should be prioritized during assisted labelling.

## Decision and gate

Phase C passes its pilot gate: all six class IDs load correctly, the training and validation pipelines operate end to end, and a reproducible baseline is recorded. The model is useful for generating CVAT proposals that a human must correct.

Production promotion is blocked by evidence quality, not by a pipeline failure:

1. the later audit accepted 82 of the 84 test candidates and
   `data/dataset3-interim-v2/` now locks the 82 images in 81 leakage groups;
2. At the time of this pilot, thousands of Dataset3 images still needed
   annotations, so this is not the final detector,
   production-training corpus;
3. the model has not been evaluated once on the accepted untouched test set;
4. class support is imbalanced, particularly for Nasi Lemak and Roti Canai;
5. Char Kuey Teow needs targeted error reduction.

Do not copy this checkpoint to `data/weights/best.pt`. Use it to initialize the
new interim run without `resume=True`, keep the locked test split untouched
during assisted-labelling cycles, and run final holdout evaluation exactly once
after annotation freeze and validation-only model selection.
