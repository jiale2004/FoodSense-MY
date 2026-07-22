# Dataset3 YOLO11n Interim — `dataset3_interim_v4`

**Training completed:** 22 July 2026
**Assessment:** accepted for assisted-labelling proposals; not approved for production inference

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

## Configuration and training artifacts

- initialization: `runs/detect/dataset3_interim_v3/weights/best.pt`
- dataset: `data/dataset3-interim-v4/data.yaml`
- image size: 640
- batch size: 16
- requested epochs: 100
- early-stopping patience: 20
- completed epochs: 21
- `lr0`: 0.002 (lowered from the interim-v3 default 0.01)
- seed: 42, deterministic mode enabled
- accelerator: HPC NVIDIA GPU (`device=0`)
- run directory: `runs/detect/dataset3_interim_v4/`
- best checkpoint SHA-256: `c9aa14e8ee171a1e6f57e2b4c9572e9348c3c9af97a84bb4c6d969647ed087fb`
- last checkpoint SHA-256: `1e59df0011b974247750e484102a60b6d853e2910d1769a36931689a02db783a`
- results CSV SHA-256: `d9f1d688593540f336b2397666abdb7bbacc06c7d81af62f612705b7d6ff2958`
- args SHA-256: `f0ecb725cb73cee6a3e830dc7fcb59e6c35f7fc65f7ee2ffbfa3eafba106f34b`
- HPC confusion matrix SHA-256: `eabe3ae7675a404a56da329a32948fb35c931ae38dcae856526fa52c24ad656b`
- HPC normalized confusion matrix SHA-256: `4033aa37964c04f9705bb6fe9ce72ee31d4f226e22ad2e05ce595040285b31a2`

The run correctly used `resume: false` with the interim v3 checkpoint as
initialization and `lr0=0.002`. The checkpoint embeds the canonical class IDs:
`nasi_lemak`, `roti_canai`, `char_kuey_teow`, `chicken_rice`, `laksa`, and
`mee_goreng` at IDs 0–5.

## Training result

Ultralytics uses mAP50–95 as detection fitness. Best mAP50–95 (0.7834) was
again reached at epoch 1. No later epoch improved fitness, so the configured
20-epoch patience stopped training normally after epoch 21. The run was not
interrupted. Validation mAP50 peaked later at epoch 20 (0.949), and last-epoch
recall (0.885) exceeded the best-checkpoint recall (0.848).

| Checkpoint/epoch | Precision | Recall | mAP50 | mAP50–95 |
|------------------|----------:|-------:|------:|----------:|
| Best, epoch 1 | 0.911 | 0.848 | 0.938 | 0.783 |
| Last, epoch 21 | 0.899 | 0.885 | 0.940 | 0.776 |
| Peak mAP50, epoch 20 | 0.927 | 0.876 | 0.949 | 0.782 |

Relative to the interim v3 embedded validation result, mAP50–95 increased from
0.761 to 0.783 while mAP50 rose from 0.932 to 0.938. These comparisons are
directional because the interim v4 validation set is larger (825 vs 418
images), although every surviving interim v3 train/validation assignment was
preserved.

Lowering `lr0` to 0.002 did not move the fitness peak off epoch 1, but the
mid-run dip recovered more cleanly than interim v3, and last-epoch mAP50–95
(0.776) stayed close to the best checkpoint rather than collapsing to 0.737 as
in interim v3.

## Validation-only local review

The best checkpoint was revalidated locally on the 825-image validation split
using Ultralytics `8.4.90`, PyTorch `2.12.1`, and CPU. This pass did not access
the locked test split. Version differences can cause a small aggregate
difference, so the embedded HPC metrics above remain authoritative; the local
pass is used for per-class diagnosis.

| Class | Validation instances | Precision | Recall | mAP50 | mAP50–95 |
|-------|---------------------:|----------:|-------:|------:|----------:|
| Nasi Lemak | 150 | 0.992 | 0.835 | 0.969 | 0.851 |
| Roti Canai | 161 | 0.864 | 0.888 | 0.928 | 0.751 |
| Char Kuey Teow | 205 | 0.925 | 0.838 | 0.949 | 0.823 |
| Chicken Rice | 143 | 0.918 | 0.863 | 0.946 | 0.698 |
| Laksa | 151 | 0.893 | 0.967 | 0.975 | 0.826 |
| Mee Goreng | 60 | 0.861 | 0.717 | 0.863 | 0.746 |
| **All classes** | **870** | **0.909** | **0.851** | **0.938** | **0.783** |

## Error review

- Mee Goreng improved substantially: local recall rose from 0.513 (interim v3)
  to 0.717, and mAP50–95 from 0.731 to 0.746. Remaining confusion still routes
  true Mee Goreng into Char Kuey Teow (~18% on the HPC normalized matrix).
- Char Kuey Teow ↔ Mee Goreng remains the dominant inter-class disagreement;
  reverse confusion (Char → Mee) is lower (~6%).
- Roti Canai still shows the largest background false-positive share on the
  normalized matrix (~33% of background predictions).
- Chicken Rice again has strong mAP50 (0.946) but the lowest mAP50–95 (0.698),
  pointing to box tightness/localization rather than recognition.
- Laksa remains strongest on recall (0.967) with high mAP50 (0.975).
- Nasi Lemak precision is excellent (0.992) with moderate recall (0.835).

Local diagnostic hashes:

- confusion matrix: `727acbb8d36169f7f4625cb3d4eb468a00520e0671d8ab33935925ab0f652106`
- normalized confusion matrix: `7a4744da83e607376a6f72cc1189c508c5ae790a63d086fb0772f6f1d4aafd2e`

## Decision and next gate

The run passes the interim assisted-labelling gate. Use
`runs/detect/dataset3_interim_v4/weights/best.pt` for subsequent assisted-batch
proposals. Continue human correction on every frame; prioritize remaining
missing annotations (Laksa, Mee Goreng, Nasi Lemak, Roti Canai) and Char Kuey
Teow / Mee Goreng disagreements.

Do not copy this checkpoint to `data/weights/best.pt` and do not run
`split=test`. Dataset3 still has 1,153 missing annotations, so the locked
holdout remains untouched until the final annotation freeze and validation-only
model selection are complete. Further fine-tuning may try a still-lower `lr0`
or shorter warmup if epoch-1 fitness selection persists; consider `yolo11s` only
if the nano model remains capacity-limited on the noodle classes.
