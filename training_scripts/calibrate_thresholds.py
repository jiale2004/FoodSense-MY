#!/usr/bin/env python3
"""Calibrate confidence and NMS-IoU thresholds on a validation split only.

This script never evaluates the locked test split. It runs the detector on the
validation images at a low confidence floor, matches predictions to ground truth
at a fixed evaluation IoU, and sweeps a confidence grid to find the operating
point that maximizes macro-averaged F1. A small NMS-IoU grid is also swept.

Outputs a deterministic JSON report (and prints a summary table) recommending a
single global confidence threshold plus per-class diagnostics. The application
consumes one global ``confidence_threshold`` and ``iou_threshold`` (NMS), so the
recommended global values map directly onto ``app/core/config.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data_yaml(data_yaml: Path) -> tuple[Path, list[str]]:
    spec = yaml.safe_load(data_yaml.read_text())
    root = Path(spec["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names = spec["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    return root, list(names)


def gather_split(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    pairs: list[tuple[Path, Path]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in exts:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))
    return pairs


def load_gt(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (classes[int], boxes[N,4] xyxy pixels) for a YOLO label file."""
    if not label_path.exists():
        return np.zeros((0,), dtype=int), np.zeros((0, 4), dtype=float)
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:5])
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height
        classes.append(cls)
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0,), dtype=int), np.zeros((0, 4), dtype=float)
    return np.asarray(classes, dtype=int), np.asarray(boxes, dtype=float)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a[N,4] and b[M,4] in xyxy; returns [N,M]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def collect_predictions(
    model: YOLO,
    pairs: list[tuple[Path, Path]],
    conf_floor: float,
    nms_iou: float,
    imgsz: int,
    device: str,
    predict_batch_size: int,
) -> list[dict[str, Any]]:
    """Run inference and return per-image predictions and ground truth."""
    records: list[dict[str, Any]] = []
    image_paths = [str(p[0]) for p in pairs]
    label_paths = [p[1] for p in pairs]
    idx = 0
    for start in range(0, len(image_paths), predict_batch_size):
        chunk = image_paths[start : start + predict_batch_size]
        results = model.predict(
            source=chunk,
            conf=conf_floor,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
        )
        for result in results:
            height, width = result.orig_shape
            gt_cls, gt_boxes = load_gt(label_paths[idx], width, height)
            if result.boxes is not None and len(result.boxes) > 0:
                pred_cls = result.boxes.cls.cpu().numpy().astype(int)
                pred_conf = result.boxes.conf.cpu().numpy().astype(float)
                pred_boxes = result.boxes.xyxy.cpu().numpy().astype(float)
            else:
                pred_cls = np.zeros((0,), dtype=int)
                pred_conf = np.zeros((0,), dtype=float)
                pred_boxes = np.zeros((0, 4), dtype=float)
            records.append(
                {
                    "gt_cls": gt_cls,
                    "gt_boxes": gt_boxes,
                    "pred_cls": pred_cls,
                    "pred_conf": pred_conf,
                    "pred_boxes": pred_boxes,
                }
            )
            idx += 1
    return records


def evaluate_threshold(
    records: list[dict[str, Any]],
    num_classes: int,
    conf_thr: float,
    eval_iou: float,
) -> dict[int, dict[str, int]]:
    """Greedy per-class TP/FP/FN counting at a confidence threshold."""
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)}
    for rec in records:
        keep = rec["pred_conf"] >= conf_thr
        pred_cls = rec["pred_cls"][keep]
        pred_conf = rec["pred_conf"][keep]
        pred_boxes = rec["pred_boxes"][keep]
        gt_cls = rec["gt_cls"]
        gt_boxes = rec["gt_boxes"]
        for c in range(num_classes):
            p_mask = pred_cls == c
            g_mask = gt_cls == c
            n_pred = int(p_mask.sum())
            n_gt = int(g_mask.sum())
            if n_pred == 0:
                counts[c]["fn"] += n_gt
                continue
            if n_gt == 0:
                counts[c]["fp"] += n_pred
                continue
            pb = pred_boxes[p_mask]
            pc = pred_conf[p_mask]
            gb = gt_boxes[g_mask]
            order = np.argsort(-pc)
            ious = iou_matrix(pb[order], gb)
            gt_taken = np.zeros(n_gt, dtype=bool)
            tp = 0
            for i in range(ious.shape[0]):
                candidates = np.where((ious[i] >= eval_iou) & (~gt_taken))[0]
                if candidates.size == 0:
                    continue
                best = candidates[np.argmax(ious[i, candidates])]
                gt_taken[best] = True
                tp += 1
            counts[c]["tp"] += tp
            counts[c]["fp"] += n_pred - tp
            counts[c]["fn"] += n_gt - tp
    return counts


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(counts: dict[int, dict[str, int]], present: list[int]) -> float:
    if not present:
        return 0.0
    return float(np.mean([prf(counts[c]["tp"], counts[c]["fp"], counts[c]["fn"])[2] for c in present]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml")
    parser.add_argument("--split", default="val", help="split to calibrate on (test is refused)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-iou", type=float, default=0.5, help="IoU for TP matching")
    parser.add_argument(
        "--nms-iou-grid",
        type=float,
        nargs="+",
        default=[0.45, 0.6, 0.7],
        help="candidate NMS IoU values",
    )
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--conf-min", type=float, default=0.05)
    parser.add_argument("--conf-max", type=float, default=0.95)
    parser.add_argument("--conf-step", type=float, default=0.01)
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit("Refusing to calibrate on the locked test split.")

    root, names = load_data_yaml(args.data)
    num_classes = len(names)
    pairs = gather_split(root, args.split)
    if not pairs:
        raise SystemExit(f"No images found for split '{args.split}' under {root}")

    print(f"Calibrating on {len(pairs)} '{args.split}' images ({num_classes} classes).")
    print(f"Model: {args.model}")
    print(f"NMS-IoU grid: {args.nms_iou_grid} | eval IoU: {args.eval_iou}")

    model = YOLO(str(args.model))
    conf_grid = [
        round(args.conf_min + i * args.conf_step, 4)
        for i in range(int(round((args.conf_max - args.conf_min) / args.conf_step)) + 1)
    ]

    best_overall: dict[str, Any] | None = None
    per_nms_summary: list[dict[str, Any]] = []

    for nms_iou in args.nms_iou_grid:
        print(f"\n[NMS IoU {nms_iou}] running inference at conf floor {args.conf_floor} ...")
        records = collect_predictions(
            model, pairs, args.conf_floor, nms_iou, args.imgsz, args.device, args.predict_batch_size
        )
        present = sorted({int(c) for rec in records for c in rec["gt_cls"].tolist()})

        best_for_nms: dict[str, Any] | None = None
        curve: list[dict[str, float]] = []
        for conf_thr in conf_grid:
            counts = evaluate_threshold(records, num_classes, conf_thr, args.eval_iou)
            m_f1 = macro_f1(counts, present)
            tot_tp = sum(counts[c]["tp"] for c in range(num_classes))
            tot_fp = sum(counts[c]["fp"] for c in range(num_classes))
            tot_fn = sum(counts[c]["fn"] for c in range(num_classes))
            micro_p, micro_r, micro_f1 = prf(tot_tp, tot_fp, tot_fn)
            curve.append(
                {
                    "conf": conf_thr,
                    "macro_f1": round(m_f1, 5),
                    "micro_f1": round(micro_f1, 5),
                    "micro_precision": round(micro_p, 5),
                    "micro_recall": round(micro_r, 5),
                }
            )
            if best_for_nms is None or m_f1 > best_for_nms["macro_f1"]:
                best_for_nms = {
                    "nms_iou": nms_iou,
                    "conf": conf_thr,
                    "macro_f1": round(m_f1, 5),
                    "micro_precision": round(micro_p, 5),
                    "micro_recall": round(micro_r, 5),
                    "micro_f1": round(micro_f1, 5),
                    "per_class": {
                        names[c]: {
                            **{k: counts[c][k] for k in ("tp", "fp", "fn")},
                            **dict(
                                zip(
                                    ("precision", "recall", "f1"),
                                    (round(v, 5) for v in prf(counts[c]["tp"], counts[c]["fp"], counts[c]["fn"])),
                                )
                            ),
                        }
                        for c in range(num_classes)
                    },
                }

        # Per-class best confidence (independent), diagnostic only.
        per_class_best: dict[str, dict[str, float]] = {}
        for c in range(num_classes):
            best_c: dict[str, float] | None = None
            for conf_thr in conf_grid:
                counts = evaluate_threshold(records, num_classes, conf_thr, args.eval_iou)
                p, r, f1 = prf(counts[c]["tp"], counts[c]["fp"], counts[c]["fn"])
                if best_c is None or f1 > best_c["f1"]:
                    best_c = {"conf": conf_thr, "precision": round(p, 5), "recall": round(r, 5), "f1": round(f1, 5)}
            per_class_best[names[c]] = best_c or {}
        best_for_nms["per_class_best_conf"] = per_class_best

        per_nms_summary.append(
            {
                "nms_iou": nms_iou,
                "best_conf": best_for_nms["conf"],
                "macro_f1": best_for_nms["macro_f1"],
                "micro_precision": best_for_nms["micro_precision"],
                "micro_recall": best_for_nms["micro_recall"],
                "micro_f1": best_for_nms["micro_f1"],
                "curve": curve,
            }
        )
        print(
            f"[NMS IoU {nms_iou}] best conf={best_for_nms['conf']} macro-F1={best_for_nms['macro_f1']} "
            f"(micro P={best_for_nms['micro_precision']} R={best_for_nms['micro_recall']})"
        )

        if best_overall is None or best_for_nms["macro_f1"] > best_overall["macro_f1"]:
            best_overall = best_for_nms

    report = {
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "data_yaml": str(args.data),
        "data_yaml_sha256": sha256_file(args.data),
        "split": args.split,
        "num_images": len(pairs),
        "num_classes": num_classes,
        "class_names": names,
        "eval_iou": args.eval_iou,
        "conf_grid": {"min": args.conf_min, "max": args.conf_max, "step": args.conf_step},
        "recommended": {
            "confidence_threshold": best_overall["conf"],
            "iou_threshold": best_overall["nms_iou"],
            "macro_f1": best_overall["macro_f1"],
            "micro_precision": best_overall["micro_precision"],
            "micro_recall": best_overall["micro_recall"],
            "micro_f1": best_overall["micro_f1"],
        },
        "best_operating_point": best_overall,
        "nms_iou_summary": per_nms_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Recommended operating point (validation split only) ===")
    print(f"  confidence_threshold = {best_overall['conf']}")
    print(f"  iou_threshold (NMS)  = {best_overall['nms_iou']}")
    print(
        f"  macro-F1={best_overall['macro_f1']}  micro P={best_overall['micro_precision']} "
        f"R={best_overall['micro_recall']} F1={best_overall['micro_f1']}"
    )
    print("\nPer-class at recommended global threshold:")
    print(f"  {'class':<16}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for cls_name, m in best_overall["per_class"].items():
        print(
            f"  {cls_name:<16}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
            f"{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}"
        )
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
