#!/usr/bin/env python3
"""Generate multi-page results PDFs for interim detector runs (v5-style)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "logs"

INK = "#1f2933"
MUTED = "#52606d"
LINE = "#cbd2d9"
ACCENT = "#243b53"
OK = "#2f6f4e"
WARN = "#8a5a00"
INFO = "#3e4c59"
SERIES = ["#243b53", "#486581", "#829ab1", "#9fb3c8"]

CLASSES = ["Nasi Lemak", "Roti Canai", "Char Kuey Teow", "Chicken Rice", "Laksa", "Mee Goreng"]
SHORT = ["NL", "RC", "CKT", "CR", "LK", "MG"]
SPLIT_TRAIN = [779, 755, 908, 533, 811, 345]
SPLIT_VAL = [192, 186, 237, 133, 203, 82]
SPLIT_TEST = [5, 5, 15, 31, 15, 11]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e4e7eb",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def load_results_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty results CSV: {path}")
    keys = {
        "epoch": "epoch",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map5095": "metrics/mAP50-95(B)",
    }
    data = {name: [float(row[column].strip()) for row in rows] for name, column in keys.items()}
    data["epoch"] = [int(v) for v in data["epoch"]]
    return data


def style_table(table) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(fontweight="bold", color=INK)
        elif row == 1:
            cell.set_facecolor("#f5f7fa")
        else:
            cell.set_facecolor("white")


def page_footer(fig, text: str, page: str, y: float = 0.04) -> None:
    fig.text(0.08, y, text, fontsize=7, color=MUTED)
    fig.text(0.92, y, page, ha="right", fontsize=7, color=MUTED)


def add_headline_page(
    pdf: PdfPages,
    *,
    title: str,
    subtitle: str,
    metrics: list[tuple[str, str]],
    checkpoints: list[list[str]],
    sources: str,
    page: str,
    pdf_title: str,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=0.96)
    fig.text(0.5, 0.91, subtitle, ha="center", color=MUTED, fontsize=10)

    ax = fig.add_axes([0.08, 0.58, 0.84, 0.28])
    ax.axis("off")
    for i, (val, lab) in enumerate(metrics):
        x = 0.02 + (i % 3) * 0.33
        y = 0.55 if i < 3 else 0.12
        ax.text(x, y + 0.22, val, fontsize=20, fontweight="bold", color=ACCENT, transform=ax.transAxes)
        ax.text(x, y, lab, fontsize=9, color=MUTED, transform=ax.transAxes)

    ax2 = fig.add_axes([0.08, 0.12, 0.84, 0.40])
    ax2.axis("off")
    ax2.set_title("Headline checkpoints", loc="left", fontsize=11, fontweight="bold", pad=8)
    table = ax2.table(
        cellText=checkpoints,
        colLabels=["Checkpoint", "Precision", "Recall", "mAP50", "mAP50–95"],
        cellLoc="center",
        colLoc="center",
        loc="upper center",
    )
    style_table(table)
    page_footer(fig, sources, page)
    pdf.savefig(fig)
    plt.close(fig)


def add_curves_page(
    pdf: PdfPages,
    data: dict[str, list[float]],
    *,
    best_epoch: int,
    best_map5095: float,
    peak_map50_epoch: int,
    peak_map50: float,
    source: str,
    page: str,
    y_map_min: float = 0.65,
    y_pr_min: float = 0.55,
) -> None:
    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    fig.suptitle("Training curves (validation metrics by epoch)", fontsize=14, fontweight="bold", y=0.96)

    ax = axes[0]
    ax.plot(epochs, data["map50"], color=SERIES[0], lw=1.8, label="mAP50", marker="o", ms=2.5)
    ax.plot(epochs, data["map5095"], color=SERIES[1], lw=1.8, label="mAP50–95", marker="o", ms=2.5)
    ax.axhline(best_map5095, color=INFO, ls="--", lw=1, label=f"Best mAP50–95 (ep {best_epoch})")
    ax.axhline(peak_map50, color=OK, ls=":", lw=1, label=f"Peak mAP50 (ep {peak_map50_epoch})")
    ax.set_ylabel("Score")
    ax.set_ylim(y_map_min, 1.0)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("mAP50 and mAP50–95", loc="left")

    ax = axes[1]
    ax.plot(epochs, data["precision"], color=SERIES[0], lw=1.8, label="Precision", marker="o", ms=2.5)
    ax.plot(epochs, data["recall"], color=SERIES[2], lw=1.8, label="Recall", marker="o", ms=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(y_pr_min, 1.0)
    step = max(1, len(epochs) // 20)
    ax.set_xticks(epochs[::step])
    ax.legend(loc="lower right")
    ax.set_title("Precision and recall", loc="left")

    page_footer(
        fig,
        f"Source: {source} · early-stop patience 20 · fitness = mAP50–95 · "
        f"best at epoch {best_epoch} · completed {epochs[-1]} epochs",
        page,
        y=0.03,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_split_bars(ax) -> None:
    x = np.arange(len(CLASSES))
    w = 0.25
    ax.bar(x - w, SPLIT_TRAIN, w, label="Train", color=SERIES[0])
    ax.bar(x, SPLIT_VAL, w, label="Val", color=SERIES[1])
    ax.bar(x + w, SPLIT_TEST, w, label="Test", color=SERIES[3])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Primary-class images")
    ax.set_title("Split composition (4,131 train / 1,033 val / 82 test)", loc="left")
    ax.legend(loc="upper right")


def generate_v6() -> Path:
    run = ROOT / "runs/detect/dataset3_interim_v6_s"
    data = load_results_csv(run / "results.csv")
    best_i = int(np.argmax(data["map5095"]))
    peak_i = int(np.argmax(data["map50"]))
    last_i = len(data["epoch"]) - 1
    out = LOGS / "dataset3_interim_v6_results.pdf"

    with PdfPages(out) as pdf:
        add_headline_page(
            pdf,
            title="FoodSense-MY · Dataset3 Interim v6 Results",
            subtitle="YOLO11s capacity run (v6_s) · Validation-only · Not promoted (nano production path)",
            metrics=[
                (f"{data['map5095'][best_i]:.3f}", "Best val mAP50–95"),
                (f"{data['map50'][best_i]:.3f}", "Best val mAP50"),
                (f"{data['precision'][best_i]:.3f}", "Best val Precision"),
                (f"{data['recall'][best_i]:.3f}", "Best val Recall"),
                ("0.878", "Val Mee Goreng recall"),
                ("yolo11s", "Model family"),
            ],
            checkpoints=[
                [
                    f"Best (epoch {data['epoch'][best_i]}, selected)",
                    f"{data['precision'][best_i]:.3f}",
                    f"{data['recall'][best_i]:.3f}",
                    f"{data['map50'][best_i]:.3f}",
                    f"{data['map5095'][best_i]:.3f}",
                ],
                [
                    f"Peak mAP50 (epoch {data['epoch'][peak_i]})",
                    f"{data['precision'][peak_i]:.3f}",
                    f"{data['recall'][peak_i]:.3f}",
                    f"{data['map50'][peak_i]:.3f}",
                    f"{data['map5095'][peak_i]:.3f}",
                ],
                [
                    f"Last (epoch {data['epoch'][last_i]})",
                    f"{data['precision'][last_i]:.3f}",
                    f"{data['recall'][last_i]:.3f}",
                    f"{data['map50'][last_i]:.3f}",
                    f"{data['map5095'][last_i]:.3f}",
                ],
                ["v5 nano baseline (val)", "0.894", "0.899", "0.945", "0.793"],
            ],
            sources=(
                "Sources: docs/experiments/dataset3_interim_v6.md · "
                "runs/detect/dataset3_interim_v6_s/results.csv"
            ),
            page="Page 1 / 3",
            pdf_title="v6",
        )
        add_curves_page(
            pdf,
            data,
            best_epoch=data["epoch"][best_i],
            best_map5095=data["map5095"][best_i],
            peak_map50_epoch=data["epoch"][peak_i],
            peak_map50=data["map50"][peak_i],
            source="runs/detect/dataset3_interim_v6_s/results.csv",
            page="Page 2 / 3",
            y_map_min=0.25,
            y_pr_min=0.35,
        )

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Decision context and locked interim-v5 split", fontsize=14, fontweight="bold", y=0.96)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28, left=0.08, right=0.96, top=0.88, bottom=0.10)

        ax = fig.add_subplot(gs[0, 0])
        labels = ["v5 nano", "v6_s"]
        vals = [0.793, data["map5095"][best_i]]
        colors = [SERIES[2], OK]
        ax.bar(labels, vals, color=colors)
        ax.set_ylim(0.75, 0.86)
        ax.set_ylabel("mAP50–95")
        ax.set_title("Validation mAP50–95 vs prior nano", loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9, color=INK)

        ax = fig.add_subplot(gs[0, 1])
        ax.axis("off")
        ax.set_title("Key findings", loc="left", fontsize=11, fontweight="bold", pad=8)
        ax.text(
            0.0,
            0.95,
            "\n".join(
                [
                    "Run B (yolo11s from COCO) completed on the locked",
                    "interim-v5 split (batch 16, seed 42).",
                    "",
                    f"• Best val mAP50–95 {data['map5095'][best_i]:.3f} at epoch {data['epoch'][best_i]}",
                    "• Strongest Mee Goreng recall among interim runs (0.878)",
                    "• Beats v5 nano aggregate (0.793) on validation",
                    "",
                    "Not used as production init: architecture mismatch",
                    "with the YOLO11n deployment path. Nano follow-ups",
                    "continued in interim v7 (freeze) → v8 (MG recovery).",
                    "",
                    "Run A (v6_n_cos) was the schedule/fine-tune attempt;",
                    "artifacts focus here on the completed v6_s capacity run.",
                ]
            ),
            va="top",
            fontsize=9,
            color=INK,
            linespacing=1.4,
        )

        ax = fig.add_subplot(gs[1, :])
        add_split_bars(ax)
        page_footer(
            fig,
            "Locked split reused from data/dataset3-interim-v5/ · no locked-test eval for v6_s",
            "Page 3 / 3",
            y=0.03,
        )
        pdf.savefig(fig)
        plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "FoodSense-MY Dataset3 Interim v6 Results"
        info["Author"] = "FoodSense-MY"
        info["Subject"] = "YOLO11s interim v6_s validation results"
    return out


def generate_v7() -> Path:
    run = ROOT / "runs/detect/dataset3_interim_v7_n_freeze"
    data = load_results_csv(run / "results.csv")
    best_i = int(np.argmax(data["map5095"]))
    peak_i = int(np.argmax(data["map50"]))
    last_i = len(data["epoch"]) - 1
    out = LOGS / "dataset3_interim_v7_results.pdf"

    with PdfPages(out) as pdf:
        add_headline_page(
            pdf,
            title="FoodSense-MY · Dataset3 Interim v7 Results",
            subtitle="YOLO11n freeze fine-tune (v7_n_freeze) · Validation-only · Not promoted (MG recall regress)",
            metrics=[
                (f"{data['map5095'][best_i]:.3f}", "Best val mAP50–95"),
                (f"{data['map50'][best_i]:.3f}", "Best val mAP50"),
                (f"{data['precision'][best_i]:.3f}", "Best val Precision"),
                (f"{data['recall'][best_i]:.3f}", "Best val Recall"),
                ("0.722", "Val Mee Goreng recall"),
                ("freeze=10", "Key lever"),
            ],
            checkpoints=[
                [
                    f"Best (epoch {data['epoch'][best_i]}, selected)",
                    f"{data['precision'][best_i]:.3f}",
                    f"{data['recall'][best_i]:.3f}",
                    f"{data['map50'][best_i]:.3f}",
                    f"{data['map5095'][best_i]:.3f}",
                ],
                [
                    f"Peak mAP50 (epoch {data['epoch'][peak_i]})",
                    f"{data['precision'][peak_i]:.3f}",
                    f"{data['recall'][peak_i]:.3f}",
                    f"{data['map50'][peak_i]:.3f}",
                    f"{data['map5095'][peak_i]:.3f}",
                ],
                [
                    f"Last (epoch {data['epoch'][last_i]})",
                    f"{data['precision'][last_i]:.3f}",
                    f"{data['recall'][last_i]:.3f}",
                    f"{data['map50'][last_i]:.3f}",
                    f"{data['map5095'][last_i]:.3f}",
                ],
                ["v5 nano baseline (val)", "0.894", "0.899", "0.945", "0.793"],
            ],
            sources=(
                "Sources: docs/experiments/dataset3_interim_v7.md · "
                "runs/detect/dataset3_interim_v7_n_freeze/results.csv"
            ),
            page="Page 1 / 3",
            pdf_title="v7",
        )
        add_curves_page(
            pdf,
            data,
            best_epoch=data["epoch"][best_i],
            best_map5095=data["map5095"][best_i],
            peak_map50_epoch=data["epoch"][peak_i],
            peak_map50=data["map50"][peak_i],
            source="runs/detect/dataset3_interim_v7_n_freeze/results.csv",
            page="Page 2 / 3",
            y_map_min=0.75,
            y_pr_min=0.80,
        )

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Mee Goreng tradeoff and next-step gate", fontsize=14, fontweight="bold", y=0.96)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28, left=0.08, right=0.96, top=0.88, bottom=0.10)

        ax = fig.add_subplot(gs[0, 0])
        names = ["v5", "v6_s", "v7 freeze"]
        map_vals = [0.793, 0.826, data["map5095"][best_i]]
        ax.bar(names, map_vals, color=[SERIES[2], SERIES[1], ACCENT])
        ax.set_ylim(0.78, 0.84)
        ax.set_ylabel("mAP50–95")
        ax.set_title("Validation mAP50–95 progression", loc="left")
        for i, v in enumerate(map_vals):
            ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

        ax = fig.add_subplot(gs[0, 1])
        mg = [0.778, 0.878, 0.722]
        colors = [SERIES[2], OK, WARN]
        ax.bar(names, mg, color=colors)
        ax.axhline(0.778, color=INFO, ls="--", lw=1, label="v5 MG recall gate")
        ax.set_ylim(0.65, 0.95)
        ax.set_ylabel("Recall")
        ax.set_title("Validation Mee Goreng recall", loc="left")
        ax.legend(loc="upper right")
        for i, v in enumerate(mg):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        ax.set_title("Findings", loc="left", fontsize=11, fontweight="bold", pad=8)
        ax.text(
            0.0,
            0.95,
            "\n".join(
                [
                    "freeze=10 + lr0=0.0002 + cos_lr escaped the",
                    "epoch-1 fitness peak seen on v3–v5 fine-tunes.",
                    "",
                    f"• Best at epoch {data['epoch'][best_i]}: mAP50–95 "
                    f"{data['map5095'][best_i]:.3f}",
                    "• Beats v5 aggregate (0.793) ✓",
                    "• Mee Goreng recall 0.722 < v5 0.778 ✗",
                    "• ~22% MG→CKT on HPC normalized matrix",
                    "• Chicken Rice mAP50–95 ~0.735 (still weak)",
                    "",
                    "Not promoted. Became the nano init for interim v8.",
                    "Run B (v7_n_box) was superseded by v8_n_box.",
                ]
            ),
            va="top",
            fontsize=9,
            color=INK,
            linespacing=1.4,
        )

        ax = fig.add_subplot(gs[1, 1])
        add_split_bars(ax)
        page_footer(
            fig,
            "Sources: docs/experiments/dataset3_interim_v7.md · v8 write-up prior-results table",
            "Page 3 / 3",
            y=0.03,
        )
        pdf.savefig(fig)
        plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "FoodSense-MY Dataset3 Interim v7 Results"
        info["Author"] = "FoodSense-MY"
        info["Subject"] = "YOLO11n interim v7_n_freeze validation results"
    return out


def generate_v8() -> Path:
    run = ROOT / "runs/detect/dataset3_interim_v8_n_mg"
    box_run = ROOT / "runs/detect/dataset3_interim_v8_n_box"
    data = load_results_csv(run / "results.csv")
    box = load_results_csv(box_run / "results.csv")
    test = json.loads((ROOT / "runs/detect/dataset3_interim_v8_n_mg_test/test-metrics.json").read_text())
    cal = json.loads((ROOT / "runs/detect/dataset3_interim_v8_n_mg_calibration/calibration.json").read_text())

    best_i = int(np.argmax(data["map5095"]))
    peak_i = int(np.argmax(data["map50"]))
    last_i = len(data["epoch"]) - 1
    box_best_i = int(np.argmax(box["map5095"]))
    out = LOGS / "dataset3_interim_v8_results.pdf"

    order = ["nasi_lemak", "roti_canai", "char_kuey_teow", "chicken_rice", "laksa", "mee_goreng"]
    test_map50 = [test["per_class"][k]["mAP50"] for k in order]
    test_map5095 = [test["per_class"][k]["mAP50_95"] for k in order]
    test_p = [test["per_class"][k]["precision"] for k in order]
    test_r = [test["per_class"][k]["recall"] for k in order]
    test_n = [test["per_class"][k]["instances"] for k in order]

    # Local validation per-class from experiment write-up (best.pt).
    val_map50 = [0.976, 0.936, 0.967, 0.970, 0.992, 0.916]
    val_map5095 = [0.896, 0.772, 0.869, 0.751, 0.876, 0.823]
    val_p = [0.973, 0.927, 0.908, 0.951, 0.963, 0.899]
    val_r = [0.946, 0.892, 0.955, 0.947, 0.981, 0.822]

    rec = cal["recommended"]
    per = cal["best_operating_point"]["per_class"]
    cal_f1 = [per[k]["f1"] for k in order]
    cal_p = [per[k]["precision"] for k in order]
    cal_r = [per[k]["recall"] for k in order]

    with PdfPages(out) as pdf:
        add_headline_page(
            pdf,
            title="FoodSense-MY · Dataset3 Interim v8 Results",
            subtitle="Production-approved YOLO11n detector (v8_n_mg) · Locked-test evaluation 28 Jul 2026",
            metrics=[
                (f"{test['overall']['mAP50']:.3f}", "Test mAP50"),
                (f"{test['overall']['mAP50_95']:.3f}", "Test mAP50–95"),
                (f"{test['overall']['precision']:.3f}", "Test Precision"),
                (f"{test['overall']['recall']:.3f}", "Test Recall"),
                (
                    f"{rec['confidence_threshold']:.2g} / {rec['iou_threshold']:.2g}",
                    "Conf / NMS-IoU",
                ),
                (f"{data['map5095'][best_i]:.3f}", "Best val mAP50–95"),
            ],
            checkpoints=[
                [
                    f"Best v8_n_mg (epoch {data['epoch'][best_i]})",
                    f"{data['precision'][best_i]:.3f}",
                    f"{data['recall'][best_i]:.3f}",
                    f"{data['map50'][best_i]:.3f}",
                    f"{data['map5095'][best_i]:.3f}",
                ],
                [
                    f"Last v8_n_mg (epoch {data['epoch'][last_i]})",
                    f"{data['precision'][last_i]:.3f}",
                    f"{data['recall'][last_i]:.3f}",
                    f"{data['map50'][last_i]:.3f}",
                    f"{data['map5095'][last_i]:.3f}",
                ],
                [
                    f"Best v8_n_box (epoch {box['epoch'][box_best_i]})",
                    f"{box['precision'][box_best_i]:.3f}",
                    f"{box['recall'][box_best_i]:.3f}",
                    f"{box['map50'][box_best_i]:.3f}",
                    f"{box['map5095'][box_best_i]:.3f}",
                ],
                [
                    "Locked test (one-shot)",
                    f"{test['overall']['precision']:.3f}",
                    f"{test['overall']['recall']:.3f}",
                    f"{test['overall']['mAP50']:.3f}",
                    f"{test['overall']['mAP50_95']:.3f}",
                ],
            ],
            sources=(
                "Sources: docs/experiments/dataset3_interim_v8.md · "
                "runs/detect/dataset3_interim_v8_n_mg_test/test-metrics.json · "
                "runs/detect/dataset3_interim_v8_n_mg/results.csv"
            ),
            page="Page 1 / 4",
            pdf_title="v8",
        )

        add_curves_page(
            pdf,
            data,
            best_epoch=data["epoch"][best_i],
            best_map5095=data["map5095"][best_i],
            peak_map50_epoch=data["epoch"][peak_i],
            peak_map50=data["map50"][peak_i],
            source="runs/detect/dataset3_interim_v8_n_mg/results.csv",
            page="Page 2 / 4",
            y_map_min=0.65,
            y_pr_min=0.70,
        )

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Per-class locked-test and validation metrics", fontsize=14, fontweight="bold", y=0.96)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28, left=0.08, right=0.96, top=0.88, bottom=0.12)
        x = np.arange(len(CLASSES))
        w = 0.35

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(x - w / 2, test_map50, w, label="mAP50", color=SERIES[0])
        ax.bar(x + w / 2, test_map5095, w, label="mAP50–95", color=SERIES[2])
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Locked-test mAP by class", loc="left")
        ax.legend(loc="lower right")

        ax = fig.add_subplot(gs[0, 1])
        ax.bar(x - w / 2, val_map5095, w, label="Validation", color=SERIES[1])
        ax.bar(x + w / 2, test_map5095, w, label="Locked test", color=SERIES[3])
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mAP50–95")
        ax.set_title("Localization gap (mAP50–95)", loc="left")
        ax.legend(loc="lower right")

        ax = fig.add_subplot(gs[1, 0])
        ax.bar(x - w / 2, test_p, w, label="Precision", color=SERIES[0])
        ax.bar(x + w / 2, test_r, w, label="Recall", color=SERIES[2])
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Locked-test precision / recall", loc="left")
        ax.legend(loc="lower right")

        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        ax.set_title("Locked-test per-class table", loc="left", fontsize=11, fontweight="bold", pad=8)
        rows = []
        for name, n, p, r, m50, m95 in zip(CLASSES, test_n, test_p, test_r, test_map50, test_map5095):
            rows.append([name, str(n), f"{p:.3f}", f"{r:.3f}", f"{m50:.3f}", f"{m95:.3f}"])
        tbl = ax.table(
            cellText=rows,
            colLabels=["Class", "n", "P", "R", "mAP50", "mAP50–95"],
            cellLoc="center",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.45)
        for (row, _col), cell in tbl.get_celld().items():
            cell.set_edgecolor(LINE)
            if row == 0:
                cell.set_facecolor("#f0f4f8")
                cell.set_text_props(fontweight="bold")
            elif row == 4:  # Chicken Rice
                cell.set_facecolor("#fff7ed")
            else:
                cell.set_facecolor("white")

        page_footer(
            fig,
            "NL=Nasi Lemak · RC=Roti Canai · CKT=Char Kuey Teow · CR=Chicken Rice · LK=Laksa · MG=Mee Goreng. "
            "Chicken Rice lowest test mAP50–95 (0.518) and recall 0.733 — monitor. Val MG recall recovered to 0.822.",
            "Page 3 / 4",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Threshold calibration (validation) and split composition", fontsize=14, fontweight="bold", y=0.96)
        gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28, left=0.08, right=0.96, top=0.88, bottom=0.10)

        ax = fig.add_subplot(gs[0, 0])
        w = 0.25
        ax.bar(x - w, cal_f1, w, label="F1", color=SERIES[0])
        ax.bar(x, cal_p, w, label="Precision", color=SERIES[1])
        ax.bar(x + w, cal_r, w, label="Recall", color=SERIES[2])
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_ylim(0.80, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Per-class metrics at conf {rec['confidence_threshold']} / IoU {rec['iou_threshold']}",
            loc="left",
        )
        ax.legend(loc="lower right", ncol=3, fontsize=8)

        ax = fig.add_subplot(gs[0, 1])
        ax.axis("off")
        ax.set_title("Calibration + promotion summary", loc="left", fontsize=11, fontweight="bold", pad=8)
        ax.text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"Recommended: confidence {rec['confidence_threshold']}, "
                    f"NMS-IoU {rec['iou_threshold']}",
                    f"Macro-F1 {rec['macro_f1']:.3f} · Micro P {rec['micro_precision']:.3f} / "
                    f"R {rec['micro_recall']:.3f} / F1 {rec['micro_f1']:.3f}",
                    "Swept on 1,033-image validation split only",
                    "Locked test refused by calibrate_thresholds.py",
                    "",
                    "Nano winner: v8_n_mg over v8_n_box",
                    f"  v8_n_mg val mAP50–95 {data['map5095'][best_i]:.3f}, MG recall 0.822",
                    f"  v8_n_box val mAP50–95 {box['map5095'][box_best_i]:.3f}, MG recall 0.807",
                    "",
                    "Promoted to data/weights/best.pt",
                    "SHA-256 f0cda9e1…95ee412",
                ]
            ),
            va="top",
            fontsize=9,
            color=INK,
            linespacing=1.4,
        )

        ax = fig.add_subplot(gs[1, :])
        add_split_bars(ax)
        page_footer(
            fig,
            "Sources: calibration.json · test-metrics.json · dataset3-interim-v5 primary-class counts · "
            "promoted weights SHA-256 f0cda9e1…95ee412",
            "Page 4 / 4",
            y=0.03,
        )
        pdf.savefig(fig)
        plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "FoodSense-MY Dataset3 Interim v8 Results"
        info["Author"] = "FoodSense-MY"
        info["Subject"] = "YOLO11n interim v8_n_mg training, calibration, and locked-test evaluation"
        info["Keywords"] = "YOLO11n, dataset3, interim-v8, mAP, calibration"
    return out


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    outputs = [generate_v6(), generate_v7(), generate_v8()]
    for path in outputs:
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
