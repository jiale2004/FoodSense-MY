#!/usr/bin/env python3
"""Optuna hyperparameter optimization for YOLOv11n training."""

import argparse
import logging
import sys
from pathlib import Path

import optuna
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ReproducibilityManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA = Path("data/dataset/data.yaml")
BASE_MODEL = "yolo11n.pt"


class YoloHyperparameterTuner:
    def __init__(
        self,
        data_yaml: Path,
        n_trials: int = 20,
        epochs: int = 30,
        seed: int = 42,
    ) -> None:
        self.data_yaml = data_yaml
        self.n_trials = n_trials
        self.epochs = epochs
        self.seed = seed
        self.device = self._select_device()
        ReproducibilityManager(seed=seed).seed_everything()

    def _select_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _objective(self, trial: optuna.Trial) -> float:
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.6, 0.98)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        mixup = trial.suggest_float("mixup", 0.0, 0.3)
        mosaic = trial.suggest_float("mosaic", 0.5, 1.0)

        model = YOLO(BASE_MODEL)
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            mixup=mixup,
            mosaic=mosaic,
            device=self.device,
            seed=self.seed,
            verbose=False,
            plots=False,
        )

        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        logger.info(
            "Trial %d: lr0=%.5f momentum=%.3f wd=%.5f mixup=%.2f mosaic=%.2f -> mAP50-95=%.4f",
            trial.number, lr0, momentum, weight_decay, mixup, mosaic, map50_95,
        )
        return float(map50_95)

    def optimize(self) -> dict:
        study = optuna.create_study(direction="maximize", study_name="yolo11n_tune")
        study.optimize(self._objective, n_trials=self.n_trials)

        logger.info("Best trial: %d", study.best_trial.number)
        logger.info("Best mAP50-95: %.4f", study.best_value)
        logger.info("Best params: %s", study.best_params)
        return study.best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for YOLOv11n")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.data.exists():
        parser.error(f"data.yaml not found at {args.data}. Run prepare_dataset.py first.")

    tuner = YoloHyperparameterTuner(
        data_yaml=args.data,
        n_trials=args.n_trials,
        epochs=args.epochs,
        seed=args.seed,
    )
    best = tuner.optimize()
    print("\nBest hyperparameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
