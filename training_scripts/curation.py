"""Non-destructive image validation, deduplication, and semantic curation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_download_record(
    path: Path,
    candidate_class: str,
    query: str,
    engine: str,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Build a provenance record for one downloaded image."""
    width: int | None = None
    height: int | None = None
    decode_error: str | None = None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        decode_error = str(exc)

    sha256 = sha256_file(path)
    return {
        "image_id": sha256[:16],
        "candidate_class": candidate_class,
        "query": query,
        "source_url": source_url,
        "source_domain": urlparse(source_url).netloc.lower() if source_url else None,
        "engine": engine,
        "local_path": str(path),
        "downloaded_at": utc_now(),
        "sha256": sha256,
        "file_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "decode_error": decode_error,
    }


@dataclass(slots=True)
class ValidationConfig:
    min_width: int = 256
    min_height: int = 256
    min_file_bytes: int = 2048
    max_pixels: int = 40_000_000
    max_aspect_ratio: float = 4.0
    min_blur_variance: float = 20.0


@dataclass(slots=True)
class SemanticConfig:
    enabled: bool = True
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    batch_size: int = 32
    accept_min_score: float = 0.22
    accept_min_margin: float = 0.02
    reject_max_score: float = 0.16
    reject_wrong_class_margin: float = 0.04


@dataclass(slots=True)
class CurationConfig:
    classes: dict[str, list[str]]
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    dhash_distance: int = 6
    semantic: SemanticConfig = field(default_factory=SemanticConfig)

    @classmethod
    def load(cls, path: Path) -> "CurationConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_classes = raw.get("classes", {})
        classes = {
            name: list(value.get("prompts", []))
            for name, value in raw_classes.items()
        }
        if not classes or any(not prompts for prompts in classes.values()):
            raise ValueError("Every configured class must contain at least one prompt.")
        return cls(
            classes=classes,
            validation=ValidationConfig(**raw.get("validation", {})),
            dhash_distance=int(raw.get("deduplication", {}).get("dhash_distance", 6)),
            semantic=SemanticConfig(**raw.get("semantic", {})),
        )


@dataclass(slots=True)
class CurationRecord:
    image_id: str
    source_path: str
    candidate_class: str
    sha256: str
    file_bytes: int
    width: int | None = None
    height: int | None = None
    dhash: str | None = None
    blur_variance: float | None = None
    status: str = "pending"
    reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    semantic_scores: dict[str, float] = field(default_factory=dict)
    predicted_class: str | None = None
    intended_score: float | None = None
    score_margin: float | None = None
    calibration_score: float | None = None
    calibration_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a compact perceptual dHash for near-duplicate detection."""
    grayscale = ImageOps.grayscale(image)
    resized = grayscale.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    """Metric tree used to query perceptual hashes without an O(n²) scan."""

    def __init__(self) -> None:
        self.root: tuple[int, str, dict[int, Any]] | None = None

    def add(self, value: int, image_id: str) -> None:
        if self.root is None:
            self.root = (value, image_id, {})
            return
        node = self.root
        while True:
            node_value, _, children = node
            distance = hamming_distance(value, node_value)
            child = children.get(distance)
            if child is None:
                children[distance] = (value, image_id, {})
                return
            node = child

    def find(self, value: int, max_distance: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        matches: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node_value, image_id, children = stack.pop()
            distance = hamming_distance(value, node_value)
            if distance <= max_distance:
                matches.append((distance, image_id))
            lower = distance - max_distance
            upper = distance + max_distance
            stack.extend(child for edge, child in children.items() if lower <= edge <= upper)
        return sorted(matches)


class SemanticScorer(Protocol):
    def score(self, paths: list[Path]) -> list[dict[str, float]]:
        """Return one class-to-score mapping per path."""


class OpenClipScorer:
    """OpenCLIP-backed semantic scorer with averaged prompt prototypes."""

    def __init__(self, classes: dict[str, list[str]], config: SemanticConfig) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Semantic filtering requires open-clip-torch. Install training requirements "
                "or rerun with --skip-semantic."
            ) from exc

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
            device=self.device,
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(config.model_name)
        self.class_names = list(classes)

        text_features = []
        with torch.inference_mode():
            for class_name in self.class_names:
                tokens = tokenizer(classes[class_name]).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                prototype = features.mean(dim=0)
                prototype = prototype / prototype.norm()
                text_features.append(prototype)
        self.text_features = torch.stack(text_features)

    def embed(self, paths: list[Path]) -> np.ndarray:
        """Return normalized CLIP image embeddings."""
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.preprocess(image.convert("RGB")))
        batch = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return np.asarray(features.cpu().tolist(), dtype=np.float32)

    def scores_from_embeddings(self, embeddings: np.ndarray) -> list[dict[str, float]]:
        tensor = self.torch.from_numpy(embeddings).to(self.device)
        with self.torch.inference_mode():
            similarities = tensor @ self.text_features.T
        return [
            {name: round(float(value), 6) for name, value in zip(self.class_names, row)}
            for row in similarities.cpu().tolist()
        ]

    def score(self, paths: list[Path]) -> list[dict[str, float]]:
        return self.scores_from_embeddings(self.embed(paths))


@dataclass(slots=True)
class ManualDecision:
    candidate_class: str
    sha256: str
    image_id: str
    source_path: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_manual_decisions(run_dir: Path) -> list[ManualDecision]:
    """Load accepted/rejected folder moves as authoritative pilot decisions."""
    manifest_path = run_dir / "curation.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Curation manifest not found: {manifest_path}")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    by_materialized_name = {
        (
            record["candidate_class"],
            record["image_id"] + Path(record["source_path"]).suffix.lower(),
        ): record
        for record in records
    }
    assignments: dict[tuple[str, str], str] = {}
    unknown: list[str] = []
    for label in ("accepted", "rejected"):
        root = run_dir / label
        if not root.exists():
            continue
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                key = (class_dir.name, path.name)
                if key not in by_materialized_name:
                    unknown.append(str(path))
                    continue
                previous = assignments.get(key)
                if previous and previous != label:
                    raise ValueError(f"Pilot image appears in multiple decision folders: {path.name}")
                assignments[key] = label

    unresolved: list[str] = []
    for folder_name in ("review", "manual_review"):
        root = run_dir / folder_name
        if not root.exists():
            continue
        unresolved.extend(
            str(path)
            for path in root.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
    if unresolved:
        raise ValueError(
            f"Pilot still has {len(unresolved)} unresolved review image(s); finish sorting first."
        )
    if unknown:
        raise ValueError(f"Found {len(unknown)} decision image(s) not present in the pilot manifest.")
    missing_decisions = [
        key
        for key, record in by_materialized_name.items()
        if record.get("status") != "duplicate" and key not in assignments
    ]
    if missing_decisions:
        raise ValueError(
            f"Pilot has {len(missing_decisions)} non-duplicate image(s) without a manual decision."
        )

    decisions: list[ManualDecision] = []
    for key, label in sorted(assignments.items()):
        record = by_materialized_name[key]
        decisions.append(
            ManualDecision(
                candidate_class=record["candidate_class"],
                sha256=record["sha256"],
                image_id=record["image_id"],
                source_path=record["source_path"],
                label=label,
            )
        )
    if not decisions:
        raise ValueError(f"No accepted/rejected manual decisions found in {run_dir}")
    return decisions


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length calibration vector.")
    return vector / norm


def _centroid_score(
    embeddings: np.ndarray,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> np.ndarray:
    embeddings_64 = np.asarray(embeddings, dtype=np.float64)
    positive_64 = np.asarray(positive_centroid, dtype=np.float64)
    negative_64 = np.asarray(negative_centroid, dtype=np.float64)
    if not (
        np.isfinite(embeddings_64).all()
        and np.isfinite(positive_64).all()
        and np.isfinite(negative_64).all()
    ):
        raise ValueError("Calibration received a non-finite CLIP embedding.")
    positive_similarity = np.sum(embeddings_64 * positive_64, axis=1)
    negative_similarity = np.sum(embeddings_64 * negative_64, axis=1)
    return positive_similarity - negative_similarity


@dataclass(slots=True)
class CalibrationClassModel:
    positive_centroid: list[float]
    negative_centroid: list[float]
    threshold: float
    validation_precision: float
    validation_recall: float
    validation_coverage: float
    accepted_validation_count: int
    positive_count: int
    negative_count: int

    def score(self, embedding: np.ndarray) -> float:
        positive = np.asarray(self.positive_centroid, dtype=np.float32)
        negative = np.asarray(self.negative_centroid, dtype=np.float32)
        return float(_centroid_score(embedding.reshape(1, -1), positive, negative)[0])


@dataclass(slots=True)
class CalibrationModel:
    source_run: str
    target_precision: float
    folds: int
    classes: dict[str, CalibrationClassModel]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run": self.source_run,
            "target_precision": self.target_precision,
            "folds": self.folds,
            "classes": {name: asdict(model) for name, model in self.classes.items()},
        }


def _stratified_folds(labels: np.ndarray, folds: int, seed: int = 42) -> np.ndarray:
    assignments = np.empty(len(labels), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignments[index] = position % folds
    return assignments


def _select_precision_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_precision: float,
    minimum_predictions: int,
) -> tuple[float, float, float, float, int]:
    candidates = sorted({float(score) for score in scores}, reverse=True)
    selected: tuple[float, float, float, float, int] | None = None
    positives = int(labels.sum())
    for threshold in candidates:
        predicted = scores >= threshold
        count = int(predicted.sum())
        if count < minimum_predictions:
            continue
        true_positive = int(labels[predicted].sum())
        precision = true_positive / count
        if precision < target_precision:
            continue
        recall = true_positive / positives if positives else 0.0
        coverage = count / len(labels)
        candidate = (threshold, precision, recall, coverage, count)
        if selected is None or count > selected[4]:
            selected = candidate
    if selected is not None:
        return selected
    return float(scores.max()) + 1e-6, 1.0, 0.0, 0.0, 0


def train_calibration_model(
    decisions: list[ManualDecision],
    scorer: OpenClipScorer,
    source_run: Path,
    target_precision: float = 0.98,
    folds: int = 5,
    minimum_predictions: int = 5,
    batch_size: int = 32,
) -> CalibrationModel:
    """Fit per-class CLIP centroid classifiers with out-of-fold thresholds."""
    if not 0.5 <= target_precision <= 1.0:
        raise ValueError("target_precision must be between 0.5 and 1.0")
    grouped: dict[str, list[ManualDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.candidate_class, []).append(decision)

    class_models: dict[str, CalibrationClassModel] = {}
    for class_name, class_decisions in sorted(grouped.items()):
        labels = np.asarray([decision.label == "accepted" for decision in class_decisions], dtype=np.int32)
        positive_count = int(labels.sum())
        negative_count = int(len(labels) - positive_count)
        if min(positive_count, negative_count) < 2:
            raise ValueError(
                f"Calibration class {class_name!r} needs at least two accepted and two rejected images."
            )
        class_folds = min(folds, positive_count, negative_count)
        fold_assignments = _stratified_folds(labels, class_folds)

        paths = [Path(decision.source_path) for decision in class_decisions]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pilot source image(s) are missing.")
        embedding_batches = [
            scorer.embed(paths[start : start + batch_size])
            for start in range(0, len(paths), batch_size)
        ]
        embeddings = np.concatenate(embedding_batches, axis=0)
        oof_scores = np.zeros(len(labels), dtype=np.float32)
        for fold in range(class_folds):
            train = fold_assignments != fold
            test = ~train
            positive_centroid = _normalize_vector(embeddings[train & (labels == 1)].mean(axis=0))
            negative_centroid = _normalize_vector(embeddings[train & (labels == 0)].mean(axis=0))
            oof_scores[test] = _centroid_score(
                embeddings[test], positive_centroid, negative_centroid
            )

        threshold, precision, recall, coverage, accepted_count = _select_precision_threshold(
            oof_scores,
            labels,
            target_precision=target_precision,
            minimum_predictions=min(minimum_predictions, positive_count),
        )
        final_positive = _normalize_vector(embeddings[labels == 1].mean(axis=0))
        final_negative = _normalize_vector(embeddings[labels == 0].mean(axis=0))
        full_scores = _centroid_score(embeddings, final_positive, final_negative)
        full_threshold, _, _, _, _ = _select_precision_threshold(
            full_scores,
            labels,
            target_precision=target_precision,
            minimum_predictions=min(minimum_predictions, positive_count),
        )
        deployment_threshold = max(threshold, full_threshold)
        class_models[class_name] = CalibrationClassModel(
            positive_centroid=final_positive.astype(float).tolist(),
            negative_centroid=final_negative.astype(float).tolist(),
            threshold=round(deployment_threshold, 8),
            validation_precision=round(precision, 6),
            validation_recall=round(recall, 6),
            validation_coverage=round(coverage, 6),
            accepted_validation_count=accepted_count,
            positive_count=positive_count,
            negative_count=negative_count,
        )

    return CalibrationModel(
        source_run=str(source_run),
        target_precision=target_precision,
        folds=folds,
        classes=class_models,
    )


def validate_image(path: Path, candidate_class: str, config: ValidationConfig) -> CurationRecord:
    """Inspect one image and return its technical validation record."""
    file_bytes = path.stat().st_size
    sha256 = sha256_file(path)
    record = CurationRecord(
        image_id=sha256[:16],
        source_path=str(path),
        candidate_class=candidate_class,
        sha256=sha256,
        file_bytes=file_bytes,
    )
    if file_bytes < config.min_file_bytes:
        record.reasons.append("file_too_small")

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            record.width = width
            record.height = height
            record.dhash = f"{difference_hash(image):016x}"
            if width < config.min_width:
                record.reasons.append("width_below_minimum")
            if height < config.min_height:
                record.reasons.append("height_below_minimum")
            if width * height > config.max_pixels:
                record.reasons.append("pixel_count_above_maximum")
            ratio = max(width / max(height, 1), height / max(width, 1))
            if ratio > config.max_aspect_ratio:
                record.reasons.append("extreme_aspect_ratio")

            grayscale = np.asarray(ImageOps.grayscale(image))
            record.blur_variance = round(float(cv2.Laplacian(grayscale, cv2.CV_64F).var()), 3)
            if record.blur_variance < config.min_blur_variance:
                record.reasons.append("likely_blurry")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        record.reasons.append(f"decode_error:{type(exc).__name__}")

    hard_reasons = [reason for reason in record.reasons if reason != "likely_blurry"]
    if hard_reasons:
        record.status = "rejected"
    return record


def route_semantic_record(record: CurationRecord, config: SemanticConfig) -> None:
    if not record.semantic_scores:
        record.status = "review"
        record.reasons.append("semantic_score_missing")
        return

    ranked = sorted(record.semantic_scores.items(), key=lambda item: item[1], reverse=True)
    predicted_class, best_score = ranked[0]
    intended_score = record.semantic_scores.get(record.candidate_class)
    if intended_score is None:
        record.status = "review"
        record.reasons.append("candidate_class_not_configured")
        return

    best_competing = max(
        (score for name, score in ranked if name != record.candidate_class),
        default=-1.0,
    )
    margin = intended_score - best_competing
    record.predicted_class = predicted_class
    record.intended_score = round(intended_score, 6)
    record.score_margin = round(margin, 6)

    if intended_score >= config.accept_min_score and margin >= config.accept_min_margin:
        record.status = "review" if "likely_blurry" in record.reasons else "accepted"
        if record.status == "review":
            record.reasons.append("manual_quality_review_required")
        return
    if best_score < config.reject_max_score:
        record.status = "rejected"
        record.reasons.append("semantic_score_too_low")
        return
    if predicted_class != record.candidate_class and -margin >= config.reject_wrong_class_margin:
        record.status = "rejected"
        record.reasons.append(f"semantic_class_mismatch:{predicted_class}")
        return
    record.status = "review"
    record.reasons.append("semantic_score_borderline")


class ImageCurator:
    """Run non-destructive curation over a class-folder image dataset."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: CurationConfig,
        scorer: SemanticScorer | None,
        materialize: str = "hardlink",
        run_id: str | None = None,
        limit_per_class: int | None = None,
        manual_decisions: list[ManualDecision] | None = None,
        calibration: CalibrationModel | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.scorer = scorer
        self.materialize = materialize
        self.limit_per_class = limit_per_class
        self.manual_decisions = manual_decisions or []
        self.decision_overrides = {
            (decision.candidate_class, decision.sha256): decision.label
            for decision in self.manual_decisions
        }
        self.calibration = calibration
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = output_dir / "runs" / self.run_id

    def collect(self) -> list[tuple[Path, str]]:
        items: list[tuple[Path, str]] = []
        for class_name in self.config.classes:
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                continue
            paths = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if self.limit_per_class is not None:
                paths = paths[: self.limit_per_class]
            items.extend((path, class_name) for path in paths)
        return items

    def _deduplicate(self, records: list[CurationRecord]) -> None:
        exact: dict[str, str] = {}
        tree = BKTree()
        for record in records:
            if record.sha256 in exact:
                record.status = "duplicate"
                record.duplicate_of = exact[record.sha256]
                record.reasons.append("exact_duplicate")
                continue
            exact[record.sha256] = record.image_id
            if record.status == "rejected" or record.dhash is None:
                continue
            hash_value = int(record.dhash, 16)
            matches = tree.find(hash_value, self.config.dhash_distance)
            if matches:
                record.status = "duplicate"
                record.duplicate_of = matches[0][1]
                record.reasons.append(f"perceptual_duplicate_distance:{matches[0][0]}")
                continue
            tree.add(hash_value, record.image_id)

    def _score(self, records: list[CurationRecord]) -> None:
        pending = [record for record in records if record.status == "pending"]
        if not pending:
            return
        if self.scorer is None:
            for record in pending:
                record.status = "manual_review" if self.calibration else "review"
                record.reasons.append("semantic_filter_skipped")
            return

        if self.calibration is not None:
            self._score_calibrated(pending)
            return

        batch_size = self.config.semantic.batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            scores = self.scorer.score([Path(record.source_path) for record in batch])
            if len(scores) != len(batch):
                raise RuntimeError("Semantic scorer returned an unexpected number of results.")
            for record, result in zip(batch, scores):
                record.semantic_scores = result
                route_semantic_record(record, self.config.semantic)

    def _score_calibrated(self, pending: list[CurationRecord]) -> None:
        if not hasattr(self.scorer, "embed") or not hasattr(self.scorer, "scores_from_embeddings"):
            raise TypeError("Calibrated curation requires a scorer with CLIP embedding support.")
        batch_size = self.config.semantic.batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            embeddings = self.scorer.embed([Path(record.source_path) for record in batch])
            semantic_scores = self.scorer.scores_from_embeddings(embeddings)
            for record, embedding, scores in zip(batch, embeddings, semantic_scores):
                record.semantic_scores = scores
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                record.predicted_class = ranked[0][0]
                record.intended_score = scores.get(record.candidate_class)
                competing = [score for name, score in ranked if name != record.candidate_class]
                if record.intended_score is not None and competing:
                    record.score_margin = round(record.intended_score - max(competing), 6)

                class_model = self.calibration.classes.get(record.candidate_class)
                if class_model is None:
                    record.status = "manual_review"
                    record.reasons.append("calibration_class_missing")
                    continue
                calibration_score = class_model.score(embedding)
                record.calibration_score = round(calibration_score, 8)
                record.calibration_threshold = class_model.threshold
                if "likely_blurry" in record.reasons:
                    record.status = "manual_review"
                    record.reasons.append("manual_quality_review_required")
                elif calibration_score >= class_model.threshold:
                    record.status = "accepted"
                    record.reasons.append("calibrated_auto_accept")
                else:
                    record.status = "manual_review"
                    record.reasons.append("calibrated_below_accept_threshold")

    def _apply_manual_decisions(self, records: list[CurationRecord]) -> None:
        for record in records:
            label = self.decision_overrides.get((record.candidate_class, record.sha256))
            if label is None:
                continue
            record.status = label
            record.reasons.append(f"manual_decision_override:{label}")

    def _materialize(self, record: CurationRecord) -> None:
        if self.materialize == "none":
            return
        source = Path(record.source_path)
        destination_dir = self.run_dir / record.status / record.candidate_class
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{record.image_id}{source.suffix.lower()}"
        if destination.exists():
            return
        if self.materialize == "copy":
            shutil.copy2(source, destination)
            return
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def run(self) -> tuple[list[CurationRecord], dict[str, Any]]:
        if self.run_dir.exists():
            raise FileExistsError(f"Curation run already exists: {self.run_dir}")

        records = [validate_image(path, class_name, self.config.validation) for path, class_name in self.collect()]
        self._deduplicate(records)
        self._apply_manual_decisions(records)
        self._score(records)

        self.run_dir.mkdir(parents=True)
        for record in records:
            self._materialize(record)

        manifest_path = self.run_dir / "curation.jsonl"
        append_jsonl(manifest_path, (record.to_dict() for record in records))
        if self.manual_decisions:
            append_jsonl(
                self.run_dir / "manual_decisions.jsonl",
                (decision.to_dict() for decision in self.manual_decisions),
            )
        if self.calibration is not None:
            (self.run_dir / "calibration.json").write_text(
                json.dumps(self.calibration.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        status_counts: dict[str, int] = {}
        class_counts: dict[str, dict[str, int]] = {}
        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
            per_class = class_counts.setdefault(record.candidate_class, {})
            per_class[record.status] = per_class.get(record.status, 0) + 1
        summary = {
            "run_id": self.run_id,
            "created_at": utc_now(),
            "input_dir": str(self.input_dir),
            "manifest": str(manifest_path),
            "semantic_enabled": self.scorer is not None,
            "materialize": self.materialize,
            "total": len(records),
            "status_counts": status_counts,
            "class_counts": class_counts,
            "manual_decision_overrides": len(self.manual_decisions),
            "calibrated": self.calibration is not None,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return records, summary
