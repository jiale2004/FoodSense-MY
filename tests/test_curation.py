import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_scripts"))

from curation import (  # noqa: E402
    CalibrationClassModel,
    CalibrationModel,
    CurationConfig,
    ImageCurator,
    ManualDecision,
    SemanticConfig,
    ValidationConfig,
    build_download_record,
    sha256_file,
    train_calibration_model,
)
from scrape_images import ImageScraper  # noqa: E402


class FakeScorer:
    def __init__(self, scores_by_name):
        self.scores_by_name = scores_by_name

    def score(self, paths):
        return [self.scores_by_name[path.name] for path in paths]


class FakeEmbeddingScorer:
    def __init__(self, embeddings_by_name):
        self.embeddings_by_name = embeddings_by_name

    def embed(self, paths):
        import numpy as np

        return np.asarray([self.embeddings_by_name[path.name] for path in paths], dtype=np.float32)

    def scores_from_embeddings(self, embeddings):
        return [
            {"laksa": float(row[0]), "mee_goreng": float(row[1])}
            for row in embeddings
        ]


def create_pattern(path: Path, invert: bool = False) -> None:
    image = Image.new("RGB", (320, 320), "white" if not invert else "black")
    pixels = image.load()
    for y in range(320):
        for x in range(320):
            if (x // 20 + y // 20) % 2:
                pixels[x, y] = (20, 80, 180) if not invert else (220, 170, 40)
    image.save(path, quality=95)


class CurationTests(unittest.TestCase):
    def config(self) -> CurationConfig:
        return CurationConfig(
            classes={"laksa": ["laksa"], "mee_goreng": ["mee goreng"]},
            validation=ValidationConfig(min_blur_variance=0, min_file_bytes=1),
            dhash_distance=1,
            semantic=SemanticConfig(
                batch_size=8,
                accept_min_score=0.22,
                accept_min_margin=0.02,
                reject_max_score=0.16,
                reject_wrong_class_margin=0.04,
            ),
        )

    def test_routes_images_and_detects_global_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for class_name in ("laksa", "mee_goreng"):
                (source / class_name).mkdir(parents=True)

            accepted = source / "laksa" / "accepted.jpg"
            duplicate = source / "mee_goreng" / "duplicate.jpg"
            mismatch = source / "mee_goreng" / "mismatch.jpg"
            create_pattern(accepted)
            duplicate.write_bytes(accepted.read_bytes())
            create_pattern(mismatch, invert=True)

            scorer = FakeScorer({
                "accepted.jpg": {"laksa": 0.30, "mee_goreng": 0.18},
                "mismatch.jpg": {"laksa": 0.31, "mee_goreng": 0.20},
            })
            curator = ImageCurator(
                source,
                root / "curation",
                self.config(),
                scorer,
                materialize="none",
                run_id="test-run",
            )
            records, summary = curator.run()
            by_name = {Path(record.source_path).name: record for record in records}

            self.assertEqual(by_name["accepted.jpg"].status, "accepted")
            self.assertEqual(by_name["duplicate.jpg"].status, "duplicate")
            self.assertEqual(by_name["mismatch.jpg"].status, "rejected")
            self.assertIn("semantic_class_mismatch:laksa", by_name["mismatch.jpg"].reasons)
            self.assertEqual(summary["total"], 3)
            manifest = root / "curation" / "runs" / "test-run" / "curation.jsonl"
            self.assertEqual(len(manifest.read_text().splitlines()), 3)

    def test_corrupt_image_is_rejected_without_semantic_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "laksa"
            source.mkdir(parents=True)
            (source / "broken.jpg").write_bytes(b"not an image")
            curator = ImageCurator(
                root / "source",
                root / "curation",
                self.config(),
                scorer=None,
                materialize="none",
                run_id="broken-run",
            )
            records, _ = curator.run()
            self.assertEqual(records[0].status, "rejected")
            self.assertTrue(any(reason.startswith("decode_error:") for reason in records[0].reasons))

    def test_exact_duplicate_is_detected_even_when_technically_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "laksa"
            source.mkdir(parents=True)
            (source / "broken-a.jpg").write_bytes(b"same invalid image")
            (source / "broken-b.jpg").write_bytes(b"same invalid image")
            curator = ImageCurator(
                root / "source",
                root / "curation",
                self.config(),
                scorer=None,
                materialize="none",
                run_id="duplicate-reject-run",
            )
            records, _ = curator.run()
            self.assertEqual([record.status for record in records], ["rejected", "duplicate"])
            self.assertIn("exact_duplicate", records[1].reasons)

    def test_download_record_includes_source_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.jpg"
            create_pattern(path)
            record = build_download_record(
                path,
                candidate_class="laksa",
                query="Malaysian laksa",
                engine="uc",
                source_url="https://Example.COM/images/laksa.jpg",
            )
            self.assertEqual(record["source_domain"], "example.com")
            self.assertEqual(record["candidate_class"], "laksa")
            json.dumps(record)

    def test_scraper_appends_manifest_for_new_downloads(self):
        class FakeCrawler:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir

            def crawl(self, keyword: str, max_num: int) -> None:
                create_pattern(self.output_dir / "000001.jpg")

        class TestScraper(ImageScraper):
            def _create_crawler(self, class_dir: Path, engine: str | None = None):
                return FakeCrawler(class_dir)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifests" / "downloads.jsonl"
            scraper = TestScraper(
                output_dir=root / "raw",
                keywords={"laksa": "Malaysian laksa"},
                max_images=1,
                engine="bing",
                manifest_path=manifest,
            )
            self.assertEqual(scraper.scrape_all(), 1)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["query"], "Malaysian laksa")
            self.assertEqual(records[0]["engine"], "bing")
            self.assertEqual(records[0]["candidate_class"], "laksa")

    def test_calibrated_mode_preserves_decisions_and_uses_one_review_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source" / "laksa"
            source.mkdir(parents=True)
            decided = source / "decided.jpg"
            new_image = source / "new.jpg"
            create_pattern(decided)
            create_pattern(new_image, invert=True)

            decision = ManualDecision(
                candidate_class="laksa",
                sha256=sha256_file(decided),
                image_id=sha256_file(decided)[:16],
                source_path=str(decided),
                label="accepted",
            )
            class_model = CalibrationClassModel(
                positive_centroid=[1.0, 0.0],
                negative_centroid=[0.0, 1.0],
                threshold=0.5,
                validation_precision=1.0,
                validation_recall=1.0,
                validation_coverage=0.5,
                accepted_validation_count=5,
                positive_count=5,
                negative_count=5,
            )
            calibration = CalibrationModel(
                source_run="pilot",
                target_precision=0.98,
                folds=5,
                classes={"laksa": class_model},
            )
            scorer = FakeEmbeddingScorer({"new.jpg": [0.0, 1.0]})
            config = self.config()
            config.dhash_distance = -1
            curator = ImageCurator(
                root / "source",
                root / "curation",
                config,
                scorer,
                materialize="none",
                run_id="calibrated-run",
                manual_decisions=[decision],
                calibration=calibration,
            )
            records, summary = curator.run()
            by_name = {Path(record.source_path).name: record for record in records}
            self.assertEqual(by_name["decided.jpg"].status, "accepted")
            self.assertIn("manual_decision_override:accepted", by_name["decided.jpg"].reasons)
            self.assertEqual(by_name["new.jpg"].status, "manual_review")
            self.assertTrue(summary["calibrated"])
            run_dir = root / "curation" / "runs" / "calibrated-run"
            self.assertTrue((run_dir / "manual_decisions.jsonl").exists())
            self.assertTrue((run_dir / "calibration.json").exists())

    def test_centroid_calibration_meets_precision_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decisions = []
            embeddings = {}
            for index in range(12):
                path = root / f"sample-{index}.jpg"
                path.write_bytes(b"placeholder")
                accepted = index < 6
                embeddings[path.name] = [0.95, 0.05] if accepted else [0.05, 0.95]
                decisions.append(
                    ManualDecision(
                        candidate_class="laksa",
                        sha256=f"{index:064x}",
                        image_id=f"{index:016x}",
                        source_path=str(path),
                        label="accepted" if accepted else "rejected",
                    )
                )
            model = train_calibration_model(
                decisions,
                FakeEmbeddingScorer(embeddings),
                source_run=root,
                target_precision=1.0,
                folds=3,
                minimum_predictions=2,
                batch_size=4,
            )
            laksa = model.classes["laksa"]
            self.assertEqual(laksa.validation_precision, 1.0)
            self.assertGreater(laksa.validation_recall, 0.0)
            self.assertGreaterEqual(
                laksa.score(__import__("numpy").array([1.0, 0.0])),
                laksa.threshold,
            )


if __name__ == "__main__":
    unittest.main()
