#!/usr/bin/env python3
"""Scrape supplemental food images using icrawler or SeleniumBase UC mode."""

import argparse
import logging
from pathlib import Path

from icrawler.builtin import BingImageCrawler

from google_crawler import FixedGoogleImageCrawler
from curation import append_jsonl, build_download_record
from uc_crawler import UCImageScraper, _normalize_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = {
    "nasi_lemak": "nasi lemak malaysian food",
    "roti_canai": "roti canai malaysian mamak",
    "char_kuey_teow": "char kuey teow malaysian penang ",
    "chicken_rice": "hainanese chicken rice",
    "laksa": "laksa malaysian food penang asam laksa laksa kedah",
    "mee_goreng": "mee goreng malaysian mamak",
}

# Extra Google/Bing queries to reach 1.5k+ images per class (UC rotates through these).
QUERY_VARIANTS: dict[str, list[str]] = {
    "nasi_lemak": [
        "nasi lemak malaysia banana leaf",
        "nasi lemak sambal egg anchovies",
        "nasi lemak hawker stall",
    ],
    "roti_canai": [
        "roti canai mamak malaysia",
        "roti prata malaysia food",
        "roti canai curry malaysia",
    ],
    "char_kuey_teow": [
        "char kuey teow penang malaysia",
        "char koay teow malaysia wok",
        "fried flat rice noodles malaysia",
    ],
    "chicken_rice": [
        "soy chicken rice malaysia",
        "steam chicken rice malaysian",
        "hainanese chicken rice malaysia",
        "chicken rice malaysia plate",
        "malaysian chicken rice soy sauce",
    ],
    "laksa": [
        "penang asam laksa malaysia",
        "curry laksa malaysia",
        "laksa kedah malaysia",
        "sarawak laksa malaysia",
    ],
    "mee_goreng": [
        "mee goreng mamak malaysia yellow noodles",
        "indian mee goreng malaysia mamak plate",
        "mee goreng mamak egg tomato malaysia",
        "mee goreng mamak restaurant malaysia",
        "malaysian mee goreng yellow noodle wok",
    ],
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Classes still being filled in dataset3 (max 1100 each).
DEFAULT_SCRAPE_CLASSES = ["char_kuey_teow", "chicken_rice"]
DEFAULT_MAX_IMAGES = 1100


class ImageScraper:
    """Downloads food images per class using icrawler or SeleniumBase UC mode."""

    def __init__(
        self,
        output_dir: Path,
        keywords: dict[str, str] | None = None,
        max_images: int = DEFAULT_MAX_IMAGES,
        engine: str = "google",
        google_fallback: bool = False,
        uc_headless: bool = False,
        uc_scroll_pause: float = 1.5,
        uc_max_scrolls: int = 40,
        uc_max_pages: int = 10,
        uc_wait_for_captcha: bool = True,
        manifest_path: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.max_images = max_images
        self.engine = engine
        self.google_fallback = google_fallback
        self.manifest_path = manifest_path
        self.uc_scraper = UCImageScraper(
            headless=uc_headless,
            scroll_pause=uc_scroll_pause,
            max_scrolls=uc_max_scrolls,
            max_pages=uc_max_pages,
            wait_for_captcha=uc_wait_for_captcha,
        )

    def _queries_for_class(self, class_name: str, keyword: str) -> list[str]:
        variants = QUERY_VARIANTS.get(class_name, [])
        return _normalize_queries([keyword, *variants])

    def _create_crawler(self, class_dir: Path, engine: str | None = None):
        """Create an icrawler instance for the given engine."""
        selected_engine = engine or self.engine
        if selected_engine == "bing":
            return BingImageCrawler(storage={"root_dir": str(class_dir)})
        return FixedGoogleImageCrawler(storage={"root_dir": str(class_dir)})

    def _count_images(self, class_dir: Path) -> int:
        return sum(1 for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    def scrape_class(self, class_name: str, keyword: str) -> int:
        """Scrape images for a single class. Returns number of images downloaded."""
        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        before = {
            path.resolve()
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }

        existing = self._count_images(class_dir)
        if existing >= self.max_images:
            logger.info("%s: already has %d images, skipping.", class_name, existing)
            return existing

        to_fetch = self.max_images - existing
        logger.info(
            "Scraping '%s' with keyword '%s' (max %d, engine=%s)...",
            class_name,
            keyword,
            to_fetch,
            self.engine,
        )

        if self.engine == "uc":
            queries = self._queries_for_class(class_name, keyword)
            downloaded = self.uc_scraper.scrape(queries, class_dir, self.max_images)
        else:
            crawler = self._create_crawler(class_dir)
            crawler.crawl(keyword=keyword, max_num=to_fetch)
            downloaded = self._count_images(class_dir)

        fallback_metadata: dict[Path, dict[str, str]] = {}
        if self.engine in {"google", "uc"} and downloaded < self.max_images and self.google_fallback:
            queries = self._queries_for_class(class_name, keyword)
            for query in queries:
                if downloaded >= self.max_images:
                    break
                remaining = self.max_images - downloaded
                logger.warning(
                    "%s: %s fetched %d/%d; Bing fallback query=%r for %d more.",
                    class_name,
                    self.engine,
                    downloaded,
                    self.max_images,
                    query,
                    remaining,
                )
                before_fallback = {
                    path.resolve()
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                }
                fallback = self._create_crawler(class_dir, engine="bing")
                fallback.crawl(keyword=query, max_num=remaining)
                downloaded = self._count_images(class_dir)
                for path in class_dir.iterdir():
                    resolved = path.resolve()
                    if (
                        path.is_file()
                        and path.suffix.lower() in IMAGE_SUFFIXES
                        and resolved not in before_fallback
                    ):
                        fallback_metadata[resolved] = {"engine": "bing", "query": query}

        if self.manifest_path:
            uc_metadata = {
                Path(item["path"]).resolve(): item
                for item in self.uc_scraper.download_log
            } if self.engine == "uc" else {}
            new_paths = sorted(
                (
                    path
                    for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_SUFFIXES
                    and path.resolve() not in before
                ),
                key=lambda path: path.name,
            )
            records = [
                build_download_record(
                    path=path,
                    candidate_class=class_name,
                    query=(
                        fallback_metadata.get(path.resolve(), {}).get("query")
                        or uc_metadata.get(path.resolve(), {}).get("query")
                        or keyword
                    ),
                    engine=fallback_metadata.get(path.resolve(), {}).get("engine", self.engine),
                    source_url=uc_metadata.get(path.resolve(), {}).get("source_url"),
                )
                for path in new_paths
            ]
            written = append_jsonl(self.manifest_path, records)
            logger.info("  %s: appended %d provenance records to %s", class_name, written, self.manifest_path)
        logger.info("  %s: %d images in %s", class_name, downloaded, class_dir)
        return downloaded

    def scrape_all(self, classes: list[str] | None = None) -> int:
        """Scrape images for all (or specified) classes. Returns total image count."""
        target = classes or list(self.keywords.keys())
        total = 0
        for class_name in target:
            if class_name not in self.keywords:
                logger.warning("Unknown class: %s, skipping.", class_name)
                continue
            total += self.scrape_class(class_name, self.keywords[class_name])
        return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape food images for dataset3")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="Max images per class (default: 1100)",
    )
    parser.add_argument(
        "--engine",
        choices=["google", "bing", "uc"],
        default="google",
        help="google=icrawler, uc=SeleniumBase Undetected Chrome, bing=icrawler Bing",
    )
    parser.add_argument(
        "--google-fallback",
        action="store_true",
        help="Retry with Bing when google/uc returns fewer images than requested",
    )
    parser.add_argument(
        "--uc-headless",
        action="store_true",
        help="Run UC browser headless (less reliable against bot detection)",
    )
    parser.add_argument(
        "--uc-scroll-pause",
        type=float,
        default=1.5,
        help="Seconds to wait between UC scrolls (default: 1.5)",
    )
    parser.add_argument(
        "--uc-max-scrolls",
        type=int,
        default=60,
        help="Maximum scroll attempts per UC page (default: 60)",
    )
    parser.add_argument(
        "--uc-max-pages",
        type=int,
        default=10,
        help="Maximum paginated UC pages per query, 100 results each (default: 10)",
    )
    parser.add_argument(
        "--uc-no-wait-captcha",
        action="store_true",
        help="Do not pause for manual CAPTCHA solving in headed UC mode",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_SCRAPE_CLASSES,
        help="Classes to scrape (default: char_kuey_teow chicken_rice)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/downloads.jsonl"),
        help="Append provenance for newly downloaded images",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable download provenance records",
    )
    args = parser.parse_args()

    scraper = ImageScraper(
        output_dir=args.output_dir,
        max_images=args.max_images,
        engine=args.engine,
        google_fallback=args.google_fallback,
        uc_headless=args.uc_headless,
        uc_scroll_pause=args.uc_scroll_pause,
        uc_max_scrolls=args.uc_max_scrolls,
        uc_max_pages=args.uc_max_pages,
        uc_wait_for_captcha=not args.uc_no_wait_captcha,
        manifest_path=None if args.no_manifest else args.manifest,
    )
    total = scraper.scrape_all(args.classes)
    print(f"Done. Total images across classes: {total}")


if __name__ == "__main__":
    main()
