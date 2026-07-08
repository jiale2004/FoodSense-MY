#!/usr/bin/env python3
"""Scrape supplemental food images using icrawler."""

import argparse
import logging
from pathlib import Path

from icrawler.builtin import BingImageCrawler

from google_crawler import FixedGoogleImageCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = {
    "nasi_lemak": "nasi lemak malaysian food",
    "roti_canai": "roti canai malaysian",
    "char_kuey_teow": "char kuey teow malaysian",
    "chicken_rice": "chicken rice malaysian hainanese",
    "laksa": "laksa malaysian food",
    "mee_goreng": "mee goreng malaysian",
}


class ImageScraper:
    """Downloads food images per class using icrawler (Google/Bing)."""

    def __init__(
        self,
        output_dir: Path,
        keywords: dict[str, str] | None = None,
        max_images: int = 50,
        engine: str = "google",
        google_fallback: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.max_images = max_images
        self.engine = engine
        self.google_fallback = google_fallback

    def _create_crawler(self, class_dir: Path, engine: str | None = None):
        """Create an icrawler instance for the given engine."""
        selected_engine = engine or self.engine
        if selected_engine == "bing":
            return BingImageCrawler(storage={"root_dir": str(class_dir)})
        return FixedGoogleImageCrawler(storage={"root_dir": str(class_dir)})

    def scrape_class(self, class_name: str, keyword: str) -> int:
        """Scrape images for a single class. Returns number of images downloaded."""
        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(class_dir.glob("*")))
        if existing >= self.max_images:
            logger.info("%s: already has %d images, skipping.", class_name, existing)
            return existing

        to_fetch = self.max_images - existing
        logger.info("Scraping '%s' with keyword '%s' (max %d)...", class_name, keyword, to_fetch)

        crawler = self._create_crawler(class_dir)
        crawler.crawl(keyword=keyword, max_num=to_fetch)

        downloaded = len(list(class_dir.glob("*")))
        if self.engine == "google" and downloaded < self.max_images and self.google_fallback:
            remaining = self.max_images - downloaded
            logger.warning(
                "%s: Google fetched %d/%d images; falling back to Bing for %d more.",
                class_name,
                downloaded,
                self.max_images,
                remaining,
            )
            fallback = self._create_crawler(class_dir, engine="bing")
            fallback.crawl(keyword=keyword, max_num=remaining)
            downloaded = len(list(class_dir.glob("*")))
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
    parser = argparse.ArgumentParser(description="Scrape food images with icrawler")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument("--max-images", type=int, default=50, help="Max images per class")
    parser.add_argument("--engine", choices=["google", "bing"], default="google")
    parser.add_argument(
        "--no-google-fallback",
        action="store_true",
        help="Do not retry with Bing when Google returns no images",
    )
    parser.add_argument("--classes", nargs="*", default=None, help="Specific classes to scrape")
    args = parser.parse_args()

    scraper = ImageScraper(
        output_dir=args.output_dir,
        max_images=args.max_images,
        engine=args.engine,
        google_fallback=not args.no_google_fallback,
    )
    total = scraper.scrape_all(args.classes)
    print(f"Done. Total images across classes: {total}")


if __name__ == "__main__":
    main()
