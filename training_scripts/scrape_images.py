#!/usr/bin/env python3
"""Scrape supplemental food images for the myFood11 dataset."""

import argparse
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests

DEFAULT_KEYWORDS = {
    "nasi_lemak": "nasi lemak malaysian food",
    "roti_canai": "roti canai malaysian",
    "char_kuey_teow": "char kuey teow malaysian",
    "nasi_goreng": "nasi goreng malaysian",
    "laksa": "laksa malaysian food",
    "satay": "satay malaysian",
    "rendang": "rendang malaysian",
    "roti_tissue": "roti tissue malaysian",
    "cendol": "cendol malaysian dessert",
    "teh_tarik": "teh tarik malaysian",
    "murtabak": "murtabak malaysian",
}

HEADERS = {
    "User-Agent": "FoodSense-MY/1.0 (research; educational dataset builder)",
}


def check_robots_txt(base_url: str) -> bool:
    """Basic robots.txt check stub. Returns True if scraping should proceed."""
    robots_url = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200 and "Disallow: /" in resp.text:
            print(f"WARNING: {robots_url} may disallow scraping. Review before proceeding.")
            return False
    except requests.RequestException:
        pass
    return True


def download_image(url: str, save_path: Path) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        save_path.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"  Failed: {url} — {exc}")
        return False


def scrape_class(
    class_name: str,
    keyword: str,
    output_dir: Path,
    max_images: int,
    delay: float,
) -> int:
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: uses a simple image search API pattern.
    # Replace with your preferred source when myFood11 data is available.
    print(f"Scraping '{class_name}' with keyword '{keyword}' (max {max_images})...")
    print("  NOTE: Configure a real image source API before production use.")

    search_url = f"https://www.google.com/search?q={quote_plus(keyword)}&tbm=isch"
    check_robots_txt(search_url)

    downloaded = 0
    for i in range(max_images):
        save_path = class_dir / f"{class_name}_{i:04d}.jpg"
        if save_path.exists():
            downloaded += 1
            continue
        time.sleep(delay)

    print(f"  {class_name}: {downloaded} images in {class_dir}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape supplemental food images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_images"),
        help="Output directory for scraped images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum images per class",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Specific classes to scrape (default: all)",
    )
    args = parser.parse_args()

    keywords = DEFAULT_KEYWORDS
    target_classes = args.classes or list(keywords.keys())

    total = 0
    for class_name in target_classes:
        if class_name not in keywords:
            print(f"Unknown class: {class_name}, skipping.")
            continue
        total += scrape_class(
            class_name, keywords[class_name], args.output_dir, args.max_images, args.delay
        )

    print(f"Done. Total images across classes: {total}")


if __name__ == "__main__":
    main()
