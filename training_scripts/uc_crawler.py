"""Google Images scraper using SeleniumBase Undetected Chrome (UC) mode."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
BLOCK_URL_MARKERS = ("/sorry/", "google.com/sorry")
BLOCK_TEXT_MARKERS = (
    "our systems have detected unusual traffic",
    "unusual traffic from your computer network",
    "verify you're not a robot",
    "to continue, please verify that you are not a robot",
)
COUNT_RESULT_IMAGES_JS = """
return [...document.querySelectorAll("img")].filter((img) => {
    const src = img.currentSrc || img.src || "";
    return src.startsWith("http") && !src.includes("gstatic.com/images/branding");
}).length;
"""
CONSENT_SELECTORS = (
    "#L2AGLb",
    'button[aria-label="Accept all"]',
    'button[aria-label="Reject all"]',
    "form[action*='consent'] button",
)
SKIP_URL_PARTS = (
    "gstatic.com/images/branding",
    "google.com/images",
    "favicon",
    "logo",
)
EXTRACT_URLS_JS = """
const urls = new Set();
for (const img of document.querySelectorAll("img")) {
    for (const value of [img.currentSrc, img.src, img.getAttribute("data-src")]) {
        if (value && value.startsWith("http")) urls.add(value);
    }
}
for (const anchor of document.querySelectorAll("a[href*='imgurl=']")) {
    try {
        const parsed = new URL(anchor.href);
        const imgurl = parsed.searchParams.get("imgurl");
        if (imgurl) urls.add(imgurl);
    } catch (err) {}
}
return [...urls];
"""
CLICK_MORE_RESULTS_JS = """
const nodes = [...document.querySelectorAll("a, button, input, span")];
const target = nodes.find((el) => {
    const text = (el.innerText || el.value || el.getAttribute("aria-label") || "").toLowerCase();
    return text.includes("more results") || text.includes("show more");
});
if (target) {
    target.click();
    return true;
}
return false;
"""

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
ORIGINAL_URL_PATTERN = re.compile(r'"ou":"(https?://[^"]+)"')


def _google_images_url(keyword: str, start: int = 0) -> str:
    query = quote(keyword)
    url = f"https://www.google.com/search?q={query}&udm=2&hl=en&gl=us"
    if start > 0:
        url += f"&start={start}"
    return url


def _normalize_queries(keyword: str | list[str]) -> list[str]:
    if isinstance(keyword, list):
        queries = keyword
    else:
        queries = [keyword]
    seen: set[str] = set()
    ordered: list[str] = []
    for query in queries:
        cleaned = " ".join(query.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def _url_priority(url: str) -> tuple[int, str]:
    lower = url.lower()
    if "encrypted-tbn0.gstatic.com" in lower:
        return (2, url)
    if "gstatic.com" in lower:
        return (1, url)
    return (0, url)


def _is_blocked(page_source: str, current_url: str) -> bool:
    lower_url = current_url.lower()
    if any(marker in lower_url for marker in BLOCK_URL_MARKERS):
        return True
    lower_source = page_source.lower()
    return any(marker in lower_source for marker in BLOCK_TEXT_MARKERS)


def _count_result_images(sb) -> int:
    try:
        return int(sb.execute_script(COUNT_RESULT_IMAGES_JS) or 0)
    except Exception:
        return 0


def _page_has_results(sb) -> bool:
    if _count_result_images(sb) >= 3:
        return True
    return len(ORIGINAL_URL_PATTERN.findall(sb.get_page_source())) >= 3


def _should_skip_url(url: str) -> bool:
    lower = url.lower()
    if not lower.startswith("http"):
        return True
    if any(part in lower for part in SKIP_URL_PARTS):
        return True
    parsed = urlparse(url)
    if parsed.path.endswith(".svg"):
        return True
    return False


def _guess_suffix(url: str, content_type: str | None) -> str:
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in IMAGE_SUFFIXES:
        return path_suffix

    if content_type:
        lowered = content_type.lower()
        if "jpeg" in lowered or "jpg" in lowered:
            return ".jpg"
        if "png" in lowered:
            return ".png"
        if "webp" in lowered:
            return ".webp"
        if "gif" in lowered:
            return ".gif"
        if "bmp" in lowered:
            return ".bmp"
    return ".jpg"


def _next_file_index(class_dir: Path) -> int:
    indices = [
        int(path.stem)
        for path in class_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isdigit()
    ]
    return (max(indices) if indices else 0) + 1


class UCImageScraper:
    """Scrape Google Images with SeleniumBase UC mode and download files locally."""

    def __init__(
        self,
        headless: bool = False,
        scroll_pause: float = 1.5,
        max_scrolls: int = 40,
        max_pages: int = 10,
        request_timeout: int = 20,
        wait_for_captcha: bool = True,
    ) -> None:
        self.headless = headless
        self.scroll_pause = scroll_pause
        self.max_scrolls = max_scrolls
        self.max_pages = max_pages
        self.request_timeout = request_timeout
        self.wait_for_captcha = wait_for_captcha and not headless
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _dismiss_consent(self, sb) -> None:
        for selector in CONSENT_SELECTORS:
            try:
                if sb.is_element_visible(selector):
                    sb.click(selector)
                    sb.sleep(2)
                    logger.info("Dismissed Google consent dialog.")
                    return
            except Exception:
                continue

    def _wait_for_manual_captcha(self, sb, keyword: str) -> bool:
        if not self.wait_for_captcha:
            return False

        logger.warning(
            "Google block page detected at %s. If you see a challenge in Chrome, "
            "complete it and wait for image results, then press Enter here.",
            sb.get_current_url(),
        )
        try:
            input()
        except EOFError:
            logger.warning("No interactive terminal available for manual CAPTCHA wait.")
            return False

        if _page_has_results(sb):
            return True

        search_url = _google_images_url(keyword)
        sb.uc_open_with_reconnect(search_url, reconnect_time=4)
        sb.sleep(3)
        self._dismiss_consent(sb)
        return _page_has_results(sb) and not _is_blocked(sb.get_page_source(), sb.get_current_url())

    def _try_handle_block(self, sb, keyword: str) -> bool:
        if _page_has_results(sb):
            return False

        if not _is_blocked(sb.get_page_source(), sb.get_current_url()):
            logger.info(
                "No image grid yet at %s; waiting for page to finish loading.",
                sb.get_current_url(),
            )
            sb.sleep(5)
            self._dismiss_consent(sb)
            if _page_has_results(sb):
                return False

        if not _is_blocked(sb.get_page_source(), sb.get_current_url()):
            logger.warning(
                "Google page loaded but no image results found yet at %s.",
                sb.get_current_url(),
            )
            return False

        logger.warning("Google block page detected at %s; trying UC CAPTCHA handler.", sb.get_current_url())
        try:
            sb.uc_gui_click_captcha()
            sb.sleep(4)
        except Exception as exc:
            logger.warning("UC CAPTCHA handler failed: %s", exc)

        if _page_has_results(sb) or not _is_blocked(sb.get_page_source(), sb.get_current_url()):
            return False

        if self._wait_for_manual_captcha(sb, keyword):
            return False

        return True

    def _collect_urls_on_page(self, sb, keyword: str, start: int, seen: set[str]) -> list[str]:
        search_url = _google_images_url(keyword, start=start)
        logger.info("Loading Google Images page: start=%d query=%r", start, keyword)
        sb.uc_open_with_reconnect(search_url, reconnect_time=4)
        sb.sleep(3)
        self._dismiss_consent(sb)

        if start == 0 and self._try_handle_block(sb, keyword):
            logger.error(
                "Google blocked the UC browser session at %s. Try again later or pass --google-fallback.",
                sb.get_current_url(),
            )
            return []

        ordered: list[str] = []
        stale_rounds = 0

        for scroll_idx in range(self.max_scrolls):
            dom_urls = sb.execute_script(EXTRACT_URLS_JS)
            page_urls = ORIGINAL_URL_PATTERN.findall(sb.get_page_source())
            candidates = sorted(set(list(dom_urls or []) + page_urls), key=_url_priority)

            added = 0
            for url in candidates:
                if _should_skip_url(url) or url in seen:
                    continue
                seen.add(url)
                ordered.append(url)
                added += 1

            logger.info(
                "Query=%r start=%d scroll %d/%d: +%d URLs (%d on this page, %d total seen).",
                keyword,
                start,
                scroll_idx + 1,
                self.max_scrolls,
                added,
                len(ordered),
                len(seen),
            )

            if added == 0:
                stale_rounds += 1
            else:
                stale_rounds = 0

            if stale_rounds >= 5:
                break

            clicked_more = sb.execute_script(CLICK_MORE_RESULTS_JS)
            if clicked_more:
                sb.sleep(self.scroll_pause)
                continue

            sb.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sb.sleep(self.scroll_pause)

        return ordered

    def _download_batch(
        self,
        urls: list[str],
        class_dir: Path,
        file_index: int,
        needed: int,
        downloaded: int,
    ) -> tuple[int, int]:
        for url in sorted(urls, key=_url_priority):
            if downloaded >= needed:
                break
            dest = class_dir / f"{file_index:06d}"
            if self._download_image(url, dest):
                downloaded += 1
                file_index += 1
                if downloaded % 50 == 0 or downloaded == needed:
                    logger.info(
                        "Saved %d/%d images for %s",
                        downloaded,
                        needed,
                        class_dir.name,
                    )
        return downloaded, file_index

    def scrape(self, keyword: str | list[str], class_dir: Path, max_images: int) -> int:
        """Download up to max_images for keyword(s) into class_dir. Returns total image count."""
        class_dir.mkdir(parents=True, exist_ok=True)
        starting = sum(1 for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if starting >= max_images:
            return starting

        queries = _normalize_queries(keyword)
        logger.info("UC scrape using %d query variant(s) for %s", len(queries), class_dir.name)

        try:
            from seleniumbase import SB
        except ImportError as exc:
            raise ImportError(
                "seleniumbase is required for --engine uc. Install with: pip install seleniumbase"
            ) from exc

        needed = max_images - starting
        file_index = _next_file_index(class_dir)
        downloaded = 0
        seen: set[str] = set()

        with SB(uc=True, headless=self.headless, incognito=False) as sb:
            for query in queries:
                if downloaded >= needed:
                    break
                for page_idx in range(self.max_pages):
                    if downloaded >= needed:
                        break
                    start = page_idx * 100
                    page_urls = self._collect_urls_on_page(sb, query, start, seen)
                    if not page_urls:
                        if page_idx == 0:
                            logger.warning("No URLs found for query=%r", query)
                        break
                    downloaded, file_index = self._download_batch(
                        page_urls, class_dir, file_index, needed, downloaded
                    )

        total = starting + downloaded
        logger.info("UC scrape finished: %d/%d images in %s", total, max_images, class_dir)
        return total

    def _download_image(self, url: str, dest: Path) -> bool:
        try:
            response = self.session.get(url, timeout=self.request_timeout, stream=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Failed to download %s: %s", url, exc)
            return False

        content_type = response.headers.get("Content-Type", "")
        if content_type and "image" not in content_type.lower():
            return False

        suffix = _guess_suffix(url, content_type)
        target = dest.with_suffix(suffix)
        try:
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        except OSError as exc:
            logger.debug("Failed to write %s: %s", target, exc)
            if target.exists():
                target.unlink(missing_ok=True)
            return False

        if target.stat().st_size < 1024:
            target.unlink(missing_ok=True)
            return False

        return True
