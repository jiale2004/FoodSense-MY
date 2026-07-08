"""Patched Google image crawler for icrawler 0.6.x.

Google frequently changes image search markup and may serve JS-only or block
pages to bots. This module hardens the stock icrawler Google parser so it no
longer crashes and tries several URL extraction strategies.
"""

from __future__ import annotations

import queue
import re
import time
from threading import current_thread
from urllib.parse import unquote, urlencode, urlsplit

from bs4 import BeautifulSoup
from icrawler.builtin.google import GoogleFeeder, GoogleImageCrawler, GoogleParser

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

BLOCK_MARKERS = (
    "enablejs",
    "/sorry/",
    "unusual traffic",
    "before you continue",
    "update your browser",
)

IMAGE_PATTERNS = (
    re.compile(r"http[^\[\]\"\\]+?\.(?:jpg|jpeg|png|bmp|webp)(?:\?[^\[\]\"\\]*)?", re.I),
    re.compile(r'"ou":"(https?://[^"]+)"'),
    re.compile(r'"ow":\d+,"oh":\d+,"ou":"(https?://[^"]+)"'),
    re.compile(r"imgurl=(https?[^&\"]+)"),
    re.compile(r'\\"(https?:\\\\/\\\\/[^\\"]+\.(?:jpg|jpeg|png|webp)[^\\"]*)\\"'),
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize_url(url: str) -> str:
    url = url.replace("\\/", "/")
    url = bytes(url, "utf-8").decode("unicode-escape", errors="ignore")
    return unquote(url)


def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from a Google image search HTML response."""
    urls: list[str] = []
    seen: set[str] = set()

    soup = BeautifulSoup(text, "lxml")
    for script in soup.find_all(name="script"):
        script_text = str(script)
        for pattern in IMAGE_PATTERNS:
            for match in pattern.findall(script_text):
                candidate = _normalize_url(match)
                if candidate.startswith("http") and candidate not in seen:
                    seen.add(candidate)
                    urls.append(candidate)

    for pattern in IMAGE_PATTERNS:
        for match in pattern.findall(text):
            candidate = _normalize_url(match)
            if candidate.startswith("http") and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)

    return urls


def is_blocked_page(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in BLOCK_MARKERS)


class FixedGoogleFeeder(GoogleFeeder):
    """Use Google's newer image search mode (udm=2)."""

    def feed(self, keyword, offset, max_num, language=None, filters=None):
        base_url = "https://www.google.com/search?"
        self.filter = self.get_filter()
        filter_str = self.filter.apply(filters, sep=",")
        for i in range(offset, offset + max_num, 100):
            params = dict(q=keyword, ijn=int(i / 100), start=i, tbs=filter_str, udm=2)
            if language:
                params["lr"] = "lang_" + language
            url = base_url + urlencode(params)
            self.out_queue.put(url)
            self.logger.debug("put url to url_queue: %s", url)


class FixedGoogleParser(GoogleParser):
    """Return an empty list instead of None and extract more URL formats."""

    def parse(self, response):
        text = response.content.decode("utf-8", "ignore")
        if not text.strip():
            self.logger.warning("Google returned an empty response body.")
            return []

        if is_blocked_page(text):
            self.logger.warning(
                "Google returned a block/consent/JS-only page; try --engine bing "
                "or rerun later with a slower rate."
            )
            return []

        urls = extract_image_urls(text)
        if not urls:
            self.logger.warning("Google page parsed successfully but contained no image URLs.")
            return []

        return [{"file_url": url} for url in urls]

    def worker_exec(self, queue_timeout=2, req_timeout=5, max_retry=3, **kwargs):
        """Fetch pages with a modern browser User-Agent header."""
        while True:
            if self.signal.get("reach_max_num"):
                self.logger.info(
                    "downloaded image reached max num, thread %s is ready to exit",
                    current_thread().name,
                )
                break
            try:
                url = self.in_queue.get(timeout=queue_timeout)
            except queue.Empty:
                if self.signal.get("feeder_exited"):
                    self.logger.info("no more page urls for thread %s to parse", current_thread().name)
                    break
                self.logger.info("%s is waiting for new page urls", current_thread().name)
                continue
            except Exception:
                self.logger.error("exception in thread %s", current_thread().name)
                continue
            else:
                self.logger.debug("start fetching page %s", url)

            retry = max_retry
            while retry > 0:
                try:
                    base_url = "{0.scheme}://{0.netloc}".format(urlsplit(url))
                    headers = {**REQUEST_HEADERS, "Referer": base_url}
                    response = self.session.get(url, timeout=req_timeout, headers=headers)
                except Exception as exc:
                    self.logger.error(
                        "Exception caught when fetching page %s, error: %s, remaining retry times: %d",
                        url,
                        exc,
                        retry - 1,
                    )
                else:
                    self.logger.info("parsing result page %s", url)
                    for task in self.parse(response, **kwargs) or []:
                        while not self.signal.get("reach_max_num"):
                            try:
                                if isinstance(task, dict):
                                    self.output(task, timeout=1)
                                elif isinstance(task, str):
                                    self.input(task, timeout=1)
                            except queue.Full:
                                time.sleep(1)
                            except Exception as exc:
                                self.logger.error(
                                    "Exception caught when put task %s into queue, error: %s",
                                    task,
                                    exc,
                                )
                            else:
                                break
                        if self.signal.get("reach_max_num"):
                            break
                    self.in_queue.task_done()
                    break
                finally:
                    retry -= 1

        self.logger.info("thread %s exit", current_thread().name)


class FixedGoogleImageCrawler(GoogleImageCrawler):
    """Google image crawler with parser and request hardening."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            feeder_cls=FixedGoogleFeeder,
            parser_cls=FixedGoogleParser,
            *args,
            **kwargs,
        )
        self.session.headers.update(REQUEST_HEADERS)
        self.session.cookies.set(
            "CONSENT",
            "YES+cb.20210328-17-p0.en+FX+667",
            domain=".google.com",
        )
