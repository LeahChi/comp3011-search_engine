from __future__ import annotations

import time
import logging
from collections import deque               # for BFS
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%d/%m/%Y %H:%M:%S")
# changed it to UK format DDMMYYYY
logger = logging.getLogger(__name__)

@dataclass
class CrawledPage:
    """Represents a crawled page and its extracted content."""
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


class Crawler:
    """
    BFS web crawler that respects politeness and handles retries.

    Attributes:
        base_url:         The root URL to start crawling from.
        politeness_delay: Seconds to wait between requests.
        max_retries:      Number of times to retry a failed request.
        timeout:          Seconds before a request times out.
    """

    def __init__(
        self,
        base_url: str = "https://quotes.toscrape.com",
        politeness_delay: float = 6.0,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url
        self.politeness_delay = max(politeness_delay, 6.0)  # enforce minimum
        self.max_retries = max_retries
        self.timeout = timeout
        self.visited: set[str] = set()
        self.session = requests.Session()                   # reuse TCP connection

    def _is_internal(self, url: str) -> bool:
        """Return True if the URL belongs to the same domain as base_url."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """Extract and normalise all internal links from a page."""
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = urljoin(current_url, href)
            full_url = full_url.split("#")[0].rstrip("/")
            if self._is_internal(full_url) and full_url not in self.visited:
                links.append(full_url)
        return links

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean visible text from a page, stripping scripts and styles."""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())

    def _fetch(self, url: str) -> Optional[requests.Response]:
        """
        Fetch a URL with retry logic.

        Returns the Response on success, None on failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error on {url}: {e} (attempt {attempt})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on {url}: {e} (attempt {attempt})")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {url} (attempt {attempt})")
            if attempt < self.max_retries:
                time.sleep(2)
        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def crawl(self) -> dict[str, CrawledPage]:
        """
        Crawl the website using BFS and return all crawled pages.

        Returns:
            A dict mapping URL -> CrawledPage for every successfully crawled page.
        Time complexity: O(V + E) where V = pages, E = links between pages.
        """
        queue: deque[str] = deque([self.base_url])
        self.visited.add(self.base_url)
        pages: dict[str, CrawledPage] = {}

        while queue:
            url = queue.popleft()
            logger.info(f"Crawling: {url}")

            response = self._fetch(url)
            if response is None:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "No title"
            text = self._extract_text(soup)
            links = self._extract_links(soup, url)

            pages[url] = CrawledPage(url=url, title=title, text=text, links=links)

            for link in links:
                if link not in self.visited:
                    self.visited.add(link)
                    queue.append(link)

            logger.info(f"Done. Found {len(links)} links. Sleeping {self.politeness_delay}s...")
            time.sleep(self.politeness_delay)

        logger.info(f"Crawl complete. {len(pages)} pages crawled.")
        return pages

