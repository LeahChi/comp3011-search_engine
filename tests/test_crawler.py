from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup

from src.crawler import Crawler, CrawledPage

SIMPLE_HTML = """
<html>
  <head><title>Test Page</title></head>
  <body>
    <p>Hello!!! This is a test page.</p>
    <a href="/page/2">Next</a>
    <a href="https://external.com">External</a>
  </body>
</html>
"""

EMPTY_HTML = """
<html>
  <head></head>
  <body></body>
</html>
"""

SCRIPT_HTML = """
<html>
  <head><title>Script Page</title></head>
  <body>
    <script>var x = 'this should not appear';</script>
    <style>.css { color: red; }</style>
    <p>Visible text only.</p>
  </body>
</html>
"""


def make_mock_response(html: str, status_code: int = 200) -> MagicMock:
    """Create a fake requests.Response with controllable content."""
    mock = MagicMock()
    mock.text = html
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    return mock


class TestCrawlerInit:
    """Tests for Crawler initialisation and configuration."""

    def test_default_politeness_delay(self):
        crawler = Crawler()
        assert crawler.politeness_delay == 6.0

    def test_politeness_delay_enforced_minimum(self):
        # Even if a user tries to set it lower, it should be enforced to 6.0
        crawler = Crawler(politeness_delay=1.0)
        assert crawler.politeness_delay == 6.0

    def test_visited_starts_empty(self):
        crawler = Crawler()
        assert len(crawler.visited) == 0


class TestIsInternal:
    """
    Tests for _is_internal URL classification.
    True (internal): the URL gets added to the BFS queue and crawled.
    False (external): the URL gets completely ignored and never visited.
    Checking that https://external.com never ends up in the crawl queue.
    
    """


    def setup_method(self):
        self.crawler = Crawler()

    def test_internal_url(self):
        assert self.crawler._is_internal("https://quotes.toscrape.com/page/2") is True

    def test_external_url(self):
        assert self.crawler._is_internal("https://external.com") is False

    def test_external_subdomain(self):
        assert self.crawler._is_internal("https://sub.quotes.toscrape.com") is False


class TestExtractLinks:
    """Tests for _extract_links parsing."""

    def setup_method(self):
        self.crawler = Crawler()

    def test_extracts_internal_links(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        links = self.crawler._extract_links(soup, "https://quotes.toscrape.com")
        assert "https://quotes.toscrape.com/page/2" in links

    def test_excludes_external_links(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        links = self.crawler._extract_links(soup, "https://quotes.toscrape.com")
        assert "https://external.com" not in links

    def test_no_links_on_empty_page(self):
        soup = BeautifulSoup(EMPTY_HTML, "html.parser")
        links = self.crawler._extract_links(soup, "https://quotes.toscrape.com")
        assert links == []

    def test_no_duplicate_links(self):
        self.crawler.visited.add("https://quotes.toscrape.com/page/2")
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        links = self.crawler._extract_links(soup, "https://quotes.toscrape.com")
        assert "https://quotes.toscrape.com/page/2" not in links


class TestExtractText:
    """Tests for _extract_text cleaning."""

    def setup_method(self):
        self.crawler = Crawler()

    def test_extracts_visible_text(self):
        soup = BeautifulSoup(SIMPLE_HTML, "html.parser")
        text = self.crawler._extract_text(soup)
        assert "Hello!!! This is a test page" in text

    def test_strips_script_tags(self):
        # The CSS and JS content should not appear in the extracted text
        soup = BeautifulSoup(SCRIPT_HTML, "html.parser")
        text = self.crawler._extract_text(soup)
        assert "this should not appear" not in text

    def test_strips_style_tags(self):
        soup = BeautifulSoup(SCRIPT_HTML, "html.parser")
        text = self.crawler._extract_text(soup)
        assert "color: red" not in text

    def test_visible_text_preserved(self):
        soup = BeautifulSoup(SCRIPT_HTML, "html.parser")
        text = self.crawler._extract_text(soup)
        assert "Visible text only" in text

    def test_empty_page_returns_empty_string(self):
        soup = BeautifulSoup(EMPTY_HTML, "html.parser")
        text = self.crawler._extract_text(soup)
        assert text.strip() == ""


class TestFetch:
    """Tests for _fetch retry logic and error handling.
        Checking that if a request fails (e.g., due to a timeout), the crawler retries the request up to the specified max_retries. 
        If all retries fail, it should return None and not crash the crawler.
        Checking if the politeness delay is respected between retries, and that successful fetches return the expected content.
        Checking that if all retries fail, _fetch returns None and does not raise an exception.
    """

    def setup_method(self):
        self.crawler = Crawler()

    @patch("src.crawler.time.sleep")
    def test_successful_fetch(self, mock_sleep):
        with patch.object(self.crawler.session, "get", return_value=make_mock_response(SIMPLE_HTML)):
            response = self.crawler._fetch("https://quotes.toscrape.com")
            assert response is not None
            assert response.text == SIMPLE_HTML

    @patch("src.crawler.time.sleep")
    def test_returns_none_after_all_retries_fail(self, mock_sleep):
        import requests as req
        mock_response = make_mock_response("", 500)
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError("500")
        with patch.object(self.crawler.session, "get", return_value=mock_response):
            result = self.crawler._fetch("https://quotes.toscrape.com")
            assert result is None

    @patch("src.crawler.time.sleep")
    def test_retries_on_timeout(self, mock_sleep):
        import requests as req
        with patch.object(
            self.crawler.session, "get",
            side_effect=req.exceptions.Timeout
        ) as mock_get:
            self.crawler._fetch("https://quotes.toscrape.com")
            assert mock_get.call_count == self.crawler.max_retries


class TestCrawl:
    """Integration-style tests for the full crawl method.
       Testing the crawl() method which calls all above helper methods together.
    """

    @patch("src.crawler.time.sleep")
    def test_crawl_returns_crawled_pages(self, mock_sleep):
        crawler = Crawler()
        mock_response = make_mock_response(SIMPLE_HTML)
        with patch.object(crawler.session, "get", return_value=mock_response):
            pages = crawler.crawl()
        assert isinstance(pages, dict)
        assert len(pages) > 0

    @patch("src.crawler.time.sleep")
    def test_crawl_marks_pages_as_visited(self, mock_sleep):
        crawler = Crawler()
        mock_response = make_mock_response(SIMPLE_HTML)
        with patch.object(crawler.session, "get", return_value=mock_response):
            crawler.crawl()
        assert "https://quotes.toscrape.com" in crawler.visited

    @patch("src.crawler.time.sleep")
    def test_crawl_skips_failed_pages(self, mock_sleep):
        import requests as req
        crawler = Crawler()
        mock_response = make_mock_response("", 500)
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError("500")
        with patch.object(crawler.session, "get", return_value=mock_response):
            pages = crawler.crawl()
        assert isinstance(pages, dict)