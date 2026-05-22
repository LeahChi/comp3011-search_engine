from __future__ import annotations

import pytest
from unittest.mock import patch
from src.crawler import Crawler, CrawledPage
from src.indexer import Indexer
from src.search import Search

# --- realistic fake pages using Shakespeare quotes ---
# Note: stopwords are filtered during tokenization, so only the "surviving" words are listed in comments.

PAGE_HOME = CrawledPage(
    url="https://quotes.toscrape.com",
    title="Quotes to Scrape",
    # surviving words: little, water, clears, deed
    text="A little water clears us of this deed. Lady Macbeth Act 2 Scene 2.",
    links=["https://quotes.toscrape.com/page/2"]
)

PAGE_2 = CrawledPage(
    url="https://quotes.toscrape.com/page/2",
    title="Quotes to Scrape - Page 2",
    # surviving words: look, like, innocent, flower, serpent
    text="Look like the innocent flower but be the serpent under it. Lady Macbeth Act 1 Scene 5.",
    links=["https://quotes.toscrape.com/page/3"]
)

PAGE_3 = CrawledPage(
    url="https://quotes.toscrape.com/page/3",
    title="Quotes to Scrape - Page 3",
    # surviving words: tale, told, idiot, sound, fury, signifying, nothing
    text="It is a tale told by an idiot full of sound and fury signifying nothing. Macbeth Act 5 Scene 5.",
    links=[]
)

PAGES = {
    PAGE_HOME.url: PAGE_HOME,
    PAGE_2.url: PAGE_2,
    PAGE_3.url: PAGE_3,
}


@pytest.fixture
def pipeline(tmp_path):
    """
    Build a complete pipeline: Indexer built from fake pages, Search on top.
    Returns (indexer, search) tuple.
    """
    indexer = Indexer(index_path=str(tmp_path / "index.json"))
    indexer.build(PAGES)
    search = Search(indexer)
    return indexer, search


class TestFullPipeline:
    """
    Integration tests for the full crawler -> indexer -> search pipeline.
    Tests components working together rather than in isolation.
    """

    def test_index_built_from_pages(self, pipeline):
        """Index should contain words from all three pages."""
        indexer, _ = pipeline
        assert len(indexer.index) > 0

    def test_words_from_all_pages_indexed(self, pipeline):
        """Words from every page should appear in the index."""
        indexer, _ = pipeline
        assert "water" in indexer.index    # PAGE_HOME
        assert "serpent" in indexer.index  # PAGE_2
        assert "fury" in indexer.index     # PAGE_3

    def test_document_frequency_across_pages(self, pipeline):
        """DF should correctly count how many pages contain a word."""
        indexer, _ = pipeline
        # "macbeth" appears in all three pages as attribution
        assert indexer.index["macbeth"]["df"] == 3

    def test_tfidf_scores_computed(self, pipeline):
        """All postings should have TF-IDF scores computed."""
        indexer, _ = pipeline
        for word, data in indexer.index.items():
            for url, stats in data["postings"].items():
                assert "tfidf" in stats
                assert isinstance(stats["tfidf"], float)

    def test_rare_word_has_higher_tfidf(self, pipeline):
        """A rare word should have a higher TF-IDF than a common word."""
        indexer, _ = pipeline
        # "fury" only appears on PAGE_3 — should score higher
        # than "macbeth" which appears on all three pages
        fury_tfidf = indexer.index["fury"]["postings"][PAGE_3.url]["tfidf"]
        macbeth_tfidf = indexer.index["macbeth"]["postings"][PAGE_3.url]["tfidf"]
        assert fury_tfidf > macbeth_tfidf

    def test_save_and_reload_preserves_index(self, pipeline, tmp_path):
        """Saving and reloading the index should produce identical results."""
        indexer, _ = pipeline
        original = indexer.index.copy()

        # reload from disk into a fresh indexer
        new_indexer = Indexer(index_path=str(tmp_path / "index.json"))
        new_indexer.load()
        assert new_indexer.index == original

    def test_search_find_returns_correct_page(self, pipeline, capsys):
        """Find should return the page containing the searched word."""
        _, search = pipeline
        # "fury" only appears on PAGE_3
        search.find("fury")
        captured = capsys.readouterr()
        assert PAGE_3.url in captured.out

    def test_search_multi_word_intersection(self, pipeline, capsys):
        """Multi-word find should only return pages containing all words."""
        _, search = pipeline
        # "water" only on PAGE_HOME, "deed" only on PAGE_HOME
        # so result should only be PAGE_HOME
        search.find("water deed")
        captured = capsys.readouterr()
        assert PAGE_HOME.url in captured.out
        assert PAGE_2.url not in captured.out
        assert PAGE_3.url not in captured.out

    def test_search_not_operator_excludes_page(self, pipeline, capsys):
        """NOT operator should exclude pages containing the excluded word."""
        _, search = pipeline
        # "macbeth" appears on all three pages
        # "water" only on PAGE_HOME
        # "macbeth NOT water" should return PAGE_2 and PAGE_3 only
        search.find("macbeth NOT water")
        captured = capsys.readouterr()
        # append "\n" so we match PAGE_HOME's URL as a complete line, not as a
        # substring of PAGE_2's or PAGE_3's URL
        assert PAGE_HOME.url + "\n" not in captured.out
        assert PAGE_2.url in captured.out
        assert PAGE_3.url in captured.out

    def test_search_word_not_in_index(self, pipeline, capsys):
        """Searching for a word not in index should return no results."""
        _, search = pipeline
        search.find("zzzzzzz")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_search_suggests_correction(self, pipeline, capsys):
        """Typo in search should trigger did-you-mean suggestion."""
        _, search = pipeline
        # "furry" is close enough to "fury" to trigger suggestion
        search.find("furry")
        captured = capsys.readouterr()
        assert "Did you mean" in captured.out

    def test_print_word_shows_correct_page(self, pipeline, capsys):
        """Print should show the page the word appears on."""
        _, search = pipeline
        # "water" only appears on PAGE_HOME
        search.print_word("water")
        captured = capsys.readouterr()
        assert PAGE_HOME.url in captured.out
        assert "Document frequency" in captured.out

    def test_print_word_shows_all_pages(self, pipeline, capsys):
        """Print should show every page the word appears on."""
        _, search = pipeline
        # "macbeth" appears on all three pages
        search.print_word("macbeth")
        captured = capsys.readouterr()
        assert PAGE_HOME.url in captured.out
        assert PAGE_2.url in captured.out
        assert PAGE_3.url in captured.out

    def test_results_ranked_by_tfidf(self, pipeline, capsys):
        """Results should appear with relevance scores."""
        _, search = pipeline
        search.find("macbeth")
        captured = capsys.readouterr()
        assert "Relevance score" in captured.out


class TestCrawlerToIndexer:
    """
    Tests for the crawler -> indexer handoff using mocked HTTP.
    Verifies the output of the crawler feeds correctly into the indexer.
    """

    @patch("src.crawler.time.sleep")
    def test_crawler_output_feeds_indexer(self, mock_sleep, tmp_path):
        """Pages returned by crawler should be indexable without errors."""
        fake_pages = {PAGE_HOME.url: PAGE_HOME}
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(fake_pages)
        assert len(indexer.index) > 0

    @patch("src.crawler.time.sleep")
    def test_empty_crawl_produces_empty_index(self, mock_sleep, tmp_path):
        """Empty crawler output should produce an empty index."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build({})
        assert indexer.index == {}