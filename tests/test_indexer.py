from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.crawler import CrawledPage
from src.indexer import Indexer
from src.tokenizer import tokenize
from src.ranking import compute_tf, compute_idf, compute_tfidf


# --- fake pages used across multiple tests ---

PAGE_1 = CrawledPage(
    url="https://quotes.toscrape.com",
    title="Page 1",
    text="apple orange banana apple pear orange",
    links=[]
)

PAGE_2 = CrawledPage(
    url="https://quotes.toscrape.com/page/2",
    title="Page 2",
    text="orange lemon lime mango",
    links=[]
)

PAGE_3 = CrawledPage(
    url="https://quotes.toscrape.com/page/3",
    title="Page 3",
    text="",  # empty page — edge case
    links=[]
)

PAGES = {
    PAGE_1.url: PAGE_1,
    PAGE_2.url: PAGE_2,
}

PAGES_WITH_EMPTY = {
    PAGE_1.url: PAGE_1,
    PAGE_2.url: PAGE_2,
    PAGE_3.url: PAGE_3,
}


class TestTokenizer:
    """Tests for the tokenize function in tokenizer.py."""

    def test_lowercases_text(self):
        """All tokens should be lowercase."""
        tokens = tokenize("Apple ORANGE Banana")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stopwords(self):
        """Common stopwords should be filtered out."""
        tokens = tokenize("the boy sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens

    def test_removes_punctuation(self):
        """Punctuation should be stripped from tokens."""
        tokens = tokenize("lucy, keep running! don't stop.")
        assert "lucy" in tokens
        assert "running" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string_returns_empty_list(self):
        """Empty input should return empty list."""
        assert tokenize("") == []

    def test_only_stopwords_returns_empty_list(self):
        """Text containing only stopwords should return empty list."""
        tokens = tokenize("the a is and")
        assert tokens == []

    def test_returns_list(self):
        """tokenize should always return a list."""
        assert isinstance(tokenize("apple orange"), list)


class TestRanking:
    """Tests for TF-IDF ranking functions in ranking.py."""

    def test_compute_tf_basic(self):
        """TF should be word count divided by total words."""
        assert compute_tf(2, 10) == 0.2

    def test_compute_tf_zero_total_words(self):
        """TF should return 0.0 when page has no words."""
        assert compute_tf(0, 0) == 0.0

    def test_compute_idf_basic(self):
        """IDF should be log(total_docs / docs_containing_word)."""
        import math
        assert compute_idf(10, 2) == pytest.approx(math.log(5), rel=1e-5)

    def test_compute_idf_zero_docs_containing_word(self):
        """IDF should return 0.0 when no documents contain the word."""
        assert compute_idf(10, 0) == 0.0

    def test_compute_idf_word_in_all_docs(self):
        """IDF should be 0.0 when word appears in every document."""
        import math
        assert compute_idf(10, 10) == pytest.approx(math.log(1), rel=1e-5)

    def test_compute_tfidf_basic(self):
        """TF-IDF should be TF multiplied by IDF."""
        assert compute_tfidf(0.2, 1.5) == pytest.approx(0.3, rel=1e-5)

    def test_compute_tfidf_zero(self):
        """TF-IDF should be 0.0 when either TF or IDF is 0."""
        assert compute_tfidf(0.0, 1.5) == 0.0
        assert compute_tfidf(0.2, 0.0) == 0.0


class TestIndexerInit:
    """Tests for Indexer initialisation."""

    def test_index_starts_empty(self):
        """Index should be empty on initialisation."""
        indexer = Indexer()
        assert indexer.index == {}

    def test_default_index_path(self):
        """Default index path should be data/index.json."""
        indexer = Indexer()
        assert indexer.index_path == Path("data/index.json")

    def test_custom_index_path(self):
        """Custom index path should be stored correctly."""
        indexer = Indexer(index_path="data/custom.json")
        assert indexer.index_path == Path("data/custom.json")


class TestIndexerBuild:
    """Tests for the build method."""

    def test_build_creates_index(self, tmp_path):
        """Build should populate the index with words from pages."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        assert len(indexer.index) > 0

    def test_build_indexes_known_word(self, tmp_path):
        """A known word from the pages should appear in the index."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        # "apple" appears in PAGE_1, should be indexed
        assert "apple" in indexer.index

    def test_build_correct_df(self, tmp_path):
        """Document frequency should reflect how many pages contain the word."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        # "orange" appears in both PAGE_1 and PAGE_2
        assert indexer.index["orange"]["df"] == 2

    def test_build_correct_tf(self, tmp_path):
        """Term frequency should reflect how many times word appears on page."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        # "apple" appears twice in PAGE_1
        assert indexer.index["apple"]["postings"][PAGE_1.url]["tf"] == 2

    def test_build_stores_positions(self, tmp_path):
        """Positions list should be stored for each word on each page."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        positions = indexer.index["apple"]["postings"][PAGE_1.url]["positions"]
        assert isinstance(positions, list)
        assert len(positions) == 2  # "apple" appears twice

    def test_build_computes_tfidf(self, tmp_path):
        """TF-IDF scores should be computed and stored for each posting."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        tfidf = indexer.index["apple"]["postings"][PAGE_1.url]["tfidf"]
        assert isinstance(tfidf, float)

    def test_build_handles_empty_page(self, tmp_path):
        """Build should handle pages with no text without crashing."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES_WITH_EMPTY)
        assert isinstance(indexer.index, dict)

    def test_build_excludes_stopwords(self, tmp_path):
        """Stopwords should not appear in the index."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        assert "the" not in indexer.index
        assert "and" not in indexer.index


class TestIndexerSaveLoad:
    """Tests for save and load methods."""

    def test_save_creates_file(self, tmp_path):
        """Save should create a JSON file on disk."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        assert (tmp_path / "index.json").exists()

    def test_save_creates_valid_json(self, tmp_path):
        """Saved file should be valid JSON."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        with open(tmp_path / "index.json", "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_load_restores_index(self, tmp_path):
        """Load should restore the exact same index that was saved."""
        indexer = Indexer(index_path=str(tmp_path / "index.json"))
        indexer.build(PAGES)
        original_index = indexer.index.copy()
        # clear and reload
        indexer.index = {}
        indexer.load()
        assert indexer.index == original_index

    def test_load_raises_when_no_file(self, tmp_path):
        """Load should raise FileNotFoundError when no index exists."""
        indexer = Indexer(index_path=str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError):
            indexer.load()
