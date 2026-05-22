from __future__ import annotations

import pytest
from unittest.mock import patch
from src.crawler import CrawledPage
from src.indexer import Indexer
from src.search import Search

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


@pytest.fixture # no need to build the index from scratch in every test — can reuse it
def search(tmp_path) -> Search:
    """Build a real index from fake pages and return a Search instance."""
    indexer = Indexer(index_path=str(tmp_path / "index.json"))
    indexer.build(PAGES)
    return Search(indexer)


class TestSuggest:
    """Tests for the did-you-mean suggestion feature."""

    def test_suggests_close_match(self, search):
        """Should suggest a similar word from the index."""
        # "appel" is a typo of "apple" which is in the index
        suggestion = search._suggest("appel")
        assert suggestion == "apple"

    def test_returns_none_for_no_match(self, search):
        """Should return None when no similar word exists."""
        suggestion = search._suggest("xdftyuiklmnbvfgj")
        assert suggestion is None

    def test_returns_none_for_empty_string(self, search):
        """Should return None for empty input."""
        suggestion = search._suggest("")
        assert suggestion is None


class TestPrintWord:
    """Tests for the print command."""

    def test_prints_word_found(self, search, capsys):
        """Should print index entry for a known word."""
        search.print_word("apple")
        captured = capsys.readouterr()
        assert "apple" in captured.out
        # confirms the full postings entry was actually printed, not just an error containing the word.
        assert "Document frequency" in captured.out

    def test_prints_not_found_message(self, search, capsys):
        """Should print not found message for unknown word."""
        search.print_word("xdftyuiklmnbvfgj")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_prints_suggestion_when_close_match(self, search, capsys):
        """Should print did-you-mean when word not found but similar word exists."""
        search.print_word("appel")
        captured = capsys.readouterr()
        assert "Did you mean" in captured.out

    def test_empty_query_prints_error(self, search, capsys):
        """Should print error message for empty input."""
        search.print_word("")
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_case_insensitive(self, search, capsys):
        """Should handle uppercase input correctly."""
        search.print_word("APPLE")
        captured = capsys.readouterr()
        assert "apple" in captured.out
        assert "not found" not in captured.out


class TestFindSingle:
    """Tests for the internal _find_single method."""

    def test_finds_known_word(self, search):
        """Should return results for a word that exists in the index."""
        results = search._find_single("apple")
        assert len(results) > 0

    def test_returns_empty_for_unknown_word(self, search):
        """Should return empty dict for word not in index."""
        results = search._find_single("xdftyuiklmnbvfgj")
        assert results == {}

    def test_results_contain_tfidf_scores(self, search):
        """Results should map URLs to float TF-IDF scores."""
        results = search._find_single("apple")
        for url, score in results.items():
            assert isinstance(score, float)

    def test_word_on_multiple_pages(self, search):
        """Word appearing on multiple pages should return multiple results."""
        # "orange" appears on PAGE_1 and PAGE_2
        results = search._find_single("orange")
        assert PAGE_1.url in results
        assert PAGE_2.url in results


class TestRankResults:
    """Tests for the ranking method."""

    def test_ranks_by_score_descending(self, search):
        """Results should be sorted highest score first."""
        results = {"url1": 0.1, "url2": 0.9, "url3": 0.5}
        ranked = search._rank_results(results)
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_returns_list_of_tuples(self, search):
        """Should return a list of (url, score) tuples."""
        results = {"url1": 0.5}
        ranked = search._rank_results(results)
        assert isinstance(ranked, list)
        assert isinstance(ranked[0], tuple)

    def test_empty_results_returns_empty_list(self, search):
        """Empty input should return empty list."""
        assert search._rank_results({}) == []


class TestParseBoolean:
    """Tests for the Boolean query parser."""

    def test_simple_query(self, search):
        """Single word query should go into must_include."""
        must, exclude = search._parse_boolean("apple")
        assert "apple" in must
        assert exclude == []

    def test_multi_word_query(self, search):
        """Multiple words should all go into must_include."""
        must, exclude = search._parse_boolean("apple orange")
        assert "apple" in must
        assert "orange" in must
        assert exclude == []

    def test_and_operator(self, search):
        """AND should be ignored as it is implicit."""
        must, exclude = search._parse_boolean("apple AND orange")
        assert "apple" in must
        assert "orange" in must
        assert exclude == []

    def test_not_operator(self, search):
        """NOT should move the following word to must_exclude."""
        must, exclude = search._parse_boolean("apple NOT orange")
        assert "apple" in must
        assert "orange" in exclude

    def test_stopwords_ignored_in_query(self, search):
        """Stopwords in query should be filtered out."""
        must, exclude = search._parse_boolean("the apple")
        assert "the" not in must
        assert "apple" in must

class TestFind:
    """Tests for the main find command."""

    def test_find_single_word(self, search, capsys):
        """Should return results for a known word."""
        search.find("apple")
        captured = capsys.readouterr()
        assert "result" in captured.out

    def test_find_word_not_in_index(self, search, capsys):
        """Should print not found message for unknown word."""
        # "unknown" = similarity < 60% in "difflib.get_close_matches"
        search.find("xdftyuiklmnbvfgj")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_find_empty_query(self, search, capsys):
        """Should print error for empty query."""
        search.find("")
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_find_multi_word(self, search, capsys):
        """Should return pages containing all words."""
        # both PAGE_1 and PAGE_2 contain "orange"
        # only PAGE_1 contains "apple"
        # so result should only be PAGE_1
        search.find("apple orange")
        captured = capsys.readouterr()
        assert PAGE_1.url in captured.out
        assert PAGE_2.url not in captured.out

    def test_find_with_not_operator(self, search, capsys):
        """NOT operator should exclude pages containing that word."""
        # PAGE_1 and PAGE_2 both have orange
        # PAGE_1 also has apple
        # "orange NOT apple" should only return PAGE_2
        search.find("orange NOT apple")
        captured = capsys.readouterr()
        assert PAGE_2.url in captured.out
        assert PAGE_1.url + "\n" not in captured.out
        # append "\n" so we match PAGE_1's URL as a complete line, not as a
        # substring of PAGE_2's URL ("https://quotes.toscrape.com/page/2")

    def test_find_with_and_operator(self, search, capsys):
        """AND operator should work same as multi-word query."""
        search.find("apple AND orange")
        captured = capsys.readouterr()
        assert PAGE_1.url in captured.out

    def test_find_only_stopwords(self, search, capsys):
        """Query of only stopwords should print error."""
        search.find("the and is")
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_find_shows_relevance_score(self, search, capsys):
        """Results should display a relevance score."""
        search.find("apple")
        captured = capsys.readouterr()
        assert "Relevance score" in captured.out

    def test_find_results_case_insensitive(self, search, capsys):
        """Find should handle uppercase query correctly."""
        search.find("APPLE")
        captured = capsys.readouterr()
        assert "result" in captured.out

    def test_find_special_characters(self, search, capsys):
        """Special characters in query should be stripped and handled gracefully."""
        # "app!e" should be cleaned to "appe" which won't match anything but also shouldn't cause an error.
        search.find("app!e")
        captured = capsys.readouterr()
        assert "Error" not in captured.out or "No results" in captured.out

    def test_find_second_word_not_in_index(self, search, capsys):
        """Should handle second word in multi-word query not being in index."""
        # "apple" exists but "xdftyuiklmnbvfgj" does not
        search.find("apple xdftyuiklmnbvfgj")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_find_no_page_contains_all_words(self, search, capsys):
        """Should handle case where no page contains all search terms together."""
        # "apple" only on PAGE_1, "lemon" only on PAGE_2
        # no page has both so intersection is empty
        search.find("apple lemon")
        captured = capsys.readouterr()
        assert "No pages match" in captured.out

    def test_find_second_word_suggests_correction(self, search, capsys):
        """Should suggest correction when second word in query is not found but similar word exists."""
        # "apple" exists, "appel" is a typo of "apple" but treated as second word
        search.find("orange appel")
        captured = capsys.readouterr()
        assert "Did you mean" in captured.out