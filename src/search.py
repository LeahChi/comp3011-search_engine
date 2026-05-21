from __future__ import annotations

import logging
import difflib                     # for "did you mean" suggestions
from src.indexer import Indexer
from src.tokenizer import tokenize # for cleaning the user's query the same way we cleaned the page text during indexing


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


class Search:
    """
    Searches the inverted index for words and phrases.

    Supports:
        - Single word search
        - Multi-word search with TF-IDF ranked results
        - Boolean operators: AND, NOT
        - Did-you-mean suggestions via difflib

    Attributes:
        indexer: The Indexer instance containing the inverted index.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def _suggest(self, word: str) -> str | None:
        """
        "Did you mean" functionality that helps users find results even if they misspell their query.

        Uses difflib to find the most similar word to the query.
        Returns None if no close match is found.

        Args:
            word: The word to find a suggestion for.

        Returns:
            Closest matching word or None.

        Time complexity: O(n) where n is number of words in index.
        """
        matches = difflib.get_close_matches(
            word,
            self.indexer.index.keys(),
            n=1,          # return at most 1 suggestion
            cutoff=0.6    # minimum similarity score (0-1)
        )
        return matches[0] if matches else None
    

    def print_word(self, word: str) -> None:
        """
        Print the full inverted index entry for a word.

        Shows document frequency, and for each page:
        term frequency, positions, and TF-IDF score.

        Args:
            word: The word to look up in the index.
        """
        # empty query blocker
        if not word.strip():
            print("Error: please provide a word to print.")
            return

        word = word.lower().strip()

        if word not in self.indexer.index:
            print(f"'{word}' not found in index.")
            suggestion = self._suggest(word)
            if suggestion:
                print(f"Did you mean: '{suggestion}'?")
            return

        data = self.indexer.index[word]
        print(f"\nWord: '{word}'")
        print(f"Document frequency: {data['df']}")
        print(f"Appears in {data['df']} page(s):\n")

        for url, stats in data["postings"].items():
            print(f"URL:       {url}")
            print(f"TF:        {stats['tf']}")
            print(f"Positions: {stats['positions']}")
            print(f"TF-IDF:    {stats['tfidf']}")
            print()

    def _find_single(self, word: str) -> dict[str, float]:
        """
        Find all pages containing a single word.

        Args:
            word: Clean lowercase word to search for.

        Returns:
            Dict mapping URL -> TF-IDF score. Empty if not found.

        Time complexity: O(1) index lookup.
        """
        if word not in self.indexer.index:
            return {}
        return {
            url: stats["tfidf"]
            for url, stats in self.indexer.index[word]["postings"].items()
        }

    def _rank_results(self, results: dict[str, float]) -> list[tuple[str, float]]:
        """
        Sort results by TF-IDF score descending.

        Args:
            results: Dict mapping URL -> TF-IDF score.

        Returns:
            List of (url, score) tuples sorted by score descending.

        Time complexity: O(n log n) where n is number of results.
        """
        return sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    def _parse_boolean(self, query: str) -> tuple[list[str], list[str]]:
        """
        Parse a query into must-include and must-exclude word lists.

        Supports AND and NOT operators.
        Examples:
            "good friends"        -> must=["good","friends"], exclude=[]
            "good AND friends"    -> must=["good","friends"], exclude=[]
            "good NOT bad"        -> must=["good"], exclude=["bad"]

        Args:
            query: Raw query string from the user.

        Returns:
            Tuple of (must_include, must_exclude) word lists.

        Time complexity: O(n) where n is number of query tokens.
        """
        must_include = []
        must_exclude = []
        exclude_next = False

        tokens = query.lower().split()

        for token in tokens:
            if token == "and":
                continue  # AND is implicit, just skip it
            elif token == "not":
                exclude_next = True
            else:
                clean = tokenize(token)
                if not clean:
                    continue
                if exclude_next:
                    must_exclude.append(clean[0])
                    exclude_next = False
                else:
                    must_include.append(clean[0])

        return must_include, must_exclude
    
    def find(self, query: str) -> None:
        """
        Find pages matching a query and print ranked results.

        Supports multi-word queries and Boolean operators (AND, NOT).
        Results are ranked by combined TF-IDF score.

        Args:
            query: Raw query string from the user.
        """
        # empty query guard
        if not query.strip():
            print("Error: please provide a search term.")
            return

        must_include, must_exclude = self._parse_boolean(query)

        if not must_include:
            print("Error: query contains no searchable terms.")
            return

        # --- find pages containing ALL must_include words ---
        # start with results for first word then intersect
        combined: dict[str, float] = self._find_single(must_include[0])

        if not combined:
            print(f"No results found for '{must_include[0]}'.")
            suggestion = self._suggest(must_include[0])
            if suggestion:
                print(f"Did you mean: '{suggestion}'?")
            return

        for word in must_include[1:]:
            word_results = self._find_single(word)
            if not word_results:
                print(f"No results found containing '{word}'.")
                suggestion = self._suggest(word)
                if suggestion:
                    print(f"Did you mean: '{suggestion}'?")
                return
            # intersect: only keep pages that contain this word too
            # add scores together for combined relevance ranking
            combined = {
                url: combined[url] + word_results[url]
                for url in combined
                if url in word_results
            }

        # --- remove pages containing any must_exclude words ---
        for word in must_exclude:
            exclude_urls = set(self._find_single(word).keys())
            combined = {
                url: score
                for url, score in combined.items()
                if url not in exclude_urls
            }

        if not combined:
            print("No pages match all search terms.")
            return

        # --- rank and display ---
        ranked = self._rank_results(combined)
        print(f"\nFound {len(ranked)} result(s):\n")
        for rank, (url, score) in enumerate(ranked, start=1):
            print(f"  {rank}. {url}")
            print(f"     Relevance score: {round(score, 6)}\n")