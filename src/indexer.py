from __future__ import annotations

import json
import logging
from pathlib import Path

from src.crawler import CrawledPage # The dataclass from crawler.py that represents a crawled page
from src.tokenizer import tokenize
from src.ranking import compute_tf, compute_idf, compute_tfidf

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",datefmt="%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)

# alias for "dict[str, dict]" to represent the structure of the index
Index = dict[str, dict]


class Indexer:
    """
    Builds and manages an inverted index from crawled pages.

    # Below describes how the dict would look after build() has finished.
    The index structure is:
        {
            word: {
                "df": int,
                "postings": {
                    url: {
                        "tf": int,
                        "positions": list,
                        "tfidf": float
                    }
                }
            }
        }

    Attributes:
        index:      The inverted index dictionary.
        index_path: Path to save/load the index file.
    """

    def __init__(self, index_path: str = "data/index.json") -> None:
        self.index: Index = {}
        self.index_path = Path(index_path)

    def _build_page_postings(
        self, tokens: list[str]
    ) -> dict[str, dict]:
        """
        Build postings for a single page from its tokens.

        Records term frequency and positions for each word.

        Args:
            tokens: List of clean tokens from a page.

        Returns:
            Dict mapping word -> {tf, positions} for this page.

        Time complexity: O(n) where n is number of tokens.
        """
        postings: dict[str, dict] = {}

        for position, word in enumerate(tokens):
            # enumerate gives us both the index (position) and the word itself
            if word not in postings:
                postings[word] = {"tf": 0, "positions": []}
            postings[word]["tf"] += 1
            postings[word]["positions"].append(position)

        return postings

    def build(self, pages: dict[str, CrawledPage]) -> None:
        """
        Build the inverted index from all crawled pages.

        Processes each page, computes TF-IDF scores, and stores
        results in the index. Saves automatically after building.

        Args:
            pages: Dict mapping URL -> CrawledPage from the crawler.

        Time complexity: O(P * T) where P = pages, T = tokens per page.
        """
        logger.info(f"Building index from {len(pages)} pages...")
        self.index = {}
        total_docs = len(pages)

        # --- Pass 1: build raw postings and document frequency ---
        for url, page in pages.items():
            tokens = tokenize(page.text)
            total_words = len(tokens)
            page_postings = self._build_page_postings(tokens)

            for word, stats in page_postings.items():
                if word not in self.index:
                    self.index[word] = {"df": 0, "postings": {}}

                self.index[word]["df"] += 1
                self.index[word]["postings"][url] = {
                    "tf": stats["tf"],
                    "positions": stats["positions"],
                    "tfidf": 0.0,
                    "total_words": total_words,
                }

        # --- Pass 2: compute TF-IDF now we know df for every word ---
        for word, data in self.index.items():
            idf = compute_idf(total_docs, data["df"])
            for url, stats in data["postings"].items():
                tf = compute_tf(stats["tf"], stats["total_words"])
                stats["tfidf"] = compute_tfidf(tf, idf)
                del stats["total_words"]

        logger.info(f"Index built. {len(self.index)} unique terms indexed.")
        self.save()

    def save(self) -> None:
        """
        Save the index to disk as a JSON file.

        Creates the data directory if it doesn't exist.

        Time complexity: O(n) where n is number of index entries.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
            # Characters like é, and ü are saved properly rather than as escaped unicode sequences.
        logger.info(f"Index saved to {self.index_path}")

    def load(self) -> None:
        """
        Load the index from disk.

        Raises:
            FileNotFoundError: If no index file exists.
                Run 'build' first to create one.

        Time complexity: O(n) where n is number of index entries.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_path}. Run 'build' first."
            )
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        logger.info(f"Index loaded from {self.index_path}. {len(self.index)} terms.")