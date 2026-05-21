from __future__ import annotations

import math


def compute_tf(word_count: int, total_words: int) -> float:
    """
    Compute Term Frequency for a word on a single page.

    TF = number of times word appears / total words on page

    Args:
        word_count:  How many times the word appears on this page.
        total_words: Total number of words on this page.

    Returns:
        TF score as a float. Returns 0.0 if page has no words.

    Time complexity: O(1)
    """
    if total_words == 0:
        return 0.0
    return word_count / total_words


def compute_idf(total_docs: int, docs_containing_word: int) -> float:
    """
    Compute Inverse Document Frequency for a word across all pages.

    IDF = log(total documents / documents containing the word)

    Args:
        total_docs:           Total number of pages crawled.
        docs_containing_word: Number of pages containing this word.

    Returns:
        IDF score as a float. Returns 0.0 if no documents contain the word.

    Time complexity: O(1)
    """
    if docs_containing_word == 0:
        return 0.0
    return math.log(total_docs / docs_containing_word)


def compute_tfidf(tf: float, idf: float) -> float:
    """
    TF-IDF score = TF * IDF.

    Args:
        tf:  Term frequency score.
        idf: Inverse document frequency score.

    Returns:
        TF-IDF score rounded to 6 decimal places.

    Time complexity: O(1)
    """
    return round(tf * idf, 6)