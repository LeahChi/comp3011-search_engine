from __future__ import annotations

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)  # stops nltk printing download messages every time the code runs

"""
Original: ['this', 'is', 'a', 'sample', 'sentence', 'showing', 'stopword', 'removal', '.'] 
Filtered: ['sample', 'sentence', 'showing', 'stopword', 'removal', '.']
"""

STOPWORDS: set[str] = set(stopwords.words("english"))
# Pre loads the stopwords into a set for O(1) lookups

def tokenize(text: str) -> list[str]:
    """
    Convert raw text into a list of meaningful tokens.

    Steps:
        1. Lowercase everything
        2. Strip punctuation and non-alphabetic characters
        3. Remove stopwords
        4. Remove empty strings

    Args:
        text: Raw text extracted from a web page.

    Returns:
        List of clean tokens ready for indexing.

    Time complexity: O(n) where n is the number of words in the text.
    """
    # 1. lowercase
    text = text.lower()

    # 2. replace anything that isn't a letter or space with a space {REGEX}
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. split into individual words
    words = text.split()

    # 4. remove stopwords and empty strings
    tokens = [word for word in words if word and word not in STOPWORDS]

    return tokens