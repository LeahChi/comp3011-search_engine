# comp3011-search_engine
# COMP3011 Search Engine

A command-line search engine that crawls, indexes, and searches [quotes.toscrape.com](https://quotes.toscrape.com). Built in Python using BFS crawling, an inverted index with TF-IDF ranking, Boolean query operators, and did-you-mean suggestions.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Complexity Analysis](#complexity-analysis)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Overview

Three core components work together in a pipeline:

- **Crawler** — BFS crawl with a 6 second politeness window, retry logic, and graceful error handling
- **Indexer** — Builds an inverted index storing term frequency, positions, and TF-IDF scores for every word on every page
- **Search** — Ranked retrieval using TF-IDF scoring, Boolean operators (AND/NOT), and did-you-mean suggestions

---

## Architecture
src/
├── crawler.py      # BFS crawler with politeness window and retry logic
├── indexer.py      # Inverted index builder with TF-IDF scoring
├── search.py       # Query processor with ranking and Boolean operators
├── ranking.py      # TF-IDF computation functions
├── tokenizer.py    # Tokenization, lowercasing, stopword removal
└── main.py         # Interactive CLI shell
tests/
├── test_crawler.py       # 22 unit tests
├── test_indexer.py       # 28 unit tests
├── test_search.py        # 32 unit tests
└── test_integration.py   # 16 integration tests
data/
└── index.json      # Compiled inverted index (214 pages)

### Index Structure

```json
{
  "word": {
    "df": 3,
    "postings": {
      "https://quotes.toscrape.com/page/1": {
        "tf": 4,
        "positions": [2, 15, 34, 67],
        "tfidf": 0.482691
      }
    }
  }
}
```

- `df` — document frequency: number of pages containing this word
- `tf` — term frequency: number of times the word appears on this page
- `positions` — word positions on the page
- `tfidf` — precomputed relevance score

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/comp3011-search-engine.git
cd comp3011-search-engine

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK stopwords
python3 -c "import nltk; nltk.download('stopwords')"
```

---

## Usage

Start the interactive shell:

```bash
python3 -m src.main
```

### `build`
Crawls the website and builds the inverted index. Takes approximately 20 minutes due to the 6 second politeness window.

### `load`
Loads a previously built index from disk.

### `print <word>`
Prints the full index entry for a word.

### `find <query>`
Finds pages containing the query, ranked by TF-IDF score. Supports multi-word queries and Boolean operators.


---

## Complexity Analysis

| Operation | Complexity | Notes |
|---|---|---|
| Crawl | O(V + E) | V = pages, E = links |
| Build index | O(P × T) | P = pages, T = tokens per page |
| Index lookup | O(1) | Hash map lookup |
| Multi-word find | O(k × m) | k = query words, m = results per word |
| Rank results | O(n log n) | n = matching pages |
| Did-you-mean | O(n) | n = words in index |

### Key Design Decisions

- **`deque` for BFS** — O(1) `popleft()` vs O(n) for a list
- **`set` for visited URLs** — O(1) lookup vs O(n) for a list
- **`dict` for index** — O(1) word lookup at query time
- **TF-IDF precomputed at build time** — trades build time for instant query speed
- **Two-pass indexing** — pass 1 builds raw postings, pass 2 computes TF-IDF once all document frequencies are known
- **`requests.Session()`** — reuses TCP connections across all requests

---

## Testing

```bash
# run all tests
pytest tests/ -v

# run with coverage report
pytest --cov=src --cov-report=term-missing -v
```

### Coverage

| File | Coverage |
|---|---|
| crawler.py | 100% |
| indexer.py | 100% |
| ranking.py | 100% |
| search.py | 100% |
| tokenizer.py | 100% |

### Strategy

- **Unit tests** — each component tested in isolation with mocked HTTP calls
- **Integration tests** — full crawler → indexer → search pipeline end to end
- **Edge cases** — empty queries, unknown words, special characters, connection errors, timeouts
- **Coverage-driven** — used `pytest --cov` to identify uncovered lines and added targeted tests

---

## Dependencies
requests
beautifulsoup4
nltk
pytest
pytest-cov
responses

```bash
pip install -r requirements.txt
```