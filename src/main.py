from __future__ import annotations

import logging
from src.crawler import Crawler
from src.indexer import Indexer
from src.search import Search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)

def run_shell() -> None:
    """
    Run the interactive command-line shell.

    Available commands:
        build           Crawl the website and build the index.
        load            Load a previously built index from disk.
        print <word>    Print the index entry for a word.
        find <query>    Find pages matching a query.
        help            Show available commands.
        exit            Exit the shell.
    """
    indexer = Indexer()
    search = Search(indexer)

    print("Search Engine Shell")
    print("Type 'help' for available commands.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLeaving... Goodbye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        # split into command and arguments, e.g. "print apple" -> command="print", args="apple"

        if command == "build":
            _handle_build(indexer)

        elif command == "load":
            _handle_load(indexer)

        elif command == "print":
            _handle_print(search, args)

        elif command == "find":
            _handle_find(search, args)

        elif command == "help":
            _handle_help()

        elif command == "exit":
            print("Exiting... Goodbye!")
            break

        else:
            print(f"Unknown command: '{command}'. Type 'help' for available commands.")


def _handle_build(indexer: Indexer) -> None:
    """Crawl the website and build the index."""
    print("Starting crawl... this will take a few minutes due to the politeness window.")
    crawler = Crawler()
    pages = crawler.crawl()
    indexer.build(pages)
    print(f"Build complete. {len(pages)} pages indexed.")


def _handle_load(indexer: Indexer) -> None:
    """Load the index from disk."""
    try:
        indexer.load()
        print(f"Index loaded successfully! {len(indexer.index)} terms available.")
    except FileNotFoundError as e:
        print(f"Error: {e}")


def _handle_print(search: Search, args: str) -> None:
    """Print the index entry for a word."""
    if not args.strip():
        print("Usage: print <word>")
        print("Example: print indifference")
        return
    search.print_word(args)


def _handle_find(search: Search, args: str) -> None:
    """Find pages matching a query."""
    if not args.strip():
        print("Usage: find <query>")
        print("Example: find good friends")
        print("Example: find love NOT hate")
        return
    search.find(args)


def _handle_help() -> None:
    """Print available commands."""
    print("""
    Available commands:
    build              Crawl the website and build the inverted index
    load               Load a previously built index from disk
    print <word>       Print the index entry for a single word
    find <query>       Find pages matching a query (supports AND, NOT)
    help               Show this help message
    exit               Exit the shell

    Examples:
    > build
    > load
    > print indifference
    > find good friends
    > find love AND life
    > find hope NOT despair""")


if __name__ == "__main__":
    run_shell()