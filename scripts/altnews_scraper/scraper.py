import argparse
import json
import time
import logging
import os
from config import URLS_FILE, ARTICLES_FILE, DATA_DIR, LOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_tier1():
    from listing_scraper import scrape_listing_pages
    print("\n" + "="*60)
    print("TIER 1: Collecting article URLs")
    print("="*60)
    start = time.time()
    urls = scrape_listing_pages()
    elapsed = time.time() - start
    print(f"Collected {len(urls)} URLs in {elapsed:.1f}s")
    return urls


def run_tier2(urls=None):
    from article_scraper import scrape_articles
    print("\n" + "="*60)
    print("TIER 2: Scraping articles")
    print("="*60)

    if urls is None:
        if not os.path.exists(URLS_FILE):
            print(f"ERROR: {URLS_FILE} not found. Run --tier 1 first.")
            return
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            urls = json.load(f)

    print(f"Processing {len(urls)} articles with concurrency=20...")
    start = time.time()
    articles = scrape_articles(urls)
    elapsed = time.time() - start
    print(f"Scraped {len(articles)} articles in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return articles


def main():
    parser = argparse.ArgumentParser(description="Alt News Scraper")
    parser.add_argument("--tier", choices=["1", "2", "all"], default="all",
                        help="Which tier to run (1=URLs, 2=articles, all=both)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    total_start = time.time()

    if args.tier in ("1", "all"):
        urls = run_tier1()
    else:
        urls = None

    if args.tier in ("2", "all"):
        run_tier2(urls)

    total_elapsed = time.time() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
