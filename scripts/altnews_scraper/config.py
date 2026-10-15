# config.py
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data", "scraped")
URLS_FILE = os.path.join(DATA_DIR, "altnews_urls.json")
ARTICLES_FILE = os.path.join(DATA_DIR, "altnews_articles.json")
LOG_DIR = os.path.join(_BASE_DIR, "logs")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


ALTNEWS_SELECTORS = {
    "base_url": "https://www.altnews.in/type/fact-check",
    "pagination": "https://www.altnews.in/type/fact-check/page/{page}/",
    "card": "article.card.card--standard",
    "title": "h3.card__title a",
    "verdict": "span.verdict-badge",
    "date": "time.card__date",
    "date_attr": "datetime",
    "author": "a.card__author-name",
    "content_type": "span.card__content-type",
    "image": ".card__image img",
    "next_page": "a.next.page-numbers",
    "articles_per_page": 10,
    "total_pages": 498,
}

ARTICLE_SELECTORS = {
    "title": "h1.article-headline",
    "body": ".article-body",
    "structured_data_type": "ClaimReview",
}

VERDICT_MAP = {
    "false": "FAKE",
    "fake": "FAKE",
    "fabricated": "FAKE",
    "doctored": "FAKE",
    "ai-generated": "AI-GENERATED",
    "deepfake": "DEEPFAKE",
    "misleading": "MISLEADING",
    "partly false": "MISLEADING",
    "partially false": "MISLEADING",
    "out of context": "MISLEADING",
    "true": "TRUE",
    "real": "TRUE",
    "unrelated": "FALSE_CONTEXT",
    "old": "FALSE_CONTEXT",
}

CLASS_VERDICT_MAP = {
    "verdict-badge--false": "FAKE",
    "verdict-badge--true": "TRUE",
    "verdict-badge--misleading": "MISLEADING",
    "verdict-badge--partially-false": "MISLEADING",
}

# ─── Scraper limits ────────────────────────────────────────────────────────────
# Maximum number of article URLs to collect in Tier 1.
# Each listing page has ~10 articles, so MAX_URLS / 10 ≈ pages fetched.
MAX_URLS = 2000  # change this to scrape more or fewer articles

# ─── Concurrency ──────────────────────────────────────────────────────────────
TIER1_WORKERS = 6   # parallel Playwright browser pages for listing scrape
TIER2_WORKERS = 10  # parallel requests threads for article enrichment

# ─── Rate limits ──────────────────────────────────────────────────────────────
RATE_LIMIT = {
    # Tier 1: one small delay per page (browser already takes time to render)
    "delay_seconds_tier1": 0.5,
    "jitter_seconds": 0.3,
    # Tier 2: minimal delay — lightweight HTTP requests
    "delay_seconds_tier2": 0.5,
    "retry_attempts": 3,
    "backoff_base": 2,
    "rate_limit_wait": 30,
    "request_timeout": 20,
}
