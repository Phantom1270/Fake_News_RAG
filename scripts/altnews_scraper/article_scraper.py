import asyncio
import httpx
import json
import os
import logging
from bs4 import BeautifulSoup
from config import ARTICLES_FILE, DATA_DIR, HEADERS, ARTICLE_SELECTORS

logger = logging.getLogger(__name__)

CONCURRENCY = 20
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 5]  # seconds between retries
SAVE_EVERY = 50  # save progress every N articles


# ── Extraction helpers (keep all existing ones) ──

def extract_title(soup):
    sel = ARTICLE_SELECTORS.get("title", "h1")
    el = soup.select_one(sel)
    if not el:
        # Fallback selectors
        el = soup.select_one("h1.article-headline") or soup.select_one("h1.entry-title") or soup.find("h1")
    return el.get_text().strip() if el else None


def extract_body(soup):
    sel = ARTICLE_SELECTORS.get("body", ".article-body")
    el = soup.select_one(sel)
    if not el:
        # Fallback
        el = soup.select_one(".entry-content") or soup.select_one("article")
    if not el:
        return None
    paras = el.find_all("p")
    if not paras:
        return None
    text = "\n\n".join(p.get_text().strip() for p in paras if p.get_text().strip())
    return text if text else None


def extract_author(soup):
    el = soup.select_one(".author-name") or soup.select_one(".byline") or soup.select_one("a[rel='author']")
    return el.get_text().strip() if el else None


def extract_date(soup):
    el = soup.select_one("time[datetime]")
    if el:
        return el.get("datetime", el.get_text().strip())
    el = soup.select_one(".posted-on time") or soup.select_one(".entry-date")
    return el.get_text().strip() if el else None


def extract_category(soup):
    el = soup.select_one(".cat-links a") or soup.select_one(".entry-category")
    return el.get_text().strip() if el else "Fact Check"


def extract_verdict_raw(soup):
    el = soup.select_one(".verdict-badge") or soup.select_one("[class*='verdict']")
    return el.get_text().strip() if el else None


def extract_verdict_class(soup):
    el = soup.select_one(".verdict-badge") or soup.select_one("[class*='verdict']")
    return " ".join(el.get("class", [])) if el else ""


def normalize_verdict(raw):
    if not raw:
        return "UNKNOWN"
    raw_lower = raw.lower().strip()
    if any(w in raw_lower for w in ["false", "fake", "misleading", "fabricated", "unsubstantiated", "incorrect"]):
        return "FAKE"
    if any(w in raw_lower for w in ["true", "correct", "genuine", "authentic"]):
        return "TRUE"
    if any(w in raw_lower for w in ["partly", "half", "mixed", "mostly"]):
        return "MIXED"
    return raw.upper()


def extract_featured_image(soup):
    el = soup.select_one("meta[property='og:image']")
    if el:
        return el.get("content")
    img = soup.select_one("article img") or soup.select_one(".featured-image img")
    return img.get("src") if img else None


def extract_tags(soup):
    tags = []
    for el in soup.select(".tags-links a") or soup.select("a[rel='tag']"):
        t = el.get_text().strip()
        if t:
            tags.append(t)
    return tags


def extract_claim_review(soup):
    """Extract ClaimReview from JSON-LD, including @graph structures."""
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(data, dict):
            if data.get("@type") == "ClaimReview":
                return _parse_claim_review(data)
            if "@graph" in data:
                for item in data["@graph"]:
                    if isinstance(item, dict) and item.get("@type") == "ClaimReview":
                        return _parse_claim_review(item)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "ClaimReview":
                    return _parse_claim_review(item)
    return {}


def _parse_claim_review(cr):
    claim_text = cr.get("claimReviewed", "")
    review = cr.get("reviewRating", {})
    rating_value = review.get("alternateName") or review.get("ratingValue") or review.get("bestRating", "")
    author = ""
    item_reviewed = cr.get("itemReviewed", {})
    if isinstance(item_reviewed, dict):
        author = item_reviewed.get("author", {}).get("name", "")
    return {
        "claim": claim_text,
        "verdict": str(rating_value),
        "rating": str(review.get("ratingValue", "")),
        "author": author
    }


# ── Validation ──

def validate_article(article):
    """Check that an article has minimum viable data."""
    if not article.get("title"):
        return False, "missing title"
    if not article.get("body"):
        return False, "missing body"
    return True, "ok"


# ── Async scraping engine ──

async def _fetch_one(client, semaphore, url, index, total):
    """Fetch and parse one article with retry logic."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, headers=HEADERS, timeout=20)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")

                article = {
                    "source": "altnews",
                    "url": url,
                    "title": extract_title(soup),
                    "author": extract_author(soup),
                    "date": extract_date(soup),
                    "category": extract_category(soup),
                    "verdict_raw": extract_verdict_raw(soup),
                    "verdict_class": extract_verdict_class(soup),
                    "verdict_normalized": normalize_verdict(extract_verdict_raw(soup)),
                    "verdict_source": "badge",
                    "summary": None,
                    "featured_image": extract_featured_image(soup),
                    "body": extract_body(soup),
                    "tags": extract_tags(soup),
                    "claim_review": extract_claim_review(soup),
                    "enriched": True
                }

                if article["body"]:
                    article["summary"] = article["body"][:300]

                valid, reason = validate_article(article)
                if valid:
                    print(f"  [{index}/{total}] OK: {article['title'][:65]}")
                    return article
                else:
                    # Data extraction failed — retry might help if it was a bad response
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_BACKOFF[attempt])
                        continue
                    else:
                        print(f"  [{index}/{total}] INCOMPLETE ({reason}): {article['title'][:50]}")
                        return article  # Return what we have, mark as incomplete

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited — back off hard
                    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)] * 3
                    print(f"  [{index}/{total}] RATE LIMITED, waiting {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                else:
                    print(f"  [{index}/{total}] HTTP ERROR {e.response.status_code}: {url}")

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                else:
                    print(f"  [{index}/{total}] FAIL: {url} — {e}")

    # All retries exhausted
    return {
        "source": "altnews", "url": url, "title": None, "body": None,
        "summary": None, "claim_review": {}, "enriched": False, "_error": True
    }


async def _scrape_all_articles(urls):
    """Scrape all articles with controlled concurrency and incremental saving."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(urls)
    
    os.makedirs(DATA_DIR, exist_ok=True)

    # Better approach: batched processing with periodic saves
    results = []
    batch_size = SAVE_EVERY

    async with httpx.AsyncClient(http2=False, follow_redirects=True) as client:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_urls = urls[batch_start:batch_end]

            tasks = [
                _fetch_one(client, semaphore, url, batch_start + i + 1, total)
                for i, url in enumerate(batch_urls)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Incremental save
            _save_progress(results, total)

    return results


def _save_progress(results, total):
    """Save current results to file (incremental checkpoint)."""
    successful = [r for r in results if r and not r.get("_error")]
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(successful, f, indent=2, ensure_ascii=False)
    print(f"  [SAVE] Saved {len(successful)}/{total} articles to disk")


def scrape_articles(urls):
    """Sync wrapper. Returns list of article dicts."""
    results = asyncio.run(_scrape_all_articles(urls))

    # Final save — only successful articles
    successful = [r for r in results if r and not r.get("_error")]
    failed = [r for r in results if r and r.get("_error")]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(successful, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Saved to:   {ARTICLES_FILE}")
    print(f"{'='*60}")

    # If there were failures, save them separately for debugging
    if failed:
        fail_file = os.path.join(DATA_DIR, "altnews_failed.json")
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f"  Failed URLs saved to: {fail_file}")

    return successful
