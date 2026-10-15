import asyncio
import httpx
import json
import logging
from bs4 import BeautifulSoup
from config import URLS_FILE, DATA_DIR, HEADERS

logger = logging.getLogger(__name__)

BASE_URL = "https://www.altnews.in/type/fact-check/page/{page}/"
CONCURRENCY = 20
MAX_PAGES = 250  # generous upper bound


async def _fetch_page(client, semaphore, page_num):
    async with semaphore:
        url = BASE_URL.format(page=page_num)
        try:
            resp = await client.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                return page_num, None  # signals end of listing
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            cards = soup.select("article.card h3.card__title a")
            if not cards:
                cards = soup.select("article a[href*='altnews.in/']")
            urls = [a["href"] for a in cards if a.get("href")]
            return page_num, urls
        except Exception as e:
            logger.warning(f"Page {page_num} failed: {e}")
            return page_num, []


async def _scrape_all_pages():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_urls = []
    reached_end = False

    async with httpx.AsyncClient(http2=False, follow_redirects=True) as client:
        # Process in batches to detect when listing ends
        batch_size = CONCURRENCY
        for batch_start in range(1, MAX_PAGES + 1, batch_size):
            if reached_end:
                break
            batch_end = min(batch_start + batch_size, MAX_PAGES + 1)
            tasks = [_fetch_page(client, semaphore, p) for p in range(batch_start, batch_end)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])

            for page_num, urls in results:
                if urls is None:
                    reached_end = True
                    break
                if urls:
                    all_urls.extend(urls)
                    logger.info(f"Page {page_num}: {len(urls)} URLs (total: {len(all_urls)})")
                else:
                    # Empty page but not 404 — might be end, might be fluke
                    # Continue to next batch; if consecutive empties, the 404 will catch it
                    logger.warning(f"Page {page_num}: no URLs found")

    return all_urls


def scrape_listing_pages(max_pages=250):
    """Sync wrapper. Returns list of article URL strings."""
    urls = asyncio.run(_scrape_all_pages())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # Save to file
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(unique)} unique URLs to {URLS_FILE}")
    return unique
