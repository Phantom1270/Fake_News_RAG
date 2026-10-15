"""
test_scraper.py
End-to-end quick test for the Alt News scraper fixes.
"""

import sys
import os
from pathlib import Path

# Add project root to path for RAG imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from listing_scraper import scrape_listing_pages
from article_scraper import scrape_articles
from rag.pipeline.adapters import adapt_scraped_facts
from rag.pipeline.normalizer import UniversalNormalizer

NUM_TEST_ARTICLES = 5

def main():
    print(f"=== STEP 1: Scraping listing pages for URLs ===")
    # 1 page should give ~10 URLs, enough for our test
    urls = scrape_listing_pages(max_pages=1)
    urls = urls[:NUM_TEST_ARTICLES]
    print(f"Found {len(urls)} URLs for testing.\n")
    
    if len(urls) < NUM_TEST_ARTICLES:
        print(f"Failed to find {NUM_TEST_ARTICLES} URLs.")
        sys.exit(1)

    print(f"=== STEP 2: Scraping {NUM_TEST_ARTICLES} articles ===")
    articles = scrape_articles(urls)
    print("")

    print(f"=== STEP 3: Adapting scraped results ===")
    adapted = adapt_scraped_facts(articles, "altnews")
    entries = adapted.get("results", {}).get("fact_check", [])
    for i, entry in enumerate(entries, 1):
        data = entry.get("data", [{}])[0]
        claim = data.get("claim", "")[:50].replace('\n', ' ')
        print(f"[{i}] Adapted - Claim: {claim}...")
    print("")
    
    print(f"=== STEP 4: Normalizing adapted results ===")
    normalizer = UniversalNormalizer()
    normalized_docs = []
    for i, entry in enumerate(entries, 1):
        docs = normalizer.normalize_entry(entry)
        if docs:
            doc = docs[0]
            normalized_docs.append(doc)
            print(f"[{i}] Normalized - ID: {doc.doc_id}, Type: {doc.content_type}")
        else:
            print(f"[{i}] Normalized - Failed to produce NormalizedDocument")
    print("")

    print(f"=== STEP 5: Summary Table ===")
    header = f"| {'#':<2} | {'Title (first 50 chars)':<50} | {'Body?':<5} | {'Claim (first 50 chars)':<50} | {'Verdict':<15} | {'ClaimReview?':<12} |"
    print(header)
    print("-" * len(header))
    
    for i, a in enumerate(articles, 1):
        title = (a.get("title") or "")[:50].replace('\n', ' ')
        body_val = "YES" if a.get("body") else "NO"
        cr = a.get("claim_review", {})
        # claim is claim_review.claim or title
        claim = (cr.get("claim") or a.get("title") or "")[:50].replace('\n', ' ')
        verdict = a.get("verdict_normalized") or ""
        cr_val = "YES" if cr else "NO"
        
        row = f"| {i:<2} | {title:<50} | {body_val:<5} | {claim:<50} | {verdict:<15} | {cr_val:<12} |"
        print(row)
        
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(articles)} articles scraped")
    
    body_count = sum(1 for a in articles if a.get("body"))
    claim_count = sum(1 for a in articles if a.get("claim_review", {}).get("claim"))
    
    adapter_pass = all(bool(e.get("data", [{}])[0].get("claim")) for e in entries)
    normalizer_pass = len(normalized_docs) == len(entries)
    
    print(f"  Body extracted:     {body_count}/{NUM_TEST_ARTICLES} {'PASS' if body_count > 0 else 'FAIL'}")
    print(f"  ClaimReview found:  {claim_count}/{NUM_TEST_ARTICLES} {'PASS' if claim_count > 0 else 'WARN'}")
    print(f"  Adapter mapping:    {'PASS' if adapter_pass else 'FAIL'}")
    print(f"  Normalizer:         {'PASS' if normalizer_pass else 'FAIL'}")
    print(f"{'='*60}")
    
    if body_count == 0 or not adapter_pass:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
