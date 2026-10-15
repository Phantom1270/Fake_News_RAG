"""
DuckDuckGo web search for live fact-check evidence.
No API key required. No rate limits for reasonable use.
"""
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    snippet: str
    url: str
    source: str  # "web_search" or "trusted_source"


class WebSearcher:
    """
    Searches DuckDuckGo for fact-check evidence.
    Two search strategies:
    1. General fact-check search
    2. Trusted Indian source search (PIB, Alt News, etc.)
    """

    TRUSTED_SOURCES = [
        "pib.gov.in",
        "altnews.in",
        "boomlive.in",
        "factly.in",
        "vishvasnews.com",
        "thequint.com",
        "factcheck.afp.com",
        "reuters.com",
    ]

    def search(
        self,
        claim: str,
        max_general: int = 3,
        max_trusted: int = 3
    ) -> List[WebSearchResult]:
        """
        Search for evidence about a claim.
        Returns combined results from general + trusted searches.
        """
        results = []

        # Search 1: General fact-check
        results.extend(
            self._search_ddg(
                query=f"{claim} fact check India",
                max_results=max_general,
                source_type="web_search"
            )
        )

        # Search 2: Trusted sources
        site_query = " OR ".join(
            [f"site:{s}" for s in self.TRUSTED_SOURCES[:4]]
        )
        results.extend(
            self._search_ddg(
                query=f"{claim} {site_query}",
                max_results=max_trusted,
                source_type="trusted_source"
            )
        )

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)

        return unique

    def _search_ddg(
        self,
        query: str,
        max_results: int,
        source_type: str
    ) -> List[WebSearchResult]:
        """Execute a single DuckDuckGo search."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    region="in-en",
                    max_results=max_results
                ):
                    results.append(WebSearchResult(
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        url=r.get("href", ""),
                        source=source_type
                    ))
            return results

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []
