"""
Two-tier search orchestrator.
Tier 1: Local fact-check database (ChromaDB + BM25)
Tier 2: DuckDuckGo web search (live evidence)
"""
from dataclasses import dataclass, field
from typing import List
import logging

from rag.retrieval.retrieval_engine import (
    RetrievalEngine,
    QueryRequest,
    RetrievalResult
)
from rag.search.web_search import WebSearcher, WebSearchResult

logger = logging.getLogger(__name__)


@dataclass
class TieredSearchResult:
    """Combined results from both tiers."""
    db_results: List[RetrievalResult] = field(default_factory=list)
    web_results: List[WebSearchResult] = field(default_factory=list)
    searched_web: bool = False
    db_confidence: str = "none"  # "none", "low", "high"


class TieredSearcher:
    """
    Searches local DB first. If no strong match,
    supplements with DuckDuckGo web search.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        db_confidence_threshold: float = 0.55,
        web_search_enabled: bool = True
    ):
        self.retriever = retrieval_engine
        self.web_searcher = WebSearcher()
        self.db_confidence_threshold = db_confidence_threshold
        self.web_search_enabled = web_search_enabled

    def search(self, claim: str, top_k: int = 3) -> TieredSearchResult:
        """
        Two-tier search:
        1. Search local fact-check DB
        2. Assess confidence
        3. If low/none, also search web
        """
        result = TieredSearchResult()

        # Tier 1: Local DB
        request = QueryRequest(
            question=claim,
            top_k=top_k,
            filter_content_type="fact_check"
        )
        result.db_results = self.retriever.retrieve(request=request)

        # Assess confidence
        if not result.db_results:
            result.db_confidence = "none"
        elif result.db_results[0].score >= self.db_confidence_threshold:
            result.db_confidence = "high"
        else:
            result.db_confidence = "low"

        logger.info(
            f"DB: {len(result.db_results)} results, "
            f"confidence: {result.db_confidence}"
        )

        # Tier 2: Web search (only if DB confidence is not high)
        if self.web_search_enabled and result.db_confidence != "high":
            logger.info("Searching web...")
            result.web_results = self.web_searcher.search(claim)
            result.searched_web = True
            logger.info(f"Web: {len(result.web_results)} results")
        else:
            logger.info("Strong DB match, skipping web search")

        return result
