import re
import logging
from typing import List, Dict
from urllib.parse import urlparse

from rag.agent.models import Evidence
from rag.agent.config import TOP_K, MAX_EVIDENCE_ARTICLES, MAX_EXCERPTS_PER_ARTICLE, MIN_SIMILARITY_THRESHOLD
from rag.retrieval.retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)

# Lines to strip from chunk content that are injected by the ingestion pipeline.
# These are line-by-line prefixes, not body text.
_HEADER_LINE_PATTERN = re.compile(
    r'^(##?\s.*|CLAIM:.*|VERDICT:.*|Category:.*|Date:.*|Description:\s*)$',
    re.MULTILINE
)


def _extract_body_text(content: str) -> str:
    """
    Strip injected metadata header lines from chunk content.
    Removes lines starting with CLAIM:, VERDICT:, Category:, Date:, and markdown headings.
    Keeps only body text.
    """
    lines = content.splitlines()
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if _HEADER_LINE_PATTERN.match(stripped):
            continue
        body_lines.append(line)
    result = '\n'.join(body_lines).strip()
    # Strip inline "Description: " prefix if the chunk text starts with it
    if result.lower().startswith("description:"):
        result = result[len("description:"):].lstrip()
    return result


def _source_name_from_url(url: str) -> str:
    """Derive a human-readable source name from the article URL."""
    if not url:
        return "Unknown Source"
    try:
        hostname = urlparse(url).netloc  # e.g. "www.altnews.in"
        # Remove www. prefix
        hostname = re.sub(r'^www\.', '', hostname)
        return hostname
    except Exception:
        return "Unknown Source"


class VerificationRetriever:
    """Retrieves and prepares evidence for the LLM."""

    def __init__(self, retrieval_engine: RetrievalEngine):
        self.retrieval_engine = retrieval_engine

    def retrieve_evidence(self, claim: str, top_k: int = TOP_K) -> List[Evidence]:
        """
        Retrieve chunks and group them by source article.
        Only groups/articles whose BEST chunk passes MIN_SIMILARITY_THRESHOLD are kept.
        """
        results = self.retrieval_engine.retrieve(query=claim)
        if not results:
            return []

        # Check if the best result passes the minimum threshold
        best_score = results[0].score
        if best_score < MIN_SIMILARITY_THRESHOLD:
            logger.info(
                f"Top result score ({best_score:.3f}) is below threshold "
                f"({MIN_SIMILARITY_THRESHOLD}). Returning empty evidence."
            )
            return []

        # Group by document URL (or title if URL is missing)
        grouped_results: Dict[str, List] = {}
        for r in results[:top_k]:
            key = (
                r.url
                or r.metadata.get("document_url", "")
                or r.metadata.get("document_title", "Unknown Source")
            )
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(r)

        evidence_list = []
        for key, chunks in grouped_results.items():
            if len(evidence_list) >= MAX_EVIDENCE_ARTICLES:
                break

            # Sort chunks in this document by score descending
            chunks.sort(key=lambda x: x.score, reverse=True)
            best_chunk = chunks[0]

            # ── Per-article threshold gate ─────────────────────────────────
            # Skip this article entirely if its best chunk is below threshold
            if best_chunk.score < MIN_SIMILARITY_THRESHOLD:
                logger.info(
                    f"Skipping article '{key[:60]}' — best score "
                    f"{best_chunk.score:.3f} < {MIN_SIMILARITY_THRESHOLD}"
                )
                continue
            # ──────────────────────────────────────────────────────────────

            # Derive source URL and name
            source_url = (
                best_chunk.url
                or best_chunk.metadata.get("document_url", "")
                or best_chunk.metadata.get("url", "")
            )
            source_name = _source_name_from_url(source_url)

            # Combine up to MAX_EXCERPTS_PER_ARTICLE, stripping metadata headers
            excerpts = []
            for c in chunks[:MAX_EXCERPTS_PER_ARTICLE]:
                body = _extract_body_text(c.content)
                if not body:
                    # Fallback: use raw content without newlines if cleaning leaves nothing
                    body = c.content.replace('\n', ' ').strip()
                # Truncate to ~600 chars
                if len(body) > 600:
                    body = body[:597] + "..."
                excerpts.append(body)

            combined_excerpt = " ... ".join(excerpts)

            evidence = Evidence(
                source_title=best_chunk.metadata.get("document_title", "Unknown Title"),
                source_url=source_url,
                source_name=source_name,
                relevant_excerpt=combined_excerpt,
                original_verdict=best_chunk.metadata.get("verdict", "Unknown"),
                similarity_score=best_chunk.score
            )
            evidence_list.append(evidence)

        # Sort evidence by similarity score descending
        evidence_list.sort(key=lambda x: x.similarity_score, reverse=True)
        return evidence_list

    def format_evidence_for_prompt(self, evidence_list: List[Evidence]) -> str:
        """Format the evidence into a string for the LLM prompt."""
        if not evidence_list:
            return "No matching fact-checks found."

        formatted = []
        for i, ev in enumerate(evidence_list):
            formatted.append(f"--- Article {i+1} ---")
            formatted.append(f"Title: {ev.source_title}")
            formatted.append(f"Source: {ev.source_name}")
            formatted.append(f"Original Fact-Check Verdict: {ev.original_verdict}")
            formatted.append(f"Excerpt: {ev.relevant_excerpt}")
            formatted.append("")

        return "\n".join(formatted)
