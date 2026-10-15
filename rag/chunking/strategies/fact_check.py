from typing import List
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.chunking.strategies.base import BaseChunkingStrategy

class FactCheckChunkingStrategy(BaseChunkingStrategy):
    """
    Fact-check chunking strategy.

    Each fact-check article becomes a SINGLE chunk with the following
    structure, ensuring claim and verdict are always embedded in the
    chunk text for semantic retrieval:

        CLAIM: <claim text>       ← from doc.title (adapter maps claim_review.claim → claim field → title role)
        VERDICT: <verdict>        ← from verdict content block
        <full article body>       ← from description/prose content blocks
    """

    def chunk(self, doc: NormalizedDocument) -> List[Chunk]:
        """
        Build one chunk per fact-check document.

        Reads claim from doc.title (where the FieldClassifier puts the 'claim'
        field — it matches TITLE_PATTERNS) and verdict from the 'verdict'
        content block. Prepends both to the body so they are always searchable
        via semantic similarity.
        """
        # ── Pull claim from doc.title ──────────────────────────────────────
        # The adapter maps: claim_review.claim → item["claim"]
        # FieldClassifier maps "claim" → "title" role → doc.title scalar
        claim = (doc.title or "").strip()

        # ── Pull verdict from the verdict content block ────────────────────
        # FieldClassifier maps "verdict" → "verdict" role → ContentBlock(block_type="verdict")
        verdict = ""
        for block in doc.content_blocks:
            if block.block_type == "verdict" and block.text:
                verdict = block.text.strip()
                break

        # ── Build the body from non-verdict content blocks ─────────────────
        body_parts = []
        for block in doc.content_blocks:
            if block.block_type == "verdict":
                continue  # already captured above; don't duplicate in body
            text = block.text.strip()
            if not text:
                continue
            if block.heading:
                body_parts.append(f"{block.heading}: {text}")
            else:
                body_parts.append(text)

        body_text = "\n".join(body_parts)

        # ── Assemble: CLAIM + VERDICT header + body ────────────────────────
        header_lines = []
        if claim:
            header_lines.append(f"CLAIM: {claim}")
        if verdict:
            header_lines.append(f"VERDICT: {verdict}")

        if header_lines:
            combined_text = "\n".join(header_lines) + "\n" + body_text
        else:
            combined_text = body_text

        if not combined_text.strip():
            return []

        # ── Build heading path ─────────────────────────────────────────────
        heading_path = [claim] if claim else (["Fact Check"])

        chunk = self._make_chunk(
            doc=doc,
            content=combined_text,
            content_type="fact_check",
            heading_path=heading_path,
            chunk_index=0,
            block_type="fact_check"
        )

        return [chunk]

    def _build_heading_path(self, doc: NormalizedDocument, block) -> List[str]:
        """Build breadcrumb: [doc_title, block_heading]"""
        path = []
        if doc.title:
            path.append(doc.title)
        if block.heading:
            path.append(block.heading)
        return path or ["Fact Check"]
