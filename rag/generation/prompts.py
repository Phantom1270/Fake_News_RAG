"""
Prompt templates and builder for the Generation Engine.
"""

from typing import List, Dict
from rag.models.retrieval import RetrievalResult

SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer the user's question using ONLY the provided context.

Rules:
1. If the context contains the answer, provide it clearly with references.
2. If the context is insufficient or does not contain the answer, say:
   "I don't have enough information in the documentation to answer this
   fully." Do NOT make up information.
3. NEVER fabricate information not present in the context.
4. When referencing documentation, cite the source using [Source: Section Title].
5. If code examples are available in the context, include them in your answer.
6. If multiple approaches or options exist in the docs, mention all of them.

Format your answer:
- Direct answer first
- Supporting details and code examples if available
- Source references at the end
"""

NO_CONTEXT_PROMPT = """You are a technical documentation assistant.
The user asked a question but no relevant documentation was found.

Respond with: "I couldn't find relevant information in the documentation
for your question. Please check the official documentation or try
rephrasing your question."

Do NOT attempt to answer from your general knowledge.
"""

LOW_CONFIDENCE_PROMPT = """You are a technical documentation assistant.
The retrieved context may not fully answer the user's question.

Rules:
1. Answer what you can from the context.
2. Clearly state which parts of your answer come from the documentation
   and which parts you are uncertain about.
3. If the context is too vague or off-topic, say:
   "The available documentation doesn't directly address this question.
   Based on related information, here's what I found: ..."
4. Always cite your sources using [Source: Section Title].
"""


# ============================================================
# FACT-CHECK PROMPTS
# ============================================================

FACTCHECK_SYSTEM_PROMPT = """You are a fact-checker specializing in Indian news and claims. Your job is to evaluate claims based on evidence provided.

RULES:
1. Base your verdict ONLY on the evidence provided below
2. If evidence clearly supports or refutes the claim, give a definitive verdict
3. If evidence is insufficient or conflicting, say UNVERIFIABLE
4. Always cite specific sources from the evidence
5. Be objective and concise

RESPONSE FORMAT (follow this exactly):
VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIABLE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
EXPLANATION: [2-3 sentences explaining your reasoning based on the evidence]
SOURCES: [List the most relevant sources, one per line, with URLs if available]"""

FACTCHECK_LOW_CONFIDENCE_PROMPT = """You are a fact-checker specializing in Indian news and claims.

WARNING: The available evidence has LOW confidence — database matches may not be directly relevant, and web results may not be authoritative.

RULES:
1. Be cautious — prefer UNVERIFIABLE over a wrong call
2. Clearly state that evidence is limited
3. Still provide what you found, but flag uncertainty

RESPONSE FORMAT (follow this exactly):
VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIABLE]
CONFIDENCE: [LOW]
EXPLANATION: [Explain what limited evidence was found and why it's insufficient]
SOURCES: [List whatever sources were found, noting they may not be authoritative]"""

FACTCHECK_NO_CONTEXT_PROMPT = """You are a fact-checker specializing in Indian news and claims.

NO EVIDENCE was found in the fact-check database or web search for this claim.

RULES:
1. You MUST say UNVERIFIABLE — do not guess or use training data
2. Explain that no evidence was found
3. Suggest where the user might verify

RESPONSE FORMAT (follow this exactly):
VERDICT: [UNVERIFIABLE]
CONFIDENCE: [NONE]
EXPLANATION: [No fact-check evidence was found for this claim. We cannot verify or debunk it.]
SOURCES: [Suggest: PIB Fact Check (pib.gov.in), Alt News (altnews.in), BOOM Live (boomlive.in)]"""


def build_factcheck_context(
    db_results: list,
    web_results: list
) -> str:
    """Build context string from two-tier search results."""
    parts = []

    if db_results:
        parts.append("=== FACT-CHECK DATABASE MATCHES ===")
        for i, r in enumerate(db_results, 1):
            heading = " > ".join(r.heading_path) if r.heading_path else ""
            parts.append(
                f"\n[Match {i}] (relevance: {r.score:.2f})"
                f"\nSource: {heading}"
                f"\nURL: {r.url}"
                f"\nContent: {r.content}"
            )

    if web_results:
        parts.append("\n\n=== WEB SEARCH RESULTS ===")
        for i, r in enumerate(web_results, 1):
            parts.append(
                f"\n[Result {i}] ({r.source})"
                f"\nTitle: {r.title}"
                f"\nURL: {r.url}"
                f"\nSnippet: {r.snippet}"
            )

    if not parts:
        return "No evidence found."

    return "\n".join(parts)


class PromptBuilder:
    """Builds prompt messages and assesses retrieval confidence."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        self.max_context_tokens = settings.generation.max_context_tokens
        self.encoding_name = settings.chunking.encoding_name

    def build_messages(self, query: str, context: str,
                       has_context: bool = True,
                       confidence: str = "high") -> List[Dict[str, str]]:
        """
        Build the messages list for the LLM.
        """
        if confidence == "none":
            system = NO_CONTEXT_PROMPT
            user = f"Question: {query}"
        else:
            system = SYSTEM_PROMPT if confidence == "high" else LOW_CONFIDENCE_PROMPT
            user = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    def assess_confidence(self, results: List[RetrievalResult]) -> str:
        """
        Assess retrieval confidence based on result scores.
        """
        if not results:
            return "none"

        # The first result should be the highest scoring one
        best_result = results[0]
        
        # Cross-encoder reranker returns raw logits (typically -10 to +10 range).
        # A score above -3 means the chunk is a relevant match.
        if best_result.source == "reranked":
            if best_result.score < -3.0:
                return "low"
            return "high"
            
        # Standard cosine similarity or hybrid scores (0.0 – 1.0 range)
        if best_result.score < 0.3:
            return "low"
            
        if best_result.score < 0.5 and len(results) < 2:
            return "low"
            
        return "high"
