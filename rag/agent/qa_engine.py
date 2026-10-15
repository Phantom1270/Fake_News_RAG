"""
Minimal RAG Question-Answering Engine.
Pipeline: Question -> RetrievalEngine -> Context Formatting -> LLM Answer Synthesis -> Result with Sources
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure .env is loaded
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")
load_dotenv()

from rag.agent.retriever import VerificationRetriever
from rag.agent.llm_client import GroqClient, GeminiClient
from rag.agent.config import (
    DEFAULT_LLM_BACKEND, GROQ_MODEL, GEMINI_MODEL, TOP_K
)
from rag.retrieval.retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    title: str
    url: str
    source: str
    verdict: str
    excerpt: str
    score: float


class QAResult(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    provider: str


class QAEngine:
    """Orchestrates RAG Question Answering."""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_backend: Optional[str] = None):
        self.retrieval_engine = retrieval_engine
        self.retriever = VerificationRetriever(retrieval_engine)
        self.backend_name = (llm_backend or DEFAULT_LLM_BACKEND).lower()

        # Load API keys dynamically
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

        # Primary client
        self.primary_llm = None
        self.fallback_llm = None

        if self.backend_name == "groq":
            if groq_key:
                self.primary_llm = GroqClient(api_key=groq_key, model=GROQ_MODEL)
            if gemini_key:
                self.fallback_llm = GeminiClient(api_key=gemini_key, model=GEMINI_MODEL)
        else:
            if gemini_key:
                self.primary_llm = GeminiClient(api_key=gemini_key, model=GEMINI_MODEL)
            if groq_key:
                self.fallback_llm = GroqClient(api_key=groq_key, model=GROQ_MODEL)

    def ask(self, question: str, top_k: int = TOP_K) -> QAResult:
        question_clean = question.strip()
        if not question_clean:
            raise ValueError("Question cannot be empty.")

        logger.info(f"Processing question: {question_clean}")

        # 1. Retrieve relevant evidence chunks from knowledge base
        evidence_list = self.retriever.retrieve_evidence(question_clean, top_k=top_k)

        # Build citations list
        sources: List[SourceCitation] = [
            SourceCitation(
                title=e.source_title,
                url=e.source_url,
                source=e.source_name,
                verdict=e.original_verdict,
                excerpt=e.relevant_excerpt,
                score=round(e.similarity_score, 3)
            )
            for e in evidence_list
        ]

        # If no relevant documents found in knowledge base
        if not evidence_list:
            logger.info("No matching knowledge base documents found above similarity threshold.")
            return QAResult(
                question=question_clean,
                answer="I could not find any relevant information or fact-checks in the knowledge base to answer this question.",
                sources=[],
                provider=self.backend_name
            )

        # 2. Format context for LLM
        context = self.retriever.format_evidence_for_prompt(evidence_list)

        # 3. Generate answer via Primary LLM (with Fallback LLM)
        answer = None
        used_provider = self.backend_name

        # Try primary LLM
        if self.primary_llm and self.primary_llm.is_available():
            try:
                answer = self.primary_llm.generate_answer(question_clean, context)
            except Exception as e:
                logger.warning(f"Primary LLM ({self.backend_name}) failed: {e}. Attempting fallback...")

        # If primary failed or unavailable, try fallback
        if not answer and self.fallback_llm and self.fallback_llm.is_available():
            try:
                fallback_name = "gemini" if self.backend_name == "groq" else "groq"
                answer = self.fallback_llm.generate_answer(question_clean, context)
                used_provider = f"{fallback_name} (fallback)"
            except Exception as e:
                logger.error(f"Fallback LLM failed: {e}")

        # If both LLMs failed or no keys configured
        if not answer:
            answer = (
                "An error occurred while contacting the LLM generation service. "
                "However, relevant fact-checks were successfully retrieved from the knowledge base (see sources below)."
            )
            used_provider = "none (retrieval only)"

        return QAResult(
            question=question_clean,
            answer=answer,
            sources=sources,
            provider=used_provider
        )
