import logging
from typing import Optional

from rag.agent.models import VerificationResult, VerdictType
from rag.agent.retriever import VerificationRetriever
from rag.agent.llm_client import GroqClient, GeminiClient
from rag.agent.config import (
    DEFAULT_LLM_BACKEND, DEFAULT_GROQ_API_KEY, DEFAULT_GEMINI_API_KEY,
    GROQ_MODEL, GEMINI_MODEL, TOP_K
)
from rag.retrieval.retrieval_engine import RetrievalEngine

logger = logging.getLogger(__name__)


class ClaimVerifier:
    """Orchestrates the verification process."""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_backend: Optional[str] = None, user_api_key: Optional[str] = None):
        """
        Args:
            retrieval_engine: Existing RetrievalEngine instance
            llm_backend: "groq" or "gemini". Defaults to DEFAULT_LLM_BACKEND.
            user_api_key: User's own API key (BYOK). If None, uses default trial key.
        """
        self.retrieval_engine = retrieval_engine
        self.retriever = VerificationRetriever(retrieval_engine)

        self.backend_name = (llm_backend or DEFAULT_LLM_BACKEND).lower()

        if self.backend_name == "groq":
            key = user_api_key or DEFAULT_GROQ_API_KEY
            self.llm = GroqClient(api_key=key, model=GROQ_MODEL)
        elif self.backend_name == "gemini":
            key = user_api_key or DEFAULT_GEMINI_API_KEY
            self.llm = GeminiClient(api_key=key, model=GEMINI_MODEL)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend_name}. Use 'groq' or 'gemini'.")

        if not self.llm.is_available():
            raise ValueError(
                f"No API key available for {self.backend_name}. "
                f"Provide your own key or set {self.backend_name.upper()}_API_KEY environment variable."
            )

    def verify(self, claim: str, top_k: int = TOP_K) -> VerificationResult:
        """
        Verify a claim end-to-end.
        """
        logger.info(f"Verifying claim: {claim}")
        
        # 1. Retrieve evidence
        evidence_list = self.retriever.retrieve_evidence(claim, top_k=top_k)
        
        # 2. Check if we have evidence (min similarity threshold applies)
        if not evidence_list:
            logger.info("No relevant evidence found above threshold.")
            return VerificationResult(
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                explanation="No relevant fact-checks were found in the database to verify this claim.",
                reasoning="Retrieved chunks scored below the similarity threshold.",
                evidence=[],
                matched_fact_checks=0,
                llm_backend=self.backend_name
            )
            
        # 3. Format context and call LLM
        evidence_context = self.retriever.format_evidence_for_prompt(evidence_list)
        
        try:
            llm_result = self.llm.analyze_claim(claim, evidence_context)
            
            try:
                verdict_enum = VerdictType(llm_result["verdict"])
            except ValueError:
                verdict_enum = VerdictType.UNVERIFIABLE
                
            return VerificationResult(
                claim=claim,
                verdict=verdict_enum,
                confidence=llm_result.get("confidence", 0.0),
                explanation=llm_result.get("explanation", ""),
                reasoning=llm_result.get("reasoning", ""),
                evidence=evidence_list,
                matched_fact_checks=len(evidence_list),
                llm_backend=self.backend_name
            )
            
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return VerificationResult(
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                explanation=f"Analysis failed due to an error with the {self.backend_name} API.",
                reasoning=str(e),
                evidence=evidence_list,
                matched_fact_checks=len(evidence_list),
                llm_backend=self.backend_name
            )
