"""
POST /api/v1/verify — Fact-check verification endpoint.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class VerifyRequest(BaseModel):
    text: str


class SourceInfo(BaseModel):
    title: str
    url: str
    type: str
    score: Optional[float] = None


class EvidenceInfo(BaseModel):
    database_matches: int
    web_results: int
    searched_web: bool
    db_confidence: str


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: str
    explanation: str
    sources: List[SourceInfo]
    evidence: EvidenceInfo
    processing_time_ms: int


@router.post("/verify", response_model=VerifyResponse)
async def verify_claim(request: VerifyRequest):
    """Verify a claim using local DB + web search + LLM."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long. Max 5000 chars.")

    start = time.time()

    try:
        from rag.bootstrap import create_pipeline
        pipeline = create_pipeline()
        generator = pipeline["generator"]

        result = generator.generate_factcheck(request.text)
        elapsed = int((time.time() - start) * 1000)

        if "claims" not in result:
            return VerifyResponse(
                claim=result["claim"],
                verdict=result["verdict"],
                confidence=result["confidence"],
                explanation=result["explanation"],
                sources=result["sources"],
                evidence=EvidenceInfo(**result["evidence"]),
                processing_time_ms=elapsed
            )

        first = result["claims"][0]
        return VerifyResponse(
            claim=first["claim"],
            verdict=first["verdict"],
            confidence=first["confidence"],
            explanation=first["explanation"],
            sources=first["sources"],
            evidence=EvidenceInfo(**first["evidence"]),
            processing_time_ms=elapsed
        )

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
