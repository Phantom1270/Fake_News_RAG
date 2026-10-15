from dataclasses import dataclass, field
from typing import List
from enum import Enum


class VerdictType(Enum):
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    UNVERIFIABLE = "unverifiable"
    PARTIALLY_TRUE = "partially_true"


@dataclass
class Evidence:
    source_title: str
    source_url: str
    source_name: str
    relevant_excerpt: str
    original_verdict: str
    similarity_score: float


@dataclass
class VerificationResult:
    claim: str
    verdict: VerdictType
    confidence: float
    explanation: str
    evidence: List[Evidence] = field(default_factory=list)
    matched_fact_checks: int = 0
    reasoning: str = ""
    llm_backend: str = ""  # Track which backend was used
