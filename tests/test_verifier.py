import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.agent.verifier import ClaimVerifier
from rag.retrieval.retrieval_engine import RetrievalEngine

@pytest.fixture(scope="module")
def retrieval_engine():
    return RetrievalEngine()

test_cases = [
    {
        "claim": "India and Japan signed a contract allowing 50 million Hindus to settle in Japan",
        "expected_verdict": "false",
        "min_confidence": 0.7,
    },
    {
        "claim": "A Hindu family was killed by jihadists in Bangladesh",
        "expected_verdict": "false",
        "min_confidence": 0.7,
    },
    {
        "claim": "PM Modi disrespected the national anthem by walking during its playing",
        "expected_verdict": "false",
        "min_confidence": 0.7,
    },
    {
        "claim": "The sky is blue",
        "expected_verdict": "unverifiable",
        "min_confidence": 0.0,
    },
    {
        "claim": "Sonia Gandhi was photographed as a go-go dancer in Australia in 1967",
        "expected_verdict": "false",
        "min_confidence": 0.7,
    }
]

import time

@pytest.mark.parametrize("backend", ["groq", "gemini"])
def test_claim_verification(retrieval_engine, backend):
    try:
        verifier = ClaimVerifier(retrieval_engine=retrieval_engine, llm_backend=backend)
    except ValueError as e:
        pytest.skip(f"Skipping {backend} tests: {e}")

    print(f"\n=== Testing with {backend.upper()} ===")
    
    for tc in test_cases:
        time.sleep(3)  # Avoid rate limits
        result = verifier.verify(tc["claim"])
        
        passed = result.verdict.value == tc["expected_verdict"]
        print(f"  {'PASS' if passed else 'FAIL'}: {tc['claim'][:60]}...")
        print(f"        Expected: {tc['expected_verdict']}, Got: {result.verdict.value}, Confidence: {result.confidence}")
        
        assert result.verdict.value == tc["expected_verdict"], f"Failed on claim: {tc['claim']}"
        assert result.confidence >= tc["min_confidence"], f"Low confidence on claim: {tc['claim']}"
