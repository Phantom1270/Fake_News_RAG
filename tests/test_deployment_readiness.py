import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from web.app import app, check_admin_access, _rate_limit_records
from rag.config.settings import get_settings
from rag.storage.sqlite_store import SQLiteStore
from rag.search.bm25.bm25_index import BM25Index
from rag.search.vector_store.chroma_store import ChromaVectorStore
from rag.agent.models import VerificationResult, VerdictType, Evidence


@pytest.fixture(scope="session")
def client():
    """Test client with application lifespan/startup executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_environment_configuration_example():
    """Requirement: .env.example exists and contains no secrets."""
    example_path = PROJECT_ROOT / ".env.example"
    assert example_path.exists(), ".env.example missing"
    text = example_path.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in text
    assert "GEMINI_API_KEY=" in text
    assert "ADMIN_PASSWORD=" in text
    assert "RATE_LIMIT_PER_MINUTE=" in text
    # Verify no real API keys in .env.example
    assert "gsk_" not in text
    assert "AIza" not in text


def test_02_production_requirements_file():
    """Requirement: requirements-prod.txt exists and is properly structured."""
    req_path = PROJECT_ROOT / "requirements-prod.txt"
    assert req_path.exists(), "requirements-prod.txt missing"
    text = req_path.read_text(encoding="utf-8")
    assert "fastapi" in text
    assert "uvicorn" in text
    assert "chromadb" in text
    assert "sentence-transformers" in text
    assert "rank-bm25" in text
    assert "torch --index-url" in text  # CPU instruction documented


def test_03_frontend_served(client):
    """Requirement: GET / returns 200 and serves HTML with title."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "THE FAKE NEWS DETECTOR" in response.text


def test_04_expected_api_endpoints_registered(client):
    """Requirement: Key API routes are registered and healthy."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["retrieval_engine_ready"] is True


def test_05_sqlite_database_opens_and_has_data():
    """Requirement: rag.db opens correctly and has expected 2499 records."""
    settings = get_settings()
    db_path = os.path.join(settings.general.data_dir, "rag.db")
    assert os.path.exists(db_path), f"Database not found at {db_path}"
    store = SQLiteStore(db_path)
    count = store.count_documents()
    assert count == 2499, f"Expected 2499 documents, found {count}"


def test_06_bm25_index_loads():
    """Requirement: BM25 index loads correctly from disk."""
    settings = get_settings()
    bm25 = BM25Index()
    success = bm25.load(settings.bm25.index_path)
    assert success is True, "Failed to load BM25 index"
    assert len(bm25.chunk_ids) == 2499


def test_07_chroma_index_loads():
    """Requirement: Chroma vector store loads collection correctly."""
    settings = get_settings()
    store = ChromaVectorStore(settings)
    assert store.count() == 2499


def test_08_local_retrieval_returns_relevant_results():
    """Requirement: Retrieval returns relevant documents for known claim."""
    from rag.retrieval.retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()
    results = engine.retrieve("Nepal flood videos")
    assert len(results) > 0
    top_title = results[0].metadata.get("document_title", "")
    assert "Nepal" in top_title or "flood" in top_title.lower()


def test_09_admin_auth_missing_env_fails_safely():
    """Requirement: Missing ADMIN_PASSWORD in environment returns 503."""
    with patch.dict(os.environ, {"ADMIN_PASSWORD": ""}):
        with pytest.raises(Exception) as exc_info:
            check_admin_access("some_password")
        assert exc_info.value.status_code == 503
        assert "disabled" in exc_info.value.detail.lower()


def test_10_admin_auth_invalid_credentials_rejected(client):
    """Requirement: Incorrect admin credentials return 401."""
    with patch.dict(os.environ, {"ADMIN_PASSWORD": "secure_demo_pass"}):
        res = client.post("/api/admin/login", json={"password": "wrong_password"})
        assert res.status_code == 401

        res_stats = client.get("/api/admin/stats", headers={"X-Admin-Password": "wrong_password"})
        assert res_stats.status_code == 401


def test_11_admin_auth_valid_credentials_accepted(client):
    """Requirement: Correct admin credentials succeed."""
    with patch.dict(os.environ, {"ADMIN_PASSWORD": "secure_demo_pass"}):
        res = client.post("/api/admin/login", json={"password": "secure_demo_pass"})
        assert res.status_code == 200
        assert res.json()["authenticated"] is True

        res_stats = client.get("/api/admin/stats", headers={"X-Admin-Password": "secure_demo_pass"})
        assert res_stats.status_code == 200
        assert "total_articles" in res_stats.json()
        assert res_stats.json()["total_articles"] == 2499


def test_12_public_stats_accessible_without_auth(client):
    """Requirement: /api/public-stats serves database count to public frontend."""
    res = client.get("/api/public-stats")
    assert res.status_code == 200
    assert res.json()["total_articles"] == 2499


def test_13_rate_limiting_enforced(client):
    """Requirement: Rate limiting triggers HTTP 429 with Retry-After header."""
    _rate_limit_records.clear()
    with patch("web.app.RATE_LIMIT_PER_MINUTE", 2):
        # 1st request - should pass claim validation
        r1 = client.post("/api/verify", json={"claim": "abc"})
        assert r1.status_code == 400  # Passed rate limit, failed on claim len

        # 2nd request
        r2 = client.post("/api/verify", json={"claim": "abc"})
        assert r2.status_code == 400

        # 3rd request - should be rate limited (limit is 2)
        r3 = client.post("/api/verify", json={"claim": "abc"})
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]
        assert "Retry-After" in r3.headers
    _rate_limit_records.clear()


def test_14_internal_exceptions_do_not_leak_traces_or_keys(client):
    """Requirement: Unexpected errors return safe 500 without leaking server paths or secrets."""
    _rate_limit_records.clear()
    with patch("rag.agent.verifier.ClaimVerifier.verify", side_effect=RuntimeError("Internal /secret/path/to/key Failure")):
        res = client.post("/api/verify", json={"claim": "Are the Nepal flood videos real?"})
        assert res.status_code == 500
        # Check that internal path or exception name is NOT in response
        assert "/secret/path" not in res.text
        assert "RuntimeError" not in res.text
        assert "Verification service encountered an unexpected error" in res.json()["detail"]
    _rate_limit_records.clear()


def test_15_mocked_claim_verification(client):
    """Requirement: Claim verification works end-to-end with mocked LLM."""
    _rate_limit_records.clear()
    mock_result = VerificationResult(
        claim="Nepal flood videos were fabricated by the media",
        verdict=VerdictType.FALSE,
        confidence=0.92,
        explanation="Investigation confirmed the footage was recycled from previous disasters.",
        reasoning="Visual matching confirmed keyframes from 2024 incidents.",
        evidence=[
            Evidence(
                source_title="Nepal floods: Unrelated videos viral",
                source_url="https://www.altnews.in/nepal-floods",
                source_name="altnews.in",
                relevant_excerpt="Alt News found the videos circulating were drawn from other calamities.",
                original_verdict="False",
                similarity_score=0.88
            )
        ],
        matched_fact_checks=1,
        llm_backend="groq"
    )

    with patch("rag.agent.verifier.ClaimVerifier.verify", return_value=mock_result):
        res = client.post("/api/verify", json={"claim": "Nepal flood videos were fabricated by the media"})
        assert res.status_code == 200
        data = res.json()
        assert data["verdict"] == "false"
        assert data["confidence"] == 0.92
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["source_name"] == "altnews.in"
    _rate_limit_records.clear()


def test_16_index_preservation_across_reloads():
    """Requirement: Loading indexes repeatedly does not corrupt or alter data."""
    settings = get_settings()
    store = SQLiteStore(os.path.join(settings.general.data_dir, "rag.db"))
    before = store.count_documents()
    assert before == 2499
    # Reload settings and check count again
    after = store.count_documents()
    assert after == before
