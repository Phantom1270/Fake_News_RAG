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
from web.app import app
from rag.config.settings import get_settings
from rag.storage.sqlite_store import SQLiteStore
from rag.search.bm25.bm25_index import BM25Index
from rag.search.vector_store.chroma_store import ChromaVectorStore
from rag.agent.qa_engine import QAResult, SourceCitation


@pytest.fixture(scope="session")
def client():
    """Test client with application lifespan/startup executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_environment_configuration_example():
    """Requirement: .env.example exists and contains proper keys with no leaked secrets."""
    example_path = PROJECT_ROOT / ".env.example"
    assert example_path.exists(), ".env.example missing"
    text = example_path.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in text
    assert "GEMINI_API_KEY=" in text
    assert "DEFAULT_LLM_BACKEND=" in text
    # Verify no real credentials in .env.example
    assert "gsk_" not in text
    assert "AIza" not in text


def test_02_production_requirements_file():
    """Requirement: requirements.txt and requirements-prod.txt exist and are complete."""
    for req_file in ["requirements.txt", "requirements-prod.txt"]:
        req_path = PROJECT_ROOT / req_file
        assert req_path.exists(), f"{req_file} missing"
        text = req_path.read_text(encoding="utf-8")
        assert "fastapi" in text
        assert "uvicorn" in text
        assert "chromadb" in text
        assert "sentence-transformers" in text
        assert "rank-bm25" in text
        assert "groq" in text
        assert "torch --index-url" in text  # CPU instruction documented


def test_03_frontend_served(client):
    """Requirement: GET / returns 200 and serves HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Knowledge Base Q&A" in response.text
    assert "question-input" in response.text
    assert "ask-btn" in response.text


def test_04_health_endpoint(client):
    """Requirement: GET /api/health reports operational status and loaded indexes."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qa_engine_ready"] is True
    assert data["retrieval_engine_ready"] is True
    assert data["indexed_documents"] > 0


def test_05_favicon_endpoint(client):
    """Requirement: /favicon.ico returns 204 to prevent browser console errors."""
    response = client.get("/favicon.ico")
    assert response.status_code == 204


def test_06_sqlite_database_opens_and_has_data():
    """Requirement: rag.db opens correctly and has expected 2499 records."""
    settings = get_settings()
    db_path = os.path.join(settings.general.data_dir, "rag.db")
    assert os.path.exists(db_path), f"Database not found at {db_path}"
    store = SQLiteStore(db_path)
    count = store.count_documents()
    assert count == 2499, f"Expected 2499 documents, found {count}"


def test_07_bm25_index_loads():
    """Requirement: BM25 index loads correctly from disk."""
    settings = get_settings()
    bm25 = BM25Index()
    success = bm25.load(settings.bm25.index_path)
    assert success is True, "Failed to load BM25 index"
    assert len(bm25.chunk_ids) == 2499


def test_08_chroma_index_loads():
    """Requirement: Chroma vector store loads collection correctly."""
    settings = get_settings()
    store = ChromaVectorStore(settings)
    assert store.count() == 2499


def test_09_local_retrieval_returns_relevant_results():
    """Requirement: Retrieval returns relevant documents for known fact-check topic."""
    from rag.retrieval.retrieval_engine import RetrievalEngine
    engine = RetrievalEngine()
    results = engine.retrieve("Nepal flood videos")
    assert len(results) > 0
    top_title = results[0].metadata.get("document_title", "")
    assert "Nepal" in top_title or "flood" in top_title.lower()


def test_10_empty_question_is_rejected(client):
    """Requirement: Empty or whitespace question returns 400 Bad Request."""
    res = client.post("/api/ask", json={"question": "   "})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_11_out_of_domain_question_handled_cleanly(client):
    """Requirement: Out-of-domain question returns a safe response without hallucination."""
    res = client.post("/api/ask", json={"question": "What is the orbital resonance of Jupiter's moons?"})
    assert res.status_code == 200
    data = res.json()
    assert "could not find any relevant information" in data["answer"].lower()
    assert len(data["sources"]) == 0


def test_12_internal_exceptions_do_not_leak_traces(client):
    """Requirement: Unexpected errors return safe 500 without leaking server paths or secrets."""
    with patch("web.app.qa_engine.ask", side_effect=RuntimeError("Internal /secret/path/to/key Failure")):
        res = client.post("/api/ask", json={"question": "Are the Nepal flood videos real?"})
        assert res.status_code == 500
        assert "/secret/path" not in res.text
        assert "RuntimeError" not in res.text
        assert "An error occurred" in res.json()["detail"]


def test_13_mocked_qa_succeeds(client):
    """Requirement: /api/ask endpoint serializes answer and citations properly."""
    mock_result = QAResult(
        question="Were the Nepal flood videos real?",
        answer="Investigation found the footage was from older disasters.",
        sources=[
            SourceCitation(
                title="Nepal floods: Unrelated videos viral",
                url="https://www.altnews.in/nepal-floods",
                source="altnews.in",
                verdict="False",
                excerpt="Alt News verified that the footage originated elsewhere.",
                score=0.91
            )
        ],
        provider="groq"
    )

    with patch("web.app.qa_engine.ask", return_value=mock_result):
        res = client.post("/api/ask", json={"question": "Were the Nepal flood videos real?"})
        assert res.status_code == 200
        data = res.json()
        assert data["question"] == "Were the Nepal flood videos real?"
        assert "older disasters" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "altnews.in"
        assert data["sources"][0]["score"] == 0.91


def test_14_index_preservation_across_reloads():
    """Requirement: Loading indexes repeatedly does not corrupt or alter data."""
    settings = get_settings()
    store = SQLiteStore(os.path.join(settings.general.data_dir, "rag.db"))
    before = store.count_documents()
    assert before == 2499
    after = store.count_documents()
    assert after == before
