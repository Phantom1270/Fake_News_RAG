import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from web.app import app
from rag.agent.qa_engine import QAResult, SourceCitation


@pytest.fixture(scope="session")
def client():
    """Test client with startup lifespan executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_homepage_loads(client):
    """Test that GET / returns the Q&A homepage with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Knowledge Base Q&A" in response.text
    assert "question-input" in response.text
    assert "ask-btn" in response.text


def test_02_health_endpoint(client):
    """Test that GET /api/health returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qa_engine_ready"] is True
    assert data["retrieval_engine_ready"] is True
    assert data["indexed_documents"] > 0


def test_03_favicon_returns_no_content(client):
    """Test that /favicon.ico returns 204 to prevent 404 console errors."""
    response = client.get("/favicon.ico")
    assert response.status_code == 204


def test_04_removed_admin_endpoints_return_404(client):
    """Verify that all removed admin and upload endpoints return 404 Not Found."""
    admin_routes = [
        ("POST", "/api/admin/login"),
        ("POST", "/api/admin/upload"),
        ("GET", "/api/admin/stats"),
        ("GET", "/api/public-stats"),
        ("POST", "/api/verify"),
    ]
    for method, route in admin_routes:
        if method == "POST":
            res = client.post(route, json={})
        else:
            res = client.get(route)
        assert res.status_code == 404, f"Route {route} expected 404, got {res.status_code}"


def test_05_empty_question_is_rejected(client):
    """Verify that empty, whitespace-only, or missing questions are rejected with 400 Bad Request."""
    res1 = client.post("/api/ask", json={"question": ""})
    assert res1.status_code == 400
    assert res1.json()["detail"] == "Question cannot be empty."

    res2 = client.post("/api/ask", json={"question": "   "})
    assert res2.status_code == 400
    assert res2.json()["detail"] == "Question cannot be empty."

    res3 = client.post("/api/ask", json={})
    assert res3.status_code == 400
    assert res3.json()["detail"] == "Question cannot be empty."


def test_06_ask_endpoint_valid_question_with_sources(client):
    """
    Test that a valid question reaches the backend, retrieval runs,
    an answer is generated, and sources are returned.
    """
    payload = {
        "question": "Were the Nepal flood videos authentic?",
        "top_k": 3
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["question"] == "Were the Nepal flood videos authentic?"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 20
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    
    first_source = data["sources"][0]
    assert "title" in first_source
    assert "url" in first_source
    assert "source" in first_source
    assert "score" in first_source
    assert "verdict" in first_source


def test_07_ask_endpoint_out_of_domain_question(client):
    """
    Test question where no documents meet similarity threshold.
    Should gracefully return helpful message without hallucinating or erroring.
    """
    payload = {
        "question": "What is the quantum state of a black hole singularity in string theory?",
        "top_k": 3
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "could not find any relevant information" in data["answer"].lower()
    assert len(data["sources"]) == 0


def test_08_api_failure_handling(client):
    """
    Test that internal backend failures produce clean 500 error responses
    without crashing the server or exposing traceback leaks.
    """
    with patch("web.app.qa_engine.ask", side_effect=Exception("Simulated internal error")):
        response = client.post("/api/ask", json={"question": "Test question that triggers failure"})
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"].lower()
