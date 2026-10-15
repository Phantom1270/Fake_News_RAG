"""
FastAPI application for RAG Question-Answering.
Simplified from legacy Fake News Detector.

Endpoints:
  GET  /              - Serves the frontend Q and A interface
  POST /api/ask       - Accepts a question, retrieves context, returns synthesized answer and sources
  GET  /api/health    - Health check endpoint
  GET  /favicon.ico   - Returns 204 No Content
"""

import os
import sqlite3
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag.config.settings import get_settings
from rag.retrieval.retrieval_engine import RetrievalEngine
from rag.agent.qa_engine import QAEngine, QAResult, SourceCitation

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Shared instances initialized at startup
retrieval_engine: Optional[RetrievalEngine] = None
qa_engine: Optional[QAEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_engine, qa_engine
    logger.info("Starting RAG QA service...")

    settings = get_settings()

    # 1. Verify existence of required pre-built database and indexes
    db_path = os.path.join(settings.general.data_dir, "rag.db")
    bm25_path = settings.bm25.index_path
    chroma_dir = settings.vector_store.persist_dir
    chroma_db = os.path.join(chroma_dir, "chroma.sqlite3")

    missing_assets = []
    if not os.path.exists(db_path):
        missing_assets.append(f"SQLite database ({db_path})")
    if not os.path.exists(bm25_path):
        missing_assets.append(f"BM25 index ({bm25_path})")
    if not os.path.exists(chroma_db):
        missing_assets.append(f"Chroma vector store ({chroma_db})")

    if missing_assets:
        details = "\n".join(f"  - {item}" for item in missing_assets)
        error_msg = f"Startup aborted: Required pre-built RAG assets are missing:\n{details}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 2. Initialize RetrievalEngine and QAEngine once
    retrieval_engine = RetrievalEngine(settings=settings)
    qa_engine = QAEngine(retrieval_engine=retrieval_engine)
    logger.info("RAG QAEngine initialized and ready.")
    yield


app = FastAPI(
    title="Fact-Check Knowledge Base Q&A",
    description="Minimal RAG Question-Answering Application",
    lifespan=lifespan
)


class AskRequest(BaseModel):
    question: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of passages to retrieve")


class SourceItem(BaseModel):
    title: str
    url: str
    source: str
    verdict: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    provider: str


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid 404 browser console noise."""
    return Response(status_code=204)


@app.get("/api/health")
async def health():
    """Lightweight health check."""
    doc_count = 0
    try:
        settings = get_settings()
        db_path = os.path.join(settings.general.data_dir, "rag.db")
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except Exception:
        pass

    return {
        "status": "ok",
        "qa_engine_ready": qa_engine is not None,
        "retrieval_engine_ready": retrieval_engine is not None,
        "indexed_documents": doc_count,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Accepts a question, retrieves relevant context from the knowledge base,
    synthesizes an answer using the LLM, and returns the answer with sources.
    """
    cleaned_question = req.question.strip() if req.question else ""
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not qa_engine:
        raise HTTPException(status_code=503, detail="QA engine is not initialized.")

    try:
        result: QAResult = qa_engine.ask(cleaned_question, top_k=req.top_k or 5)
        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                SourceItem(
                    title=s.title,
                    url=s.url,
                    source=s.source,
                    verdict=s.verdict,
                    excerpt=s.excerpt,
                    score=s.score,
                )
                for s in result.sources
            ],
            provider=result.provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /api/ask: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while answering your question. Please try again."
        )


# Serve frontend static assets and index
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "RAG Q&A API is operational. Static index.html not found."}
