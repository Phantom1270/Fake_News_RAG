# ==============================================================================
# Production Dockerfile for Fake News Detector RAG Q&A
# Optimized for CPU AWS Hosting (App Runner, ECS Fargate, or EC2 Docker)
# ==============================================================================

FROM python:3.12-slim

# Avoid interactive prompts and python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt ./

# 1. Install CPU-only PyTorch first (avoids ~4GB CUDA drivers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 2. Pre-cache SentenceTransformer model during build for zero runtime startup delay
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code and pre-indexed databases
COPY rag/ ./rag/
COPY web/ ./web/
COPY .env.example ./

# Expose web server port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application with Uvicorn
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
