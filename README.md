# 🛡️ Fake News Detector & Fact-Check Knowledge Base RAG

[![Live Deployment](https://img.shields.io/badge/AWS%20Elastic%20Beanstalk-Live-brightgreen?style=for-the-badge&logo=amazon-aws)](http://fakenews-rag-demo.ap-south-1.elasticbeanstalk.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)](https://hub.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![AWS Free Tier](https://img.shields.io/badge/AWS%20Free%20Tier-Optimized%20($0)-orange?style=for-the-badge)](https://aws.amazon.com/free/)

An enterprise-ready **Retrieval-Augmented Generation (RAG)** system designed to detect disinformation and debunk viral rumors using a verified knowledge base of **2,499+ structured fact-checks** (AltNews, BOOM Live, PIB Fact Check, and independent fact-checkers).

---

## 🌐 Live Hosted Application

| Access Method | URL | Description | Status |
| :--- | :--- | :--- | :--- |
| **Global HTTPS (Mobile & Desktop)** | **[https://d2nll19ill5bq3.cloudfront.net/](https://d2nll19ill5bq3.cloudfront.net/)** | 🔒 **Universal SSL Link** (Opens on all iPhones, Androids, & PCs) | 🟢 Operational |
| **Direct Elastic Beanstalk** | **[http://fakenews-rag-demo.ap-south-1.elasticbeanstalk.com/](http://fakenews-rag-demo.ap-south-1.elasticbeanstalk.com/)** | Direct HTTP origin endpoint | 🟢 Operational |
| **System Health Probe** | **[https://d2nll19ill5bq3.cloudfront.net/api/health](https://d2nll19ill5bq3.cloudfront.net/api/health)** | Live retriever & database verification | 🟢 `{"status":"ok","indexed_documents":2499}` |
| **Interactive API Docs (Swagger)** | **[https://d2nll19ill5bq3.cloudfront.net/docs](https://d2nll19ill5bq3.cloudfront.net/docs)** | OpenAPI / Swagger interactive documentation | 🟢 Available |

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [Architecture](#-architecture)
3. [AWS Free-Tier Engineering ($0 Spend)](#-aws-free-tier-engineering-0-spend)
4. [Running Locally](#-running-locally)
   - [Method A: Docker (Recommended)](#method-a-docker-recommended)
   - [Method B: Native Python Virtual Environment](#method-b-native-python-virtual-environment)
5. [Automated Testing](#-automated-testing)
6. [API Documentation](#-api-documentation)
7. [AWS Cloud CI/CD Pipeline](#-aws-cloud-cicd-pipeline)
8. [AWS CLI Operations & Teardown Playbook](#-aws-cli-operations--teardown-playbook)
9. [Project Directory Layout](#-project-directory-layout)

---

## ✨ Key Features

- **Hybrid Retrieval Architecture**:
  - **BM25 Lexical Search**: Captures exact keywords, names, numbers, and viral hashtags.
  - **ChromaDB Dense Vector Search**: Semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dimensional).
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings with score normalization for high-precision recall.
- **Grounded LLM Answer Synthesis**:
  - Strict grounding prompt ensures the model **never hallucinates** facts. If no supporting fact-check exists, it explicitly states so.
  - Supported inference providers: **Groq** (`llama-3.3-70b-versatile` with ultra-low latency) and **Google Gemini** (`gemini-1.5-flash`).
  - Returns verified verdicts (`TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIED`), source article links, and similarity scores.
- **Self-Contained & Lightweight**:
  - Pre-built SQLite and Chroma databases embedded in the repository (~35 MB).
  - Pre-cached SentenceTransformer weights inside the Docker image—zero download lag during boot.
- **Sleek Modern Web UI**:
  - Glassmorphic dark UI with responsive design.
  - Interactive claim tester, preset viral rumor prompts, and collapsible citation cards.

---

## 🏗️ Architecture

```text
[ Web Browser / API Client ]
             │ HTTP (Port 80)
             ▼
[ AWS Elastic Beanstalk (Single Instance Docker on AL2023) ]
  ├── Nginx Reverse Proxy (Port 80 -> 8000)
  └── Docker Container (Port 8000)
        ├── FastAPI Web Server (Uvicorn)
        ├── 2 GB Virtual Swap (prevents OOM on t3.micro)
        ├── Hybrid Retrieval Engine
        │     ├── SQLite Metadata Store (rag/data/rag.db - 2,499 documents)
        │     ├── ChromaDB Vector Store (rag/data/indexes/chroma/)
        │     └── BM25 Lexical Index (rag/data/indexes/bm25.pkl)
        └── Cloud LLM Gateway
              ├── Primary: Groq API (Llama-3.3-70b-versatile)
              └── Secondary: Google Gemini API (gemini-1.5-flash)
```

---

## 💰 AWS Free-Tier Engineering ($0 Spend)

The entire deployment is engineered to run strictly within the **AWS Free Tier** limits:

| Component | AWS Resource | Free Tier Allowance | How We Stay Free |
| :--- | :--- | :--- | :--- |
| **Compute** | Amazon EC2 (`t3.micro`) | 750 hours/month | Deployed as a single `t3.micro` instance. |
| **Load Balancing** | **None** (`SingleInstance`) | $0 | Configured `EnvironmentType=SingleInstance` to completely avoid Elastic Load Balancer (ALB/NLB) hourly charges (~$18-$22/month saved). |
| **Storage** | Amazon EBS (`gp3`) | 30 GB/month | Root volume sized at 20 GB. |
| **Container Registry** | Amazon ECR | 500 MB/month | Attached an ECR Lifecycle Policy to **retain strictly 1 image** tag (`latest`), preventing storage creep. |
| **CI/CD** | AWS CodePipeline | 1 active pipeline | 1 free pipeline per month in AWS Free Tier. |
| **Memory Optimization** | Linux Swapfile | N/A | Added a 2 GB swapfile via `.ebextensions/01_swap.config` so CPU PyTorch runs smoothly on a 1 GB RAM instance. |

---

## 💻 Running Locally

### Prerequisites
- Python 3.10 or 3.11 installed
- Git
- Docker Desktop (optional, for containerized run)
- A Groq API key ([Get one free at console.groq.com](https://console.groq.com/)) or Google Gemini API key

---

### Method A: Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Phantom1270/Fake_News_RAG.git
   cd Fake_News_RAG
   ```

2. **Create your `.env` file:**
   ```bash
   cp .env.example .env
   ```
   Add your API keys to `.env`:
   ```ini
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   DEFAULT_LLM_BACKEND=groq
   PYTHONPATH=.
   ```

3. **Build the Docker container:**
   ```bash
   docker build -t fakenews-rag:latest .
   ```

4. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --env-file .env --name fakenews-local fakenews-rag:latest
   ```

5. **Open in browser:**
   Navigate to **`http://localhost:8000`**.

---

### Method B: Native Python Virtual Environment

1. **Clone and setup virtual environment:**
   ```bash
   git clone https://github.com/Phantom1270/Fake_News_RAG.git
   cd Fake_News_RAG

   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install CPU-only PyTorch first (avoids downloading ~4 GB CUDA drivers):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install application dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your GROQ_API_KEY
   ```

5. **Start the local server:**
   ```bash
   uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   Visit **`http://localhost:8000`** in your browser.

---

## 🧪 Automated Testing

The repository contains an automated test suite verifying both database integrity and API endpoints:

```bash
# Run all unit and integration tests
pytest tests/ -v
```

**Results (22/22 Passing):**
- `tests/test_deployment_readiness.py`: 14 checks (SQLite schema, Chroma index, BM25 manifest, CPU torch compatibility, environment settings, Dockerfile syntax, swap configuration).
- `tests/test_qa_app.py`: 8 checks (`/api/health`, `/api/ask` query routing, validation logic, empty payload handling, favicon 204 handler).

---

## 📡 API Documentation

### 1. Health Probe
- **Endpoint**: `GET /api/health`
- **Description**: Verifies database readiness, retriever status, and total indexed fact-checks.
- **Example Response**:
  ```json
  {
    "status": "ok",
    "qa_engine_ready": true,
    "retrieval_engine_ready": true,
    "indexed_documents": 2499
  }
  ```

### 2. Fact-Check Claim Verification
- **Endpoint**: `POST /api/ask`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "question": "Did the RBI announce 2000 rupee notes demonetisation?",
    "top_k": 5
  }
  ```
- **Example Response**:
  ```json
  {
    "question": "Did the RBI announce 2000 rupee notes demonetisation?",
    "answer": "Based on the verified fact-check database, there is no record of the RBI banning ₹500 or ₹2000 notes in the manner claimed by viral messages...",
    "sources": [
      {
        "title": "RBI has not asked banks to stop disbursing ₹500 notes; viral claim is false",
        "url": "https://www.altnews.in/fact-check-did-rbi-ban-500-rupee-note-again-false-claim/",
        "source": "altnews.in",
        "verdict": "FALSE",
        "excerpt": "CLAIM: RBI has not asked banks to stop disbursing ₹500 notes...",
        "score": 0.517
      }
    ],
    "provider": "groq"
  }
  ```

---

## 🚀 AWS Cloud CI/CD Pipeline

The project features a continuous deployment pipeline powered by **AWS CodePipeline**:

```text
[ GitHub: Phantom1270/Fake_News_RAG (main) ]
                      │ (git push)
                      ▼
[ AWS CodeStar Connection / Webhook ]
                      │
                      ▼
[ AWS CodePipeline: fakenews-pipeline ]
  ├── Stage 1: Source (pulls repository snapshot)
  └── Stage 2: Deploy (packages Dockerrun.aws.json & .ebextensions)
                      │
                      ▼
[ AWS Elastic Beanstalk: fakenews-env ]
  └── Zero-downtime rolling update pulls container from Amazon ECR
```

Whenever you push commits to `main`, CodePipeline automatically redeploys the latest code to your live Elastic Beanstalk environment.

---

## 🛠️ AWS CLI Operations & Teardown Playbook

All cloud resources can be inspected, monitored, or torn down directly from the **AWS CLI**:

### 1. Check Live Environment Status
```bash
aws elasticbeanstalk describe-environments --environment-names fakenews-env --region ap-south-1 --query "Environments[0].{Status:Status,Health:Health,CNAME:CNAME}"
```

### 2. View Live Server Logs
```bash
aws elasticbeanstalk request-environment-info --environment-name fakenews-env --info-type tail --region ap-south-1
aws elasticbeanstalk retrieve-environment-info --environment-name fakenews-env --info-type tail --region ap-south-1 --query "EnvironmentInfo[0].Message" --output text
```

### 3. Check CI/CD Pipeline Status
```bash
aws codepipeline get-pipeline-state --name fakenews-pipeline --region ap-south-1 --query "stageStates[].{Stage:stageName,Status:latestExecution.status}"
```

### 4. Complete Resource Teardown (Post-Demo Clean Up)
When your demonstration or evaluation is complete, run these commands to safely decommission the cloud resources:

```bash
# 1. Terminate the Elastic Beanstalk Environment (deletes the EC2 instance & EBS volume)
aws elasticbeanstalk terminate-environment --environment-name fakenews-env --region ap-south-1

# 2. Delete the Elastic Beanstalk Application
aws elasticbeanstalk delete-application --application-name fakenews-detector --terminate-env-by-force --region ap-south-1

# 3. Delete the CodePipeline
aws codepipeline delete-pipeline --name fakenews-pipeline --region ap-south-1

# 4. Delete the Amazon ECR Repository and container image
aws ecr delete-repository --repository-name fakenews-rag --force --region ap-south-1
```

---

## 📁 Project Directory Layout

```text
Fake_News_Detector/
├── .ebextensions/
│   └── 01_swap.config              # Creates 2GB swap space for t3.micro
├── Dockerrun.aws.json              # Elastic Beanstalk Docker deployment descriptor
├── Dockerfile                      # Production container spec with CPU PyTorch
├── buildspec.yml                   # AWS CodeBuild / pipeline specification
├── requirements.txt                # Production Python dependencies
├── requirements-prod.txt           # Minimal deployment requirements
├── .env.example                    # Template for API keys
├── DEPLOYMENT.md                   # Detailed manual deployment guide
├── AWS_CLEANUP.md                  # Comprehensive AWS billing protection guide
├── rag/
│   ├── agent/                      # LLM generation & synthesis agent
│   ├── config/                     # Settings and YAML configuration
│   ├── data/
│   │   ├── rag.db                  # SQLite database (2,499 fact checks)
│   │   └── indexes/
│   │       ├── bm25.pkl            # BM25 sparse keyword index
│   │       ├── manifest.json       # Index metadata & checksums
│   │       └── chroma/             # ChromaDB vector store (embeddings)
│   ├── retrieval/                  # Dense, sparse, and hybrid retrieval logic
│   └── utils/                      # Text preprocessing & scoring
├── tests/
│   ├── test_deployment_readiness.py # 14 deployment validation checks
│   └── test_qa_app.py              # 8 API and QA logic tests
└── web/
    ├── app.py                      # FastAPI application
    └── static/
        └── index.html              # Modern glassmorphism web UI
```

---

## 👥 Authors & Acknowledgments

- **Author**: Ujjwal ([Phantom1270](https://github.com/Phantom1270))
- **Knowledge Sources**: Fact-check data curated from verified IFCN (International Fact-Checking Network) signatories including AltNews, BOOM Live, and PIB Fact Check.
- **Technologies**: FastAPI, SentenceTransformers, ChromaDB, Rank-BM25, Groq Llama 3.3, AWS Elastic Beanstalk, Amazon ECR, AWS CodePipeline.
