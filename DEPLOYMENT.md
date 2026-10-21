# Fake News Detector & Fact-Check Q&A - AWS Deployment Guide

This guide walks you through deploying the **Fake News Detector / Fact-Check Knowledge Base RAG** system to **AWS Cloud** (AWS EC2 or AWS App Runner / Docker).

### Deployment Goals:
- **Cost Efficiency**: Runs under the AWS Free Tier (`t2.micro` / `t3.micro`) or low-cost container instances.
- **Simplicity**: Self-contained architecture using pre-built SQLite and Chroma databases (~2,500 fact checks). No external database services or GPU drivers needed.
- **Reliability**: Uses CPU-only PyTorch and cloud LLM APIs (Groq and Gemini) for fast inference without RAM bottlenecks.

---

## 1. Architecture Overview

```text
[ Browser / Client ]
         │ HTTP (Port 80)
         ▼
[ Nginx Reverse Proxy ] (Port 80 -> 8000)
         │
         ▼
[ FastAPI / Uvicorn Server ] (Port 8000)
    ├── Serves Frontend UI (web/static/index.html)
    ├── Endpoints:
    │     ├── GET  /              (Fact-Check Q&A UI)
    │     ├── POST /api/ask       (RAG retrieval + LLM synthesized answer)
    │     ├── GET  /api/health    (System & index health check)
    │     └── GET  /favicon.ico   (204 No Content)
    ├── RAG Retrieval Engine (all-MiniLM-L6-v2 on CPU)
    │     ├── SQLite Metadata Store (rag/data/rag.db)
    │     ├── Chroma Vector Store (rag/data/indexes/chroma/)
    │     └── BM25 Lexical Index (rag/data/indexes/bm25.pkl)
    └── LLM Generation via Cloud APIs (Groq / Google Gemini)
```

---

## 2. Required Files Checklist

Only the following core folders and files are needed for deployment:

```text
Fake_News_Detector/
├── .env.example
├── Dockerfile                   <-- For containerized hosting (App Runner / ECS)
├── requirements.txt            <-- Complete production dependencies
├── requirements-prod.txt
├── rag/
│   ├── config/ (settings.py, default.yaml)
│   ├── models/
│   ├── pipeline/
│   ├── retrieval/
│   ├── search/
│   ├── agent/
│   ├── storage/
│   ├── utils/
│   └── data/                   <-- CRITICAL: Pre-built databases (~35 MB)
│       ├── rag.db              (~9.4 MB SQLite)
│       └── indexes/
│           ├── manifest.json
│           ├── bm25.pkl        (~2.6 MB BM25 index)
│           └── chroma/         (~19 MB Chroma vector store)
└── web/
    ├── app.py
    └── static/
        └── index.html
```

---

## 3. Local Verification Before Deployment

Before deploying to AWS, verify the repository locally:

```powershell
# 1. Run deployment readiness test suite (14 checks)
pytest tests/test_deployment_readiness.py -v

# 2. Run application test suite (8 checks)
pytest tests/test_qa_app.py -v

# 3. Start the local server
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in your browser and confirm:
1. The Fact-Check Knowledge Base Q&A page loads.
2. The health check at `http://localhost:8000/api/health` reports status `ok`.
3. Submit a test question: *"Were the Nepal flood videos authentic?"* and verify the answer and source citations appear.

---

## 4. Deployment Option A: AWS EC2 (Step-by-Step)

### Step 1: Launch EC2 Instance via AWS Console
1. Open the [AWS Management Console](https://console.aws.amazon.com/).
2. Select Region: **Asia Pacific (Mumbai) `ap-south-1`** (or your closest region).
3. Navigate to **EC2** and click **Launch Instance**.
4. **Name**: `fakenews-detector`.
5. **OS Image**: **Ubuntu Server 22.04 LTS (HVM)**, 64-bit (x86).
6. **Instance Type**:
   - `t2.micro` (Free Tier eligible) or `t3.small` (recommended for smooth memory headroom).
7. **Key Pair**: Create new or select existing `.pem` key pair.
8. **Network Settings (Security Group)**:
   - Allow **SSH** (Port 22) from **My IP**.
   - Allow **HTTP** (Port 80) from `0.0.0.0/0`.
9. **Storage**: Set **20 GiB gp3**.
10. Click **Launch Instance**.

---

### Step 2: Connect to EC2
```powershell
ssh -i your-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

---

### Step 3: Configure 2 GB Swap Space (CRITICAL FOR 1 GB RAM INSTANCES)
PyTorch and sentence-transformers require swap on 1 GB instances to prevent the Linux OOM killer from terminating Uvicorn during startup:

```bash
# Allocate 2 GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -m
```

---

### Step 4: Install System Packages
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx git
```

---

### Step 5: Transfer or Clone Repository
On the EC2 instance:
```bash
# Clone the repository
git clone https://github.com/Phantom1270/Fake_News_RAG.git ~/Fake_News_Detector
cd ~/Fake_News_Detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# CRITICAL: Install CPU-only PyTorch first (avoids ~4 GB CUDA drivers)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
pip install -r requirements.txt
```

---

### Step 6: Configure Environment Variables
```bash
cp .env.example .env
nano .env
```

Enter your API keys:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key
GEMINI_API_KEY=your_actual_gemini_api_key
DEFAULT_LLM_BACKEND=groq
PYTHONPATH=.
```
Press `Ctrl+O`, `Enter`, and `Ctrl+X` to save and exit. Secure the file:
```bash
chmod 600 .env
```

---

### Step 7: Configure Nginx Reverse Proxy
```bash
sudo nano /etc/nginx/sites-available/fakenews
```

Paste:
```nginx
server {
    listen 80 default_server;
    server_name _;

    client_max_body_size 15M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

Enable and restart Nginx:
```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/fakenews /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

### Step 8: Configure Systemd Service (Auto-Restart & Persistence)
```bash
sudo nano /etc/systemd/system/fakenews.service
```

Paste:
```ini
[Unit]
Description=Fake News Detector FastAPI RAG Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/Fake_News_Detector
EnvironmentFile=/home/ubuntu/Fake_News_Detector/.env
Environment="PATH=/home/ubuntu/Fake_News_Detector/venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ubuntu/Fake_News_Detector/venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000 --workers 1

Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fakenews
sudo systemctl start fakenews
```

Check status:
```bash
sudo systemctl status fakenews
```

---

## 5. Deployment Option B: Docker / AWS App Runner

If you prefer containerized hosting without managing an EC2 virtual machine, use the included [Dockerfile](file:///d:/projects/Fake_News_Detector/Dockerfile):

### Run with Docker Locally or on EC2:
```bash
# Build the image (pre-caches SentenceTransformer model)
docker build -t fakenews-rag .

# Run container with your API key
docker run -d \
  -p 80:8000 \
  -e GROQ_API_KEY="your_groq_api_key" \
  -e GEMINI_API_KEY="your_gemini_api_key" \
  --name fakenews-app \
  fakenews-rag
```

### Deploy to AWS App Runner:
1. Push the container image to **Amazon ECR** (Elastic Container Registry).
2. Go to **AWS App Runner** $ightarrow$ **Create Service**.
3. Source: **Container registry** $ightarrow$ Select your ECR image.
4. CPU / Memory: Select **1 vCPU, 2 GB RAM**.
5. Environment variables: Add `GROQ_API_KEY` and `DEFAULT_LLM_BACKEND`.
6. Port: `8000`.
7. Click **Create & Deploy**. App Runner provides an automatic HTTPS endpoint with built-in health monitoring.

---

## 6. Verification & Health Checks

1. **Verify Health Endpoint:**
   ```bash
   curl http://<YOUR_PUBLIC_IP_OR_DOMAIN>/api/health
   # Response: {"status":"ok","qa_engine_ready":true,"retrieval_engine_ready":true,"indexed_documents":2499}
   ```

2. **Verify Frontend UI:**
   Open `http://<YOUR_PUBLIC_IP_OR_DOMAIN>/` in your browser.
   - Enter a test question: *"Were the Nepal flood videos authentic?"*
   - Verify that the response includes a synthesized summary and citations pointing to Alt News articles with match scores.

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Process killed` during startup | Insufficient RAM on `t2.micro` | Ensure Section 4 Step 3 swapfile is enabled (`free -m`). |
| Missing database error at startup | `rag/data` not transferred | Verify `rag/data/rag.db` and `rag/data/indexes/` exist. |
| `502 Bad Gateway` in browser | FastAPI service is stopped | Run `sudo systemctl status fakenews` and check `journalctl -u fakenews -n 50`. |
| Connection timeout in browser | Security Group missing port 80 | Edit Security Group in AWS Console and add Inbound Rule: HTTP (Port 80) from `0.0.0.0/0`. |
| LLM fallback message in response | Missing or expired Groq key | Check `GROQ_API_KEY` in `.env` and restart service (`sudo systemctl restart fakenews`). |
