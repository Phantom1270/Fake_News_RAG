# Fake News Detector - AWS EC2 Deployment Guide (College Demo)

This guide walks you through deploying the **Fake News Detector RAG** system to a single **AWS EC2** instance for a temporary college project demonstration.

Priorities for this deployment:
- **Zero or lowest practical cost** (under AWS Free Tier allowances where eligible).
- **Simplicity**: Single-instance architecture; no external databases, no load balancers, no Kubernetes.
- **Data Preservation**: Uses the existing pre-built SQLite and Chroma databases (~2,500 fact checks). No re-embedding or scraping required.
- **Easy Cleanup**: Straightforward procedures to turn off and terminate resources when your demo concludes.

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
    ├── In-Memory Rate Limiting (default: 20 req/min/IP)
    ├── RAG Retrieval Engine (all-MiniLM-L6-v2 on CPU)
    │     ├── SQLite Store (rag/data/rag.db)
    │     ├── Chroma Vector Store (rag/data/indexes/chroma/)
    │     └── BM25 Lexical Index (rag/data/indexes/bm25.pkl)
    └── LLM Generation via Cloud APIs (Groq / Google Gemini)
```

- **Inference & Retrieval**: Pure CPU execution.
- **LLM Reasoning**: Cloud APIs (Groq / Gemini) so no massive local LLM weights are stored in RAM.
- **Host Specifications**: 1x AWS EC2 instance (`t2.micro` or `t3.micro`) running Ubuntu 22.04 LTS with 2 GB Swap space.

---

## 2. Prerequisites & Required Files

### Supported Python Version
- **Python 3.10 – 3.12** (Tested and verified on **Python 3.12.2**).

### Deployment File Checklist
Only transfer these files and folders to your server:

```text
Fake_News_Detector/
├── .env.example
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
│   └── data/                    <-- CRITICAL: Existing pre-indexed data
│       ├── rag.db               (~9.4 MB)
│       └── indexes/
│           ├── manifest.json
│           ├── bm25.pkl         (~2.6 MB)
│           └── chroma/          (~30 MB)
└── web/
    ├── app.py
    └── static/
        └── index.html
```

> **DO NOT TRANSFER:** `.venv/`, `__pycache__/`, `tests/`, `scripts/altnews_scraper/data/` (raw 13.7 MB JSON not needed), `Data/IFND.csv` (raw CSV not needed), or your local `.env` file.

---

## 3. Local Verification Before Deployment

Before touching AWS, verify that the application runs locally:

### PowerShell (Windows)
```powershell
# 1. From the project root
cd d:\projects\Fake_News_Detector

# 2. Run the deployment readiness test suite
pytest tests/test_deployment_readiness.py -v

# 3. Start the local server
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open your browser to `http://localhost:8000` and confirm:
1. The newsroom homepage loads.
2. The welcome message displays fact-checks count.
3. You can verify a sample claim (e.g., *"Were the Nepal flood videos real?"*).

---

## 4. AWS Free Tier & Cost Checklist

Before launching an instance, check your AWS account status:

1. **Account Age & Free Tier Period**:
   - The 12-month Free Tier (750 hours/month of `t2.micro` or `t3.micro`) applies to accounts within 12 months of signup.
   - If your account is older than 12 months, running a `t2.micro` in Mumbai (`ap-south-1`) costs approximately **$0.0116/hour** (~**$0.28 per 24 hours**). Remember to stop or terminate the instance when done!
2. **EBS Volume Allowance**:
   - Free Tier allows up to **30 GB** total EBS storage across all volumes.
   - Allocate **20 GB gp3** for your instance to leave safety margin.
3. **Public IPv4 Charges**:
   - AWS charges ~$0.005/hour for in-use public IPv4 addresses (~$0.12/day).
4. **Services to AVOID**:
   - Do NOT launch an Application Load Balancer (ALB) (~$18/month).
   - Do NOT launch a NAT Gateway (~$32/month).
   - Do NOT allocate an Elastic IP (unattached EIPs incur hourly idle charges).
   - Do NOT use RDS (managed database).

---

## 5. Step-by-Step EC2 Launch Instructions

### Step 1: Launch EC2 Instance via AWS Console
1. Open the [AWS Management Console](https://console.aws.amazon.com/).
2. Select Region: **Asia Pacific (Mumbai) `ap-south-1`** (top right dropdown).
3. Navigate to **EC2** and click **Launch Instance**.
4. **Name**: `fakenews-detector-demo`.
5. **Application and OS Images (AMI)**: Select **Ubuntu** $ightarrow$ **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type** (64-bit x86).
6. **Instance Type**:
   - Select **`t2.micro`** (1 vCPU, 1 GB RAM) or **`t3.micro`** (2 vCPU, 1 GB RAM).
7. **Key Pair (login)**:
   - Click **Create new key pair**.
   - Name: `fakenews-key`.
   - Type: `RSA`, Format: `.pem`.
   - Download and save `fakenews-key.pem` securely on your computer.
8. **Network Settings (Security Group)**:
   - Choose **Create security group**.
   - Check **Allow SSH traffic from** $ightarrow$ Select **My IP** (never allow `0.0.0.0/0` for SSH).
   - Check **Allow HTTP traffic from the internet** (Port 80).
9. **Configure Storage**:
   - Set **20 GiB**, Volume type: **gp3**.
10. Click **Launch Instance**.

---

## 6. Server Configuration (Linux)

### Step 2: Connect via SSH
From your Windows PowerShell or Git Bash:

```powershell
# Set read-only permissions on key file (PowerShell)
icacls fakenews-key.pem /inheritance:r
icacls fakenews-key.pem /grant:r "$($env:USERNAME):(R)"

# Connect to EC2 (replace with your instance Public IP)
ssh -i fakenews-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

---

### Step 3: Configure 2 GB Swap Space (CRITICAL FOR 1 GB RAM)
Because `t2.micro` has 1.0 GB RAM, running `sentence-transformers` and PyTorch requires swap space to prevent the Linux Out-Of-Memory (OOM) killer from terminating Uvicorn during startup.

Run these commands inside your EC2 terminal:

```bash
# Check existing swap (usually 0)
free -m

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

### Step 4: Install System Dependencies & Nginx
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx rsync
```

---

## 7. Project Code Transfer

From your **local Windows machine** (open PowerShell in the project directory):

```powershell
# In PowerShell: create a clean zip archive excluding caches and dev data
# Exclude: .venv, .git, scripts/altnews_scraper/data, Data/IFND.csv, __pycache__
tar --exclude='.venv' `
    --exclude='__pycache__' `
    --exclude='.pytest_cache' `
    --exclude='Data/IFND.csv' `
    --exclude='scripts/altnews_scraper/data' `
    --exclude='tests' `
    -czf deploy_package.tar.gz rag web requirements-prod.txt .env.example

# Copy the archive to your EC2 instance
scp -i fakenews-key.pem deploy_package.tar.gz ubuntu@<YOUR_EC2_PUBLIC_IP>:~/
```

---

## 8. Server Setup & Dependencies

Back inside your **EC2 SSH terminal**:

```bash
# Extract the archive into application directory
mkdir -p ~/Fake_News_Detector
tar -xzf ~/deploy_package.tar.gz -C ~/Fake_News_Detector
cd ~/Fake_News_Detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# CRITICAL: Install CPU-only PyTorch first (avoids 4 GB of CUDA drivers)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
pip install -r requirements-prod.txt
```

---

## 9. Environment Variables Configuration

Create the `.env` file on EC2:

```bash
cd ~/Fake_News_Detector
cp .env.example .env
nano .env
```

Fill in your actual API keys:
```ini
GROQ_API_KEY=gsk_your_actual_groq_key
GEMINI_API_KEY=your_actual_gemini_key
DEFAULT_LLM_BACKEND=groq
ADMIN_PASSWORD=your_strong_admin_password
RATE_LIMIT_PER_MINUTE=20
PYTHONPATH=.
```
*(Press `Ctrl+O` then `Enter` to save, and `Ctrl+X` to exit nano).*

Verify permissions on `.env`:
```bash
chmod 600 .env
```

---

## 10. Configure Nginx Reverse Proxy

Nginx allows users to access the application on standard HTTP port 80 without running Uvicorn as root.

Create the Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/fakenews
```

Paste this configuration:

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

        # Disable buffering for SSE streaming (admin upload progress)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }
}
```

Enable the site and restart Nginx:
```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/fakenews /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 11. Systemd Service (Process Persistence)

To keep the application running continuously and automatically restart on reboot:

```bash
sudo nano /etc/systemd/system/fakenews.service
```

Paste this service definition:

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

# Resource safety limits
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

Check the status:
```bash
sudo systemctl status fakenews
```

View live logs:
```bash
journalctl -u fakenews -f
```

---

## 12. Verification & Testing on AWS

1. **Check local health via curl on the instance:**
   ```bash
   curl http://127.0.0.1:8000/api/health
   # Expected output: {"status":"ok","retrieval_engine_ready":true}
   ```

2. **Access via browser:**
   Open: `http://<YOUR_EC2_PUBLIC_IP>`

3. **Verify Demo Functionality:**
   - Type a claim: *"Were the Nepal flood videos real?"*
   - Select Backend: Groq or Gemini.
   - Click submit and verify the verdict stamp, confidence bar, and clickable citations.

---

## 13. Troubleshooting Common Issues

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `Process killed` or `Out of memory` during startup | Missing swap file on 1 GB RAM instance | Run Section 6 Step 3 to enable the 2 GB swapfile. |
| `Startup aborted: Required pre-built RAG assets are missing` | `rag/data/` was not transferred properly | Ensure `rag/data/rag.db`, `rag/data/indexes/bm25.pkl`, and `rag/data/indexes/chroma/chroma.sqlite3` exist. |
| `502 Bad Gateway` in browser | FastAPI service is stopped or restarting | Run `sudo systemctl status fakenews` and check `journalctl -u fakenews -n 50`. |
| `429 Too Many Requests` | Rate limit exceeded | Wait 60 seconds or increase `RATE_LIMIT_PER_MINUTE` in `.env`. |
| `Admin functionality is disabled (503)` | `ADMIN_PASSWORD` is not set in `.env` | Add `ADMIN_PASSWORD=...` to `/home/ubuntu/Fake_News_Detector/.env` and restart service (`sudo systemctl restart fakenews`). |
| Cannot connect via browser | Security Group missing port 80 | Edit Security Group in AWS Console $ightarrow$ Add Inbound Rule: HTTP (Port 80) from `0.0.0.0/0`. |

---

## 14. Stopping the Application (When Not Presenting)

To temporarily stop the application to conserve resources:

```bash
# Stop application
sudo systemctl stop fakenews

# In AWS Console:
# Select Instance -> Instance state -> Stop instance
```

> **Note:** Stopping an instance stops compute charges, but EBS volume storage charges (~$1.60/month for 20 GB if outside free tier) still accrue. When your college evaluation is finished, perform permanent termination as described in `AWS_CLEANUP.md`.
