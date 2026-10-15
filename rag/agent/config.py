import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env")
load_dotenv()  # also load from current working directory

# Default LLM Backend ("groq" or "gemini")
DEFAULT_LLM_BACKEND = os.environ.get("DEFAULT_LLM_BACKEND", "groq")

def get_groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")

def get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")

# Server-side API keys (evaluated with .env loaded)
DEFAULT_GROQ_API_KEY = get_groq_api_key()
DEFAULT_GEMINI_API_KEY = get_gemini_api_key()

# Validated active models
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Retrieval configuration
TOP_K = 5
MAX_EVIDENCE_ARTICLES = 5
MAX_EXCERPTS_PER_ARTICLE = 2
MIN_SIMILARITY_THRESHOLD = 0.45
