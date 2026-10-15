"""
Groq API backend client.
"""

from typing import List, Dict, Optional
import time
import logging
from rag.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """Groq API backend."""

    def __init__(self, api_key: str = None, model: str = "qwen/qwen3.8-27b", max_retries: int = 3):
        try:
            from groq import Groq, GroqError
            self.GroqError = GroqError
        except ImportError:
            raise ImportError(
                "groq package required for Groq backend. "
                "Install with: pip install groq"
            )
        
        self.client = Groq(api_key=api_key)
        self.model_name = model
        self.max_retries = max_retries

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1,
             max_tokens: int = 300) -> str:
        """Send a chat completion request to Groq with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limits or other transient errors
                if "rate limit" in error_msg or "429" in error_msg or "connection" in error_msg or "network" in error_msg:
                    if retries < self.max_retries:
                        sleep_time = 2 ** retries
                        logger.warning(f"Groq API error: {e}. Retrying in {sleep_time} seconds (attempt {retries + 1}/{self.max_retries})...")
                        time.sleep(sleep_time)
                        retries += 1
                        continue
                    else:
                        raise RuntimeError(f"Groq rate limit exceeded / connection failed after {self.max_retries} retries: {e}") from e
                elif "authentication" in error_msg or "invalid api key" in error_msg or "unauthorized" in error_msg:
                    raise RuntimeError("Invalid Groq API key") from e
                else:
                    raise RuntimeError(f"Groq API error: {e}") from e

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
