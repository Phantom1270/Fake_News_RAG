import os
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def analyze_claim(self, claim: str, evidence_context: str) -> dict:
        """
        Send claim + evidence to LLM, get structured analysis back.

        Args:
            claim: The user's claim text
            evidence_context: Formatted string of retrieved fact-check evidence

        Returns:
            dict with keys: verdict, confidence, explanation, reasoning, matched_articles
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this client has a valid API key and can make requests."""
        pass

    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        """Generate a direct, factual answer to the question using retrieved context."""
        pass


class GroqClient(LLMClient):
    """Groq API client. Fast inference, generous free tier."""

    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def analyze_claim(self, claim: str, evidence_context: str) -> dict:
        import httpx

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(claim, evidence_context)

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2500,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        return _parse_llm_response(content)

    def generate_answer(self, question: str, context: str) -> str:
        import httpx

        system_prompt = (
            "You are a helpful and accurate question-answering assistant for an investigative fact-checking knowledge base. "
            "Your task is to answer the user's question clearly, objectively, and comprehensively using only the provided fact-check context.\n\n"
            "Guidelines:\n"
            "1. Base your answer strictly on the provided context. If the context does not contain enough information to answer the question, state: 'Based on the available knowledge base, there is no verified information to answer this question.'\n"
            "2. Provide direct, factual explanations. Mention specific facts, names, dates, and conclusions from the sources.\n"
            "3. Do not speculate or introduce unverified outside claims."
        )
        user_prompt = f"FACT-CHECK KNOWLEDGE BASE CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip()



class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def analyze_claim(self, claim: str, evidence_context: str) -> dict:
        import httpx

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(claim, evidence_context)

        response = httpx.post(
            f"{self.base_url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {"parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2500,
                    "responseMimeType": "application/json"
                }
            },
            timeout=30
        )
        response.raise_for_status()

        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_response(content)

    def generate_answer(self, question: str, context: str) -> str:
        import httpx

        system_prompt = (
            "You are a helpful and accurate question-answering assistant for an investigative fact-checking knowledge base. "
            "Your task is to answer the user's question clearly, objectively, and comprehensively using only the provided fact-check context.\n\n"
            "Guidelines:\n"
            "1. Base your answer strictly on the provided context. If the context does not contain enough information to answer the question, state: 'Based on the available knowledge base, there is no verified information to answer this question.'\n"
            "2. Provide direct, factual explanations. Mention specific facts, names, dates, and conclusions from the sources.\n"
            "3. Do not speculate or introduce unverified outside claims."
        )
        user_prompt = f"FACT-CHECK KNOWLEDGE BASE CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"

        response = httpx.post(
            f"{self.base_url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {"parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1000
                }
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return content.strip()



# ── Shared prompt building and response parsing ──

def _build_system_prompt() -> str:
    return """You are an expert fact-checking analyst working for a news verification service. Your job is to thoroughly analyze a user's claim against retrieved fact-check articles and provide a detailed, informative verdict.

YOUR RESPONSE STYLE:
- Write like an investigative journalist, not a robot
- Explain the FULL context: what the claim says, what actually happened, and why the claim is wrong/right/misleading
- Reference specific details from the evidence (names, dates, events, sources)
- If the claim is false, explain what the REAL story is based on the evidence
- If the claim is misleading, explain what's true and what's distorted
- Be thorough but concise — aim for 3-5 sentences, not 1-2

RULES:
1. Only use the provided evidence. Do not use external knowledge.
2. If the retrieved evidence does not contain information relevant to the claim, respond with verdict "unverifiable" and explain that no matching fact-checks were found in the database.
3. A claim is "false" if the core assertion itself is factually incorrect — the event didn't happen, the people involved are wrong, or the stated facts are fabricated.
4. A claim is "misleading" if the underlying event is real but the specific evidence presented (images, videos, statistics) is fabricated, doctored, or from a different context. For example, if a real flood occurred but the viral videos showing it are from a different event or are AI-generated, the claim that "these videos show the flood" is misleading, not simply false.
5. A claim is "partially_true" if some parts are correct but others are not. Specify which parts are correct and which are not.
6. Always reference which evidence article supports your conclusion.

RESPOND IN THIS EXACT json FORMAT:
{
    "verdict": "true|false|misleading|partially_true|unverifiable",
    "confidence": 0.0 to 1.0,
    "explanation": "A detailed 3-5 sentence explanation that covers: (1) what the claim states, (2) what the evidence actually shows, (3) why the claim is true/false/misleading. Write like a journalist explaining to a curious reader.",
    "reasoning": "Step-by-step analysis: How does each piece of evidence relate to the claim? What specific details from the evidence support your conclusion?",
    "matched_articles": ["title of evidence article 1", "title of evidence article 2"]
}"""


def _build_user_prompt(claim: str, evidence_context: str) -> str:
    return f"""CLAIM TO VERIFY:
{claim}

RETRIEVED FACT-CHECK EVIDENCE:
{evidence_context}

Analyze the claim against the evidence above. Respond in json format."""


def _parse_llm_response(content: str) -> dict:
    """Parse LLM JSON response with fallback handling."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return _fallback_unverifiable(content)
        else:
            return _fallback_unverifiable(content)

    # Normalize verdict
    verdict = parsed.get("verdict", "unverifiable").lower().strip()
    valid_verdicts = {"true", "false", "misleading", "partially_true", "unverifiable"}
    if verdict not in valid_verdicts:
        verdict = "unverifiable"

    # Clamp confidence
    try:
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.0

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": parsed.get("explanation", ""),
        "reasoning": parsed.get("reasoning", ""),
        "matched_articles": parsed.get("matched_articles", [])
    }

def _fallback_unverifiable(content: str) -> dict:
    logger.warning(f"Failed to parse LLM response as JSON: {content[:200]}")
    return {
        "verdict": "unverifiable",
        "confidence": 0.0,
        "explanation": "The analysis system was unable to process the response.",
        "reasoning": content,
        "matched_articles": []
    }
