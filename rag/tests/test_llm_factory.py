import pytest
import os
from unittest.mock import patch

from rag.llm import create_llm_client
from rag.llm.groq_client import GroqClient
from rag.config.settings import Settings


@pytest.fixture
def base_settings():
    return Settings()


class TestLLMFactory:
    def test_factory_returns_groq_client_by_default(self, base_settings):
        base_settings.llm.provider = "groq"
        base_settings.llm.api_key = "fake_key"
        client = create_llm_client(base_settings)
        assert isinstance(client, GroqClient)

    def test_factory_returns_openai_client_when_configured(self, base_settings):
        base_settings.llm.provider = "openai"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
            try:
                import openai
                from rag.llm.openai_client import OpenAIClient
                
                with patch("rag.llm.openai_client.OpenAI"):
                    client = create_llm_client(base_settings)
                    assert isinstance(client, OpenAIClient)
            except ImportError:
                pytest.skip("openai not installed")

    def test_factory_raises_for_openai_without_key(self, base_settings):
        base_settings.llm.provider = "openai"
        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable required"):
                create_llm_client(base_settings)

    def test_factory_raises_for_unknown_provider(self, base_settings):
        base_settings.llm.provider = "invalid"
        with pytest.raises(ValueError, match="Unknown LLM provider: 'invalid'"):
            create_llm_client(base_settings)
