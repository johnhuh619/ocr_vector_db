"""LLM client abstraction for generation layer.

Provides unified interface for LLM providers, starting with Gemini.
Uses the same google-generativeai library as embedding/provider.py.
"""

import os
from typing import Optional, Protocol

from .models import LLMResponse


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients (dependency inversion)."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response from prompt."""
        ...


class GeminiLLMClient:
    """Gemini LLM client using google-generativeai.

    Uses the same library pattern as GeminiEmbeddings (embedding/provider.py).

    Example:
        >>> client = GeminiLLMClient()
        >>> response = client.generate("Explain Python decorators")
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        """Initialize Gemini LLM client.

        Args:
            model: Gemini model to use (default: gemini-2.0-flash)
            api_key: Google API key (falls back to GOOGLE_API_KEY env var)
        """
        import google.generativeai as genai

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini LLM")

        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)
        self._model_name = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response using Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Generation temperature (0-1)
            max_tokens: Maximum output tokens

        Returns:
            LLMResponse with generated content
        """
        # Build full prompt with system instruction
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            response = self._model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            # Handle blocked or empty responses
            if not response.candidates:
                return LLMResponse(
                    content="I couldn't generate a response. Please try rephrasing your question.",
                    model=self._model_name,
                )

            # Extract text from response
            text = response.text if hasattr(response, "text") else ""

            return LLMResponse(
                content=text.strip(),
                model=self._model_name,
                usage=None,  # Gemini doesn't expose token usage easily
            )

        except Exception as e:
            print(f"[llm] Generation failed: {e}")
            return LLMResponse(
                content=f"Generation error: {str(e)}",
                model=self._model_name,
            )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name


__all__ = ["LLMClientProtocol", "GeminiLLMClient", "LLMResponse"]
