"""Gemini LLM client for teaching demonstrations and prompt engineering.

Provides a simple interface to Google's Gemini API with artifact generation
matching the repo's config-driven pattern.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types


@dataclass
class GeminiResponse:
    """Response object from Gemini API."""

    text: str
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    model: str | None = None
    timestamp: str | None = None
    tool_usage: dict | None = None

    def to_dict(self) -> dict:
        """Convert response to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "timestamp": self.timestamp,
            "tool_usage": self.tool_usage,
        }


class GeminiClient:
    """Client for interacting with Google Gemini API.
    
    Usage:
        client = GeminiClient(api_key="your-api-key", model_name="gemini-2.0-flash")
        response = client.generate("Your prompt here")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        safety_settings: dict | None = None,
    ) -> None:
        """Initialize Gemini client.
        
        Args:
            api_key: Google API key (falls back to GEMINI_API_KEY env var)
            model_name: Model to use (e.g., 'gemini-2.0-flash' for free tier)
            temperature: Creativity parameter (0-1)
            max_tokens: Maximum response length
            safety_settings: Safety configuration dict
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or self._load_api_key_from_dotenv()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not provided and not set in environment. "
                "Get free key at https://ai.google.dev/"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _load_api_key_from_dotenv() -> str | None:
        """Load GEMINI_API_KEY from project-level .env file when available."""
        # Project root is two levels above this file: src/models/gemini_client.py
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"
        if not dotenv_path.exists():
            return None

        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("GEMINI_API_KEY="):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
        return None

    @staticmethod
    def _default_safety_settings() -> dict:
        """Return default safety settings for classroom use (unused in new SDK)."""
        return {}

    @staticmethod
    def _to_json_safe(value):
        """Convert SDK objects into JSON-safe data structures."""
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return str(value)

    def _extract_tool_usage(self, response) -> dict:
        """Extract tool-call and grounding evidence from a Gemini response."""
        function_calls: list[dict] = []
        execution_events: list[str] = []
        grounding_metadata = None

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            candidate_grounding = getattr(candidate, "grounding_metadata", None)
            if candidate_grounding and grounding_metadata is None:
                grounding_metadata = self._to_json_safe(candidate_grounding)

            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    function_calls.append(
                        {
                            "name": getattr(function_call, "name", None),
                            "args": self._to_json_safe(getattr(function_call, "args", None)),
                        }
                    )

                if getattr(part, "executable_code", None) is not None:
                    execution_events.append("executable_code")
                if getattr(part, "code_execution_result", None) is not None:
                    execution_events.append("code_execution_result")

        return {
            "tool_calls_detected": bool(function_calls or execution_events or grounding_metadata),
            "function_calls": function_calls,
            "execution_events": execution_events,
            "has_grounding_metadata": grounding_metadata is not None,
            "grounding_metadata": grounding_metadata,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_config: dict | None = None,
    ) -> GeminiResponse:
        """Generate response from Gemini API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system context (prepended to prompt)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            tools: Optional Gemini tool definitions
            tool_config: Optional Gemini tool config
            
        Returns:
            GeminiResponse object with text and metadata
        """
        # Prepare final prompt
        contents = []
        if system_prompt:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"{system_prompt}\n\nUser Query: {prompt}")],
            ))
        else:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ))

        # Use provided overrides or defaults
        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens

        # Call API with timing
        start_time = time.time()
        try:
            config_kwargs: dict = {
                "temperature": temp,
                "max_output_tokens": max_toks,
            }
            if tools:
                config_kwargs["tools"] = tools
            if tool_config:
                config_kwargs["tool_config"] = tool_config

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            latency_ms = (time.time() - start_time) * 1000

            # Extract response text
            response_text = response.text if response.text else "[No response]"

            # Extract token usage metadata when available
            prompt_tokens = None
            response_tokens = None
            total_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                response_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)

            tool_usage = self._extract_tool_usage(response)

            return GeminiResponse(
                text=response_text,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                model=self.model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                tool_usage=tool_usage,
            )

        except Exception as e:
            error_msg = f"Gemini API Error: {str(e)}"
            return GeminiResponse(
                text=error_msg,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                model=self.model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def generate_with_examples(
        self,
        user_prompt: str,
        examples: list[tuple[str, str]] | None = None,
        system_prompt: str | None = None,
        **gen_kwargs,
    ) -> GeminiResponse:
        """Generate with few-shot examples for better results.
        
        Args:
            user_prompt: Main query
            examples: List of (input, output) tuples for in-context learning
            system_prompt: System context
            **gen_kwargs: Additional generation parameters
            
        Returns:
            GeminiResponse object
        """
        # Build few-shot prompt
        few_shot = ""
        if examples:
            few_shot = "\n\nExamples:\n"
            for inp, outp in examples:
                few_shot += f"Input: {inp}\nOutput: {outp}\n"

        full_prompt = user_prompt + few_shot if few_shot else user_prompt
        return self.generate(full_prompt, system_prompt, **gen_kwargs)
