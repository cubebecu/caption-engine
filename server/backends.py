"""LLM backend abstraction for caption-engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger("caption-engine")


@dataclass
class CaptionResult:
    """Result from a single image captioning call."""
    caption_text: str
    model_name: str


@runtime_checkable
class LLMBackend(Protocol):
    """Interface for LLM backends used by caption-engine."""

    async def health_check(self) -> bool: ...
    async def caption(self, image_base64: str, system_prompt: str) -> CaptionResult: ...


class AnthropicBackend:
    """Anthropic API backend using raw httpx (no SDK dependency for inference)."""

    def __init__(self):
        self._http = None  # Shared httpx.AsyncClient
        self._model_name: str | None = None
        self._api_key: str | None = None

    async def initialize(self) -> None:
        import os
        import httpx

        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            logger.error("ANTHROPIC_API_KEY is not set")
            raise RuntimeError("ANTHROPIC_API_KEY environment variable required")

        # Single shared client for all requests — creating per-request hangs in this env
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=5.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1, keepalive_expiry=10),
        )

        await self._detect_model()
        logger.info("Anthropic backend initialized with model: %s", self._model_name)

    async def _detect_model(self) -> None:
        """List available models and pick the latest Sonnet."""
        try:
            resp = await self._http.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            models_list = data.get("data", [])
            sonnets = [m for m in models_list if "sonnet" in m["id"].lower()]
            if not sonnets:
                logger.warning("No Sonnet models found, using default")
                self._model_name = "claude-sonnet-4-5-20250929"
                return

            sonnets.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            self._model_name = sonnets[0]["id"]
            logger.info(
                "Auto-detected latest Sonnet model: %s (created: %s)",
                self._model_name,
                sonnets[0].get("created_at"),
            )
        except Exception as e:
            logger.warning("Model detection failed (%s), using default", e)
            self._model_name = "claude-sonnet-4-5-20250929"

    async def health_check(self) -> bool:
        return self._api_key is not None and self._model_name is not None

    async def caption(self, image_base64: str, system_prompt: str) -> CaptionResult:
        """Generate a caption using shared httpx client."""

        payload = {
            "model": self._model_name,
            "max_tokens": 2048,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Focus on technical content.",
                        },
                    ],
                }
            ],
        }

        logger.info(
            "[ANTHROPIC] START: model=%s, image_b64_len=%d, prompt_len=%d",
            self._model_name, len(image_base64), len(system_prompt),
        )
        start = time.time()

        try:
            logger.info("[ANTHROPIC] HTTP POST to /v1/messages")
            resp = await self._http.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            elapsed = (time.time() - start) * 1000
            logger.info(
                "[ANTHROPIC] HTTP response: status=%d, after %.1fms",
                resp.status_code, elapsed,
            )

            if resp.status_code != 200:
                body = resp.text[:500]
                raise Exception(f"Anthropic API error {resp.status_code}: {body}")

            result = resp.json()
            caption_text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    caption_text += block["text"]

            total_elapsed = (time.time() - start) * 1000
            logger.info(
                "[ANTHROPIC] OK: %.1fms, tokens=%d",
                total_elapsed, len(caption_text.split()),
            )
            return CaptionResult(
                caption_text=caption_text,
                model_name=self._model_name or "claude-sonnet",
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(
                "[ANTHROPIC] FAILED after %.1fms: %s",
                elapsed, str(e), exc_info=True,
            )
            raise

    async def close(self) -> None:
        """Close the shared httpx client."""
        if self._http:
            await self._http.aclose()
