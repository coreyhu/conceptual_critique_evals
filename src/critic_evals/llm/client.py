"""Thin async wrapper over the Anthropic Messages API."""

from __future__ import annotations

import json
from dataclasses import dataclass

from anthropic import AsyncAnthropic, Omit
from anthropic.types.output_config_param import OutputConfigParam


@dataclass(frozen=True, slots=True)
class Completion:
    model_id: str
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    request_id: str | None
    parsed: dict[str, object] | None = None  # populated when a `schema` is given


class AnthropicClient:
    """Async client for single-turn completions.

    `api_key=None` lets the SDK resolve `ANTHROPIC_API_KEY` from the
    environment (load a `.env` first if you keep the key there).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_retries: int = 8,
        timeout: float = 120.0,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    async def complete(
        self,
        *,
        model_id: str,
        system: str,
        prompt: str,
        max_tokens: int = 4096,
        schema: dict[str, object] | None = None,
    ) -> Completion:
        """Single-turn completion. With `schema`, constrains output to it and fills `.parsed`."""
        output_config: OutputConfigParam | Omit = (
            {"format": {"type": "json_schema", "schema": schema}}
            if schema is not None
            else Omit()
        )
        message = await self._client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            model_id=message.model,
            text=text,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            request_id=message._request_id,  # public despite the underscore
            parsed=json.loads(text) if schema is not None else None,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "AnthropicClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
