"""Thin async wrapper over the Anthropic Messages API."""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic


@dataclass(frozen=True, slots=True)
class Completion:
    model_id: str
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    request_id: str | None


class AnthropicClient:
    """Async client for single-turn completions.

    `api_key=None` lets the SDK resolve `ANTHROPIC_API_KEY` from the
    environment (load a `.env` first if you keep the key there).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_retries: int = 4,
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
    ) -> Completion:
        message = await self._client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            model_id=message.model,
            text=text,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            request_id=message._request_id,  # public despite the underscore
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "AnthropicClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
