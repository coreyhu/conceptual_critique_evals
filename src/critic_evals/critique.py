"""Critique prompt construction and one-shot critique generation."""

from __future__ import annotations

from datetime import datetime, timezone

from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import ModelSpec
from critic_evals.schema import ArgumentItem, CritiqueRecord, Usage

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful, intellectually honest critic of arguments. "
    "Engage with the specific argument in front of you on its own terms."
)

CRITIQUE_PROMPT = (
    "You will be given a question and an argument that attempts to answer it.\n\n"
    "Critique the argument. Assess how well it actually supports its conclusion. "
    "If it has weaknesses — unsupported leaps, internal tensions, overlooked "
    "considerations, or claims that do not hold up — identify the most important "
    "ones and explain why they matter. If the argument is basically sound, say so "
    "rather than manufacturing objections. Engage with the argument as written.\n\n"
    "Question:\n{question}\n\n"
    "Argument:\n{argument}"
)


def build_prompt(item: ArgumentItem) -> tuple[str, str]:
    """Return `(system, user_prompt)` for eliciting a critique of `item`."""
    return CRITIQUE_SYSTEM_PROMPT, CRITIQUE_PROMPT.format(
        question=item.question, argument=item.argument
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def generate_critique(
    client: AnthropicClient,
    model: ModelSpec,
    item: ArgumentItem,
    *,
    sample: int = 0,
    max_tokens: int = 4096,
) -> CritiqueRecord:
    """Ask `model` to critique `item` and return the result as a `CritiqueRecord`.

    Exceptions propagate; callers that fan out decide how to record failures.
    """
    system, prompt = build_prompt(item)
    completion = await client.complete(
        model_id=model.model_id, system=system, prompt=prompt, max_tokens=max_tokens
    )
    return CritiqueRecord(
        item_id=item.id,
        model=model.label,
        model_id=completion.model_id,
        sample=sample,
        system=system,
        prompt=prompt,
        response=completion.text,
        success=True,
        stop_reason=completion.stop_reason,
        usage=Usage(completion.input_tokens, completion.output_tokens),
        request_id=completion.request_id,
        timestamp=_utcnow_iso(),
    )
