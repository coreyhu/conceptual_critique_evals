"""Generating a critique from a model.

A single source for the critique prompt and the call that turns a model's
response into a `CritiqueRecord`, so every caller elicits critiques identically.

The default prompt is natural and single-pass (no decomposition scaffolding) and
offers a calibration off-ramp ("if it's basically sound, say so"), so a sound
argument can be endorsed rather than forced into a teardown.
"""

from __future__ import annotations

from datetime import datetime, timezone

from critic_evals.llm.client import AnthropicClient
from critic_evals.llm.models import ModelSpec
from critic_evals.schema import ArgumentItem, CritiqueRecord, Usage

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful, intellectually honest critic of arguments. "
    "Engage with the specific argument in front of you on its own terms."
)

CRITIQUE_PROMPTS: dict[str, str] = {
    "default": (
        "You will be given a question and an argument that attempts to answer it.\n\n"
        "Critique the argument. Assess how well it actually supports its conclusion. "
        "If it has weaknesses — unsupported leaps, internal tensions, overlooked "
        "considerations, or claims that do not hold up — identify the most important "
        "ones and explain why they matter. If the argument is basically sound, say so "
        "rather than manufacturing objections. Engage with the argument as written.\n\n"
        "Question:\n{question}\n\n"
        "Argument:\n{argument}"
    ),
}

DEFAULT_VARIANT = "default"


def build_prompt(item: ArgumentItem, variant: str = "default") -> tuple[str, str]:
    """Return `(system, user_prompt)` for eliciting a critique of `item`."""
    try:
        template = CRITIQUE_PROMPTS[variant]
    except KeyError:
        known = ", ".join(CRITIQUE_PROMPTS)
        raise KeyError(f"unknown prompt variant {variant!r}; known: {known}") from None
    return CRITIQUE_SYSTEM_PROMPT, template.format(
        question=item.question, argument=item.argument
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_critique(
    client: AnthropicClient,
    model: ModelSpec,
    item: ArgumentItem,
    *,
    variant: str = "default",
    sample: int = 0,
    max_tokens: int = 4096,
) -> CritiqueRecord:
    """Ask `model` to critique `item` and return the result as a `CritiqueRecord`.

    Exceptions propagate; callers that fan out decide how to record failures.
    """
    system, prompt = build_prompt(item, variant)
    completion = await client.complete(
        model_id=model.model_id, system=system, prompt=prompt, max_tokens=max_tokens
    )
    return CritiqueRecord(
        item_id=item.id,
        model=model.label,
        model_id=completion.model_id,
        prompt_variant=variant,
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
