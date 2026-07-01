"""Generate and load reference metadata for argument items."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from critic_evals.llm.client import AnthropicClient

REF_DIR = Path(__file__).resolve().parents[3] / "references"

_SYSTEM = (
    "You are a logician mapping an argument from the text alone — no outside answer key. Be "
    "rigorous and charitable: engage the strongest reading, and neither invent gaps nor excuse "
    "real ones."
)

_DECOMPOSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["soundness", "strongest_reading", "concessions"],
    "properties": {
        "soundness": {
            "type": "string",
            "enum": ["sound", "flawed"],
            "description": "'sound' if the premises plausibly entail the conclusion with no decisive unstated premise, inconsistency, equivocation, inductive overreach, or fix-vs-diagnosis mismatch; else 'flawed'",
        },
        "strongest_reading": {
            "type": "string",
            "description": "the argument's load-bearing idea in its STRONGEST form (steelman) — what a critique should engage rather than weak phrasing",
        },
        "concessions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["concession", "grounding"],
                "properties": {
                    "concession": {
                        "type": "string",
                        "description": "an objection the argument PRE-EMPTS via a qualifier, scope limit, or concession",
                    },
                    "grounding": {
                        "type": "string",
                        "description": "verbatim quote of the qualifying span",
                    },
                },
            },
        },
    },
}

_GAPS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["gaps"],
    "properties": {
        "gaps": {
            "type": "array",
            "description": "every reasoning gap in the argument — deductive AND inductive",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["gap", "type", "grounding", "load_bearing"],
                "properties": {
                    "gap": {
                        "type": "string",
                        "description": "the gap, phrased concretely (for an unstated premise: 'the argument assumes X but never argues it')",
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "unstated_premise",
                            "invalid_step",
                            "equivocation",
                            "inductive_overreach",
                            "inconsistency",
                        ],
                    },
                    "grounding": {
                        "type": "string",
                        "description": "verbatim quote of the span where the gap sits",
                    },
                    "load_bearing": {
                        "type": "boolean",
                        "description": "true if the gap is in the MAIN chain to the conclusion and its rejection would damage the conclusion; false for peripheral gaps",
                    },
                },
            },
        },
    },
}

_DECOMPOSE_PROMPT = (
    "Decompose this ARGUMENT and decide whether it is fundamentally SOUND or FLAWED, from the text "
    "alone. Check rigorously and charitably: do the stated premises plausibly entail the "
    "conclusion; is it consistent and free of equivocation; if it proposes a fix, does the fix "
    "address the cause it diagnosed, or does it target a different cause; and do any inductive "
    "leaps overreach? Give the argument's STRONGEST reading, "
    "and list its CONCESSIONS — objections it already pre-empts via qualifiers or scope limits, "
    "each with a verbatim quote.\n\nARGUMENT:\n{argument}"
)
_GAPS_PROMPT = (
    "This ARGUMENT is flawed. Enumerate EVERY reasoning gap — where it can legitimately be "
    "attacked — using two families of lens:\n"
    "- DEDUCTIVE: unstated premises the conclusion needs but never argues (negate the conclusion: "
    "what must it be leaning on — especially about any alternative or baseline it implicitly "
    "endorses); invalid steps; equivocations; internal inconsistencies.\n"
    "- INDUCTIVE: overgeneralization from thin or unrepresentative evidence; weak analogies.\n"
    "For each gap give: the gap (an unstated premise phrased 'the argument assumes X but never "
    "argues it'), its type, a VERBATIM grounding quote, and load_bearing=true only if it sits in "
    "the MAIN chain and its rejection would damage the conclusion. Be exhaustive but do not invent "
    "gaps the argument already closes.\n\nARGUMENT:\n{argument}"
)


@dataclass(frozen=True, slots=True)
class Gap:
    gap: str
    type: str
    grounding: str
    load_bearing: bool


@dataclass(frozen=True, slots=True)
class AutoKey:
    soundness: str
    strongest_reading: str
    gaps: tuple[Gap, ...]
    concessions: tuple[str, ...]  # concession text with «grounding» inline
    raw: dict[str, object]

    @property
    def load_bearing_gaps(self) -> tuple[str, ...]:
        return tuple(g.gap for g in self.gaps if g.load_bearing)

    def to_key_string(self) -> str:
        """Human-readable rendering of the reference (for inspection / auditing)."""
        lines = [
            f"SOUNDNESS: {self.soundness}.",
            f"STRONGEST READING: {self.strongest_reading}",
        ]
        for g in self.gaps:
            tag = "LOAD-BEARING" if g.load_bearing else "peripheral"
            lines.append(f"GAP [{tag}, {g.type}]: {g.gap}  «{g.grounding[:70]}»")
        if self.concessions:
            lines.append("CONCESSIONS (pre-empted): " + "; ".join(self.concessions))
        return "\n".join(lines)


async def preprocess_argument(
    client: AnthropicClient,
    *,
    model_id: str,
    argument: str,
) -> AutoKey:
    """Decompose `argument` into a grounded attack-surface map (no human key)."""
    dec = (
        await client.complete(
            model_id=model_id,
            system=_SYSTEM,
            prompt=_DECOMPOSE_PROMPT.format(argument=argument),
            schema=_DECOMPOSE_SCHEMA,
            max_tokens=1800,
        )
    ).parsed or {}
    soundness = str(dec.get("soundness", "flawed"))
    strongest = str(dec.get("strongest_reading", ""))
    raw_conc = dec.get("concessions", [])
    concessions = (
        tuple(
            f"{c['concession']}  «{c['grounding'][:60]}»"
            if isinstance(c, dict) and c.get("grounding")
            else str(c.get("concession", c))
            for c in raw_conc
        )
        if isinstance(raw_conc, list)
        else ()
    )

    gaps: tuple[Gap, ...] = ()
    deep: dict[str, object] = {}
    if soundness == "flawed":
        deep = (
            await client.complete(
                model_id=model_id,
                system=_SYSTEM,
                prompt=_GAPS_PROMPT.format(argument=argument),
                schema=_GAPS_SCHEMA,
                max_tokens=2400,
            )
        ).parsed or {}
        raw_gaps = deep.get("gaps", [])
        gaps = (
            tuple(
                Gap(
                    str(g["gap"]),
                    str(g["type"]),
                    str(g.get("grounding", "")),
                    bool(g.get("load_bearing")),
                )
                for g in raw_gaps
                if isinstance(g, dict) and g.get("gap")
            )
            if isinstance(raw_gaps, list)
            else ()
        )

    return AutoKey(
        soundness=soundness,
        strongest_reading=strongest,
        gaps=gaps,
        concessions=concessions,
        raw={"decompose": dec, "gaps": deep},
    )


def load_reference(item_id: str, ref_dir: Path = REF_DIR) -> dict[str, object]:
    """Load a committed auto-reference (built by scripts/build_references.py)."""
    return json.loads((ref_dir / f"{item_id}.json").read_text(encoding="utf-8"))
