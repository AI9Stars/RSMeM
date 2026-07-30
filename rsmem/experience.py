"""Grounded full-trajectory summaries for Summary+FTRO reflection."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


SUMMARY_SYSTEM_PROMPT = """You are a trajectory state compressor, not a solver and not a critic.

Summarize what the previous tool-using agent was trying to do and what state it
actually reached. Every factual claim must cite source step IDs from the
numbered full trajectory.

Hard constraints:
- Do not recommend an answer or option letter.
- Do not invent tools, paths, artifacts, values, successes, or failures.
- Do not turn an intended action into a completed action.
- Separate verified tool evidence from unverified assistant reasoning.
- Preserve unresolved work and visible repetition/exhaustion signals.
- Do not copy absolute paths; refer to artifacts by semantic role.
- Return exactly one JSON object and no markdown.
"""

SUMMARY_USER_TEMPLATE = """question_id: {question_id}

---BEGIN NUMBERED FULL TRAJECTORY---
{rendered_full_trajectory}
---END NUMBERED FULL TRAJECTORY---

Return exactly:
{{
  "question_id": "{question_id}",
  "task_summary": {{
    "text": "What analysis the actor was trying to complete, without an answer",
    "source_step_ids": [1]
  }},
  "attempted_strategy": [
    {{"text": "A strategy actually attempted", "source_step_ids": [2]}}
  ],
  "verified_state": [
    {{"text": "A fact or artifact directly supported by tool output", "source_step_ids": [3]}}
  ],
  "observed_failures": [
    {{"text": "An observed error or contradiction", "source_step_ids": [4]}}
  ],
  "unfinished_subgoals": [
    {{"text": "Work left incomplete", "source_step_ids": [5]}}
  ],
  "budget_signals": [
    {{"text": "Visible repetition, expansion, or exhaustion", "source_step_ids": [6]}}
  ]
}}

Use empty arrays when no grounded item exists.
Keep each list to at most four consolidated items. Merge repeated errors into one
claim with multiple source step IDs. Target 800-1200 output tokens.
"""

SUMMARY_FTRO_REFLECTION_SYSTEM_PROMPT = """You are a repair reflector for a fixed tool-using actor.

Diagnose the previous failure and produce one concise, actor-executable local
repair. Use only the evidence channels supplied in the user message. If a
summary conflicts with raw tool output, trust raw tool output.

Hard constraints:
- Do not answer the multiple-choice question or provide an option letter.
- Preserve already verified work; do not restart the whole workflow.
- Prefer the smallest correction that can close the unfinished subgoal.
- Do not invent unavailable tools, paths, arguments, artifacts, or values.
- Avoid opening multiple fallback branches.
- Include a concrete stopping condition.
- Return exactly one JSON object and no markdown.
"""

SUMMARY_FTRO_REFLECTION_USER_TEMPLATE = """question_id: {question_id}
source_attempt: {source_attempt}

Question:
{question_text}

Retrieved context:
{retrieved_context}

LLM full-trajectory state summary:
{summary_channel}

Raw full-tool-result-only evidence:
{ftro_channel}

In 3-5 sentences (100-300 words), identify the earliest decisive failure,
state what verified work should be kept, give the minimal next actions, and
state when the actor should stop.

Return exactly:
{{
  "question_id": "{question_id}",
  "reflective_text": "Concise local repair for the next attempt."
}}
"""


class GroundedClaim(BaseModel):
    text: str = Field(min_length=3, max_length=900)
    source_step_ids: list[int] = Field(min_length=1, max_length=512)


class TrajectorySummary(BaseModel):
    question_id: str
    task_summary: GroundedClaim
    attempted_strategy: list[GroundedClaim] = Field(default_factory=list, max_length=8)
    verified_state: list[GroundedClaim] = Field(default_factory=list, max_length=10)
    observed_failures: list[GroundedClaim] = Field(default_factory=list, max_length=10)
    unfinished_subgoals: list[GroundedClaim] = Field(default_factory=list, max_length=8)
    budget_signals: list[GroundedClaim] = Field(default_factory=list, max_length=6)


class ReflectionResponse(BaseModel):
    question_id: str
    reflective_text: str = Field(min_length=20, max_length=3000)


_ANSWER_LEAK = re.compile(
    r"<\s*/?\s*answer\s*>|"
    r"\b(?:choose|select|answer(?:\s+is)?)\s+(?:option\s+)?[A-D]\b|"
    r"\b(?:matches?|corresponds?\s+to|closest\s+to)\s+(?:the\s+)?option\s+[A-D]\b|"
    r'"answer"\s*:',
    flags=re.I,
)
_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|root|mnt|data|tmp|workspace|ultrafast_share)/)"
    r"[^\s\"']*",
    flags=re.I,
)


def extract_object(text: str, required_key: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            return value
    raise ValueError(f"no JSON object containing {required_key!r}")


def parse_and_validate_summary(
    text: str, *, question_id: str, step_count: int
) -> TrajectorySummary:
    value = extract_object(text, "task_summary")
    summary = TrajectorySummary.model_validate(value)
    if summary.question_id != str(question_id):
        raise ValueError(
            f"question_id mismatch: expected {question_id}, got {summary.question_id}"
        )
    claims = [
        summary.task_summary,
        *summary.attempted_strategy,
        *summary.verified_state,
        *summary.observed_failures,
        *summary.unfinished_subgoals,
        *summary.budget_signals,
    ]
    for claim in claims:
        invalid = [step for step in claim.source_step_ids if step < 1 or step > step_count]
        if invalid:
            raise ValueError(f"q{question_id}: invalid source steps {invalid}; maximum={step_count}")
        if _ANSWER_LEAK.search(claim.text):
            raise ValueError(f"q{question_id}: answer leakage in summary")
        if _ABSOLUTE_PATH.search(claim.text):
            raise ValueError(f"q{question_id}: absolute path copied into summary")
    return summary


def parse_and_validate_reflection(text: str, *, question_id: str) -> ReflectionResponse:
    reflection = ReflectionResponse.model_validate(extract_object(text, "reflective_text"))
    if reflection.question_id != str(question_id):
        raise ValueError(
            f"question_id mismatch: expected {question_id}, got {reflection.question_id}"
        )
    if _ANSWER_LEAK.search(reflection.reflective_text):
        raise ValueError(f"q{question_id}: answer leakage in reflection")
    return reflection
