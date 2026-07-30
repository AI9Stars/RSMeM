"""Strategy registry for pluggable truncation, trajectory, and reflection variants.

Usage::

    from rsmem.strategies import get_truncation, get_trajectory, get_reflection

    truncate = get_truncation("default")   # returns a callable
    extract  = get_trajectory("head-30")
    reflect  = get_reflection("default")

To register a new strategy, use the ``@register_*`` decorators in the
corresponding module, or call ``register_truncation`` / etc. directly.

This module also hosts :func:`generate_reflections_for_failed_questions`, the
batch reflection orchestrator that wires the three strategy families together
(used by ``rsmem.runner``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

_TRUNCATION_REGISTRY: dict[str, Callable] = {}
_TRAJECTORY_REGISTRY: dict[str, Callable] = {}
_REFLECTION_REGISTRY: dict[str, Callable] = {}


def register_truncation(name: str, fn: Callable) -> None:
    _TRUNCATION_REGISTRY[name] = fn


def register_trajectory(name: str, fn: Callable) -> None:
    _TRAJECTORY_REGISTRY[name] = fn


def register_reflection(name: str, fn: Callable) -> None:
    _REFLECTION_REGISTRY[name] = fn


def get_truncation(name: str = "default") -> Callable:
    if name not in _TRUNCATION_REGISTRY:
        raise KeyError(
            f"Unknown truncation strategy {name!r}. "
            f"Available: {list(_TRUNCATION_REGISTRY)}"
        )
    return _TRUNCATION_REGISTRY[name]


def get_trajectory(name: str = "head-30") -> Callable:
    if name not in _TRAJECTORY_REGISTRY:
        raise KeyError(
            f"Unknown trajectory strategy {name!r}. "
            f"Available: {list(_TRAJECTORY_REGISTRY)}"
        )
    return _TRAJECTORY_REGISTRY[name]


def get_reflection(name: str = "default") -> Callable:
    if name not in _REFLECTION_REGISTRY:
        raise KeyError(
            f"Unknown reflection strategy {name!r}. "
            f"Available: {list(_REFLECTION_REGISTRY)}"
        )
    return _REFLECTION_REGISTRY[name]


def available_strategies() -> dict[str, list[str]]:
    return {
        "truncation": list(_TRUNCATION_REGISTRY),
        "trajectory": list(_TRAJECTORY_REGISTRY),
        "reflection": list(_REFLECTION_REGISTRY),
    }


async def generate_reflections_for_failed_questions(
    llm: Any,
    chat_log_path: Path,
    failed_results: list,
    attempt_num: int,
    first_attempt_only: bool = False,
    *,
    truncation_strategy: str = "default",
    trajectory_strategy: str = "head-30",
    reflection_strategy: str = "default",
    reflection_kwargs: dict[str, Any] | None = None,
    trajectory_kwargs: dict[str, Any] | None = None,
    reflection_failure_policy: str = "fallback",
) -> tuple[list, dict[str, int]]:
    """Batch-generate reflections (Reflexion-style FAR) using pluggable strategies.

    Selects a truncation/trajectory/reflection strategy by name, extracts a
    compressed trajectory per failed question, and asks the LLM for a structured
    reflection. ``reflection_kwargs`` is forwarded to the reflection strategy;
    the base ``default`` strategy ignores extras, but custom strategies may
    consume keys such as ``previous_reflection`` or ``num_candidates``.
    """
    reflections: list[dict[str, Any]] = []
    total_reflection_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    extra = reflection_kwargs or {}
    trajectory_extra = trajectory_kwargs or {}
    if reflection_failure_policy not in {"fallback", "error"}:
        raise ValueError("reflection_failure_policy must be 'fallback' or 'error'")

    truncate_fn = get_truncation(truncation_strategy)
    traj_fn = get_trajectory(trajectory_strategy)
    reflect_fn = get_reflection(reflection_strategy)

    print(f"\n{'='*60}")
    print(f"Generating Reflections for {len(failed_results)} Failed Questions")
    print(
        f"  strategies: trunc={truncation_strategy}, traj={trajectory_strategy}, "
        f"refl={reflection_strategy}"
    )
    print(f"{'='*60}")

    for result in failed_results:
        question_id = result["question_id"]

        trajectory_data = traj_fn(
            chat_log_path,
            question_id,
            first_attempt_only=first_attempt_only,
            truncate_fn=truncate_fn,
            **trajectory_extra,
        )

        reflection = await reflect_fn(
            llm=llm,
            question_id=question_id,
            attempt=attempt_num,
            question_text=trajectory_data["question_text"],
            retrieved_context=trajectory_data["retrieved_context"],
            trajectory=trajectory_data["trajectory"],
            **extra,
        )

        if reflection:
            reflections.append(reflection)
            if "token_usage" in reflection:
                tu = reflection["token_usage"]
                total_reflection_tokens["prompt_tokens"] += tu.get("prompt_tokens", 0)
                total_reflection_tokens["completion_tokens"] += tu.get("completion_tokens", 0)
                total_reflection_tokens["total_tokens"] += tu.get("total_tokens", 0)
        else:
            if reflection_failure_policy == "error":
                raise RuntimeError(
                    f"reflection generation failed for Q{question_id}; fallback disabled"
                )
            fallback_reflection = {
                "question_id": str(question_id),
                "reflective_text": (
                    "Previous attempt failed. Please review the tool workflow and "
                    "ensure all steps are completed correctly, paying special attention "
                    "to data paths and final answer format."
                ),
            }
            reflections.append(fallback_reflection)
            print(f"  → Using fallback reflection for Q{question_id}")

    print(f"\n✓ Generated {len(reflections)} reflections")
    if total_reflection_tokens["total_tokens"] > 0:
        print(f"  Reflection tokens: {total_reflection_tokens['total_tokens']} total")
    return reflections, total_reflection_tokens
