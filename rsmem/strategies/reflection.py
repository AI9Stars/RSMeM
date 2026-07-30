"""Reflection generation strategies."""

from __future__ import annotations

import json
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from rsmem.prompt import DEFAULT_REFLECTION_TEMPLATE
from rsmem.strategies import register_reflection
from rsmem.experience import (
    SUMMARY_FTRO_REFLECTION_SYSTEM_PROMPT,
    SUMMARY_FTRO_REFLECTION_USER_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
    parse_and_validate_reflection,
    parse_and_validate_summary,
)


def _build_llm_messages(prompt: str) -> list:
    try:
        from langchain.schema import HumanMessage
        return [HumanMessage(content=prompt)]
    except ImportError:
        try:
            from langchain_core.messages import HumanMessage
            return [HumanMessage(content=prompt)]
        except ImportError:
            from types import SimpleNamespace
            return [SimpleNamespace(content=prompt)]


def _build_system_user_messages(system_prompt: str, user_prompt: str) -> list:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    except ImportError:
        from types import SimpleNamespace
        return [
            SimpleNamespace(type="system", content=system_prompt),
            SimpleNamespace(type="human", content=user_prompt),
        ]


def _extract_token_usage(response: Any) -> dict[str, int]:
    tu = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        if hasattr(um, "input_tokens"):
            tu["prompt_tokens"] = um.input_tokens
            tu["completion_tokens"] = um.output_tokens
            tu["total_tokens"] = um.input_tokens + um.output_tokens
        elif hasattr(um, "prompt_tokens"):
            tu["prompt_tokens"] = um.prompt_tokens
            tu["completion_tokens"] = um.completion_tokens
            tu["total_tokens"] = um.total_tokens
        elif isinstance(um, dict):
            tu["prompt_tokens"] = um.get("input_tokens", um.get("prompt_tokens", 0))
            tu["completion_tokens"] = um.get("output_tokens", um.get("completion_tokens", 0))
            tu["total_tokens"] = um.get("total_tokens", 0)
    if tu["total_tokens"] == 0 and (tu["prompt_tokens"] > 0 or tu["completion_tokens"] > 0):
        tu["total_tokens"] = tu["prompt_tokens"] + tu["completion_tokens"]
    return tu


def _parse_reflection_json(text: str) -> Optional[dict]:
    """Try multiple strategies to extract a reflection JSON from LLM output."""
    try:
        data = json.loads(text)
        if "reflective_text" in data:
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if "reflective_text" in data:
                return data
        except json.JSONDecodeError:
            pass
    m = re.search(r'({\s*"question_id".*?"reflective_text".*?})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


async def reflect_default(
    llm: Any,
    question_id: str,
    attempt: int,
    question_text: str,
    retrieved_context: str,
    trajectory: list,
    max_retries: int = 2,
    reflection_template: str = DEFAULT_REFLECTION_TEMPLATE,
    **kwargs: Any,
) -> Optional[dict]:
    trajectory_json = json.dumps(trajectory, ensure_ascii=False, indent=2)
    prompt = reflection_template.format(
        question_id=question_id,
        attempt=attempt,
        question_text=question_text[:1000] if question_text else "(Question text not available)",
        retrieved_context=retrieved_context[:500] if retrieved_context else "(No retrieved context)",
        trajectory_json=trajectory_json,
    )
    for retry in range(max_retries):
        try:
            response = await llm.ainvoke(_build_llm_messages(prompt))
            token_usage = _extract_token_usage(response)
            data = _parse_reflection_json(response.content.strip())
            if data:
                data["token_usage"] = token_usage
                return data
            if retry < max_retries - 1:
                prompt += "\n\nIMPORTANT: Output ONLY the JSON object, no markdown code blocks, no explanations."
        except Exception:
            pass
    return None


register_reflection("default", reflect_default)


def _add_token_usage(*values: dict[str, int]) -> dict[str, int]:
    return {
        key: sum(int(value.get(key, 0)) for value in values)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _latest_success(
    path: Path,
    question_id: str,
    *,
    full_trajectory_sha256: str | None = None,
    ftro_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("question_id")) != str(question_id) or row.get("status") != "ok":
            continue
        if full_trajectory_sha256 and row.get("full_trajectory_sha256") != full_trajectory_sha256:
            continue
        if ftro_sha256 and row.get("ftro_sha256") != ftro_sha256:
            continue
        if str(row.get("question_id")) == str(question_id):
            latest = row
    return latest


async def _invoke_validated(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    parser: Any,
    correction: str,
) -> tuple[Any, str, dict[str, int], int]:
    prompt = user_prompt
    last_error: Exception | None = None
    for format_attempt in range(1, 3):
        response = await llm.ainvoke(_build_system_user_messages(system_prompt, prompt))
        raw = str(response.content or "").strip()
        if not raw:
            last_error = ValueError("model returned empty content")
        else:
            try:
                return parser(raw), raw, _extract_token_usage(response), format_attempt
            except Exception as exc:
                last_error = exc
        if format_attempt == 1:
            prompt = user_prompt + "\n\n" + correction
    assert last_error is not None
    raise last_error


async def reflect_summary_ftro(
    llm: Any,
    question_id: str,
    attempt: int,
    question_text: str,
    retrieved_context: str,
    trajectory: dict[str, Any],
    *,
    compressor_llm: Any,
    artifact_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compress a canonical full trace, then reflect over Summary + legacy FTRO."""
    if not isinstance(trajectory, dict) or not trajectory.get("full_trajectory"):
        raise ValueError(f"q{question_id}: summary_ftro_source produced no full trajectory")
    full = trajectory["full_trajectory"]
    rendered = trajectory["rendered_full_trajectory"]
    ftro = trajectory.get("ftro", [])
    root = Path(artifact_dir)
    started = time.time()
    case = {
        "question_id": str(question_id),
        "source_attempt": attempt,
        "question_text": question_text,
        "retrieved_context": retrieved_context,
        "full_trajectory": full,
        "rendered_full_trajectory": rendered,
        "ftro": ftro,
        "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
        "ftro_sha256": trajectory["ftro_sha256"],
    }
    _write_json(root / "cases" / f"q{question_id}.json", case)

    cached_reflection = _latest_success(
        root / "reflections.jsonl",
        str(question_id),
        full_trajectory_sha256=trajectory["full_trajectory_sha256"],
        ftro_sha256=trajectory["ftro_sha256"],
    )
    if cached_reflection:
        parsed = parse_and_validate_reflection(
            json.dumps(cached_reflection["reflection"]), question_id=str(question_id)
        )
        experience_tokens = cached_reflection.get("experience_token_usage") or {}
        total_tokens = experience_tokens.get("total") or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return {
            "question_id": str(question_id),
            "reflective_text": parsed.reflective_text,
            "trajectory_summary": cached_reflection.get("trajectory_sent_to_teacher", {}).get(
                "full_trajectory_summary"
            ),
            "experience_token_usage": experience_tokens,
            "token_usage": total_tokens,
            "reused_artifact": True,
        }

    summary_user = SUMMARY_USER_TEMPLATE.format(
        question_id=question_id,
        rendered_full_trajectory=rendered,
    )
    summary_request = {
        "question_id": str(question_id),
        "source_attempt": attempt,
        "system_prompt": SUMMARY_SYSTEM_PROMPT,
        "user_prompt": summary_user,
        "prompt_sha256": hashlib.sha256(summary_user.encode()).hexdigest(),
        "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
    }
    _write_json(root / "summary_requests" / f"q{question_id}.json", summary_request)

    cached_summary = _latest_success(
        root / "summaries.jsonl",
        str(question_id),
        full_trajectory_sha256=trajectory["full_trajectory_sha256"],
    )
    if cached_summary:
        summary = parse_and_validate_summary(
            json.dumps(cached_summary["summary"]),
            question_id=str(question_id),
            step_count=len(full),
        )
        summary_raw = str(cached_summary.get("raw_response", ""))
        summary_tokens = cached_summary.get("token_usage") or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        summary_formats = int(cached_summary.get("format_attempts", 1))
    else:
        try:
            summary, summary_raw, summary_tokens, summary_formats = await _invoke_validated(
                compressor_llm,
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=summary_user,
                parser=lambda raw: parse_and_validate_summary(
                    raw, question_id=str(question_id), step_count=len(full)
                ),
                correction=(
                    "The previous response failed schema or grounding validation. "
                    "Return one corrected JSON object only."
                ),
            )
        except Exception as exc:
            _append_jsonl(
                root / "summaries.jsonl",
                {
                    "question_id": str(question_id),
                    "source_attempt": attempt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
                },
            )
            raise
    summary_dict = summary.model_dump()
    if not cached_summary:
        _append_jsonl(
            root / "summaries.jsonl",
            {
                "question_id": str(question_id),
                "source_attempt": attempt,
                "status": "ok",
                "summary": summary_dict,
                "raw_response": summary_raw,
                "format_attempts": summary_formats,
                "token_usage": summary_tokens,
                "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
            },
        )

    reflection_user = SUMMARY_FTRO_REFLECTION_USER_TEMPLATE.format(
        question_id=question_id,
        source_attempt=attempt,
        question_text=question_text or "(Question text not available)",
        retrieved_context=retrieved_context or "(none)",
        summary_channel=json.dumps(summary_dict, ensure_ascii=False, indent=2),
        ftro_channel=json.dumps(ftro, ensure_ascii=False, indent=2),
    )
    teacher_payload = {
        "full_trajectory_summary": summary_dict,
        "ftro": ftro,
    }
    reflection_request = {
        "question_id": str(question_id),
        "source_attempt": attempt,
        "trajectory_sent_to_teacher": teacher_payload,
        "system_prompt": SUMMARY_FTRO_REFLECTION_SYSTEM_PROMPT,
        "user_prompt": reflection_user,
        "prompt_sha256": hashlib.sha256(reflection_user.encode()).hexdigest(),
        "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
        "ftro_sha256": trajectory["ftro_sha256"],
    }
    _write_json(root / "reflection_requests" / f"q{question_id}.json", reflection_request)

    try:
        reflection, reflection_raw, reflection_tokens, reflection_formats = await _invoke_validated(
            llm,
            system_prompt=SUMMARY_FTRO_REFLECTION_SYSTEM_PROMPT,
            user_prompt=reflection_user,
            parser=lambda raw: parse_and_validate_reflection(raw, question_id=str(question_id)),
            correction=(
                "The previous response failed schema validation. Return one corrected "
                "JSON object only, without an answer."
            ),
        )
    except Exception as exc:
        _append_jsonl(
            root / "reflections.jsonl",
            {
                "question_id": str(question_id),
                "source_attempt": attempt,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
                "ftro_sha256": trajectory["ftro_sha256"],
            },
        )
        raise
    total_tokens = _add_token_usage(summary_tokens, reflection_tokens)
    row = {
        "question_id": str(question_id),
        "source_attempt": attempt,
        "status": "ok",
        "trajectory_sent_to_teacher": teacher_payload,
        "reflection": reflection.model_dump(),
        "raw_response": reflection_raw,
        "format_attempts": reflection_formats,
        "experience_token_usage": {
            "compressor": summary_tokens,
            "reflector": reflection_tokens,
            "total": total_tokens,
        },
        "latency_seconds": round(time.time() - started, 3),
        "full_trajectory_sha256": trajectory["full_trajectory_sha256"],
        "ftro_sha256": trajectory["ftro_sha256"],
    }
    _append_jsonl(root / "reflections.jsonl", row)
    return {
        "question_id": str(question_id),
        "reflective_text": reflection.reflective_text,
        "trajectory_summary": summary_dict,
        "experience_token_usage": row["experience_token_usage"],
        "token_usage": total_tokens,
    }


register_reflection("summary_ftro", reflect_summary_ftro)
