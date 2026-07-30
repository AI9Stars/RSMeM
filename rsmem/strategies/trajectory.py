"""Trajectory extraction strategies."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

from rsmem.strategies import register_trajectory


def _read_chat_messages(chat_log_path: Path) -> list[dict]:
    msgs: list[dict] = []
    with open(chat_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return msgs


def _capture_question_messages(
    messages: list[dict], qid: str, first_attempt_only: bool
) -> tuple[list[dict], str, str]:
    question_messages: list[dict] = []
    capturing = False
    first_captured = False
    user_prompt = ""
    retrieved_context = ""

    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("metadata"), dict):
            if str(msg["metadata"].get("question_id")) == qid:
                if not first_attempt_only or not first_captured:
                    capturing = True
                    question_messages = []
                    user_prompt = msg.get("content", "")
                    retrieved_context = msg["metadata"].get("retrieved_context", "")
                    first_captured = True
                continue
        if capturing and msg.get("name") == qid:
            question_messages.append(msg)
        elif capturing and msg.get("role") == "assistant":
            question_messages.append(msg)
        elif capturing and msg.get("role") in ("system", "tool"):
            question_messages.append(msg)
        elif capturing and msg.get("role") == "user":
            capturing = False
            if first_attempt_only:
                break

    return question_messages, user_prompt, retrieved_context


def _compress_message(msg: dict, truncate_fn: Callable) -> list[dict]:
    """Turn a single raw chat message into compressed trajectory items."""
    items: list[dict] = []
    if msg.get("role") == "assistant":
        content = msg.get("content", [])
        if isinstance(content, str):
            items.append({"type": "assistant_text", "text": content})
            return items
        for item in content:
            if item.get("type") == "text":
                items.append({"type": "assistant_text", "text": item.get("text", "")})
            elif item.get("type") == "tool_use":
                raw_input = str(item.get("input", {}))
                items.append({
                    "type": "tool_use",
                    "name": item.get("name", ""),
                    "input": truncate_fn(raw_input, max_chars=300),
                })
    elif msg.get("role") == "system":
        for item in msg.get("content", []):
            if item.get("type") == "tool_result":
                out = item.get("output")
                output_text = ""
                if out:
                    first = out[0] if isinstance(out, list) else out
                    raw = str(first.get("text", "")) if isinstance(first, dict) else str(first)
                    output_text = truncate_fn(raw, max_chars=300)
                items.append({
                    "type": "tool_result",
                    "name": item.get("name", ""),
                    "output": output_text,
                })
    elif msg.get("role") == "tool":
        items.append({"type": "assistant_text", "text": str(msg.get("content", ""))})
    return items


def _tool_result_text(block: dict[str, Any]) -> str:
    output = block.get("output", block.get("content", ""))
    if isinstance(output, list):
        return "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in output
        )
    return json.dumps(output, ensure_ascii=False, sort_keys=True, default=str)


def _question_text(user_prompt: str) -> str:
    marker = "### Question"
    return user_prompt.split(marker, 1)[1].strip() if marker in user_prompt else user_prompt.strip()


def _normalize_numbered_trajectory(
    q_msgs: list[dict], question_text: str
) -> list[dict[str, Any]]:
    """Build the lossless, step-addressable trajectory used by the compressor."""
    steps: list[dict[str, Any]] = []

    def emit(kind: str, **payload: Any) -> None:
        steps.append({"step_id": len(steps) + 1, "type": kind, **payload})

    emit("question", text=question_text)
    for message in q_msgs:
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                emit("assistant_text" if role == "assistant" else role, text=content)
            continue
        if not isinstance(content, list):
            if content is not None:
                emit(role or "message", value=content)
            continue
        for block in content:
            if not isinstance(block, dict):
                emit(role or "message", value=block)
                continue
            kind = str(block.get("type", "")).lower()
            if kind == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    emit("assistant_text", text=text)
            elif kind == "tool_use":
                emit(
                    "tool_use",
                    call_id=str(block.get("id", "")),
                    name=str(block.get("name", "")),
                    input=block.get("input", {}),
                )
            elif kind == "tool_result":
                emit(
                    "tool_result",
                    call_id=str(block.get("id") or block.get("tool_use_id") or ""),
                    name=str(block.get("name", "")),
                    output=_tool_result_text(block),
                )
            else:
                emit(kind or role or "message", value=block)
    return steps


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 80) // 2)
    omitted = len(text) - 2 * half
    return text[:half] + f"\n...[TRUNCATED {omitted} CHARACTERS]...\n" + text[-half:]


def render_numbered_trajectory(
    steps: list[dict[str, Any]], *, max_step_chars: int, max_trace_chars: int
) -> str:
    rows = []
    for step in steps:
        payload = {key: value for key, value in step.items() if key != "step_id"}
        rows.append(
            f"[{step['step_id']}] "
            + _truncate_middle(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                max_step_chars,
            )
        )
    return _truncate_middle("\n".join(rows), max_trace_chars)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Strategy: head-30
# ---------------------------------------------------------------------------

def extract_head_30(
    chat_log_path: Path,
    question_id: str,
    max_messages: int = 30,
    first_attempt_only: bool = False,
    truncate_fn: Optional[Callable] = None,
) -> dict[str, Any]:
    if truncate_fn is None:
        from rsmem.strategies.truncation import truncate_default
        truncate_fn = truncate_default

    try:
        if not chat_log_path.exists():
            return {"question_text": "", "retrieved_context": "", "trajectory": []}
        messages = _read_chat_messages(chat_log_path)
        q_msgs, user_prompt, retrieved_ctx = _capture_question_messages(
            messages, str(question_id), first_attempt_only
        )
        compressed: list[dict] = []
        for msg in q_msgs[:max_messages]:
            compressed.extend(_compress_message(msg, truncate_fn))
        return {
            "question_text": user_prompt,
            "retrieved_context": retrieved_ctx,
            "trajectory": compressed,
        }
    except Exception:
        return {"question_text": "", "retrieved_context": "", "trajectory": []}


# ---------------------------------------------------------------------------
# Strategy: tail_last_30  (chronological tail — mirror of ``head-30``)
# ---------------------------------------------------------------------------
#
# ``head-30`` keeps the *head* ``q_msgs[:max_messages]`` (first ~15 tool steps).
# ``tail_last_30`` keeps the *tail* ``q_msgs[-max_messages:]`` instead, so the
# reflector sees where the agent actually got stuck / ran out of steps. When the
# trajectory is short (``<= max_messages``) the whole thing is kept, which makes
# the result identical to ``head-30``.

def extract_tail_last_30(
    chat_log_path: Path,
    question_id: str,
    max_messages: int = 30,
    first_attempt_only: bool = False,
    truncate_fn: Optional[Callable] = None,
) -> dict[str, Any]:
    if truncate_fn is None:
        from rsmem.strategies.truncation import truncate_default
        truncate_fn = truncate_default

    try:
        if not chat_log_path.exists():
            return {"question_text": "", "retrieved_context": "", "trajectory": []}
        messages = _read_chat_messages(chat_log_path)
        q_msgs, user_prompt, retrieved_ctx = _capture_question_messages(
            messages, str(question_id), first_attempt_only
        )
        if len(q_msgs) > max_messages:
            kept = q_msgs[-max_messages:]
        else:
            kept = q_msgs
        compressed: list[dict] = []
        for msg in kept:
            compressed.extend(_compress_message(msg, truncate_fn))
        return {
            "question_text": user_prompt,
            "retrieved_context": retrieved_ctx,
            "trajectory": compressed,
        }
    except Exception:
        return {"question_text": "", "retrieved_context": "", "trajectory": []}


# ---------------------------------------------------------------------------
# Strategy: full_trajectory  (whole decision chain, no message-count cap)
# ---------------------------------------------------------------------------


def extract_full_trajectory(
    chat_log_path: Path,
    question_id: str,
    max_messages: int = 30,
    first_attempt_only: bool = False,
    truncate_fn: Optional[Callable] = None,
) -> dict[str, Any]:
    if truncate_fn is None:
        from rsmem.strategies.truncation import truncate_default
        truncate_fn = truncate_default

    try:
        if not chat_log_path.exists():
            return {"question_text": "", "retrieved_context": "", "trajectory": []}
        messages = _read_chat_messages(chat_log_path)
        q_msgs, user_prompt, retrieved_ctx = _capture_question_messages(
            messages, str(question_id), first_attempt_only
        )
        compressed: list[dict] = []
        for msg in q_msgs:  # NO cap — keep the whole trajectory
            compressed.extend(_compress_message(msg, truncate_fn))
        return {
            "question_text": user_prompt,
            "retrieved_context": retrieved_ctx,
            "trajectory": compressed,
        }
    except Exception:
        return {"question_text": "", "retrieved_context": "", "trajectory": []}


# ---------------------------------------------------------------------------
# Strategy: full_tool_result_only  (tool outputs only, full coverage)
# ---------------------------------------------------------------------------


def extract_full_tool_result_only(
    chat_log_path: Path,
    question_id: str,
    max_messages: int = 30,
    first_attempt_only: bool = False,
    truncate_fn: Optional[Callable] = None,
) -> dict[str, Any]:
    if truncate_fn is None:
        from rsmem.strategies.truncation import truncate_default
        truncate_fn = truncate_default

    try:
        if not chat_log_path.exists():
            return {"question_text": "", "retrieved_context": "", "trajectory": []}
        messages = _read_chat_messages(chat_log_path)
        q_msgs, user_prompt, retrieved_ctx = _capture_question_messages(
            messages, str(question_id), first_attempt_only
        )
        compressed: list[dict] = []
        for msg in q_msgs:  # NO cap
            for item in _compress_message(msg, truncate_fn):
                if item.get("type") == "tool_result":
                    compressed.append(item)
        return {
            "question_text": user_prompt,
            "retrieved_context": retrieved_ctx,
            "trajectory": compressed,
        }
    except Exception:
        return {"question_text": "", "retrieved_context": "", "trajectory": []}


def extract_summary_ftro_source(
    chat_log_path: Path,
    question_id: str,
    max_messages: int = 30,
    first_attempt_only: bool = False,
    truncate_fn: Optional[Callable] = None,
    max_step_chars: int = 6000,
    max_trace_chars: int = 120000,
) -> dict[str, Any]:
    if truncate_fn is None:
        from rsmem.strategies.truncation import truncate_default
        truncate_fn = truncate_default
    if not chat_log_path.exists():
        return {"question_text": "", "retrieved_context": "", "trajectory": {}}
    messages = _read_chat_messages(chat_log_path)
    q_msgs, user_prompt, retrieved_ctx = _capture_question_messages(
        messages, str(question_id), first_attempt_only
    )
    if not q_msgs:
        return {"question_text": user_prompt, "retrieved_context": retrieved_ctx, "trajectory": {}}
    full = _normalize_numbered_trajectory(q_msgs, _question_text(user_prompt))
    ftro: list[dict[str, Any]] = []
    for msg in q_msgs:
        for item in _compress_message(msg, truncate_fn):
            if item.get("type") == "tool_result":
                ftro.append(item)
    return {
        "question_text": _question_text(user_prompt),
        "retrieved_context": retrieved_ctx,
        "trajectory": {
            "full_trajectory": full,
            "rendered_full_trajectory": render_numbered_trajectory(
                full,
                max_step_chars=max_step_chars,
                max_trace_chars=max_trace_chars,
            ),
            "ftro": ftro,
            "full_trajectory_sha256": _sha256_json(full),
            "ftro_sha256": _sha256_json(ftro),
        },
    }


register_trajectory("head-30", extract_head_30)
register_trajectory("tail_last_30", extract_tail_last_30)
register_trajectory("full_trajectory", extract_full_trajectory)
register_trajectory("full_tool_result_only", extract_full_tool_result_only)
register_trajectory("summary_ftro_source", extract_summary_ftro_source)
