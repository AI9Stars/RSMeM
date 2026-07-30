"""Pure evaluation helpers (extracted from scripts/langchain_deepseek_autoplan_RSMeM.py)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

ISSUE_PATTERNS = {
    "refusal_or_incomplete": [
        r"\bsorry\b",
        r"need more steps",
        r"i (can't|cannot) continue",
        r"unable to complete",
    ],
    "data_access": [
        r"cannot access",
        r"missing file",
        r"no such file or directory",
        r"permission denied",
    ],
    "guessing_language": [
        r"educated guess",
        r"most reasonable estimate",
        r"most likely",
    ],
}

# DeepSeek V4 can namespace user-requested XML-like tags with its native DSML
# marker.  Accept both the canonical full-width form and degraded ASCII-pipe
# variants, in addition to the two Answer closing conventions supported by
# this project.
ANSWER_TAG_RE = re.compile(
    r"<\s*/?\s*"
    r"(?:(?:[|｜]\s*){1,2}DSML\s*(?:[|｜]\s*){1,2})?"
    r"\s*answer\s*>",
    flags=re.IGNORECASE,
)


def extract_tagged_answer(text: str) -> str | None:
    """Return the content between the final pair of supported Answer tags."""

    tags = list(ANSWER_TAG_RE.finditer(text))
    if len(tags) < 2:
        return None
    return text[tags[-2].end() : tags[-1].start()].strip()


def normalize_answer_tags(text: str) -> str:
    """Canonicalize supported Answer tags while preserving the full response."""

    return ANSWER_TAG_RE.sub("<Answer>", text)


def message_content_to_text(content: Any) -> str:
    """Convert LangChain/provider message content to text without list repr noise."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text", block.get("content"))
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "".join(parts)


def heuristic_critic_has_issue(final_answer_raw: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if not final_answer_raw or not final_answer_raw.strip():
        return True, ["empty_answer"]

    text = final_answer_raw.strip()
    low = text.lower()

    if len(ANSWER_TAG_RE.findall(text)) < 2:
        reasons.append("missing_answer_tag")

    inner_raw = extract_tagged_answer(text)
    if inner_raw is None:
        reasons.append("cannot_parse_answer_tag")
    else:
        inner = inner_raw.upper()
        if not re.search(r"\b[A-D]\b", inner):
            if not re.search(r"^\s*[A-D]\s*[\.\)]", inner):
                reasons.append("answer_tag_missing_choice_letter")

    for group, patterns in ISSUE_PATTERNS.items():
        for p in patterns:
            if re.search(p, low):
                reasons.append(group)
                break

    return (len(reasons) > 0), reasons


def extract_answer_from_response(response: Any) -> str:
    messages = response.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, "type") and message.type == "ai":
            content = message_content_to_text(message.content)
            return normalize_answer_tags(content)
    return "No answer found"




def aggregate_results(attempt_results_history: List[List[dict]]) -> dict:
    aggregated: Dict[Any, dict] = {}

    for attempt_idx, attempt_results in enumerate(attempt_results_history):
        for result in attempt_results:
            qid = result.get("question_id", result.get("qid"))
            if qid:
                aggregated[qid] = {**result, "latest_attempt": attempt_idx + 1}

    return aggregated


def compute_statistics(
    aggregated_results: dict,
    attempt_results_history: List[Any],
) -> dict:
    total_questions = len(aggregated_results)

    per_attempt_stats: List[dict] = []
    for attempt_idx, attempt_data in enumerate(attempt_results_history):
        if isinstance(attempt_data, dict):
            attempt_results = attempt_data.get("results", [])
            attempt_tokens = attempt_data.get("token_usage", {})
            reflection_tokens = attempt_data.get("reflection_token_usage", {})
        elif isinstance(attempt_data, list):
            attempt_results = attempt_data
            attempt_tokens = {}
            reflection_tokens = {}
        else:
            attempt_results = []
            attempt_tokens = {}
            reflection_tokens = {}

        heuristic_success = sum(
            1
            for r in attempt_results
            if isinstance(r, dict) and r.get("heuristic_success", False)
        )
        heuristic_fail = len(attempt_results) - heuristic_success

        per_attempt_stats.append(
            {
                "attempt": attempt_idx + 1,
                "total_questions": len(attempt_results),
                "heuristic_success": heuristic_success,
                "heuristic_fail": heuristic_fail,
                "heuristic_success_rate": heuristic_success / len(attempt_results)
                if attempt_results
                else 0.0,
                "token_usage": attempt_tokens,
                "reflection_token_usage": reflection_tokens,
            }
        )

    improvement_tracking = {
        "questions_improved": [],
        "questions_degraded": [],
        "questions_consistently_failed": [],
    }

    question_attempt_map: Dict[Any, list] = {}
    for attempt_idx, attempt_data in enumerate(attempt_results_history):
        if isinstance(attempt_data, dict):
            attempt_results = attempt_data.get("results", [])
        elif isinstance(attempt_data, list):
            attempt_results = attempt_data
        else:
            attempt_results = []

        for result in attempt_results:
            if not isinstance(result, dict):
                continue
            mqid = result.get("question_id", result.get("qid"))
            if mqid not in question_attempt_map:
                question_attempt_map[mqid] = []
            question_attempt_map[mqid].append(result)

    for qid, attempts in question_attempt_map.items():
        if len(attempts) < 2:
            continue

        first_success = attempts[0].get("heuristic_success", False)
        last_success = attempts[-1].get("heuristic_success", False)
        all_failed = all(not r.get("heuristic_success", False) for r in attempts)

        if not first_success and last_success:
            improvement_tracking["questions_improved"].append(qid)
        elif first_success and not last_success:
            improvement_tracking["questions_degraded"].append(qid)
        elif all_failed:
            improvement_tracking["questions_consistently_failed"].append(qid)

    total_tokens_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_reflection_tokens_all = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for stat in per_attempt_stats:
        tu = stat.get("token_usage", {})
        total_tokens_all["prompt_tokens"] += tu.get("prompt_tokens", 0)
        total_tokens_all["completion_tokens"] += tu.get("completion_tokens", 0)
        total_tokens_all["total_tokens"] += tu.get("total_tokens", 0)

        rtu = stat.get("reflection_token_usage", {})
        total_reflection_tokens_all["prompt_tokens"] += rtu.get("prompt_tokens", 0)
        total_reflection_tokens_all["completion_tokens"] += rtu.get(
            "completion_tokens", 0
        )
        total_reflection_tokens_all["total_tokens"] += rtu.get("total_tokens", 0)

    grand_total_tokens = {
        "prompt_tokens": total_tokens_all["prompt_tokens"]
        + total_reflection_tokens_all["prompt_tokens"],
        "completion_tokens": total_tokens_all["completion_tokens"]
        + total_reflection_tokens_all["completion_tokens"],
        "total_tokens": total_tokens_all["total_tokens"]
        + total_reflection_tokens_all["total_tokens"],
    }

    return {
        "total_questions": total_questions,
        "per_attempt_stats": per_attempt_stats,
        "improvement_tracking": improvement_tracking,
        "total_attempts": len(attempt_results_history),
        "token_summary": {
            "question_answering_tokens": total_tokens_all,
            "reflection_generation_tokens": total_reflection_tokens_all,
            "grand_total": grand_total_tokens,
        },
    }
