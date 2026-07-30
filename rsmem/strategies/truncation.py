"""Truncation strategies for tool I/O compression.

The ``default`` strategy implements **GeoTraceCompress** (paper, Methodology):
it preserves the assistant's reasoning while compressing tool inputs/outputs into
a compact, human-readable form — summarizing long lists and explicitly retaining
salient error signals — to keep reflection overhead low.
"""

from __future__ import annotations

import ast
from typing import Optional

from rsmem.strategies import register_truncation


def _truncate_plain(text: str, max_chars: int) -> str:
    if len(text) > max_chars * 2:
        half = max_chars // 2
        return text[:half] + f" ... [TRUNCATED: {len(text) - max_chars} chars] ... " + text[-half:]
    return text[:max_chars] + f" [TRUNCATED: {len(text) - max_chars} chars]"


def _error_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars * 6:
        return text
    return (
        text[: max_chars * 3]
        + f" ... [TRUNCATED: {len(text) - max_chars * 6} chars] ... "
        + text[-max_chars * 3 :]
    )


_ERROR_KEYWORDS = [
    "error", "exception", "failed", "traceback",
    "invalid", "not found", "cannot", "unable",
]


def _has_error(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# Strategy: default  ==  GeoTraceCompress (paper, Methodology)
# ---------------------------------------------------------------------------

def truncate_default(text: Optional[str], max_chars: int = 500) -> Optional[str]:
    """GeoTraceCompress: compress a tool input/output string (lists, dicts,
    errors, long text) into a compact, human-readable form."""
    if not text or len(text) <= max_chars:
        return text

    if text.strip().startswith("["):
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list) and len(data) > 5:
                truncated = data[:3] + [f"... [{len(data) - 5} items omitted] ..."] + data[-2:]
                result = str(truncated)
                if len(result) > max_chars:
                    return result[:max_chars] + f" [TRUNCATED: {len(result) - max_chars} chars]"
                return result + " [TRUNCATED]"
        except (ValueError, SyntaxError, TypeError, MemoryError):
            pass

    if text.strip().startswith("{"):
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict) and len(str(data)) > max_chars:
                sample_keys = list(data.keys())[:2]
                sample = {k: data[k] for k in sample_keys}
                extra = max(len(data) - len(sample_keys), 0)
                out = f"{sample} ... [TRUNCATED: {extra} more keys]"
                if len(out) > max_chars:
                    return _truncate_plain(out, max_chars)
                return out
        except (ValueError, SyntaxError, TypeError, MemoryError):
            pass

    if _has_error(text):
        return _error_truncate(text, max_chars)
    return _truncate_plain(text, max_chars)


register_truncation("default", truncate_default)
